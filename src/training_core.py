"""Shared batching, optimization, evaluation and accounting for A/B.

Sampling uses an independent RNG. Every update averages over its actual examples,
including an incomplete last update. The model supplies the per-microbatch mean loss.
"""
from __future__ import annotations
from contextlib import nullcontext
import hashlib
import json
import math
import random
import time

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def make_update_plan(num_examples, microbatch_size, accumulation_steps, max_updates, seed):
    if min(num_examples, microbatch_size, accumulation_steps, max_updates) <= 0:
        raise ValueError("Batch-plan dimensions must be positive.")
    plan = []
    epoch = 0
    while len(plan) < max_updates:
        order = list(range(num_examples))
        random.Random(seed + epoch).shuffle(order)
        micros = [order[i:i + microbatch_size] for i in range(0, num_examples, microbatch_size)]
        for i in range(0, len(micros), accumulation_steps):
            plan.append({"epoch": epoch, "microbatches": micros[i:i + accumulation_steps]})
            if len(plan) == max_updates:
                break
        epoch += 1
    return plan


def plan_identity(plan, native_ids):
    values = [{"epoch": u["epoch"], "microbatches": [[native_ids[i] for i in b] for b in u["microbatches"]]}
              for u in plan]
    return hashlib.sha256(json.dumps(values, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def model_fingerprint(model):
    """Hash tensor names, dtypes, shapes and exact values; independent of pickle bytes."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def make_optimizer(model, config):
    decay, no_decay = [], []
    for name, value in model.named_parameters():
        if value.requires_grad:
            (no_decay if name.endswith("bias") or "LayerNorm.weight" in name else decay).append(value)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": config["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=config["learning_rate"], betas=tuple(config["betas"]), eps=config["eps"], foreach=False)


def make_scheduler(optimizer, total_updates, warmup_updates, end_lr):
    base = optimizer.defaults["lr"]
    if not 0 <= warmup_updates < total_updates or not 0 <= end_lr < base:
        raise ValueError("Invalid polynomial schedule configuration.")

    def factor(step):
        if step < warmup_updates:
            return step / max(1, warmup_updates)
        if step >= total_updates:
            return end_lr / base
        progress = (step - warmup_updates) / (total_updates - warmup_updates)
        return ((base - end_lr) * (1 - progress) + end_lr) / base

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class ForwardCounter:
    def __init__(self, model):
        self.phase = "unassigned"
        self.counts = {}
        self.handle = model.bert.register_forward_hook(self._hook, with_kwargs=True)

    def _hook(self, module, args, kwargs, output):
        ids = args[0] if args and args[0] is not None else kwargs["input_ids"]
        item = self.counts.setdefault(self.phase, {"encoder_forward_calls": 0, "encoder_example_passes": 0,
                                                   "padded_token_positions": 0})
        item["encoder_forward_calls"] += 1
        item["encoder_example_passes"] += ids.shape[0]
        item["padded_token_positions"] += ids.numel()

    def close(self):
        self.handle.remove()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def take_batch(tensors, indices, device):
    index = torch.tensor(indices, dtype=torch.long)
    return {key: value.index_select(0, index).to(device) for key, value in tensors.items()}


def train_update(model, tensors, group, optimizer, scheduler, scaler, device,
                 max_grad_norm, use_amp, expected_forwards, counter):
    """One optimizer step, with sample-weighted accumulation and measured forwards."""
    model.train()
    counter.phase = "train"
    count_before = counter.counts.get("train", {}).get("encoder_forward_calls", 0)
    count = sum(len(indices) for indices in group["microbatches"])
    optimizer.zero_grad(set_to_none=True)
    synchronize(device)
    started = time.perf_counter()
    loss_value = 0.0
    lr_used = optimizer.param_groups[0]["lr"]
    for indices in group["microbatches"]:
        batch = take_batch(tensors, indices, device)
        context = torch.cuda.amp.autocast(dtype=torch.float16) if use_amp else nullcontext()
        with context:
            outputs = model(**batch)
            loss = outputs.loss
        if not torch.isfinite(loss).item():
            raise RuntimeError("Nonfinite training loss.")
        weight = len(indices) / count
        loss_value += loss.detach().float().item() * weight
        scaler.scale(loss * weight).backward()
    scaler.unscale_(optimizer)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if any(p.grad is None for p in trainable):
        raise RuntimeError("Some trainable parameters did not receive gradients.")
    norm = torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm, error_if_nonfinite=True)
    # A post-hook counts actual optimizer.step calls; GradScaler skips do not count.
    actual_calls = []
    hook = optimizer.register_step_post_hook(lambda *args, **kwargs: actual_calls.append(1))
    scale_before = scaler.get_scale()
    try:
        scaler.step(optimizer)
        scaler.update()
    finally:
        hook.remove()
    if len(actual_calls) != 1:
        raise RuntimeError("AMP skipped an optimizer step; stop the diagnostic instead of counting a false update.")
    scheduler.step()
    synchronize(device)
    seconds = time.perf_counter() - started
    calls = counter.counts["train"]["encoder_forward_calls"] - count_before
    if calls != len(group["microbatches"]) * expected_forwards:
        raise RuntimeError(f"Unexpected number of encoder forwards: {calls}")
    optimizer.zero_grad(set_to_none=True)
    return {"loss_mean": loss_value, "grad_norm_before_clip": float(norm), "lr_used": lr_used,
            "lr_next": optimizer.param_groups[0]["lr"], "amp_scale_before": scale_before,
            "amp_scale_after": scaler.get_scale(), "optimizer_step_calls": len(actual_calls),
            "examples_seen": count, "microbatches": len(group["microbatches"]),
            "encoder_forward_calls": calls, "diagnostic_update_seconds": seconds}


def binary_metrics(labels, predictions):
    if len(labels) != len(predictions) or not labels or any(x not in (0, 1) for x in labels + predictions):
        raise ValueError("Nonempty aligned binary labels and predictions required.")
    tp = sum(y == p == 1 for y, p in zip(labels, predictions))
    tn = sum(y == p == 0 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    denominator = 2 * tp + fp + fn
    return {"accuracy": (tp + tn) / len(labels), "f1": 2 * tp / denominator if denominator else 0.0,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn, "num_examples": len(labels)}


def evaluate(model, tensors, batch_size, device, counter, phase="eval"):
    """FP32, eval mode, single forward per batch; model adapter must ensure that path."""
    model.eval()
    counter.phase = phase
    rows = tensors["labels"].shape[0]
    before = counter.counts.get(phase, {}).get("encoder_forward_calls", 0)
    logits_list = []
    summed_loss = 0.0
    synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, rows, batch_size):
            indices = list(range(start, min(start + batch_size, rows)))
            outputs = model(**take_batch(tensors, indices, device))
            if not torch.isfinite(outputs.logits).all().item() or not torch.isfinite(outputs.loss).item():
                raise RuntimeError("Nonfinite evaluation output.")
            logits_list.append(outputs.logits.float().cpu())
            summed_loss += outputs.loss.item() * len(indices)
    synchronize(device)
    seconds = time.perf_counter() - started
    if counter.counts[phase]["encoder_forward_calls"] - before != math.ceil(rows / batch_size):
        raise RuntimeError("Evaluation must perform exactly one encoder forward per batch.")
    logits = torch.cat(logits_list)
    metrics = binary_metrics(tensors["labels"].tolist(), logits.argmax(dim=-1).tolist())
    metrics.update({"loss_mean": summed_loss / rows, "evaluation_seconds": seconds, "precision": "fp32"})
    return metrics, logits
