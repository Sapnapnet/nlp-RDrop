"""Instrumented full-run loop shared with future B adapters; no CLI/network access."""
from __future__ import annotations
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import time

import torch
from .training_core import (ForwardCounter, evaluate, make_optimizer, make_scheduler, make_update_plan,
                            model_fingerprint, plan_identity, synchronize, train_update)


def write_json(path, value):
    """Atomically replace a small report, so polling never reads half-written JSON."""
    path = Path(path)
    partial = path.with_name(path.name + ".tmp")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    partial.replace(path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_plan(plan):
    sizes = [sum(len(b) for b in u["microbatches"]) for u in plan]
    return {"optimizer_updates": len(plan), "microbatches": sum(len(u["microbatches"]) for u in plan),
            "example_visits": sum(sizes), "update_size_counts": {str(n): sizes.count(n) for n in sorted(set(sizes))},
            "epochs_entered": max(u["epoch"] for u in plan) + 1}


def run_experiment(model, tokenizer, tensors, native_ids, output, protocol, run_config, device, report,
                   on_progress=None, reload_tokenizer_check=None):
    """Train and evaluate one seed. Model adapter supplies mean loss and single-pass eval.

    CPU/tiny models are allowed for local tests, but the server CLI independently
    enforces the fixed real resources, CUDA environment and full-data budget.
    """
    output = Path(output)
    model_class = type(model)
    total = run_config["max_updates"]
    forwards_per_microbatch = run_config.get("forwards_per_training_microbatch", 1)
    report["overall_status"] = "RUNNING"
    report["initial_state_sha256"] = model_fingerprint(model)
    expected_initial = run_config.get("validated_seed13_initial_state_sha256")
    if expected_initial and report["initial_state_sha256"] != expected_initial:
        raise RuntimeError("Seed-13 model initialization differs from the accepted preflight.")
    plan = make_update_plan(len(native_ids["train"]), protocol["microbatch_size"],
                            protocol["gradient_accumulation_steps"], total, run_config["seed"])
    report["planned_workload"] = summarize_plan(plan)
    report["batch_plan_sha256"] = plan_identity(plan, native_ids["train"])
    for actual, configured in (("microbatches", "expected_microbatches"), ("example_visits", "expected_training_example_visits")):
        if configured in run_config and report["planned_workload"][actual] != run_config[configured]:
            raise RuntimeError("Workload plan mismatch: " + actual)
    write_json(output / "data_ids.json", native_ids)
    model.to(device)
    counter = ForwardCounter(model)
    optimizer = make_optimizer(model, protocol["optimizer"])
    scheduler = make_scheduler(optimizer, total, run_config["warmup_updates"], protocol["scheduler"]["end_learning_rate"])
    use_amp = device.type == "cuda" and protocol["train_precision"] == "fp16_amp"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp, init_scale=protocol["amp_init_scale"],
                                      growth_interval=protocol["amp_growth_interval"])
    report["effective_precision"] = "fp16_amp" if use_amp else "fp32_test_only"
    report["phase"] = "training"
    report["completed_updates"] = 0
    write_json(output / "formal_report.json", report)
    query = model.bert.encoder.layer[0].attention.self.query.weight
    snapshots = (model.classifier.weight.detach().clone(), query.detach().clone())
    events = []
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    started = time.perf_counter()
    try:
        with (output / "training_updates.jsonl").open("w", encoding="utf-8") as stream:
            for number, group in enumerate(plan, 1):
                event = train_update(model, tensors["train"], group, optimizer, scheduler, scaler, device,
                                     protocol["max_grad_norm"], use_amp, forwards_per_microbatch, counter)
                event["update_seconds"] = event.pop("diagnostic_update_seconds")
                event.update({"update": number, "epoch": group["epoch"],
                              "native_ids": [native_ids["train"][i] for b in group["microbatches"] for i in b]})
                stream.write(json.dumps(event, allow_nan=False) + "\n")
                stream.flush()
                events.append(event)
                report["completed_updates"] = number
                if number == 1 or number % protocol["log_every_updates"] == 0 or number == total:
                    elapsed = time.perf_counter() - started
                    progress = {"status": "TRAINING", "completed_updates": number, "target_updates": total,
                                "last_loss": event["loss_mean"], "lr_used": event["lr_used"],
                                "elapsed_train_seconds": elapsed,
                                "estimated_remaining_train_seconds": elapsed / number * (total - number)}
                    write_json(output / "progress.json", progress)
                    write_json(output / "formal_report.json", report)
                    print(f"step {number}/{total} loss={event['loss_mean']:.6f} lr={event['lr_used']:.3g} elapsed={elapsed:.1f}s", flush=True)
                    if on_progress:
                        on_progress(progress)
        synchronize(device)
        wall_seconds = time.perf_counter() - started
        memory = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)} if device.type == "cuda" else {
                      "peak_allocated_bytes": None, "peak_reserved_bytes": None}
        report["training"] = {
            "optimizer_step_calls": sum(e["optimizer_step_calls"] for e in events),
            "microbatches": sum(e["microbatches"] for e in events),
            "examples_seen": sum(e["examples_seen"] for e in events),
            "distinct_native_ids_seen": len({idx for e in events for idx in e["native_ids"]}),
            "forward_counts": dict(counter.counts["train"]),
            "train_loop_wall_seconds": wall_seconds,
            "mean_wall_seconds_per_update": wall_seconds / total,
            "sum_update_body_seconds": sum(e["update_seconds"] for e in events),
            "mean_update_body_seconds": sum(e["update_seconds"] for e in events) / total,
            "amp_skipped_updates": 0, "all_gradient_norms_finite": True,
            "classifier_max_abs_change": (model.classifier.weight.detach() - snapshots[0]).abs().max().item(),
            "encoder_probe_max_abs_change": (query.detach() - snapshots[1]).abs().max().item(), **memory}
        training = report["training"]
        counts = training["forward_counts"]
        if (training["optimizer_step_calls"] != total or training["examples_seen"] != report["planned_workload"]["example_visits"]
                or counts["encoder_forward_calls"] != training["microbatches"] * forwards_per_microbatch
                or counts["encoder_example_passes"] != training["examples_seen"] * forwards_per_microbatch):
            raise RuntimeError("Measured optimizer/sample/encoder work differs from the plan.")
        if training["classifier_max_abs_change"] <= 0 or training["encoder_probe_max_abs_change"] <= 0:
            raise RuntimeError("Expected both classifier and encoder updates.")
        for key in ("encoder_forward_calls", "encoder_example_passes"):
            if "expected_" + key in run_config and counts[key] != run_config["expected_" + key]:
                raise RuntimeError("Measured work differs from the frozen configuration: " + key)
        report["epoch_training_summary"] = []
        for epoch in sorted({e["epoch"] for e in events}):
            selected = [e for e in events if e["epoch"] == epoch]
            samples = sum(e["examples_seen"] for e in selected)
            report["epoch_training_summary"].append({"epoch": epoch, "updates": len(selected), "sample_visits": samples,
                "sample_weighted_mean_training_loss": sum(e["loss_mean"] * e["examples_seen"] for e in selected) / samples})
        del snapshots, query
        report["phase"] = "evaluating_final_checkpoint"
        write_json(output / "formal_report.json", report)
        metrics, logits = evaluate(model, tensors["validation"], protocol["eval_batch_size"], device, counter)
        report["dev_metrics"] = {**metrics, "split": "validation", "checkpoint_selection": "last_update"}
        predictions = logits.argmax(dim=-1).tolist()
        with (output / "predictions_dev.jsonl").open("w", encoding="utf-8") as stream:
            for idx, label, prediction, values in zip(native_ids["validation"], tensors["validation"]["labels"].tolist(), predictions, logits.tolist()):
                stream.write(json.dumps({"idx": idx, "label": label, "prediction": prediction, "logits": values}, allow_nan=False) + "\n")
        report["phase"] = "saving"
        write_json(output / "formal_report.json", report)
        checkpoint = output / "checkpoint-final"
        checkpoint.mkdir(exist_ok=False)
        save_started = time.perf_counter()
        model.save_pretrained(str(checkpoint))
        if tokenizer is not None:
            tokenizer.save_pretrained(str(checkpoint))
        state_sha = model_fingerprint(model)
        report["checkpoint_save_and_state_hash_seconds"] = time.perf_counter() - save_started
        report["final_state_sha256"] = state_sha
        write_json(output / "checkpoint_manifest.json", {
            "path": str(checkpoint), "state_sha256": state_sha, "optimizer_saved": False,
            "files": [{"name": p.name, "size": p.stat().st_size, "sha256": file_sha256(p)}
                      for p in sorted(checkpoint.iterdir()) if p.is_file()]})
        report["forward_counts"] = counter.counts
        counter.close()
        # Keep the original on CPU while reloading to avoid holding two CUDA models.
        model.to("cpu")
        del optimizer, scheduler, scaler
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        report["phase"] = "reloading_for_inference"
        write_json(output / "formal_report.json", report)
        reload_started = time.perf_counter()
        restored, loading = model_class.from_pretrained(str(checkpoint), local_files_only=True, output_loading_info=True)
        if loading["missing_keys"] or loading["unexpected_keys"] or loading["error_msgs"] or model_fingerprint(restored) != state_sha:
            raise RuntimeError("Saved checkpoint state did not reload exactly.")
        if reload_tokenizer_check:
            reload_tokenizer_check(checkpoint)
        restored.to(device)
        second_counter = ForwardCounter(restored)
        try:
            after_metrics, after_logits = evaluate(restored, tensors["validation"], protocol["eval_batch_size"], device, second_counter, "reload_eval")
        finally:
            second_counter.close()
        if not torch.allclose(logits, after_logits, atol=1e-6, rtol=0) or after_logits.argmax(-1).tolist() != predictions:
            raise RuntimeError("Reloaded checkpoint predictions/logits differ.")
        if any(metrics[k] != after_metrics[k] for k in ("accuracy", "f1", "tp", "tn", "fp", "fn")):
            raise RuntimeError("Reloaded metrics differ.")
        report["checkpoint_reload"] = {"state_sha256_equal": True, "predictions_equal": True, "metrics_equal": True,
            "max_abs_logit_difference": (logits - after_logits).abs().max().item(), "tokenizer_checked": reload_tokenizer_check is not None,
            "reload_seconds": time.perf_counter() - reload_started, "training_resume_tested": False,
            "forward_counts": second_counter.counts}
        report["phase"] = "complete"
        report["overall_status"] = "PASS"
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "formal_report.json", report)
        write_json(output / "progress.json", {"status": "PASS", "completed_updates": total, "target_updates": total})
        return report
    finally:
        counter.close()
