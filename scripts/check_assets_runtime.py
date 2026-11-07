"""Child process for step 3: load real MRPC/BERT offline, tokenize, and run forwards.

This is a resource check. No backward pass, optimizer, checkpoint, or dev metrics.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata as metadata
import inspect
import json
from pathlib import Path
import sys
import traceback

from asset_utils import ROOT, SPLITS, audit_rows, canonical, read_rows, sha256_file


def check(report, record):
    import torch
    import transformers
    import datasets
    import pyarrow.parquet as pq
    from torch.utils.data import DataLoader
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizer, BertTokenizerFast
    from transformers import DataCollatorWithPadding

    expected = ROOT / "vendor/R-Drop/huggingface_transformer_src/src/transformers/__init__.py"
    if Path(transformers.__file__).resolve() != expected.resolve() or transformers.__version__ != "4.6.0.dev0":
        raise RuntimeError("Expected the editable author Transformers fork from step 2.")
    bert_path = Path(inspect.getfile(BertForSequenceClassification))
    bert_sha = sha256_file(bert_path)
    if bert_sha != "1316e7d6990cd9a12f08689dbb3cf0fcd5b165bcc981e519f46babe5f66cbf06":
        raise RuntimeError("The author BERT algorithm source differs from step 2; send this report before continuing.")
    if str(torch.__version__) != "2.1.2+cu118" or torch.version.cuda != "11.8" or not torch.cuda.is_available():
        raise RuntimeError("Expected the validated PyTorch 2.1.2+cu118 environment with CUDA available.")
    versions = {name: metadata.version(name) for name in ("tokenizers", "datasets", "pyarrow", "numpy")}
    if versions != {"tokenizers": "0.13.3", "datasets": "2.19.2", "pyarrow": "16.1.0", "numpy": "1.26.4"}:
        raise RuntimeError(f"Resource-check dependency versions changed: {versions}")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(13)
    torch.cuda.manual_seed_all(13)
    report["runtime"] = {"python": sys.version, "executable": sys.executable, "torch": str(torch.__version__),
                         "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                         "transformers": transformers.__version__, "transformers_file": str(transformers.__file__),
                         "bert_source_sha256": bert_sha, "package_versions": versions, "probe_seed": 13}
    record("runtime", "Validated project environment and unchanged author BERT source.")

    protocol = json.loads((ROOT / "configs/input_protocol.json").read_text(encoding="utf-8"))
    rows = {s: read_rows(ROOT / "data/mrpc" / (s + ".jsonl")) for s in SPLITS}
    for split in SPLITS:
        table = pq.read_table(ROOT / f"data/mrpc/raw/{split}.parquet")
        if table.to_pylist() != rows[split]:
            raise RuntimeError(f"{split}: JSONL rows differ from the original Parquet, including order.")
        label_names = json.loads(table.schema.metadata[b"huggingface"])["info"]["features"]["label"]["names"]
        if label_names != ["not_equivalent", "equivalent"]:
            raise RuntimeError("Unexpected source label mapping.")
    report["data_audit"] = audit_rows(rows)
    record("real_mrpc", "All 3668 train and 408 validation rows match source Parquet exactly; IDs and labels verified.")

    model_dir = ROOT / protocol["model"]["local_path"]
    fast = BertTokenizerFast.from_pretrained(str(model_dir), local_files_only=True)
    slow = BertTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    if fast.do_lower_case or slow.do_lower_case or fast.vocab_size != 28996:
        raise RuntimeError("Expected the original case-sensitive BERT tokenizer with vocabulary size 28996.")
    if fast("Apple")['input_ids'] == fast("apple")['input_ids']:
        raise RuntimeError("Tokenizer unexpectedly lost case information.")
    options = {k: v for k, v in protocol["tokenization"].items() if k not in ("class", "do_lower_case")}
    keys = ("input_ids", "attention_mask", "token_type_ids")
    tokenized = {}
    report["tokenization"] = {"options": protocol["tokenization"], "splits": {}}
    for split, split_rows in rows.items():
        first = [r["sentence1"] for r in split_rows]
        second = [r["sentence2"] for r in split_rows]
        enc = fast(first, second, **options)
        slow_enc = slow(first, second, **options)
        mismatches = [split_rows[i]["idx"] for i in range(len(split_rows))
                      if any(enc[k][i] != slow_enc[k][i] for k in keys)]
        if mismatches:
            report["tokenizer_mismatch"] = {"split": split, "count": len(mismatches), "first_native_ids": mismatches[:20]}
            raise RuntimeError("Real-data fast/slow tokenizer mismatch; inspect the recorded IDs before training.")
        untruncated = fast(first, second, padding=False, truncation=False, add_special_tokens=True)
        lengths = [len(x) for x in untruncated["input_ids"]]
        for i in range(len(split_rows)):
            if any(len(enc[k][i]) != 128 for k in keys):
                raise RuntimeError("Expected fixed 128-token pair encodings.")
            ids, mask, types = (enc[k][i] for k in keys)
            valid = sum(mask)
            if (valid < 3 or mask != [1] * valid + [0] * (128 - valid)
                    or ids[0] != fast.cls_token_id or ids[valid-1] != fast.sep_token_id
                    or any(t not in (0, 1) for t in types) or 1 not in types[:valid]
                    or any(t < 0 or t >= fast.vocab_size for t in ids)
                    or any(t != fast.pad_token_id for t in ids[valid:])):
                raise RuntimeError(f"{split}: malformed sentence-pair encoding for idx={split_rows[i]['idx']}")
        encoding_sha = hashlib.sha256(canonical({k: enc[k] for k in keys}).encode("utf-8")).hexdigest()
        report["tokenization"]["splits"][split] = {
            "num_rows": len(split_rows), "slow_fast_mismatch_count": len(mismatches),
            "untruncated_length_min": min(lengths), "untruncated_length_max": max(lengths),
            "untruncated_length_mean": sum(lengths) / len(lengths),
            "truncated_examples": sum(n > 128 for n in lengths),
            "truncated_fraction": sum(n > 128 for n in lengths) / len(lengths),
            "encoding_sha256": encoding_sha}
        tokenized[split] = datasets.Dataset.from_dict({**{k: enc[k] for k in keys},
                                                      "label": [r["label"] for r in split_rows]})
    report["tokenization"]["example"] = {"split": "train", "native_idx": rows["train"][0]["idx"],
                                         **{k: tokenized["train"][0][k] for k in keys}}
    record("full_data_tokenization", "Fast/slow tokenizers agree on every MRPC pair; fixed length 128 and truncation counts recorded.")

    loader = DataLoader(tokenized["train"], batch_size=8, shuffle=False, num_workers=0,
                        collate_fn=DataCollatorWithPadding(fast, padding="max_length", max_length=128))
    batch = next(iter(loader))
    if tuple(batch["input_ids"].shape) != (8, 128) or tuple(batch["labels"].shape) != (8,):
        raise RuntimeError("Real-data batch/collator shape mismatch.")
    record("real_batch", "Dataset/DataLoader produced a fixed batch of 8 real pairs, shape [8,128].")

    config = BertConfig.from_pretrained(str(model_dir), local_files_only=True, num_labels=2)
    config.id2label = {0: "not_equivalent", 1: "equivalent"}
    config.label2id = {value: key for key, value in config.id2label.items()}
    if (config.num_hidden_layers, config.hidden_size, config.num_attention_heads, config.num_labels) != (12, 768, 12, 2):
        raise RuntimeError("Unexpected BERT model dimensions.")
    print("Loading the verified pretrained encoder from the local weight file...", flush=True)
    model, loading = BertForSequenceClassification.from_pretrained(str(model_dir), config=config,
                                                                   local_files_only=True, output_loading_info=True)
    report["pretrained_loading"] = loading
    if set(loading["missing_keys"]) != {"classifier.weight", "classifier.bias"}:
        raise RuntimeError("Expected only a new binary classifier; some encoder weights may be missing.")
    if loading.get("error_msgs") or loading.get("mismatched_keys") or any(not k.startswith("cls.") for k in loading["unexpected_keys"]):
        raise RuntimeError("Unexpected pretrained state-dict mismatch; inspect the loading report.")
    report["model"] = {"parameters": sum(p.numel() for p in model.parameters()),
                       "encoder_parameters": sum(p.numel() for p in model.bert.parameters()),
                       "classifier_parameters": sum(p.numel() for p in model.classifier.parameters()),
                       "hidden_dropout_prob": config.hidden_dropout_prob,
                       "attention_probs_dropout_prob": config.attention_probs_dropout_prob,
                       "classifier_is_new": True, "encoder_missing_keys": []}
    record("pretrained_encoder", "All encoder weights loaded; only the binary classifier is newly initialized, as expected.")

    model = model.cuda().eval()
    cuda_batch = {key: value.cuda() for key, value in batch.items()}
    report["forward_probes"] = {}
    for use_amp in (False, True):
        calls = []
        hook = model.bert.register_forward_hook(lambda module, inputs, outputs: calls.append(1))
        try:
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                # Use the author's existing single-forward path, not its default bert_kl forward.
                loss, logits, outputs = model.bert_cross_entropy(
                    input_ids=cuda_batch["input_ids"], attention_mask=cuda_batch["attention_mask"],
                    token_type_ids=cuda_batch["token_type_ids"], position_ids=None, head_mask=None,
                    inputs_embeds=None, labels=cuda_batch["labels"], output_attentions=False,
                    output_hidden_states=False, return_dict=True)
            torch.cuda.synchronize()
            if len(calls) != 1 or tuple(logits.shape) != (8, 2):
                raise RuntimeError("Expected exactly one encoder forward and [8,2] logits.")
            if not torch.isfinite(logits).all().item() or not torch.isfinite(loss).item():
                raise RuntimeError("Nonfinite logits/loss on real data.")
            if use_amp and logits.dtype != torch.float16:
                raise RuntimeError("Expected FP16 logits under AMP.")
            name = "fp16_amp" if use_amp else "fp32"
            report["forward_probes"][name] = {"encoder_calls": len(calls), "logits_shape": list(logits.shape),
                                               "logits_dtype": str(logits.dtype), "finite_logits": True,
                                               "diagnostic_loss": loss.item(), "training": False,
                                               "backward_performed": False, "optimizer_updates": 0}
            record(name + "_real_forward", "One pretrained encoder forward, finite logits/loss; no parameter updates.")
        finally:
            hook.remove()
    if any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("Unexpected parameter gradients in inference-only resource check.")
    del model, outputs, logits, loss, cuda_batch
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    output = parser.parse_args().output_dir
    output.mkdir(parents=True, exist_ok=True)
    report = {"step": 3, "overall_status": "RUNNING", "checks": [],
              "scope": "Real resources and inference only; no training, dev metrics or formal costs."}

    def record(name, detail):
        report["checks"].append({"name": name, "status": "PASS", "detail": detail})
        print(f"[PASS] {name}: {detail}", flush=True)

    try:
        check(report, record)
        report["overall_status"] = "PASS"
    except Exception:
        report["overall_status"] = "FAIL"
        report["traceback"] = traceback.format_exc()
        print(report["traceback"], flush=True)
    (output / "runtime_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if report["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
