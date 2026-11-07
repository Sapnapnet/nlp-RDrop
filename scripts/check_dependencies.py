#!/usr/bin/env python3
"""Offline checks for the author's fork and the pinned Python 3.10 environment.

Uses a tiny random BERT and synthetic text, not pretrained BERT or MRPC data.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib
import importlib.metadata as metadata
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback

from prepare_source import prepare_source

ROOT = Path(__file__).resolve().parents[1]


def run_checks(report, record):
    import torch
    if sys.version_info[:2] != (3, 10) or str(torch.__version__) != "2.1.2+cu118" or torch.version.cuda != "11.8":
        raise RuntimeError("Expected Python 3.10 and PyTorch 2.1.2+cu118.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this server check.")
    source = prepare_source()
    report["source"] = source
    record("source_integrity", f"Verified commit {source['commit']}; two dependency edits only.")
    manifest = json.loads((ROOT / "provenance/wheels.json").read_text(encoding="utf-8"))
    versions = {}
    for item in manifest["wheels"]:
        actual = metadata.version(item["name"])
        versions[item["name"]] = actual
        if actual != item["version"]:
            raise RuntimeError(f"Package mismatch: {item['name']}={actual}, expected {item['version']}")
    report["package_versions"] = versions
    record("pinned_versions", f"All {len(versions)} dependency versions match the offline lock.")
    modules = ("transformers", "tokenizers", "datasets", "huggingface_hub", "pyarrow", "pandas",
               "sklearn", "scipy", "sacremoses", "numpy", "fsspec", "psutil", "yaml")
    report["import_paths"] = {name: str(importlib.import_module(name).__file__) for name in modules}
    import transformers
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizer, BertTokenizerFast
    from transformers import DataCollatorWithPadding, Trainer, TrainingArguments
    import datasets
    from sklearn.metrics import accuracy_score, f1_score
    expected_init = ROOT / "vendor/R-Drop/huggingface_transformer_src/src/transformers/__init__.py"
    if Path(transformers.__file__).resolve() != expected_init.resolve() or transformers.__version__ != "4.6.0.dev0":
        raise RuntimeError("Imported transformers is not the pinned author fork.")
    if not hasattr(BertForSequenceClassification, "bert_kl") or not hasattr(BertForSequenceClassification, "bert_cross_entropy"):
        raise RuntimeError("Author R-Drop / cross-entropy methods are missing.")
    bert_file = Path(inspect.getfile(BertForSequenceClassification))
    report["author_model"] = {"file": str(bert_file), "sha256": hashlib.sha256(bert_file.read_bytes()).hexdigest(),
                              "transformers_version": transformers.__version__}
    record("author_fork_import", "Transformers loads from the bundled author source; bert_kl and bert_cross_entropy exist.")

    with tempfile.TemporaryDirectory(prefix="offline_probe_", dir=ROOT / "tmp") as scratch:
        scratch = Path(scratch)
        vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "A", "man", "woman", "person",
                 "sits", "walks", "runs", ".", "The", "dog", "cat"]
        vocab_file = scratch / "vocab.txt"
        vocab_file.write_text("\n".join(vocab) + "\n", encoding="utf-8")
        slow = BertTokenizer(vocab_file=str(vocab_file), do_lower_case=False)
        fast = BertTokenizerFast(vocab_file=str(vocab_file), do_lower_case=False)
        texts = ["A man sits .", "A woman walks .", "The dog runs .", "The cat sits ."]
        pairs = ["A person sits .", "A person runs .", "The cat sits .", "A woman sits ."]
        options = {"padding": "max_length", "truncation": True, "max_length": 16}
        slow_enc = slow(texts, pairs, **options)
        fast_enc = fast(texts, pairs, **options)
        for key in ("input_ids", "attention_mask", "token_type_ids"):
            if slow_enc[key] != fast_enc[key]:
                raise RuntimeError(f"Slow/fast BERT tokenizer mismatch: {key}")
        saved_tokenizer = scratch / "tokenizer"
        fast.save_pretrained(saved_tokenizer)
        restored = BertTokenizerFast.from_pretrained(saved_tokenizer, local_files_only=True)
        if restored(texts, pairs, **options)["input_ids"] != fast_enc["input_ids"]:
            raise RuntimeError("Tokenizer save/load roundtrip mismatch.")
        record("tokenizer_compatibility", "tokenizers 0.13.3: sentence pairs, padding, truncation, slow/fast parity and save/load passed.")

        dataset = datasets.Dataset.from_dict({"sentence1": texts, "sentence2": pairs, "label": [1, 0, 0, 0]})
        dataset = dataset.map(lambda batch: fast(batch["sentence1"], batch["sentence2"], **options), batched=True)
        dataset.save_to_disk(str(scratch / "dataset"))
        dataset = datasets.load_from_disk(str(scratch / "dataset"))
        args = TrainingArguments(output_dir=str(scratch / "trainer"), per_device_train_batch_size=2,
                                 max_steps=1, report_to=[], disable_tqdm=True, skip_memory_metrics=True)
        config = BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2,
                            num_attention_heads=4, intermediate_size=64, max_position_embeddings=32,
                            num_labels=2, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1)
        trainer = Trainer(model=BertForSequenceClassification(config), args=args,
                          train_dataset=dataset, tokenizer=fast, data_collator=DataCollatorWithPadding(fast))
        batch = next(iter(trainer.get_train_dataloader()))
        if tuple(batch["input_ids"].shape) != (2, 16) or tuple(batch["labels"].shape) != (2,):
            raise RuntimeError("Dataset / Trainer collator shape mismatch.")
        record("dataset_trainer_interop", "Arrow save/load, author Trainer initialization and DataLoader batch succeeded.")
        del trainer
        labels = [0, 1, 1, 0]
        predictions = [0, 1, 0, 0]
        if accuracy_score(labels, predictions) != 0.75 or abs(f1_score(labels, predictions) - 2/3) > 1e-12:
            raise RuntimeError("Accuracy / binary F1 sanity check failed.")
        record("metrics", "Known-label Accuracy and binary F1 checks passed (synthetic values, not MRPC scores).")

        report["gpu_probes"] = {}
        for use_amp in (False, True):
            torch.manual_seed(2026)
            torch.cuda.manual_seed_all(2026)
            model = BertForSequenceClassification(config).cuda().train()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp, init_scale=128.0)
            cuda_batch = {k: v.cuda() for k, v in batch.items()}
            calls = []
            hook = model.bert.register_forward_hook(lambda module, inputs, outputs: calls.append(1))
            before = model.classifier.weight.detach().clone()
            try:
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                    outputs = model(**cuda_batch)
                if len(calls) != 2:
                    raise RuntimeError(f"Author default forward should call BERT twice, got {len(calls)}.")
                if outputs.logits.shape != (2, 2) or not torch.isfinite(outputs.loss).item():
                    raise RuntimeError("Invalid author model logits or loss.")
                if use_amp and outputs.logits.dtype != torch.float16:
                    raise RuntimeError("Expected FP16 logits under autocast.")
                scaler.scale(outputs.loss).backward()
                scaler.unscale_(optimizer)
                grads = [p.grad for p in model.parameters() if p.requires_grad]
                if not grads or any(g is None or not torch.isfinite(g).all().item() for g in grads):
                    raise RuntimeError("Author model gradient check failed.")
                scaler.step(optimizer)
                scaler.update()
                torch.cuda.synchronize()
                if torch.equal(before, model.classifier.weight.detach()):
                    raise RuntimeError("Classifier parameters did not update.")
                name = "fp16_amp" if use_amp else "fp32"
                report["gpu_probes"][name] = {"status": "PASS", "bert_forward_calls": len(calls),
                                              "loss": outputs.loss.item(), "logits_dtype": str(outputs.logits.dtype),
                                              "finite_gradients": True, "parameters_updated": True}
                record("author_tiny_bert_" + name, "Two author BERT forwards, finite loss/gradients and update passed.")
            finally:
                hook.remove()
            del model, optimizer, scaler, outputs, cuda_batch, before, grads
            torch.cuda.empty_cache()
    report["runtime"] = {"python": sys.version, "executable": sys.executable,
                         "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
                         "gpu": torch.cuda.get_device_name(0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    os.environ.update({"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "USE_TORCH": "1", "USE_TF": "0", "USE_FLAX": "0", "TOKENIZERS_PARALLELISM": "false"})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / "tmp").mkdir(exist_ok=True)
    report = {"step": 2, "overall_status": "RUNNING", "checks": [],
              "checked_at_china_time": datetime.now(timezone(timedelta(hours=8))).isoformat(),
              "scope": "Source and dependency compatibility. Tiny random BERT only; no pretrained model or MRPC data."}

    def record(name, detail):
        report["checks"].append({"name": name, "status": "PASS", "detail": detail})
        print(f"[PASS] {name}: {detail}", flush=True)

    try:
        run_checks(report, record)
        report["overall_status"] = "PASS"
    except Exception:
        report["overall_status"] = "FAIL"
        report["traceback"] = traceback.format_exc()
        print(report["traceback"], flush=True)
    (args.output_dir / "dependency_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# 第 2 步源码与依赖检查", "", f"总体状态：**{report['overall_status']}**", "",
             "这是离线兼容性诊断，使用临时词表、合成句子和随机初始化的小型 BERT。",
             "不代表 MRPC Baseline、正式 R-Drop 消融或完整训练已完成。", "",
             "| 检查项 | 状态 | 说明 |", "|---|---|---|"]
    lines += [f"| {c['name']} | {c['status']} | {c['detail']} |" for c in report["checks"]]
    if "traceback" in report:
        lines += ["", "## 错误详情", "", "```text", report["traceback"], "```"]
    lines += ["", "详细包版本、实际导入路径和诊断结果见同目录 dependency_report.json。", ""]
    (args.output_dir / "dependency_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("Dependency checks: " + report["overall_status"], flush=True)
    return 0 if report["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
