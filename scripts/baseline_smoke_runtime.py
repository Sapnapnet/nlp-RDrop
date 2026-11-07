"""Step-4 GPU child: 20 baseline updates, dev-subset metrics, save and inference reload."""
from __future__ import annotations
from datetime import datetime, timezone
import argparse
import gc
import inspect
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from asset_utils import canonical, read_rows, sha256_file


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(output, report, record):
    import torch
    import transformers
    from transformers import BertConfig, BertTokenizerFast, BertForSequenceClassification
    from sklearn.metrics import accuracy_score, f1_score
    from src.baseline_model import BaselineBertForSequenceClassification
    from src.training_core import (ForwardCounter, binary_metrics, evaluate, make_optimizer, make_scheduler,
                                   make_update_plan, model_fingerprint, plan_identity, seed_everything,
                                   synchronize, take_batch, train_update)

    protocol = json.loads((ROOT / "configs/training_protocol.json").read_text(encoding="utf-8"))
    smoke = json.loads((ROOT / "configs/baseline_smoke.json").read_text(encoding="utf-8"))
    inputs = json.loads((ROOT / "configs/input_protocol.json").read_text(encoding="utf-8"))
    report["resolved_config"] = {"shared": protocol, "smoke": smoke, "inputs": inputs}
    if smoke["purpose"] != "smoke_only_not_formal_results" or smoke["condition"] != "baseline" or smoke["max_updates"] != 20:
        raise RuntimeError("This entry point is restricted to the approved 20-update baseline smoke test.")
    if protocol["input_manifest_sha256"] != sha256_file(ROOT / "provenance/assets.json"):
        raise RuntimeError("Shared input manifest differs from the accepted step-3 version.")
    expected_import = ROOT / "vendor/R-Drop/huggingface_transformer_src/src/transformers/__init__.py"
    bert_sha = sha256_file(Path(inspect.getfile(BertForSequenceClassification)))
    if Path(transformers.__file__).resolve() != expected_import.resolve() or bert_sha != "1316e7d6990cd9a12f08689dbb3cf0fcd5b165bcc981e519f46babe5f66cbf06":
        raise RuntimeError("Expected the unchanged author BERT source and the project fork import path.")
    if str(torch.__version__) != "2.1.2+cu118" or not torch.cuda.is_available():
        raise RuntimeError("Use the previously validated server PyTorch 2.1.2+cu118.")
    report["runtime"] = {"python": sys.version, "executable": sys.executable, "torch": str(torch.__version__),
                         "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                         "transformers": transformers.__version__, "bert_source_sha256": bert_sha}
    torch.set_num_threads(4)
    seed_everything(smoke["seed"])
    device = torch.device("cuda:0")
    record("environment_and_source", "Validated environment; author algorithm file remains unchanged.")

    rows = {s: read_rows(ROOT / f"data/mrpc/{s}.jsonl")[:smoke[key]["count"]]
            for s, key in (("train", "train_subset"), ("validation", "dev_subset"))}
    report["subset"] = {s: {"count": len(r), "native_ids": [x["idx"] for x in r],
                            "label_counts": {str(label): sum(x["label"] == label for x in r) for label in (0, 1)}}
                        for s, r in rows.items()}
    write_json(output / "subset_ids.json", report["subset"])
    model_dir = ROOT / inputs["model"]["local_path"]
    tokenizer = BertTokenizerFast.from_pretrained(str(model_dir), local_files_only=True)
    options = {k: v for k, v in inputs["tokenization"].items() if k not in ("class", "do_lower_case")}
    tensors = {}
    for split, data in rows.items():
        enc = tokenizer([r["sentence1"] for r in data], [r["sentence2"] for r in data], **options)
        tensors[split] = {key: torch.tensor(enc[key], dtype=torch.long)
                          for key in ("input_ids", "attention_mask", "token_type_ids")}
        tensors[split]["labels"] = torch.tensor([r["label"] for r in data], dtype=torch.long)
    config = BertConfig.from_pretrained(str(model_dir), local_files_only=True, num_labels=2)
    config.id2label = {0: "not_equivalent", 1: "equivalent"}
    config.label2id = {v: k for k, v in config.id2label.items()}
    model, loading = BaselineBertForSequenceClassification.from_pretrained(
        str(model_dir), config=config, local_files_only=True, output_loading_info=True)
    if set(loading["missing_keys"]) != {"classifier.weight", "classifier.bias"} or loading["error_msgs"] or any(not k.startswith("cls.") for k in loading["unexpected_keys"]):
        raise RuntimeError("Unexpected pretrained encoder loading mismatch.")
    report["pretrained_loading"] = loading
    report["initial_state_sha256"] = model_fingerprint(model)
    model.to(device)
    counter = ForwardCounter(model)
    sample = take_batch(tensors["train"], [0, 1, 2, 3], device)
    rng_cpu, rng_cuda = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    counter.phase = "dropout_probe"
    with torch.no_grad():
        model.train()
        p, q = model(**sample).logits, model(**sample).logits
        if torch.equal(p, q):
            raise RuntimeError("Training dropout produced identical probe logits unexpectedly.")
        model.eval()
        e1, e2 = model(**sample), model(**sample)
        if not torch.equal(e1.logits, e2.logits):
            raise RuntimeError("Evaluation is not deterministic with dropout disabled.")
        ce = torch.nn.functional.cross_entropy(e1.logits, sample["labels"])
        if not torch.allclose(e1.loss, ce, atol=1e-7, rtol=0):
            raise RuntimeError("Baseline loss differs from mean cross entropy.")
    torch.set_rng_state(rng_cpu)
    torch.cuda.set_rng_state_all(rng_cuda)
    if counter.counts["dropout_probe"]["encoder_forward_calls"] != 4:
        raise RuntimeError("Baseline unexpectedly performed extra encoder forwards.")
    report["dropout_probe"] = {"train_logits_differ": True, "eval_logits_equal": True,
                               "mean_ce_matches": True, "rng_restored_before_training": True}
    del sample, p, q, e1, e2, ce
    record("baseline_and_dropout", "Mean CE and one encoder forward; train dropout on, evaluation dropout off.")

    plan = make_update_plan(len(rows["train"]), protocol["microbatch_size"], protocol["gradient_accumulation_steps"],
                            smoke["max_updates"], smoke["seed"])
    ids = [r["idx"] for r in rows["train"]]
    report["batch_plan_sha256"] = plan_identity(plan, ids)
    optimizer = make_optimizer(model, protocol["optimizer"])
    scheduler = make_scheduler(optimizer, smoke["max_updates"], smoke["warmup_updates"], protocol["scheduler"]["end_learning_rate"])
    scaler = torch.cuda.amp.GradScaler(init_scale=protocol["amp_init_scale"], growth_interval=protocol["amp_growth_interval"])
    snapshots = {"classifier": model.classifier.weight.detach().clone(),
                 "encoder_query": model.bert.encoder.layer[0].attention.self.query.weight.detach().clone()}
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    loop_started = time.perf_counter()
    events = []
    with (output / "training_updates.jsonl").open("w", encoding="utf-8") as stream:
        for number, group in enumerate(plan, 1):
            event = train_update(model, tensors["train"], group, optimizer, scheduler, scaler, device,
                                 protocol["max_grad_norm"], True, 1, counter)
            event.update({"update": number, "epoch": group["epoch"],
                          "native_ids": [ids[i] for batch in group["microbatches"] for i in batch]})
            events.append(event)
            stream.write(json.dumps(event, allow_nan=False) + "\n")
            stream.flush()
            print(f"update {number:02d}/20  loss={event['loss_mean']:.6f}  grad={event['grad_norm_before_clip']:.4f}  lr={event['lr_used']:.3g}", flush=True)
    synchronize(device)
    loop_seconds = time.perf_counter() - loop_started
    report["training"] = {
        "optimizer_step_calls": sum(x["optimizer_step_calls"] for x in events),
        "microbatches": sum(x["microbatches"] for x in events),
        "examples_seen": sum(x["examples_seen"] for x in events),
        "distinct_native_ids_seen": len({idx for x in events for idx in x["native_ids"]}),
        "forward_counts": dict(counter.counts["train"]),
        "diagnostic_train_loop_seconds": loop_seconds,
        "diagnostic_sum_update_seconds": sum(x["diagnostic_update_seconds"] for x in events),
        "diagnostic_mean_update_seconds": sum(x["diagnostic_update_seconds"] for x in events) / len(events),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "amp_skipped_updates": 0, "all_gradient_norms_finite": True,
        "classifier_max_abs_change": (model.classifier.weight.detach() - snapshots["classifier"]).abs().max().item(),
        "encoder_probe_max_abs_change": (model.bert.encoder.layer[0].attention.self.query.weight.detach() - snapshots["encoder_query"]).abs().max().item()}
    t = report["training"]
    if (t["optimizer_step_calls"], t["microbatches"], t["examples_seen"]) != (20, 40, 320):
        raise RuntimeError("Actual training accounting differs from the approved smoke workload.")
    if t["forward_counts"]["encoder_forward_calls"] != 40 or t["forward_counts"]["encoder_example_passes"] != 320:
        raise RuntimeError("Baseline forward accounting is inconsistent.")
    if t["classifier_max_abs_change"] <= 0 or t["encoder_probe_max_abs_change"] <= 0:
        raise RuntimeError("Classifier or encoder parameters did not change.")
    del snapshots
    record("twenty_updates", "20 optimizer calls, 40 single-forward microbatches, 320 sample visits; finite gradients and changed encoder/classifier.")

    metrics, before_logits = evaluate(model, tensors["validation"], protocol["eval_batch_size"], device, counter)
    labels = tensors["validation"]["labels"].tolist()
    predictions = before_logits.argmax(dim=-1).tolist()
    if abs(metrics["accuracy"] - accuracy_score(labels, predictions)) > 1e-12 or abs(metrics["f1"] - f1_score(labels, predictions, zero_division=0)) > 1e-12:
        raise RuntimeError("Metrics differ from scikit-learn.")
    report["diagnostic_dev_metrics"] = {**metrics, "split": "first_64_dev_rows", "formal_result": False}
    with (output / "predictions_dev_smoke.jsonl").open("w", encoding="utf-8") as stream:
        for row, logits, prediction in zip(rows["validation"], before_logits.tolist(), predictions):
            stream.write(json.dumps({"idx": row["idx"], "label": row["label"], "prediction": prediction, "logits": logits}) + "\n")
    record("dev_metrics", "64-row diagnostic dev evaluation agrees with sklearn Accuracy/F1; predictions retained.")

    checkpoint = output / "checkpoint-final"
    checkpoint.mkdir()
    save_started = time.perf_counter()
    model.save_pretrained(str(checkpoint))
    tokenizer.save_pretrained(str(checkpoint))
    final_state_sha = model_fingerprint(model)
    report["save_seconds"] = time.perf_counter() - save_started
    files = [{"name": p.name, "size": p.stat().st_size, "sha256": sha256_file(p)} for p in sorted(checkpoint.iterdir()) if p.is_file()]
    write_json(output / "checkpoint_manifest.json", {"path": str(checkpoint), "files": files,
                                                    "state_sha256": final_state_sha, "optimizer_saved": False,
                                                    "scope": "Inference reload only; optimizer/scheduler/RNG resume not tested."})
    report["forward_counts"] = counter.counts
    counter.close()
    del model, optimizer, scheduler, scaler
    gc.collect()
    torch.cuda.empty_cache()
    reload_started = time.perf_counter()
    reloaded, loading = BaselineBertForSequenceClassification.from_pretrained(str(checkpoint), local_files_only=True, output_loading_info=True)
    if loading["missing_keys"] or loading["unexpected_keys"] or loading["error_msgs"]:
        raise RuntimeError("Saved Baseline did not reload completely.")
    if model_fingerprint(reloaded) != final_state_sha:
        raise RuntimeError("Reloaded parameters differ from the trained checkpoint.")
    reloaded_tok = BertTokenizerFast.from_pretrained(str(checkpoint), local_files_only=True)
    enc = reloaded_tok([r["sentence1"] for r in rows["validation"]], [r["sentence2"] for r in rows["validation"]], **options)
    if any(enc[k] != tensors["validation"][k].tolist() for k in ("input_ids", "attention_mask", "token_type_ids")):
        raise RuntimeError("Reloaded tokenizer changed encodings.")
    reloaded.to(device)
    reload_counter = ForwardCounter(reloaded)
    after_metrics, after_logits = evaluate(reloaded, tensors["validation"], protocol["eval_batch_size"], device, reload_counter, "reload_eval")
    difference = (after_logits - before_logits).abs().max().item()
    if not torch.allclose(after_logits, before_logits, atol=1e-6, rtol=0) or after_logits.argmax(-1).tolist() != predictions:
        raise RuntimeError("Reloaded predictions/logits differ.")
    if any(after_metrics[k] != metrics[k] for k in ("accuracy", "f1", "tp", "tn", "fp", "fn")):
        raise RuntimeError("Reloaded metrics differ.")
    report["checkpoint_reload"] = {"state_sha256_equal": True, "tokenizer_equal": True,
                                   "max_abs_logit_difference": difference, "predictions_equal": True,
                                   "metrics_equal": True, "reload_seconds": time.perf_counter() - reload_started,
                                   "training_resume_tested": False, "forward_counts": reload_counter.counts}
    reload_counter.close()
    record("checkpoint_reload", "Saved model/tokenizer reload locally; exact state hash and matching 64-row logits/predictions/metrics.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    output = parser.parse_args().output_dir
    report = {"step": 4, "overall_status": "RUNNING", "checks": [], "purpose": "smoke_only_not_formal_results",
              "started_at_utc": datetime.now(timezone.utc).isoformat()}

    def record(name, detail):
        report["checks"].append({"name": name, "status": "PASS", "detail": detail})
        print(f"[PASS] {name}: {detail}", flush=True)
        write_json(output / "baseline_smoke_report.json", report)

    try:
        run(output, report, record)
        report["overall_status"] = "PASS"
    except Exception:
        report["overall_status"] = "FAIL"
        report["traceback"] = traceback.format_exc()
        print(report["traceback"], flush=True)
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "baseline_smoke_report.json", report)
    print("Baseline smoke: " + report["overall_status"], flush=True)
    return 0 if report["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
