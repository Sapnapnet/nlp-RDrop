"""Offline full MRPC baseline for the approved additional seeds 21 and 42."""
from __future__ import annotations
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.metadata as metadata
import inspect
import json
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from asset_utils import canonical, read_rows, sha256_file


def load_run_config(seed):
    if seed not in (21, 42):
        raise ValueError("Only the two approved additional seeds are allowed.")
    config = json.loads((ROOT / f"configs/baseline_seed{seed}.json").read_text(encoding="utf-8"))
    expected = json.loads((ROOT / "configs/baseline_seed13.json").read_text(encoding="utf-8"))
    expected.pop("validated_seed13_initial_state_sha256")
    expected["seed"] = seed
    if config != expected:
        raise RuntimeError("Additional seed settings must exactly match seed 13, except seed and its initial-state check.")
    return config


def nvidia_snapshot():
    commands = [
        ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu,power.limit", "--format=csv"],
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv"]]
    result = []
    for command in commands:
        try:
            completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10)
            result.append({"command": command, "returncode": completed.returncode,
                           "stdout": completed.stdout, "stderr": completed.stderr})
        except Exception as error:
            result.append({"command": command, "error": str(error)})
    return result


def run(output, report, seed):
    import torch
    import transformers
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
    from src.baseline_model import BaselineBertForSequenceClassification
    from src.training_core import seed_everything
    from src.experiment_runner import run_experiment, write_json

    protocol = json.loads((ROOT / "configs/training_protocol_v1.json").read_text(encoding="utf-8"))
    config = load_run_config(seed)
    inputs = json.loads((ROOT / "configs/input_protocol.json").read_text(encoding="utf-8"))
    if (config["seed"], config["max_updates"], config["warmup_updates"], config["train_rows"], config["dev_rows"]) != (seed, 2296, 137, 3668, 408):
        raise RuntimeError("This phase authorizes only full-data baseline seeds 21/42, 2296 updates and warmup 137.")
    if config["condition"] != "baseline" or config["resume_training"] or config["start_weights"] != inputs["model"]["local_path"]:
        raise RuntimeError("Start a fresh baseline from the fixed pretrained model.")
    for key in ("max_updates", "warmup_updates", "train_rows", "dev_rows"):
        if config[key] != protocol["formal"][key]:
            raise RuntimeError("Run configuration differs from the shared protocol: " + key)
    if sha256_file(ROOT / "provenance/assets.json") != protocol["input_manifest_sha256"]:
        raise RuntimeError("Input manifest changed.")
    expected_import = ROOT / "vendor/R-Drop/huggingface_transformer_src/src/transformers/__init__.py"
    source_sha = sha256_file(Path(inspect.getfile(BertForSequenceClassification)))
    if (Path(transformers.__file__).resolve() != expected_import.resolve() or transformers.__version__ != "4.6.0.dev0"
            or source_sha != "1316e7d6990cd9a12f08689dbb3cf0fcd5b165bcc981e519f46babe5f66cbf06"):
        raise RuntimeError("Expected the verified author fork with unchanged BERT algorithm source.")
    if str(torch.__version__) != "2.1.2+cu118" or torch.version.cuda != "11.8" or not torch.cuda.is_available():
        raise RuntimeError("Expected the validated PyTorch/CUDA environment.")
    versions = {}
    for item in json.loads((ROOT / "provenance/wheels.json").read_text(encoding="utf-8"))["wheels"]:
        versions[item["name"]] = metadata.version(item["name"])
        if versions[item["name"]] != item["version"]:
            raise RuntimeError("Pinned dependency version changed: " + item["name"])
    report["runtime"] = {"python": sys.version, "executable": sys.executable, "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), "cudnn": torch.backends.cudnn.version(),
        "transformers": transformers.__version__, "bert_source_sha256": source_sha, "package_versions": versions}
    report["resolved_config"] = {"protocol": protocol, "run": config, "inputs": inputs}
    report["identities"] = {"protocol_sha256": sha256_file(ROOT / "configs/training_protocol_v1.json"),
        "run_config_sha256": sha256_file(ROOT / f"configs/baseline_seed{seed}.json"),
        "input_manifest_sha256": sha256_file(ROOT / "provenance/assets.json"),
        "code_manifest_sha256": sha256_file(ROOT / "provenance/step6_bundle.json"),
        "training_core_sha256": sha256_file(ROOT / "src/training_core.py")}
    report["nvidia_before"] = nvidia_snapshot()
    from baseline_summary import gpu_identity, load_json
    anchor = load_json(ROOT / "provenance/step6_seed13_anchor.json")
    if gpu_identity(report["nvidia_before"]) != anchor["gpu_identity"]:
        raise RuntimeError("GPU UUID/driver/power limit differs from the accepted seed 13 run.")
    torch.set_num_threads(4)
    seed_everything(config["seed"])
    rows = {s: read_rows(ROOT / f"data/mrpc/{s}.jsonl") for s in ("train", "validation")}
    if len(rows["train"]) != 3668 or len(rows["validation"]) != 408:
        raise RuntimeError("Use the complete fixed train/dev split.")
    model_dir = ROOT / inputs["model"]["local_path"]
    tokenizer = BertTokenizerFast.from_pretrained(str(model_dir), local_files_only=True)
    options = {k: v for k, v in inputs["tokenization"].items() if k not in ("class", "do_lower_case")}
    keys = ("input_ids", "attention_mask", "token_type_ids")
    expected_encodings = {"train": "e3386878dc71f69f76c65c3a6bed847b755e9a4709ccd802303ecf0527df7d93",
                          "validation": "29f536e548417d0bb2bc37eade5b2e7359242947fbbd3f72425057b2614b228e"}
    tensors, native_ids = {}, {}
    report["encoding_sha256"] = {}
    for split, data in rows.items():
        enc = tokenizer([r["sentence1"] for r in data], [r["sentence2"] for r in data], **options)
        identity = hashlib.sha256(canonical({k: enc[k] for k in keys}).encode("utf-8")).hexdigest()
        if identity != expected_encodings[split]:
            raise RuntimeError("Full-data encoding differs from step 3: " + split)
        report["encoding_sha256"][split] = identity
        tensors[split] = {k: torch.tensor(enc[k], dtype=torch.long) for k in keys}
        tensors[split]["labels"] = torch.tensor([r["label"] for r in data], dtype=torch.long)
        native_ids[split] = [r["idx"] for r in data]
    model_config = BertConfig.from_pretrained(str(model_dir), local_files_only=True, num_labels=2)
    model_config.id2label = {0: "not_equivalent", 1: "equivalent"}
    model_config.label2id = {v: k for k, v in model_config.id2label.items()}
    model, loading = BaselineBertForSequenceClassification.from_pretrained(str(model_dir), config=model_config,
                                                                         local_files_only=True, output_loading_info=True)
    if set(loading["missing_keys"]) != {"classifier.weight", "classifier.bias"} or loading["error_msgs"] or any(not k.startswith("cls.") for k in loading["unexpected_keys"]):
        raise RuntimeError("Unexpected pretrained loading mismatch.")
    report["pretrained_loading"] = loading

    def check_reloaded_tokenizer(checkpoint):
        tok = BertTokenizerFast.from_pretrained(str(checkpoint), local_files_only=True)
        dev = rows["validation"]
        enc = tok([r["sentence1"] for r in dev], [r["sentence2"] for r in dev], **options)
        if any(enc[k] != tensors["validation"][k].tolist() for k in keys):
            raise RuntimeError("Saved tokenizer changed dev encodings.")

    print(f"Validated complete MRPC encodings and pretrained source. Starting seed {seed}.", flush=True)
    run_experiment(model, tokenizer, tensors, native_ids, output, protocol, config, torch.device("cuda:0"), report,
                   reload_tokenizer_check=check_reloaded_tokenizer)
    predictions = read_rows(output / "predictions_dev.jsonl")
    labels, predicted = [r["label"] for r in predictions], [r["prediction"] for r in predictions]
    metrics = report["dev_metrics"]
    if metrics["accuracy"] != accuracy_score(labels, predicted) or abs(metrics["f1"] - f1_score(labels, predicted, zero_division=0)) > 1e-12:
        raise RuntimeError("Full-dev metrics differ from sklearn.")
    report["metrics_verified_with_sklearn"] = True
    report["nvidia_after"] = nvidia_snapshot()
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "formal_report.json", report)
    write_json(output / "result.json", {
        "status": "PASS", "condition": "baseline", "seed": seed, "protocol_id": protocol["protocol_id"],
        "protocol_sha256": report["identities"]["protocol_sha256"], "code_manifest_sha256": report["identities"]["code_manifest_sha256"],
        "input_manifest_sha256": report["identities"]["input_manifest_sha256"], "dev_examples": len(predictions),
        "accuracy": metrics["accuracy"], "f1": metrics["f1"],
        "optimizer_updates": report["training"]["optimizer_step_calls"],
        "train_loop_wall_seconds": report["training"]["train_loop_wall_seconds"],
        "mean_wall_seconds_per_update": report["training"]["mean_wall_seconds_per_update"],
        "peak_allocated_bytes": report["training"]["peak_allocated_bytes"],
        "peak_reserved_bytes": report["training"]["peak_reserved_bytes"],
        "encoder_forward_calls": report["training"]["forward_counts"]["encoder_forward_calls"],
        "encoder_example_passes": report["training"]["forward_counts"]["encoder_example_passes"],
        "examples_seen": report["training"]["examples_seen"], "checkpoint_selection": "last_update",
        "statistical_scope": "One seed only; no multi-seed standard deviation yet."})
    print(f"Complete dev: Accuracy={100*metrics['accuracy']:.4f}%  F1={100*metrics['f1']:.4f}%", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(21, 42), required=True)
    args = parser.parse_args()
    output, seed = args.output_dir, args.seed
    from src.experiment_runner import write_json
    report = {"step": 6, "condition": "baseline", "seed": seed, "purpose": "formal_baseline_single_seed",
              "overall_status": "RUNNING", "phase": "preparing", "started_at_utc": datetime.now(timezone.utc).isoformat()}
    try:
        run(output, report, seed)
    except BaseException:
        report["overall_status"] = "FAIL"
        report["traceback"] = traceback.format_exc()
        print(report["traceback"], flush=True)
        write_json(output / "formal_report.json", report)
        write_json(output / "progress.json", {"status": "FAIL", "completed_updates": report.get("completed_updates", 0)})
    return 0 if report["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
