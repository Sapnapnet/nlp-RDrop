"""Two-update real-BERT R-Drop engineering preflight.

This is NOT a formal experiment.
It reuses A's frozen inputs, training utilities and schedule.
Only the first two updates of the full 2296-update plan are executed.
"""

from __future__ import annotations

import os

# Set offline/runtime variables before importing torch or transformers.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("OMP_NUM_THREADS", "4")

from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import importlib.metadata as metadata
import inspect
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]

os.environ.setdefault("HF_HOME", str(ROOT / "cache/huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(ROOT / "cache/transformers"))

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch
import transformers

from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizerFast,
)

from asset_utils import (
    canonical,
    read_rows,
    sha256_file,
    verify_assets,
)

from src.rdrop_model import RDropBertForSequenceClassification

from src.training_core import (
    ForwardCounter,
    evaluate,
    make_optimizer,
    make_scheduler,
    make_update_plan,
    model_fingerprint,
    plan_identity,
    seed_everything,
    train_update,
    synchronize,
)

from src.experiment_runner import write_json


EXPECTED_BERT_SOURCE_SHA256 = (
    "1316e7d6990cd9a12f08689dbb3cf0fcd5b165bcc981e519"
    "f46babe5f66cbf06"
)

EXPECTED_ENCODINGS = {
    "train": (
        "e3386878dc71f69f76c65c3a6bed847b755e9a4709ccd802303"
        "ecf0527df7d93"
    ),
    "validation": (
        "29f536e548417d0bb2bc37eade5b2e7359242947fbbd3f724"
        "25057b2614b228e"
    ),
}

SEED = 13
ALPHA = 0.0
PREFLIGHT_UPDATES = 2
EXPECTED_INITIAL_SHA256 = (
    "55e6adb49031efc8018f7d2733471c77"
    "c5f2df5df61aed15b20223b51f7499bd"
)


def nvidia_snapshot():
    commands = [
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,"
            "memory.total,memory.used,utilization.gpu",
            "--format=csv",
        ],
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv",
        ],
    ]

    results = []

    for command in commands:
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
            )

            results.append({
                "command": command,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            })

        except Exception as error:
            results.append({
                "command": command,
                "error": str(error),
            })

    return results


def verify_environment(report):
    if (
        platform.system() != "Linux"
        or platform.machine() != "x86_64"
        or sys.version_info[:2] != (3, 10)
    ):
        raise RuntimeError("Expected Linux x86_64 and Python 3.10.")

    if Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        raise RuntimeError("Use the project's .venv Python.")

    if os.environ.get("PYTHONHASHSEED") != "13":
        raise RuntimeError("PYTHONHASHSEED must be set to 13 before Python starts.")

    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be :4096:8.")

    expected_import = (
        ROOT
        / "vendor/R-Drop/huggingface_transformer_src/src/transformers/__init__.py"
    )

    source_path = Path(inspect.getfile(BertForSequenceClassification))
    source_sha = sha256_file(source_path)

    if (
        Path(transformers.__file__).resolve() != expected_import.resolve()
        or transformers.__version__ != "4.6.0.dev0"
        or source_sha != EXPECTED_BERT_SOURCE_SHA256
    ):
        raise RuntimeError("Author Transformers source differs from the verified version.")

    if (
        str(torch.__version__) != "2.1.2+cu118"
        or torch.version.cuda != "11.8"
        or not torch.cuda.is_available()
    ):
        raise RuntimeError("Expected PyTorch 2.1.2+cu118 and available CUDA.")

    if not hasattr(BertForSequenceClassification, "bert_kl"):
        raise RuntimeError("Pinned author's bert_kl implementation is unavailable.")

    if not hasattr(BertForSequenceClassification, "bert_cross_entropy"):
        raise RuntimeError("Pinned author's bert_cross_entropy is unavailable.")

    versions = {}

    wheel_manifest = json.loads(
        (ROOT / "provenance/wheels.json").read_text(encoding="utf-8")
    )

    for item in wheel_manifest["wheels"]:
        actual = metadata.version(item["name"])

        if actual != item["version"]:
            raise RuntimeError(
                f"Pinned dependency version changed: {item['name']}"
            )

        versions[item["name"]] = actual

    report["runtime"] = {
        "python": sys.version,
        "executable": sys.executable,
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "cudnn": torch.backends.cudnn.version(),
        "transformers": transformers.__version__,
        "transformers_path": str(Path(transformers.__file__).resolve()),
        "bert_source_path": str(source_path.resolve()),
        "bert_source_sha256": source_sha,
        "package_versions": versions,
    }


def verify_frozen_files():
    bundle_path = ROOT / "provenance/step5_bundle.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))

    for item in bundle["files"]:
        path = (ROOT / item["path"]).resolve()

        if (
            not path.is_relative_to(ROOT.resolve())
            or not path.is_file()
            or sha256_file(path) != item["sha256"]
        ):
            raise RuntimeError(
                "Original A code bundle changed: " + item["path"]
            )

    return sha256_file(bundle_path)


def load_fixed_inputs(protocol, baseline_config, inputs, report):
    if (
        protocol["formal"]["max_updates"] != 2296
        or protocol["formal"]["warmup_updates"] != 137
        or protocol["formal"]["train_rows"] != 3668
        or protocol["formal"]["dev_rows"] != 408
    ):
        raise RuntimeError("Frozen formal training workload changed.")

    if (
        protocol["microbatch_size"] != 8
        or protocol["gradient_accumulation_steps"] != 2
        or protocol["train_precision"] != "fp16_amp"
        or protocol["eval_precision"] != "fp32"
        or protocol["eval_batch_size"] != 16
        or protocol["max_grad_norm"] != 1.0
    ):
        raise RuntimeError("Frozen training protocol changed.")

    if protocol["loss_contract"]["alpha_values"] != [0.0, 0.5, 1.0]:
        raise RuntimeError("Frozen R-Drop alpha values changed.")

    if (
        baseline_config["condition"] != "baseline"
        or baseline_config["seed"] != SEED
        or baseline_config["start_weights"] != inputs["model"]["local_path"]
        or baseline_config["validated_seed13_initial_state_sha256"]
        != EXPECTED_INITIAL_SHA256
    ):
        raise RuntimeError("Baseline Seed-13 reference configuration changed.")

    manifest_path = ROOT / "provenance/assets.json"

    if sha256_file(manifest_path) != protocol["input_manifest_sha256"]:
        raise RuntimeError("Input manifest SHA256 mismatch.")

    asset_audit = verify_assets(ROOT)

    report["asset_verification"] = asset_audit

    rows = {
        split: read_rows(ROOT / f"data/mrpc/{split}.jsonl")
        for split in ("train", "validation")
    }

    if len(rows["train"]) != 3668 or len(rows["validation"]) != 408:
        raise RuntimeError("MRPC full-data row count mismatch.")

    model_dir = ROOT / inputs["model"]["local_path"]

    if not model_dir.is_dir():
        raise RuntimeError("Local BERT model directory does not exist.")

    tokenizer = BertTokenizerFast.from_pretrained(
        str(model_dir),
        local_files_only=True,
    )

    options = {
        key: value
        for key, value in inputs["tokenization"].items()
        if key not in ("class", "do_lower_case")
    }

    keys = ("input_ids", "attention_mask", "token_type_ids")

    tensors = {}
    native_ids = {}
    encoding_hashes = {}

    for split, data in rows.items():
        enc = tokenizer(
            [row["sentence1"] for row in data],
            [row["sentence2"] for row in data],
            **options,
        )

        identity = hashlib.sha256(
            canonical({key: enc[key] for key in keys}).encode("utf-8")
        ).hexdigest()

        if identity != EXPECTED_ENCODINGS[split]:
            raise RuntimeError(
                f"MRPC {split} tokenization SHA256 mismatch."
            )

        encoding_hashes[split] = identity

        tensors[split] = {
            key: torch.tensor(enc[key], dtype=torch.long)
            for key in keys
        }

        tensors[split]["labels"] = torch.tensor(
            [row["label"] for row in data],
            dtype=torch.long,
        )

        native_ids[split] = [row["idx"] for row in data]

    report["encoding_sha256"] = encoding_hashes

    return rows, tokenizer, options, tensors, native_ids, model_dir


def load_rdrop_model(model_dir, report):
    model_config = BertConfig.from_pretrained(
        str(model_dir),
        local_files_only=True,
        num_labels=2,
    )

    model_config.id2label = {
        0: "not_equivalent",
        1: "equivalent",
    }

    model_config.label2id = {
        value: key for key, value in model_config.id2label.items()
    }

    model_config.rdrop_alpha = ALPHA

    model, loading = (
        RDropBertForSequenceClassification.from_pretrained(
            str(model_dir),
            config=model_config,
            local_files_only=True,
            output_loading_info=True,
        )
    )

    if (
        set(loading["missing_keys"])
        != {"classifier.weight", "classifier.bias"}
        or loading["error_msgs"]
        or any(not key.startswith("cls.") for key in loading["unexpected_keys"])
    ):
        raise RuntimeError("Unexpected pretrained model loading mismatch.")

    if model.rdrop_alpha != ALPHA:
        raise RuntimeError("R-Drop alpha initialization mismatch.")

    initial_sha = model_fingerprint(model)

    if initial_sha != EXPECTED_INITIAL_SHA256:
        raise RuntimeError(
            "R-Drop Seed-13 initial weights differ from Baseline."
        )

    report["pretrained_loading"] = loading
    report["initial_state_sha256"] = initial_sha

    return model


def verify_reloaded_tokenizer(checkpoint, rows, tensors, options):
    tokenizer = BertTokenizerFast.from_pretrained(
        str(checkpoint),
        local_files_only=True,
    )

    dev = rows["validation"]

    encoded = tokenizer(
        [row["sentence1"] for row in dev],
        [row["sentence2"] for row in dev],
        **options,
    )

    for key in ("input_ids", "attention_mask", "token_type_ids"):
        if encoded[key] != tensors["validation"][key].tolist():
            raise RuntimeError("Reloaded tokenizer changed validation encodings.")


def run_preflight(output, report):
    protocol = json.loads(
        (ROOT / "configs/training_protocol_v1.json").read_text(encoding="utf-8")
    )

    baseline_config = json.loads(
        (ROOT / "configs/baseline_seed13.json").read_text(encoding="utf-8")
    )

    inputs = json.loads(
        (ROOT / "configs/input_protocol.json").read_text(encoding="utf-8")
    )

    report["resolved_config"] = {
        "protocol": protocol,
        "baseline_reference": baseline_config,
        "inputs": inputs,
        "preflight": {
            "seed": SEED,
            "alpha": ALPHA,
            "updates_executed": PREFLIGHT_UPDATES,
            "schedule_total_updates": protocol["formal"]["max_updates"],
            "schedule_warmup_updates": protocol["formal"]["warmup_updates"],
            "purpose": "engineering_preflight_not_formal_result",
        },
    }

    report["identities"] = {
        "protocol_sha256": sha256_file(
            ROOT / "configs/training_protocol_v1.json"
        ),
        "baseline_seed13_config_sha256": sha256_file(
            ROOT / "configs/baseline_seed13.json"
        ),
        "input_manifest_sha256": sha256_file(
            ROOT / "provenance/assets.json"
        ),
        "code_manifest_sha256": verify_frozen_files(),
        "training_core_sha256": sha256_file(
            ROOT / "src/training_core.py"
        ),
        "experiment_runner_sha256": sha256_file(
            ROOT / "src/experiment_runner.py"
        ),
        "rdrop_model_sha256": sha256_file(
            ROOT / "src/rdrop_model.py"
        ),
        "preflight_script_sha256": sha256_file(__file__),
    }

    report["nvidia_before"] = nvidia_snapshot()

    torch.set_num_threads(4)
    seed_everything(SEED)

    (
        rows,
        tokenizer,
        options,
        tensors,
        native_ids,
        model_dir,
    ) = load_fixed_inputs(
        protocol,
        baseline_config,
        inputs,
        report,
    )

    model = load_rdrop_model(model_dir, report)

    device = torch.device("cuda:0")
    model.to(device)

    # Generate the original full 2296-update plan and take its exact prefix.
    full_plan = make_update_plan(
        len(native_ids["train"]),
        protocol["microbatch_size"],
        protocol["gradient_accumulation_steps"],
        protocol["formal"]["max_updates"],
        SEED,
    )

    plan = full_plan[:PREFLIGHT_UPDATES]

    if len(plan) != PREFLIGHT_UPDATES:
        raise RuntimeError("Could not construct the preflight update plan.")

    for group in plan:
        if (
            len(group["microbatches"]) != 2
            or any(len(batch) != 8 for batch in group["microbatches"])
        ):
            raise RuntimeError("Unexpected preflight microbatch structure.")

    selected_indices = [
        index
        for group in plan
        for batch in group["microbatches"]
        for index in batch
    ]

    selected_native_ids = [
        native_ids["train"][index]
        for index in selected_indices
    ]

    if len(selected_native_ids) != 32 or len(set(selected_native_ids)) != 32:
        raise RuntimeError("Preflight sample plan contains unexpected duplicates.")

    report["planned_workload"] = {
        "optimizer_updates": len(plan),
        "microbatches": sum(len(group["microbatches"]) for group in plan),
        "example_visits": len(selected_indices),
        "distinct_native_ids": len(set(selected_native_ids)),
        "encoder_forwards_expected": 8,
        "encoder_example_passes_expected": 64,
    }

    report["batch_plan_prefix_sha256"] = plan_identity(
        plan,
        native_ids["train"],
    )

    optimizer = make_optimizer(model, protocol["optimizer"])

    # IMPORTANT:
    # Use the full formal schedule, not a shortened 2-step schedule.
    scheduler = make_scheduler(
        optimizer,
        protocol["formal"]["max_updates"],
        protocol["formal"]["warmup_updates"],
        protocol["scheduler"]["end_learning_rate"],
    )

    use_amp = True

    scaler = torch.cuda.amp.GradScaler(
        enabled=use_amp,
        init_scale=protocol["amp_init_scale"],
        growth_interval=protocol["amp_growth_interval"],
    )

    report["effective_precision"] = "fp16_amp"

    classifier_before = model.classifier.weight.detach().clone()

    query_before = (
        model.bert.encoder.layer[0]
        .attention.self.query.weight.detach().clone()
    )

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)

    counter = ForwardCounter(model)
    events = []

    training_started = time.perf_counter()

    try:
        with (output / "training_updates.jsonl").open(
            "w",
            encoding="utf-8",
        ) as stream:

            for number, group in enumerate(plan, 1):

                event = train_update(
                    model,
                    tensors["train"],
                    group,
                    optimizer,
                    scheduler,
                    scaler,
                    device,
                    protocol["max_grad_norm"],
                    use_amp,
                    2,
                    counter,
                )

                event.update({
                    "update": number,
                    "epoch": group["epoch"],
                    "native_ids": [
                        native_ids["train"][index]
                        for batch in group["microbatches"]
                        for index in batch
                    ],
                })

                stream.write(
                    json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n"
                )
                stream.flush()

                events.append(event)

                print(
                    f"Preflight update {number}/{PREFLIGHT_UPDATES}: "
                    f"loss={event['loss_mean']:.6f}, "
                    f"lr={event['lr_used']:.10g}, "
                    f"forwards={event['encoder_forward_calls']}",
                    flush=True,
                )

        synchronize(device)
        training_seconds = time.perf_counter() - training_started

        counts = dict(counter.counts["train"])

        if sum(e["optimizer_step_calls"] for e in events) != 2:
            raise RuntimeError("Expected exactly two optimizer.step calls.")

        if sum(e["microbatches"] for e in events) != 4:
            raise RuntimeError("Expected exactly four training microbatches.")

        if sum(e["examples_seen"] for e in events) != 32:
            raise RuntimeError("Expected exactly 32 training example visits.")

        if counts["encoder_forward_calls"] != 8:
            raise RuntimeError("Expected exactly eight encoder forwards.")

        if counts["encoder_example_passes"] != 64:
            raise RuntimeError("Expected exactly 64 encoder example passes.")

        expected_second_lr = (
            protocol["optimizer"]["learning_rate"]
            / protocol["formal"]["warmup_updates"]
        )

        if not math.isclose(events[0]["lr_used"], 0.0, abs_tol=1e-15):
            raise RuntimeError("First warmup learning rate should be zero.")

        if not math.isclose(
            events[1]["lr_used"],
            expected_second_lr,
            rel_tol=1e-7,
            abs_tol=1e-15,
        ):
            raise RuntimeError("Second warmup learning rate mismatch.")

        classifier_change = (
            model.classifier.weight.detach() - classifier_before
        ).abs().max().item()

        query_change = (
            model.bert.encoder.layer[0].attention.self.query.weight.detach()
            - query_before
        ).abs().max().item()

        if classifier_change <= 0 or query_change <= 0:
            raise RuntimeError(
                "Classifier or BERT encoder did not change after the nonzero-LR update."
            )

        if any(
            not math.isfinite(event["grad_norm_before_clip"])
            for event in events
        ):
            raise RuntimeError("Nonfinite gradient norm detected.")

        memory = {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }

        report["training"] = {
            "optimizer_step_calls": 2,
            "microbatches": 4,
            "examples_seen": 32,
            "distinct_native_ids_seen": len(set(selected_native_ids)),
            "forward_counts": counts,
            "train_loop_wall_seconds": training_seconds,
            "mean_wall_seconds_per_update": training_seconds / 2,
            "sum_update_body_seconds": sum(
                event["diagnostic_update_seconds"]
                for event in events
            ),
            "learning_rates_used": [
                event["lr_used"] for event in events
            ],
            "amp_scales": [
                {
                    "before": event["amp_scale_before"],
                    "after": event["amp_scale_after"],
                }
                for event in events
            ],
            "amp_skipped_updates": 0,
            "all_gradient_norms_finite": True,
            "classifier_max_abs_change": classifier_change,
            "encoder_probe_max_abs_change": query_change,
            **memory,
        }

        report["phase"] = "evaluating_preflight_checkpoint"
        write_json(output / "preflight_report.json", report)

        metrics, logits = evaluate(
            model,
            tensors["validation"],
            protocol["eval_batch_size"],
            device,
            counter,
            phase="preflight_eval",
        )

        predictions = logits.argmax(dim=-1).tolist()
        labels = tensors["validation"]["labels"].tolist()

        if (
            metrics["accuracy"] != accuracy_score(labels, predictions)
            or abs(
                metrics["f1"]
                - f1_score(labels, predictions, zero_division=0)
            ) > 1e-12
        ):
            raise RuntimeError("Preflight metrics differ from sklearn.")

        report["dev_metrics"] = {
            **metrics,
            "split": "validation",
            "purpose": "engineering_preflight_only_not_formal_result",
        }

        report["metrics_verified_with_sklearn"] = True

        with (output / "predictions_dev.jsonl").open(
            "w",
            encoding="utf-8",
        ) as stream:
            for idx, label, prediction, values in zip(
                native_ids["validation"],
                labels,
                predictions,
                logits.tolist(),
            ):
                stream.write(
                    json.dumps({
                        "idx": idx,
                        "label": label,
                        "prediction": prediction,
                        "logits": values,
                    }, allow_nan=False) + "\n"
                )

        report["forward_counts"] = counter.counts

    finally:
        counter.close()

    # Save the two-update checkpoint separately from formal experiments.
    checkpoint = output / "checkpoint"
    checkpoint.mkdir(exist_ok=False)

    save_started = time.perf_counter()

    model.save_pretrained(str(checkpoint))
    tokenizer.save_pretrained(str(checkpoint))

    state_sha = model_fingerprint(model)

    report["checkpoint_save_and_hash_seconds"] = (
        time.perf_counter() - save_started
    )

    report["final_state_sha256"] = state_sha

    report["checkpoint_manifest"] = {
        "state_sha256": state_sha,
        "optimizer_saved": False,
        "files": [
            {
                "name": path.name,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(checkpoint.iterdir())
            if path.is_file()
        ],
    }

    write_json(
        output / "checkpoint_manifest.json",
        report["checkpoint_manifest"],
    )

    # Release the original GPU model before loading the saved copy.
    model.to("cpu")

    del optimizer, scheduler, scaler
    gc.collect()
    torch.cuda.empty_cache()

    report["phase"] = "reloading_preflight_checkpoint"
    write_json(output / "preflight_report.json", report)

    model_class = type(model)

    restored, loading = model_class.from_pretrained(
        str(checkpoint),
        local_files_only=True,
        output_loading_info=True,
    )

    if (
        loading["missing_keys"]
        or loading["unexpected_keys"]
        or loading["error_msgs"]
        or model_fingerprint(restored) != state_sha
    ):
        raise RuntimeError("Preflight checkpoint did not reload exactly.")

    if restored.rdrop_alpha != ALPHA:
        raise RuntimeError("Checkpoint did not preserve R-Drop alpha.")

    verify_reloaded_tokenizer(
        checkpoint,
        rows,
        tensors,
        options,
    )

    restored.to(device)

    second_counter = ForwardCounter(restored)

    try:
        after_metrics, after_logits = evaluate(
            restored,
            tensors["validation"],
            protocol["eval_batch_size"],
            device,
            second_counter,
            phase="reload_eval",
        )
    finally:
        second_counter.close()

    if not torch.allclose(logits, after_logits, atol=1e-6, rtol=0):
        raise RuntimeError("Checkpoint reload changed validation logits.")

    if after_logits.argmax(dim=-1).tolist() != predictions:
        raise RuntimeError("Checkpoint reload changed validation predictions.")

    for key in ("accuracy", "f1", "tp", "tn", "fp", "fn"):
        if metrics[key] != after_metrics[key]:
            raise RuntimeError("Checkpoint reload changed evaluation metrics.")

    report["checkpoint_reload"] = {
        "state_sha256_equal": True,
        "predictions_equal": True,
        "metrics_equal": True,
        "max_abs_logit_difference": (
            logits - after_logits
        ).abs().max().item(),
        "tokenizer_checked": True,
        "rdrop_alpha_restored": restored.rdrop_alpha,
        "forward_counts": second_counter.counts,
    }

    report["nvidia_after"] = nvidia_snapshot()
    report["overall_status"] = "PASS"
    report["phase"] = "complete"
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()

    write_json(output / "preflight_report.json", report)

    write_json(
        output / "result.json",
        {
            "status": "PASS",
            "condition": "rdrop_alpha0",
            "seed": SEED,
            "alpha": ALPHA,
            "purpose": "engineering_preflight_not_formal_experiment",
            "initial_state_sha256": report["initial_state_sha256"],
            "final_state_sha256": state_sha,
            "optimizer_updates": 2,
            "microbatches": 4,
            "examples_seen": 32,
            "encoder_forward_calls": 8,
            "encoder_example_passes": 64,
            "dev_accuracy": metrics["accuracy"],
            "dev_f1": metrics["f1"],
        },
    )

    print("\n========== R-Drop PREFLIGHT PASS ==========", flush=True)
    print("Output:", output, flush=True)
    print("Initial SHA256:", report["initial_state_sha256"], flush=True)
    print("Final SHA256:", state_sha, flush=True)
    print("Dev Accuracy:", metrics["accuracy"], flush=True)
    print("Dev F1:", metrics["f1"], flush=True)


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "reports/rdrop_preflight",
    )

    args = parser.parse_args()

    if shutil_disk_free(ROOT) < 2 * 1024**3:
        raise RuntimeError("Keep at least 2 GiB free before preflight.")

    lock_path = ROOT / "runs/gpu_training.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    lock = lock_path.open("a+")

    try:
        fcntl.flock(
            lock.fileno(),
            fcntl.LOCK_EX | fcntl.LOCK_NB,
        )
    except BlockingIOError:
        lock.close()
        print(
            "ABORT: another project training job holds gpu_training.lock.",
            flush=True,
        )
        return 2

    output = None

    try:
        # Do not compete with another CUDA compute process.
        process_check = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )

        if process_check.returncode == 0 and process_check.stdout.strip():
            raise RuntimeError(
                "Another GPU compute process is present:\n"
                + process_check.stdout
            )

        args.output_root.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = args.output_root / f"seed13_alpha0_{stamp}"
        output.mkdir(exist_ok=False)

        report = {
            "step": "rdrop_preflight",
            "condition": "rdrop_alpha0",
            "seed": SEED,
            "alpha": ALPHA,
            "overall_status": "RUNNING",
            "phase": "preparing",
            "output_directory": str(output),
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        }

        write_json(output / "preflight_report.json", report)

        try:
            verify_environment(report)
            run_preflight(output, report)

        except BaseException:
            report["overall_status"] = "FAIL"
            report["phase"] = "failed"
            report["traceback"] = traceback.format_exc()
            print(report["traceback"], flush=True)

            write_json(output / "preflight_report.json", report)

        return 0 if report["overall_status"] == "PASS" else 1

    finally:
        lock.close()


def shutil_disk_free(path):
    import shutil
    return shutil.disk_usage(path).free


if __name__ == "__main__":
    raise SystemExit(main())
