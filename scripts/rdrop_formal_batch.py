"""Sequential, auditable nine-run R-Drop MRPC formal experiment."""
from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import importlib.metadata as metadata
import inspect
import json
import math
import shutil
import statistics
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from asset_utils import canonical, read_rows, sha256_file
from src.experiment_runner import run_experiment, write_json
from src.training_core import seed_everything, model_fingerprint

ALPHAS = (0.0, 0.5, 1.0)
SEEDS = (13, 21, 42)

EXPECTED = {
    "updates": 2296,
    "microbatches": 4583,
    "examples": 36628,
    "distinct_ids": 3668,
    "encoder_forwards": 9166,
    "encoder_passes": 73256,
    "padded_positions": 9376768,
}

EXPECTED_ENCODING = {
    "train": "e3386878dc71f69f76c65c3a6bed847b755e9a4709ccd802303ecf0527df7d93",
    "validation": "29f536e548417d0bb2bc37eade5b2e7359242947fbbd3f72425057b2614b228e",
}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha_json(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def nvidia_snapshot():
    commands = [
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu,power.limit",
            "--format=csv",
        ],
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv",
        ],
    ]
    result = []
    for command in commands:
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=10
            )
            result.append({
                "command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            })
        except Exception as error:
            result.append({"command": command, "error": str(error)})
    return result


def code_manifest():
    files = [
        "scripts/rdrop_formal_batch.py",
        "scripts/asset_utils.py",
        "src/rdrop_model.py",
        "src/training_core.py",
        "src/experiment_runner.py",
    ]
    entries = []
    for name in files:
        path = ROOT / name
        if not path.is_file():
            raise RuntimeError("Missing code file: " + name)
        entries.append({
            "path": name,
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return sha_json(entries), entries


def verify_shared_resources():
    import torch
    import transformers
    from transformers import BertForSequenceClassification
    from baseline_summary import gpu_identity

    protocol = load_json(ROOT / "configs/training_protocol_v1.json")
    inputs = load_json(ROOT / "configs/input_protocol.json")

    if protocol["formal"]["max_updates"] != EXPECTED["updates"]:
        raise RuntimeError("Unexpected max_updates.")
    if protocol["formal"]["warmup_updates"] != 137:
        raise RuntimeError("Unexpected warmup_updates.")
    if protocol["formal"]["train_rows"] != EXPECTED["distinct_ids"]:
        raise RuntimeError("Unexpected training row count.")
    if protocol["formal"]["dev_rows"] != 408:
        raise RuntimeError("Unexpected validation row count.")
    if protocol["loss_contract"]["alpha_values"] != [0.0, 0.5, 1.0]:
        raise RuntimeError("R-Drop alpha values differ from frozen protocol.")

    for alpha in ALPHAS:
        condition = f"rdrop_alpha{alpha:g}"
        if condition not in protocol["conditions"]:
            raise RuntimeError("Condition missing from frozen protocol: " + condition)

    if sha256_file(ROOT / "provenance/assets.json") != protocol["input_manifest_sha256"]:
        raise RuntimeError("Frozen input manifest changed.")

    gpu_before = nvidia_snapshot()

    if str(torch.__version__) != "2.1.2+cu118":
        raise RuntimeError("Unexpected PyTorch version.")
    if torch.version.cuda != "11.8" or not torch.cuda.is_available():
        raise RuntimeError("Expected CUDA 11.8 and an available GPU.")
    if torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 3090":
        raise RuntimeError("Expected the validated RTX 3090.")

    expected_import = (
        ROOT / "vendor/R-Drop/huggingface_transformer_src/src/transformers/__init__.py"
    ).resolve()
    source_sha = sha256_file(Path(inspect.getfile(BertForSequenceClassification)))

    if (
        Path(transformers.__file__).resolve() != expected_import
        or transformers.__version__ != "4.6.0.dev0"
        or source_sha != "1316e7d6990cd9a12f08689dbb3cf0fcd5b165bcc981e519f46babe5f66cbf06"
    ):
        raise RuntimeError("Pinned author Transformers source mismatch.")

    versions = {}
    wheel_manifest = load_json(ROOT / "provenance/wheels.json")
    for item in wheel_manifest["wheels"]:
        actual = metadata.version(item["name"])
        if actual != item["version"]:
            raise RuntimeError("Dependency version mismatch: " + item["name"])
        versions[item["name"]] = actual

    baseline = load_json(ROOT / "reports/summary/baseline_results.json")
    if baseline.get("condition") != "baseline":
        raise RuntimeError("Baseline reference summary is missing or invalid.")

    baseline_by_seed = {int(r["seed"]): r for r in baseline["results"]}
    if set(baseline_by_seed) != set(SEEDS):
        raise RuntimeError("Baseline must contain exactly seeds 13, 21 and 42.")

    # Verify seed-specific configurations have not changed.
    seed13_config = load_json(ROOT / "configs/baseline_seed13.json")
    for seed in SEEDS:
        actual = load_json(ROOT / f"configs/baseline_seed{seed}.json")
        expected = dict(seed13_config)
        if seed != 13:
            expected.pop("validated_seed13_initial_state_sha256")
            expected["seed"] = seed
        if actual != expected:
            raise RuntimeError(f"Baseline seed-{seed} configuration changed.")

    compute_lines = [
        line.strip()
        for line in gpu_before[1].get("stdout", "").splitlines()
        if line.strip()
    ]
    if len(compute_lines) > 1:
        raise RuntimeError("Another GPU compute process is active. Stop before training.")

    anchor = load_json(ROOT / "provenance/step6_seed13_anchor.json")
    if gpu_identity(gpu_before) != anchor["gpu_identity"]:
        raise RuntimeError("GPU identity differs from the accepted Baseline GPU.")

    free_bytes = shutil.disk_usage(ROOT).free
    if free_bytes < 8 * 1024**3:
        raise RuntimeError("Less than 8 GiB disk space remains.")

    torch.set_num_threads(4)

    return {
        "protocol": protocol,
        "inputs": inputs,
        "baseline": baseline,
        "baseline_by_seed": baseline_by_seed,
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "torch": str(torch.__version__),
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "cudnn": torch.backends.cudnn.version(),
            "transformers": transformers.__version__,
            "bert_source_sha256": source_sha,
            "package_versions": versions,
        },
        "gpu_before": gpu_before,
        "gpu_identity": anchor["gpu_identity"],
    }


def validate_loading(loading):
    if set(loading["missing_keys"]) != {"classifier.weight", "classifier.bias"}:
        raise RuntimeError("Unexpected missing pretrained weights.")
    if loading["error_msgs"]:
        raise RuntimeError("Pretrained model loading errors.")
    if any(not key.startswith("cls.") for key in loading["unexpected_keys"]):
        raise RuntimeError("Unexpected pretrained checkpoint keys.")


def run_one(alpha, seed, batch_dir, shared):
    import torch
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import BertConfig, BertTokenizerFast
    from src.baseline_model import BaselineBertForSequenceClassification
    from src.rdrop_model import RDropBertForSequenceClassification

    protocol = shared["protocol"]
    inputs = shared["inputs"]
    condition = f"rdrop_alpha{alpha:g}"
    output = batch_dir / condition / f"seed{seed}"
    output.mkdir(parents=True, exist_ok=False)

    report = {
        "step": "B-RDrop",
        "condition": condition,
        "alpha": alpha,
        "seed": seed,
        "purpose": "formal_rdrop_experiment",
        "overall_status": "RUNNING",
        "phase": "preparing",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    model = None

    try:
        config_path = ROOT / f"configs/baseline_seed{seed}.json"
        run_config = load_json(config_path)
        run_config.update({
            "purpose": "formal_rdrop_experiment",
            "condition": condition,
            "alpha": alpha,
            "forwards_per_training_microbatch": 2,
            "expected_microbatches": EXPECTED["microbatches"],
            "expected_training_example_visits": EXPECTED["examples"],
            "expected_encoder_forward_calls": EXPECTED["encoder_forwards"],
            "expected_encoder_example_passes": EXPECTED["encoder_passes"],
        })

        model_dir = ROOT / inputs["model"]["local_path"]
        rows = {
            split: read_rows(ROOT / f"data/mrpc/{split}.jsonl")
            for split in ("train", "validation")
        }

        if len(rows["train"]) != 3668 or len(rows["validation"]) != 408:
            raise RuntimeError("MRPC row count mismatch.")

        tokenizer = BertTokenizerFast.from_pretrained(
            str(model_dir), local_files_only=True
        )
        options = {
            k: v for k, v in inputs["tokenization"].items()
            if k not in ("class", "do_lower_case")
        }
        keys = ("input_ids", "attention_mask", "token_type_ids")

        tensors, native_ids, encodings = {}, {}, {}
        for split, data in rows.items():
            enc = tokenizer(
                [r["sentence1"] for r in data],
                [r["sentence2"] for r in data],
                **options,
            )
            identity = hashlib.sha256(
                canonical({k: enc[k] for k in keys}).encode("utf-8")
            ).hexdigest()

            if identity != EXPECTED_ENCODING[split]:
                raise RuntimeError("Tokenization changed: " + split)

            encodings[split] = identity
            tensors[split] = {
                k: torch.tensor(enc[k], dtype=torch.long) for k in keys
            }
            tensors[split]["labels"] = torch.tensor(
                [r["label"] for r in data], dtype=torch.long
            )
            native_ids[split] = [r["idx"] for r in data]

        # Independently reproduce the Baseline initialization for this seed.
        seed_everything(seed)
        baseline_config = BertConfig.from_pretrained(
            str(model_dir), local_files_only=True, num_labels=2
        )
        baseline_config.id2label = {
            0: "not_equivalent", 1: "equivalent"
        }
        baseline_config.label2id = {
            value: key for key, value in baseline_config.id2label.items()
        }

        baseline_model, baseline_loading = (
            BaselineBertForSequenceClassification.from_pretrained(
                str(model_dir),
                config=baseline_config,
                local_files_only=True,
                output_loading_info=True,
            )
        )
        validate_loading(baseline_loading)
        baseline_initial_hash = model_fingerprint(baseline_model)
        del baseline_model
        gc.collect()

        accepted_hash = shared["baseline_by_seed"][seed]["initial_state_sha256"]
        if baseline_initial_hash != accepted_hash:
            raise RuntimeError(
                f"Seed {seed} Baseline initialization differs from its accepted run."
            )

        # Reset the RNG so R-Drop starts from the exact same initialization.
        seed_everything(seed)
        model_config = BertConfig.from_pretrained(
            str(model_dir), local_files_only=True, num_labels=2
        )
        model_config.id2label = {
            0: "not_equivalent", 1: "equivalent"
        }
        model_config.label2id = {
            value: key for key, value in model_config.id2label.items()
        }
        model_config.rdrop_alpha = float(alpha)

        model, loading = RDropBertForSequenceClassification.from_pretrained(
            str(model_dir),
            config=model_config,
            local_files_only=True,
            output_loading_info=True,
        )
        validate_loading(loading)

        initial_hash = model_fingerprint(model)
        if initial_hash != baseline_initial_hash:
            raise RuntimeError("R-Drop and Baseline initial weights differ.")

        if seed == 13:
            expected_hash = run_config["validated_seed13_initial_state_sha256"]
            if initial_hash != expected_hash:
                raise RuntimeError("Seed-13 accepted initial hash mismatch.")

        manifest_hash, manifest_files = code_manifest()
        run_config_hash = sha_json(run_config)

        report["runtime"] = shared["runtime"]
        report["nvidia_before"] = nvidia_snapshot()
        report["gpu_identity"] = shared["gpu_identity"]
        report["resolved_config"] = {
            "protocol": protocol,
            "run": run_config,
            "inputs": inputs,
        }
        report["identities"] = {
            "protocol_sha256": sha256_file(
                ROOT / "configs/training_protocol_v1.json"
            ),
            "run_config_sha256": run_config_hash,
            "input_manifest_sha256": sha256_file(
                ROOT / "provenance/assets.json"
            ),
            "code_manifest_sha256": manifest_hash,
            "code_manifest_files": manifest_files,
            "training_core_sha256": sha256_file(
                ROOT / "src/training_core.py"
            ),
            "rdrop_model_sha256": sha256_file(
                ROOT / "src/rdrop_model.py"
            ),
            "experiment_runner_sha256": sha256_file(
                ROOT / "src/experiment_runner.py"
            ),
        }
        report["encoding_sha256"] = encodings
        report["pretrained_loading"] = loading
        report["baseline_initial_state_sha256"] = baseline_initial_hash
        report["rdrop_loss_contract"] = (
            "0.5*(CE1_mean+CE2_mean) + alpha*0.5*(KL12_batchmean+KL21_batchmean)"
        )
        report["rdrop_alpha"] = float(alpha)

        def check_reloaded_tokenizer(checkpoint):
            restored = BertTokenizerFast.from_pretrained(
                str(checkpoint), local_files_only=True
            )
            dev = rows["validation"]
            enc = restored(
                [r["sentence1"] for r in dev],
                [r["sentence2"] for r in dev],
                **options,
            )
            if any(
                enc[k] != tensors["validation"][k].tolist()
                for k in keys
            ):
                raise RuntimeError("Reloaded tokenizer changed dev encodings.")

        print(
            f"\n========== START {condition}, seed {seed} ==========",
            flush=True,
        )
        print("Initial SHA256:", initial_hash, flush=True)

        run_experiment(
            model,
            tokenizer,
            tensors,
            native_ids,
            output,
            protocol,
            run_config,
            torch.device("cuda:0"),
            report,
            reload_tokenizer_check=check_reloaded_tokenizer,
        )

        metrics = report["dev_metrics"]
        predictions = read_rows(output / "predictions_dev.jsonl")

        if len(predictions) != 408:
            raise RuntimeError("Incomplete dev predictions.")

        expected_order = [
            (row["idx"], row["label"]) for row in rows["validation"]
        ]
        actual_order = [
            (row["idx"], row["label"]) for row in predictions
        ]
        if actual_order != expected_order:
            raise RuntimeError("Dev prediction order/labels changed.")

        labels = [row["label"] for row in predictions]
        predicted = [row["prediction"] for row in predictions]

        sklearn_accuracy = accuracy_score(labels, predicted)
        sklearn_f1 = f1_score(
            labels, predicted, average="binary",
            pos_label=1, zero_division=0
        )

        if not math.isclose(
            metrics["accuracy"], sklearn_accuracy, abs_tol=1e-12
        ):
            raise RuntimeError("Accuracy differs from sklearn.")
        if not math.isclose(metrics["f1"], sklearn_f1, abs_tol=1e-12):
            raise RuntimeError("F1 differs from sklearn.")

        training = report["training"]
        train_counts = training["forward_counts"]
        expected_train_counts = {
            "encoder_forward_calls": EXPECTED["encoder_forwards"],
            "encoder_example_passes": EXPECTED["encoder_passes"],
            "padded_token_positions": EXPECTED["padded_positions"],
        }

        if training["optimizer_step_calls"] != EXPECTED["updates"]:
            raise RuntimeError("Optimizer update count mismatch.")
        if training["microbatches"] != EXPECTED["microbatches"]:
            raise RuntimeError("Microbatch count mismatch.")
        if training["examples_seen"] != EXPECTED["examples"]:
            raise RuntimeError("Training example visit count mismatch.")
        if training["distinct_native_ids_seen"] != EXPECTED["distinct_ids"]:
            raise RuntimeError("Distinct training IDs mismatch.")
        if train_counts != expected_train_counts:
            raise RuntimeError("R-Drop training forward accounting mismatch.")
        if training["amp_skipped_updates"] != 0:
            raise RuntimeError("AMP skipped an optimizer update.")
        if not training["all_gradient_norms_finite"]:
            raise RuntimeError("Nonfinite gradient norm.")
        if training["classifier_max_abs_change"] <= 0:
            raise RuntimeError("Classifier did not update.")
        if training["encoder_probe_max_abs_change"] <= 0:
            raise RuntimeError("Encoder did not update.")

        eval_counts = report["forward_counts"]["eval"]
        if eval_counts != {
            "encoder_forward_calls": 26,
            "encoder_example_passes": 408,
            "padded_token_positions": 52224,
        }:
            raise RuntimeError("Evaluation must use one forward per batch.")

        reload_check = report["checkpoint_reload"]
        for key in (
            "state_sha256_equal",
            "predictions_equal",
            "metrics_equal",
            "tokenizer_checked",
        ):
            if reload_check[key] is not True:
                raise RuntimeError("Checkpoint reload check failed: " + key)

        checkpoint_manifest = load_json(
            output / "checkpoint_manifest.json"
        )
        if checkpoint_manifest["state_sha256"] != report["final_state_sha256"]:
            raise RuntimeError("Checkpoint final-state hash mismatch.")

        for item in checkpoint_manifest["files"]:
            path = output / "checkpoint-final" / item["name"]
            if not path.is_file() or path.stat().st_size != item["size"]:
                raise RuntimeError("Checkpoint file missing/size mismatch.")

        report["metrics_verified_with_sklearn"] = True
        report["nvidia_after"] = nvidia_snapshot()
        report["phase"] = "complete"
        report["overall_status"] = "PASS"
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()

        result = {
            "status": "PASS",
            "condition": condition,
            "alpha": float(alpha),
            "seed": seed,
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": report["identities"]["protocol_sha256"],
            "run_config_sha256": run_config_hash,
            "code_manifest_sha256": manifest_hash,
            "input_manifest_sha256": report["identities"]["input_manifest_sha256"],
            "dev_examples": 408,
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "optimizer_updates": training["optimizer_step_calls"],
            "microbatches": training["microbatches"],
            "examples_seen": training["examples_seen"],
            "distinct_native_ids_seen": training["distinct_native_ids_seen"],
            "train_loop_wall_seconds": training["train_loop_wall_seconds"],
            "mean_wall_seconds_per_update": training["mean_wall_seconds_per_update"],
            "peak_allocated_bytes": training["peak_allocated_bytes"],
            "peak_reserved_bytes": training["peak_reserved_bytes"],
            "encoder_forward_calls": train_counts["encoder_forward_calls"],
            "encoder_example_passes": train_counts["encoder_example_passes"],
            "checkpoint_selection": "last_update",
            "initial_state_sha256": initial_hash,
            "final_state_sha256": report["final_state_sha256"],
            "batch_plan_sha256": report["batch_plan_sha256"],
            "run_path": output.resolve().relative_to(ROOT.resolve()).as_posix(),
        }

        write_json(output / "formal_report.json", report)
        write_json(output / "result.json", result)

        print(
            f"PASS {condition} seed={seed}: "
            f"Accuracy={metrics['accuracy']*100:.4f}% "
            f"F1={metrics['f1']*100:.4f}% "
            f"time={training['train_loop_wall_seconds']:.2f}s",
            flush=True,
        )
        return result

    except BaseException:
        report["overall_status"] = "FAIL"
        report["phase"] = "failed"
        report["traceback"] = traceback.format_exc()
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        print(report["traceback"], flush=True)
        write_json(output / "formal_report.json", report)
        write_json(output / "progress.json", {
            "status": "FAIL",
            "completed_updates": report.get("completed_updates", 0),
        })
        raise

    finally:
        if model is not None:
            try:
                model.to("cpu")
            except Exception:
                pass
            del model
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def summarize(results, baseline):
    if len(results) != 9:
        raise RuntimeError("Exactly nine successful R-Drop runs are required.")

    expected_pairs = {
        (f"rdrop_alpha{alpha:g}", seed)
        for alpha in ALPHAS for seed in SEEDS
    }
    actual_pairs = {(r["condition"], r["seed"]) for r in results}

    if actual_pairs != expected_pairs:
        raise RuntimeError("Missing or duplicate R-Drop conditions/seeds.")
    if any(r["status"] != "PASS" for r in results):
        raise RuntimeError("Only PASS runs may enter the summary.")

    baseline_by_seed = {
        int(r["seed"]): r for r in baseline["results"]
    }

    by_alpha = {}
    for alpha in ALPHAS:
        condition = f"rdrop_alpha{alpha:g}"
        group = sorted(
            [r for r in results if r["alpha"] == alpha],
            key=lambda r: r["seed"],
        )
        if [r["seed"] for r in group] != list(SEEDS):
            raise RuntimeError("Incomplete seeds for " + condition)

        accuracy = [r["accuracy"] for r in group]
        f1 = [r["f1"] for r in group]
        times = [r["train_loop_wall_seconds"] for r in group]
        allocated = [r["peak_allocated_bytes"] for r in group]

        delta_accuracy = [
            r["accuracy"] - baseline_by_seed[r["seed"]]["accuracy"]
            for r in group
        ]
        delta_f1 = [
            r["f1"] - baseline_by_seed[r["seed"]]["f1"]
            for r in group
        ]
        cost_ratios = [
            r["train_loop_wall_seconds"]
            / baseline_by_seed[r["seed"]]["train_loop_wall_seconds"]
            for r in group
        ]

        by_alpha[condition] = {
            "alpha": alpha,
            "n": 3,
            "seeds": list(SEEDS),
            "mean_accuracy": statistics.mean(accuracy),
            "sample_std_accuracy": statistics.stdev(accuracy),
            "mean_f1": statistics.mean(f1),
            "sample_std_f1": statistics.stdev(f1),
            "mean_train_loop_wall_seconds": statistics.mean(times),
            "sample_std_train_loop_wall_seconds": statistics.stdev(times),
            "mean_peak_allocated_bytes": statistics.mean(allocated),
            "paired_delta_accuracy_vs_baseline": delta_accuracy,
            "mean_paired_delta_accuracy": statistics.mean(delta_accuracy),
            "sample_std_paired_delta_accuracy": statistics.stdev(delta_accuracy),
            "paired_delta_f1_vs_baseline": delta_f1,
            "mean_paired_delta_f1": statistics.mean(delta_f1),
            "sample_std_paired_delta_f1": statistics.stdev(delta_f1),
            "mean_training_time_ratio_vs_baseline": statistics.mean(cost_ratios),
            "runs": group,
        }

    return {
        "overall_status": "PASS",
        "protocol_id": "mrpc_bert_cased_v1",
        "condition": "rdrop",
        "planned_alphas": list(ALPHAS),
        "planned_seeds": list(SEEDS),
        "completed_runs": 9,
        "std_ddof": 1,
        "baseline_reference": {
            "mean_accuracy": baseline["mean_accuracy"],
            "sample_std_accuracy": baseline["sample_std_accuracy"],
            "mean_f1": baseline["mean_f1"],
            "sample_std_f1": baseline["sample_std_f1"],
            "mean_train_loop_wall_seconds": baseline["mean_train_loop_wall_seconds"],
        },
        "by_alpha": by_alpha,
        "results": sorted(
            results, key=lambda r: (r["alpha"], r["seed"])
        ),
        "interpretation": (
            "All scores are MRPC validation/dev results, not test-set results. "
            "Alpha=0 is still two stochastic forwards with two CE terms and "
            "is not the single-forward Baseline. Standard deviations are "
            "sample standard deviations across three seeds (ddof=1). "
            "Encoder passes and sample visits describe measured work, not FLOPs."
        ),
    }


def write_summary(batch_dir, summary):
    write_json(batch_dir / "rdrop_results.json", summary)

    lines = [
        "# R-Drop MRPC 正式实验结果",
        "",
        "数据集：MRPC；验证集：408 条；三个随机种子：13、21、42。",
        "每组 2296 次优化器更新；使用最后一次更新的 checkpoint。",
        "",
        "## 各 Alpha 的三种子结果",
        "",
        "| Alpha | Accuracy (%) | F1 (%) | 训练时间 (s) | 相对 Baseline 时间倍数 |",
        "|---:|---:|---:|---:|---:|",
    ]

    for alpha in ALPHAS:
        item = summary["by_alpha"][f"rdrop_alpha{alpha:g}"]
        lines.append(
            f"| {alpha:g} | "
            f"{item['mean_accuracy']*100:.4f} ± {item['sample_std_accuracy']*100:.4f} | "
            f"{item['mean_f1']*100:.4f} ± {item['sample_std_f1']*100:.4f} | "
            f"{item['mean_train_loop_wall_seconds']:.2f} ± "
            f"{item['sample_std_train_loop_wall_seconds']:.2f} | "
            f"{item['mean_training_time_ratio_vs_baseline']:.3f} |"
        )

    lines += [
        "",
        "## 与 Baseline 的配对差值",
        "",
        "| Alpha | Accuracy 差值 (百分点) | F1 差值 (百分点) |",
        "|---:|---:|---:|",
    ]

    for alpha in ALPHAS:
        item = summary["by_alpha"][f"rdrop_alpha{alpha:g}"]
        lines.append(
            f"| {alpha:g} | "
            f"{item['mean_paired_delta_accuracy']*100:+.4f} ± "
            f"{item['sample_std_paired_delta_accuracy']*100:.4f} | "
            f"{item['mean_paired_delta_f1']*100:+.4f} ± "
            f"{item['sample_std_paired_delta_f1']*100:.4f} |"
        )

    lines += [
        "",
        "## 解释与限制",
        "",
        "- ± 表示三个种子的样本标准差（ddof=1），不是置信区间。",
        "- Alpha=0 仍然执行两次随机前向和两次 CE，不能当作 Baseline。",
        "- Baseline 使用单次前向；R-Drop 使用双次前向，计算成本不能忽略。",
        "- 结果来自 validation/dev，不是独立 test 集。",
        "- Encoder 前向次数和样本通过次数是工作量统计，不是 FLOPs。",
        "- 所有种子均保留，不根据最高分挑选实验。",
        "",
        "## 每次实验",
        "",
        "| Alpha | Seed | Accuracy (%) | F1 (%) | Encoder forwards | 训练时间 (s) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]

    for result in summary["results"]:
        lines.append(
            f"| {result['alpha']:g} | {result['seed']} | "
            f"{result['accuracy']*100:.4f} | {result['f1']*100:.4f} | "
            f"{result['encoder_forward_calls']} | "
            f"{result['train_loop_wall_seconds']:.2f} |"
        )

    path = batch_dir / "rdrop_results.md"
    partial = path.with_suffix(".md.tmp")
    partial.write_text("\n".join(lines) + "\n", encoding="utf-8")
    partial.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan-only", action="store_true",
        help="Print the nine-run plan without starting training."
    )
    args = parser.parse_args()

    if args.plan_only:
        print("R-Drop formal plan; no training will be performed.")
        for alpha in ALPHAS:
            for seed in SEEDS:
                print(f"  rdrop_alpha{alpha:g} / seed {seed}: 2296 updates")
        print("Total: 9 runs; 20664 optimizer updates.")
        print("Training forwards per run: 9166.")
        return 0

    output_root = ROOT / "reports/rdrop_formal"
    output_root.mkdir(parents=True, exist_ok=True)

    lock_path = output_root / ".gpu.lock"
    lock_stream = lock_path.open("a+")

    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Another formal GPU experiment holds the R-Drop lock.")
        lock_stream.close()
        return 2

    batch_name = "batch_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    batch_dir = output_root / batch_name
    batch_dir.mkdir(parents=True, exist_ok=False)

    status = {
        "overall_status": "RUNNING",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "planned_alphas": list(ALPHAS),
        "planned_seeds": list(SEEDS),
        "completed_runs": [],
        "failed_run": None,
        "output_directory": str(batch_dir),
    }
    write_json(batch_dir / "batch_status.json", status)

    try:
        shared = verify_shared_resources()
        results = []

        for alpha in ALPHAS:
            for seed in SEEDS:
                result = run_one(alpha, seed, batch_dir, shared)
                results.append(result)

                status["completed_runs"].append({
                    "condition": result["condition"],
                    "seed": seed,
                    "accuracy": result["accuracy"],
                    "f1": result["f1"],
                    "run_path": result["run_path"],
                })
                write_json(batch_dir / "batch_status.json", status)

        summary = summarize(results, shared["baseline"])
        write_summary(batch_dir, summary)

        status["overall_status"] = "PASS"
        status["summary_json"] = str(batch_dir / "rdrop_results.json")
        status["summary_markdown"] = str(batch_dir / "rdrop_results.md")
        status["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(batch_dir / "batch_status.json", status)

        print("\n========== ALL NINE R-DROP RUNS PASS ==========")
        print("Output:", batch_dir)
        print("Summary:", batch_dir / "rdrop_results.md")
        return 0

    except BaseException:
        status["overall_status"] = "FAIL"
        status["traceback"] = traceback.format_exc()
        status["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(batch_dir / "batch_status.json", status)
        print(status["traceback"], flush=True)
        print("Failed artifacts retained at:", batch_dir, flush=True)
        return 1

    finally:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
        lock_stream.close()


if __name__ == "__main__":
    raise SystemExit(main())
