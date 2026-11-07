"""Audit fixed-seed artifacts and summarize scores/costs; standard library only."""
from __future__ import annotations
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import random
import statistics

from asset_utils import ROOT, canonical, read_rows, sha256_file

EVIDENCE_FILES = ("formal_report.json", "result.json", "run.log", "training_updates.jsonl",
                  "data_ids.json", "predictions_dev.jsonl", "checkpoint_manifest.json", "progress.json")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    partial = path.with_name(path.name + ".tmp")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    partial.replace(path)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def gpu_identity(snapshots):
    item = snapshots[0]
    require(item.get("returncode") == 0, "Cannot verify nvidia-smi GPU identity.")
    rows = list(csv.reader(io.StringIO(item["stdout"]), skipinitialspace=True))
    require(len(rows) == 2 and len(rows[1]) == 8, "Expected exactly one visible GPU.")
    values = [v.strip() for v in rows[1]]
    return dict(zip(("index", "uuid", "name", "driver", "memory_total", "power_limit"), values[:5] + [values[7]]))


def verify_bundle(root, name):
    manifest = load_json(root / "provenance" / name)
    for item in manifest["files"]:
        path = (root / item["path"]).resolve()
        require(path.is_relative_to(root.resolve()) and path.is_file(), "Missing/invalid bundle path: " + item["path"])
        require(path.stat().st_size == item["size"] and sha256_file(path) == item["sha256"], "Bundle file changed: " + item["path"])
    return sha256_file(root / "provenance" / name)


def verify_anchor(root=ROOT):
    anchor = load_json(root / "provenance/step6_seed13_anchor.json")
    path = root / anchor["run_path"]
    for item in anchor["files"]:
        require(sha256_file(path / item["name"]) == item["sha256"], "Accepted seed-13 evidence changed: " + item["name"])
    return path


def expected_plan(native_ids, seed):
    """Independent replay of the fixed protocol for auditing stored sample order."""
    plan = []
    epoch = 0
    while len(plan) < 2296:
        order = list(native_ids)
        random.Random(seed + epoch).shuffle(order)
        for start in range(0, len(order), 16):
            group = order[start:start + 16]
            plan.append({"epoch": epoch, "microbatches": [group[i:i + 8] for i in range(0, len(group), 8)]})
            if len(plan) == 2296:
                break
        epoch += 1
    return plan


def binary_scores(rows):
    tp = sum(r["label"] == 1 and r["prediction"] == 1 for r in rows)
    tn = sum(r["label"] == 0 and r["prediction"] == 0 for r in rows)
    fp = sum(r["label"] == 0 and r["prediction"] == 1 for r in rows)
    fn = sum(r["label"] == 1 and r["prediction"] == 0 for r in rows)
    denominator = 2 * tp + fp + fn
    return {"accuracy": (tp + tn) / len(rows), "f1": 2 * tp / denominator if denominator else 0.0,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def verify_run(output, seed, root=ROOT, check_checkpoint=False):
    """No model execution. Verify source identity, full dev and training accounting."""
    output = Path(output)
    result, report = load_json(output / "result.json"), load_json(output / "formal_report.json")
    require(seed in (13, 21, 42), "Unexpected seed.")
    require(result["status"] == report["overall_status"] == "PASS", "Run did not pass.")
    require(result["seed"] == report["seed"] == seed and result["condition"] == report["condition"] == "baseline", "Wrong seed/condition.")
    protocol = load_json(root / "configs/training_protocol_v1.json")
    config_path = root / f"configs/baseline_seed{seed}.json"
    require(report["resolved_config"]["protocol"] == protocol and report["resolved_config"]["run"] == load_json(config_path), "Resolved settings changed.")
    require(report["resolved_config"]["inputs"] == load_json(root / "configs/input_protocol.json"), "Input settings changed.")
    manifests = {"protocol_sha256": root / "configs/training_protocol_v1.json",
                 "run_config_sha256": config_path, "input_manifest_sha256": root / "provenance/assets.json",
                 "training_core_sha256": root / "src/training_core.py",
                 "code_manifest_sha256": root / f"provenance/step{5 if seed == 13 else 6}_bundle.json"}
    for key, path in manifests.items():
        require(report["identities"][key] == sha256_file(path), "Identity mismatch: " + key)
        if key in result:
            require(result[key] == report["identities"][key], "Result identity mismatch: " + key)
    require(result["protocol_id"] == protocol["protocol_id"], "Protocol ID mismatch.")
    anchor = load_json(root / "provenance/step6_seed13_anchor.json")
    reference = load_json(root / anchor["run_path"] / "formal_report.json")
    for key in ("python", "torch", "cuda_runtime", "gpu", "cudnn", "transformers", "bert_source_sha256", "package_versions"):
        require(report["runtime"][key] == reference["runtime"][key], "Runtime changed: " + key)
    require(gpu_identity(report["nvidia_before"]) == anchor["gpu_identity"], "GPU identity/driver/power changed.")
    require(report["encoding_sha256"] == reference["encoding_sha256"], "Tokenization changed.")
    require(report["effective_precision"] == "fp16_amp", "Not a formal CUDA AMP run.")
    require(report["completed_updates"] == result["optimizer_updates"] == 2296 and result["dev_examples"] == 408, "Incomplete formal run.")
    require(load_json(output / "progress.json")["status"] == "PASS", "Final progress not PASS.")
    train_rows = read_rows(root / "data/mrpc/train.jsonl")
    dev_rows = read_rows(root / "data/mrpc/validation.jsonl")
    ids = {"train": [r["idx"] for r in train_rows], "validation": [r["idx"] for r in dev_rows]}
    require(load_json(output / "data_ids.json") == ids and len(train_rows) == 3668 and len(dev_rows) == 408, "Data ID mismatch.")
    predictions = read_rows(output / "predictions_dev.jsonl")
    require([(r["idx"], r["label"]) for r in predictions] == [(r["idx"], r["label"]) for r in dev_rows], "Dev labels/order mismatch.")
    for row in predictions:
        logits = row["logits"]
        require(len(logits) == 2 and all(math.isfinite(x) for x in logits), "Invalid logits.")
        require(row["prediction"] == int(logits[1] > logits[0]), "Prediction not argmax.")
    scores = binary_scores(predictions)
    for key, value in scores.items():
        require(math.isclose(report["dev_metrics"][key], value, rel_tol=0, abs_tol=1e-12), "Dev metric mismatch: " + key)
        if key in ("accuracy", "f1"):
            require(math.isclose(result[key], value, rel_tol=0, abs_tol=1e-12), "Result score mismatch: " + key)
    plan = expected_plan(ids["train"], seed)
    digest = hashlib.sha256(canonical(plan).encode("utf-8")).hexdigest()
    require(report["batch_plan_sha256"] == digest, "Batch-plan hash mismatch.")
    events = read_rows(output / "training_updates.jsonl")
    require(len(events) == 2296, "Incomplete training log.")
    for step, (event, group) in enumerate(zip(events, plan), 1):
        flat = [idx for micro in group["microbatches"] for idx in micro]
        require((event["update"], event["epoch"], event["native_ids"]) == (step, group["epoch"], flat), "Training sample order mismatch.")
        require(event["examples_seen"] == len(flat) and event["microbatches"] == event["encoder_forward_calls"] == len(group["microbatches"]), "Per-step work mismatch.")
        require(event["optimizer_step_calls"] == 1 and event["amp_scale_after"] >= event["amp_scale_before"], "AMP skipped an update.")
        require(all(math.isfinite(event[k]) for k in ("loss_mean", "grad_norm_before_clip", "update_seconds")) and event["update_seconds"] > 0, "Nonfinite/invalid event.")
    training = report["training"]
    require(training["optimizer_step_calls"] == 2296 and training["microbatches"] == 4583 and training["examples_seen"] == 36628, "Training totals mismatch.")
    require(training["amp_skipped_updates"] == 0 and training["all_gradient_norms_finite"] is True, "Invalid optimization.")
    require(training["distinct_native_ids_seen"] == 3668 and training["classifier_max_abs_change"] > 0 and training["encoder_probe_max_abs_change"] > 0, "Incomplete updates/coverage.")
    require(training["forward_counts"] == {"encoder_forward_calls": 4583, "encoder_example_passes": 36628, "padded_token_positions": 4688384}, "Forward cost mismatch.")
    require(report["forward_counts"]["eval"] == {"encoder_forward_calls": 26, "encoder_example_passes": 408, "padded_token_positions": 52224}, "Evaluation forward count mismatch.")
    for key in ("train_loop_wall_seconds", "mean_wall_seconds_per_update", "peak_allocated_bytes", "peak_reserved_bytes", "examples_seen"):
        require(result[key] == training[key] and math.isfinite(result[key]) and result[key] > 0, "Cost mismatch: " + key)
    for key in ("encoder_forward_calls", "encoder_example_passes"):
        require(result[key] == training["forward_counts"][key], "Forward result mismatch.")
    body_seconds = sum(e["update_seconds"] for e in events)
    require(math.isclose(training["sum_update_body_seconds"], body_seconds, rel_tol=1e-10), "Update-time sum mismatch.")
    require(training["train_loop_wall_seconds"] >= body_seconds and math.isclose(result["mean_wall_seconds_per_update"], result["train_loop_wall_seconds"] / 2296, rel_tol=1e-12), "Wrong wall-clock scope.")
    require(result["checkpoint_selection"] == report["dev_metrics"]["checkpoint_selection"] == "last_update", "Wrong checkpoint selection.")
    checkpoint = load_json(output / "checkpoint_manifest.json")
    require(checkpoint["state_sha256"] == report["final_state_sha256"] and not checkpoint["optimizer_saved"], "Checkpoint state mismatch.")
    for key in ("state_sha256_equal", "predictions_equal", "metrics_equal", "tokenizer_checked"):
        require(report["checkpoint_reload"][key] is True, "Reload check failed: " + key)
    if check_checkpoint:
        for item in checkpoint["files"]:
            path = output / "checkpoint-final" / item["name"]
            require(path.is_file() and path.stat().st_size == item["size"] and sha256_file(path) == item["sha256"], "Checkpoint missing/changed: " + item["name"])
    return {**result, "run_path": output.resolve().relative_to(root.resolve()).as_posix() if output.resolve().is_relative_to(root.resolve()) else str(output),
            "initial_state_sha256": report["initial_state_sha256"], "final_state_sha256": report["final_state_sha256"],
            "batch_plan_sha256": digest, "training_core_sha256": report["identities"]["training_core_sha256"],
            "evidence_sha256": {name: sha256_file(output / name) for name in EVIDENCE_FILES},
            "artifact_checks": "PASS", "checkpoint_files_verified_now": check_checkpoint}


def summarize(results):
    require(len(results) == 3 and sorted(r["seed"] for r in results) == [13, 21, 42], "Need exactly seeds 13, 21 and 42; no missing or duplicate seeds.")
    results = sorted(results, key=lambda r: r["seed"])
    require(all(r["status"] == "PASS" and r["artifact_checks"] == "PASS" for r in results), "Only verified PASS runs can be summarized.")
    for key in ("protocol_id", "protocol_sha256", "input_manifest_sha256", "training_core_sha256", "condition", "optimizer_updates", "encoder_forward_calls", "encoder_example_passes", "examples_seen", "checkpoint_selection"):
        require(len({r[key] for r in results}) == 1, "Runs differ in " + key)
    require(len({r["initial_state_sha256"] for r in results}) == 3, "Different seeds unexpectedly share initial states.")
    summary = {"protocol_id": results[0]["protocol_id"], "protocol_sha256": results[0]["protocol_sha256"],
               "condition": "baseline", "planned_seeds": [13, 21, 42], "accepted_seeds": [13, 21, 42],
               "pending_seeds": [], "complete": True, "verification": "Automatic artifact checks; local review of returned feedback pending.",
               "n": 3, "std_ddof": 1, "results": results,
               "score_units": "JSON: fraction [0,1]; Markdown: percent, standard deviation in percentage points.",
               "cost_scope": "CUDA-synchronized full training loop including first-step and logging/check overhead; excludes preprocessing, evaluation, checkpoint save/reload.",
               "interpretation": "Fixed 3-seed MRPC dev baseline; no test-set claim or R-Drop improvement claim. Three seeds give a limited variability estimate; sample visits/forward counts are not FLOPs."}
    for metric in ("accuracy", "f1", "train_loop_wall_seconds", "mean_wall_seconds_per_update", "peak_allocated_bytes", "peak_reserved_bytes"):
        values = [r[metric] for r in results]
        require(all(math.isfinite(v) for v in values), "Nonfinite summary value.")
        summary["mean_" + metric] = statistics.mean(values)
        summary["sample_std_" + metric] = statistics.stdev(values)
    return summary


def write_summary(directory, summary):
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "baseline_results.json", summary)
    lines = ["# Baseline 三种子正式结果", "", "完整 MRPC dev：408 条；固定种子 13、21、42；每次 2296 次优化器更新；均使用末次 checkpoint。", "",
             "| Seed | Accuracy (%) | F1 (%) | 训练循环 (s) | 每步 (s) | 峰值 allocated (GiB) | 峰值 reserved (GiB) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in summary["results"]:
        lines.append(f"| {r['seed']} | {r['accuracy']*100:.4f} | {r['f1']*100:.4f} | {r['train_loop_wall_seconds']:.3f} | {r['mean_wall_seconds_per_update']:.6f} | {r['peak_allocated_bytes']/2**30:.4f} | {r['peak_reserved_bytes']/2**30:.4f} |")
    lines += ["", f"Accuracy：**{summary['mean_accuracy']*100:.4f}% ± {summary['sample_std_accuracy']*100:.4f} 个百分点**；F1：**{summary['mean_f1']*100:.4f}% ± {summary['sample_std_f1']*100:.4f} 个百分点**。",
              f"训练循环：**{summary['mean_train_loop_wall_seconds']:.3f} ± {summary['sample_std_train_loop_wall_seconds']:.3f} 秒**。",
              "", "± 为跨种子的样本标准差（n=3，ddof=1），不是标准误或置信区间。保留全部预定种子，不挑选最高分。",
              "", "每个种子：4583 次训练编码器前向、36628 次训练样本访问/编码器样本通过、4688384 个填充 token 位置。该计数描述计算工作量，不是精确 FLOPs。",
              "", "计时包含首次更新、数据搬运、检查及日志；不含预处理、评估、保存和重新加载。设备与协议核验通过，但训练前后的 GPU 快照不代表全程独占证明。",
              "", "本表为服务器自动核验结果，回传后仍需本地验收。只代表 Baseline 的 dev 结果；尚不能得出 R-Drop 增益或免费提升结论。", "",
              "原始结果路径、初始/最终模型及 batch-plan 指纹、各证据文件哈希见同目录 baseline_results.json。", ""]
    partial = directory / "baseline_results.md.tmp"
    partial.write_text("\n".join(lines), encoding="utf-8")
    partial.replace(directory / "baseline_results.md")
