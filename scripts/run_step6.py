"""Run approved baseline seeds 21 then 42, reusing accepted seed 13, entirely offline."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import shutil
import sys
import traceback
import zipfile

from asset_utils import ROOT, verify_assets
from baseline_summary import (EVIDENCE_FILES, load_json, require, summarize, verify_anchor,
                              verify_bundle, verify_run, write_json, write_summary)
from run_step5 import stream_child


def child_environment(seed):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update({"PYTHONNOUSERSITE": "1", "PYTHONHASHSEED": str(seed), "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
                "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1", "USE_TORCH": "1", "USE_TF": "0", "USE_FLAX": "0",
                "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "4", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "HF_HOME": str(ROOT / "cache/huggingface"), "TRANSFORMERS_CACHE": str(ROOT / "cache/transformers")})
    return env


def run_remaining(batch, launch, validate, say):
    """A failed/partial run is retained and never silently restarted or overwritten."""
    results = []
    for seed in (21, 42):
        output = batch / f"seed{seed}"
        if output.exists():
            require((output / "result.json").is_file(), f"Seed {seed} has partial artifacts. Stop and return feedback; do not delete or rerun to choose a better score.")
            say(f"Checking existing seed {seed}; it will be reused only if all checks pass.")
        else:
            output.mkdir()
            say(f"Starting full baseline seed {seed}, 2296 optimizer updates.")
            launch(seed, output)
        results.append(validate(output, seed))
        say(f"Seed {seed} verified PASS.")
    return results


def package_feedback(batch, seed13=None):
    target = batch / "feedback_step6.zip"
    partial = batch / "feedback_step6.zip.tmp"
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("step6_result.json", "launcher_failure.json", "run.log", "summary/baseline_results.json", "summary/baseline_results.md"):
            if (batch / name).is_file():
                archive.write(batch / name, arcname=name)
        for seed in (13, 21, 42):
            source = seed13 if seed == 13 else batch / f"seed{seed}"
            if source is None:
                continue
            for name in EVIDENCE_FILES + (("step5_result.json",) if seed == 13 else ()):
                if (source / name).is_file():
                    archive.write(source / name, arcname=f"seed{seed}/{name}")
    partial.replace(target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/baseline_batch/step6_baseline_seeds_21_42")
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    batch = args.output_dir.resolve()
    batch.mkdir(parents=True, exist_ok=True)
    lock = None
    try:
        require(platform.system() == "Linux" and platform.machine() == "x86_64" and sys.version_info[:2] == (3, 10), "Run on the validated Linux x86_64 server with Python 3.10.")
        require(Path(sys.prefix).resolve() == (ROOT / ".venv").resolve(), "Use .venv/bin/python scripts/run_step6.py")
        import fcntl
        (ROOT / "runs").mkdir(exist_ok=True)
        lock = (ROOT / "runs/gpu_training.lock").open("a+")
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            print("Another project job holds gpu_training.lock. No training started; existing reports untouched.", flush=True)
            return 2
    except BaseException:
        if lock:
            lock.close()
        failure = {"overall_status": "FAIL", "traceback": traceback.format_exc()}
        write_json(batch / "launcher_failure.json", failure)
        print(failure["traceback"], flush=True)
        print("Feedback ZIP: " + str(package_feedback(batch)), flush=True)
        return 1
    seed13 = None
    report = {"step": 6, "overall_status": "RUNNING", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "planned_seeds": [13, 21, 42], "training_order": [21, 42], "output_directory": str(batch)}
    try:
        with (batch / "run.log").open("a", encoding="utf-8") as log:
            def say(message):
                print(message, flush=True)
                log.write(message + "\n")
                log.flush()
            say("Step 6 started at " + report["started_at_utc"])
            say("Batch directory: " + str(batch))
            write_json(batch / "step6_result.json", report)
            try:
                require(shutil.disk_usage(batch).free >= 3 * 1024**3, "Keep at least 3 GiB free for two final checkpoints and logs.")
                report["code_manifest_sha256"] = verify_bundle(ROOT, "step6_bundle.json")
                report["historical_code_manifest_sha256"] = verify_bundle(ROOT, "step5_bundle.json")
                report["asset_verification"] = verify_assets()
                seed13 = verify_anchor()
                first = verify_run(seed13, 13, check_checkpoint=True)
                report["reused_seed13"] = first["run_path"]
                say("Accepted seed 13 verified; no seed-13 training will run.")

                def launch(seed, output):
                    command = [sys.executable, str(ROOT / "scripts/baseline_multiseed_runtime.py"),
                               "--seed", str(seed), "--output-dir", str(output)]
                    with (output / "run.log").open("w", encoding="utf-8") as child_log:
                        def child_say(line):
                            child_log.write(line + "\n")
                            child_log.flush()
                            say(f"[seed {seed}] {line}")
                        stream_child(command, child_environment(seed), child_say)

                new_results = run_remaining(batch, launch, lambda path, seed: verify_run(path, seed, check_checkpoint=True), say)
                summary = summarize([first] + new_results)
                write_summary(batch / "summary", summary)
                write_summary(ROOT / "reports/summary", summary)
                report["overall_status"] = "PASS"
                report["completed_seeds"] = [13, 21, 42]
                report["summary"] = str(batch / "summary/baseline_results.json")
                say(f"Accuracy: {summary['mean_accuracy']*100:.4f}% +/- {summary['sample_std_accuracy']*100:.4f} percentage points (sample SD)")
                say(f"F1: {summary['mean_f1']*100:.4f}% +/- {summary['sample_std_f1']*100:.4f} percentage points (sample SD)")
            except BaseException:
                report["overall_status"] = "FAIL"
                report["traceback"] = traceback.format_exc()
                say(report["traceback"])
            report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            say("Overall: " + report["overall_status"])
        write_json(batch / "step6_result.json", report)
        feedback = package_feedback(batch, seed13)
        print("Feedback ZIP: " + str(feedback), flush=True)
        print("Model checkpoints remain in each seed directory on the server.", flush=True)
        return 0 if report["overall_status"] == "PASS" else 1
    finally:
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
