"""Validated server launcher for the approved full-data baseline seed 13."""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import traceback
import zipfile

from asset_utils import ROOT, sha256_file, verify_assets

FEEDBACK_FILES = ("step5_result.json", "formal_report.json", "result.json", "run.log", "training_updates.jsonl",
                  "data_ids.json", "predictions_dev.jsonl", "checkpoint_manifest.json", "progress.json")


def package_feedback(output):
    target = output / "feedback_step5_seed13.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in FEEDBACK_FILES:
            if (output / name).is_file():
                archive.write(output / name, arcname=name)
    return target


def stream_child(command, env, say, timeout=7200):
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace", bufsize=1)
    timed_out = threading.Event()
    def expire():
        if process.poll() is None:
            timed_out.set()
            process.kill()
    timer = threading.Timer(timeout, expire)
    timer.daemon = True
    timer.start()
    try:
        for line in process.stdout:
            say(line.rstrip("\r\n"))
        code = process.wait()
        if timed_out.is_set():
            raise TimeoutError(f"Training process exceeded {timeout} seconds; partial logs retained.")
        if code:
            raise RuntimeError(f"Training process exited with {code}; see run.log.")
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    finally:
        timer.cancel()
        process.stdout.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "reports/baseline")
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args.output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("seed13_%Y%m%d_%H%M%S_CN_")
    output = Path(tempfile.mkdtemp(prefix=stamp, dir=args.output_root)).resolve()
    report = {"step": 5, "condition": "baseline", "seed": 13, "overall_status": "RUNNING", "output_directory": str(output)}
    lock = None
    with (output / "run.log").open("w", encoding="utf-8") as log:
        def say(message):
            print(message, flush=True)
            log.write(message + "\n")
            log.flush()
        say("Step 5: complete MRPC baseline, seed 13, 2296 optimizer steps")
        say("Report directory: " + str(output))
        try:
            if platform.system() != "Linux" or platform.machine() != "x86_64" or sys.version_info[:2] != (3, 10):
                raise RuntimeError("Run on the validated Linux x86_64 server with Python 3.10.")
            if Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
                raise RuntimeError("Use .venv/bin/python scripts/run_step5.py")
            if shutil.disk_usage(output).free < 2 * 1024**3:
                raise RuntimeError("Keep at least 2 GiB free for the final checkpoint and logs.")
            import fcntl
            (ROOT / "runs").mkdir(exist_ok=True)
            lock = (ROOT / "runs/gpu_training.lock").open("a+")
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("Another project training job holds gpu_training.lock. Run jobs sequentially.") from None
            bundle_path = ROOT / "provenance/step5_bundle.json"
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            for item in bundle["files"]:
                path = (ROOT / item["path"]).resolve()
                if not path.is_relative_to(ROOT.resolve()) or sha256_file(path) != item["sha256"]:
                    raise RuntimeError("Step-5 file hash mismatch: " + item["path"])
            report["code_manifest_sha256"] = sha256_file(bundle_path)
            report["asset_verification"] = verify_assets()
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            env.update({"PYTHONNOUSERSITE": "1", "PYTHONHASHSEED": "13", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
                        "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                        "HF_HUB_DISABLE_TELEMETRY": "1", "USE_TORCH": "1", "USE_TF": "0", "USE_FLAX": "0",
                        "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "4", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                        "HF_HOME": str(ROOT / "cache/huggingface"), "TRANSFORMERS_CACHE": str(ROOT / "cache/transformers")})
            command = [sys.executable, str(ROOT / "scripts/baseline_formal_runtime.py"), "--output-dir", str(output)]
            report["command"] = command
            stream_child(command, env, say)
            child = json.loads((output / "formal_report.json").read_text(encoding="utf-8"))
            result = json.loads((output / "result.json").read_text(encoding="utf-8"))
            if child["overall_status"] != "PASS" or result["status"] != "PASS" or child["completed_updates"] != 2296 or result["dev_examples"] != 408:
                raise RuntimeError("Full-run completion checks failed.")
            report["overall_status"] = "PASS"
        except BaseException:
            report["overall_status"] = "FAIL"
            report["traceback"] = traceback.format_exc()
            say(report["traceback"])
        finally:
            if lock is not None:
                lock.close()
        say("Overall: " + report["overall_status"])
    (output / "step5_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    feedback = package_feedback(output)
    print("Feedback ZIP: " + str(feedback), flush=True)
    print("Final checkpoint stays in this run's checkpoint-final directory on the server.", flush=True)
    return 0 if report["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
