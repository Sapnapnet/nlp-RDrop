"""Run the approved offline baseline smoke check and package small feedback files."""
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
import traceback
import zipfile

from asset_utils import ROOT, sha256_file, verify_assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "reports/baseline_smoke")
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args.output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_CN_")
    output = Path(tempfile.mkdtemp(prefix=stamp, dir=args.output_root)).resolve()
    report = {"step": 4, "overall_status": "RUNNING", "output_directory": str(output), "scope": "Baseline smoke only"}
    with (output / "run.log").open("w", encoding="utf-8") as log:
        def say(message):
            print(message, flush=True)
            log.write(message + "\n")
            log.flush()
        say("Step 4: offline baseline smoke, 20 optimizer calls")
        say("Report directory: " + str(output))
        try:
            if platform.system() != "Linux" or platform.machine() != "x86_64" or sys.version_info[:2] != (3, 10):
                raise RuntimeError("Run on the rented Linux x86_64 server with Python 3.10.")
            if Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
                raise RuntimeError("Use .venv/bin/python scripts/check_step4.py")
            if shutil.disk_usage(output).free < 2 * 1024**3:
                raise RuntimeError("Keep at least 2 GiB free for checkpoint writing and logs.")
            bundle_path = ROOT / "provenance/step4_bundle.json"
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            for item in bundle["files"]:
                path = (ROOT / item["path"]).resolve()
                if not path.is_relative_to(ROOT.resolve()) or sha256_file(path) != item["sha256"]:
                    raise RuntimeError("Step-4 code/config SHA256 mismatch: " + item["path"])
            report["code_manifest_sha256"] = sha256_file(bundle_path)
            report["asset_verification"] = verify_assets()
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            env.update({"PYTHONNOUSERSITE": "1", "PYTHONHASHSEED": "13", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
                        "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                        "HF_HUB_DISABLE_TELEMETRY": "1", "USE_TORCH": "1", "USE_TF": "0", "USE_FLAX": "0",
                        "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "4", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                        "HF_HOME": str(ROOT / "cache/huggingface"), "TRANSFORMERS_CACHE": str(ROOT / "cache/transformers")})
            command = [sys.executable, str(ROOT / "scripts/baseline_smoke_runtime.py"), "--output-dir", str(output)]
            report["command"] = command
            say("Running training, evaluation and checkpoint reload. Detailed progress is in run.log.")
            log.flush()
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=900, check=True)
            child = json.loads((output / "baseline_smoke_report.json").read_text(encoding="utf-8"))
            if child["overall_status"] != "PASS":
                raise RuntimeError("Baseline smoke did not pass.")
            report["overall_status"] = "PASS"
            for check in child["checks"]:
                print(f"[PASS] {check['name']}: {check['detail']}", flush=True)
        except Exception:
            report["overall_status"] = "FAIL"
            report["traceback"] = traceback.format_exc()
            say(report["traceback"])
        say("Overall: " + report["overall_status"])
    (output / "step4_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    feedback = output / "feedback_step4.zip"
    with zipfile.ZipFile(feedback, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("step4_result.json", "baseline_smoke_report.json", "run.log", "training_updates.jsonl",
                     "subset_ids.json", "predictions_dev_smoke.jsonl", "checkpoint_manifest.json"):
            if (output / name).exists():
                archive.write(output / name, arcname=name)
    print("Report directory: " + str(output), flush=True)
    print("Feedback ZIP: " + str(feedback), flush=True)
    print("Checkpoint stays on the server; it is not included in the feedback ZIP.", flush=True)
    return 0 if report["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
