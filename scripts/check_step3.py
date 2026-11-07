"""Offline step-3 entry point. Run using .venv/bin/python on the validated server."""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import traceback
import zipfile

from asset_utils import ROOT, sha256_file, verify_assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "reports/assets")
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args.output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_CN_")
    output = Path(tempfile.mkdtemp(prefix=stamp, dir=args.output_root)).resolve()
    report = {"step": 3, "overall_status": "RUNNING", "output_directory": str(output),
              "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Offline data/model preparation check; no formal experiments.",
              "script_sha256": {name: sha256_file(ROOT / "scripts" / name)
                                for name in ("asset_utils.py", "check_step3.py", "check_assets_runtime.py")}}
    with (output / "check.log").open("w", encoding="utf-8") as log:
        def say(message):
            print(message, flush=True)
            log.write(message + "\n")
            log.flush()

        say("Step 3: verify and load MRPC + pretrained BERT offline")
        say("Report directory: " + str(output))
        try:
            if platform.system() != "Linux" or platform.machine() != "x86_64" or sys.version_info[:2] != (3, 10):
                raise RuntimeError("Run this check on the Linux x86_64 server with project Python 3.10.")
            if Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
                raise RuntimeError("Use the project interpreter: .venv/bin/python scripts/check_step3.py")
            report["asset_verification"] = verify_assets()
            say("[PASS] Asset SHA256, unchanged MRPC splits, labels and sample identities.")
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            env.update({"PYTHONNOUSERSITE": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
                        "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                        "HF_HUB_DISABLE_TELEMETRY": "1", "USE_TORCH": "1", "USE_TF": "0", "USE_FLAX": "0",
                        "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "4",
                        "HF_HOME": str(ROOT / "cache/huggingface"),
                        "HF_DATASETS_CACHE": str(ROOT / "cache/datasets"),
                        "TRANSFORMERS_CACHE": str(ROOT / "cache/transformers")})
            command = [sys.executable, str(ROOT / "scripts/check_assets_runtime.py"), "--output-dir", str(output)]
            report["command"] = command
            say("Running full-data tokenization and pretrained GPU checks; output is in check.log.")
            log.flush()
            # Bound the subprocess; all diagnostics survive failures and timeouts.
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                           timeout=600, check=True)
            runtime = json.loads((output / "runtime_report.json").read_text(encoding="utf-8"))
            if runtime["overall_status"] != "PASS":
                raise RuntimeError("Resource runtime checks did not pass.")
            report["runtime_check_count"] = len(runtime["checks"])
            report["overall_status"] = "PASS"
            for item in runtime["checks"]:
                print(f"[PASS] {item['name']}: {item['detail']}", flush=True)
        except Exception:
            report["overall_status"] = "FAIL"
            report["traceback"] = traceback.format_exc()
            say(report["traceback"])
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        say("Overall: " + report["overall_status"])
    (output / "asset_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# 第 3 步真实数据与模型检查", "", f"总体状态：**{report['overall_status']}**", "",
             "本检查核验离线资源、全部数据分词及预训练 BERT 单前向。没有反向传播、参数更新或正式 Accuracy/F1。", "",
             "详细信息见 asset_report.json、runtime_report.json（若已生成）和 check.log。", ""]
    (output / "asset_report.md").write_text("\n".join(lines), encoding="utf-8")
    feedback = output / "feedback_step3.zip"
    with zipfile.ZipFile(feedback, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("asset_report.json", "asset_report.md", "runtime_report.json", "check.log"):
            path = output / name
            if path.exists():
                archive.write(path, arcname=name)
    print("Report directory: " + str(output), flush=True)
    print("Feedback ZIP: " + str(feedback), flush=True)
    return 0 if report["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
