#!/usr/bin/env python3
"""Offline step-2 setup. Run with the validated server base Python 3.10.

Creates a project .venv with access to the already validated base PyTorch.
All pip writes target .venv; no network is used. Logs are retained on failure.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import tempfile
import traceback
import zipfile

from prepare_source import prepare_source
from verify_bundle import verify_bundle

ROOT = Path(__file__).resolve().parents[1]
PROBE = """import json,sys,torch,numpy
print(json.dumps({'python':sys.version.split()[0], 'torch':str(torch.__version__),
'torch_cuda':torch.version.cuda, 'torch_file':torch.__file__, 'numpy':numpy.__version__,
'cuda_available':torch.cuda.is_available()}))"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "reports/setup")
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args.output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_CN_")
    output = Path(tempfile.mkdtemp(prefix=stamp, dir=args.output_root)).resolve()
    result = {"step": 2, "overall_status": "RUNNING", "output_directory": str(output),
              "started_at_utc": datetime.now(timezone.utc).isoformat(), "commands": [],
              "scope": "Author-source and dependency setup; no MRPC data or pretrained model downloaded."}
    log = (output / "install.log").open("w", encoding="utf-8")

    def say(text):
        print(text, flush=True)
        log.write(text + "\n")
        log.flush()

    child_env = os.environ.copy()
    child_env.pop("PYTHONPATH", None)
    child_env.update({"PYTHONNOUSERSITE": "1", "PYTHONIOENCODING": "utf-8",
                      "PIP_NO_INDEX": "1", "PIP_NO_INPUT": "1", "PIP_CONFIG_FILE": os.devnull,
                      "PIP_DISABLE_PIP_VERSION_CHECK": "1", "HF_HUB_OFFLINE": "1",
                      "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                      "USE_TORCH": "1", "USE_TF": "0", "USE_FLAX": "0",
                      "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "4"})
    child_env.update({"HF_HOME": str(ROOT / "cache/huggingface"),
                      "HF_DATASETS_CACHE": str(ROOT / "cache/datasets"),
                      "TRANSFORMERS_CACHE": str(ROOT / "cache/transformers")})

    def run(command, timeout=600):
        command = [str(x) for x in command]
        say("\n$ " + shlex.join(command))
        entry = {"argv": command}
        result["commands"].append(entry)
        # Bounded subprocess; its output is replayed into both console and install.log.
        completed = subprocess.run(command, cwd=ROOT, env=child_env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                   errors="replace", timeout=timeout, check=False)
        if completed.stdout.strip():
            say(completed.stdout.rstrip())
        entry["returncode"] = completed.returncode
        if completed.returncode:
            raise RuntimeError(f"Command failed with exit code {completed.returncode}; see install.log.")
        return completed.stdout

    def probe(python):
        output_text = run([python, "-c", PROBE], timeout=60)
        return json.loads(output_text.strip().splitlines()[-1])

    say("Step 2: offline author-source and dependency setup")
    say("Report directory: " + str(output))
    try:
        if platform.system() != "Linux" or platform.machine() != "x86_64" or sys.version_info[:2] != (3, 10):
            raise RuntimeError("Run on the rented Linux x86_64 server with Python 3.10, not on the Windows workstation.")
        venv_dir = ROOT / ".venv"
        if Path(sys.prefix).resolve() == venv_dir.resolve():
            raise RuntimeError("Run setup with base Python: /root/miniconda3/bin/python scripts/setup_step2.py")
        result["bundle_verification"] = verify_bundle()
        say("PASS: source archive, 44 wheels and requirement locks verified.")
        before = probe(sys.executable)
        result["base_before"] = before
        if before["torch"] != "2.1.2+cu118" or before["torch_cuda"] != "11.8" or before["numpy"] != "1.26.4":
            raise RuntimeError("Base PyTorch/CUDA/NumPy differs from step-1 report. Preserve it and send this report.")
        if not before["cuda_available"]:
            raise RuntimeError("Base environment CUDA is unavailable.")
        result["source_preparation"] = prepare_source()
        (output / "source_preparation.json").write_text(
            json.dumps(result["source_preparation"], indent=2) + "\n", encoding="utf-8")
        venv_python = venv_dir / "bin/python"
        marker = venv_dir / ".rdrop-step2.json"
        if venv_dir.exists():
            if not (venv_python.exists() and marker.exists()):
                raise RuntimeError("Existing .venv is incomplete or not created by this script; preserving it for review.")
            previous = json.loads(marker.read_text(encoding="utf-8"))
            if previous["base_python"] != str(Path(sys.executable).resolve()):
                raise RuntimeError("Existing .venv uses a different base interpreter; preserving it for review.")
        else:
            say("Creating project .venv; reusing base PyTorch through system-site-packages.")
            run([sys.executable, "-m", "venv", "--system-site-packages", venv_dir])
            marker.write_text(json.dumps({"base_python": str(Path(sys.executable).resolve()),
                                          "base_torch": before["torch"], "created_by": "setup_step2.py"}, indent=2),
                              encoding="utf-8")
        result["project_python"] = str(venv_python)
        install_flags = ["--no-index", "--find-links", str(ROOT / "wheelhouse"), "--no-deps",
                         "--require-hashes", "--disable-pip-version-check"]
        say("Installing pinned build tools from local wheelhouse...")
        run([venv_python, "-m", "pip", "install", *install_flags, "-r", ROOT / "requirements-bootstrap.lock"])
        say("Installing pinned project dependencies from local wheelhouse...")
        run([venv_python, "-m", "pip", "install", *install_flags, "-r", ROOT / "requirements-step2.lock"])
        say("Installing the author's local Transformers fork in editable mode...")
        run([venv_python, "-m", "pip", "install", "--no-index", "--no-deps", "--no-build-isolation",
             "--disable-pip-version-check", "--editable", ROOT / "vendor/R-Drop/huggingface_transformer_src"])
        project_probe = probe(venv_python)
        result["project_environment"] = project_probe
        if project_probe["torch"] != before["torch"] or Path(project_probe["torch_file"]).resolve() != Path(before["torch_file"]).resolve():
            raise RuntimeError("Project environment is not reusing the validated base PyTorch.")
        run([venv_python, ROOT / "scripts/check_dependencies.py", "--output-dir", output])
        after = probe(sys.executable)
        result["base_after"] = after
        if after != before:
            raise RuntimeError("Base environment changed during setup; send the report for review.")
        result["overall_status"] = "PASS"
    except Exception:
        result["overall_status"] = "FAIL"
        result["traceback"] = traceback.format_exc()
        say(result["traceback"])
    result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    (output / "install_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    say("\nOverall: " + result["overall_status"])
    say("Report directory: " + str(output))
    feedback = output / "feedback_step2.zip"
    say("Return feedback_step2.zip to the local server/ folder for review.")
    log.close()
    with zipfile.ZipFile(feedback, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.iterdir()):
            if path.is_file() and path.suffix in (".json", ".md", ".log", ".txt"):
                archive.write(path, path.name)
    print("Feedback ZIP: " + str(feedback), flush=True)
    return 0 if result["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
