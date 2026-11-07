#!/usr/bin/env python3
"""Step 1: inventory and a tiny CUDA forward/backward/optimizer check.

No package installs, downloads, model loading, or training runs are performed.
The standard library is sufficient to save a failure report if torch is missing.
Exit codes: 0 = checks passed (possibly with warnings); 1 = a required check failed.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback


SCRIPT_VERSION = "1.0.0"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "gpu_name": "NVIDIA GeForce RTX 3090",
    "python": "3.10",
    "torch": "2.1.2",
    "torch_cuda_runtime": "11.8",
    "os": "Ubuntu 22.04",
}
PACKAGES = (
    "torch", "numpy", "transformers", "datasets", "tokenizers",
    "huggingface-hub", "accelerate", "evaluate", "scikit-learn",
    "scipy", "safetensors", "protobuf", "packaging",
)


def read_text(path):
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def read_positive_limit(path):
    text = read_text(path)
    try:
        value = int(text)
        return value if 0 < value < 2**60 else None
    except (TypeError, ValueError):
        return None


def run_command(args):
    """Read-only command with bounded waiting; never invoke a shell."""
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=20, check=False,
        )
        return {"command": args, "returncode": result.returncode,
                "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": args, "returncode": None, "error": str(exc)}


def collect_system():
    os_release = {}
    for line in (read_text("/etc/os-release") or "").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            os_release[key] = value.strip('"')

    memory = {}
    for line in (read_text("/proc/meminfo") or "").splitlines():
        key, _, value = line.partition(":")
        if key in ("MemTotal", "MemAvailable"):
            memory[key + "_bytes"] = int(value.split()[0]) * 1024
    limits = [read_positive_limit(p) for p in (
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    )]
    limits = [v for v in limits if v is not None]
    memory["cgroup_limit_bytes"] = min(limits) if limits else None
    capacities = [v for v in (memory.get("MemTotal_bytes"),
                              memory["cgroup_limit_bytes"]) if v is not None]
    memory["effective_capacity_estimate_bytes"] = min(capacities) if capacities else None
    memory["note"] = "Host RAM and container limits are separate; limits may be unavailable."

    cpu_model = None
    for line in (read_text("/proc/cpuinfo") or "").splitlines():
        if line.startswith("model name"):
            cpu_model = line.split(":", 1)[1].strip()
            break
    try:
        affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = None
    quota_cores = None
    cpu_max = (read_text("/sys/fs/cgroup/cpu.max") or "").split()
    if len(cpu_max) == 2 and cpu_max[0] != "max":
        if int(cpu_max[1]) > 0:
            quota_cores = int(cpu_max[0]) / int(cpu_max[1])
    if quota_cores is None:
        quota = read_positive_limit("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        period = read_positive_limit("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        if quota and period:
            quota_cores = quota / period
    return {
        "platform": platform.platform(), "os_release": os_release,
        "architecture": platform.machine(), "python": sys.version,
        "python_executable": sys.executable,
        "cpu": {"model": cpu_model, "os_logical_cpus": os.cpu_count(),
                "affinity_cpus": affinity, "cgroup_quota_cores": quota_cores},
        "memory": memory,
        "project_root": str(PROJECT_ROOT),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>"),
    }


class Checker:
    def __init__(self, output_dir, device):
        self.output_dir = output_dir
        self.log = (output_dir / "console.log").open("w", encoding="utf-8")
        now = datetime.now(timezone.utc)
        self.report = {
            "schema_version": 1, "script_version": SCRIPT_VERSION,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "started_at_utc": now.isoformat(),
            "started_at_china_time": now.astimezone(timezone(timedelta(hours=8))).isoformat(),
            "expected_environment": EXPECTED,
            "selected_cuda_logical_device": device,
            "output_directory": str(output_dir), "checks": [],
            "scope": "Environment check only; not MRPC training or a performance benchmark.",
        }

    def say(self, message):
        print(message, flush=True)
        self.log.write(message + "\n")
        self.log.flush()

    def add(self, name, status, detail):
        self.report["checks"].append({"name": name, "status": status, "detail": detail})
        self.say(f"[{status}] {name}: {detail}")

    def inventory(self):
        self.say("1/3 Collecting system, package, storage and driver information...")
        self.report["system"] = collect_system()
        py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        self.add("python_version", "PASS" if py_version == EXPECTED["python"] else "WARN",
                 f"actual={py_version}, expected={EXPECTED['python']}")
        os_name = self.report["system"]["os_release"].get("PRETTY_NAME", platform.system())
        os_ok = platform.system() == "Linux" and "Ubuntu 22.04" in os_name
        self.add("operating_system", "PASS" if os_ok else "WARN", os_name)

        packages = {}
        for name in PACKAGES:
            try:
                packages[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                packages[name] = None
        self.report["packages"] = packages
        missing = [name for name, version in packages.items() if version is None and name != "torch"]
        self.add("optional_packages", "INFO",
                 "Not installed (prepare in next step): " + (", ".join(missing) or "none"))
        installed = sorted({(d.metadata.get("Name", "unknown"), d.version)
                            for d in metadata.distributions()}, key=lambda pair: pair[0].lower())
        (self.output_dir / "installed_packages.txt").write_text(
            "# Inventory only, not a validated requirements lockfile.\n" +
            "\n".join(f"{name}=={version}" for name, version in installed) + "\n",
            encoding="utf-8",
        )

        disks = []
        for path in dict.fromkeys([str(PROJECT_ROOT), "/root/autodl-tmp", "/"]):
            if Path(path).exists():
                usage = shutil.disk_usage(path)
                disks.append({"path": path, "total_bytes": usage.total,
                              "used_bytes": usage.used, "free_bytes": usage.free})
        self.report["storage"] = {
            "filesystems": disks,
            "note": "Filesystem capacity may exceed the instance quota. Also check AutoDL's disk panel.",
        }
        with tempfile.TemporaryFile(dir=PROJECT_ROOT) as handle:
            handle.write(b"rdrop-environment-write-check\n")
            handle.flush()
        self.add("project_directory_writable", "PASS", str(PROJECT_ROOT))
        free_gib = shutil.disk_usage(PROJECT_ROOT).free / 2**30
        self.add("project_disk_space", "PASS" if free_gib >= 15 else "WARN",
                 f"Filesystem reports {free_gib:.2f} GiB free; also check instance quota.")
        data_root = Path("/root/autodl-tmp")
        if platform.system() == "Linux":
            on_data_disk = data_root.exists() and (
                PROJECT_ROOT == data_root.resolve() or data_root.resolve() in PROJECT_ROOT.parents)
            self.add("project_location", "PASS" if on_data_disk else "WARN",
                     "Project is under /root/autodl-tmp." if on_data_disk else
                     "Verify project is on the data disk; usual AutoDL path is /root/autodl-tmp.")
        self.report["nvidia_smi"] = run_command([
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ])
        smi = self.report["nvidia_smi"]
        if smi["returncode"] == 0:
            fields = ("physical_index", "name", "uuid", "driver_version", "memory_total_mib",
                      "memory_used_mib", "utilization_percent", "temperature_c")
            smi["gpus"] = [dict(zip(fields, (v.strip() for v in row)))
                           for row in csv.reader(smi["stdout"].splitlines()) if row]
            self.add("nvidia_smi", "PASS", "GPU/driver inventory saved; logical CUDA indices may differ.")
        else:
            self.add("nvidia_smi", "WARN", "Command unavailable or failed; see JSON for details.")

    def cuda_checks(self, index):
        self.say("2/3 Importing PyTorch and checking CUDA...")
        try:
            import torch
        except Exception:
            self.report["torch_import_error"] = traceback.format_exc()
            self.add("torch_import", "FAIL", "PyTorch cannot be imported; error saved in environment.json.")
            return
        self.add("torch_import", "PASS", str(torch.__version__))
        actual_version = str(torch.__version__).split("+", 1)[0]
        self.add("torch_version", "PASS" if actual_version == EXPECTED["torch"] else "WARN",
                 f"actual={torch.__version__}, expected={EXPECTED['torch']}")
        self.report["torch"] = {
            "version": str(torch.__version__), "cuda_runtime": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cuda_available": torch.cuda.is_available(),
            "visible_device_count": torch.cuda.device_count(),
        }
        self.add("torch_cuda_runtime",
                 "PASS" if torch.version.cuda == EXPECTED["torch_cuda_runtime"] else "WARN",
                 f"actual={torch.version.cuda}, expected={EXPECTED['torch_cuda_runtime']}; separate from nvidia-smi.")
        if not self.report["torch"]["cuda_available"]:
            self.add("cuda_available", "FAIL", "CUDA is unavailable. No CPU fallback is used.")
            return
        if not 0 <= index < torch.cuda.device_count():
            self.add("cuda_device_index", "FAIL", f"Logical CUDA device {index} is not visible.")
            return
        torch.cuda.set_device(index)
        device = torch.device(f"cuda:{index}")
        props = torch.cuda.get_device_properties(device)
        free, total = torch.cuda.mem_get_info(device)
        self.report["torch"]["selected_gpu"] = {
            "logical_index": index, "name": props.name,
            "compute_capability": [props.major, props.minor],
            "total_memory_bytes": props.total_memory,
            "free_memory_before_test_bytes": free,
            "cuda_reported_total_memory_bytes": total,
            "supported_architectures_in_build": torch.cuda.get_arch_list(),
        }
        self.add("cuda_available", "PASS", f"Using {device}: {props.name}")
        self.add("gpu_model", "PASS" if props.name == EXPECTED["gpu_name"] else "WARN",
                 f"actual={props.name}, expected={EXPECTED['gpu_name']}")
        self.add("gpu_free_memory", "PASS" if free >= 4 * 2**30 else "WARN",
                 f"{free / 2**30:.2f} GiB free before tiny check; other GPU tasks can affect this.")
        self.say("3/3 Running tiny FP32 and FP16 AMP forward/backward/optimizer checks...")
        self.report["gpu_smoke_tests"] = {}
        for use_amp in (False, True):
            name = "fp16_amp" if use_amp else "fp32"
            try:
                result = tiny_gpu_step(torch, device, use_amp)
                self.report["gpu_smoke_tests"][name] = result
                self.add(name + "_forward_backward_update", "PASS",
                         f"loss={result['loss']:.6f}; finite gradients; parameters updated.")
            except Exception:
                self.report["gpu_smoke_tests"][name] = {
                    "status": "FAIL", "traceback": traceback.format_exc(),
                }
                self.add(name + "_forward_backward_update", "FAIL",
                         "CUDA test failed; traceback saved in environment.json.")
                break

    def finish(self):
        checks = self.report["checks"]
        failed = any(item["status"] == "FAIL" for item in checks)
        required = ("torch_import", "cuda_available", "fp32_forward_backward_update",
                    "fp16_amp_forward_backward_update", "project_directory_writable")
        passed = {item["name"] for item in checks if item["status"] == "PASS"}
        missing = [name for name in required if name not in passed]
        if missing and not failed:
            self.add("required_checks", "FAIL", "Not completed: " + ", ".join(missing))
            failed = True
        warned = any(item["status"] == "WARN" for item in checks)
        self.report["overall_status"] = "FAIL" if failed else ("PASS_WITH_WARNINGS" if warned else "PASS")
        self.report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        (self.output_dir / "environment.json").write_text(
            json.dumps(self.report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (self.output_dir / "environment.md").write_text(make_markdown(self.report), encoding="utf-8")
        self.say("\nOverall: " + self.report["overall_status"])
        self.say("Reports: " + str(self.output_dir))
        self.say("Send environment.md and environment.json back for review.")
        self.say("This checks only a tiny network; it is NOT evidence of BERT memory use or training speed.")
        self.log.close()
        return 1 if failed else 0


def tiny_gpu_step(torch, device, use_amp):
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model = torch.nn.Sequential(
        torch.nn.Linear(128, 64), torch.nn.ReLU(),
        torch.nn.Dropout(0.1), torch.nn.Linear(64, 2),
    ).to(device).train()
    inputs = torch.randn(16, 128, device=device)
    labels = torch.randint(0, 2, (16,), device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp, init_scale=128.0)
    first_parameter = next(model.parameters())
    before = first_parameter.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
        logits = model(inputs)
        loss = torch.nn.functional.cross_entropy(logits, labels)
    if logits.device.type != "cuda" or not torch.isfinite(loss).item():
        raise RuntimeError("Expected CUDA logits and a finite loss.")
    if use_amp and logits.dtype != torch.float16:
        raise RuntimeError(f"AMP did not produce float16 logits: {logits.dtype}")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grads = [p.grad for p in model.parameters()]
    if any(g is None or not torch.isfinite(g).all().item() for g in grads):
        raise RuntimeError("Missing or non-finite gradients.")
    grad_norm = sum(g.detach().float().square().sum().item() for g in grads) ** 0.5
    if grad_norm <= 0:
        raise RuntimeError("All gradients are zero.")
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize(device)
    changed = not torch.equal(before, first_parameter.detach())
    if not changed:
        raise RuntimeError("Optimizer did not change model parameters.")
    return {
        "status": "PASS", "device": str(device), "logits_dtype": str(logits.dtype),
        "loss": loss.detach().float().item(), "gradient_l2_norm": grad_norm,
        "parameters_updated": changed, "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "note": "Tiny diagnostic only. Timing includes setup; not a training cost measurement.",
    }


def make_markdown(report):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    system = report.get("system", {})
    torch_info = report.get("torch", {})
    gpu = torch_info.get("selected_gpu", {})
    lines = [
        "# R-Drop / MRPC 环境检查报告", "",
        f"- 检查时间（北京时间）：{report['started_at_china_time']}",
        f"- 总体状态：**{report['overall_status']}**",
        f"- 脚本版本：{report['script_version']}",
        f"- 项目路径：`{system.get('project_root', PROJECT_ROOT)}`",
        f"- Python：`{system.get('python', 'unknown')}`",
        f"- Python 可执行文件：`{system.get('python_executable', 'unknown')}`",
        f"- PyTorch：`{torch_info.get('version', 'unavailable')}`",
        f"- PyTorch CUDA 运行时：`{torch_info.get('cuda_runtime', 'unavailable')}`",
        f"- 实测 GPU 名称：`{gpu.get('name', 'unavailable')}`", "",
        "本报告用于环境预检、组员交接和报告素材。小网络检查不代表 BERT 已跑通，",
        "其显存和耗时不能作为正式实验成本。PASS_WITH_WARNINGS 需结合下表核对。", "",
        "## 检查结果", "", "| 项目 | 状态 | 说明 |", "|---|---|---|",
    ]
    lines += [f"| {cell(c['name'])} | {c['status']} | {cell(c['detail'])} |" for c in report["checks"]]
    lines += ["", "## 依赖版本", "", "| 包 | 版本 |", "|---|---|"]
    lines += [f"| {name} | {version or '未安装'} |" for name, version in report.get("packages", {}).items()]
    lines += ["", "## CPU、内存与驱动", "", "```json",
              json.dumps({"cpu": system.get("cpu"), "memory": system.get("memory"),
                          "nvidia_smi": report.get("nvidia_smi")}, ensure_ascii=False, indent=2),
              "```", "", "## 磁盘", "", "```json",
              json.dumps(report.get("storage", {}), ensure_ascii=False, indent=2), "```", "",
              "磁盘容量来自文件系统查询，可能不是 AutoDL 实例配额；请同时查看平台面板。", "",
              "详细 GPU 检查、异常堆栈和脚本哈希见同目录 environment.json。",
              "installed_packages.txt 是当前包清单，不是经过验证的安装依赖锁文件。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0, help="Visible logical CUDA device index (default: 0).")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "reports" / "environment",
                        help="Parent directory for timestamped report folders.")
    args = parser.parse_args()
    if args.device < 0:
        parser.error("--device must be non-negative")
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    output_root = args.output_root.expanduser().resolve()
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_CN_")
        output_dir = Path(tempfile.mkdtemp(prefix=timestamp, dir=output_root))
    except OSError as exc:
        print(f"Cannot create report directory: {exc}", file=sys.stderr)
        return 1
    checker = Checker(output_dir, args.device)
    checker.say("R-Drop environment check v" + SCRIPT_VERSION)
    checker.say("Output directory: " + str(output_dir))
    try:
        checker.inventory()
        checker.cuda_checks(args.device)
    except Exception:
        checker.report["unexpected_error"] = traceback.format_exc()
        checker.add("unexpected_error", "FAIL", "Error saved in environment.json; send the report for diagnosis.")
    return checker.finish()


if __name__ == "__main__":
    raise SystemExit(main())
