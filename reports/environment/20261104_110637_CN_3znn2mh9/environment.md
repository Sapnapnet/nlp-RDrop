# R-Drop / MRPC 环境检查报告

- 检查时间（北京时间）：2026-11-04T11:06:37.708494+08:00
- 总体状态：**PASS**
- 脚本版本：1.0.0
- 项目路径：`/root/autodl-tmp/rdrop-mrpc-project`
- Python：`3.10.8 (main, Nov 24 2022, 14:13:03) [GCC 11.2.0]`
- Python 可执行文件：`/root/miniconda3/bin/python`
- PyTorch：`2.1.2+cu118`
- PyTorch CUDA 运行时：`11.8`
- 实测 GPU 名称：`NVIDIA GeForce RTX 3090`

本报告用于环境预检、组员交接和报告素材。小网络检查不代表 BERT 已跑通，
其显存和耗时不能作为正式实验成本。PASS_WITH_WARNINGS 需结合下表核对。

## 检查结果

| 项目 | 状态 | 说明 |
|---|---|---|
| python_version | PASS | actual=3.10, expected=3.10 |
| operating_system | PASS | Ubuntu 22.04.1 LTS |
| optional_packages | INFO | Not installed (prepare in next step): transformers, datasets, tokenizers, huggingface-hub, accelerate, evaluate, scikit-learn, scipy, safetensors |
| project_directory_writable | PASS | /root/autodl-tmp/rdrop-mrpc-project |
| project_disk_space | PASS | Filesystem reports 50.00 GiB free; also check instance quota. |
| project_location | PASS | Project is under /root/autodl-tmp. |
| nvidia_smi | PASS | GPU/driver inventory saved; logical CUDA indices may differ. |
| torch_import | PASS | 2.1.2+cu118 |
| torch_version | PASS | actual=2.1.2+cu118, expected=2.1.2 |
| torch_cuda_runtime | PASS | actual=11.8, expected=11.8; separate from nvidia-smi. |
| cuda_available | PASS | Using cuda:0: NVIDIA GeForce RTX 3090 |
| gpu_model | PASS | actual=NVIDIA GeForce RTX 3090, expected=NVIDIA GeForce RTX 3090 |
| gpu_free_memory | PASS | 23.30 GiB free before tiny check; other GPU tasks can affect this. |
| fp32_forward_backward_update | PASS | loss=0.710565; finite gradients; parameters updated. |
| fp16_amp_forward_backward_update | PASS | loss=0.710587; finite gradients; parameters updated. |

## 依赖版本

| 包 | 版本 |
|---|---|
| torch | 2.1.2+cu118 |
| numpy | 1.26.4 |
| transformers | 未安装 |
| datasets | 未安装 |
| tokenizers | 未安装 |
| huggingface-hub | 未安装 |
| accelerate | 未安装 |
| evaluate | 未安装 |
| scikit-learn | 未安装 |
| scipy | 未安装 |
| safetensors | 未安装 |
| protobuf | 4.25.3 |
| packaging | 24.1 |

## CPU、内存与驱动

```json
{
  "cpu": {
    "model": "Intel(R) Xeon(R) Gold 6330 CPU @ 2.00GHz",
    "os_logical_cpus": 112,
    "affinity_cpus": 112,
    "cgroup_quota_cores": 14.0
  },
  "memory": {
    "MemTotal_bytes": 810969751552,
    "MemAvailable_bytes": 760158978048,
    "cgroup_limit_bytes": 96636764160,
    "effective_capacity_estimate_bytes": 96636764160,
    "note": "Host RAM and container limits are separate; limits may be unavailable."
  },
  "nvidia_smi": {
    "command": [
      "nvidia-smi",
      "--query-gpu=index,name,uuid,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu",
      "--format=csv,noheader,nounits"
    ],
    "returncode": 0,
    "stdout": "0, NVIDIA GeForce RTX 3090, GPU-deee7a72-8a0c-61a0-daaf-601d37c7b23e, 580.105.08, 24576, 0, 0, 28",
    "stderr": "",
    "gpus": [
      {
        "physical_index": "0",
        "name": "NVIDIA GeForce RTX 3090",
        "uuid": "GPU-deee7a72-8a0c-61a0-daaf-601d37c7b23e",
        "driver_version": "580.105.08",
        "memory_total_mib": "24576",
        "memory_used_mib": "0",
        "utilization_percent": "0",
        "temperature_c": "28"
      }
    ]
  }
}
```

## 磁盘

```json
{
  "filesystems": [
    {
      "path": "/root/autodl-tmp/rdrop-mrpc-project",
      "total_bytes": 53687091200,
      "used_bytes": 102400,
      "free_bytes": 53686988800
    },
    {
      "path": "/root/autodl-tmp",
      "total_bytes": 53687091200,
      "used_bytes": 102400,
      "free_bytes": 53686988800
    },
    {
      "path": "/",
      "total_bytes": 32212254720,
      "used_bytes": 1340813312,
      "free_bytes": 30871441408
    }
  ],
  "note": "Filesystem capacity may exceed the instance quota. Also check AutoDL's disk panel."
}
```

磁盘容量来自文件系统查询，可能不是 AutoDL 实例配额；请同时查看平台面板。

详细 GPU 检查、异常堆栈和脚本哈希见同目录 environment.json。
installed_packages.txt 是当前包清单，不是经过验证的安装依赖锁文件。
