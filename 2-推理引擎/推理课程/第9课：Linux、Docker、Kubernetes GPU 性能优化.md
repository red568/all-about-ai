---
title: "第9课：Linux、Docker、Kubernetes GPU 性能优化"
source: "https://everythingai.top/learn/ai-systems-performance-engineering-text/lesson-09"
author:
published:
created: 2026-09-26
description: "面向 AI 工程学习者的专业课程学习空间。"
tags:
  - "clippings"
---
**导航**

本课从 Linux、容器与 Kubernetes 全链路定位 GPU 之外的系统瓶颈，让训练和推理环境稳定高效。

## 课程定位

GPU 利用率低，不一定是 GPU Kernel 慢。数据可能卡在磁盘、CPU 解码、NUMA 远端内存、Pinned Memory、容器共享内存、CPU CFS 配额或 Kubernetes 错误调度上。此时继续优化 CUDA Kernel，往往等于给堵车中的跑车换发动机。

本课建立从 Linux 主机到容器再到 Kubernetes Pod 的性能诊断链。课程不会提供一份“复制粘贴后永久修改所有 sysctl”的激进脚本，而是采用可复现的工程方法：先采集基线，再只改一个变量，记录回滚方法，最后用端到端 Goodput 验证。

## 学习目标

完成本课后，你能够：

1. 解释 Linux 调度、NUMA、Page Cache、Swap、THP、Pinned Memory 如何影响 GPU。
2. 区分 NVIDIA Driver、CUDA Driver API、CUDA Runtime、Toolkit 与容器内用户态库。
1. 正确配置和验证 NVIDIA Container Toolkit，而不是把完整驱动打进镜像。
2. 使用 cgroup v2 指标定位 CPU Throttling、Memory Pressure 与 OOM。
1. 解释 Docker 的 CPU/Memory/ `/dev/shm` /IPC/NUMA 配置对训练与推理的影响。
2. 正确配置 Kubernetes GPU Resource、QoS、CPU Manager 与 Topology Manager。
1. 区分整卡、Time-Slicing、MPS 与 MIG 的共享和隔离语义。
2. 在 CPU-only、通用 NVIDIA GPU 和数据中心专项环境中完成分层实验。

## 前置知识

- 会使用 Linux Shell、Python、Docker；Level 2 需要可选 Kubernetes。
- 理解 GPU Memory、PCIe、NVLink、NUMA 与 Pinned Memory。
- 理解 Latency、Throughput、P50/P99、Goodput。
- 了解 Pod、Container、Request、Limit 的基本概念。

## 核心直觉：GPU 是流水线末端的消费者

训练或推理数据路径可抽象为：

```
Storage / Network
        ↓
Page Cache / Filesystem
        ↓
CPU Read → Decode → Tokenize → Collate
        ↓
Pinned Host Memory
        ↓ PCIe / NVLink-C2C
GPU Memory
        ↓
GPU Kernels
```

系统吞吐受最慢阶段限制：

$$
R_{pipeline}\leq \min(R_{storage},R_{decode},R_{collate},R_{H2D},R_{GPU})
$$

若 GPU 每个 Batch 计算 40 ms，数据准备却要 55 ms，那么 GPU 再快 50% 也不会让端到端吞吐提高 50%。真正要优化的是 55 ms 的上游阶段，或者用预取把它与 GPU 计算重叠。

## 从主机到容器的 GPU 软件栈

```
应用：PyTorch / JAX / TensorFlow / vLLM
用户态库：CUDA Runtime、cuBLAS、cuDNN、NCCL
容器接口：NVIDIA Container Toolkit / CDI / Runtime Hook
主机内核态：NVIDIA Kernel Driver
硬件：GPU、PCIe/NVLink、NUMA、NIC、Storage
```

关键规则：

- NVIDIA 内核驱动由主机安装并与主机 Kernel 匹配；
- 容器镜像通常携带应用所需的 CUDA 用户态 Runtime 和库；
- NVIDIA Container Toolkit 把设备和兼容的驱动能力暴露给容器；
- 容器内的 CUDA Toolkit 版本不应超过主机驱动可支持范围；
- `nvidia-smi` 显示的 CUDA Version 是驱动最高兼容能力，不等于当前 Python Wheel 编译所用版本；
- `nvcc --version` 只说明本地 Toolkit 编译器，不证明 PyTorch 正在使用它。

因此，排查时至少同时记录：

```
uname -a
nvidia-smi
nvcc --version || true
python -c 'import torch; print(torch.__version__, torch.version.cuda)'
```

## Linux 性能原理

### CPU 调度与 GPU Bubble

GPU Launch、DataLoader、Tokenizer、NCCL Progress 和网络处理都需要 CPU。若线程被频繁迁移、被 CFS 配额 Throttle、与系统中断争抢，GPU Timeline 会出现空洞。

一次 Step 可分解为：

$$
T_{step}=T_{GPU}+T_{input,exposed}+T_{comm,exposed}+T_{runtime}+T_{sync}
$$

其中 `exposed` 表示没有被并行隐藏的部分。CPU 优化的目标不是追求 100% CPU 利用率，而是降低关键路径上的等待和抖动。

诊断命令：

```
mpstat -P ALL 1
pidstat -wt -p <PID> 1
vmstat 1
perf stat -p <PID> -e context-switches,cpu-migrations,page-faults
```

### NUMA：CPU、内存、GPU 和 NIC 的距离

双路服务器中，CPU 访问本地 NUMA Memory 通常比远端更快、更稳定。GPU 也挂在特定 PCIe Root/NUMA Node 下。若 GPU 0 的 DataLoader 在线程属于 NUMA 0，却从 NUMA 1 分配大块 Pinned Memory，H2D 路径会多一次跨 Socket 传输。

```
numactl -H
nvidia-smi topo -m
cat /sys/bus/pci/devices/0000:XX:YY.Z/numa_node
numastat -p <PID>
```

仅用于 A/B 的绑定示例：

```
numactl --cpunodebind=0 --membind=0 python train.py
```

不要假设 NUMA 0 一定连接 GPU 0；先看实际 PCI Bus ID 与拓扑。

### CPU Affinity、Thread Pool 与 DataLoader

过少 CPU Worker 会饿死 GPU；过多 Worker 会造成：

- Context Switch 增多；
- Page Cache 和内存压力增加；
- 多个 BLAS/OpenMP Thread Pool 过度订阅；
- 每个 Worker 复制 Python 对象，放大 Host RAM；
- 容器 CPU Limit 内发生严重 Throttling。

先固定隐式线程数再扫描 Worker：

```
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

PyTorch 常用起点：

```
loader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=workers > 0,
    prefetch_factor=2 if workers > 0 else None,
)
```

`pin_memory=True` 不是无条件越多越好。Pinned Page 不能被普通方式换出，占用过大时会挤压系统内存，并受 `ulimit -l` 、cgroup 和 OS 策略影响。

### Swap 与内存压力

训练进程发生 Swap-in 会带来毫秒到秒级停顿。先观察，再决定策略：

```
free -h
swapon --show
vmstat 1
cat /proc/pressure/memory
cat /proc/<PID>/status | rg 'VmRSS|VmSwap|VmLck'
```

`vm.swappiness=0` 并不是所有机器的通用答案。它倾向于尽量避免匿名页换出，但不能解决内存容量不足，也不保证绝不 Swap。生产节点应结合 OOM 策略、Checkpoint、作业优先级和共享负载决定。

临时 A/B：

```
sysctl vm.swappiness
sudo sysctl -w vm.swappiness=10
# 测试后恢复记录中的原值
```

### Page Cache、Readahead 与 Checkpoint

Linux Page Cache 能显著加速重复读取，但首次读取、随机小文件和远程文件系统的行为不同。Checkpoint 写入会制造 Dirty Page Burst，可能与训练读数据、NCCL 网络或同盘日志争用。

```
iostat -xz 1
cat /proc/meminfo | rg 'Cached|Dirty|Writeback'
cat /proc/pressure/io
```

不要在共享生产机用 `drop_caches` 作为日常“优化”。它会破坏其他任务的缓存，只适合获得授权的隔离基准环境。

### Transparent Huge Pages

THP 用更大页降低 CPU TLB 压力，但也可能引入 Compaction、分配延迟和尾延迟变化。检查：

```
cat /sys/kernel/mm/transparent_hugepage/enabled
cat /proc/meminfo | rg 'AnonHugePages|HugePages'
```

`always` 、 `madvise` 、 `never` 应以工作负载实测选择。GPU HBM Page、CUDA Unified Memory 和 Linux Host THP 是不同层次，不能混为一谈。

### CPU Frequency、C-State 与能耗

Latency 敏感服务可对比 `performance` Governor 与默认策略：

```
cpupower frequency-info
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
```

更高频率和更浅 C-State 可能降低抖动，也会增加功耗、温度和噪声。对长时间 GPU 训练，CPU 不是瓶颈时强制最高频只会降低性能/瓦特。

### GPU Persistence、Clock 与功耗

```
nvidia-smi --query-gpu=index,name,persistence_mode,pstate,clocks.sm,power.draw,temperature.gpu --format=csv
sudo nvidia-smi -pm 1
```

Persistence Mode 主要减少作业之间反复初始化的 Cold Start，不会直接提高 Kernel FLOPS。锁定 Application Clock、Power Limit 或 Exclusive Process Mode 具有型号、权限和多租户影响，必须由集群管理员操作并记录回滚值。

## Docker 性能原理

### 容器不是虚拟 GPU

Linux 容器共享主机 Kernel，GPU Kernel 仍在真实设备执行。正确配置时，计算 Kernel 本身通常没有传统虚拟机式模拟开销。容器场景的性能问题更多来自：

- CPU Quota/Cpuset；
- Memory Limit、OOM 与 Swap；
- `/dev/shm` 太小；
- OverlayFS 小文件与写放大；
- 容器网络；
- 缺失 RDMA、IPC Lock、NUMA 或设备权限；
- 镜像和 Host Driver/Runtime 不兼容。

### CPU Quota 与 CFS Throttling

假设容器每周期允许 CPU 时间 Q，周期长度为 P：

$$
CPU_{quota}=\frac{Q}{P}
$$

达到 Quota 后，即使主机还有空闲 CPU，容器也可能等到下一周期。DataLoader 常表现为平均 CPU 不高，但 `nr_throttled` 和 `throttled_usec` 增长。

cgroup v2 检查：

```
cat /sys/fs/cgroup/cpu.max
cat /sys/fs/cgroup/cpu.stat
cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.events
```

Docker A/B：

```
docker run --rm --cpus=4 ubuntu:24.04 nproc
docker run --rm --cpuset-cpus=0-3 --cpuset-mems=0 ubuntu:24.04 nproc
```

### /dev/shm 与多进程 DataLoader

Docker 默认共享内存通常较小。PyTorch 多进程、NCCL SHM Transport、Python Multiprocessing 和浏览器型服务可能出现：

- `Bus error` ；
- `No space left on device` ，但磁盘仍有空间；
- DataLoader Worker 意外退出；
- 隐式回退或吞吐下降。

```
df -h /dev/shm
docker run --rm --shm-size=8g <image> <command>
```

`--ipc=host` 可共享主机 IPC Namespace，但扩大隔离边界，不应作为默认答案。优先为容器设置明确的 `--shm-size` ；Kubernetes 使用 Memory-backed `emptyDir` 。

### 镜像与文件系统

最佳实践：

- 数据集和 Checkpoint 使用显式 Volume，不写入容器可写层；
- 大量小文件先做 Shard/Tar/Parquet 等布局优化；
- 多阶段构建减小镜像；
- 依赖版本锁定，镜像使用不可变 Digest；
- 把编译缓存放在持久 Volume，并按架构隔离；
- 镜像拉取时间与运行时吞吐分开统计。

### NVIDIA Container Toolkit 验证

主机先有可用 NVIDIA Driver，再安装 Toolkit。官方当前配置入口是：

```
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

验证：

```
docker run --rm --gpus all ubuntu:24.04 nvidia-smi
```

不要把主机的 `/usr/lib` 整目录手工挂入容器。Toolkit 会按设备请求和 Driver Capability 处理设备与库。Podman 等 Runtime 可使用 CDI；是否采用 CDI 取决于 Runtime 和 Toolkit 版本。

## Kubernetes GPU 性能原理

### GPU 是扩展资源

安装 NVIDIA Device Plugin 或 GPU Operator 后，节点通常暴露 `nvidia.com/gpu` 。GPU Resource 只能按整数请求，不能像 CPU 那样请求 `500m` 。标准 Pod 中可只写 `limits` ，Kubernetes 会用同值作为 Request；若同时写 Request 与 Limit，两者必须相等。

```
resources:
  requests:
    cpu: "4"
    memory: 16Gi
    nvidia.com/gpu: "1"
  limits:
    cpu: "4"
    memory: 16Gi
    nvidia.com/gpu: "1"
```

### QoS 与 CPU 独占

要获得 `Guaranteed` QoS，Pod 内每个 Container 的 CPU/Memory Request 与 Limit 必须满足相应条件。结合 kubelet `CPU Manager static` ，具有整数 CPU Request 的合格容器可获得独占 CPU Set，减少迁移和共享干扰。

检查：

```
kubectl get pod <pod> -o jsonpath='{.status.qosClass}{"\n"}'
kubectl exec <pod> -- cat /sys/fs/cgroup/cpuset.cpus.effective
kubectl exec <pod> -- cat /sys/fs/cgroup/cpu.stat
```

CPU Limit 过紧时，即使是 Guaranteed Pod，也可能在突发预处理阶段 Throttle。优化目标是稳定供应而非盲目压缩 Request。

### Topology Manager 的真实边界

Kubelet Topology Manager 协调 CPU Manager、Device Manager 等 Hint Provider，在\*\*单节点\*\*内尝试把 CPU、内存和设备对齐到合适 NUMA Node。策略包括：

- `none` ：不做对齐；
- `best-effort` ：尽量对齐，失败仍准入；
- `restricted` ：无可接受拓扑时拒绝；
- `single-numa-node` ：要求单 NUMA Node 的首选对齐。

它不是跨节点网络拓扑调度器，也不保证一个多 GPU Request 总能得到同一 NVLink Clique。机架、交换机、Rail、NVLink Domain 等信息仍需 Node Label、Affinity、DRA/专用调度扩展或集群平台能力表达。

### 整卡、Time-Slicing、MPS 与 MIG

| 方式 | 共享方式 | 显存隔离 | 故障隔离 | 适合场景 |
| --- | --- | --- | --- | --- |
| 整卡 | 独占设备 | 强 | 较强 | 训练、稳定推理基线 |
| Time-Slicing | 时间复用 | 无 | 无，同一 Fault Domain | 低负载开发/轻推理 |
| CUDA MPS | 多进程并发与配额能力 | 有限/依配置 | 弱于 MIG | 小 Kernel、多进程吞吐 |
| MIG | 硬件分区 | 有 | 有 | 支持 MIG 的数据中心 GPU 多租户 |

NVIDIA Device Plugin 的 Time-Slicing Replica 不是一块“虚拟小 GPU”，也不提供显存或故障隔离。一个 Pod OOM/异常可能影响共享者。MPS 与 Time-Slicing 互斥；MIG 只在支持的 GPU 与配置上可用。

消费卡 RTX 3080/3090、RTX 4090、RTX 5090 不支持 MIG。A100、H100/H200 以及部分更新数据中心产品可按具体 SKU 和当前官方支持矩阵配置 MIG。没有 MIG 硬件时不能用容器配额模拟等价隔离。

### GPU Operator 的角色

GPU Operator 可管理 Driver、Container Toolkit、Device Plugin、GPU Feature Discovery、DCGM Exporter、MIG Manager 等组件。它减少生命周期管理工作，但不会自动解决：

- 错误的 Pod CPU/Memory Request；
- 数据集局部性；
- 跨节点拓扑；
- 用户代码中的同步和 DataLoader 问题；
- 共享策略带来的干扰。

## 瓶颈分析方法

### 四层漏斗

```
业务：Tokens/s、Samples/s、P99、SLO、失败率
应用：Data Time、Compute Time、Communication Time、Queue Time
容器：CPU Throttle、Memory Event、/dev/shm、Block/Network I/O
节点：NUMA、Frequency、PSI、Storage、PCIe、GPU/NIC Topology
```

只有四层指标在同一时间轴上，才能避免“GPU Util 低所以加 GPU”这种错误结论。

### 诊断决策树

```
GPU 利用率低？
├─ GPU Timeline 有 Kernel，但间隔大
│  ├─ DataLoader/Data Time 高 → CPU、I/O、Worker、Pinned Memory
│  ├─ NCCL 高 → 拓扑与通信
│  └─ Launch 碎片 → Runtime/Graph/Fusion
├─ GPU 无进程或初始化慢 → Driver、Runtime、权限、Persistence
├─ 容器比主机慢
│  ├─ cpu.stat throttled 增长 → CPU Quota
│  ├─ memory.events 增长 → Limit/OOM/Pressure
│  ├─ /dev/shm 满 → SHM 配额
│  └─ Volume/OverlayFS → 存储路径
└─ Pod 间性能不稳定
   ├─ QoS/Cpuset/NUMA 不一致
   ├─ GPU 共享干扰
   ├─ 节点型号/时钟/温度不一致
   └─ 网络/存储邻居噪声
```

## 完整可运行实验：主机、容器与 Pod 审计

实验分为三级：

- Level 0：任何 Linux/CPU 机器可运行的系统审计；
- Level 1：自动检测 NVIDIA Driver、GPU、拓扑和 PyTorch；
- Level 2：Docker 与 Kubernetes 对照部署。

### Level 0/1 完整代码：system\_gpu\_audit.py

```
#!/usr/bin/env python3
"""只读系统审计：不修改 sysctl、CPU Governor、GPU Clock 或容器配置。"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import statistics
import subprocess
import time
from pathlib import Path

def read_text(path: str, default: str = "unavailable") -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").strip()
    except (OSError, PermissionError):
        return default

def run(argv: list[str], timeout: int = 10) -> dict:
    if not shutil.which(argv[0]):
        return {"available": False, "command": argv, "output": "command not found"}
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                           check=False)
        output = (p.stdout or p.stderr).strip()
        return {"available": True, "command": argv, "returncode": p.returncode,
                "output": output}
    except subprocess.TimeoutExpired:
        return {"available": True, "command": argv, "timeout": True}

def parse_cpu_max(text: str) -> dict:
    parts = text.split()
    if len(parts) != 2:
        return {"raw": text, "quota_cpus": None}
    quota, period = parts
    if quota == "max":
        return {"raw": text, "quota_cpus": "unlimited"}
    try:
        return {"raw": text, "quota_cpus": round(int(quota) / int(period), 3)}
    except (ValueError, ZeroDivisionError):
        return {"raw": text, "quota_cpus": None}

def parse_key_values(text: str) -> dict:
    result = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) == 2:
            key, value = fields
            try:
                result[key] = int(value)
            except ValueError:
                result[key] = value
    return result

def cgroup_v2_path() -> Path:
    # /proc/self/cgroup 常见格式：0::/user.slice/...
    for line in read_text("/proc/self/cgroup", "").splitlines():
        fields = line.split(":", 2)
        if len(fields) == 3 and fields[0] == "0":
            return Path("/sys/fs/cgroup") / fields[2].lstrip("/")
    return Path("/sys/fs/cgroup")

def cpu_jitter(samples: int, work: int) -> dict:
    """微型 CPU 调度抖动探针，不是通用 CPU Benchmark。"""
    times_ms = []
    checksum = 0
    for _ in range(samples):
        start = time.perf_counter_ns()
        local = 0
        for i in range(work):
            local = (local + i * i) % 1_000_000_007
        checksum ^= local
        times_ms.append((time.perf_counter_ns() - start) / 1e6)
    ordered = sorted(times_ms)
    p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return {
        "samples": samples,
        "work_per_sample": work,
        "median_ms": round(statistics.median(times_ms), 4),
        "p95_ms": round(ordered[p95_index], 4),
        "max_ms": round(max(times_ms), 4),
        "jitter_max_over_median": round(max(times_ms) / statistics.median(times_ms), 3),
        "checksum": checksum,
    }

def linux_audit(args: argparse.Namespace) -> dict:
    cg = cgroup_v2_path()
    cpu_max = read_text(str(cg / "cpu.max"))
    return {
        "platform": platform.platform(),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "pid": os.getpid(),
        "logical_cpus_visible": os.cpu_count(),
        "affinity_cpus": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [],
        "loadavg": os.getloadavg() if hasattr(os, "getloadavg") else None,
        "cgroup_v2_path": str(cg),
        "cpu_max": parse_cpu_max(cpu_max),
        "cpu_stat": parse_key_values(read_text(str(cg / "cpu.stat"), "")),
        "memory_current": read_text(str(cg / "memory.current")),
        "memory_max": read_text(str(cg / "memory.max")),
        "memory_events": parse_key_values(read_text(str(cg / "memory.events"), "")),
        "cpuset_effective": read_text(str(cg / "cpuset.cpus.effective")),
        "memory_nodes_effective": read_text(str(cg / "cpuset.mems.effective")),
        "transparent_hugepage": read_text(
            "/sys/kernel/mm/transparent_hugepage/enabled"
        ),
        "swappiness": read_text("/proc/sys/vm/swappiness"),
        "memory_psi": read_text("/proc/pressure/memory"),
        "io_psi": read_text("/proc/pressure/io"),
        "shm": run(["df", "-h", "/dev/shm"]),
        "numa": run(["numactl", "-H"]),
        "cpu_jitter_probe": cpu_jitter(args.samples, args.work),
    }

def nvidia_audit() -> dict:
    result = {
        "nvidia_smi": run(["nvidia-smi"]),
        "gpu_query": run([
            "nvidia-smi",
            "--query-gpu=index,name,compute_cap,pci.bus_id,memory.total,"
            "persistence_mode,pstate,temperature.gpu,power.draw",
            "--format=csv",
        ]),
        "topology": run(["nvidia-smi", "topo", "-m"]),
    }
    try:
        import torch
        result["pytorch"] = {
            "version": torch.__version__,
            "compiled_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "visible_devices": torch.cuda.device_count(),
            "cudnn_version": torch.backends.cudnn.version(),
        }
        if torch.cuda.is_available():
            result["pytorch"]["devices"] = [
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "compute_capability": list(torch.cuda.get_device_capability(i)),
                }
                for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        result["pytorch"] = {"available": False, "reason": "PyTorch not installed"}
    return result

def recommendations(audit: dict) -> list[str]:
    notes = []
    cpu_stat = audit["linux"]["cpu_stat"]
    if cpu_stat.get("nr_throttled", 0) > 0:
        notes.append("cgroup has observed CPU throttling; compare delta during workload.")
    mem_events = audit["linux"]["memory_events"]
    if mem_events.get("oom", 0) > 0 or mem_events.get("oom_kill", 0) > 0:
        notes.append("cgroup has OOM events; inspect host RAM and container/Pod limits.")
    if audit["linux"]["memory_current"] == "unavailable":
        notes.append("cgroup v2 metrics unavailable; system may use v1 or restrict access.")
    if not audit["nvidia"]["nvidia_smi"].get("available", False):
        notes.append("nvidia-smi unavailable; Level 0 is still valid.")
    notes.append("Compare counters before/after the target workload; cumulative values alone are not rates.")
    notes.append("This audit is read-only and intentionally changes no system setting.")
    return notes

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=30)
    p.add_argument("--work", type=int, default=100_000)
    p.add_argument("--output", default="-")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    audit = {"linux": linux_audit(args), "nvidia": nvidia_audit()}
    audit["recommendations"] = recommendations(audit)
    text = json.dumps(audit, ensure_ascii=False, indent=2)
    if args.output == "-":
        print(text)
    else:
        Path(args.output).write_text(text + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
```

### Level 0：通用 Linux 实验

```
python3 system_gpu_audit.py --output audit-host.json
python3 -m json.tool audit-host.json | less
```

### 预期现象

- 无 GPU 的机器仍会输出 Kernel、Affinity、cgroup、THP、Swap、PSI、SHM 和 Jitter；
- 容器内 `logical_cpus_visible` 可能大于 Affinity CPU 数；
- `cpu.max` 为 `max 100000` 表示没有 cgroup CPU Quota；
- `memory.events` 中 `oom_kill` 增长说明发生过 cgroup OOM Kill；
- CPU Jitter 的 Max/Median 在系统有邻居负载时通常更大。

脚本中的微型循环不是 CPU 选型基准，只用于同一机器、同一环境、单变量 A/B。

### Level 1：通用 NVIDIA GPU 实验

安装与 Driver 匹配的稳定 PyTorch 后：

```
python3 system_gpu_audit.py --output audit-gpu.json
nvidia-smi topo -m
watch -n 1 nvidia-smi
```

双 RTX 3080 20GB 环境重点检查：

- 两卡 Compute Capability、PCI Bus ID 与显存是否正确；
- 两卡路径是 `PIX/PXB/PHB/SYS` 中哪一种；
- Persistence Mode、P-State、温度、功耗是否一致；
- DataLoader CPU Affinity 是否覆盖 GPU 所在 NUMA 的本地核心；
- 3080 不支持 MIG，也没有 NVLink，不要执行 MIG/NVSwitch 专项命令。

### Docker 对照实验

先在主机生成基线，然后在容器执行同一份脚本。假设脚本在当前目录：

```
docker run --rm \
  --gpus all \
  --cpus=4 \
  --memory=8g \
  --shm-size=2g \
  -v "$PWD:/work:ro" \
  -w /work \
  python:3.12-slim \
  python system_gpu_audit.py
```

这个基础 Python 镜像没有 PyTorch，但能完成 Linux/cgroup 审计并调用被 Toolkit 注入的 `nvidia-smi` 。若需要 PyTorch，使用与硬件和 Driver 兼容、来源可信且固定 Digest 的 CUDA/PyTorch 镜像。

做三组 A/B：

```
# A：严格 CPU 配额
docker run --rm --cpus=1 -v "$PWD:/work:ro" -w /work \
  python:3.12-slim python system_gpu_audit.py

# B：更宽 CPU 配额
docker run --rm --cpus=4 -v "$PWD:/work:ro" -w /work \
  python:3.12-slim python system_gpu_audit.py

# C：绑定 CPU/NUMA，CPU 列表要按自己的拓扑修改
docker run --rm --cpuset-cpus=0-3 --cpuset-mems=0 \
  -v "$PWD:/work:ro" -w /work \
  python:3.12-slim python system_gpu_audit.py
```

对比 `cpu.max` 、Affinity、 `nr_throttled` 增量和 Jitter。不要拿不同宿主机的一次结果下结论。

## Level 2：Kubernetes 可执行实验

### Pod Manifest：gpu-system-audit.yaml

```
apiVersion: v1
kind: Pod
metadata:
  name: gpu-system-audit
  labels:
    app: gpu-system-audit
spec:
  restartPolicy: Never
  containers:
    - name: audit
      # 生产中替换为内部已验证镜像及不可变 digest
      image: nvcr.io/nvidia/cuda:13.0.2-runtime-ubuntu24.04
      command: ["bash", "-lc"]
      args:
        - |
          set -euo pipefail
          echo "qos/cgroup"
          cat /sys/fs/cgroup/cpu.max || true
          cat /sys/fs/cgroup/cpu.stat || true
          cat /sys/fs/cgroup/memory.events || true
          echo "shm"
          df -h /dev/shm
          echo "gpu"
          nvidia-smi
          nvidia-smi topo -m
          sleep 3600
      resources:
        requests:
          cpu: "4"
          memory: 8Gi
          nvidia.com/gpu: "1"
        limits:
          cpu: "4"
          memory: 8Gi
          nvidia.com/gpu: "1"
      volumeMounts:
        - name: dshm
          mountPath: /dev/shm
  volumes:
    - name: dshm
      emptyDir:
        medium: Memory
        sizeLimit: 2Gi
```

运行：

```
kubectl apply -f gpu-system-audit.yaml
kubectl wait --for=condition=Ready pod/gpu-system-audit --timeout=180s
kubectl get pod gpu-system-audit -o wide
kubectl get pod gpu-system-audit -o jsonpath='{.status.qosClass}{"\n"}'
kubectl logs gpu-system-audit
kubectl exec gpu-system-audit -- cat /sys/fs/cgroup/cpuset.cpus.effective
kubectl delete pod gpu-system-audit
```

### 预期现象

- GPU Plugin 正常时，Pod 内只看到分配的 GPU；
- CPU/Memory Request 与 Limit 相等时应获得 Guaranteed QoS；
- 配置 `CPU Manager static` 且满足条件时，Pod 可获得独占 CPU Set；
- `/dev/shm` 来自 Memory-backed `emptyDir` ，容量由 `sizeLimit` 和 Pod Memory 共同约束；
- Topology Manager 是否对齐不能只看 Pod Running，要结合 kubelet 配置、CPU Set、GPU PCI/NUMA 信息验证。

### 节点与 Operator 检查

```
kubectl get nodes -o wide
kubectl describe node <node-name> | sed -n '/Capacity:/,/Allocated resources:/p'
kubectl get pods -A | rg 'nvidia|gpu-operator|device-plugin|dcgm'
kubectl get node <node-name> --show-labels
kubectl get runtimeclass
```

### Topology Manager 节点配置边界

典型 kubelet 配置片段由集群管理员管理：

```
cpuManagerPolicy: static
topologyManagerPolicy: single-numa-node
topologyManagerScope: pod
reservedSystemCPUs: "0-1"
```

修改 Kubelet Policy 前必须 Drain Node，并遵循 Kubernetes 官方关于 CPU Manager State Checkpoint 的迁移步骤。直接改 Policy 后重启可能导致 Kubelet CrashLoop。课程实验不自动修改节点配置。

## 架构专项与不可等价边界

### Ampere：RTX 3080/3090、A100

- RTX 3080/3090：验证 PCIe、NUMA、Docker、Kubernetes 整卡/MPS；无 MIG；3080 无 NVLink，3090 仅特定双卡桥接。
- A100：可选验证 MIG、MPS、NVLink/NVSwitch 和 GPUDirect；PCIe 与 SXM/HGX 不能混写。

### Ada：RTX 4090、L40/L40S

- 重点验证 PCIe、CPU/NUMA、容器推理和 MPS；
- RTX 4090/L40S 无 NVLink；
- 不支持 MIG 的型号不能用 Time-Slicing 冒充硬件分区。

### Hopper：H100/H200

- 可选验证 MIG、NVLink 4/NVSwitch、Transformer Engine、Fabric Manager；
- PCIe、SXM、NVL/HGX 的互联和部署能力不同；
- MIG 重配置可能要求停止 GPU Client，必须走平台维护流程。

### Blackwell：RTX 5090、B100/B200/GB200

- RTX 5090 是消费级 Blackwell：PCIe，无 NVLink，不等于 B200/GB200；
- B100/B200/GB200 的 MIG、NVLink 5、NVSwitch、Fabric Manager 与新 Driver/Toolkit 要按具体平台支持矩阵验证；
- 双 3080 只能模拟“带宽降低对吞吐的影响”，不能模拟 MIG、NVLink 5、NVSwitch、FP4 或机架级 Fabric。

## 优化前后对照

| 层级 | 常见低效配置 | 优化方向 | 验证指标 |
| --- | --- | --- | --- |
| Linux CPU | Worker 迁移、线程过订阅 | Affinity、线程池与 Worker 扫描 | Data Time、Context Switch |
| NUMA | CPU/Memory/GPU 跨 Socket | 按真实拓扑局部绑定 | H2D GB/s、P99 |
| Memory | Swap、Pinned 过量 | 容量规划、锁页预算 | PSI、VmSwap、VmLck |
| Storage | 小文件、Checkpoint 阻塞 | Shard、异步/分布式写 | IOPS、Await、Step P99 |
| Docker CPU | CPU Quota 过紧 | 合理 Request/Limit/Cpuset | `throttled_usec` 增量 |
| Docker IPC | `/dev/shm` 默认过小 | 明确 SHM 容量 | Worker 错误、吞吐 |
| Container FS | 数据写 OverlayFS | 显式 Volume/本地缓存 | Block I/O、Startup |
| Kubernetes | Burstable、随机 NUMA | Guaranteed + CPU/Topology Manager | CPU Set、NUMA、Jitter |
| GPU Sharing | Time-Slicing 当隔离 | 整卡/MPS/MIG 按目标选择 | P99、OOM、干扰率 |
| Scheduling | 只按 GPU 数调度 | 型号、显存、拓扑、数据局部性 | Goodput、排队时间 |

## 结果分析方法

### 不要比较“裸机一次”与“容器另一次”

正确 A/B 要固定：

- 同一节点、GPU、Driver、Clock、温度；
- 同一镜像/依赖/模型/数据；
- 同一 Batch、Seed、精度和正确性标准；
- Warmup 后多次采样；
- 同一 CPU Set、NUMA Node、存储缓存状态；
- 记录邻居负载。

### 用增量而不是累计值

`cpu.stat` 、 `memory.events` 是累计计数。应在实验前后采样：

$$
\Delta throttled=throttled_{after}-throttled_{before}
$$

再计算 Throttle Ratio：

$$
R_{throttle}=\frac{\Delta throttled_usec}{\Delta wall_time\times CPU_{allocated}}
$$

如果 Ratio 高且 Data Time/P99 同时恶化，才有证据指向 CPU Quota。

### 端到端胜利条件

至少同时满足：

- Samples/s 或 Tokens/s 上升；
- P95/P99 不恶化；
- 正确性一致；
- OOM/Worker Failure 不增加；
- GPU 功耗、CPU 功耗与成本可接受；
- 连续多轮无明显回退。

## 常见错误与排查

### 错误 1：容器里 nvidia-smi 正常，PyTorch 却 CUDA Unavailable

`nvidia-smi` 只证明 Driver 设备暴露。检查 PyTorch 是否为 CUDA Build、Wheel CUDA 版本、 `LD_LIBRARY_PATH` 污染、Compute Capability 支持和容器设备权限。

### 错误 2：把主机 Driver 安装进容器

容器不应加载自己的 Kernel Driver。主机负责内核模块，容器使用匹配的用户态库和 Toolkit 注入的 Driver 能力。

### 错误 3：DataLoader 报 Bus Error

先看 `df -h /dev/shm` 。增加 `--shm-size` 或 Kubernetes Memory-backed `emptyDir` ，同时检查 Pod Memory Limit；不要第一时间无限增大 Worker。

### 错误 4：GPU 利用率周期性掉零

关联检查 Data Time、Disk Await、Page Fault、CPU Throttle、Checkpoint、GC、NCCL Synchronization 和 Thermal Throttling。

### 错误 5：CPU 使用率不高，却仍被 CPU Limit 影响

平均值掩盖了 CFS 周期性 Throttling。检查 `cpu.stat` 的 `nr_throttled` 与 `throttled_usec` 增量。

### 错误 6：Pod 是 Running，却没有 NUMA 对齐

`best-effort` 允许非首选对齐；CPU Manager 默认 `none` 也不给独占 CPU。检查 Kubelet Policy、QoS、整数 CPU Request、Cpuset 与 GPU PCI NUMA。

### 错误 7：请求 4 块 GPU 就认为它们必在同一 NVLink 域

标准扩展资源主要表达数量。需要平台额外标签、Affinity、DRA/拓扑调度能力或整节点分配，并在 Pod 内用 `nvidia-smi topo -m` 验证。

### 错误 8：Time-Slicing Replica 当成显存切片

Time-Slicing 没有显存和 Fault Isolation。需要强隔离时选择整卡或支持硬件上的 MIG，并验证资源名称和 Profile。

### 错误 9：将 vm.swappiness=0、关闭 THP、关闭 C-State 当万能脚本

这些是工作负载和平台相关旋钮。任何永久修改都要有基线、授权、回滚和功耗/尾延迟验证。

### 错误 10：每个 Worker 都启动大量 OpenMP Thread

例如 16 个 DataLoader Worker × 16 个 OpenMP Thread 会造成严重过订阅。固定线程池为 1 起步，再扫描 Worker 数。

### 错误 11：Kubernetes Memory Limit 只算 Python RSS

还要考虑 Page Cache、共享内存、Pinned Memory、子进程和部分映射。检查 cgroup `memory.current` 、 `memory.stat` 、 `memory.events` ，不要只看进程 RSS。

### 错误 12：性能测试时修改了十几个系统参数

无法归因，也无法安全回滚。一次只改一个假设，使用相同脚本重复运行并记录差异。

## 面试题与答案

### 1\. 为什么 GPU Util 低不一定要优化 GPU？

GPU 是流水线末端。CPU 解码、I/O、H2D、通信或 Runtime Launch 任何阶段供给不足，都会让 GPU 空闲。要用 Timeline 和分阶段时间确认瓶颈。

### 2\. Docker 会让 CUDA Kernel 变慢吗？

容器共享主机 Kernel，GPU Kernel 运行在真实硬件。正确配置时主要差异通常来自 cgroup、NUMA、SHM、存储、网络和依赖，而不是 GPU 指令被模拟。

### 3\. nvidia-smi 中 CUDA Version 与 torch.version.cuda 有什么区别？

前者表示 Driver 可支持的最高 CUDA 兼容能力，后者表示当前 PyTorch Binary 编译使用的 CUDA Runtime 版本。二者不必相同，但必须满足兼容关系。

### 4\. CPU Request=4、Limit=4 为什么更稳定？

它有助于获得 Guaranteed QoS，并在 CPU Manager Static 下满足独占整数 CPU 的条件。但是否真正独占仍取决于 Kubelet 配置和 Pod 全部 Container 资源声明。

### 5\. Topology Manager 能解决跨机架调度吗？

不能。它是 Kubelet 单节点组件，协调 NUMA Hint Provider。跨节点/机架网络拓扑需要 Scheduler 扩展、标签/Affinity、DRA 或平台能力。

### 6\. Time-Slicing、MPS 和 MIG 的主要区别？

Time-Slicing 主要是时间复用且无显存/故障隔离；MPS 改善多进程并发并可管理部分配额；MIG 在支持硬件上提供更强的硬件资源和故障隔离。

### 7\. 为什么 /dev/shm 会影响 PyTorch DataLoader？

多进程间传输 Tensor 和元数据可使用共享内存。SHM 太小会导致 Bus Error、Worker 退出或传输失败。

### 8\. NUMA 绑定为什么可能适得其反？

如果绑定到错误 Node、容量不足或数据/NIC 位于另一 Node，会增加远端访问或 OOM。必须先确认 GPU/CPU/Memory/NIC 拓扑。

### 9\. Persistence Mode 提升什么性能？

主要减少 GPU 无 Client 后再次初始化的启动延迟，适合频繁启动任务；不会提高稳定态 Kernel 的理论吞吐。

### 10\. 如何判断 CPU Quota 是瓶颈？

测量工作负载期间 `nr_throttled` 、 `throttled_usec` 增量，并与 Data Time、GPU Bubble 和吞吐做 A/B。仅看到 Limit 不足以证明瓶颈。

### 11\. 为什么不能只看平均 GPU Util？

平均值会掩盖周期性 Data Stall、Checkpoint、最慢 Rank 和尾延迟。至少结合时间序列、Step P99 和 Goodput。

### 12\. Kubernetes 中 GPU Request 和 Limit 怎么写？

GPU 是整数扩展资源。可以只写 Limit，Request 默认等于 Limit；若同时写，两者必须相等，不能只写 GPU Request。

## 课后练习

1. 在主机和 Docker 中运行审计脚本，对比 Affinity、 `cpu.max` 、SHM 和 Jitter。
2. 扫描 DataLoader `num_workers=0/2/4/8/16` ，记录 Data Time、Step Time 和 Host RAM。
1. 对比 `pin_memory=False/True` ，同时记录 H2D 时间、VmLck 与系统可用内存。
2. 找出双 GPU 分别对应的 NUMA Node，设计合理的 CPU Worker 绑定方案。
1. 构造 `--cpus=1` 与 `--cpus=4` 的容器 A/B，计算 Throttle Ratio。
2. 把 Kubernetes Pod 改为 Burstable，再比较 QoS、Cpuset 和抖动。
1. 将 `/dev/shm` 从 64MiB 扫描到 2GiB，观察多进程作业的错误和吞吐。
2. 为一台 8-GPU 双路服务器设计 CPU/GPU/NIC/NUMA 资源标签。
1. 写出 Time-Slicing 适用与禁止使用的场景清单。
2. 制作一份永久修改 sysctl/Clock 前的审批、基线与回滚模板。

## Checklist

### Linux

- 已记录 Kernel、Driver、CUDA Runtime、Framework 版本。
- 已检查 CPU Affinity、Thread Pool 和 Worker 过订阅。
- 已确认 CPU/Memory/GPU/NIC 的 NUMA 距离。
- 已检查 Swap、PSI、Page Fault、THP 与 Pinned Memory。
- 已区分 CPU 平均利用率和 cgroup Throttling。
- 系统参数修改有基线、授权和回滚。

### Docker

- 主机 Driver 正常，容器未重复安装 Kernel Driver。
- NVIDIA Container Toolkit/Runtime 能正确暴露设备。
- CPU、Memory、Cpuset、NUMA 与 SHM 配置明确。
- 数据和 Checkpoint 不写入 OverlayFS 可写层。
- 主机与容器使用相同工作负载做 A/B。
- 已检查 cgroup v2 cpu.stat 和 memory.events 增量。

### Kubernetes

- GPU 以整数扩展资源请求，Request/Limit 语义正确。
- 延迟敏感 Pod 的 QoS 与 CPU Manager 条件已验证。
- Topology Manager 策略与 Scope 已记录。
- Pod 内实际 CPU Set、GPU 与 NUMA 对齐已验证。
- /dev/shm 使用合适的 Memory-backed emptyDir。
- 节点型号、显存、互联和数据局部性已用于调度。

### GPU 共享

- 已区分整卡、Time-Slicing、MPS 与 MIG。
- 没有把 Time-Slicing 当作显存或故障隔离。
- 只在支持的 GPU 上配置 MIG。
- 多租户干扰以 P95/P99 和错误率验证。
- 双 RTX 3080 按 PCIe、无 MIG、无 NVLink 基线处理。

### 验收

- Samples/s 或 Tokens/s 提升。
- P95/P99、失败率和正确性未恶化。
- GPU Bubble 与 Exposed Data/Communication Time 降低。
- 功耗、温度和性能/瓦特可接受。
- 优化可复现、可回滚、可自动化检查。