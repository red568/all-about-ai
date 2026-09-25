---
title: "第3课：现代 AI 系统全栈架构"
source: "https://everythingai.top/learn/ai-systems-performance-engineering-text/lesson-03"
author:
published:
created: 2026-09-26
description: "面向 AI 工程学习者的专业课程学习空间。"
tags:
  - "clippings"
---
**导航**

本课拆解模型、框架、编译器、运行时、硬件与基础设施之间的协作关系，建立现代 AI 系统的全栈视图。

课程进度：第 3/28 课

所属模块：AI 系统性能工程基础

实验环境：Level 0 仅需 Python 3.9+；Level 1 可选 PyTorch 与任意可用的 NVIDIA CUDA GPU；Level 2 面向特定互联与集群能力

## 课程定位

前两课回答了两个问题：性能工程师做什么，以及怎样用 Latency、Throughput、Goodput、MFU 和 Roofline 描述系统。本课把镜头拉远，回答第三个问题：\*\*一次训练 step 或一次推理请求，究竟穿过了哪些层？\*\*

AI 系统不是“模型 + GPU”。它是一条从入口流量、数据、调度、框架、编译器、算子库、CUDA Runtime、驱动、操作系统，一直延伸到 CPU、内存、PCIe、GPU、网络与存储的执行链。任何一层都可能让昂贵的加速器等待。

本课不会提前深挖某个 CUDA Kernel，也不会把 Kubernetes 配置罗列成运维手册。目标是建立一张可以反复使用的全栈地图：看到现象时，能够定位候选层；做优化时，能够判断收益是否会被上下游吞掉；设计系统时，能够让可观测性、回压和拓扑成为架构的一部分。

## 学习目标

学完本课，你应该能够：

1. 画出训练系统和 LLM 推理系统的端到端关键路径。
2. 区分数据平面、控制平面和可观测平面，不再把所有问题都归因于 GPU。
1. 解释框架、编译器、算子库、CUDA Runtime、驱动与硬件之间的边界。
2. 用阶段和、流水线瓶颈、传输模型与简化通信模型估算性能上限。
1. 根据“GPU 利用率低”“P99 抖动”“扩卡反而变慢”等症状选择第一批证据和工具。
2. 在无 GPU、常见 NVIDIA GPU和专用数据中心集群上使用同一套分层实验方法。
1. 识别只在 NVLink/NVSwitch、RDMA、MIG、FP8/FP4 等环境中成立的结论，避免伪等价模拟。

## 前置知识

- 能运行 Python 脚本并理解均值、百分位数和吞吐。
- 理解 CPU、GPU、内存、磁盘与网络的基本概念。
- 建议先完成第 2 课，熟悉 Latency、QPS、Tokens/s、Goodput 和 Roofline。
- Level 1 可选实验要求已安装与本机驱动兼容的 PyTorch；Level 0 不依赖第三方包。

## 核心直觉：GPU 是工厂里最贵的机器，不是整座工厂

把 AI 服务想象成一座工厂：

- 网关接单，准入控制决定哪些订单进入；
- tokenizer、数据加载器和 CPU 预处理准备原料；
- 调度器把原料凑成批次；
- GPU 执行矩阵乘、Attention 和通信；
- KV Cache、检查点和对象存储保存中间品；
- 监控系统记录每个工位的等待与产出；
- Kubernetes 或作业调度器负责分配厂房和机器。

若 GPU 每 2 ms 完成一次计算，但上游每 8 ms 才送来一个批次，换一张算力翻倍的 GPU，端到端吞吐几乎不会翻倍。相反，如果 H2D 复制和计算能并行，原本相加的两段时间可能变成取最大值。这就是全栈性能工程的第一原则：

第二个直觉是：\*\*平均利用率不是因果证据。\*\* 50% GPU 利用率可能来自数据加载不足、频繁小 Kernel、CPU launch gap、跨 NUMA 访问、同步通信、显存换入换出，也可能只是采样窗口把“满载—空闲”的周期平均了。先建立跨层时间线，再动参数。

## 一张现代 AI 系统全栈地图

### 逻辑分层

从上到下，可以把常见训练和推理系统划分为以下层：

| 层 | 典型组件 | 性能职责 | 常见问题 |
| --- | --- | --- | --- |
| 产品与入口 | API Gateway、队列、限流、SLA | 接收、排队、优先级、回压 | 排队爆炸、重试风暴、P99 抖动 |
| 应用与模型 | 训练循环、LLM Server、模型结构 | 定义工作负载与正确性 | 序列过长、batch 不合适、动态形状 |
| 调度与并行 | Continuous Batching、TP/PP/DP/EP | 合批、放置、切分、重叠 | 气泡、负载不均、队头阻塞 |
| 框架 | PyTorch、JAX 等 | 张量语义、自动微分、分布式接口 | eager 开销、隐式同步、对象分配 |
| 编译器与 DSL | torch.compile、XLA、Triton | 图捕获、融合、代码生成 | graph break、重编译、错误特化 |
| 加速库 | cuBLAS、cuDNN、NCCL、FlashAttention | 高性能算子和集合通信 | 算法选择不佳、形状不友好 |
| Runtime 与驱动 | CUDA Runtime/Driver、streams、events | 内存、任务提交、同步、设备管理 | launch gap、同步、上下文抖动 |
| OS 与容器 | Linux、cgroup、NUMA、Docker/Containerd | CPU/内存/I/O 隔离与设备暴露 | CPU 限流、page fault、拓扑错配 |
| 集群控制 | Kubernetes、Device Plugin、GPU Operator | 资源发现、调度、生命周期 | 驱动漂移、资源标签缺失、错误放置 |
| 硬件与网络 | CPU/DRAM/PCIe/GPU/HBM/NVLink/NIC/存储 | 真正执行计算与移动字节 | 带宽、延迟、容量、热/功耗限制 |

这不是严格的调用栈。例如 NCCL 会穿过 CUDA、驱动、PCIe/NVLink 或 NIC；一个 `torch.matmul` 可能由编译器融合，也可能落到 cuBLAS；容器并不虚拟出一块 GPU，而是通过主机驱动和设备接口使用它。分层的价值在于划清证据范围，而不是制造绝对边界。

### 三个正交平面

同一套分层还要从三个平面观察：

1. \*\*数据平面\*\*：张量、token、梯度、KV Cache、检查点真正移动和计算的路径。它直接决定 steady-state 吞吐与单请求关键路径。
2. \*\*控制平面\*\*：资源声明、调度、模型版本、健康检查、扩缩容、故障恢复。它不一定出现在每个 token 的热路径上，却决定系统能否稳定得到资源并维持 Goodput。
1. \*\*可观测平面\*\*：metrics、trace、profile、日志和实验元数据。它必须能够把入口请求、CPU 线程、CUDA stream、collective 与具体 worker 关联起来。

一个常见误区是让控制平面介入每一个细粒度操作。例如每个小请求都经过昂贵的远程决策，会把控制延迟放进数据热路径。更稳妥的设计是控制平面下发策略，数据平面在本地快速执行，并通过有界队列提供回压。

## 训练系统：一条 step 的真实旅程

### 端到端关键路径

典型训练 step 可以抽象为：

```
对象存储/并行文件系统
        ↓ 读取、解码、shuffle
CPU DataLoader → pinned host memory
        ↓ H2D copy
GPU forward → loss → backward
        ↓ 梯度 bucket
NCCL collective（AllReduce / ReduceScatter / AllGather）
        ↓
optimizer update → checkpoint / metrics
```

性能好的系统不会机械地串行执行所有箭头，而会建立流水线：worker 预取下一个 batch；pinned memory 配合独立 stream 发起异步 H2D；反向传播生成梯度 bucket 后立即启动通信；checkpoint 后台写入；下一 step 的计算与当前 step 的非关键工作重叠。

但“异步 API”不等于真正重叠。要同时满足：

- 两段工作之间没有未满足的数据依赖；
- 硬件有可并行的执行资源与路径；
- 使用了正确的 stream、event 和缓冲区生命周期；
- 没有 `.item()` 、日志、分配器或默认 stream 引入隐式同步；
- 上下游速率相匹配，有足够但不过量的缓冲。

### 训练中的控制平面

控制平面负责创建 worker、注入 rank/world size、建立 rendezvous、分配 GPU/NIC、处理失败与重新调度。Kubernetes 原生通过 device plugin 暴露 `nvidia.com/gpu` 等扩展资源；NVIDIA GPU Operator 可自动化驱动、Container Toolkit、device plugin、GPU Feature Discovery 与 DCGM 等组件的部署。它们解决的是“资源可被可靠发现和使用”，不会自动让模型获得最佳拓扑或最佳 batch size。

当作业跨节点时，调度决策会改变数据平面：同样的 AllReduce，可能走机内 NVLink，也可能绕经 PCIe 和跨机网络。全栈架构因此必须把 GPU、NUMA、NIC 与网络拓扑当成调度输入，而不是部署后的偶然事实。

## LLM 推理系统：一次请求与许多个 token

### 请求路径

典型生成式推理可以表示为：

```
Client
  ↓
Gateway / authentication / quota
  ↓
Admission control + request queue
  ↓
Tokenizer + scheduler + continuous batching
  ↓
Model worker
  ├─ Prefill：处理整段 prompt，通常更偏计算
  └─ Decode：逐 token 迭代，频繁访问权重与 KV Cache
  ↓
Detokenize / stream response
```

现代引擎通常把 API server、调度器和 GPU worker 分开。worker 内部再由模型执行器、KV Cache 管理器、attention backend 与通信后端协作。这个拆分带来三个关键约束：

- \*\*准入控制必须知道容量\*\*：只看 QPS 而不看输入/输出 token 长度，队列会突然失稳。
- \*\*调度器必须理解阶段差异\*\*：prefill 和 decode 的计算、带宽和延迟目标不同。
- \*\*KV Cache 是容量与带宽问题\*\*：它既决定最大并发，也参与每一步 decode 的内存访问。

控制平面可以做服务发现、模型发布和扩缩容，但请求级调度、batch 重组和 KV block 分配通常位于低延迟数据路径。第 23～28 课会逐步展开这些机制。

### 为什么只看“模型推理耗时”会误判

客户端体验的延迟至少包含：入口排队、tokenize、调度等待、prefill、逐 token decode、网络发送和客户端背压。模型 worker 的 CUDA 时间下降 20%，若排队占了总延迟的 70%，端到端收益仍然有限。另一方面，盲目加大 batch 可能提高总 Tokens/s，却恶化 TTFT 或单用户 TPOT。

架构设计必须同时保留：

- 请求级 trace：端到端、TTFT、每 token 间隔；
- 调度器指标：队列长度、被抢占请求、batch 组成、KV 使用率；
- worker 指标：CPU gap、Kernel、显存与通信；
- 业务结果：超时率、成功 token、SLO 内完成的 Goodput。

## 执行层如何接力

以 PyTorch 中的一次矩阵乘为例：

1. Python/框架创建张量操作，执行 eager 调度或进入捕获的计算图。
2. 编译器可能进行形状特化、算子融合并生成代码；未编译路径可能直接分派到后端。
1. cuBLAS 等库根据 dtype、形状、布局和硬件选择实现。
2. CUDA Runtime 管理 stream、event、内存与 Kernel launch，并调用驱动。
1. 驱动把工作提交给 GPU；GPU 在 SM/Tensor Core 上计算，并通过缓存、HBM 与互联移动数据。
2. CPU 只有在显式/隐式同步、数据依赖或资源不足时必须等待完成。

这解释了为什么“GPU Kernel 已经很快”仍不代表系统快：Python 可能发不满；图可能频繁重编译；张量布局可能触发额外复制；进程可能被 cgroup 限流；PCIe 可能跨 NUMA；多卡通信可能走低带宽路径。

## 容器与 Kubernetes：隔离不是性能魔法

### 容器边界

GPU 容器通常共享宿主机内核和 NVIDIA 驱动，容器内携带用户态 CUDA 相关库与应用依赖。最常见的兼容性原则是：\*\*宿主机驱动必须支持容器所需的 CUDA 用户态版本\*\*。不要仅因为容器里能看到 `nvcc` 就断定运行路径正确；也不要在镜像中随意覆盖宿主驱动组件。

容器还会继承 CPU quota、cpuset、memory limit、shared memory、ulimit 和 IPC 配置。一个 GPU 训练容器若只得到很少 CPU，DataLoader 可能饿死 GPU； `/dev/shm` 太小可能导致多进程数据加载报错；跨 NUMA 的 CPU 与 GPU 绑定则可能降低 H2D 或网络吞吐。

### Kubernetes GPU 资源模型

Kubernetes 通过厂商 device plugin 把 GPU 作为扩展资源交给 Pod。标准资源请求通常放在 `limits` 中，例如：

```
apiVersion: v1
kind: Pod
metadata:
  name: gpu-smoke-test
spec:
  restartPolicy: Never
  containers:
    - name: test
      image: nvcr.io/nvidia/cuda:12.8.1-base-ubuntu22.04
      command: ["nvidia-smi"]
      resources:
        limits:
          nvidia.com/gpu: 1
```

镜像 tag 只是示例，部署前应选择与组织驱动策略匹配且仍受维护的版本。申请到“1 张 GPU”也不代表申请到特定架构、显存容量或互联。生产环境需要借助节点标签、亲和性、污点/容忍、队列和拓扑感知策略描述这些约束。Lesson 9 会专门讨论 Linux、Docker 和 Kubernetes 调优。

## 拓扑：数据走哪条路，常常比算多少更重要

### CPU、NUMA、PCIe 与 GPU

双路服务器通常有多个 NUMA node。每张 GPU 和 NIC 挂在某个 CPU/PCIe root complex 下。若数据加载线程在 NUMA 0、GPU 在 NUMA 1，host memory 访问和 H2D 可能跨 socket；若负责通信的 NIC 离 GPU 很远，网络路径也会增加跳数。

第一批只读检查命令是：

```
lscpu
numactl --hardware                 # 安装 numactl 后可用
nvidia-smi topo -m                 # NVIDIA GPU 环境
nvidia-smi --query-gpu=name,pci.bus_id,memory.total --format=csv
```

这些命令告诉你“可能走什么路径”，但不能替代实际带宽、延迟与端到端 profile。

### 架构能力不要靠型号字符串猜

课程采用以下正确映射：

| 架构 | 常见 GPU | 本课处理方式 |
| --- | --- | --- |
| Ampere | RTX 3080/3090、A100 | Level 1 常见基线；消费卡与 A100 的显存/互联能力不同 |
| Ada Lovelace | RTX 4090、L40/L40S | Level 1 常见推理基线；RTX 4090 不是 Blackwell |
| Hopper | H100/H200 | Level 2；可进一步验证 FP8、Transformer Engine、NVLink 等 |
| Blackwell | RTX 5090、B100/B200/GB200 | Level 2；能力因消费卡与数据中心产品而异 |

代码应优先查询 CUDA Compute Capability、dtype 支持、显存与库版本，再开启优化。即使同属一个架构，RTX 3090 与 A100、RTX 5090 与 B200 的互联、显存类型、MIG 与数据中心功能也不等价。

NVLink/NVSwitch、GPUDirect RDMA、MIG、FP8/FP4 或 Transformer Engine 无法用无对应硬件的机器“等价模拟”。没有这些硬件时，可以验证调度逻辑、阶段模型和回退分支，但不能据此声称验证了真实链路带宽或数值吞吐。

## 关键性能模型

### 1\. 串行关键路径

若一次请求的各阶段完全串行：

$$
T_{e2e}=T_{queue}+T_{pre}+T_{transfer}+T_{compute}+T_{comm}+T_{post}
$$

优化某一阶段的端到端上限受 Amdahl 定律约束。若阶段占比为 p，加速倍数为 s：

$$
S_{e2e}=\frac{1}{(1-p)+p/s}
$$

因此一个只占 10% 的阶段即使加速 10 倍，端到端也只有约 $1/(0.9+0.01)=1.10\times$ 。

### 2\. 流水线吞吐

对稳定的多阶段流水线，忽略气泡和额外开销后：

$$
T_{interval}\approx \max_i(T_i), \qquad Throughput\approx\frac{1}{\max_i(T_i)}
$$

首个请求仍需穿过所有阶段，所以流水线通常提高 steady-state 吞吐，而不会同等降低单个请求的首次延迟。队列太深还会增大排队时间。

### 3\. 计算与传输重叠

不重叠时：

$$
T_{serial}=T_{copy}+T_{compute}
$$

理想重叠时：

$$
T_{overlap}\approx\max(T_{copy},T_{compute})+T_{unhidden}
$$

其中 $T_{unhidden}$ 包含依赖、同步、启动、气泡和资源竞争。重叠不是免费收益：复制引擎和计算可能竞争内存带宽。

### 4\. 数据移动模型

移动 S 字节的近似耗时：

$$
T_{move}=\alpha+\frac{S}{BW_{effective}}
$$

$\alpha$ 是固定启动延迟， $BW_{effective}$ 是有效带宽。许多小传输受 $\alpha$ 支配；少量大传输更接近带宽上限。合并传输、减少往返和改善拓扑，往往比单纯提升峰值 FLOPS 更有效。

### 5\. 简化 Ring AllReduce 模型

对 P 个 rank、每个 rank 有 M 字节数据，忽略协议细节与拥塞，可粗略写为：

$$
T_{allreduce}\approx 2(P-1)\alpha+2\frac{P-1}{P}\frac{M}{BW_{effective}}
$$

它用于建立直觉：小消息更受延迟影响，大消息更受带宽与拓扑影响。真实 NCCL 会根据拓扑、算法、协议和版本选择路径，必须通过 NCCL 日志与 profiler 验证，不能拿此式替代实测。

## 瓶颈分析方法：从现象到证据链

### 第一步：先确定优化目标和边界

必须写清：

- 优化训练 step、端到端 epoch、在线 TTFT、TPOT、吞吐还是 Goodput？
- 是否保持相同模型、精度、输出正确性与请求分布？
- 范围包含排队、tokenizer、数据加载、checkpoint 与网络吗？
- 测量是在 warmup 后，还是混入了初始化、编译和缓存建立？

### 第二步：画出关键路径与并发路径

把请求 ID、batch ID、step ID、rank、CPU 线程和 CUDA stream 对齐到同一时间线。找出：

- GPU 空洞前 CPU 在做什么；
- copy 与 compute 是否重叠；
- rank 是否同时进入 collective；
- 排队时间是在入口、scheduler 还是 worker；
- 周期性尖峰是否与 GC、checkpoint、日志或扩缩容一致。

### 第三步：按层选择最小证据

| 症状 | 首选候选层 | 第一批证据/工具 |
| --- | --- | --- |
| GPU 周期性空闲 | DataLoader、CPU launch、同步 | Nsight Systems、PyTorch Profiler、CPU profile、队列深度 |
| Kernel 很忙但吞吐低 | 算子、内存、形状 | Nsight Compute、算子形状、Roofline、显存带宽 |
| 加 GPU 后缩放差 | 并行策略、collective、拓扑 | NCCL 日志、通信/计算时间线、 `nvidia-smi topo -m` |
| P50 好而 P99 很差 | 排队、调度、抖动、重试 | 请求 trace、队列时间、batch 组成、GC/CPU throttling |
| H2D 时间高 | pageable memory、NUMA、碎片传输 | pinned/pageable 对照、消息大小、NUMA/PCIe 拓扑 |
| 容器慢、宿主机快 | cgroup、CPU/shm、挂载、驱动路径 | CPU quota、cpuset、 `/dev/shm` 、容器 runtime 配置 |
| OOM 但显存均值不高 | 峰值、碎片、KV/激活增长 | memory snapshot、按阶段峰值、请求长度分布 |
| 首次请求慢 | 初始化、JIT/compile、缓存 | 冷/热分组、编译日志、模型加载时间线 |

### 第四步：每次只验证一个因果假设

一个可验证假设应长这样：

它比“把 `num_workers` 调大试试”更好，因为说明了证据、动作、预期和失效条件。优化后同时检查正确性、P50/P99、资源利用、峰值内存和失败率，避免把问题推给下一层。

## 完整可运行实验：观察串行全栈与预取流水线

本实验把一个 AI 数据路径缩小为四段：存储读取、CPU 预处理、设备传输/计算、后处理。它不会假装模拟 NVLink 或真实分布式集群；它用同一脚本展示“阶段相加”和“阶段重叠”的差别，并收集运行环境能力。

将下面代码保存为 `ai_full_stack_lab.py` ：

```
#!/usr/bin/env python3
"""A dependency-free pipeline lab with optional PyTorch/CUDA compute."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import queue
import random
import shutil
import subprocess
import tempfile
import threading
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

def timed_command(argv: List[str]) -> Dict[str, Any]:
    """Run a read-only inventory command without shell expansion."""
    if shutil.which(argv[0]) is None:
        return {"available": False}
    try:
        result = subprocess.run(
            argv, capture_output=True, text=True, timeout=5, check=False
        )
        return {
            "available": True,
            "returncode": result.returncode,
            "stdout": result.stdout.strip()[:4000],
            "stderr": result.stderr.strip()[:1000],
        }
    except (subprocess.SubprocessError, OSError) as exc:
        return {"available": True, "error": repr(exc)}

def host_memory_gib() -> Optional[float]:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) / 1024 / 1024, 2)
    except (OSError, ValueError, IndexError):
        pass
    return None

def inventory() -> Dict[str, Any]:
    inv: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "host_memory_gib": host_memory_gib(),
        "container_hint": Path("/.dockerenv").exists()
        or Path("/run/.containerenv").exists(),
    }
    inv["nvidia_smi"] = timed_command([
        "nvidia-smi",
        "--query-gpu=name,uuid,memory.total,driver_version,pci.bus_id",
        "--format=csv,noheader",
    ])
    inv["gpu_topology"] = timed_command(["nvidia-smi", "topo", "-m"])
    return inv

def percentile(values: List[float], q: float) -> float:
    data = sorted(values)
    if not data:
        return 0.0
    pos = (len(data) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(data) - 1)
    frac = pos - lo
    return data[lo] * (1 - frac) + data[hi] * frac

def make_dataset(path: Path, items: int, block_bytes: int) -> None:
    rng = random.Random(20260805)
    with path.open("wb") as f:
        for _ in range(items):
            f.write(rng.randbytes(block_bytes))

def read_and_prepare(
    f, item_id: int, block_bytes: int, io_delay_ms: float
) -> Tuple[bytes, Dict[str, float]]:
    t0 = time.perf_counter()
    f.seek(item_id * block_bytes)
    payload = f.read(block_bytes)
    if io_delay_ms:
        # Explicitly models remote/object-storage latency; it is not a disk claim.
        time.sleep(io_delay_ms / 1000.0)
    t1 = time.perf_counter()
    prepared = hashlib.blake2b(payload, digest_size=32).digest()
    t2 = time.perf_counter()
    return prepared, {
        "read_ms": (t1 - t0) * 1000,
        "preprocess_ms": (t2 - t1) * 1000,
    }

class ComputeEngine:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.torch = None
        self.device = "cpu"
        self.capability = None
        if args.torch:
            try:
                import torch
            except ImportError as exc:
                raise SystemExit("--torch requires PyTorch in this environment") from exc
            self.torch = torch
            if args.device == "cuda":
                if not torch.cuda.is_available():
                    raise SystemExit("--device cuda requested, but CUDA is unavailable")
                self.device = "cuda"
                self.capability = list(torch.cuda.get_device_capability())
            torch.manual_seed(20260805)
            self.base = torch.randn(args.matrix_n, args.matrix_n, dtype=torch.float32)

    def run(self, prepared: bytes) -> Tuple[str, Dict[str, float]]:
        if self.torch is None:
            t0 = time.perf_counter()
            digest = prepared
            for _ in range(self.args.compute_rounds):
                digest = hashlib.sha256(digest).digest()
            t1 = time.perf_counter()
            return digest.hex()[:16], {
                "transfer_ms": 0.0,
                "compute_ms": (t1 - t0) * 1000,
                "postprocess_ms": 0.0,
            }

        torch = self.torch
        scale = 1.0 + prepared[0] / 2550.0
        host = self.base * scale
        if self.args.pin_memory and self.device == "cuda":
            host = host.pin_memory()
        t0 = time.perf_counter()
        x = host.to(
            self.device,
            non_blocking=self.args.pin_memory and self.device == "cuda",
        )
        if self.device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        y = x @ x
        if self.device == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        checksum = f"{float(y[0, 0].cpu()):.6f}"
        t3 = time.perf_counter()
        return checksum, {
            "transfer_ms": (t1 - t0) * 1000,
            "compute_ms": (t2 - t1) * 1000,
            "postprocess_ms": (t3 - t2) * 1000,
        }

    def metadata(self) -> Dict[str, Any]:
        if self.torch is None:
            return {"backend": "python-cpu"}
        torch = self.torch
        data: Dict[str, Any] = {
            "backend": "pytorch",
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "device": self.device,
        }
        if self.device == "cuda":
            data.update({
                "gpu_name": torch.cuda.get_device_name(),
                "compute_capability": self.capability,
                "memory_gib": round(
                    torch.cuda.get_device_properties(0).total_memory / 2**30, 2
                ),
            })
        return data

def merge_stage(dst: Dict[str, List[float]], src: Dict[str, float]) -> None:
    for key, value in src.items():
        dst[key].append(value)

def summarize(
    name: str,
    total_s: float,
    stages: Dict[str, List[float]],
    checksums: List[str],
) -> Dict[str, Any]:
    return {
        "name": name,
        "total_ms": round(total_s * 1000, 3),
        "items_per_second": round(len(checksums) / total_s, 3),
        "stage_mean_ms": {k: round(mean(v), 4) for k, v in stages.items()},
        "stage_p95_ms": {k: round(percentile(v, 0.95), 4) for k, v in stages.items()},
        "checksum_sample": checksums[:3],
        "checksums": checksums,
    }

def run_serial(
    path: Path, args: argparse.Namespace, engine: ComputeEngine
) -> Dict[str, Any]:
    stages: Dict[str, List[float]] = defaultdict(list)
    checksums: List[str] = []
    start = time.perf_counter()
    with path.open("rb") as f:
        for item_id in range(args.items):
            prepared, stage = read_and_prepare(
                f, item_id, args.block_bytes, args.io_delay_ms
            )
            merge_stage(stages, stage)
            checksum, stage = engine.run(prepared)
            merge_stage(stages, stage)
            checksums.append(checksum)
    return summarize("serial", time.perf_counter() - start, stages, checksums)

def run_prefetch(
    path: Path, args: argparse.Namespace, engine: ComputeEngine
) -> Dict[str, Any]:
    work_queue: queue.Queue[Any] = queue.Queue(maxsize=args.prefetch)
    stages: Dict[str, List[float]] = defaultdict(list)
    checksums: List[str] = []
    sentinel = object()

    def producer() -> None:
        with path.open("rb") as f:
            for item_id in range(args.items):
                prepared, stage = read_and_prepare(
                    f, item_id, args.block_bytes, args.io_delay_ms
                )
                work_queue.put((item_id, prepared, stage))
        work_queue.put(sentinel)

    start = time.perf_counter()
    thread = threading.Thread(target=producer, name="prefetch", daemon=True)
    thread.start()
    while True:
        item = work_queue.get()
        if item is sentinel:
            break
        _, prepared, producer_stage = item
        merge_stage(stages, producer_stage)
        checksum, compute_stage = engine.run(prepared)
        merge_stage(stages, compute_stage)
        checksums.append(checksum)
    thread.join()
    return summarize("prefetch", time.perf_counter() - start, stages, checksums)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--items", type=int, default=40)
    p.add_argument("--block-bytes", type=int, default=256 * 1024)
    p.add_argument("--io-delay-ms", type=float, default=3.0)
    p.add_argument("--compute-rounds", type=int, default=8000)
    p.add_argument("--prefetch", type=int, default=2)
    p.add_argument("--torch", action="store_true")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--matrix-n", type=int, default=512)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--json-out", type=Path, default=Path("stack_report.json"))
    args = p.parse_args()
    if args.items < 2 or args.block_bytes < 1024 or args.prefetch < 1:
        p.error("items >= 2, block-bytes >= 1024, prefetch >= 1 are required")
    return args

def main() -> None:
    args = parse_args()
    engine = ComputeEngine(args)
    with tempfile.TemporaryDirectory(prefix="ai-stack-lab-") as tmp:
        dataset = Path(tmp) / "dataset.bin"
        make_dataset(dataset, args.items, args.block_bytes)
        # Warm up the selected compute path outside the reported interval.
        engine.run(b"warmup".ljust(32, b"0"))
        serial = run_serial(dataset, args, engine)
        prefetch = run_prefetch(dataset, args, engine)

    if serial["checksums"] != prefetch["checksums"]:
        raise RuntimeError("correctness check failed: pipeline changed item results")
    speedup = serial["total_ms"] / prefetch["total_ms"]
    report = {
        "config": vars(args) | {"json_out": str(args.json_out)},
        "inventory": inventory(),
        "compute": engine.metadata(),
        "serial": {k: v for k, v in serial.items() if k != "checksums"},
        "prefetch": {k: v for k, v in prefetch.items() if k != "checksums"},
        "speedup": round(speedup, 3),
        "correctness": "PASS",
        "interpretation": (
            "Prefetch overlaps producer work with consumer work; "
            "speedup depends on the slowest stage and available resources."
        ),
    }
    args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
```

### Level 0：通用 CPU 回退实验

无需 GPU，也无需安装 PyTorch：

```
python3 --version
python3 ai_full_stack_lab.py \
  --items 40 \
  --io-delay-ms 3 \
  --compute-rounds 8000 \
  --json-out stack_report.json
```

再去掉人为存储等待，观察优化边界：

```
python3 ai_full_stack_lab.py \
  --items 40 \
  --io-delay-ms 0 \
  --compute-rounds 8000 \
  --json-out stack_report_no_io_delay.json
```

`--io-delay-ms` 明确表示可控的远程/对象存储等待模型，不代表你的本地磁盘真实性能。第一次实验的目的，是让生产者等待与消费者计算存在可重叠区间；第二次用于证明当生产者不再显著等待时，线程预取可能收益很小甚至变慢。

### Level 1：常见 NVIDIA GPU 基线

在隔离环境中安装与你的 Python、驱动和 CUDA 路径匹配的 PyTorch。安装命令会变化，应从 PyTorch 官方安装选择器获取，不要盲目复制某台机器的 wheel URL。安装后先做能力检测：