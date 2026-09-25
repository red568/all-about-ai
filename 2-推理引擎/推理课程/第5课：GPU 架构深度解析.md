---
title: "第5课：GPU 架构深度解析"
source: "https://everythingai.top/learn/ai-systems-performance-engineering-text/lesson-05"
author:
published:
created: 2026-09-26
description: "面向 AI 工程学习者的专业课程学习空间。"
tags:
  - "clippings"
---
**导航**

本课深入 GPU 微架构与执行模型，帮助你理解 SM、Warp、调度与并行效率之间的关系。

## 课程定位

写出能在 GPU 上运行的程序并不难，难的是解释它为什么快、为什么慢，以及换一张卡后为什么结论可能完全改变。

本课建立后续 CUDA、GPU Memory、Tensor Core、Kernel Fusion、Triton 和分布式优化共同依赖的硬件心智模型。我们不把 GPU 当成“很多个更快的 CPU 核”，而把它看成一台依靠大规模并行、快速线程切换和分层存储隐藏延迟的吞吐机器。

学完后，你应能从一个 Kernel 的线程块大小、寄存器、共享内存、访存步长和数据类型，推断它可能受到哪类硬件资源限制；也能正确区分 Ampere、Ada、Hopper 与 Blackwell，而不是只比较产品宣传中的峰值数字。

## 学习目标

完成本课后，你能够：

1. 解释 Grid、Block、Warp、Thread 与 GPC、SM 的映射关系。
2. 解释 SIMT、分支发散、合并访存、延迟隐藏和指令级并行。
1. 计算寄存器、共享内存、线程数共同限制下的理论 Occupancy。
2. 区分 Occupancy、SM 利用率、Tensor Core 利用率和实际性能。
1. 用 Roofline 判断一个算子更可能受计算还是显存带宽限制。
2. 正确识别 Ampere、Ada Lovelace、Hopper 与 Blackwell 的能力边界。
1. 在 CPU、常见 NVIDIA GPU 和架构专项硬件上完成分层实验。

## 前置知识

- 会运行 Python 脚本并阅读命令行输出。
- 理解字节、带宽、FLOP、延迟和吞吐的基本含义。
- 了解张量与矩阵乘法；不要求已经会写 CUDA C++。
- 可选：安装 PyTorch，以完成 Level 1 GPU 实测。

## 核心直觉：GPU 是吞吐机器，不是低延迟机器

CPU 的典型目标是让少量复杂线程尽快完成，因此投入大量晶体管做分支预测、乱序执行和大缓存。GPU 的典型目标是让海量相似工作在单位时间内完成，因此把更多资源投入算术单元、线程上下文和高带宽数据通路。

一个 GPU 操作访问 HBM 时，单个 Warp 仍可能等待很久。GPU 的办法通常不是把这次访问变得像寄存器一样快，而是让当前 Warp 等待时，Warp Scheduler 立刻发射另一个已经就绪的 Warp：

```
Warp A 发起访存 ─────────────── 等待数据 ── 继续计算
Warp B             计算 ── 访存等待 ─────── 继续
Warp C                    计算 ── 计算 ── 访存
时间 ─────────────────────────────────────────>
```

这叫延迟隐藏。它成立需要三个条件：

- 有足够多的就绪 Warp；
- 指令之间没有过长的依赖链；
- 数据布局没有制造严重的额外内存事务。

所以 GPU 优化的核心问题不是“线程越多越好吗”，而是：

## 从软件层级到硬件层级

### CUDA 执行层级

CUDA 程序通常以四层结构组织工作：

```
Grid
└── Thread Block
    └── Warp（当前 NVIDIA CUDA 架构固定为 32 个线程）
        └── Thread
```

- \*\*Grid\*\*：一次 Kernel launch 产生的全部线程块。
- \*\*Thread Block\*\*：能在共享内存中协作、能用块级同步原语同步的一组线程。
- \*\*Warp\*\*：硬件调度和发射的基本线程组，包含 32 个线程。
- \*\*Thread\*\*：执行同一 Kernel 程序的逻辑线程实例。

线程块被分配到 SM（Streaming Multiprocessor）上。一个块开始执行后，其线程、寄存器和共享内存资源都驻留在同一个 SM；普通 CUDA 编程模型不会把一个线程块拆到多个 SM 上执行。

### 物理层级

不同产品的芯片布局不同，但可以用以下抽象理解：

```
GPU
├── GPC / 更高层处理分区
│   ├── SM
│   │   ├── Warp Scheduler / Dispatch
│   │   ├── 标量与浮点计算管线
│   │   ├── Tensor Core
│   │   ├── Load/Store 与特殊函数单元
│   │   ├── Register File
│   │   └── Shared Memory / L1
│   └── SM ...
├── 共享 L2 Cache
├── Memory Controller
└── GDDR 或 HBM
```

不要把一个 CUDA Thread 机械地对应为一个“CUDA Core”。线程是程序状态；Core 是执行资源。大量线程的指令会在时间上复用有限的执行管线。

### SM 内真正竞争的资源

同一个 SM 上可以同时驻留多个线程块，但会共同竞争：

- 最大驻留线程数与 Warp 数；
- 最大驻留线程块数；
- Register File；
- Shared Memory；
- Warp Scheduler 与执行管线；
- Load/Store 单元和缓存带宽。

任何一项达到上限，新的线程块都无法驻留。正因如此，只看 `threads_per_block` 无法判断 Occupancy。

## SIMT、Warp 与分支发散

### SIMT 不是简单 SIMD

NVIDIA 使用 SIMT（Single Instruction, Multiple Threads）组织线程。一个 Warp 中的线程执行同一 Kernel，但每个线程拥有自己的寄存器状态、线程索引和控制流状态。

当 Warp 中所有线程走同一分支时，执行效率高：

```
if (blockIdx.x < split) {
    // 整个 block 通常走相同路径
}
```

如果同一 Warp 内线程按奇偶走不同分支：

```
if ((threadIdx.x & 1) == 0) {
    path_a();
} else {
    path_b();
}
```

硬件必须分别执行有效线程掩码对应的路径，再汇合控制流。若两条路径代价相近，理想情况下的有效线程工作比例可能接近一半。

可用一个简化的分支效率描述：

$$
\eta_{branch}=\frac{\text{有效线程指令数}}{\text{发射槽覆盖的线程指令数}}
$$

这不是分析工具中所有分支指标的严格定义，但能帮助理解：同一 Warp 内路径越碎，执行槽位浪费通常越多。

### 发散不一定来自 if

以下现象也可能制造控制流或工作量不均衡：

- 不同 Token 经过长度差异很大的序列；
- MoE 中不同专家收到的 Token 数差异过大；
- 稀疏计算中每行非零元素数量不同；
- 线程内循环次数由数据决定；
- 尾块只有少量线程处理有效元素。

优化时要问的不是“有没有 `if` ”，而是“同一 Warp 中的活跃掩码和工作量是否一致”。

## 内存层级与合并访问

从靠近线程到远离线程，可以建立以下近似层级：

| 层级 | 典型作用域 | 特征 |
| --- | --- | --- |
| Register | 单线程 | 最靠近执行管线，容量有限，过多会降低驻留度或发生 spill |
| Shared Memory / L1 | 单 SM / 单 Block 协作 | 软件可控复用、低延迟，但容量有限 |
| L2 Cache | 整个 GPU | SM 间共享，产品容量差异很大 |
| GDDR / HBM | 整个 GPU | 容量与带宽大，但访问延迟高 |
| Host Memory | CPU 侧 | 通常经 PCIe/NVLink-C2C 等链路访问或搬运 |

下一课会深入分析容量、带宽、缓存和数据搬运；本课先掌握最重要的一条：一个 Warp 的全局内存访问会被合并为若干内存事务。

假设每个线程读取一个 4 字节元素：

- 连续 32 个线程读取连续地址时，请求的 128 字节能由少量连续事务覆盖；
- 步长增大时，地址散落到更多内存段；
- 总请求字节不变，实际搬运字节却可能成倍增加。

可定义简化的传输效率：

$$
\eta_{mem}=\frac{\text{线程实际请求字节}}{\text{内存事务搬运字节}}
$$

步长为 1 不保证所有场景都完美，还要考虑地址对齐、元素宽度、缓存命中和具体指令；但它是最可靠的第一直觉。

## Occupancy：资源约束下能驻留多少 Warp

### 理论模型

设一个 SM 的资源上限分别为：

$T_{SM}$ ：最大驻留线程数；

B\\\_{SM}：最大驻留块数；

$R_{SM}$ ：32 位寄存器总数；

$S_{SM}$ ：共享内存总量；

$W_{SM}$ ：最大驻留 Warp 数。

Kernel 每个线程块使用：

$T_b$ 个线程；

- 每线程 $R_t$ 个寄存器；
- 每块 $S_b$ 字节共享内存。

忽略寄存器和共享内存的实际分配粒度时，可驻留块数的简化上界为：

$$
B_{active}=\min\left(B_{SM},\left\lfloor\frac{T_{SM}}{T_b}\right\rfloor,\left\lfloor\frac{R_{SM}}{R_tT_b}\right\rfloor,\left\lfloor\frac{S_{SM}}{S_b}\right\rfloor\right)
$$

其中 $S_b=0$ 时不构成限制。每块 Warp 数为：

$$
W_b=\left\lceil\frac{T_b}{32}\right\rceil
$$

理论 Occupancy 为：

$$
Occupancy=\frac{\min(B_{active}W_b,W_{SM})}{W_{SM}}
$$

实际硬件还存在寄存器和共享内存分配粒度、块级上限、编译器资源分配等细节，因此手算模型用于定位限制因素，最终应以 CUDA Occupancy API、编译器输出与 Nsight Compute 为准。

### 为什么 Occupancy 不是越高越好

高 Occupancy 能提供更多候选 Warp，有助于隐藏延迟，但不能保证：

- 每个 Warp 都已就绪；
- 指令混合能填满目标执行管线；
- 数据访问是合并的；
- Tensor Core 或内存带宽被充分利用；
- 算法做的是有效工作。

一个寄存器复用充分、指令级并行高的计算 Kernel，可能在 50% Occupancy 时已经达到峰值附近。反过来，一个 100% Occupancy 的随机访存 Kernel 仍可能很慢。

因此优化目标不是最大化 Occupancy，而是消除阻止性能继续提升的真实瓶颈。

## 性能模型：计算、带宽、延迟和依赖

### Roofline 上界

设算子完成 F 次浮点运算，从目标内存层级搬运 Q 字节，则算术强度为：

$$
AI=\frac{F}{Q}\quad [FLOP/Byte]
$$

设设备峰值计算吞吐为 $P_{peak}$ ，目标内存层级带宽为 BW，则性能上界为：

$$
P\le\min(P_{peak}, BW\times AI)
$$

转折点为：

$$
AI_{ridge}=\frac{P_{peak}}{BW}
$$

AI：更可能受带宽限制；

$AI>AI_{ridge}$ ：更可能受计算吞吐限制；

- 若实测远低于两条上界，还应检查访存事务、指令依赖、发散、Launch、同步或负载不均衡。

### 端到端时间不是单一 Roofline

对一个真实 Kernel，可先用以下近似拆解：

$$
T\gtrsim T_{launch}+\max(T_{compute},T_{memory})+T_{dependency}+T_{imbalance}+T_{sync}
$$

这不是严格可加的物理公式，因为计算与访存可能重叠，各项也会相互作用；它的价值是提醒我们：峰值 FLOPS 和显存带宽只能解释上界，不能解释所有低效。

## 四代架构：不要只背产品名

Compute Capability（CC）是 CUDA 暴露的硬件能力版本。市场架构名、具体芯片、产品型号和 CC 不是同一个概念。优化代码必须同时检查产品与 CC，不能根据“年份新”直接启用特性。

### 架构映射

| 架构 | 常见产品示例 | 典型 CC | 本课关注点 |
| --- | --- | --- | --- |
| Ampere | RTX 3080/3090、A100 | 8.6 / 8.0 | 第三代 Tensor Core、TF32/BF16、异步全局到共享内存拷贝 |
| Ada Lovelace | RTX 4090、L40/L40S | 8.9 | 第四代 Tensor Core、产品相关的大 L2、FP8 硬件能力与软件栈边界 |
| Hopper | H100/H200 | 9.0 | TMA、Thread Block Cluster、分布式共享内存、FP8 训练 |
| Blackwell 数据中心 | B100/B200/GB200 | 10.x | 第五代 Tensor Core、FP4/FP6、增强 TMA 与集群能力 |
| Blackwell GeForce | RTX 5090 | 12.0 | Blackwell 消费级路径；资源和可移植代码目标不同于 B200 |

关键纠错：

- RTX 4090 是 \*\*Ada Lovelace\*\*，不是 Blackwell。
- RTX 5090 与 B200 都属于 Blackwell，但 CC 12.0 与 CC 10.x 不是同一执行目标。
- A100 与 RTX 3090 都是 Ampere，但 A100 是 CC 8.0，RTX 3090 是 CC 8.6；每 SM 最大 Warp 数、共享内存和块驻留上限并不相同。
- “硬件指令格式存在”不等于任意框架、驱动和算子都能直接获得对应加速。

### 资源差异为何会改变最佳参数

依据当前 CUDA 架构说明，可用以下代表性资源建立直觉：

| CC 示例 | 最大 Warp/SM | 最大线程/SM | Shared Memory/SM | 说明 | | --------------------------------------------- | -- | ---- | ------ | ------------------ | | 8.0 A100 | 64 | 2048 | 164 KB | 数据中心 Ampere | | 8.6 RTX 3080/3090 | 48 | 1536 | 100 KB | 消费级 Ampere | | 8.9 Ada | 48 | 1536 | 100 KB | 产品具体缓存容量另查规格 | | 9.0 Hopper | 64 | 2048 | 228 KB | 支持 TMA 与 Cluster | | 10.0 B200 | 64 | 2048 | 228 KB | 数据中心 Blackwell | | 12.0 RTX 5090 | 48 | 1536 | 128 KB | 单块可用共享内存上限低于 SM 总量 |

这些是编程模型资源上限，不等于性能排名。时钟、SM 数量、显存类型、缓存、功耗限制和具体指令吞吐都依产品而异。

同一个每块使用 64 KB 共享内存的 Kernel，在 100 KB/SM 的设备上通常最多驻留 1 块，在 164 KB 或 228 KB/SM 的设备上可能驻留更多块。于是最佳 Tile 大小、流水级数和每线程寄存器预算都会改变。

### Ampere：现代通用基线

Ampere 的重要工程意义包括：

- Tensor Core 支持 TF32，让许多 FP32 矩阵乘法在允许时走 Tensor Core；
- 支持 BF16/FP16 等训练格式；
- CUDA 异步拷贝可直接把数据从全局内存搬到共享内存，减少中间寄存器占用并支持搬运与计算重叠；
- A100（CC 8.0）与 RTX 30 系列（CC 8.6）的资源上限不同。

对 RTX 3080/3090，课程实验优先使用 FP32、TF32、FP16/BF16 能力检测与通用 CUDA 路径；不假设具备数据中心互联或 FP8 训练栈。

### Ada Lovelace：4090 不是 Blackwell

Ada 的第四代 Tensor Core 和更大的产品级 L2 可改善许多图形与 AI 工作负载。CC 8.9 的 Tensor Core 指令能力包括 FP8，但能否从 PyTorch 或特定推理引擎稳定使用，取决于 GPU 产品、驱动、CUDA、框架、算子实现和数据缩放策略。

因此对 RTX 4090，正确教学方式是：先检测数据类型与算子是否受支持，再测速度与误差；不能只看到“FP8”字样就假设它等价于 H100 上的 Transformer Engine 训练。

### Hopper：数据搬运也成为可编排管线

Hopper 的 TMA（Tensor Memory Accelerator）可以描述 1D 到 5D 张量搬运，在全局内存、共享内存及集群相关存储之间执行异步传输。它减少参与搬运的线程和寄存器开销，并支持更明确的 Warp 专门化：部分 Warp 管搬运，部分 Warp 管计算。

Thread Block Cluster 允许一组线程块进行硬件级协作，并通过 Distributed Shared Memory 访问集群内其他块的共享内存。它不等于“所有 GPU 都能模拟的普通共享内存技巧”；无 H100/H200 时，只能学习调度与流水模型，不能声称完成了等价硬件验证。

### Blackwell：数据中心与消费级要分开看

Blackwell 引入第五代 Tensor Core，并在相关产品上扩展 FP4/FP6 等低精度能力。但 B200/GB200 常用的 CC 10.x 路径与 RTX 5090 的 CC 12.0 路径有实质差别：

- 每 SM 最大驻留 Warp 数不同；
- Shared Memory 总量与单块上限不同；
- 某些架构专用 PTX、Tensor Memory 和集群能力不能跨目标原样复用；
- 软件栈对新数据类型与 Kernel 的支持需要逐项确认。

移植 Blackwell 专项代码时，应把 `sm_100` 、 `sm_100a` 、 `sm_120` 等目标当成不同能力集合管理，提供运行时检测与回退 Kernel。

## 瓶颈分析方法

### 第一步：先确认工作是否正确

性能分析前先做：

1. 与可信实现比较输出。
2. 固定随机种子和输入形状。
1. 明确精度容差与数据类型。
2. 确认没有把异步 GPU 时间误当成完成时间。

错误的结果没有优化价值。

### 第二步：判断瓶颈类别

| 现象 | 可能瓶颈 | 优先检查 |
| --- | --- | --- |
| DRAM 吞吐接近平台可达上限，计算管线偏低 | 显存带宽 | 算术强度、合并访存、数据复用 |
| Tensor Core 活跃度低，普通浮点管线高 | 未走 Tensor Core | 形状、dtype、布局、框架开关 |
| 两者都低，Warp 经常无可发射指令 | 延迟或依赖 | Occupancy、长依赖链、随机访存 |
| Occupancy 受寄存器限制 | 每线程状态过多 | 编译器寄存器报告、spill、本地内存 |
| Occupancy 受共享内存限制 | Tile 或流水级过大 | 每块共享内存、双缓冲级数 |
| 分支效率低 | Warp 发散 | 活跃掩码、数据分组、尾部处理 |
| Kernel 很短且调用极多 | Launch/调度 | Fusion、CUDA Graph、批量化 |
| 单卡快，多卡慢 | 通信或负载失衡 | 拓扑、NCCL、重叠、分片均衡 |

### 第三步：建立可证伪假设

不要说“GPU 没跑满”，要说：

- “每块 96 KB 共享内存使 CC 8.6 上只能驻留一个块，导致内存延迟难以隐藏。”
- “Warp 以 $stride=16$ 读取 4 字节元素，请求字节与事务字节比例很低。”
- “矩阵维度与 dtype 没触发目标 Tensor Core 路径。”

然后一次只改变一个变量，并记录正确性、延迟、吞吐和关键硬件计数器。

### 第四步：选择工具

- `nvidia-smi` ：产品、驱动、功耗、显存和粗粒度利用率。
- PyTorch Profiler：框架算子、CPU/GPU 时间和形状。
- Nsight Systems：端到端时间线、Kernel 间空洞、CPU/GPU 与通信重叠。
- Nsight Compute：单 Kernel 的 Warp、Occupancy、内存事务和管线计数器。
- `nvcc --ptxas-options=-v` ：寄存器、共享内存、spill 等编译资源报告。

`nvidia-smi` 的 GPU Utilization 只表示采样窗口内是否有 Kernel 活动，不能替代 Kernel 级诊断。

## 完整可运行实验：资源、发散、访存与 Roofline

实验脚本分三级：

- \*\*Level 0\*\*：仅依赖 Python 标准库，任何机器都能运行资源上限、Warp 发散、合并访存和 Roofline 模型。
- \*\*Level 1\*\*：可选 PyTorch，自动选择 CPU/CUDA，测量逐元素加法、连续/跨步归约和矩阵乘法。
- \*\*Level 2\*\*：按 CC 检测架构能力，再决定是否进入 TMA、Cluster、FP8/FP4 等专项实验。

### 环境准备

纯 CPU 模型无需安装依赖：

```
python3 --version
```

可选 PyTorch 环境：

```
python3 -m venv .venv-gpu-arch
source .venv-gpu-arch/bin/activate
python -m pip install --upgrade pip
# 从 https://pytorch.org/get-started/locally/ 选择与本机驱动匹配的稳定版命令
python -m pip install torch
```

### 保存脚本

将以下内容保存为 `gpu_arch_lab.py` ：

```
#!/usr/bin/env python3
import argparse
import json
import math
import random
import statistics
import time

ARCH = {
    "80": {"name": "Ampere CC 8.0 / A100", "threads": 2048,
           "warps": 64, "blocks": 32, "registers": 65536,
           "shared_kb": 164},
    "86": {"name": "Ampere CC 8.6 / RTX 3080/3090", "threads": 1536,
           "warps": 48, "blocks": 16, "registers": 65536,
           "shared_kb": 100},
    "89": {"name": "Ada CC 8.9 / RTX 4090, L40/L40S", "threads": 1536,
           "warps": 48, "blocks": 24, "registers": 65536,
           "shared_kb": 100},
    "90": {"name": "Hopper CC 9.0 / H100/H200", "threads": 2048,
           "warps": 64, "blocks": 32, "registers": 65536,
           "shared_kb": 228},
    "100": {"name": "Blackwell CC 10.0 / B200", "threads": 2048,
            "warps": 64, "blocks": 32, "registers": 65536,
            "shared_kb": 228},
    "120": {"name": "Blackwell CC 12.0 / RTX 5090", "threads": 1536,
            "warps": 48, "blocks": 32, "registers": 65536,
            "shared_kb": 128},
}

def architecture_from_cc(major, minor):
    code = major * 10 + minor
    if code == 80:
        return "Ampere CC 8.0"
    if 80 < code < 89:
        return "Ampere CC 8.x"
    if code == 89:
        return "Ada Lovelace CC 8.9"
    if 90 <= code < 100:
        return "Hopper CC 9.x"
    if 100 <= code < 120:
        return "Blackwell Data Center CC 10.x/11.x"
    if 120 <= code < 130:
        return "Blackwell CC 12.x"
    return f"未在本实验表中建模的 CC {major}.{minor}"

def occupancy(profile, threads_per_block, regs_per_thread, shared_kb):
    warps_per_block = math.ceil(threads_per_block / 32)
    by_threads = profile["threads"] // threads_per_block
    by_warps = profile["warps"] // warps_per_block
    regs_per_block = regs_per_thread * threads_per_block
    by_regs = profile["registers"] // regs_per_block if regs_per_block else 10**9
    shared_bytes = shared_kb * 1024
    by_shared = ((profile["shared_kb"] * 1024) // shared_bytes
                 if shared_bytes else 10**9)
    limits = {
        "hardware_blocks": profile["blocks"],
        "threads": by_threads,
        "warps": by_warps,
        "registers": by_regs,
        "shared_memory": by_shared,
    }
    active_blocks = max(0, min(limits.values()))
    active_warps = min(profile["warps"], active_blocks * warps_per_block)
    limiting = [k for k, v in limits.items() if v == active_blocks]
    return {
        "threads_per_block": threads_per_block,
        "regs_per_thread": regs_per_thread,
        "shared_kb_per_block": shared_kb,
        "active_blocks_per_sm": active_blocks,
        "active_warps_per_sm": active_warps,
        "occupancy": round(active_warps / profile["warps"], 4),
        "limiting_resources": limiting,
    }

def divergence_model(prob_true, warps=20000, seed=7):
    rng = random.Random(seed)
    useful = 0
    issued_slots = 0
    divergent = 0
    for _ in range(warps):
        true_lanes = sum(rng.random() < prob_true for _ in range(32))
        false_lanes = 32 - true_lanes
        useful += 32
        if true_lanes == 0 or false_lanes == 0:
            issued_slots += 32
        else:
            divergent += 1
            # 简化模型：两条等成本路径分别发射一次，每次覆盖 32 个槽位。
            issued_slots += 64
    return {
        "p_true": prob_true,
        "divergent_warp_ratio": round(divergent / warps, 4),
        "simplified_branch_efficiency": round(useful / issued_slots, 4),
    }

def coalescing_model(stride, element_bytes=4, segment_bytes=32):
    addresses = [lane * stride * element_bytes for lane in range(32)]
    segments = {addr // segment_bytes for addr in addresses}
    requested = 32 * element_bytes
    transferred = len(segments) * segment_bytes
    return {
        "stride": stride,
        "segments": len(segments),
        "requested_bytes": requested,
        "modeled_transferred_bytes": transferred,
        "modeled_efficiency": round(requested / transferred, 4),
    }

def roofline_model(peak_tflops, bandwidth_gbs):
    rows = []
    for ai in [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128]:
        bandwidth_roof = bandwidth_gbs * ai / 1000.0
        rows.append({
            "arithmetic_intensity_flop_per_byte": ai,
            "roof_tflops": round(min(peak_tflops, bandwidth_roof), 4),
            "bound": "memory" if bandwidth_roof < peak_tflops else "compute",
        })
    return {
        "peak_tflops": peak_tflops,
        "bandwidth_gbs": bandwidth_gbs,
        "ridge_point_flop_per_byte": round(peak_tflops * 1000 / bandwidth_gbs, 4),
        "rows": rows,
    }

def bench_torch(device_name, elements, matrix_n, repeat):
    try:
        import torch
    except ImportError:
        return {"available": False, "reason": "未安装 PyTorch"}

    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    info = {
        "available": True,
        "torch_version": torch.__version__,
        "device": str(device),
    }
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        prop = torch.cuda.get_device_properties(idx)
        major, minor = torch.cuda.get_device_capability(idx)
        info.update({
            "gpu_name": prop.name,
            "compute_capability": f"{major}.{minor}",
            "architecture": architecture_from_cc(major, minor),
            "memory_gib": round(prop.total_memory / 2**30, 2),
            "cuda_runtime": torch.version.cuda,
        })

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def median_ms(fn, warmup=5):
        for _ in range(warmup):
            fn()
        sync()
        samples = []
        for _ in range(repeat):
            start = time.perf_counter()
            fn()
            sync()
            samples.append((time.perf_counter() - start) * 1000)
        return statistics.median(samples)

    # 每个用例先分配，再计时，避免把 allocator 时间算进 Kernel。
    x = torch.randn(elements, device=device, dtype=torch.float32)
    y = torch.randn(elements, device=device, dtype=torch.float32)
    add_ms = median_ms(lambda: torch.add(x, y))
    add_gbs = (elements * 4 * 3) / (add_ms / 1000) / 1e9

    # 两个视图都归约 elements 个元素；跨步视图来自更大的底层张量。
    stride = 8
    base = torch.randn(elements * stride, device=device, dtype=torch.float32)
    contiguous = base[:elements]
    strided = base[::stride]
    contiguous_ms = median_ms(lambda: contiguous.sum())
    strided_ms = median_ms(lambda: strided.sum())

    a = torch.randn((matrix_n, matrix_n), device=device, dtype=torch.float32)
    b = torch.randn((matrix_n, matrix_n), device=device, dtype=torch.float32)
    matmul_ms = median_ms(lambda: torch.mm(a, b), warmup=3)
    matmul_tflops = (2 * matrix_n**3) / (matmul_ms / 1000) / 1e12

    info["measurements"] = {
        "elementwise_add": {
            "elements": elements,
            "median_ms": round(add_ms, 4),
            "estimated_gbs": round(add_gbs, 2),
        },
        "sum_layout": {
            "elements_each": elements,
            "contiguous_ms": round(contiguous_ms, 4),
            "stride": stride,
            "strided_ms": round(strided_ms, 4),
            "slowdown": round(strided_ms / contiguous_ms, 2),
        },
        "fp32_matmul": {
            "n": matrix_n,
            "median_ms": round(matmul_ms, 4),
            "estimated_tflops": round(matmul_tflops, 3),
        },
    }
    return info

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cc", choices=ARCH, default="86")
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--regs", type=int, default=64)
    parser.add_argument("--shared-kb", type=int, default=32)
    parser.add_argument("--peak-tflops", type=float, default=30.0)
    parser.add_argument("--bandwidth-gbs", type=float, default=700.0)
    parser.add_argument("--torch", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--elements", type=int, default=1_000_000)
    parser.add_argument("--matrix-n", type=int, default=1024)
    parser.add_argument("--repeat", type=int, default=10)
    args = parser.parse_args()

    profile = ARCH[args.cc]
    result = {
        "architecture_profile": profile,
        "occupancy": occupancy(
            profile, args.threads, args.regs, args.shared_kb
        ),
        "occupancy_sweep": [
            occupancy(profile, t, args.regs, args.shared_kb)
            for t in (64, 128, 256, 512, 1024)
            if t <= 1024
        ],
        "divergence": [divergence_model(p) for p in (0.0, 0.1, 0.5, 0.9, 1.0)],
        "coalescing": [coalescing_model(s) for s in (1, 2, 4, 8, 16)],
        "roofline": roofline_model(args.peak_tflops, args.bandwidth_gbs),
    }
    if args.torch:
        result["torch_benchmark"] = bench_torch(
            args.device, args.elements, args.matrix_n, args.repeat
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
```

### Level 0：通用 CPU 回退实验

运行默认的 CC 8.6 模型：

```
python3 gpu_arch_lab.py > result_cc86.json
python3 -m json.tool result_cc86.json | less
```

比较同一个 Kernel 资源配置在 A100、RTX 3090、H100、B200 和 RTX 5090 抽象模型上的差异：

```
for cc in 80 86 90 100 120; do
  python3 gpu_arch_lab.py \
    --cc "$cc" --threads 256 --regs 64 --shared-kb 64 \
    > "result_cc${cc}.json"
done
```

构造寄存器压力与共享内存压力：

```
# 寄存器增加
python3 gpu_arch_lab.py --cc 86 --threads 256 --regs 128 --shared-kb 0

# 每块共享内存增加
python3 gpu_arch_lab.py --cc 86 --threads 256 --regs 32 --shared-kb 64
```

### 预期现象

1. 当分支概率为 0 或 1 时，没有 Warp 发散，简化分支效率为 1。
2. 当每个线程独立地以 0.5 概率选分支时，几乎所有 Warp 都发散，等成本双路径模型的效率接近 0.5。
1. 4 字节元素的读取步长从 1 增至 16 时，触及的 32 字节内存段显著增加，简化传输效率下降。
2. 对 CC 8.6，每块使用 64 KB 共享内存时，100 KB/SM 通常只允许一个这样的块驻留；该配置的理论 Occupancy 会较低。
1. 相同资源配置在 CC 8.0、9.0 或 10.0 的 64 Warp/SM 与更大共享内存模型上可能得到不同结果。

### 结果分析

Level 0 不是 GPU 周期精确模拟器。它有意忽略了分配粒度、缓存、Bank Conflict、指令吞吐和调度细节，用来训练“资源约束先验”：

- 若模型显示寄存器是唯一限制项，下一步应查看真实编译器寄存器报告，并检查 spill。
- 若共享内存是限制项，应测试较小 Tile、减少流水级或用寄存器/缓存重构复用。
- 若内存段数量随步长快速增加，应优先调整数据布局和线程到元素的映射。
- 若 Roofline 显示算术强度远低于转折点，单纯增加数学指令峰值通常不是第一优化方向。

### Level 1：常见 NVIDIA GPU 与 CPU 实测

先检测环境：

```
nvidia-smi --query-gpu=name,driver_version,memory.total \
  --format=csv,noheader 2>/dev/null || true

python3 - <<'PY'
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda runtime:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
        print("cc:", torch.cuda.get_device_capability(0))
except ImportError:
    print("PyTorch not installed")
PY
```

自动使用 CUDA；无 CUDA 时回退 CPU：

```
python3 gpu_arch_lab.py --torch --device auto \
  --elements 1000000 --matrix-n 1024 --repeat 10 \
  > result_torch.json
```

显存充足时可扩大问题规模，减少 Launch 与计时噪声占比：

```
python3 gpu_arch_lab.py --torch --device cuda \
  --elements 8000000 --matrix-n 4096 --repeat 20 \
  > result_cuda_large.json
```

### 预期现象

- 脚本打印 GPU 名称、Compute Capability、显存、CUDA Runtime 与架构判断，不硬编码设备型号。
- 跨步视图与连续视图都归约相同数量的 FP32 元素，但跨步访问通常更慢、缓存与内存事务效率更差。
- 逐元素加法算术强度很低，更接近带宽型任务。
- 大矩阵乘法有较高数据复用，更可能利用 Tensor Core 或浮点计算管线；具体路径取决于 dtype、形状、框架设置和 GPU。
- CPU 回退结果用于比较布局和规模趋势，不能用来推断 GPU Warp 或 Tensor Core 的真实效率。

### 结果解释边界

脚本中逐元素加法的 `estimated_gbs` 按两次读、一次写估算为 `3 × N × 4` 字节，没有纳入缓存、写分配和内部流量。矩阵乘法的 TFLOPS 按 2N^3 估算。两者都是工作量除以墙钟时间，不是硬件计数器读数。

PyTorch 的 GPU 操作是异步的，脚本在计时前后同步设备。删除同步会得到虚假的微秒级结果，这是最常见的基准错误之一。

### Level 2：架构专项实验与能力检测

先查询 Compute Capability：

```
python3 - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("未发现 CUDA GPU：仅完成 Level 0/CPU 回退")
name = torch.cuda.get_device_name(0)
cc = torch.cuda.get_device_capability(0)
print({"name": name, "compute_capability": cc})
PY
```

按能力选择实验：

| 能力 | 最低硬件路径 | 无对应硬件时的处理 |
| --- | --- | --- |
| TF32 Tensor Core | Ampere CC 8.0+ | 用 FP32 完成正确性与测量流程，不宣称等价吞吐 |
| 异步 global→shared copy | Ampere CC 8.0+ | 用普通 load/store 学习双缓冲概念 |
| FP8 Tensor Core 指令 | Ada CC 8.9+/Hopper；软件支持另查 | 用 FP16/BF16 学习缩放和误差，不宣称 FP8 实测 |
| TMA、Thread Block Cluster | Hopper CC 9.0+ | 只完成流水和 Cluster 资源模型 |
| FP4/FP6 Tensor Core | Blackwell 相关目标 | 用 INT4/FP16 做格式对照，不宣称等价计算 |
| B200 专项 Kernel | 对应 CC 10.x 目标 | RTX 5090 CC 12.0 也必须走独立回退路径 |

编译 CUDA 程序时打印资源占用：

```
nvcc -O3 --ptxas-options=-v kernel.cu -o kernel
```

常见输出中的 `registers` 、 `smem` 、 `cmem` 和 spill 信息应与理论模型交叉验证。若已安装 Nsight Compute，可对自己的 Kernel 收集基础集合：

```
ncu --set basic --target-processes all ./kernel
```

进一步查看 Occupancy、Warp 状态和内存工作负载时，优先从图形界面的 Speed of Light、Occupancy、Warp State Statistics、Memory Workload Analysis 分区入手。指标名称会随 Nsight Compute 版本与芯片变化，不建议把一长串底层 metric 名硬编码进通用课程脚本。

### Level 2 的不可等价模拟边界

- 在 RTX 3080/3090 上模拟 FP8 数值范围，不等于运行 FP8 Tensor Core。
- 在 RTX 4090 上看到 CC 8.9，不等于自动拥有 H100 的 Transformer Engine 训练体验。
- 在双 RTX 3090 上用 PCIe 通信，不能验证 H100 NVLink/NVSwitch 拓扑。
- 在 CPU 上实现 Tile 搬运，只能理解算法，不能验证 TMA 的硬件开销。
- RTX 5090 是 Blackwell，但不能直接替代 B200/GB200 的数据中心功能与软件目标。

## 优化前后对照

| 维度 | 优化前 | 优化后 | 验证方法 |
| --- | --- | --- | --- |
| 线程块配置 | 固定复制 1024 threads/block | 根据寄存器、共享内存和设备资源扫参 | Occupancy API + 实测延迟 |
| Warp 控制流 | 同 Warp 随机走不同重路径 | 按数据或任务分组，使 Warp 工作更一致 | Warp State / branch 指标 |
| 全局访存 | lane 访问大步长地址 | lane 访问连续或可合并地址 | Memory Workload Analysis |
| 数据复用 | 每次从显存重读 | Shared Memory、Register 或 Cache 复用 | DRAM 字节与算术强度 |
| 搬运计算重叠 | 加载后等待，再计算 | 双缓冲或支持架构上的异步搬运 | Timeline 与 stall 原因 |
| 架构适配 | 把 4090 当 Blackwell，所有卡用同一 Kernel | 依据 CC 和软件栈选择主路径与回退 | 能力检测 + 正确性测试 |
| 优化目标 | 只追求 100% Occupancy | 追求端到端 Goodput 与目标 SLO | 端到端 Benchmark |

应特别警惕“降低寄存器数后 Occupancy 提升，但性能下降”：编译器可能把值 spill 到本地内存，或者减少了有用的指令级并行。任何资源调优都必须以实测为裁判。

## 常见错误与排查

### 错误 1：把 CUDA Core 数当成可同时独立运行的线程数

\*\*原因\*\*：混淆逻辑线程、Warp 和执行单元。

\*\*排查\*\*：从 Grid→Block→Warp→Thread 重新画映射；用 Nsight Compute 观察 Warp 发射和管线利用率。

### 错误 2：看到 100% GPU Utilization 就认为没有优化空间

\*\*原因\*\*：粗粒度采样只说明一段时间内有 Kernel 活动。

\*\*排查\*\*：查看单 Kernel 时间、DRAM/Tensor/Core 管线、Warp Stall 和端到端时间线。

### 错误 3：盲目把线程块改成 1024

\*\*原因\*\*：忽略每块寄存器和共享内存总量，可能使每 SM 只能驻留一个块。

\*\*排查\*\*：扫 128、256、512 等配置；记录资源、Occupancy 与真实耗时。

### 错误 4：为提高 Occupancy 强制限制寄存器

\*\*原因\*\*：寄存器不足可能产生 spill，数据落入本地内存。

\*\*排查\*\*：检查 `ptxas` 输出和 local memory 流量；比较限制前后的 Kernel 时间。

### 错误 5：把所有非连续张量都归因于分支发散

\*\*原因\*\*：非连续布局通常首先影响访存事务和缓存，不一定改变控制流。

\*\*排查\*\*：分别检查地址模式、活跃线程掩码和指令路径。

### 错误 6：GPU 计时没有同步

\*\*症状\*\*：大矩阵乘法只需几微秒，或结果波动异常。

\*\*排查\*\*：使用 CUDA Event，或在墙钟计时边界调用 `torch.cuda.synchronize()` ；预热后取中位数。

### 错误 7：把架构名当作完整能力判断

\*\*原因\*\*：同代产品可能使用不同 CC、资源规模和软件功能。

\*\*排查\*\*：同时记录 GPU 名称、CC、驱动、CUDA、框架版本、dtype 与算子实现。

### 错误 8：运行 Level 1 时内存不足

\*\*原因\*\*：跨步归约会分配 `elements × stride` 的底层张量，矩阵乘法也需要多块矩阵存储。

\*\*排查\*\*：先把 `--elements` 降为 `250000` 、 `--matrix-n` 降为 `512` ；确认没有其他进程占用显存。

## 面试题与答案

### 1\. Warp 是什么？为什么通常是 32 个线程？

Warp 是 NVIDIA GPU 调度与指令发射的基本线程组。当前 CUDA 支持的 NVIDIA 架构中 Warp Size 为 32。程序不应把线程当成完全独立的 32 个 CPU 线程；同 Warp 控制流和访存模式会共同影响效率。

### 2\. Occupancy 与 GPU Utilization 有什么区别？

Occupancy 是一个 SM 上实际驻留 Warp 数相对硬件最大 Warp 数的比例；GPU Utilization 通常是采样窗口内 GPU 是否在执行工作的粗粒度比例。前者是资源驻留指标，后者是活动指标，都不直接等于应用性能。

### 3\. 高 Occupancy 为什么不保证高性能？

驻留 Warp 可能因相同内存依赖而全部等待，也可能存在糟糕的内存事务、分支发散或低效指令。部分计算密集 Kernel 在较低 Occupancy 下依靠寄存器复用和指令级并行即可达到高吞吐。

### 4\. 什么会限制一个 SM 的活跃线程块数？

硬件最大块数、最大线程/ Warp 数、寄存器总量、共享内存总量以及架构规定的每块资源上限。实际分配还受粒度和编译结果影响。

### 5\. 什么是合并访存？

一个 Warp 的线程访问全局内存时，硬件把地址请求组合为内存事务。地址越连续、对齐越合理，通常需要的事务越少；地址跨度越大，搬运了却未被线程使用的字节越多。

### 6\. 分支发散怎样产生性能损失？

同一 Warp 的线程走不同路径时，硬件要用不同活跃掩码执行各路径，未参与当前路径的 lane 暂时闲置。损失大小取决于路径代价、发散模式和编译后的控制流。

### 7\. RTX 4090 属于哪一代？RTX 5090 呢？

RTX 4090 属于 Ada Lovelace，典型 CC 8.9；RTX 5090 属于 Blackwell，典型 CC 12.0。RTX 5090 与 B200 虽同属 Blackwell，但不能视为完全相同的 CUDA 目标。

### 8\. Ampere 的 A100 与 RTX 3090 为什么不能只用一个资源表？

A100 通常是 CC 8.0，RTX 3090 是 CC 8.6。两者每 SM 最大驻留 Warp、共享内存和块数量等资源上限不同，最佳 Tile、线程块和流水配置可能不同。

### 9\. TMA 解决什么问题？

TMA 让 Hopper 及后续相关架构可以用硬件描述和执行多维张量异步搬运，降低参与搬运的线程与寄存器开销，并帮助构建搬运和计算重叠的 Warp 专门化流水线。

### 10\. 如何判断一个 Kernel 是计算受限还是带宽受限？

先计算目标内存层级的算术强度，用 Roofline 比较计算上界与 `带宽 × 算术强度` 上界；再结合实测管线、DRAM 吞吐、缓存和 Warp Stall 指标验证。只靠一个利用率数字不足以下结论。

### 11\. 降低寄存器使用一定会更快吗？

不一定。它可能提高 Occupancy，也可能导致 spill、增加指令或破坏数据复用。必须同时检查正确性、spill、Occupancy 和 Kernel 耗时。

### 12\. 为什么消费级 Blackwell 不能直接替代 B200 实验？

两者产品资源、显存系统、互联、CC 目标和部分架构特性不同。RTX 5090 可验证部分 Blackwell 通用能力，但 B200/GB200 专项 Kernel 和集群能力需要对应硬件与软件栈。

## 课后练习

1. 用 Level 0 脚本比较 CC 8.0 与 8.6 在 `256 threads、64 registers/thread、48 KB shared/block` 下的限制资源。
2. 固定 CC 8.6 和 256 threads/block，将寄存器从 16 扫到 160，画出理论 Occupancy 曲线。
1. 固定寄存器为 32，将共享内存从 0 扫到 96 KB，解释为什么曲线呈阶梯变化。
2. 修改合并访存模型，加入 16 字节元素和起始地址偏移，观察事务数量变化。
1. 把发散模型改为 Warp 内相邻连续线程分组，而不是每线程独立随机，比较发散 Warp 比例。
2. 在 CPU 与 CUDA 上分别运行 Level 1，解释为什么 CPU 结果不能证明 Warp 合并访存。
1. 在一张可用 GPU 上记录名称、CC、驱动、CUDA 和 PyTorch 版本，并判断它属于哪代架构。
2. 若有 Nsight Compute，选择一个矩阵乘法或逐元素 Kernel，写出一个可证伪的瓶颈假设并验证。
1. 解释为什么将每块共享内存从 48 KB 增至 64 KB，可能在某些 GPU 上突然减少驻留块数。
2. 为一个同时支持 RTX 3090、RTX 4090、H100 和 RTX 5090 的程序设计能力检测与回退表。

## Checklist

### 概念

- 能画出 Grid、Block、Warp、Thread 与 SM 的关系。
- 知道 Warp Size 为 32，但线程不等于物理 CUDA Core。
- 能解释 SIMT、分支发散和活跃掩码。
- 能解释合并访存为何减少无效传输。
- 能区分延迟、带宽、吞吐和算术强度。

### 资源与模型

- 能根据线程、寄存器、共享内存估算 Occupancy。
- 知道理论 Occupancy 模型忽略了实际分配粒度。
- 不把 100% Occupancy 当作最终优化目标。
- 能用 Roofline 形成计算受限或带宽受限假设。
- 会用正确性和实测数据验证模型。

### 架构边界

- Ampere：RTX 3080/3090、A100。
- Ada Lovelace：RTX 4090、L40/L40S。
- Hopper：H100/H200。
- Blackwell：RTX 5090、B100/B200/GB200。
- 知道 CC 8.0 与 8.6、CC 10.x 与 12.0 不能混为一谈。
- 不把 CPU 模拟、FP16 或 INT4 实验声称为 TMA、FP8 或 FP4 的等价验证。

### 实验

- 已运行 Level 0 脚本并保存 JSON。
- 已比较至少两种 CC 的资源约束结果。
- 有 PyTorch 时，已完成自动设备检测与 Level 1 实测。
- GPU 计时包含预热、同步、多次重复与中位数。
- 性能结果只描述本次环境，不推广为所有 GPU 的固定结论。