---
title: "第6课：GPU Memory 体系优化"
source: "https://everythingai.top/learn/ai-systems-performance-engineering-text/lesson-06"
author:
published:
created: 2026-09-26
description: "面向 AI 工程学习者的专业课程学习空间。"
tags:
  - "clippings"
---
**导航**

本课系统梳理 GPU 显存层次与数据搬运路径，掌握容量、带宽和访问延迟的核心优化方法。

## 课程定位

GPU 有很高的峰值计算能力，但算术单元只有拿到数据才能工作。很多 AI Kernel 的真正瓶颈不是“算得不够快”，而是数据从错误的位置、以错误的布局、在错误的时间到达计算单元。

本课把上一课的 GPU 架构模型推进到 Memory 体系：从线程私有寄存器、SM 内共享内存和 L1，到全 GPU 共享的 L2，再到 GDDR/HBM、主机内存和存储。我们会建立三个可以直接用于工程诊断的指标：有效带宽、内存访问效率和数据复用率；随后用 CPU 回退、通用 PyTorch 与架构专项三层实验验证。

## 学习目标

完成本课后，你能够：

1. 解释 Register、Local Memory、Shared Memory、L1、L2、GDDR/HBM 与 Host Memory 的作用域和性能关系。
2. 用有效带宽、算术强度和事务效率判断 Memory Bottleneck。
1. 解释连续访问、跨步访问、对齐、AoS/SoA 布局对内存事务的影响。
2. 识别 Shared Memory Bank Conflict，并用 padding 等方法消除冲突。
1. 解释寄存器压力、spill、Occupancy 与数据复用之间的权衡。
2. 正确使用 Pinned Memory、 `non_blocking=True` 、Stream 与双缓冲。
1. 理解 PyTorch `memory_allocated` 、 `memory_reserved` 、缓存分配器、碎片与 OOM 的区别。
2. 为 Ampere、Ada、Hopper、Blackwell 选择能力检测和回退路径。

## 前置知识

- 已完成第 5 课，理解 SM、Warp、Occupancy 和 Roofline。
- 理解 Byte、GB/s、FLOP、Latency 与 Throughput。
- 会运行 Python；PyTorch 与 CUDA GPU 为可选环境。
- 不要求会写 CUDA C++，但应能阅读简单的索引表达式。

## 核心直觉：优化 Memory 不是“少用显存”这么简单

GPU Memory 优化至少包含四个彼此不同的问题：

1. \*\*容量\*\*：模型、激活、优化器状态和临时 Workspace 能否放下？
2. \*\*带宽\*\*：单位时间能搬运多少有效数据？
1. \*\*延迟\*\*：一次依赖性访问要等待多久，能否被其他 Warp 隐藏？
2. \*\*流量\*\*：算法到底需要从目标层级搬多少字节？

一段代码显存占用更少，不代表它更快；一段代码达到很高 DRAM 带宽，也不代表它最优。如果它重复读取本可复用的数据，那么“带宽跑满”可能只是高效地搬运了大量无效流量。

可以用一句话概括本课：

## GPU Memory 层级

### 统一心智模型

```
每个线程
  Register
     │  寄存器不足时可能 spill
     ▼
  Local Memory（逻辑上线程私有，物理上位于设备内存）

每个 SM / Thread Block 协作
  Shared Memory ↔ L1 Cache
             │
             ▼
全 GPU 共享
  L2 Cache
     │
     ▼
设备显存
  GDDR / HBM
     │
     ▼
系统链路
  PCIe / NVLink-C2C 等
     │
     ▼
Host DRAM（pageable / pinned）
     │
     ▼
SSD / 网络存储
```

“越靠上越快”是有用直觉，但不能机械理解：

- Register 延迟低，但容量小，使用过多会降低驻留 Warp 数或造成 spill。
- Shared Memory 可编程、适合 Block 内复用，但会消耗每 SM 的有限资源，还可能出现 Bank Conflict。
- L1/L2 由硬件管理，命中率取决于工作集、访问模式和与其他数据的竞争。
- HBM/GDDR 提供大容量和高带宽，但比片上存储有更高延迟。
- Host Memory 容量更大，但离离散 GPU 更远，跨链路搬运通常远慢于设备内访问。

### Register 与 Local Memory

寄存器是线程最直接使用的存储。循环累加器、索引、地址和中间值通常放在寄存器中。

“Local Memory”这个名字容易误导：它的作用域对单线程是 local，但物理存储通常在片外设备内存中，并经过缓存层级。以下情况可能让编译器使用 Local Memory：

- 每线程寄存器需求过大，发生 register spill；
- 线程私有数组无法静态索引到寄存器；
- 大型线程局部对象；
- 某些动态索引或编译器决策。

因此，减少寄存器并不总是优化。如果从 96 个寄存器强行压到 64 个，Occupancy 可能提高，但 spill 带来的设备内存访问可能使 Kernel 更慢。

### Shared Memory 与 L1

Shared Memory 位于 SM 上，由 Thread Block 中的线程显式管理。典型用途：

- 将全局内存 Tile 搬入片上后重复使用；
- 把全局内存中的非连续访问重排为连续事务；
- 在线程之间交换部分结果；
- 构建归约、扫描、矩阵乘法和卷积 Tile；
- 构建多级异步流水。

Shared Memory 与 L1 的容量组织会随架构变化；写通用代码时应查询设备属性，而不是假设固定为某个 KB 数值。

### L2 与设备显存

L2 对所有 SM 可见，是跨 Block 数据复用、原子操作和 DRAM 流量之间的重要缓冲层。L2 容量是具体产品属性，不能只按架构名推断。

设备显存可能是 GDDR，也可能是 HBM：

- RTX 3080/3090、RTX 4090、RTX 5090 等消费级卡通常使用 GDDR。
- A100、H100/H200、B100/B200 等数据中心产品通常使用 HBM。

HBM 的高总线并行度适合大带宽工作负载，但“HBM 一定比所有 GDDR 产品更快”仍不是严谨结论；必须比较具体产品、数据类型、功耗状态和可达带宽。

### Host Memory：pageable、pinned 与 unified

普通 pageable 内存可能被操作系统换出或移动。Pinned Memory 被锁定在物理内存，便于 DMA，在离散 GPU 的 Host↔Device 大块传输中通常能获得更好吞吐，也是可靠异步传输的重要条件。

但 Pinned Memory 是稀缺系统资源：

- 分配和锁页本身有成本；
- 过量 pin 会影响整个系统；
- 将普通 Tensor 临时 `.pin_memory()` 再立即拷贝，可能比直接 `.to()` 更慢；
- 更适合在 DataLoader 或复用缓冲池中预先准备，而不是每次动态创建。

Unified Memory 提供统一地址与按需迁移能力，适合简化编程或让超出显存的数据仍能运行。但页面故障和迁移会产生不可预测延迟。对稳定低延迟和高吞吐工作负载，它通常是容量安全网，而不是自动性能优化。

## 性能模型

### 有效带宽

若一次操作从内存读取 R 字节、写入 W 字节，耗时 t 秒，则按算法请求量估算的有效带宽为：

$$
BW_{effective}=\frac{R+W}{t}
$$

以 FP32 向量加法 $C=A+B$ 为例，忽略缓存和额外事务：

$$
R=2\times N\times4,\quad W=N\times4
$$

$$
BW_{effective}=\frac{12N}{t}
$$

它衡量完成有效工作所处理的逻辑字节，不等于硬件计数器看到的真实 DRAM 字节。

### 请求带宽、实际带宽与效率

设线程实际需要 $Q_{requested}$ 字节，但由于对齐、跨步、事务粒度等原因，内存系统实际搬运 $Q_{actual}$ 字节：

$$
\eta_{memory}=\frac{Q_{requested}}{Q_{actual}}
$$

例如，一个 Warp 读取 32 个连续 FP32，共请求 128 字节。在 Compute Capability 6.0+ 的常见全局内存合并模型中，这通常可由 4 个 32 字节事务覆盖。若每个 lane 间隔 32 字节，可能触及 32 个不同的 32 字节段：仍只使用 128 字节，却可能搬运 1024 字节，简化效率为 12.5%。

真实结果还会受到缓存复用、地址对齐、写策略和硬件代际影响，因此事务模型用于解释原因，硬件计数器用于给出最终证据。

### 数据复用与算术强度

设同一批数据从 DRAM 读一次后，在 Shared Memory 或 Register 中使用 K 次。理想情况下，DRAM 字节数不会随 K 成比例增加，于是算术强度上升：

$$
AI=\frac{FLOPs}{Bytes_{target\ memory\ level}}
$$

必须明确“相对哪个层级”计算 AI。一个 Kernel 对 HBM 有高复用，仍可能受 Shared Memory 带宽或 Register 依赖限制。

### 容量模型

训练时的显存不只包含参数。粗略模型为：

$$
M_{total}\approx M_{param}+M_{grad}+M_{optimizer}+M_{activation}+M_{workspace}+M_{fragmentation}
$$

不同精度、优化器、Activation Checkpoint、并行策略和编译器 Workspace 会改变各项。推理还应加入 KV Cache：

$$
M_{inference}\approx M_{weights}+M_{KV}+M_{workspace}+M_{runtime}
$$

这解释了为什么“模型权重只有 14 GB”不代表 20 GB 显存一定能训练或承载目标并发。

### 拷贝与计算重叠

若 Host→Device 拷贝耗时 $T_{copy}$ ，GPU 计算耗时 $T_{compute}$ ：

- 串行流水单步近似为 $T_{copy}+T_{compute}$ ；
- 双缓冲理想稳态近似为 $\max(T_{copy},T_{compute})$ 。

实际能否重叠取决于：

- pinned host buffer；
- 非阻塞拷贝；
- 不同 CUDA Stream 与正确事件依赖；
- 设备是否支持对应方向的并发 copy/compute；
- 缓冲区不能在拷贝完成前被覆盖；
- 数据准备是否及时。

`non_blocking=True` 只是不在调用点主动同步，不等于自动产生端到端重叠。

## 全局内存访问优化

### 连续、对齐和合并

最常见的好模式是：Warp 中相邻 lane 访问相邻元素。

```
int i = blockIdx.x * blockDim.x + threadIdx.x;
y[i] = x[i];
```

跨步模式可能浪费事务：

```
int i = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
y[i] = x[i];
```

但“Tensor 不是 contiguous”不等于一定慢。转置视图可能被后续高性能算子直接支持；某些维度跨步也可能与实际线程映射匹配。正确做法是看生成 Kernel 的地址模式和实际事务，而不是看到 `is_contiguous=False` 就无条件复制。

### AoS 与 SoA

假设每个对象有 `x、y、z、type` 四个字段。

Array of Structures：

```
[x0 y0 z0 t0][x1 y1 z1 t1][x2 y2 z2 t2]...
```

Structure of Arrays：

```
x0 x1 x2 ...
y0 y1 y2 ...
z0 z1 z2 ...
t0 t1 t2 ...
```

若 Kernel 只读取 `x` ，SoA 更容易让相邻 lane 读取相邻有效数据；AoS 会把未使用字段一起带入缓存线和事务。若每次总要读取完整对象，AoS 也可能合理。布局必须服务于访问模式。

### 向量化访问

`float4` 、128-bit load/store 或编译器生成的向量指令可以减少指令数量，但前提是：

- 地址满足对齐要求；
- 尾部元素正确处理；
- 向量宽度不会增加无效字节；
- 寄存器压力没有抵消收益。

向量化改善的可能是指令吞吐，而不一定提升 DRAM 峰值。

## Shared Memory 优化

### Tile：用一次 DRAM 读取换多次片上复用

矩阵乘法中，一个输入 Tile 会被多个线程重复使用。朴素实现从全局内存反复加载，Tile 实现则：

1. 合并读取 A、B Tile 到 Shared Memory；
2. 块内同步；
1. 多次复用 Tile 计算；
2. 继续加载下一个 Tile。

Tile 过小，复用不足；Tile 过大，Shared Memory 与 Register 压力会降低 Occupancy。最佳值需按架构、dtype、形状和 Kernel 流水扫参。

### Bank Conflict

现代 NVIDIA GPU 的 Shared Memory 通常组织为 32 个 Bank，相邻 32-bit word 映射到相邻 Bank。一个 Warp 同时访问不同 Bank 时可并行；多个 lane 访问同一 Bank 的不同地址时，会被拆成多次请求。多个 lane 读取完全相同地址可以广播，不按普通冲突处理。

二维 Tile 转置的经典问题：

```
__shared__ float tile[32][32];
```

按列访问时，相邻线程地址相差 32 个 `float` ，Bank 索引取模后可能都落到同一 Bank。增加一列 padding：

```
__shared__ float tile[32][33];
```

步长从 32 变为 33，Bank 映射错开，通常可以消除这种 32-way 冲突。

Padding 会增加 Shared Memory 占用；在 Tile 很多或流水级很多时，也要重新检查 Occupancy。

### 同步成本

Shared Memory 是显式协作空间，数据依赖必须正确同步：

- 只在单 Warp 内交换时，某些场景可用 `__syncwarp()` ；
- 跨 Warp 读写通常需要 `__syncthreads()` 或更精确的同步原语；
- 所有需要到达 Block Barrier 的线程都必须遵守控制流要求；
- 少一次必要同步会错误，多一次不必要同步会阻塞并行。

## 缓存、预取与异步搬运

### L1/L2：先测工作集，再谈命中

缓存受以下因素影响：

- 工作集相对缓存容量；
- 时间和空间局部性；
- 多个 Block 或 Kernel 的竞争；
- 数据是否只使用一次；
- 访问是否跨越大量缓存线；
- L2 persistence 等高级策略是否适用。

若数据只流式读取一次，追求高 L2 命中率没有意义；此时更重要的是合并、并发和减少总字节数。

### Ampere：异步 global→shared

Ampere（RTX 3080/3090、A100）引入可直接把数据从 Global Memory 异步搬到 Shared Memory 的能力。相比“load 到寄存器，再 store 到 Shared”，它可以减少中间寄存器使用，并帮助构建 copy/compute 流水。

Ada（RTX 4090、L40/L40S）沿用相关 CUDA 异步拷贝能力，但产品资源、缓存和最佳流水深度需要单独测量。

### Hopper：TMA

Hopper（H100/H200）的 TMA 能描述多维张量搬运，减少参与地址计算和数据移动的线程开销，并支持 Warp-specialized producer/consumer 流水。TMA 是硬件能力；在 RTX 3090 或 4090 上用普通 `memcpy` 模拟流水，只能验证算法结构，不能声称验证了 TMA 性能。

### Blackwell：分数据中心与消费级

Blackwell 数据中心产品 B100/B200/GB200 与 RTX 5090 都属于 Blackwell，但 Compute Capability、Shared Memory 资源和部分专用目标不同。B200 专项 TMA/Cluster Kernel 不能假设在 RTX 5090 上原样可用；必须编译不同目标并提供通用 CUDA 回退。

## PyTorch 内存管理

### allocated、reserved 与 nvidia-smi

PyTorch CUDA caching allocator 会保留已释放的显存块，以降低频繁调用底层分配器的成本。

- `torch.cuda.memory_allocated()` ：当前存活 Tensor 占用的显存。
- `torch.cuda.memory_reserved()` ：分配器从 CUDA 保留的显存池。
- `nvidia-smi` ：进程视角的设备占用，还可能包含 CUDA Context、库 Workspace 与非 PyTorch 分配。

因此常见关系是：

$$
allocated \le reserved \le process\ memory\ seen\ by\ nvidia\!-smi
$$

三者口径不同，不应要求完全相等。

### empty\_cache() 的真实作用

`torch.cuda.empty_cache()` 可以把缓存分配器中当前未使用的块归还给 CUDA，使它们更容易被其他进程使用。它不能：

- 释放仍被 Python 引用的 Tensor；
- 降低当前模型真正需要的峰值显存；
- 自动修复持有计算图造成的泄漏；
- 保证当前进程之后更快。

循环中频繁调用 `empty_cache()` 往往增加同步和重新分配成本。

### 真泄漏与“缓存看起来没降”

常见真泄漏模式：

```
losses.append(loss)          # 保留整个 autograd graph
outputs_history.append(out)  # 长期持有 GPU Tensor
```

更安全的记录方式：

```
losses.append(loss.item())
outputs_history.append(out.detach().cpu())
```

如果 `allocated` 随迭代持续增长，优先找仍存活的 Tensor；如果 `allocated` 稳定而 `reserved` 较高，可能只是缓存池在复用。

### 碎片与 OOM

当存在足够总空闲字节，却没有满足大块请求的连续可用块时，会表现为碎片问题。排查顺序：

1. 记录峰值 `max_memory_allocated` 和 `max_memory_reserved` 。
2. 查看 `memory_summary()` 中块分布与 OOM 次数。
1. 检查形状频繁变化是否制造多种块尺寸。
2. 检查计算图、临时 Tensor 和 Workspace 生命周期。
1. 再评估 PyTorch 当前版本支持的 allocator 配置。

不要把旧资料中的某个环境变量参数原样复制到所有 PyTorch/CUDA 版本。Allocator 选项会变化，应以当前官方 CUDA Notes 为准并记录版本。

## 瓶颈分析方法

### 四步诊断

### 第一步：确认容量与生命周期

记录：

- 参数、梯度、优化器、激活与 Workspace 的估算；
- `allocated/reserved/peak` ；
- 输入形状和 batch/token 长度；
- OOM 发生在 forward、backward、optimizer step 还是临时编译阶段。

### 第二步：计算有效带宽与算术强度

按算法字节数算 `effective GB/s` ，与 Roofline 中的带宽上界比较。若有效带宽低，不要立即断言“DRAM 慢”，还可能是：

- 访问不合并；
- 发散或依赖使请求不足；
- 工作规模过小；
- Launch/同步占比大；
- 数据主要命中缓存，公式口径与 DRAM 口径不一致。

### 第三步：查看硬件事务与 Stall

用 Nsight Compute 的 Memory Workload Analysis、Speed of Light、Warp State Statistics 等分区检查：

- Requested bytes 与实际 sector/byte；
- DRAM、L2、L1/Texture 吞吐；
- Shared Memory Bank Conflict；
- Local Memory load/store；
- Memory dependency、long scoreboard 等 Stall 原因。

### 第四步：一次只改变一个变量

按以下优先级做实验：

1. 改布局或线程映射，使访问合并。
2. 消除重复读取，提高复用。
1. 修复 Bank Conflict。
2. 调整 Tile、寄存器和 Shared Memory 的资源平衡。
1. 引入异步拷贝和多级流水。
2. 最后再尝试细粒度 cache hint 或架构专用指令。

## 完整可运行实验：Memory 层级与数据搬运

### 环境准备

Level 0 仅需 Python 3：

```
python3 --version
```

Level 1 可选 PyTorch：

```
python3 -m venv .venv-gpu-memory
source .venv-gpu-memory/bin/activate
python -m pip install --upgrade pip
# 根据 https://pytorch.org/get-started/locally/ 选择与驱动匹配的稳定版
python -m pip install torch
```

### 保存脚本

将以下内容保存为 `gpu_memory_lab.py` ：

```
#!/usr/bin/env python3
import argparse
import gc
import json
import math
import statistics
import time

def transaction_model(stride, offset_bytes=0, element_bytes=4,
                      lanes=32, segment_bytes=32):
    addresses = [offset_bytes + lane * stride * element_bytes
                 for lane in range(lanes)]
    segments = {addr // segment_bytes for addr in addresses}
    requested = lanes * element_bytes
    actual = len(segments) * segment_bytes
    return {
        "stride_elements": stride,
        "offset_bytes": offset_bytes,
        "segments": len(segments),
        "requested_bytes": requested,
        "modeled_actual_bytes": actual,
        "efficiency": round(requested / actual, 4),
    }

def bank_model(stride_words, lanes=32, banks=32):
    bank_hits = [0] * banks
    for lane in range(lanes):
        bank = (lane * stride_words) % banks
        bank_hits[bank] += 1
    nonzero = [x for x in bank_hits if x]
    return {
        "stride_words": stride_words,
        "banks_touched": len(nonzero),
        "max_conflict_degree": max(nonzero),
        "idealized_rounds": max(nonzero),
        "bank_hits": bank_hits,
    }

def capacity_model(params_billion, bytes_per_param,
                   grad_bytes, optimizer_bytes, activation_gib,
                   workspace_gib, fragmentation_ratio):
    n = params_billion * 1e9
    param = n * bytes_per_param
    grad = n * grad_bytes
    optimizer = n * optimizer_bytes
    base = param + grad + optimizer + (activation_gib + workspace_gib) * 2**30
    total = base * (1 + fragmentation_ratio)
    return {
        "parameter_gib": round(param / 2**30, 3),
        "gradient_gib": round(grad / 2**30, 3),
        "optimizer_gib": round(optimizer / 2**30, 3),
        "activation_gib": activation_gib,
        "workspace_gib": workspace_gib,
        "fragmentation_ratio": fragmentation_ratio,
        "estimated_total_gib": round(total / 2**30, 3),
        "warning": "教学估算，不包含框架、通信和所有临时张量",
    }

def cpu_copy_benchmark(size_mib, repeat):
    size = size_mib * 2**20
    src = bytearray(size)
    dst = bytearray(size)
    samples = []
    for _ in range(3):
        dst[:] = src
    for _ in range(repeat):
        start = time.perf_counter()
        dst[:] = src
        samples.append(time.perf_counter() - start)
    median = statistics.median(samples)
    return {
        "size_mib": size_mib,
        "median_ms": round(median * 1000, 4),
        "logical_copy_gbs": round(size / median / 1e9, 3),
        "note": "Python bytearray CPU 回退结果，不代表 GPU 带宽",
    }

def torch_benchmark(elements, matrix_rows, matrix_cols, repeat,
                    fragmentation_demo):
    try:
        import torch
    except ImportError:
        return {"available": False, "reason": "未安装 PyTorch"}

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    result = {
        "available": True,
        "torch_version": torch.__version__,
        "device": str(device),
    }
    if use_cuda:
        prop = torch.cuda.get_device_properties(0)
        cc = torch.cuda.get_device_capability(0)
        result.update({
            "gpu": prop.name,
            "compute_capability": f"{cc[0]}.{cc[1]}",
            "total_memory_gib": round(prop.total_memory / 2**30, 3),
            "cuda_runtime": torch.version.cuda,
        })

    def sync():
        if use_cuda:
            torch.cuda.synchronize()

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

    # 低算术强度：两读一写，按逻辑字节估算有效带宽。
    x = torch.randn(elements, device=device, dtype=torch.float32)
    y = torch.randn(elements, device=device, dtype=torch.float32)
    add_ms = median_ms(lambda: torch.add(x, y))
    add_bytes = elements * 4 * 3

    # 连续与跨步视图都归约相同数量的元素。
    stride = 8
    base = torch.randn(elements * stride, device=device, dtype=torch.float32)
    contiguous = base[:elements]
    strided = base[::stride]
    contiguous_ms = median_ms(lambda: contiguous.sum())
    strided_ms = median_ms(lambda: strided.sum())

    # 同样的元素数，比较 contiguous 和 transpose view 的逐元素读取。
    matrix = torch.randn((matrix_rows, matrix_cols),
                         device=device, dtype=torch.float32)
    transposed = matrix.t()
    transpose_view_ms = median_ms(lambda: transposed * 1.0001)
    transpose_copy_ms = median_ms(lambda: transposed.contiguous())

    result["device_memory"] = {
        "elementwise_add": {
            "elements": elements,
            "median_ms": round(add_ms, 4),
            "effective_gbs": round(add_bytes / (add_ms / 1000) / 1e9, 3),
        },
        "contiguous_vs_stride_sum": {
            "elements_each": elements,
            "contiguous_ms": round(contiguous_ms, 4),
            "stride": stride,
            "strided_ms": round(strided_ms, 4),
            "slowdown": round(strided_ms / contiguous_ms, 3),
        },
        "transpose": {
            "shape": [matrix_rows, matrix_cols],
            "is_contiguous": transposed.is_contiguous(),
            "elementwise_read_ms": round(transpose_view_ms, 4),
            "materialize_contiguous_ms": round(transpose_copy_ms, 4),
        },
    }

    if use_cuda:
        # 分配 pageable 与预先 pinned 的 Host Tensor；拷贝后同步以测完成时间。
        host_pageable = torch.randn(elements, dtype=torch.float32)
        try:
            host_pinned = torch.randn(elements, dtype=torch.float32,
                                      pin_memory=True)

            pageable_ms = median_ms(
                lambda: host_pageable.to("cuda", non_blocking=False), warmup=3
            )
            pinned_sync_ms = median_ms(
                lambda: host_pinned.to("cuda", non_blocking=False), warmup=3
            )
            pinned_async_ms = median_ms(
                lambda: host_pinned.to("cuda", non_blocking=True), warmup=3
            )
            copy_bytes = elements * 4
            result["host_to_device"] = {
                "bytes": copy_bytes,
                "pageable_blocking_ms": round(pageable_ms, 4),
                "pinned_blocking_ms": round(pinned_sync_ms, 4),
                "pinned_non_blocking_completed_ms": round(pinned_async_ms, 4),
                "pinned_non_blocking_effective_gbs": round(
                    copy_bytes / (pinned_async_ms / 1000) / 1e9, 3
                ),
                "note": "计时包含最终同步，测的是完成时间；单拷贝不等于重叠流水",
            }
        except RuntimeError as exc:
            result["host_to_device"] = {
                "skipped": True,
                "reason": str(exc).splitlines()[0],
            }

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        before = {
            "allocated_mib": round(torch.cuda.memory_allocated() / 2**20, 3),
            "reserved_mib": round(torch.cuda.memory_reserved() / 2**20, 3),
        }
        chunks = [torch.empty(16 * 2**20 // 4, device="cuda")
                  for _ in range(8)]
        live = {
            "allocated_mib": round(torch.cuda.memory_allocated() / 2**20, 3),
            "reserved_mib": round(torch.cuda.memory_reserved() / 2**20, 3),
        }
        del chunks[::2]
        gc.collect()
        after_delete = {
            "allocated_mib": round(torch.cuda.memory_allocated() / 2**20, 3),
            "reserved_mib": round(torch.cuda.memory_reserved() / 2**20, 3),
        }
        torch.cuda.empty_cache()
        after_empty_cache = {
            "allocated_mib": round(torch.cuda.memory_allocated() / 2**20, 3),
            "reserved_mib": round(torch.cuda.memory_reserved() / 2**20, 3),
        }
        result["allocator"] = {
            "before": before,
            "with_live_tensors": live,
            "after_delete_half": after_delete,
            "after_empty_cache": after_empty_cache,
            "peak_allocated_mib": round(
                torch.cuda.max_memory_allocated() / 2**20, 3
            ),
        }
        del chunks
        gc.collect()
        torch.cuda.empty_cache()

        if fragmentation_demo:
            result["allocator"]["memory_summary"] = \
                torch.cuda.memory_summary(abbreviated=True)
    else:
        result["host_to_device"] = {
            "skipped": True,
            "reason": "CPU 回退没有 CUDA Host↔Device 链路",
        }
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size-mib", type=int, default=64)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--torch", action="store_true")
    parser.add_argument("--elements", type=int, default=1_000_000)
    parser.add_argument("--matrix-rows", type=int, default=1024)
    parser.add_argument("--matrix-cols", type=int, default=1024)
    parser.add_argument("--fragmentation-demo", action="store_true")
    parser.add_argument("--params-billion", type=float, default=7.0)
    parser.add_argument("--bytes-per-param", type=float, default=2.0)
    parser.add_argument("--grad-bytes", type=float, default=2.0)
    parser.add_argument("--optimizer-bytes", type=float, default=8.0)
    parser.add_argument("--activation-gib", type=float, default=4.0)
    parser.add_argument("--workspace-gib", type=float, default=1.0)
    parser.add_argument("--fragmentation-ratio", type=float, default=0.1)
    args = parser.parse_args()

    output = {
        "global_transactions": [
            transaction_model(stride, offset)
            for stride in (1, 2, 4, 8, 16)
            for offset in (0, 4)
        ],
        "shared_memory_banks": [
            bank_model(stride) for stride in (1, 2, 4, 8, 16, 32, 33)
        ],
        "capacity": capacity_model(
            args.params_billion, args.bytes_per_param,
            args.grad_bytes, args.optimizer_bytes,
            args.activation_gib, args.workspace_gib,
            args.fragmentation_ratio,
        ),
        "cpu_copy": cpu_copy_benchmark(args.size_mib, args.repeat),
    }
    if args.torch:
        output["torch"] = torch_benchmark(
            args.elements, args.matrix_rows, args.matrix_cols,
            args.repeat, args.fragmentation_demo
        )
    print(json.dumps(output, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
```

### Level 0：任何机器可运行

运行默认实验：

```
python3 gpu_memory_lab.py > memory_level0.json
python3 -m json.tool memory_level0.json | less
```

改变 7B 模型的内存假设：

```
python3 gpu_memory_lab.py \
  --params-billion 7 \
  --bytes-per-param 2 \
  --grad-bytes 2 \
  --optimizer-bytes 8 \
  --activation-gib 8 \
  --workspace-gib 2 \
  --fragmentation-ratio 0.15 \
  > memory_7b_estimate.json
```

### 预期现象

- Global Memory `stride=1、offset=0` 的简化模型使用 4 个 32-byte segment，效率为 1。
- `stride=8` 时 32 个 lane 通常触及 32 个 segment，简化效率降为 0.125。
- 对连续 FP32 地址增加 4-byte offset，可能从 4 个 segment 增至 5 个，说明对齐会影响事务数量。
- Shared Memory `stride_words=1` 触及 32 个 Bank，冲突度为 1。
- `stride_words=32` 的不同地址映射到同一 Bank，简化冲突度为 32。
- `stride_words=33` 重新均匀映射到 32 个 Bank，这就是 `[32][33]` padding 的直觉。

### 结果边界

Level 0 模型没有模拟缓存、广播、事务合并细节、Bank 双端口行为和编译器优化。它用于在没有 GPU 时验证地址到 segment/Bank 的映射逻辑，不是硬件性能预测器。

`cpu_copy` 使用 Python `bytearray` 测量 CPU 内存复制，只能训练“预热、重复、中位数、工作字节”方法，不能与 GPU HBM/GDDR 规格直接比较。

### Level 1：通用 PyTorch CPU/CUDA 实测

先检测环境：

```
nvidia-smi --query-gpu=name,driver_version,memory.total \
  --format=csv,noheader 2>/dev/null || true

python3 - <<'PY'
try:
    import torch
    print("torch", torch.__version__)
    print("cuda runtime", torch.version.cuda)
    print("cuda available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu", torch.cuda.get_device_name(0))
        print("cc", torch.cuda.get_device_capability(0))
except ImportError:
    print("PyTorch not installed")
PY
```

自动使用 CUDA，无 CUDA 时回退 CPU：

```
python3 gpu_memory_lab.py --torch \
  --elements 1000000 \
  --matrix-rows 1024 --matrix-cols 1024 \
  --repeat 10 \
  > memory_torch.json
```

显存与主机内存充足时扩大规模：

```
python3 gpu_memory_lab.py --torch \
  --elements 8000000 \
  --matrix-rows 4096 --matrix-cols 4096 \
  --repeat 20 --fragmentation-demo \
  > memory_cuda_large.json
```

### 预期现象

- 逐元素加法的有效带宽随问题规模增大后趋于稳定，小规模时 Launch 与计时开销占比较高。
- 跨步归约处理相同元素数，但读取更分散，通常比连续归约慢；具体倍数取决于缓存、设备和规模。
- 转置 Tensor 是 view， `is_contiguous=False` ；对 view 做逐元素运算未必需要先复制。
- `transposed.contiguous()` 本身有一次完整的数据重排成本。只有后续算子累计收益超过该成本，预先 contiguous 才值得。
- CUDA 环境中，预先 pinned 的 Host Tensor 往往有更好的大块 H2D 吞吐，但小 Tensor 可能被分配、Launch 与同步成本主导。
- 删除一半 Tensor 后， `allocated` 应下降； `reserved` 可能仍保持较高。 `empty_cache()` 可归还未使用缓存块，但不会释放仍存活的另一半 Tensor。

### 正确解释 H2D 结果

脚本测量每次拷贝的完成时间，因此在每个样本后同步。 `pinned_non_blocking` 的数值说明异步 API 的单次完成成本，不直接证明与计算发生了重叠。

要验证重叠，应建立双缓冲训练循环，并在 Nsight Systems 时间线上确认 memcpy 与 Kernel 横向重叠。仅看 Python 调用返回很快会漏掉尚未完成的 GPU 工作。

### Level 2：架构专项实验

### 能力检测

```
python3 - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("无 CUDA GPU：完成 Level 0/CPU 回退即可")
p = torch.cuda.get_device_properties(0)
print({
    "name": p.name,
    "cc": torch.cuda.get_device_capability(0),
    "total_memory_GiB": round(p.total_memory / 2**30, 2),
})
PY
```

### 支持矩阵

| 架构 | 示例 | 可选 Memory 专项 | 通用回退 |
| --- | --- | --- | --- |
| Ampere | RTX 3080/3090、A100 | `cp.async` global→shared 流水 | 普通 load/store + 双缓冲模型 |
| Ada Lovelace | RTX 4090、L40/L40S | 异步拷贝、产品相关 L2 行为 | Ampere 风格通用 CUDA Kernel |
| Hopper | H100/H200 | TMA、多维搬运、Cluster DSM | `cp.async` 或同步 Tile |
| Blackwell 数据中心 | B100/B200/GB200 | 增强 TMA/Cluster、架构专用目标 | 按 CC 编译通用 Tile Kernel |
| Blackwell 消费级 | RTX 5090 | CC 12.0 对应能力 | 不直接运行 B200 专属 `sm_100a` Kernel |

### Nsight Compute 诊断

对已有 CUDA 程序：

```
ncu --set basic --target-processes all ./your_kernel
```

在图形界面重点查看：

- Memory Workload Analysis；
- L1/TEX、L2、DRAM 吞吐；
- Global Load/Store Efficiency 或相应 requested/actual 指标；
- Shared Memory Bank Conflict；
- Local Memory 访问；
- Occupancy 与 Warp Stall。

指标名称会随 Nsight Compute 和架构更新。课程不硬编码一套所有版本都存在的底层 metric 名；以本机 `ncu --query-metrics` 和报告分区为准。

### 编译器资源检查

```
nvcc -O3 --ptxas-options=-v memory_kernel.cu -o memory_kernel
```

重点观察：

- 每线程寄存器数；
- Shared Memory 字节数；
- spill stores / spill loads；
- 编译目标是否与当前 CC 匹配。

没有对应硬件时，可以完成地址模型、流水依赖和回退实现，但不能把普通 CUDA copy 宣称为 TMA，也不能把 RTX 5090 当作 B200 的等价替代。

## 优化前后对照

| 场景 | 优化前 | 优化后 | 验证指标 |
| --- | --- | --- | --- |
| Global Memory | Warp 跨步、错位访问 | 相邻 lane 连续且合理对齐 | requested/actual、有效 GB/s |
| 数据布局 | 只用单字段却采用 AoS | 按访问模式改 SoA | 实际事务、Kernel 时间 |
| 数据复用 | 每次都从 DRAM 读 | Shared/Register Tile 复用 | DRAM bytes、AI |
| Shared Memory | `[32][32]` 按列访问 | `[32][33]` padding | Bank Conflict、耗时 |
| 寄存器 | 为高 Occupancy 强制压低 | 在 spill 与驻留度间扫参 | local load/store、Occupancy |
| H2D | 每步 pageable 同步拷贝 | 复用 pinned buffer + async + stream | 时间线重叠、step time |
| Tensor 布局 | 每次无条件 `.contiguous()` | 只在收益覆盖重排成本时复制 | 端到端耗时、额外 bytes |
| Allocator | 循环频繁 `empty_cache()` | 稳定形状、合理生命周期和复用 | peak allocated、OOM、step time |
| 超显存 | 依赖按需迁移且延迟抖动 | 分片、checkpoint、量化或显式 offload | P99、page migration、吞吐 |

## 常见错误与排查

### 错误 1：显存使用率高就是 Memory Bandwidth 跑满

\*\*原因\*\*：容量占用与单位时间搬运量是不同概念。

\*\*排查\*\*：同时记录 allocated/reserved、DRAM 吞吐、实际字节和 Kernel 时间。

### 错误 2：nvidia-smi 与 memory\_allocated() 不相等就是泄漏

\*\*原因\*\*：CUDA Context、库 Workspace、缓存池和非 PyTorch 分配的统计口径不同。

\*\*排查\*\*：看 allocated 是否随迭代持续增长，并检查仍存活的 Tensor 和计算图。

### 错误 3：OOM 后反复调用 empty\_cache()

\*\*原因\*\*：活跃 Tensor 的峰值需求没有下降。

\*\*排查\*\*：降低 batch/sequence、Activation Checkpoint、缩短生命周期、减少 Workspace 或使用分片；把 `empty_cache()` 视作缓存管理，不是容量魔法。

### 错误 4：把.contiguous() 当作免费优化

\*\*原因\*\*：它可能产生完整复制和额外峰值显存。

\*\*排查\*\*：比较“直接算”与“一次 contiguous + 后续多次算”的端到端时间。

### 错误 5：pin\_memory=True 一定更快

\*\*原因\*\*：小 batch、CPU 内存压力、锁页成本或数据准备瓶颈可能抵消收益。

\*\*排查\*\*：扫 batch size、worker、prefetch；监控主机内存并看时间线。

### 错误 6：non\_blocking=True 就代表已完成拷贝

\*\*原因\*\*：它可能只让 Host 提前返回。

\*\*排查\*\*：访问结果前建立 Stream/Event 依赖；计时完成时间时同步；Pinned 源缓冲在拷贝结束前不可修改。

### 错误 7：Shared Memory 一定比 L1 快

\*\*原因\*\*：显式搬运、同步、Bank Conflict 和 Occupancy 损失都有成本。

\*\*排查\*\*：比较硬件缓存基线与 Shared Tile 实现，不能只比较理论延迟。

### 错误 8：Local Memory 是片上高速内存

\*\*原因\*\*：名称表示线程私有作用域，不代表物理位置。

\*\*排查\*\*：看 ptxas spill 和 Nsight Compute local load/store。

### 错误 9：只根据架构名启用专用 Kernel

\*\*原因\*\*：同属 Blackwell 的 CC 10.x 与 12.0 也有能力差异。

\*\*排查\*\*：运行时检查 CC，构建独立二进制目标并保留正确性回退。

### 错误 10：实验规模过大导致主机 OOM

本课脚本的跨步基数组为 `elements × 8 × 4 bytes` ，还会同时存在其他 Tensor。若资源有限：

```
python3 gpu_memory_lab.py --torch \
  --elements 250000 --matrix-rows 512 --matrix-cols 512 --repeat 5
```

## 面试题与答案

### 1\. GPU Memory 优化与减少显存占用有什么区别？

减少显存主要解决容量问题；Memory 优化还包括减少流量、提高事务效率、增加片上复用、隐藏延迟和重叠传输。容量更小的实现可能因为重计算或不规则访问反而更慢。

### 2\. 为什么跨步访问会浪费带宽？

Warp 的地址被合并为固定粒度事务。若每个 lane 访问不同且相距很远的 segment，硬件必须搬运多个 segment，而线程只使用其中少量字节，请求字节与实际字节之比下降。

### 3\. Shared Memory 的三个主要用途是什么？

块内数据复用、把全局内存访问重排为合并访问、在线程间交换数据。代价包括显式搬运、同步、Bank Conflict 和资源占用。

### 4\. 什么是 Bank Conflict？如何处理？

同一 Warp 的多个线程访问同一 Shared Memory Bank 中不同地址时，请求需要拆分。可通过改变布局、padding、索引映射或使用 Warp Shuffle 减少冲突；广播同一地址属于特殊情况。

### 5\. 为什么 \[32\]\[33\] 能改善转置 Tile？

32 列时按列访问的地址步长为 32 个 word，Bank 索引取模后重复；增加一列使步长为 33，等价于 Bank 索引每次前进 1，从而分散到不同 Bank。

### 6\. Register spill 是什么？

线程需要的寄存器无法全部分配时，部分值被放到 Local Memory。Local Memory 物理上位于设备内存并经过缓存，访问成本可能远高于寄存器。

### 7\. memory\_allocated 与 memory\_reserved 有何区别？

allocated 统计当前存活 Tensor 使用量；reserved 统计 PyTorch allocator 从 CUDA 保留的内存池。删除 Tensor 后 allocated 可下降，而 reserved 可能保留以供复用。

### 8\. empty\_cache() 为什么不能解决所有 OOM？

它只能释放缓存池中未被活跃 Tensor 使用的块，无法释放仍被引用的 Tensor，也不会降低模型、激活和 Workspace 的真实峰值需求。

### 9\. Pinned Memory 为什么快，又为什么不能滥用？

锁页内存便于 DMA 和异步传输，能减少临时 staging 成本；但锁页是昂贵且有限的系统资源，过量使用会影响操作系统和其他进程。

### 10\. non\_blocking=True 何时真正有用？

当多个拷贝可批量发起、源缓冲满足要求，或拷贝在独立 Stream 中与计算重叠时最有用。若每次拷贝后立即同步或立刻依赖结果，收益有限。

### 11\. 为什么 TMA 不能在 RTX 3090 上等价模拟？

TMA 是 Hopper 及后续相关数据中心路径的硬件张量搬运单元。普通 CUDA load/store 可以实现相同数学数据移动，但指令、线程参与、寄存器开销和硬件流水不同。

### 12\. 如何判断是容量、带宽还是延迟问题？

容量问题表现为峰值接近上限或 OOM；带宽问题需要结合有效字节和 DRAM/L2 吞吐；延迟问题常表现为吞吐未满且 Warp 因依赖或 long scoreboard 等原因等待。最终需要模型与计数器交叉验证。

## 课后练习

1. 修改 Level 0 的 `element_bytes` 为 2、8、16，观察 segment 效率变化。
2. 将起始 offset 从 0 扫到 31 字节，画出连续 FP32 访问的事务数量曲线。
1. 对 Shared Memory stride 1～64 建表，解释冲突度与 `gcd(stride, 32)` 的关系。
2. 为一个 7B 模型分别估算仅推理、全参数训练和 Adam 混合精度训练的显存组成。
1. 在 CPU 与 CUDA 上运行 Level 1，比较连续与 $stride=8$ 的趋势，并解释为何倍数不同。
2. 比较直接使用转置 view 与先 `.contiguous()` 后重复运行 1、10、100 次逐元素操作的总时间。
1. 修改 allocator 实验，每轮记录 allocated/reserved，构造“缓存增长”与“真引用泄漏”两个案例。
2. 用 DataLoader 扫描 `pin_memory={False,True}` 、 `num_workers={0,2,4}` ，记录端到端 step time。
1. 若有 Nsight Compute，对一个跨步 Kernel 检查 requested/actual 流量，并与 Level 0 模型比较。
2. 为 RTX 3090、RTX 4090、H100 与 RTX 5090 设计一个 Memory Kernel 能力分派表。

## Checklist

### 层级与模型

- 能区分 Register、Local、Shared、L1、L2、GDDR/HBM 与 Host Memory。
- 知道 Local Memory 的“local”表示作用域，不表示片上位置。
- 能计算向量加法的逻辑字节与有效带宽。
- 能区分 requested bytes 与 actual bytes。
- 能从复用率解释算术强度变化。

### Kernel 访问

- 能判断 Warp 访问是否连续且合理对齐。
- 能根据访问字段选择 AoS 或 SoA。
- 能解释 Shared Memory Bank Conflict 与广播例外。
- 能解释 \[32\]\[33\] padding 的作用和资源成本。
- 会同时考虑 Register spill 与 Occupancy。

### 框架与数据管线

- 能区分 allocated、reserved 和 nvidia-smi 口径。
- 不把 empty\\\_cache() 当成释放活跃 Tensor 的方法。
- 使用 Pinned Memory 前会测量主机内存和端到端收益。
- 知道 $non_blocking=True$ 不等于工作已经完成。
- 能在时间线上验证 H2D 与计算重叠。

### 架构边界

- Ampere：RTX 3080/3090、A100，可学习 cp.async。
- Ada Lovelace：RTX 4090、L40/L40S，不写成 Blackwell。
- Hopper：H100/H200，可验证 TMA 与 Cluster 相关能力。
- Blackwell：RTX 5090、B100/B200/GB200，但 CC 路径需分别检测。
- 无专项硬件时只做通用回退，不宣称等价模拟。

### 实验

- 已运行 Level 0 并保存 JSON。
- 有 PyTorch 时已完成 Level 1。
- GPU 计时包含预热、同步、重复与中位数。
- 所有性能数字只描述本次环境。
- 优化后重新验证正确性、峰值显存和端到端时间。