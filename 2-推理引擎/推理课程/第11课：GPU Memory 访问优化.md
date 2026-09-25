---
title: "第11课：GPU Memory 访问优化"
source: "https://everythingai.top/learn/ai-systems-performance-engineering-text/lesson-11"
author:
published:
created: 2026-09-26
description: "面向 AI 工程学习者的专业课程学习空间。"
tags:
  - "clippings"
---
**导航**

本课聚焦合并访存、Shared Memory、Bank Conflict 与缓存行为，系统提升 GPU Memory 访问效率。

## 课程定位

现代 GPU 的计算能力增长很快，但很多 Kernel 并不是“算不动”，而是“数据送不到”。即使 Global Memory 带宽很高，Warp 中 32 个线程的地址分散、未对齐或跨步过大，也会把一个逻辑 Load 拆成许多 Memory Transaction；Shared Memory 虽然在片上，Bank Conflict 仍可能让访问串行化。

本课把“Memory Coalescing”从一句口号变成可计算、可测量的工程模型，并进一步连接 Alignment、AoS/SoA、Vectorized Load、Shared Memory Tiling、Bank Conflict、Transpose、Warp Shuffle、 `cp.async` 与 Hopper TMA。

## 学习目标

完成本课后，你能够：

1. 从 Warp 地址集合推导 Global Memory Transaction 数与传输效率。
2. 解释连续访问、Misalignment、Stride、Scatter/Gather 的不同代价。
1. 在 AoS 与 SoA 之间按实际字段访问模式选择布局。
2. 正确使用 `float2/float4` 等 Vectorized Access，并处理 Alignment 与 Tail。
1. 解释 Shared Memory Bank Mapping、Broadcast 与 N-way Bank Conflict。
2. 使用 Padding 消除经典矩阵转置中的 Bank Conflict。
1. 使用 Shared Memory 完成 Global Memory 的合并读写与片上重排。
2. 区分 Ampere `cp.async` 、Hopper TMA 与 Blackwell 相关演进。
1. 用 CUDA Event、正确性验证和 Nsight Compute 分析 Memory Kernel。

## 前置知识

- 已完成 CUDA 编程模型，理解 Grid、Block、Thread、Warp、Stream 和 Event。
- 理解 Register、Shared Memory、L1/L2 与 Global Memory 层级。
- 会编译和运行 CUDA C++；Level 0 只需要 Python。
- 理解有效带宽和 Roofline。

## 核心直觉：硬件处理的是地址集合，不是“数组语义”

很多人会把 Coalescing 理解成：

这个直觉已经接近正确，但需要精确两点：

1. 硬件并不知道 C++ 的“行”或 Tensor 的“维度”，只看到一个 Warp 发出的地址集合；
2. 地址集合会被归并为若干对齐的 Memory Segment/Transaction，而不是任何情况下都只有一次物理读取。

对 Compute Capability 6.0+，以 32 个线程各读取一个相邻 `float` 为例：

```
32 threads × 4 bytes = 128 bytes requested
```

在理想对齐下，官方简化模型为 4 个 32-byte Transaction，而不是“一个不可再分的 128-byte 操作”。如果首地址错开 4 Byte，可能覆盖 5 个 32-byte Segment；如果每个线程跨 32 个 `float` ，则可能触及大量不同 Segment。

因此更准确的说法是：

## Global Memory Transaction 模型

### 地址到 Segment

设第 l 个 Lane 的地址为 $a_l$ ，Segment 大小为 S Byte，则它属于：

$$
segment_l=\left\lfloor\frac{a_l}{S}\right\rfloor
$$

简化 Transaction 数：

$$
N_{txn}=\left|\{segment_0,segment_1,\ldots,segment_{31}\}\right|
$$

请求效率：

$$
\eta_{request}=\frac{Bytes_{requested}}{N_{txn}\times S}
$$

这个模型非常适合建立直觉，但真实硬件还受到 Cache Line、Sector、合并规则、访问宽度、Cache Hit、ECC、Store Policy 与架构影响。最终以 Profiler Counter 为准。

### 连续访问

```
float value = x[warp_base + lane];
```

32 个 Lane 访问相邻 4-byte Word，请求 128 Byte。首地址合理对齐时，通常只覆盖必要的 4 个 32-byte Segment，请求效率接近 100%。

### Misalignment

```
float value = x[warp_base + lane + 1];
```

只偏移一个 `float` ，地址范围可能从原来的 4 个 Segment 扩展为 5 个。简化效率：

$$
\eta=\frac{128}{5\times32}=80\%
$$

Cache 可能让相邻 Warp 复用多取回的数据，所以应用实测下降不一定正好 20%。

### Stride

```
float value = x[(warp_base + lane) * stride];
```

当 `stride=32` 时，相邻 Lane 相隔 128 Byte，很可能每个 Lane 触及不同 Segment。虽然每个线程只要 4 Byte，硬件可能搬运远多于 128 Byte，造成严重 Overfetch。

### Permutation 不一定破坏 Coalescing

如果 Warp 仍只访问相同的少量 Segment，即使 Lane 顺序被打乱，Transaction 数也可能不增加。Coalescing 关注地址覆盖集合，不要求 Lane 0 必须访问最低地址。

### 部分 Warp

边界 Warp 中只有少量 Lane Active 时，硬件仍按 Segment 搬运。Tail 很短时，Bandwidth Utilization 下降；通常影响有限，但大量不规则小任务会放大该问题。

## Alignment

`cudaMalloc` 返回的基地址满足较高 Alignment，但以下操作会破坏 Alignment：

- 对指针加非对齐 Offset；
- Struct 字段布局与 Padding 不当；
- 从 Byte Buffer 中切出任意地址；
- Tensor View/Storage Offset；
- 将 `float*` 强制转换为 `float4*` ，但地址不是 16-byte 对齐。

Vector Type 的基本要求：

| 类型 | Payload | 常见自然对齐要求 |
| --- | --- | --- |
| `float` | 4 B | 4 B |
| `float2` | 8 B | 8 B |
| `float4` | 16 B | 16 B |

Vectorized Load 的主要价值常是减少 Load/Store Instruction 数、提高每线程搬运宽度；它不会自动修复跨 Warp 的随机地址，也不保证 DRAM Transaction 更少。

## 数据布局：AoS 与 SoA

### Array of Structures

```
struct TokenState {
    float score;
    int token_id;
    float probability;
};
TokenState states[N];
```

若 Kernel 只读取 `score` ，相邻线程地址间隔为 `sizeof(TokenState)` ，会取回不需要的字段。

### Structure of Arrays

```
float scores[N];
int token_ids[N];
float probabilities[N];
```

只读取 `scores` 时，Warp 访问连续，通常更适合合并读取。

但 SoA 不是永远更好：如果每个线程总是一起读取一个对象的全部字段，紧凑且对齐良好的 AoS/AoSoA 可能更合适。布局选择要依据“Warp 在同一条指令中读哪些字段”。

### AoSoA

Array of Structures of Arrays 把数据分成 Warp/Tile 大小的块，在块内采用 SoA：

```
Block 0: scores[32], ids[32], probs[32]
Block 1: scores[32], ids[32], probs[32]
```

它常用于兼顾局部性、Vectorization 和对象分块，但实现更复杂。

## Shared Memory：片上重排缓冲区

Shared Memory 的三个核心用途：

1. 把 Global Memory 的分散访问转换为合并访问；
2. 在 Block 内复用数据，减少重复 Global Load；
1. 在片上进行 Transpose、Reduction、Stencil Halo 等数据重排。

典型流程：

```
Global Memory：按连续方向 Coalesced Load
          ↓
Shared Memory：按算法需要重排/复用
          ↓
Global Memory：按连续方向 Coalesced Store
```

## Shared Memory Bank Conflict

现代 CUDA GPU 的 Shared Memory 通常可抽象为 32 个 Bank。对常见 32-bit Word：

$$
bank=\left(\frac{address}{4}\right)\bmod32
$$

### 无冲突

```
shared[lane]
```

Lane 0\\~31 映射到 Bank 0\\~31，可并行服务。

### 32-way Conflict

```
shared[lane * 32]
```

所有 Lane 的 Bank Index 都为 0，但地址不同，访问需拆分为多个 Conflict-free Request。

### Broadcast 是例外

如果多个线程读取\*\*同一个 Shared Memory 地址\*\*，硬件可以 Broadcast；这不同于“不同地址恰好落到同一 Bank”。

### Padding 为什么有效

矩阵 Tile：

```
__shared__ float tile[32][32];
```

按列访问 `tile[lane][fixed_col]` 时，相邻行跨度 32 Word，所有 Lane 落到同一 Bank。改为：

```
__shared__ float tile[32][33];
```

行跨度变成 33 Word：

$$
(lane\times33)\bmod32=lane
$$

32 个 Lane 再次分散到 32 个 Bank。代价只是每行多一个 Padding Element。

Bank Width、Dual Port、Access Width 和架构细节会影响实际冲突行为，因此“Padding +1”是经典模式，不是对任意 dtype、任意 Tile 的数学定律。

## Tiling 与数据复用

若一个 Global Element 被 Block 中 R 次复用，直接读取的数据量约为：

$$
Bytes_{direct}=R\times Bytes_{element}
$$

先加载 Shared Memory：

$$
Bytes_{tiled}\approx Bytes_{element}+R\times Bytes_{shared-access}
$$

Shared Memory 仍有 Instruction、Barrier、Bank 和 Occupancy 成本。只有 Global Traffic 减少带来的收益大于这些成本，Tiling 才值得。

### Tile 太大的代价

- Shared Memory/Block 增加；
- Resident Block 数下降；
- Register Live Range 变长；
- Barrier 开销增加；
- Edge Handling 更复杂；
- Cache 本已足够时，手工 Tiling 可能重复优化。

因此 Tile Size 必须扫描，并结合 Resource Usage 与 Profiler。

## 矩阵转置：Memory 优化的标准样板

输入按 Row-major 存储。Naive Transpose 如果读取连续，写入通常跨大 Stride；若写入连续，读取又可能跨 Stride。

Shared Memory Tile 解决方式：

1. Warp 从输入连续读取到 Shared Tile；
2. Block 内同步；
1. 交换 Block 坐标，从 Shared Tile 按转置方向读取；
2. 向输出连续写入；
1. Tile 第二维加 1，消除列访问 Bank Conflict。

这展示了 Shared Memory 的本质：不是单纯“更快的存储”，而是一个可编程的数据布局转换器。

## Warp Shuffle 与 Register 交换

Warp Shuffle 允许 Lane 直接交换 Register Value：

```
unsigned mask = __activemask();
float from_left = __shfl_up_sync(mask, value, 1);
```

适合：

- Warp Reduction/Scan；
- 小范围邻居交换；
- 避免 Shared Memory + Barrier。

限制：

- 只在参与 Mask 的 Warp Lane 间；
- 不能替代 Block-wide 交换；
- Divergence 中 Mask 必须正确；
- Shuffle 增加 Register Pressure 时也可能影响 Occupancy。

## Cache、Read-only 与 Local Memory

### L1/L2 不是万能补救

Cache 能缓解重复或邻近访问，但不能保证随机大工作集获得高命中率。Coalescing 优化的是一次 Warp Request 的 Transaction 效率，Cache 优化的是跨时间的复用，二者是不同轴。

### Local Memory 也要合并

Thread-private Array、过多 Register 或动态索引可能落到 Local Memory。Local Memory 物理上在 Device Memory，并经过 Cache。相同 Warp 的线程访问各自对象的相同相对 Offset 时，编译器布局通常利于 Coalescing；随机 Local Array Index 仍可能很慢。

### Constant Memory

Warp 所有 Lane 读取同一个 Constant Address 时适合 Broadcast；如果 Lane 读取大量不同 Constant Address，访问可能序列化。它不是“所有只读数据的自动最快存储”。

## Ampere cp.async、Hopper TMA 与 Blackwell

### Ampere cp.async / LDGSTS

Compute Capability 8.0+ 提供 Global-to-Shared 的硬件异步拷贝路径，可：

- 直接从 Global Memory 搬到 Shared Memory；
- 避免传统 `Global → Register → Shared` 的中间 Register；
- 将下一 Tile 的搬运与当前 Tile 计算流水化；
- 支持 4/8/16-byte 粒度，16-byte 路径可使用特定 Cache 行为；
- 对 Alignment、Participation 和 Pipeline Wait 有严格要求。

RTX 3080/3090（ `sm_86` ）与 A100（ `sm_80` ）可以验证这一机制。它不是 Hopper 才加入。

### Hopper TMA

TMA 面向更大块的一维/多维 Tensor 搬运，用 Tensor Map 描述 Shape、Stride 与 Layout，由少量线程发起 Bulk Transfer，并与 Barrier/Pipeline 协作。TMA 与 `cp.async` 都是异步搬运，但粒度、描述方式、支持方向和硬件路径不同。

### Blackwell 与 Swizzle

更新架构进一步扩展 TMA、Cluster/Distributed Shared Memory 与 Layout Swizzle。Swizzle 可改变 Shared Memory 布局以规避 Bank Conflict，但必须按具体 Compute Capability、CUDA Toolkit 和 Kernel API 验证。

没有 Hopper/Blackwell 时，可以在 Level 0 模拟 Tile 和 Bank Mapping，也可以在 Ampere 运行 Padding/ `cp.async` ，但不能声称验证了 TMA 或 Blackwell Swizzle。

## 关键性能模型

### Requested 与 Actual Throughput

$$
Requested\ Bandwidth=\frac{Bytes_{useful}}{Time}
$$

$$
Actual\ Bandwidth=\frac{Bytes_{memory\ transactions}}{Time}
$$

$$
Load\ Efficiency=\frac{Requested}{Actual}
$$

若 Requested Bandwidth 很低且 Actual Bandwidth 接近硬件上限，说明大量带宽被 Overfetch 消耗；若两者都低，则可能是 Latency、Occupancy、Dependency 或 Launch 问题。

### Roofline

优化布局不会增加 FLOP，但会减少 Actual Byte：

$$
AI=\frac{FLOPs}{Bytes_{actual}}
$$

减少 Transaction、提高复用，相当于把 Kernel 在 Roofline 图上向右移动。

### Shared Memory Break-even

可用简化不等式判断：

$$
Saved\ Global\ Time > Shared\ Access + Barrier + Index + Occupancy\ Loss
$$

不是所有只读数据都应该先放 Shared；只读一次且本来已经 Coalesced 的数据，手工缓存常常无收益。

## 瓶颈分析方法

### 第一步：确认 Memory-bound

证据组合：

- DRAM Throughput 接近本机稳定上限；
- Arithmetic Intensity 低；
- SM Compute Pipeline 未饱和；
- 增加计算指令影响小，减少 Byte 明显变快；
- Roofline 落在 Bandwidth Roof 附近。

### 第二步：区分哪一级 Memory

```
Global/DRAM：Transaction、Sector、Cache Miss、Stride
L2/L1：Hit Rate、Working Set、Eviction、Replay
Shared：Bank Conflict、Barrier、Tile、Occupancy
Local：Register Spill、Thread-private Array
Host-Device：Pinned、PCIe、Chunk、Stream
```

### 第三步：查看 SASS 与 Resource

```
nvcc -O3 -lineinfo --resource-usage -arch=sm_86 memory_access_lab.cu -o memory_access_lab
cuobjdump --dump-sass ./memory_access_lab | less
```

确认：

- 是否生成预期宽度的 Load/Store；
- 是否出现 Local Load/Store；
- Register/Shared Memory 是否限制 Occupancy；
- 编译器是否消除了无用工作。

### 第四步：只改变一个访问变量

推荐 A/B 顺序：

1. 连续 vs Stride；
2. 对齐 vs Offset；
1. AoS vs SoA；
2. Naive Transpose vs Shared Tile；
1. Unpadded vs Padded Tile；
2. 同步 Global-to-Shared vs `cp.async` ；
1. 不同 Tile/Block Size。

## Level 0：CPU 回退——模拟 Transaction 与 Bank

### 完整代码：memory\_access\_sim.py

```
#!/usr/bin/env python3
"""教学模拟器：计算简化 Segment 和 Shared Bank 映射，不模拟真实 Cache/性能。"""

from __future__ import annotations

import argparse
import json
from collections import Counter

def global_transactions(warp_size: int, word_bytes: int, segment_bytes: int,
                        stride_words: int, offset_bytes: int) -> dict:
    addresses = [offset_bytes + lane * stride_words * word_bytes
                 for lane in range(warp_size)]
    segments = [address // segment_bytes for address in addresses]
    unique_segments = sorted(set(segments))
    requested = warp_size * word_bytes
    transferred = len(unique_segments) * segment_bytes
    return {
        "stride_words": stride_words,
        "offset_bytes": offset_bytes,
        "addresses_first_8": addresses[:8],
        "unique_segments": len(unique_segments),
        "requested_bytes": requested,
        "modeled_transferred_bytes": transferred,
        "modeled_efficiency": round(requested / transferred, 4),
    }

def shared_banks(warp_size: int, banks: int, word_stride: int,
                 same_address: bool = False) -> dict:
    words = [0 if same_address else lane * word_stride
             for lane in range(warp_size)]
    bank_ids = [word % banks for word in words]
    counts = Counter(bank_ids)
    if same_address:
        conflict_degree = 1
        behavior = "broadcast"
    else:
        conflict_degree = max(counts.values())
        behavior = "conflict-free" if conflict_degree == 1 else f"{conflict_degree}-way conflict"
    return {
        "word_stride": word_stride,
        "bank_ids": bank_ids,
        "bank_histogram": dict(sorted(counts.items())),
        "modeled_conflict_degree": conflict_degree,
        "behavior": behavior,
    }

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--warp-size", type=int, default=32)
    p.add_argument("--word-bytes", type=int, default=4)
    p.add_argument("--segment-bytes", type=int, default=32)
    args = p.parse_args()

    global_cases = [
        global_transactions(args.warp_size, args.word_bytes,
                            args.segment_bytes, stride, offset)
        for stride, offset in [(1, 0), (1, 4), (2, 0), (8, 0), (32, 0)]
    ]
    bank_cases = [
        shared_banks(args.warp_size, 32, stride)
        for stride in (1, 2, 16, 32, 33)
    ]
    bank_cases.append(shared_banks(args.warp_size, 32, 0, same_address=True))

    print(json.dumps({
        "global_cases": global_cases,
        "shared_bank_cases": bank_cases,
        "warning": (
            "Simplified model only. Real transactions depend on architecture, "
            "access width, active mask, cache, ECC and instruction semantics."
        ),
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
```

运行：

```
python3 memory_access_sim.py
```

### 预期现象

- `stride=1, offset=0` ：32 个 `float` 覆盖 4 个 32-byte Segment；
- `stride=1, offset=4` ：覆盖 5 个 Segment，简化效率 80%；
- Stride 增大时 Segment 数快速增加；
- Shared Word Stride 1 无冲突；
- Stride 32 是 32-way Conflict；
- Stride 33 重新无冲突；
- 32 个 Lane 读取同一地址是 Broadcast，不按 32-way Conflict 处理。

Level 0 只验证地址映射，不模拟 Cache Reuse、Dual-ported Bank、Scheduler 或真实 GB/s。

## Level 1：通用 CUDA Memory 实验

实验包含三组对照：

- 连续、Offset 和 Strided Global Load；
- Naive Matrix Transpose；
- Shared Tile Transpose：Unpadded 与 `+1` Padding。

### 完整代码：memory\_access\_lab.cu

```
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t error__ = (call);                                          \
        if (error__ != cudaSuccess) {                                          \
            std::cerr << "CUDA error " << __FILE__ << ":" << __LINE__          \
                      << " -> " << cudaGetErrorString(error__) << "\n";       \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

constexpr int TILE_DIM = 32;
constexpr int BLOCK_ROWS = 8;

__global__ void init_data(float* data, size_t n) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        data[i] = static_cast<float>(i % 1024) * 0.25f;
    }
}

__global__ void read_with_stride(const float* input, float* output,
                                 size_t logical_n, int stride_words,
                                 int offset_words) {
    const size_t grid_stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < logical_n; i += grid_stride) {
        output[i] = input[i * static_cast<size_t>(stride_words) + offset_words];
    }
}

__global__ void transpose_naive(const float* input, float* output,
                                int width, int height) {
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < width && y + j < height) {
            output[x * height + (y + j)] = input[(y + j) * width + x];
        }
    }
}

template <int PAD>
__global__ void transpose_tiled(const float* input, float* output,
                               int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + PAD];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < width && y + j < height) {
            tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * width + x];
        }
    }
    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < height && y + j < width) {
            output[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

template <typename Launch>
float time_kernel(Launch launch, int warmup, int repeats) {
    for (int i = 0; i < warmup; ++i) launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{}, stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i) launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return total_ms / repeats;
}

void verify_stride(const float* d_output, size_t n, int stride, int offset) {
    std::vector<float> host(n);
    CUDA_CHECK(cudaMemcpy(host.data(), d_output, n * sizeof(float),
                          cudaMemcpyDeviceToHost));
    double max_error = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const size_t source_index = i * static_cast<size_t>(stride) + offset;
        const float expected = static_cast<float>(source_index % 1024) * 0.25f;
        if (!std::isfinite(host[i])) {
            throw std::runtime_error("stride output contains non-finite value");
        }
        max_error = std::max(max_error,
                             std::abs(static_cast<double>(host[i] - expected)));
    }
    if (max_error > 1e-6) throw std::runtime_error("stride verification failed");
}

void verify_transpose(const float* d_output, int width, int height) {
    const size_t n = static_cast<size_t>(width) * height;
    std::vector<float> host(n);
    CUDA_CHECK(cudaMemcpy(host.data(), d_output, n * sizeof(float),
                          cudaMemcpyDeviceToHost));
    for (int out_row = 0; out_row < width; ++out_row) {
        for (int out_col = 0; out_col < height; ++out_col) {
            const size_t input_index = static_cast<size_t>(out_col) * width + out_row;
            const float expected = static_cast<float>(input_index % 1024) * 0.25f;
            const float got = host[static_cast<size_t>(out_row) * height + out_col];
            if (!std::isfinite(got) || std::abs(got - expected) > 1e-6f) {
                throw std::runtime_error("transpose verification failed");
            }
        }
    }
}

int main(int argc, char** argv) {
    try {
        const size_t logical_n = argc > 1 ? std::stoull(argv[1]) : (1ULL << 22);
        const int matrix_size = argc > 2 ? std::stoi(argv[2]) : 4096;
        const int repeats = argc > 3 ? std::stoi(argv[3]) : 50;
        if (logical_n == 0 || matrix_size <= 0 || repeats <= 0) {
            throw std::invalid_argument("arguments must be positive");
        }

        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        CUDA_CHECK(cudaSetDevice(0));
        std::cout << "device=" << prop.name
                  << " cc=" << prop.major << "." << prop.minor
                  << " SMs=" << prop.multiProcessorCount
                  << " memory_GiB=" << prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0
                  << "\n";

        constexpr int max_stride = 32;
        constexpr int max_offset = 1;
        const size_t input_n = logical_n * max_stride + max_offset;
        float *d_input = nullptr, *d_output = nullptr;
        CUDA_CHECK(cudaMalloc(&d_input, input_n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output, logical_n * sizeof(float)));

        const int block = 256;
        const int init_grid = std::min<int>(
            static_cast<int>((input_n + block - 1) / block),
            prop.multiProcessorCount * 32);
        init_data<<<init_grid, block>>>(d_input, input_n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::cout << std::fixed << std::setprecision(3);
        const int access_grid = std::min<int>(
            static_cast<int>((logical_n + block - 1) / block),
            prop.multiProcessorCount * 32);
        for (auto [stride, offset] : std::vector<std::pair<int, int>>{
                 {1, 0}, {1, 1}, {2, 0}, {8, 0}, {32, 0}}) {
            auto launch = [&] {
                read_with_stride<<<access_grid, block>>>(
                    d_input, d_output, logical_n, stride, offset);
            };
            const float ms = time_kernel(launch, 5, repeats);
            verify_stride(d_output, logical_n, stride, offset);
            const double requested_bytes = 2.0 * logical_n * sizeof(float);
            const double requested_gbs = requested_bytes / (ms / 1000.0) / 1e9;
            std::cout << "access stride=" << stride << " offset=" << offset
                      << " ms=" << ms
                      << " requested_GBps=" << requested_gbs << " PASS\n";
        }

        CUDA_CHECK(cudaFree(d_output));
        CUDA_CHECK(cudaFree(d_input));

        const int width = matrix_size;
        const int height = matrix_size;
        const size_t matrix_n = static_cast<size_t>(width) * height;
        const size_t matrix_bytes = matrix_n * sizeof(float);
        float *d_matrix_in = nullptr, *d_matrix_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_matrix_in, matrix_bytes));
        CUDA_CHECK(cudaMalloc(&d_matrix_out, matrix_bytes));
        const int matrix_init_grid = std::min<int>(
            static_cast<int>((matrix_n + block - 1) / block),
            prop.multiProcessorCount * 32);
        init_data<<<matrix_init_grid, block>>>(d_matrix_in, matrix_n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        dim3 threads(TILE_DIM, BLOCK_ROWS);
        dim3 grid((width + TILE_DIM - 1) / TILE_DIM,
                  (height + TILE_DIM - 1) / TILE_DIM);

        auto naive = [&] {
            transpose_naive<<<grid, threads>>>(d_matrix_in, d_matrix_out,
                                               width, height);
        };
        const float naive_ms = time_kernel(naive, 5, repeats);
        verify_transpose(d_matrix_out, width, height);

        auto tiled_no_pad = [&] {
            transpose_tiled<0><<<grid, threads>>>(d_matrix_in, d_matrix_out,
                                                  width, height);
        };
        const float no_pad_ms = time_kernel(tiled_no_pad, 5, repeats);
        verify_transpose(d_matrix_out, width, height);

        auto tiled_pad = [&] {
            transpose_tiled<1><<<grid, threads>>>(d_matrix_in, d_matrix_out,
                                                  width, height);
        };
        const float pad_ms = time_kernel(tiled_pad, 5, repeats);
        verify_transpose(d_matrix_out, width, height);

        const double transpose_payload = 2.0 * matrix_bytes;
        auto print_transpose = [&](const std::string& name, float ms) {
            std::cout << "transpose " << name << " ms=" << ms
                      << " requested_GBps="
                      << transpose_payload / (ms / 1000.0) / 1e9
                      << " PASS\n";
        };
        print_transpose("naive", naive_ms);
        print_transpose("tiled_no_pad", no_pad_ms);
        print_transpose("tiled_pad_33", pad_ms);
        std::cout << "speedup padded_vs_naive=" << naive_ms / pad_ms
                  << " padded_vs_no_pad=" << no_pad_ms / pad_ms << "\n";

        CUDA_CHECK(cudaFree(d_matrix_out));
        CUDA_CHECK(cudaFree(d_matrix_in));
        CUDA_CHECK(cudaDeviceReset());
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "fatal: " << e.what() << "\n";
        return 1;
    }
}
```

### 编译与运行

通用自动检测：

```
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')
nvcc -O3 -std=c++17 -lineinfo -arch="sm_${CC}" \
  memory_access_lab.cu -o memory_access_lab

./memory_access_lab
./memory_access_lab 4194304 4096 100
```

双 RTX 3080：

```
nvcc -O3 -std=c++17 -lineinfo -arch=sm_86 \
  memory_access_lab.cu -o memory_access_lab
CUDA_VISIBLE_DEVICES=0 ./memory_access_lab
CUDA_VISIBLE_DEVICES=1 ./memory_access_lab
```

程序默认最大 Global 输入约 512 MiB，矩阵输入与输出各约 64 MiB，适合 16\\~20GB 常见显卡。显存紧张时：

```
./memory_access_lab 1048576 2048 50
```

### 预期现象

- `stride=1, offset=0` 通常最快；
- Offset 1 可能因多一个 Segment 而变慢，但 Cache Reuse 会影响差异；
- Stride 2/8/32 的 Requested GB/s 通常逐步降低；
- Naive Transpose 的 Strided Store 通常明显较慢；
- Shared Tile 让 Global Read/Write 都连续；
- `tile[32][33]` 通常快于 `tile[32][32]` ，因为经典列访问冲突被消除；
- 所有 Case 必须显示 `PASS` 后才允许比较性能。

性能数字只代表本次 GPU、Clock、Driver、编译目标和矩阵尺寸，不应写成所有 GPU 的固定倍数。

## Nsight Compute 验证

先列出当前版本可用 Metric：

```
ncu --query-metrics | rg 'dram|sector|bank|shared'
```

采集 Source 与 Memory Workload：

```
ncu --set full --kernel-name regex:read_with_stride ./memory_access_lab 1048576 2048 10
ncu --set full --kernel-name regex:transpose_tiled ./memory_access_lab 1048576 4096 10
```

关注指标类别：

- DRAM/L2/L1 Bytes 与 Throughput；
- Global Load/Store Sector 数；
- Requested 与 Actual Traffic；
- Shared Load/Store Bank Conflict；
- Long Scoreboard Stall；
- Achieved Occupancy；
- Register 与 Shared Memory Resource。

Metric 名称会随 Nsight Compute 与架构变化，先 `--query-metrics` ，不要把旧版本名称硬编码进自动化脚本。

## Level 2：架构专项实验

### 支持矩阵

| 架构 | 示例 GPU | 可验证内容 | 不可等价替代 |
| --- | --- | --- | --- |
| Ampere | RTX 3080/3090、A100 | Coalescing、Bank、 `cp.async` | TMA、Blackwell Swizzle |
| Ada | RTX 4090、L40/L40S | 通用访问、 `cp.async` 、Ada Cache 行为 | Hopper TMA |
| Hopper | H100/H200 | TMA、Cluster、 `cp.async` | Blackwell 特定能力 |
| Blackwell | RTX 5090、B100/B200/GB200 | 对应 CC 的 TMA/Swizzle/新路径 | 数据中心能力不能由 5090 代表 |

### Ampere cp.async 路径

先确认：

```
nvidia-smi --query-gpu=name,compute_cap --format=csv
nvcc --version
```

使用 CUDA C++ `cuda::memcpy_async` /Pipeline API，或 NVIDIA CUDA Samples 中对应 Async Copy 示例，编译时指定：

```
nvcc -O3 -std=c++17 -lineinfo -arch=sm_86 async_copy_lab.cu -o async_copy_lab
```

实验必须比较：

- 同步 `Global → Register → Shared` ；
- 单 Stage Async Copy；
- Multi-stage Pipeline；
- 4/8/16-byte Copy 和 Alignment；
- 相同结果、相同 Tile 与相同计算量。

仅看到源码使用 `cuda::memcpy_async` 不证明生成了理想 LDGSTS 路径；需要检查 SASS/Profiler，并验证 Alignment 和 Pipeline Wait。

### Hopper TMA

H100/H200 上使用支持 `sm_90` 的 Toolkit 与 CUDA Samples/TMA 官方示例。必须记录 Tensor Map、Transfer Shape、Stride、Alignment、Barrier 与 Initiating Thread。TMA 常由一个 Elected Thread 发起大块搬运，不应让所有 Thread 重复发起。

### Blackwell

B100/B200/GB200 与 RTX 5090 同属 Blackwell，但 Compute Capability、Memory System 和数据中心 Fabric 能力不同。按实际 `compute_cap` 、Toolkit 文档和产品支持构建，不能因为 RTX 5090 能编译某条普通 CUDA 路径，就声称验证了 GB200 TMA/Swizzle 的系统效果。

## 结果分析

### 为什么 Stride 越大越慢

Requested Byte 不变，但每个 Warp 覆盖的 Segment 增多。实际 DRAM/L2 Transaction 增加，Cache Line 中很多 Byte 没被使用。Requested GB/s 下降不一定代表物理总线空闲，可能恰恰是总线被无效流量占满。

### 为什么 Offset 实测下降小于简化模型

相邻 Warp 可能复用前一个 Warp 多取回的 Cache Segment；Cache、Prefetch 与 Sector 合并会缓和纯 80% 模型。模型预测 Transaction 几何，Benchmark 测整个系统。

### 为什么 Padded Tile 会变快

Global Transaction 数可能基本相同，变化发生在 Shared Memory：列访问从多个 Lane 命中同一 Bank，变为各 Lane 命中不同 Bank。这个实验说明“DRAM 带宽正常”不代表 Kernel 没有 Memory 问题。

### 为什么 Shared Memory 优化可能变慢

如果原访问已经 Coalesced 且只读一次，引入 Tile 会增加：

- Shared Store/Load；
- `__syncthreads()` ；
- Index 指令；
- Shared Memory 占用；
- Edge Branch。

此时应删除 Shared Tile，而不是继续增大它。

## 优化前后对照

| 维度 | 低效访问 | 优化方向 | 验证指标 |
| --- | --- | --- | --- |
| Global Load | Warp 内大 Stride | 线程映射/布局连续化 | Sector、Load Efficiency |
| Alignment | 任意 Offset | 对齐基址和 Vector Tail | Transaction、Replay |
| Layout | 只读一字段却用 AoS | SoA/AoSoA | Requested/Actual Byte |
| Vectorization | 标量指令过多 | 对齐 `float2/float4` | SASS、Instruction 数 |
| Reuse | 同数据反复读 Global | Shared/Register Tile | DRAM Byte、Occupancy |
| Transpose | 连续读、跨步写 | Shared Tile 重排 | Global Store Sector |
| Shared | `tile[32][32]` 列访问 | `tile[32][33]` /Swizzle | Bank Conflict |
| Warp 交换 | Shared + Block Barrier | Shuffle | Shared Traffic、Latency |
| Pipeline | 先全搬完再计算 | `cp.async` /TMA 多 Stage | Exposed Memory Stall |

## 常见错误与排查

### 错误 1：把 Coalescing 说成“显存一次读取一整行”

Tensor Row 不是硬件概念。应分析 Warp 的地址集合覆盖多少对齐 Segment，以及实际请求/事务 Byte。

### 错误 2：相邻线程连续访问，却仍然慢

可能首地址 Misaligned、dtype 很宽、Cache Miss、Store 路径、Occupancy、依赖或已经达到 DRAM 上限。连续只是必要条件之一。

### 错误 3：强制转成 float4\* 后 Illegal Access

检查 16-byte Alignment、元素数是否为 4 的倍数、Tail 和 Storage Offset。不要对任意 View 直接 Reinterpret Cast。

### 错误 4：Padding 后结果错误

Shared Pitch 从 32 变 33 后，所有索引必须仍按逻辑 Tile 坐标访问；不要把 Padding 当成真实矩阵列。

### 错误 5：\_\_syncthreads() 放在非一致分支

Block 内线程无法一致到达 Barrier，可能 Hang。边界通常通过对加载/存储加条件，但 Barrier 放在一致控制流中。

### 错误 6：Shared Memory Bank 公式机械套到所有类型

64/128-bit Access、Vector Instruction、Transaction 拆分、架构 Bank 行为会改变结果。用简化模型提出假设，用 NCU 验证。

### 错误 7：cp.async 编译成功就宣布异步重叠

检查目标 CC、Alignment、生成指令、Commit/Wait、Stage 设计和 Timeline。API 可能回退或无法形成有效 Pipeline。

### 错误 8：提高 Global Load Efficiency 后端到端不变

Kernel 可能只占小部分时间，或瓶颈转到 Compute、Store、Launch、通信。用 Amdahl 和端到端指标解释。

### 错误 9：只看 Requested GB/s

Requested 低可能因 Overfetch；必须结合 Actual DRAM/L2 Byte。反之，Actual 很高可能是浪费，不等于有效 Goodput 高。

### 错误 10：手写 Tiled GEMM 与 cuBLAS 比峰值

教学 Kernel 缺少 Multi-stage Pipeline、Tensor Core MMA、Swizzle、Warp Specialization 和 Autotuning。生产优先成熟库，手写用于理解或填补算子空白。

### 错误 11：Register Array 变大后性能突然下降

可能 Spill 到 Local Memory。查看 `--resource-usage` 、Local Load/Store 与 Occupancy，不要只看源码中的“局部变量”。

### 错误 12：用一次热 Cache 结果代表流式 DRAM 性能

工作集太小或重复运行同一地址时，数据可能来自 L2。扩大工作集、改变访问、记录 Cache Metric 并区分 Cold/Warm。

## 面试题与答案

### 1\. 什么是 Memory Coalescing？

同一 Warp 的 Memory Request 被硬件合并为尽可能少的对齐 Transaction。目标是减少 Transaction 数和无用 Byte，而不是要求 C++ 数组必须有某种形式。

### 2\. 连续 32 个线程读取 32 个 float 是一次还是四次？

在 CC 6.0+ 的常用简化模型下，128 Byte 请求由四个 32-byte Transaction 服务。口语可说“合并访问”，但不应误称只有一个不可分割物理事务。

### 3\. Misalignment 为什么增加 Transaction？

地址范围跨过额外的对齐 Segment 边界。虽然请求 Byte 不变，覆盖的 Segment 数增加。

### 4\. AoS 和 SoA 怎么选择？

看同一 Warp 同一条指令实际读取哪些字段。只读单字段通常 SoA 更合并；每线程读取完整对象时，对齐紧凑的 AoS/AoSoA 可能更好。

### 5\. Vectorized Load 一定减少 DRAM Transaction 吗？

不一定。它常减少指令数、增加每线程搬运宽度；Transaction 仍由 Warp 地址覆盖和 Alignment 决定。

### 6\. Shared Memory 为什么会有 Bank Conflict？

Shared 被分成多个可并行 Bank。同一 Warp 的不同地址落到同一 Bank 时，请求需拆分；读取同一地址的 Broadcast 是例外。

### 7\. 为什么 tile\[32\]\[33\] 能消除经典列冲突？

行跨度从 32 Word 变为 33 Word，Bank Index 从总是相同变成随 Lane 递增，即 `(lane×33) mod 32=lane` 。

### 8\. Shared Memory 比 L1 快，是否所有数据都先放 Shared？

不是。Shared 增加 Copy、Barrier、Index 和 Resource Cost。只有能重排或复用数据、减少更昂贵访问时才有收益。

### 9\. cp.async 是哪代架构加入的？

Ampere 为 Global-to-Shared 异步拷贝提供硬件加速，Compute Capability 8.0+ 可用对应机制。Hopper TMA 是更高级的 Bulk/多维搬运，不是 `cp.async` 的起点。

### 10\. Requested Bandwidth 和 Actual Bandwidth 的区别？

Requested 按程序真正需要的 Payload 计算；Actual 按 Memory Transaction 搬运 Byte 计算。两者差距反映 Overfetch/效率。

### 11\. Warp Shuffle 能替代 Shared Memory 吗？

只适合 Warp 内 Register 交换。跨 Warp、Block-wide Tile 或需要大容量缓存时仍需 Shared Memory/其他机制。

### 12\. 如何证明 Kernel 受 Bank Conflict 影响？

保持 Global Traffic、计算和 Launch 不变，只改变 Shared Layout/Padding；用 NCU Bank Conflict Counter 和 Kernel 时间共同验证。

## 课后练习

1. 扩展 Level 0，模拟 8-byte/16-byte Access 跨 32-byte Segment 的情况。
2. 对 Offset 0\\~31 个 `float` 做扫描，画出 Requested GB/s 曲线。
1. 对 Stride 1/2/4/8/16/32/64 做 NCU Sector 统计。
2. 实现 AoS 与 SoA 的单字段求和，比较 Actual Byte。
1. 增加对齐 `float4` Copy，并正确处理非 4 倍数 Tail。
2. 将 Transpose Tile 改为 16、32，比较 Shared 使用与 Occupancy。
1. 删除 Padding，用 NCU 证明 Bank Conflict 增加。
2. 用 Warp Shuffle 实现 Warp Sum，与 Shared 版本比较。
1. 在 RTX 3080 上实现单 Stage `cuda::memcpy_async` ，检查 SASS。
2. 在支持 TMA 的 GPU 上复现 Bulk Copy，说明为什么不能由 RTX 3080 等价模拟。

## Checklist

### Global Memory

- 已按 Warp 分析地址集合，而不是只看数组维度。
- 已计算 Requested Byte、Transaction 与简化效率。
- 已检查 Alignment、Offset、Stride 和 Tail。
- 已区分 Coalescing 与 Cache Reuse。
- Vectorized Access 有正确 Alignment 和回退。

### 数据布局

- AoS/SoA 选择依据真实字段访问。
- 大 Stride 已通过线程映射或 Layout 调整。
- 必要时使用 AoSoA/Tile。
- 没有为追求连续访问破坏算法正确性。

### Shared Memory

- Shared Tile 确实减少 Global Traffic 或完成重排。
- Barrier 位于一致控制流。
- 已分析 Bank Mapping 与 Broadcast。
- Padding/Swizzle 由 Profiler 验证。
- Shared 使用未造成不可接受 Occupancy 下降。

### 测量

- 所有 Case 先验证正确性。
- 使用 CUDA Event、Warmup 和多轮测量。
- 区分 Requested 与 Actual Bandwidth。
- 工作集足够大，区分 L2 与 DRAM。
- 记录 GPU、CC、Driver、Toolkit、Clock 和编译目标。

### 架构边界

- RTX 3080/3090 与 A100 正确归入 Ampere。
- RTX 4090/L40S 正确归入 Ada。
- H100/H200 正确归入 Hopper。
- RTX 5090、B100/B200/GB200 正确归入 Blackwell。
- cp.async 正确标为 Ampere CC 8.0+ 能力。
- 没有用 Ampere 模拟 Hopper TMA 或 Blackwell Swizzle。