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