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