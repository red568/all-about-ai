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