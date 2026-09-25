---
title: "第13课：Kernel Fusion 与高性能算子开发"
source: "https://everythingai.top/learn/ai-systems-performance-engineering-text/lesson-13"
author:
published:
created: 2026-09-26
description: "面向 AI 工程学习者的专业课程学习空间。"
tags:
  - "clippings"
---
**导航**

本课从中间张量与内存流量出发理解 Kernel Fusion，并掌握高性能自定义算子的设计与验证方法。

## 课程定位

很多 AI Kernel 的算术并不复杂：乘法、加法、激活、缩放、Dropout、Mask、Residual。问题在于，如果每一步都成为独立 Kernel，中间结果就要反复写入 Global Memory、再读回来，同时 CPU 还要不断提交 Kernel Launch。

Kernel Fusion 的核心并不是“把代码写进同一个函数”，而是让 Producer 产生的数据尽量停留在 Register 或 Shared Memory 中，直接交给 Consumer，而不落到显存中的中间 Tensor。

本课从可计算的显存流量模型出发，讲清 Fusion 为什么有效、什么时候会适得其反，以及如何完成一个真正可测量的 CUDA 融合算子。第 14 课才会系统学习 Triton；本课使用原生 CUDA C++ 和 PyTorch Extension，建立底层直觉。

## 学习目标

完成本课后，你能够：

1. 区分 Launch Fusion、Memory-Traffic Fusion、Epilogue Fusion 与 Algorithmic Fusion。
2. 计算 Elementwise 算子链融合前后的理论 Global Memory Traffic。
1. 建立包含 Launch、Memory、Compute 和同步开销的性能模型。
2. 判断一组算子是否适合 Vertical Fusion 或 Horizontal Fusion。
1. 识别 Fusion 导致的 Register Pressure、Occupancy 下降、Spill 和并行度损失。
2. 用纯 Python 实验观察中间列表带来的时间和峰值内存成本。
1. 编写并运行 PyTorch CUDA Extension，对比三个 Kernel 与一个融合 Kernel。
2. 用 Nsight Systems 验证 Kernel 数，用 Nsight Compute 验证 DRAM Traffic 和 Occupancy。
1. 建立高性能自定义算子的正确性、数值、性能和可维护性验收流程。

## 前置知识

- 理解 CUDA Grid、Block、Thread、Warp、Stream 和异步 Launch。
- 理解 Global Memory、Register、Shared Memory 与 Memory Coalescing。
- 理解 Effective Bandwidth、Arithmetic Intensity、Roofline 和 Amdahl 定律。
- 会使用 CUDA Event、Nsight Systems 与 Nsight Compute。
- Level 0 仅需 Python；Level 1 需要 CUDA Toolkit、PyTorch 和 C++ 编译器。

## 核心直觉：中间 Tensor 是一张昂贵的“交接单”

考虑下面的表达式：

```
tmp1 = x * y
tmp2 = tmp1 + bias
out = relu(tmp2)
```

从数学上看，它只是：

$$
out_i=\max(x_i y_i+bias,0)
$$

但在 Eager 执行中，它可能形成三个独立 Kernel：

```
Kernel 1: 读 x、y → 写 tmp1
Kernel 2: 读 tmp1 → 写 tmp2
Kernel 3: 读 tmp2 → 写 out
```

`tmp1` 和 `tmp2` 不是最终结果，却完整走了两次 Global Memory Store 和两次后续 Load。融合后：

```
Fused Kernel: 读 x、y → 在 Register 中乘、加、ReLU → 写 out
```

每个线程加载自己的 `x_i` 、 `y_i` ，中间值停留在 Register，最终只写一次 `out_i` 。这就是最典型的 Producer-Consumer Fusion。

## Fusion 的两大收益

### 收益一：减少 Kernel Launch

设一次 Launch 的有效开销为 $t_{launch}$ ，原来有 K 个 Kernel，融合后有 F 个：

$$
\Delta T_{launch}\approx(K-F)t_{launch}
$$

真实 Launch Cost 受 CPU、Driver、Python/Framework、CUDA Graph、Profiler 和系统负载影响，不能把某个固定微秒数写成通用常量。

对大 GEMM，Launch Cost 可能几乎可以忽略；对数百个只有几微秒的小 Pointwise Kernel，它可能成为主瓶颈。

### 收益二：减少 Global Memory Round Trip

以 FP32 的 `relu(x * y + bias)` 为例，不考虑 Cache，按有用字节估算：

| 阶段 | 读取 | 写入 | 每元素流量 |
| --- | --- | --- | --- |
| Multiply | `x,y` ：8 B | `tmp1` ：4 B | 12 B |
| Bias Add | `tmp1` ：4 B | `tmp2` ：4 B | 8 B |
| ReLU | `tmp2` ：4 B | `out` ：4 B | 8 B |
| 未融合合计 |  |  | 28 B |
| 融合 Kernel | `x,y` ：8 B | `out` ：4 B | 12 B |

理想流量缩减比：

$$
R_{traffic}=\frac{28}{12}\approx2.33
$$

这不是“必然加速 2.33 倍”。Cache 可能吸收部分中间流量；Kernel 还可能受 Launch、Compute、Register、Occupancy 和其他系统开销影响。