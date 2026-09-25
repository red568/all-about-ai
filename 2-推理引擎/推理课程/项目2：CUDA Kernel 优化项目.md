---
title: "项目2：CUDA Kernel 优化项目"
source: "https://everythingai.top/learn/ai-systems-performance-engineering-text/project-02"
author:
published:
created: 2026-09-26
description: "面向 AI 工程学习者的专业课程学习空间。"
tags:
  - "clippings"
---
**导航**

本项目通过 CUDA Kernel 基线、剖析、改写和回归验证，完成一次端到端算子优化。

## 项目定位

本项目要求你从一个“结果正确但性能很差”的 CUDA Reduction Kernel 出发，用 Profile-Driven 方法完成两轮优化：

```
Baseline：每个元素执行一次全局 atomicAdd
    ↓ 消除大部分全局原子争用
V1：线程局部累加 + Shared Memory Block Reduction + 每 Block 一次 atomicAdd
    ↓ 减少 Shared Memory 访问与同步
V2：线程局部累加 + Warp Shuffle + 每 Block 一次 atomicAdd
```

最终交付的不只是“更快的代码”，而是一套完整证据：相同输入、相同正确性标准、稳定计时、不同规模扫描、Nsight Systems 时间线、Nsight Compute Kernel 指标，以及清楚的结论边界。

本项目适配常见 NVIDIA GPU。Level 0 不需要 CUDA；Level 1 使用标准 CUDA C++，适合 Ampere RTX 3080/3090、A100，Ada RTX 4090/L40/L40S，Hopper H100/H200，以及工具链支持的 Blackwell RTX 5090、B100/B200/GB200。Level 2 的架构专项实验必须先检测能力，不允许把低代 GPU 的模拟结果写成 FP8、FP4、TMA 或 Cluster 的硬件实测。

## 学习目标

完成项目后，你应该能够：

- 从算法工作量、内存流量和同步次数解释 Kernel 为什么慢；
- 正确使用 Grid-Stride Loop、Shared Memory、 `__syncthreads()` 和 Warp Shuffle；
- 区分全局原子争用、Shared Memory Bank Conflict、Occupancy 与指令依赖；
- 使用 CUDA Event 测量设备执行时间；
- 用 Nsight Systems 找到 Kernel 边界，用 Nsight Compute 验证优化原因；
- 在改变实现后同时验证正确性、P50/P95、跨尺寸稳定性与架构可移植性；
- 避免用峰值带宽、Occupancy 或一次运行结果替代真实证据。

## 前置知识

- 已完成第 10～13 课；
- 理解 Thread、Warp、Block、Grid 与 Grid-Stride Loop；
- 理解 Global Memory、Shared Memory、原子操作和同步；
- Ubuntu 或其他支持 CUDA Toolkit 的系统；
- 可选 NVIDIA GPU；没有 GPU 时先完成 Level 0。

## 项目交付物

```
cuda-kernel-project/
├── reduction_cost_model.py
├── reduction_benchmark.cu
├── reports/
│   ├── environment.txt
│   ├── baseline.csv
│   ├── size_sweep.csv
│   ├── reduction_nsys.nsys-rep
│   └── reduction_ncu.ncu-rep
└── analysis.md
```

`analysis.md` 至少回答：

1. Baseline 最主要的瓶颈是什么？
2. V1 消除了什么成本，又引入了什么成本？
1. V2 为什么可能继续变快，也可能没有明显收益？
2. 哪些结论来自实测，哪些只是模型推断？
1. 结论适用于哪些输入规模、Block Size 与 GPU？