---
title: "第12课：GPU 性能分析方法"
source: "https://everythingai.top/learn/ai-systems-performance-engineering-text/lesson-12"
author:
published:
created: 2026-09-26
description: "面向 AI 工程学习者的专业课程学习空间。"
tags:
  - "clippings"
---
**导航**

本课建立从基准测量到 Nsight 分析的标准流程，把“感觉慢”转化为可验证的瓶颈假设。

## 课程定位

性能优化最危险的状态，不是不会写 CUDA，而是拿到一个“GPU 利用率 95%”就开始改 Kernel。这个数字可能只表示采样窗口内 GPU 经常有任务在运行，并不等于 Tensor Core 接近峰值、不等于显存带宽已打满，更不等于业务 Goodput 达标。

本课建立一套可重复的 GPU 性能分析流程：先定义目标和基线，再用时间线确认时间花在哪里，最后对少数热点 Kernel 做深度分析。核心工具链是：

```
业务指标 / SLO
      ↓
稳定、可复现的基准
      ↓
nvidia-smi / DCGM：长期趋势与粗粒度遥测
      ↓
Nsight Systems：端到端时间线，回答“时间去哪了”
      ↓
Nsight Compute：Kernel 计数器，回答“为什么慢”
      ↓
提出假设 → 只改一个变量 → 正确性与端到端复测
```

## 学习目标

完成本课后，你能够：

1. 建立带版本、输入、预热、重复次数和噪声统计的性能基线。
2. 正确区分 Host Wall Time、CUDA Event Time 与 Profiler Timeline。
1. 用 Nsight Systems 识别 CPU Launch、GPU Idle、Memcpy、同步、通信和 Kernel 热点。
2. 用 Nsight Compute 分析 Speed of Light、Roofline、Memory、Warp、Occupancy 和 Scheduler 指标。
1. 解释为什么 GPU Util、Occupancy、单个 Counter 都不能独立证明性能良好。
2. 使用 NVTX 把业务阶段映射到 CUDA 时间线。
1. 使用 Amdahl 定律判断某个局部优化是否值得做。
2. 避免异步计时、冷启动、Profiler Replay、频率波动和错误输入造成的伪结论。
1. 在 CPU、常见 NVIDIA GPU 和架构专项环境中执行分层实验。

## 前置知识

- 理解 Latency、Throughput、Goodput、P50/P99、MFU/HFU 与 Roofline。
- 理解 CUDA Stream、Kernel Launch、Event、同步和异步执行。
- 理解 SM、Warp、Global Memory、Shared Memory 与 Tensor Core。
- Level 0 只要求 Python 3.9+；Level 1 需要 CUDA 可用的 PyTorch。

## 核心直觉：Profiler 是显微镜，不是判决书

Profiler 提供证据，但证据必须放在正确的问题里解释。

假设一个训练 Step 需要 100 ms，其中：