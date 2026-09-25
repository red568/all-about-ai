---
title: "项目1：GPU 性能分析实验"
source: "https://everythingai.top/learn/ai-systems-performance-engineering-text/project-01"
author:
published:
created: 2026-09-26
description: "面向 AI 工程学习者的专业课程学习空间。"
tags:
  - "clippings"
---
**导航**

本项目使用真实分析流程定位 GPU 性能瓶颈，并输出可复现的测量结果与优化报告。

## 项目定位

这个项目把前 28 课里最重要的一条方法论真正跑通：\*\*不要凭 GPU 利用率猜瓶颈，要用可复现基准、系统时间线和 Kernel 指标建立证据链。\*\*

你将从一个同时包含“小算子启动开销、显存流量和矩阵乘法”的 PyTorch 工作负载出发，依次完成：

1. 固化环境与工作负载；
2. 建立无 Profiler 的墙钟基线；
1. 用 PyTorch Profiler 找到高开销算子；
2. 用 Nsight Systems 判断 CPU、CUDA API、Memcpy 和 GPU Kernel 的时间关系；
1. 用 Nsight Compute 深入一个 Kernel，判断它受计算、带宽、延迟还是资源限制；
2. 只修改一个变量，验证正确性并复测；
1. 输出一份别人可以复现、审阅和继续优化的性能报告。

本项目不要求特定 GPU。没有 NVIDIA GPU 时可以完整完成 Level 0，并在 CPU 上完成 Level 1 的方法训练；有常见 NVIDIA GPU 时完成 CUDA 路径；Nsight Compute 硬件计数器不可用时，保留 PyTorch Profiler 与 Nsight Systems 证据，不伪造 Kernel 级结论。

## 学习目标

完成项目后，你应该能够：

- 区分基准测试、系统级 Profiling 与 Kernel 级 Profiling；
- 正确测量异步 CUDA 工作负载，不把 CPU 提交时间当成 GPU 执行时间；
- 根据时间线识别 Launch-bound、Memory-bound、Compute-bound 和同步等待；
- 使用算术强度与 Roofline 为优化方向设定上界；
- 用 PyTorch Profiler、NVTX、Nsight Systems 和 Nsight Compute 形成逐层下钻的证据链；
- 避免“同时改很多变量”“只跑一次”“只看平均值”等常见实验错误；
- 交付包含环境、原始数据、报告、正确性检查和结论边界的性能分析包。

## 前置知识

- 能运行 Python 与基础 Bash 命令；
- 理解 Latency、Throughput、P50/P95/P99；
- 理解 GPU Kernel、CUDA Stream、异步执行和显存层级；
- 理解算术强度、Roofline、Occupancy 的基本含义；
- 可选：已安装 PyTorch、CUDA Toolkit、Nsight Systems、Nsight Compute。

## 最终交付物

建议建立如下目录：

```
gpu-profiling-project/
├── roofline_planner.py
├── gpu_profiling_lab.py
├── reports/
│   ├── environment.txt
│   ├── roofline.json
│   ├── benchmark.json
│   ├── pytorch_trace.json
│   ├── nsys_baseline.nsys-rep
│   └── ncu_matmul.ncu-rep
└── analysis.md
```

并在 `analysis.md` 中回答四个问题：