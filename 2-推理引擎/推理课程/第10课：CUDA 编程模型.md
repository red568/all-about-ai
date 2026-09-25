---
title: "第10课：CUDA 编程模型"
source: "https://everythingai.top/learn/ai-systems-performance-engineering-text/lesson-10"
author:
published:
created: 2026-09-26
description: "面向 AI 工程学习者的专业课程学习空间。"
tags:
  - "clippings"
---
**导航**

本课建立 CUDA 编程模型的完整认识，掌握线程组织、内存空间、Stream、Event 与异步执行。

## 课程定位

PyTorch、Triton、cuBLAS 和 vLLM 最终都要把工作提交给 GPU。理解 CUDA 编程模型，不是为了把所有算子重写成 CUDA C++，而是为了看懂 Kernel 为什么这样划分线程、为什么一次 Launch 没有立刻完成、为什么 `cudaDeviceSynchronize()` 会让程序突然变慢，以及为什么同一段源码在不同 GPU 上会选择不同指令。

本课从 CPU 与 GPU 的异构协作讲起，建立 Grid、Block、Thread、Warp、Stream、Event、Memory Space、Synchronization、PTX 与 SASS 的完整心智模型。下一阶段再专门深入 Memory Access、Profiling、Fusion 和 Triton。

## 学习目标

完成本课后，你能够：

1. 解释 Host、Device、Kernel 与 CUDA Runtime 的关系。
2. 使用 Grid、Block、Thread 把一维和二维问题映射到 GPU。
1. 理解 Warp/SIMT、分支发散、Predication 与独立线程调度的边界。
2. 区分 Global、Shared、Local、Constant、Register 等 Memory Space。
1. 正确使用 Kernel Launch、Stream、Event 与同步 API。
2. 解释异步 API 为什么不等于一定并发或一定重叠。
1. 编写带错误检查、计时、正确性验证和 Block Size 扫描的 CUDA 程序。
2. 区分源码、PTX、Cubin/Fatbin、SASS 以及 Compute Capability。
1. 为 Ampere、Ada、Hopper、Blackwell 选择正确编译目标与回退路径。

## 前置知识

- 熟悉 C/C++ 数组、指针、函数和编译命令。
- 理解 GPU 的 SM、Warp、Register、Shared Memory 和显存。
- 理解 Latency、Throughput、Roofline 与有效带宽。
- Level 1 需要 NVIDIA GPU、Driver 和 CUDA Toolkit；Level 0 只需 Python。

## 核心直觉：CPU 发任务，GPU 批量执行

CUDA 是异构编程模型：

```
Host / CPU
  ├─ 分配 Host/Device Memory
  ├─ 准备数据
  ├─ 把工作加入 CUDA Stream
  ├─ 发起 Kernel Launch / Memory Copy
  └─ 在需要结果时同步

Device / GPU
  ├─ SM 接收 Thread Block
  ├─ Warp Scheduler 发射 Warp 指令
  ├─ 线程访问寄存器、Shared Memory、Global Memory
  └─ 完成后由 Event/同步点向 Host 暴露状态
```

CPU 不是逐个遥控 GPU Thread。Host 发出一个 Kernel Launch，描述“运行哪个函数、启动多少 Block、每个 Block 多少 Thread、使用哪个 Stream”。GPU Runtime 与硬件负责把 Block 调度到 SM 上。

因此，CUDA 优化有两条主线：