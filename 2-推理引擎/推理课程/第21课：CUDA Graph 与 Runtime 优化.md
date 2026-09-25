---
title: "第21课：CUDA Graph 与 Runtime 优化"
source: "https://everythingai.top/learn/ai-systems-performance-engineering-text/lesson-21"
author:
published:
created: 2026-09-26
description: "面向 AI 工程学习者的专业课程学习空间。"
tags:
  - "clippings"
---
**导航**

本课通过 CUDA Graph 与 Runtime 优化减少 CPU 提交和 Kernel Launch 开销，提升重复工作负载效率。

## 一、课程定位

前面的课程主要优化“GPU 做一件事有多快”：访存是否合并、Kernel 是否融合、Tensor Core 是否吃满。但真实 AI 工作负载还有另一类瓶颈：GPU 很快，CPU 却来不及把工作提交给它。

一次训练 Step 或一次 LLM Decode Step 可能包含数十到数百个短 Kernel。每个 Kernel 的计算只需几微秒，Python、Dispatcher、CUDA Runtime 和驱动却要反复完成参数准备、依赖检查与 Launch。GPU 时间线中便会出现一串“短 Kernel + 空洞”。

CUDA Graph 的核心价值是：把一段重复的 GPU 工作及其依赖捕获成图，实例化一次，之后用一次 Graph Launch 重放整段工作。它主要降低 Runtime 和 Launch 开销，不会让单个 Kernel 的数学计算自动变快。

## 二、学习目标

- 理解 CUDA Runtime、Stream、Event、Graph Node、Graph Executable 的关系。
- 区分显式建图和 Stream Capture 两种方式。
- 掌握 Capture、Instantiate、Replay、Update 的生命周期。
- 理解静态地址、静态 Shape、禁止 CPU 同步和私有内存池等约束。
- 正确测量 CPU Submit Time、GPU Execution Time、端到端延迟和盈亏平衡点。
- 使用 PyTorch `torch.cuda.CUDAGraph` 完成可运行实验。
- 使用原生 CUDA C++ 对比普通 Kernel Launch 与 Graph Replay。
- 能判断何时应使用 Graph、 `torch.compile` 、Kernel Fusion、Batching 或 Stream 并发。

## 三、前置知识

- CUDA Kernel、Grid、Block、Stream 和 Event 基础。
- PyTorch CUDA 异步执行与 `torch.cuda.synchronize()` 。
- 第 20 课的 `torch.compile` 、Graph Break 和稳态测量方法。
- 第 16～17 课的 NCCL、Collective 与计算通信重叠。

## 四、核心直觉：把逐张工单变成一张固定流程单

普通 Eager 执行像工厂每完成一道工序，都回办公室领取下一张工单：

```
CPU: Launch A ─ Launch B ─ Launch C ─ Launch D
         │          │          │          │
GPU:    Kernel A   Kernel B   Kernel C   Kernel D
```

CUDA Graph 则先记录完整工艺流程，后续只提交一次：

```
首次：Capture(A→B→C→D) → Instantiate
后续：Graph Launch ─────→ GPU 执行 A→B→C→D
```

它省下的是重复提交和调度成本。若 A、B、C、D 本身都是很长的大 GEMM，CPU 早已提前把任务排入队列，Graph 收益可能很小；若它们是大量极短 Kernel，Graph 更容易显著降低空洞。

因此先记住一句判断口诀：