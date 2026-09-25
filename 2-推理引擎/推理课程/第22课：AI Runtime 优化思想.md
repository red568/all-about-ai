---
title: "第22课：AI Runtime 优化思想"
source: "https://everythingai.top/learn/ai-systems-performance-engineering-text/lesson-22"
author:
published:
created: 2026-09-26
description: "面向 AI 工程学习者的专业课程学习空间。"
tags:
  - "clippings"
---
**导航**

本课从调度、内存、执行图与设备协同角度理解 AI Runtime，建立系统级优化方法。

## 一、课程定位

高性能 Kernel 并不等于高性能系统。一个算子可以达到很高的 Tensor Core 利用率，但用户请求仍可能在队列里等待；GPU 可以拥有很高的瞬时利用率，但大量工作可能是 Padding、重复计算或最终超时的无效请求。

AI Runtime 是模型与硬件之间的“现场总指挥”。它接收工作，管理状态和内存，选择执行计划，把请求组合成 Batch，把任务投递到设备，并在超载时进行限流、降级和回退。

本课不局限于某个推理框架，而是建立一套通用于训练 Runtime、推理 Runtime、在线服务和本地多 GPU 系统的优化思想：

## 二、学习目标

- 能画出 AI Runtime 的控制面、数据面和执行面。
- 理解 Scheduler、Batcher、Memory Manager、Executor、Compiler Cache 的职责。
- 掌握 Hot Path/Cold Path、异步流水线、Shape Bucket 和内存复用思想。
- 用排队模型解释吞吐、尾延迟、并发和背压之间的关系。
- 建立端到端性能预算，而不是只优化 GPU Kernel。
- 能设计 Admission Control、超时、取消、优先级和降级策略。
- 完成 CPU 调度模拟和通用 NVIDIA GPU Runtime 实验。
- 能判断下一步应优化 Scheduler、Batch、内存、Compiler、Kernel、通信还是拓扑。

## 三、前置知识

- Latency、Throughput、Goodput、Little 定律与 P50/P99。
- CUDA Stream、异步执行、Kernel Launch 和 CUDA Graph。
- `torch.compile` 、Shape 专门化和重编译。
- GPU Memory、Pinned Memory、内存池与数据搬运。
- 分布式通信、NCCL 与计算通信重叠。

## 四、核心直觉：Runtime 优化的是“流”，不是单个点

把 AI 系统想象成机场：

- 请求是乘客；
- Scheduler 是塔台；
- Batch 是同一班飞机；
- GPU 是跑道与飞机；
- KV Cache/Activation/Workspace 是登机口和行李位；
- CUDA Stream 是不同作业通道；
- Graph/Compiled Plan 是预先批准的固定航线；
- Backpressure 是限制进入候机楼的人数。

只把飞机发动机调快，并不能解决登机口拥堵、跑道冲突和乘客错过转机。Runtime 优化要关注请求从进入到完成的整条路径：

```
接入 → 排队 → 预处理 → 调度/组批 → H2D → 计算/通信
    → 后处理 → D2H/流式输出 → 计量与回收
```

任何一段变慢，都会通过队列放大成尾延迟。

## 五、AI Runtime 的分层架构

### 5.1 控制面、数据面和执行面

```
┌──────────────────────── 控制面 ────────────────────────┐
│ 模型版本、配置、路由、配额、扩缩容、健康检查、回滚      │
└────────────────────────────────────────────────────────┘
                           │
┌──────────────────────── 数据面 ────────────────────────┐
│ Gateway → Queue → Admission → Scheduler → Batcher      │
│                   ↕ 状态/KV/缓存 ↕                      │
└────────────────────────────────────────────────────────┘
                           │
┌──────────────────────── 执行面 ────────────────────────┐
│ Executor → Compiler/Plan Cache → Memory Pool            │
│          → CUDA Stream/Graph → Kernel/Library/NCCL      │
└────────────────────────────────────────────────────────┘
                           │
                   GPU / CPU / 网络 / 存储
```