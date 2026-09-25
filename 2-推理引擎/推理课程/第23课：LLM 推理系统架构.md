---
title: "第23课：LLM 推理系统架构"
source: "https://everythingai.top/learn/ai-systems-performance-engineering-text/lesson-23"
author:
published:
created: 2026-09-26
description: "面向 AI 工程学习者的专业课程学习空间。"
tags:
  - "clippings"
---
**导航**

本课拆解现代 LLM 推理系统的端到端架构，理解 Prefill、Decode、Batching、缓存与调度如何协同工作。

## 一、课程定位

大模型推理不是调用一次 `model.generate()` 那么简单。在线服务需要同时处理长短不同、到达时间不同、优先级不同的请求；还要在有限显存中保存模型权重和每个会话的 KV Cache，并在每个解码迭代重新决定哪些请求一起执行。

因此，LLM 推理系统的核心问题不是“单次矩阵乘法有多快”，而是：

本课建立端到端架构视角。后续课程会分别深入 Continuous Batching、KV Cache、推测解码、Prefill/Decode 解耦和 MoE；本课先回答它们在一套真实服务中处于什么位置、由谁管理、怎样互相影响。

## 二、学习目标

- 能画出一次 LLM 在线请求从接入到流式返回的完整链路。
- 区分 Gateway、Tokenizer、Admission Controller、Scheduler、KV Manager、Model Executor 和 Detokenizer 的职责。
- 理解 Prefill 与 Decode 的计算形态、性能指标和瓶颈差异。
- 掌握权重、KV Cache、Workspace、CUDA Graph Pool 和碎片的显存模型。
- 能用 TTFT、TPOT、ITL、Tokens/s、Goodput 和排队模型评价服务。
- 理解离线推理、在线推理、单体服务、分布式服务和解耦式服务的取舍。
- 完成无需 GPU 的容量规划实验，以及无需下载模型的 PyTorch Prefill/Decode 实验。
- 能为单卡、双卡和多节点环境选择数据并行、张量并行或请求路由策略。

## 三、前置知识

- Transformer Self-Attention、Causal Mask 和自回归生成。
- Latency、Throughput、P50/P99、TTFT、TPOT、Goodput。
- CUDA Stream、Kernel Launch、GPU Memory 与 Tensor Core。
- 基本的 HTTP/RPC、队列和并发概念。
- Python 与 PyTorch 基础。

## 四、核心直觉：推理系统是一台“有状态的 Token 工厂”

普通图像分类请求通常输入固定 Shape，执行一次前向计算后结束。LLM 请求不同：

1. Prompt 先经过一次 Prefill，生成第一个 Token 所需状态；
2. 系统保存所有层的 Key/Value；
1. 每生成一个新 Token，都要再次调度并执行一次 Decode；
2. 请求长度和结束时间事先并不完全确定；
1. 多个请求会在不同迭代加入或离开运行集合。

所以 LLM 服务不是“请求级 Batch”一次算完，而更像一个操作系统：

- 请求是进程；
- 每个 Decode Step 是一个时间片；
- KV Cache 是进程常驻内存；
- Scheduler 决定本轮运行谁；
- Token Budget 类似 CPU 时间与内存配额；
- Prefix Cache 类似可共享的只读页；
- 超时、取消和抢占决定无效工作能否及时停止。

优化对象由单个请求变成了整个在途请求集合。