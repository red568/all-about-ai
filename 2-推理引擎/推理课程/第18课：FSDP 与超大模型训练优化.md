---
title: "第18课：FSDP 与超大模型训练优化"
source: "https://everythingai.top/learn/ai-systems-performance-engineering-text/lesson-18"
author:
published:
created: 2026-09-26
description: "面向 AI 工程学习者的专业课程学习空间。"
tags:
  - "clippings"
---
**导航**

本课掌握 FSDP 的参数分片、通信与显存模型，构建可扩展的超大模型训练方案。

## 课程定位

DDP 解决的是“同一个模型如何用更多 GPU 更快训练”，FSDP 解决的是“单卡根本放不下的模型如何训练”。它不只是把权重切开，而是把参数、梯度和优化器状态都切到不同 rank，并在计算某一层前临时拼回所需参数。

这一课不把 FSDP 当成一个开关，而把它当成一套显存与通信调度系统。学完后，你应能回答三个工程问题：为什么分片后仍会 OOM、为什么 FSDP 可能比 DDP 慢，以及如何为自己的模型选择分片粒度、混合精度、激活重计算和 CPU Offload。

截至 2026-08-05，PyTorch 官方主线已经把基于 DTensor、逐参数分片的 `fully_shard` 称为 FSDP2；旧的 `FullyShardedDataParallel` 通常称为 FSDP1。本文的新项目优先采用 FSDP2，同时讲清 FSDP1 配置的对应关系。

## 学习目标

完成本课后，你将能够：

1. 从训练状态的组成推导 DDP、ZeRO 与 FSDP 的单卡显存。
2. 解释 FSDP 的 All-Gather、Reduce-Scatter、Reshard 生命周期。
1. 区分 FSDP1 的 FlatParameter 与 FSDP2 的 DTensor 逐参数分片。
2. 选择合理的 Transformer Block 分片边界。
1. 正确组合混合精度、激活重计算、梯度累积和 CPU Offload。
2. 用峰值显存、step time、tokens/s 和通信暴露时间评价优化，而不是只看 `nvidia-smi` 。
1. 在双 RTX 3080 或其他双卡环境中完成 DDP/FSDP2 对照实验。

## 前置知识

- 理解参数、梯度、优化器状态和激活值。
- 了解 DDP、All-Reduce、All-Gather、Reduce-Scatter。
- 能使用 `torchrun` 启动多进程训练。
- 建议先完成第 15～17 课。

## 一、核心直觉：分片保存，按需借用

假设有四张 GPU 和四层模型。DDP 的做法是每张 GPU 都放完整四层，只给每张卡不同数据：

```
GPU0: L0 L1 L2 L3 + 全部梯度 + 全部优化器状态
GPU1: L0 L1 L2 L3 + 全部梯度 + 全部优化器状态
GPU2: L0 L1 L2 L3 + 全部梯度 + 全部优化器状态
GPU3: L0 L1 L2 L3 + 全部梯度 + 全部优化器状态
```

FSDP 的静止状态更像仓库分区：

```
GPU0: 每个参数的第 0 片 + 对应梯度/优化器状态
GPU1: 每个参数的第 1 片 + 对应梯度/优化器状态
GPU2: 每个参数的第 2 片 + 对应梯度/优化器状态
GPU3: 每个参数的第 3 片 + 对应梯度/优化器状态
```