---
title: "第28课：MoE 推理与未来 AI Runtime"
source: "https://everythingai.top/learn/ai-systems-performance-engineering-text/lesson-28"
author:
published:
created: 2026-09-26
description: "面向 AI 工程学习者的专业课程学习空间。"
tags:
  - "clippings"
---
**导航**

本课梳理 MoE 推理的专家路由、负载均衡与通信挑战，并展望下一代 AI Runtime 的演进方向。

## 课程定位

Dense 模型对每个 token 都激活同一套参数。模型越大，每个 token 的计算和显存带宽成本通常越高。Mixture of Experts（MoE）换了一种扩展方式：把 FFN 拆成许多专家，由 Router 为每个 token 只选择少量专家。

这让模型可以拥有很大的总参数容量，同时把单 token 激活计算控制在较小范围。但“少算参数”不等于“系统天然更快”。所有专家权重仍要被存放；token 会被动态发送到不同 GPU；热门专家形成尾部慢 Rank；小批 Decode 还会把 GEMM 切得很碎。

因此，MoE 是最能体现 AI Systems Performance Engineering 的模型之一：算法稀疏性只有经过 Router、Dispatch、All-to-All、Grouped GEMM、Combine、负载均衡和 Runtime 调度的协同，才能变成实际速度。

本课也是 28 节正式课程的收束：从 MoE 出发，连接未来 AI Runtime 的关键方向——动态并行、计算通信融合、分层 KV、推测执行、解耦推理、拓扑感知和 SLO 驱动自治。

## 学习目标

完成本课后，你应能：

- 解释 Total Parameters 与 Activated Parameters 的区别；
- 推导 Top-K Router、容量因子、负载不均衡和通信量；
- 画出 MoE 的 Route→Dispatch→Expert Compute→Combine 数据流；
- 区分 TP、EP、ETP、DP/Attention DP 与 Wide-EP；
- 用每专家 token 数、最大/平均负载、All-to-All 带宽和 Grouped GEMM 效率定位瓶颈；
- 在 CPU 上模拟路由倾斜，在通用 PyTorch 环境比较逐 token 与分组专家执行；
- 正确判断双 RTX 3080、RTX 4090、H100/H200、RTX 5090 和 B200/GB200 的实验边界；
- 理解未来 Runtime 为什么必须从“执行模型”升级为“持续控制系统”。

## 前置知识

- Transformer FFN、Softmax、Top-K；
- Tensor/Data/Pipeline/Expert Parallel；
- NCCL Collective、All-to-All、NVLink/NVSwitch 与 RDMA；
- Continuous Batching、KV Cache、Speculative Decoding、Disaggregated Inference。

## 核心直觉：MoE 把算力问题变成了数据搬运与排队问题

Dense FFN 对所有 token 执行同一个函数：

$$
y=\operatorname{FFN}(x)
$$

MoE 有 E 个专家。Router 为 token x 计算分数 r(x)，选择 Top-K 专家集合 $\mathcal{T}(x)$ ：

$$
p(x)=\operatorname{softmax}(W_rx)
$$

$$
y=\sum_{e\in\mathcal{T}(x)}p_e(x)\operatorname{Expert}_e(x)
$$

如果 $E=64,K=2$ ，一个 token 理论上只执行 2 个专家，而不是 64 个。但 Runtime 必须先回答：这 2 个专家在哪张 GPU？要发送多少 token？各专家的批量有多大？最慢专家什么时候完成？如何把结果按原 token 顺序合并？