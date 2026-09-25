---
title: "第25课：KV Cache 性能优化"
source: "https://everythingai.top/learn/ai-systems-performance-engineering-text/lesson-25"
author:
published:
created: 2026-09-26
description: "面向 AI 工程学习者的专业课程学习空间。"
tags:
  - "clippings"
---
**导航**

本课从显存占用、访问效率和分页管理出发优化 KV Cache，提高长上下文与高并发推理能力。

## 一、课程定位

自回归 LLM 每生成一个 Token，都需要关注之前的 Token。如果每一步都重新计算全部历史 Key 和 Value，生成第 t 个 Token 时会重复执行大量已经完成的工作。KV Cache 保存各层历史 Attention 的 Key/Value，让 Decode 只计算新 Token 的投影，再读取历史缓存完成 Attention。

它用显存换计算，却很快成为服务容量和 Decode 带宽的核心约束：

- 上下文越长，KV 线性增长；
- 并发越高，在途 Token 总数越多；
- Decode 每步都要读取历史 KV；
- 请求长度动态变化，连续预留会产生碎片；
- Prefix 复用、Offload 和解耦推理又把它变成跨请求、跨设备的数据资产。

本课从“为什么缓存”逐步走到容量模型、Paged KV、Prefix Cache、量化、Offload、驱逐、分布式传输和安全边界。

## 二、学习目标

- 能从 Attention 计算推导 KV Cache 的作用和容量公式。
- 区分 MHA、GQA、MQA、MLA 对 KV 容量和带宽的影响。
- 掌握连续预留、Paged KV、Block Table 和内部/外部碎片。
- 理解 Prefix Cache 的命中条件、收益、驱逐和多租户安全。
- 能评价 KV 低精度、滑动窗口、Offload、重算和跨节点传输。
- 建立 KV 容量、带宽、命中率和 Goodput 的统一模型。
- 完成 CPU 容量/碎片实验和通用 NVIDIA GPU 的追加写入实验。
- 能用真实流量诊断 OOM、Preemption、Cache Thrash 和 TPOT 上升。

## 三、前置知识

- Transformer Self-Attention、Causal Mask、MHA/GQA/MQA。
- LLM Prefill、Decode、Continuous Batching 和 Scheduler。
- GPU HBM、PCIe、NVLink、Pinned Memory 与显存分配器。
- TTFT、TPOT、ITL、Tokens/s 和 Goodput。

## 四、核心直觉：KV Cache 是“Attention 的历史索引”

单层 Self-Attention 计算：

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

$$
Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{d}}+Mask\right)V
$$

在 Decode 第 t 步，历史 Token 的 $K_1\ldots K_{t-1}$ 和 $V_1\ldots V_{t-1}$ 不会改变。只需计算新 Token 的 $Q_t$,$K_t$,$V_t$ ，把 $K_t$,$V_t$ 追加进缓存，再执行：

$$
O_t=softmax\left(\frac{Q_t[K_1,\ldots,K_t]^T}{\sqrt{d}}\right)[V_1,\ldots,V_t]
$$

因此 KV Cache 消除了历史 K/V 投影的重复计算，但没有消除对历史 KV 的读取。上下文越长，单步读取量越大；KV 优化既是容量问题，也是 HBM 带宽问题。