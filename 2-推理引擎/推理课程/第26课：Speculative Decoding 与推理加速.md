---
title: "第26课：Speculative Decoding 与推理加速"
source: "https://everythingai.top/learn/ai-systems-performance-engineering-text/lesson-26"
author:
published:
created: 2026-09-26
description: "面向 AI 工程学习者的专业课程学习空间。"
tags:
  - "clippings"
---
**导航**

本课讲清 Speculative Decoding 的验证与接受机制，掌握在保证输出质量前提下加速推理的方法。

## 课程定位

大模型 Decode 阶段有一个近乎反直觉的特点：每一步只生成一个 token，矩阵规模很小，却要把整套模型权重从显存读一遍。GPU 往往不是算不动，而是每次只干了一点活就必须等待下一步。

Speculative Decoding（推测解码）不改变目标模型，而是让一个更便宜的“草稿器”先猜若干 token，再让目标模型一次并行验证。猜得准，目标模型一次前向就提交多个 token；猜错，只丢弃错误后缀并按目标模型修正。它优化的是串行轮数，不是简单地把模型量化或缩小。

本课从正确性、性能模型、系统实现到可运行实验，回答四个问题：

1. 为什么一次验证多个 token 可能比逐 token 解码快？
2. 为什么“草稿猜错”仍能保持目标模型分布？
1. 何时推测解码会加速，何时反而变慢？
2. 如何在消费级 GPU、数据中心 GPU 和真实推理框架中测量它？

## 学习目标

完成本课后，你应能：

- 区分草稿、验证、接受、拒绝、修正和 KV Cache 回滚；
- 推导平均每轮提交 token 数和粗略加速比；
- 理解严格拒绝采样为何能保持目标分布；
- 比较 Draft/Target、N-gram、Medusa、EAGLE、MTP 等路线；
- 用 acceptance rate、accepted length、draft/verify latency 和 TPOT 判断瓶颈；
- 在 CPU 上做容量模拟，在任意 PyTorch CPU/CUDA 环境做验证批处理实验；
- 避免把论文中的单点速度或某款 GPU 结果写成普遍结论。

## 前置知识

- 自回归生成、logits、softmax、temperature、top-p；
- Prefill 与 Decode 的区别；
- KV Cache、Continuous Batching；
- GPU kernel launch、显存带宽和小 batch 利用率。

## 核心直觉：把时间上的串行改成一次空间并行

普通 Decode 生成 4 个 token，需要目标模型连续运行 4 次：

```
Target(x) -> t1 -> Target(x,t1) -> t2 -> Target(...) -> t3 -> Target(...) -> t4
```

推测解码先让便宜的草稿器提出 `d1,d2,d3,d4` ，目标模型用因果掩码一次计算这 4 个位置的分布：

```
Draft:  d1 -> d2 -> d3 -> d4
Target: [并行验证 d1,d2,d3,d4]
Result: 接受最长正确前缀 + 一个目标模型修正 token
```

目标模型仍然要读权重，但一次读权重服务了多个候选位置。小 batch Decode 原本常受显存带宽和串行依赖限制，验证阶段把更多算术工作塞进同一次前向，可能提高算术强度与 GPU 利用率。