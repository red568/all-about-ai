---
title: "第27课：Disaggregated Inference 架构"
source: "https://everythingai.top/learn/ai-systems-performance-engineering-text/lesson-27"
author:
published:
created: 2026-09-26
description: "面向 AI 工程学习者的专业课程学习空间。"
tags:
  - "clippings"
---
**导航**

本课解析 Prefill 与 Decode 分离架构，理解资源解耦、网络传输和调度带来的收益与代价。

## 课程定位

一条 LLM 请求看似只是“输入一段文本，再逐字输出”，但 GPU 实际经历的是两种性格完全不同的工作：

- Prefill：一次处理大量输入 token，矩阵较大，通常更偏计算密集，主要决定 TTFT；
- Decode：每轮只处理新 token，却反复读取整套模型权重，通常更偏显存带宽和串行延迟，主要决定 TPOT/ITL。

把两种负载放在同一批 GPU 上，部署简单，但长 Prompt 的 Prefill 可能打断正在流式输出的 Decode；同一套张量并行、批处理和扩缩容策略也必须同时迁就两个阶段。

Disaggregated Inference（解耦式推理）把 Prefill 与 Decode 放到独立 Worker 池：P 节点计算 Prompt 并生成 KV Cache，随后把 KV 状态传给 D 节点继续生成。它不是“多买一倍 GPU”这么简单，而是把计算、显存、网络、KV 生命周期和调度器重新组合成一套流水线。

本课要建立一个判断标准：

## 学习目标

完成本课后，你应能：

- 解释 Prefill 与 Decode 为什么适合不同资源和并行策略；
- 画出 Router、P Pool、KV Data Plane、D Pool 和控制面的完整架构；
- 计算 KV Cache 大小、传输时间和链路带宽下限；
- 用 TTFT、TPOT、Goodput、队列时间和 KV 传输时间定位瓶颈；
- 设计 $xP\times yD$ 的容量配比、路由、背压与故障恢复；
- 在纯 CPU 环境运行队列模拟，在常见 NVIDIA GPU 上测量 KV 搬运；
- 明确 PCIe、NVLink、RDMA/NIXL 的验证边界，不把模拟结果冒充集群实测。

## 前置知识

- LLM Prefill、Decode、KV Cache 与 PagedAttention；
- TTFT、TPOT/ITL、吞吐、P50/P99 与 Goodput；
- Tensor Parallel、Data Parallel、Continuous Batching；
- PCIe、NVLink、RDMA、GPUDirect RDMA 的基本概念。

## 核心直觉：让两种流水线分别做到擅长的事

### 共置式推理

```
Request -> [同一组 GPU]
           Prefill + Decode + Prefill + Decode ...
```

优点是没有跨实例 KV 传输，模型只部署一组，故障和调度逻辑简单。缺点是阶段互相干扰：一条长 Prompt 进入批次后，可能拉长其他请求下一 token 的等待时间；为了保 TPOT，只能限制 Prefill 或切成小块，又可能损失 TTFT 和 Prefill 吞吐。

### Prefill/Decode 解耦

```
控制面：注册、路由、负载、租约、故障恢复
                                           |
Client -> Gateway/Router -> Prefill Pool --+--> KV Data Plane --> Decode Pool -> Stream
                           计算 Prompt             传状态          逐 token 生成
```