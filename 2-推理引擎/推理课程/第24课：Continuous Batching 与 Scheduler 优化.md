---
title: "第24课：Continuous Batching 与 Scheduler 优化"
source: "https://everythingai.top/learn/ai-systems-performance-engineering-text/lesson-24"
author:
published:
created: 2026-09-26
description: "面向 AI 工程学习者的专业课程学习空间。"
tags:
  - "clippings"
---
**导航**

本课深入 Continuous Batching 与 Scheduler 的工作机制，学会在吞吐、延迟和公平性之间进行权衡。

## 一、课程定位

传统 Batch 推理像一辆“坐满才发车、全员到终点才返程”的班车。大模型在线请求却持续到达，Prompt 与输出长度差异很大：短请求早已结束，长请求仍在逐 Token 生成。如果 Batch 必须等最长请求完成，短请求释放出的计算槽位就会一直空着。

Continuous Batching，也称 In-flight Batching 或 Iteration-level Batching，把调度边界从“整个请求”缩小到“一个推理迭代”。每轮 Decode 后，已结束或已取消的请求立即退出，新请求可以在后续迭代进入；Scheduler 同时协调 Prefill、Decode、KV Cache 和 Token Budget。

本课的重点不是背诵某个框架参数，而是建立一套可迁移的调度模型：

## 二、学习目标

- 区分 Static、Dynamic 与 Continuous Batching。
- 理解请求级调度与迭代级调度的本质差异。
- 掌握 `max_num_seqs` 、 `max_num_batched_tokens` 、KV Block 和 Deadline 的联合约束。
- 理解 Prefill 优先、Decode 优先、Chunked Prefill、公平性与吞吐的取舍。
- 用队头阻塞、Padding Waste、排队稳定性和 Goodput 分析调度器。
- 设计取消、抢占、Backpressure、优先级和多租户隔离机制。
- 完成纯 Python 离散事件模拟，以及通用 NVIDIA GPU 上的连续 Decode 组批实验。
- 使用真实长度分布和开环负载调优 vLLM 等推理服务。

## 三、前置知识

- LLM 推理的 Prefill、Decode、KV Cache 和流式输出。
- TTFT、TPOT、ITL、Requests/s、Tokens/s 和 Goodput。
- Batch、Padding、CUDA Kernel Launch 与 GPU Memory。
- 基本队列、并发和 Little 定律。

## 四、核心直觉：从“整车发车”变成“每站换乘”

设一个静态 Batch 中四个请求分别需要生成 8、16、32、64 个 Token。若 Kernel 使用固定 Batch Width，前 8 步四个槽位都有效；之后请求陆续完成，但 Batch 仍要运行到第 64 步。

有效槽位数为：

$$
N_{useful}=8+16+32+64=120
$$

静态 Batch 分配的槽位为：

$$
N_{allocated}=4\times64=256
$$

仅从输出长度 Padding 看，有效率为：

$$
Efficiency=\frac{120}{256}=46.875\%
$$

Continuous Batching 在短请求结束后立刻把新请求放入空槽，使活动 Batch 尽量保持充实。它不能减少每个有效 Token 本身的模型计算，却能减少：

- 等待固定 Batch 凑齐的时间；
- 等最长序列完成造成的空槽；
- 短请求被长请求拖住的队头阻塞；
- 请求取消后仍执行的无效 Decode；
- 低负载时的大量 Padding。

## 五、三种 Batching 不要混淆

### 5.1 Static Batching

请求在执行前形成固定 Batch，直到全部完成才接收下一批。适合离线、长度分桶充分、吞吐优先的场景。

优点：实现简单、Shape 稳定、容易使用 CUDA Graph。缺点：在线等待和输出长度 Padding 明显，短请求无法释放槽位。

### 5.2 Dynamic Batching

服务器在一个短等待窗口内收集兼容请求，形成较大的请求级 Batch。它减少小请求 Launch 开销，但 Batch 一旦开始通常仍保持固定。

Dynamic Batching 常用于固定输出 Shape 的模型服务；对自回归 LLM，它只解决“开始时如何组批”，没有解决“生成过程中请求何时退出和补位”。

### 5.3 Continuous Batching

每个推理迭代重新形成活动集合：

```
Iteration 0: A(Prefill), B(Prefill)
Iteration 1: A(Decode), B(Decode), C(Prefill chunk)
Iteration 2: A(Decode), B完成, C(Prefill chunk)
Iteration 3: A(Decode), C(Decode), D(Prefill)
Iteration 4: A完成, C(Decode), D(Decode), E(Prefill)
```

Continuous Batching 是调度策略，不是单个 CUDA Kernel。生产系统还需要 Packed Input、Block Table、KV Cache Manager、变长 Attention 和输出状态管理配合。

## 六、Scheduler 每轮在做什么

一个简化的 Decode Loop 如下：

```
while server_alive:
    collect_new_requests()
    propagate_cancellation()
    reclaim_finished_kv_blocks()
    token_budget = max_num_batched_tokens

    schedule_decode_requests(token_budget, kv_budget, deadlines)
    schedule_prefill_chunks(remaining_budget, kv_budget, priorities)
    reserve_kv_blocks()

    execute_one_iteration()
    sample_and_stream_tokens()
    update_request_state_and_metrics()
```

调度器必须维护至少三类队列：

- Waiting：已准入但尚未开始；
- Running：已有 KV、正在 Prefill 或 Decode；
- Finished/Cancelled：等待回收状态和 KV Block。

有抢占或 Offload 时还会有 Paused/Swapped 队列。