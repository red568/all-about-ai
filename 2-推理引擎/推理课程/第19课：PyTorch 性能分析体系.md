---
title: "第19课：PyTorch 性能分析体系"
source: "https://everythingai.top/learn/ai-systems-performance-engineering-text/lesson-19"
author:
published:
created: 2026-09-26
description: "面向 AI 工程学习者的专业课程学习空间。"
tags:
  - "clippings"
---
**导航**

本课建立 PyTorch 训练性能分析体系，学会用 Profiler 和时间线准确定位 CPU、GPU 与数据瓶颈。

## 课程定位

性能优化最危险的状态，不是程序慢，而是“凭感觉知道它为什么慢”。GPU 利用率低不等于 GPU 算子慢，某个算子 CUDA 总时间高也不等于优化它就能缩短 step；显存 reserved 很大，也不等于发生了内存泄漏。

本课建立一套可复用的 PyTorch 性能分析体系：先用低扰动基准确认问题，再逐层增加观测强度，最终把端到端慢映射到 Python、Dispatcher、ATen 算子、CUDA Runtime、GPU Kernel、显存分配、数据管线或分布式通信中的具体环节。

## 学习目标

完成本课后，你将能够：

1. 区分 Benchmark、Profile、Trace 和 Monitor。
2. 正确测量异步 CUDA 工作负载，避免漏掉同步与预热。
1. 阅读 PyTorch Profiler 的 Self CPU、CPU Total、Self Device、Device Total。
2. 用 `record_function` 、Profiler Schedule 和 Chrome/Perfetto Trace 定位阶段瓶颈。
1. 分析 allocated、reserved、active、inactive split 与非 PyTorch 显存。
2. 识别数据加载、Python launch、同步、算子碎片化、通信和显存碎片问题。
1. 建立“证据 → 假设 → 单变量改动 → 回归验证”的闭环。

## 前置知识

- 熟悉 PyTorch 模型、Autograd 与训练循环。
- 理解 CUDA Kernel 异步提交、Stream 和同步。
- 了解 Latency、Throughput、P50/P95/P99。
- 建议先完成第 2、12、15～18 课。

## 一、核心直觉：Profiler 是显微镜，不是测速表

测速回答“慢了多少”，Profiler 回答“时间花在哪里”，Trace 回答“先后关系和重叠发生了什么”，Monitor 回答“长时间运行时状态如何变化”。四者不能互相替代：

| 工具层 | 核心问题 | 常见输出 |
| --- | --- | --- |
| Benchmark | 到底快不快、是否回归 | 中位数、尾延迟、tokens/s |
| Profiler | 哪些算子最贵 | 算子聚合表、调用栈、shape |
| Trace | 为什么不能重叠、哪里有空洞 | CPU/GPU 时间线、Flow、Kernel |
| Monitor | 是否随时间、温度、负载变化 | 利用率、功耗、时钟、显存曲线 |

正确顺序是先确定性能问题真实存在，再用足够轻的工具缩小范围。Profile 本身有开销，若打开 shape、stack、memory 和长时间 CUDA Activity，观测到的就不再是原始程序。

## 二、从端到端时间拆开 PyTorch

一次训练 step 可粗略写成：

$$
T_{step}=T_{input}+T_{H2D}+T_{forward}+T_{backward}+T_{optim}+T_{comm}+T_{sync}+T_{other}-T_{overlap}
$$

注意这不是把 Profiler 表中的每一列直接相加。GPU Kernel、Memcpy、通信和 CPU 工作可能重叠，同一个父算子也包含子算子时间。

吞吐为：

$$
Throughput=\frac{Work}{T_{step}}
$$

若只优化占 step 比例为 (p) 的部分，并把该部分加速 (s) 倍，整体加速上限为 Amdahl 定律：

$$
Speedup=\frac{1}{(1-p)+p/s}
$$

例如某个算子占端到端时间 10%，即使无限加速，整体也最多约 $(1/0.9=1.11\times)$ 。这就是为什么“Top 1 算子”不一定是最值得优化的对象。

## 三、PyTorch 执行链路

一行 `y = model(x)` 大致跨过：

```
Python
  ↓ Module / Autograd
Dispatcher
  ↓ ATen Operator
Backend Library / Generated Kernel
  ↓ CUDA Runtime launch / memcpy
GPU Stream
  ↓ Kernel、Memcpy、Collective
Hardware
```

分析时要问：

- Python 是否来不及提交工作？
- 是否产生大量细碎 ATen 算子和 Kernel？
- 算子是否触发隐式同步？
- GPU Kernel 是计算受限、显存带宽受限，还是 launch 受限？
- H2D、NCCL 与计算是否有重叠？
- 某个高层 Module 为什么展开成这些低层算子？

PyTorch Profiler 擅长把 Python/ATen 与设备活动关联起来；Nsight Systems 擅长系统级时间线；Nsight Compute 擅长单 Kernel 微架构指标。不要指望一个工具回答全部问题。

## 四、读懂 Profiler 表

### 4.1 Self 与 Total

假设 `train_step` 包含 `forward` ， `forward` 又调用 `aten::mm` ：

- `CPU total` ：该事件及其子事件在 CPU 侧的总时间。
- `Self CPU` ：扣除子事件后，该事件自身的 CPU 时间。
- `Device total` ：该事件及子事件关联的设备活动总时间。
- `Self Device` ：归属于该事件自身、不含子事件的设备活动。

父级 `forward` 的 total 很大很正常；寻找叶子热点时更关注 Self，理解模块成本时看 Total。