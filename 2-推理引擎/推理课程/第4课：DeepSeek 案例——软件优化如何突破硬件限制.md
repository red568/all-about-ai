---
title: "第4课：DeepSeek 案例——软件优化如何突破硬件限制"
source: "https://everythingai.top/learn/ai-systems-performance-engineering-text/lesson-04"
author:
published:
created: 2026-09-26
description: "面向 AI 工程学习者的专业课程学习空间。"
tags:
  - "clippings"
---
**导航**

本课通过 DeepSeek 案例理解软件、算法与系统协同优化如何突破既有硬件条件下的性能上限。

课程进度：第 4/28 课

所属模块：AI 系统性能工程基础

核心案例：DeepSeek-V3 / R1 及公开基础设施组件

实验环境：Level 0 仅需 Python 3.9+；Level 1 可选 PyTorch 与任意 CPU/CUDA 设备；Level 2 仅用于具备相应 Hopper/Blackwell 数据中心硬件和互联的读者

## 课程定位

“软件优化突破硬件限制”不是让低端 GPU 违反物理定律，也不是用一个神奇 Kernel 把慢卡变成快卡。它真正表达的是：当算力、显存、互联或成本成为硬约束时，通过\*\*模型架构、数值格式、并行策略、通信 Kernel、调度和存储的联合设计\*\*，减少本来不必发生的计算与数据移动，并把不可避免的开销藏到关键路径之外。

DeepSeek-V3 是观察这种硬件-模型-系统协同设计的典型案例。官方技术报告给出的规模是 671B 总参数、每 token 约 37B 激活，使用 2,048 张 NVIDIA H800 完成训练；模型使用 MLA、DeepSeekMoE、辅助损失无关的负载均衡、MTP、FP8 混合精度和 DualPipe 等设计。后续公开的 FlashMLA、DeepGEMM、DeepEP、DualPipe、3FS 与 profile-data，则把优化从模型结构延伸到 Kernel、通信和存储。

本课不把 DeepSeek 神话化。我们会把内容分成三类：

- \*\*资料事实\*\*：官方论文、仓库或生产快照明确公开的设计和数据。
- \*\*实验观察\*\*：你在本课脚本或自己的设备上真正测得的结果。
- \*\*作者推导\*\*：用于理解机制的简化模型，不能替代官方实现或大规模复现。

## 学习目标

学完本课，你应该能够：

1. 解释 DeepSeek-V3 如何同时处理计算、显存、互联和训练稳定性约束。
2. 区分 MoE 的“总参数容量”和“每 token 激活计算”，避免把 671B 当作每步都计算 671B。
1. 解释 MLA 为什么能压缩 KV Cache，以及它为什么不是普通的“把 KV 改成低精度”。
2. 说明 FP8 训练、DualPipe、通信-计算重叠和 node-limited routing 各自在解决什么瓶颈。
1. 正确理解 2.788M H800 GPU hours 与约 557.6 万美元的口径边界。
2. 使用通用脚本量化稀疏路由、负载不均、低精度通信和流水线重叠的收益与代价。
1. 判断 DeepSeek 官方 Kernel 在 RTX 3080/3090、4090、5090、A100、H100/H200、B100/B200 上是否可直接运行。

## 前置知识

- 理解 Transformer 的 Attention 与 FFN 大致结构。
- 理解 Latency、Throughput、Goodput、带宽和 Amdahl 定律。
- 知道数据并行、张量并行、流水线并行和专家并行的基本含义即可。
- 建议先完成第 3 课，能够画出训练与推理的端到端关键路径。

## 先把事实说准确

### DeepSeek-V3 的公开规格

根据 DeepSeek-V3 Technical Report：

| 项目 | 官方公开口径 |
| --- | --- |
| 总参数 | 671B |
| 每 token 激活参数 | 约 37B |
| 预训练数据 | 14.8T tokens |
| 训练集群 | 2,048 张 NVIDIA H800 |
| routed experts | 每个 MoE 层 256 个，token 选择 8 个 |
| shared expert | 每个 MoE 层另有共享专家路径 |
| 主要结构 | MLA、DeepSeekMoE、MTP |
| 训练系统 | FP8 混合精度、DualPipe、跨节点 EP、通信-计算重叠 |
| 完整训练运行 GPU 时间 | 2.788M H800 GPU hours |

“37B activated”是每个 token 穿过的注意力、共享路径与被选专家等活跃参数的总体口径，不是“每个专家有 37B 参数”。MoE 的价值恰恰在于让模型拥有很大的总容量，但每个 token 只支付一小部分计算。

### 557.6 万美元是什么，不是什么

技术报告用每 H800 GPU-hour 2 美元估算：

$$
2.788\times10^6\ \text{GPU hours}\times 2\ \text{USD/GPU-hour}=5.576\times10^6\ \text{USD}
$$

这个数字是\*\*所报告 DeepSeek-V3 最终训练运行的 GPU 时间成本估算\*\*。它不等于：

- DeepSeek-V3 从研究到上线的完整组织成本；
- 数据获取、清洗、失败实验、消融、工程人员、机房和网络的全部成本；
- DeepSeek-R1 的完整训练成本；
- 任何其他团队可以按同样价格复现的保证。

DeepSeek-R1 是基于 V3 系列基础模型开展推理强化学习与多阶段后训练的另一项工作。把“约 600 万美元”直接说成“R1 全部训练成本”是不准确的。

### H800 的约束也要准确描述

H800 属于 Hopper 数据中心 GPU。DeepSeek-V3 报告强调的关键约束是互联：其集群中机内 NVLink 带宽为 400 GB/s，而跨节点 InfiniBand 为 50 GB/s，远低于计算和 HBM 本地数据路径的供给能力。不能笼统写成“H800 的 HBM 带宽低，所以一切都慢”，更不能把 H800 当作消费级 GPU。

这使 MoE 的 token dispatch/combine 成为核心问题：每个 token 被路由到不同 GPU 上的专家，通信量和尾部最忙 GPU 会决定整个层何时结束。

## 核心直觉：不是把一块板做快，而是重画整条关键路径

DeepSeek 案例可以浓缩为五个动作：

1. \*\*少算\*\*：用 MoE 让每个 token 只激活少量专家；用 MTP 让一次前向承担更多训练信号，并为推测解码提供候选。
2. \*\*少存、少搬\*\*：用 MLA 压缩推理时的 KV 表示；通信时对 dispatch 使用更低精度。
1. \*\*算得更便宜\*\*：采用细粒度缩放的 FP8 混合精度训练和定制 GEMM。
2. \*\*把等待藏起来\*\*：DualPipe、双 batch overlap 与通信 Kernel 让通信和计算并行。
1. \*\*让最慢者不再拖全局\*\*：路由负载均衡、node-limited routing、冗余专家和运行时负载均衡降低 straggler。

这些动作不是互相独立的。例如 MoE 降低了计算，却增加 All-to-All；为了让专家 GEMM 够大，推理需要更大的全局 batch，又进一步推动跨节点 EP；更大的 EP 带来通信问题，于是需要 DeepEP 和计算-通信重叠。系统优化的本质是\*\*把一个瓶颈换成更可管理的瓶颈，然后继续协同设计\*\*。

## 从模型架构到基础设施的完整因果链

### 1\. DeepSeekMoE：用稀疏激活换取模型容量

对稠密 FFN，简化计算量近似为：

$$
FLOPs_{dense}\propto 2N_{token}d_{model}d_{ff}
$$

若 MoE 有 E 个 routed experts，每个 token 选择 K 个，单个专家宽度为 $d_e$ ：

$$
FLOPs_{moe}\propto 2N_{token}K d_{model}d_e + FLOPs_{router}
$$

总参数容量随 E 增长，而每 token 的主要专家计算随 K 增长。只要 $K\ll E$ ，就能把“容量规模”和“激活计算”解耦。

但这不是免费午餐。路由之后需要：

```
hidden states
  → router/top-k
  → dispatch 到专家所在 GPU
  → grouped GEMM / expert computation
  → combine 返回原 token 顺序
```

因此 MoE 把一部分稠密计算压力转换成不规则内存访问、All-to-All 通信、小矩阵效率和负载均衡问题。

### 2\. 辅助损失无关的负载均衡

传统 MoE 常在训练目标中增加负载均衡 auxiliary loss。辅助损失过强可能干扰主任务学习，过弱又压不住专家倾斜。DeepSeek-V3 提出 auxiliary-loss-free strategy：维护每个专家的动态偏置用于路由决策，根据近期负载提高欠载专家的偏置、降低过载专家的偏置，而不让该偏置直接改变专家输出值。

简化直觉为：

$$
e^*=TopK_e(s_e(x)+b_e)
$$

$$
b_e\leftarrow b_e+\eta(\bar{n}-n_e)
$$

$s_e(x)$ 是 token 对专家的原始亲和分数， $b_e$ 是仅用于选择的负载偏置， $n_e$ 是观测负载， $\bar{n}$ 是平均负载。第二式只是教学抽象，不是官方实现逐行复刻。

负载平衡的系统目标不是让所有专家永远完全相等，而是降低最忙专家的尾部时间，同时少改变语义上合理的路由。常用观察量包括：

$$
Imbalance=\frac{\max_e n_e}{\operatorname{mean}_e n_e}
$$

当同步层必须等待最慢专家时，这个最大值比平均负载更重要。

### 3\. Node-Limited Routing：限制 token 跨多少节点

若只按最高路由分数选专家，一个 token 的 8 个专家可能散布在许多节点上，造成更多跨节点流。DeepSeek-V3 报告采用 node-limited routing，使每个 token 的目标专家最多分布到受限节点数量，从而把更多流量留在机内高带宽域。

它体现了 Mechanical Sympathy：模型路由器不仅关心“哪个专家最合适”，还知道专家放在怎样的物理拓扑上。代价是路由自由度被约束，必须联合训练和验证质量。

### 4\. MLA：压缩的是 KV 表示，不是简单删掉注意力头

标准 MHA 的 KV Cache 每 token 近似占用：

$$
M_{MHA/token}=L\times 2\times H_{kv}\times D_h\times B
$$

其中 L 是层数， $H_{kv}$ 是 KV heads， $D_h$ 是 head dimension，B 是每元素字节数。

MLA 通过低秩联合压缩，把 key/value 的主要信息保存为潜变量，再在计算中吸收或重构投影。简化存储模型为：

$$
M_{MLA/token}\approx L\times(D_{latent}+D_{rope})\times B
$$

以 V3 的公开配置做教学估算， $L=61$ 、latent rank 为 512、RoPE 部分为 64、BF16 为 2 字节：

$$
61\times(512+64)\times2=70,272\ \text{bytes/token}
$$

这与 DeepSeek 后续硬件论文报告的 70.272 KB/token 口径一致。MLA 的收益会直接传导到最大并发、长上下文容量与 decode 的内存流量，但代价包括额外投影、专用 Kernel 和实现复杂度。

### 5\. FP8 混合精度：目标不是“所有东西都变成 8 bit”

DeepSeek-V3 的 FP8 框架采用细粒度量化：报告中对 activations 使用 tile-wise scaling，对 weights 使用 block-wise scaling，并在敏感路径保留更高精度。核心关系是：

$$
q=clip\left(round\left(\frac{x}{s}\right), q_{min},q_{max}\right),\qquad \hat{x}=s q
$$

缩放粒度越细，局部动态范围越容易匹配，但 scale 元数据、量化 Kernel 和布局转换的开销越高。训练稳定性还取决于累加精度、master weights、归一化、梯度与通信格式。

FP8 的系统收益可能来自三处：

- Tensor Core 计算吞吐提高；
- 权重/激活占用和 HBM 流量减少；
- MoE dispatch 的网络字节减少。

但若张量形状小、量化开销大、Kernel 不支持或系统被通信延迟支配，FP8 不会自动带来理论倍数。

### 6\. DualPipe：让两个方向的微批互相填空

流水线并行的经典问题是 bubble。若 P 个 stage、M 个 microbatches，朴素 schedule 的填充/排空开销在 microbatch 很少时占比很高。DeepSeek 的 DualPipe 让 microbatch 从流水线两端双向进入，并设计 forward/backward 与通信的重叠，目标是同时减少 bubble 和裸露通信。

简化地，对每个 microbatch 计算时间 C、通信时间 T：

$$
T_{serial}=M(C+T)
$$

理想稳态重叠：

$$
T_{overlap}\approx C+T+(M-1)\max(C,T)
$$

实际 DualPipe 还要处理 forward、backward-input、backward-weight、stage 放置、激活内存与依赖，不等于一个 Python 双线程。官方 DualPipe 仓库明确指出，真实使用需为模型实现专门的 `overlapped_forward_backward` 。

### 7\. 为通信保留执行资源

DeepSeek-V3 报告描述了定制跨节点 All-to-All Kernel，并为通信保留部分 SM，让网络搬运与计算 Kernel 并行。这里存在关键权衡：

$$
T_{step}\approx\max(T_{compute}(SM_{compute}),T_{comm}(SM_{comm}))+T_{unhidden}
$$

给通信太少 SM，网络数据无法及时注入或合并；给通信太多 SM，GEMM 变慢。最佳比例依赖消息量、EP 规模、NVLink/RDMA、GEMM 形状和 Kernel 实现，不是一个通用常数。

DeepEP 后来把 MoE dispatch/combine 作为高吞吐和低延迟通信原语公开，并支持 FP8 dispatch、BF16 combine 等路径。当前仓库的能力要求已经演进，不能把 2025 年论文实现、DeepEP V1 与 2026 年主分支混为同一版本。

### 8\. MTP：训练目标与推测解码连接起来

Multi-Token Prediction 在训练时为后续多个 token 提供预测目标，使每个位置承载更多训练信号；在推理时，可把额外预测头用作候选，再由主模型验证，从而形成推测解码路径。

若一次提出 m 个候选，平均接受 a 个，验证一次成本为 $T_v$ ，额外 draft/MTP 成本为 $T_d$ ：

$$
Throughput_{effective}\approx\frac{a}{T_v+T_d}
$$

只有接受长度足够高、验证能有效并行且额外头开销受控时才有收益。官方报告给出的接受率与加速来自特定模型和系统，不能直接外推到任意 7B 模型或消费卡。

## 推理系统：同一模型需要另一套并行答案

DeepSeek 2025 Open-Source Week 公布的生产系统采用 prefill-decode disaggregation，并在两个阶段使用不同专家并行规模：公开快照中 prefill 使用 routed-expert EP32，decode 使用 EP144。原因不是“数字越大越快”，而是两阶段瓶颈不同：

- prefill 有较多 token，可形成大 GEMM，更偏向计算吞吐；
- decode 每步 token 少，权重和 KV 访问显著，追求低延迟；
- 更大 EP 能让每张 GPU 只持有/访问更少专家权重，但会扩大跨节点通信；
- 因而需要双 batch 或多阶段流水线隐藏 dispatch/combine。

同一公开快照覆盖 2025-02-27 12:00 到 2025-02-28 12:00（UTC+8），报告了线上 V3/R1 的节点占用、token 数、KV cache 命中和每节点吞吐。这些是一个生产时间窗的观测，不是所有部署的规格保证。

## 开源基础设施地图

| 组件 | 解决的问题 | 不能误解为 |
| --- | --- | --- |
| FlashMLA | MLA prefill/decode 的高性能 Attention Kernel | 任意 GPU 上可直接替代所有 Attention |
| DeepGEMM | FP8/FP4/BF16、MoE 等 GEMM Kernel | 所有形状、所有架构都必然胜过厂商库 |
| DeepEP | MoE dispatch/combine、低延迟/高吞吐通信 | 单卡也能复现 RDMA/NVLink 收益 |
| DualPipe | 双向流水线与计算-通信重叠 schedule | 调用一个开关就能适配任意模型 |
| 3FS | 面向 AI 训练/推理的分布式文件系统 | 本地 NVMe benchmark 的同义词 |
| profile-data | 公开部分训练/推理时间线证据 | 完整训练栈和全部生产配置 |

配套 `ai-performance-engineering` 仓库提供了 MoE、通信栈、DeepSeek/vLLM 调优与 baseline/optimized 方法论，但其当前主线明确偏向 NVIDIA Blackwell、CUDA 13 和较新的 PyTorch 开发环境。本课借鉴其“正确性门禁 + 基线/优化 + 证据产物”方法，不直接复制代码，也不把 B200 结果套到常见 GPU。

## 瓶颈分析方法

### 从约束开始，而不是从优化名词开始

按以下顺序建立证据：

1. \*\*容量\*\*：模型权重、optimizer、activation、KV Cache 能否放下？
2. \*\*计算\*\*：GEMM 是否够大、dtype/shape 是否命中高效 Tensor Core 路径？
1. \*\*本地内存\*\*：HBM/L2 流量是否主导？是否重复加载专家权重或 KV？
2. \*\*机内互联\*\*：expert dispatch 是否落在 NVLink 域？
1. \*\*跨节点网络\*\*：消息大小、并发、热点、RDMA 和拓扑如何？
2. \*\*调度\*\*：专家、DP instance、prefill/decode 是否有长尾不均？
1. \*\*关键路径\*\*：通信是总量高，还是未被隐藏的部分高？

### 诊断表

| 现象 | 候选原因 | 需要的证据 |
| --- | --- | --- |
| MoE FLOPs 少但比 dense 慢 | 小 GEMM、dispatch、Python/launch 开销 | expert batch size、Kernel 时间线、All-to-All 分解 |
| 部分 GPU 长时间等待 | expert/DP 负载不均 | 每 expert token 数、max/mean、rank 结束时间 |
| FP8 没有加速 | 量化/转置开销、shape 不支持、非计算瓶颈 | FP8 GEMM 与端到端时间、scale Kernel、数值误差 |
| EP 扩大后吞吐下降 | 跨节点通信增长、每专家 batch 变小 | NVLink/RDMA 字节、消息分布、GEMM shape |
| decode P99 抖动 | 请求/KV/专家不均、同步和队列 | 每 DP 请求数、KV 占用、expert receive load |
| 显存足够但并发低 | KV Cache 过大或碎片 | 每 token KV 字节、block 使用率、峰值内存 |
| overlap API 已开但时间仍相加 | 依赖、同 stream、SM/带宽争用 | Nsight 时间线、stream/event、资源占用 |

### 性能模型

MoE 层的简化关键路径可以写为：

$$
T_{moe}=T_{route}+T_{dispatch}+T_{expert}+T_{combine}+T_{imbalance}
$$

重叠后更接近：

$$
T_{moe,overlap}\approx T_{route}+\max(T_{dispatch}+T_{combine},T_{expert})+T_{unhidden}+T_{imbalance}
$$

通信字节的粗略下界：

$$
Bytes_{dispatch}\approx N_{token}\times K\times d_{model}\times B
$$

实际还包含元数据、padding、对齐、转发和 combine。将 BF16/FP16 的 2 字节传输改为 FP8 的 1 字节，理论 payload 可减半，但端到端收益受固定延迟、量化开销、网络利用和精度约束影响。

## 完整可运行实验：把 DeepSeek 的协同设计缩小到一台机器

本实验不下载模型，不需要 DeepSeek-V3 权重。它包含四个可分离的教学组件：

1. 稀疏路由和专家负载不均模拟；
2. block-wise 低精度通信近似与误差；
1. MHA/MLA KV Cache 容量模型；
2. 串行与理想重叠流水线模型；
1. 可选 PyTorch dense-all-experts 与 routed top-k 的等价输出 benchmark。

将代码保存为 `deepseek_codesign_lab.py` ：

```
#!/usr/bin/env python3
"""Portable DeepSeek co-design lab. No model download is required."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

def topk_indices(values: Sequence[float], k: int) -> List[int]:
    return sorted(range(len(values)), key=values.__getitem__, reverse=True)[:k]

def make_scores(tokens: int, experts: int, seed: int) -> List[List[float]]:
    """Create intentionally skewed affinity scores for a repeatable lab."""
    rng = random.Random(seed)
    hot = max(1, experts // 8)
    scores: List[List[float]] = []
    for token in range(tokens):
        row = [rng.gauss(0.0, 1.0) for _ in range(experts)]
        # A data-dependent popularity skew creates realistic straggler pressure.
        for expert in range(hot):
            row[expert] += 0.9 + 0.2 * math.sin(token / 17.0 + expert)
        scores.append(row)
    return scores

def route(
    scores: Sequence[Sequence[float]], k: int, bias: Sequence[float]
) -> Tuple[List[List[int]], List[int], float]:
    experts = len(bias)
    assignments: List[List[int]] = []
    counts = [0] * experts
    quality_sum = 0.0
    for row in scores:
        chosen = topk_indices([v + b for v, b in zip(row, bias)], k)
        assignments.append(chosen)
        for expert in chosen:
            counts[expert] += 1
            quality_sum += row[expert]
    return assignments, counts, quality_sum / (len(scores) * k)

def load_stats(counts: Sequence[int]) -> Dict[str, float]:
    avg = statistics.mean(counts)
    stdev = statistics.pstdev(counts)
    return {
        "min": min(counts),
        "max": max(counts),
        "mean": round(avg, 4),
        "max_over_mean": round(max(counts) / avg, 4),
        "coefficient_of_variation": round(stdev / avg, 4),
    }

def balance_routing(
    scores: Sequence[Sequence[float]],
    experts: int,
    k: int,
    steps: int,
    learning_rate: float,
) -> Tuple[List[float], List[int], float]:
    """Pedagogical feedback bias; not DeepSeek's production implementation."""
    bias = [0.0] * experts
    target = len(scores) * k / experts
    for _ in range(steps):
        _, counts, _ = route(scores, k, bias)
        for expert, count in enumerate(counts):
            relative_error = (target - count) / target
            bias[expert] += learning_rate * relative_error
    _, counts, quality = route(scores, k, bias)
    return bias, counts, quality

def quantize_int8_blockwise(
    values: Sequence[float], block_size: int
) -> Tuple[List[float], int, Dict[str, float]]:
    restored: List[float] = []
    scale_bytes = 0
    for start in range(0, len(values), block_size):
        block = values[start : start + block_size]
        absmax = max(abs(x) for x in block) or 1.0
        scale = absmax / 127.0
        q = [max(-127, min(127, round(x / scale))) for x in block]
        restored.extend(x * scale for x in q)
        scale_bytes += 4  # one FP32 scale per block in this toy format
    errors = [a - b for a, b in zip(values, restored)]
    mse = statistics.mean(e * e for e in errors)
    signal = statistics.mean(x * x for x in values)
    return restored, len(values) + scale_bytes, {
        "mse": mse,
        "relative_rmse": math.sqrt(mse / signal) if signal else 0.0,
        "max_abs_error": max(abs(e) for e in errors),
    }

def communication_model(
    tokens: int, experts: int, topk: int, hidden: int, ranks: int
) -> Dict[str, Any]:
    assignments = tokens * topk
    fp16_payload = assignments * hidden * 2
    fp8_payload = assignments * hidden
    dense_hypothetical = tokens * experts * hidden * 2
    return {
        "tokens": tokens,
        "experts": experts,
        "topk": topk,
        "ranks": ranks,
        "fp16_dispatch_mib": round(fp16_payload / 2**20, 3),
        "fp8_dispatch_mib_before_metadata": round(fp8_payload / 2**20, 3),
        "payload_reduction_fp16_to_fp8": round(fp16_payload / fp8_payload, 3),
        "dense_all_experts_hypothetical_mib": round(dense_hypothetical / 2**20, 3),
        "note": "Real traffic also contains metadata, padding, forwarding and combine.",
    }

def kv_cache_model(
    layers: int,
    kv_heads: int,
    head_dim: int,
    latent_rank: int,
    rope_dim: int,
    dtype_bytes: int,
    sequence_length: int,
    batch_size: int,
) -> Dict[str, Any]:
    mha_per_token = layers * 2 * kv_heads * head_dim * dtype_bytes
    mla_per_token = layers * (latent_rank + rope_dim) * dtype_bytes
    return {
        "mha_bytes_per_token": mha_per_token,
        "mla_bytes_per_token": mla_per_token,
        "compression_ratio_analytical": round(mha_per_token / mla_per_token, 3),
        "mha_total_gib": round(
            mha_per_token * sequence_length * batch_size / 2**30, 3
        ),
        "mla_total_gib": round(
            mla_per_token * sequence_length * batch_size / 2**30, 3
        ),
        "boundary": "This is a capacity model, not an MLA kernel benchmark.",
    }

def overlap_model(microbatches: int, compute_ms: float, comm_ms: float) -> Dict[str, float]:
    serial = microbatches * (compute_ms + comm_ms)
    ideal = compute_ms + comm_ms + (microbatches - 1) * max(compute_ms, comm_ms)
    hidden = max(0.0, serial - ideal)
    return {
        "serial_ms": round(serial, 3),
        "ideal_overlap_ms": round(ideal, 3),
        "ideal_speedup": round(serial / ideal, 3),
        "hidden_ms": round(hidden, 3),
        "unhidden_per_steady_microbatch_ms": round(max(compute_ms, comm_ms), 3),
    }

def command_output(argv: List[str]) -> Dict[str, Any]:
    if shutil.which(argv[0]) is None:
        return {"available": False}
    try:
        result = subprocess.run(
            argv, capture_output=True, text=True, timeout=5, check=False
        )
        return {
            "available": True,
            "returncode": result.returncode,
            "stdout": result.stdout.strip()[:3000],
            "stderr": result.stderr.strip()[:800],
        }
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": True, "error": repr(exc)}

def synchronize(torch: Any, device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()

def torch_moe_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise SystemExit("--torch requires PyTorch") from exc

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False")
    device = args.device
    torch.manual_seed(args.seed)
    x = torch.randn(args.torch_tokens, args.torch_dim, device=device)
    w1 = torch.randn(
        args.torch_experts,
        args.torch_dim,
        args.torch_hidden,
        device=device,
    ) / math.sqrt(args.torch_dim)
    w2 = torch.randn(
        args.torch_experts,
        args.torch_hidden,
        args.torch_dim,
        device=device,
    ) / math.sqrt(args.torch_hidden)
    router = torch.randn(args.torch_dim, args.torch_experts, device=device)
    logits = x @ router
    top_values, top_indices = torch.topk(logits, args.torch_topk, dim=-1)
    gates = torch.softmax(top_values, dim=-1)

    def dense_all_experts() -> Any:
        outputs = []
        for expert in range(args.torch_experts):
            outputs.append(F.silu(x @ w1[expert]) @ w2[expert])
        stacked = torch.stack(outputs, dim=1)
        selected = torch.gather(
            stacked,
            1,
            top_indices.unsqueeze(-1).expand(-1, -1, args.torch_dim),
        )
        return (selected * gates.unsqueeze(-1)).sum(dim=1)

    def routed_topk() -> Any:
        out = torch.zeros_like(x)
        for expert in range(args.torch_experts):
            token_pos, slot = torch.where(top_indices == expert)
            if token_pos.numel() == 0:
                continue
            expert_out = F.silu(x[token_pos] @ w1[expert]) @ w2[expert]
            out.index_add_(0, token_pos, expert_out * gates[token_pos, slot, None])
        return out

    def measure(fn: Any) -> Tuple[Any, float]:
        for _ in range(args.warmup):
            result = fn()
        synchronize(torch, device)
        samples = []
        for _ in range(args.repeat):
            start = time.perf_counter()
            result = fn()
            synchronize(torch, device)
            samples.append((time.perf_counter() - start) * 1000)
        return result, statistics.median(samples)

    with torch.inference_mode():
        dense_out, dense_ms = measure(dense_all_experts)
        routed_out, routed_ms = measure(routed_topk)
    max_diff = float((dense_out - routed_out).abs().max().cpu())
    result: Dict[str, Any] = {
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "device": device,
        "shape": {
            "tokens": args.torch_tokens,
            "dim": args.torch_dim,
            "hidden": args.torch_hidden,
            "experts": args.torch_experts,
            "topk": args.torch_topk,
        },
        "dense_all_experts_median_ms": round(dense_ms, 4),
        "routed_topk_median_ms": round(routed_ms, 4),
        "measured_speedup": round(dense_ms / routed_ms, 4),
        "expert_compute_ratio_idealized": round(
            args.torch_experts / args.torch_topk, 4
        ),
        "max_abs_difference": max_diff,
        "correctness": "PASS" if max_diff < 2e-4 else "CHECK",
        "warning": (
            "Sparse Python loops launch many small operations; fewer FLOPs need not "
            "mean lower latency. Production MoE uses grouped/fused kernels."
        ),
    }
    if device == "cuda":
        prop = torch.cuda.get_device_properties(0)
        result.update({
            "gpu_name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "memory_gib": round(prop.total_memory / 2**30, 2),
        })
    return result

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=2048)
    p.add_argument("--experts", type=int, default=64)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--ranks", type=int, default=8)
    p.add_argument("--hidden", type=int, default=7168)
    p.add_argument("--balance-steps", type=int, default=30)
    p.add_argument("--balance-lr", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=20260805)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--microbatches", type=int, default=32)
    p.add_argument("--compute-ms", type=float, default=8.0)
    p.add_argument("--comm-ms", type=float, default=5.0)
    p.add_argument("--torch", action="store_true")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--torch-tokens", type=int, default=512)
    p.add_argument("--torch-dim", type=int, default=128)
    p.add_argument("--torch-hidden", type=int, default=256)
    p.add_argument("--torch-experts", type=int, default=8)
    p.add_argument("--torch-topk", type=int, default=2)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--json-out", type=Path, default=Path("deepseek_codesign_report.json"))
    args = p.parse_args()
    if not (1 <= args.topk <= args.experts):
        p.error("require 1 <= topk <= experts")
    if not (1 <= args.torch_topk <= args.torch_experts):
        p.error("require 1 <= torch-topk <= torch-experts")
    return args

def main() -> None:
    args = parse_args()
    scores = make_scores(args.tokens, args.experts, args.seed)
    _, base_counts, base_quality = route(scores, args.topk, [0.0] * args.experts)
    bias, balanced_counts, balanced_quality = balance_routing(
        scores,
        args.experts,
        args.topk,
        args.balance_steps,
        args.balance_lr,
    )

    rng = random.Random(args.seed + 1)
    activations = [rng.gauss(0, 1) for _ in range(8192)]
    _, quantized_bytes, error = quantize_int8_blockwise(
        activations, args.block_size
    )
    fp16_bytes = len(activations) * 2

    report: Dict[str, Any] = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "nvidia_smi": command_output([
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version,pci.bus_id",
                "--format=csv,noheader",
            ]),
        },
        "routing": {
            "baseline": load_stats(base_counts),
            "balanced": load_stats(balanced_counts),
            "baseline_mean_selected_raw_score": round(base_quality, 6),
            "balanced_mean_selected_raw_score": round(balanced_quality, 6),
            "mean_abs_bias": round(statistics.mean(abs(x) for x in bias), 6),
            "interpretation": (
                "Feedback bias lowers straggler pressure but may alter semantic routing. "
                "This is a teaching analogue, not DeepSeek source code."
            ),
        },
        "communication": communication_model(
            args.tokens, args.experts, args.topk, args.hidden, args.ranks
        ),
        "blockwise_low_precision": {
            "elements": len(activations),
            "block_size": args.block_size,
            "fp16_payload_bytes": fp16_bytes,
            "toy_int8_plus_scale_bytes": quantized_bytes,
            "storage_ratio": round(fp16_bytes / quantized_bytes, 4),
            **{key: round(value, 8) for key, value in error.items()},
            "boundary": "INT8 simulation illustrates scaling; it is not FP8 E4M3.",
        },
        "kv_cache": kv_cache_model(
            layers=61,
            kv_heads=128,
            head_dim=128,
            latent_rank=512,
            rope_dim=64,
            dtype_bytes=2,
            sequence_length=32768,
            batch_size=8,
        ),
        "pipeline": overlap_model(
            args.microbatches, args.compute_ms, args.comm_ms
        ),
    }
    if args.torch:
        report["torch_moe_benchmark"] = torch_moe_benchmark(args)
    args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
```

## 运行命令

### Level 0：无 GPU、无第三方依赖

```
python3 --version
python3 deepseek_codesign_lab.py \
  --tokens 2048 --experts 64 --topk 8 \
  --balance-steps 30 --balance-lr 0.02 \
  --json-out deepseek_codesign_report.json
```

做三组单变量实验：

```
# 1. 路由更稀疏
python3 deepseek_codesign_lab.py --experts 64 --topk 2 \
  --json-out report_topk2.json

# 2. 不做反馈平衡
python3 deepseek_codesign_lab.py --balance-steps 0 \
  --json-out report_no_balance.json

# 3. 通信比计算更慢
python3 deepseek_codesign_lab.py --compute-ms 5 --comm-ms 12 \
  --json-out report_comm_bound.json
```

Level 0 只验证模型和方法：路由不均如何制造 straggler、低精度如何减少 payload、MLA 容量模型如何变化、理想 overlap 上限多大。它不测真实 GPU Kernel、NVLink、RDMA 或 FP8。

### Level 1：常见 NVIDIA GPU / PyTorch 基线

使用隔离环境，并从 PyTorch 官方安装页选择与驱动兼容的稳定版：

```
python3 -m venv .venv-deepseek-lab
source .venv-deepseek-lab/bin/activate
python -m pip install --upgrade pip
# 从 https://pytorch.org/get-started/locally/ 选择本机对应命令

python - <<'PY'
import torch
print("torch", torch.__version__)
print("torch CUDA", torch.version.cuda)
print("available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU", torch.cuda.get_device_name())
    print("capability", torch.cuda.get_device_capability())
    print("memory GiB", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
PY
```

CPU PyTorch：

```
python deepseek_codesign_lab.py --torch --device cpu \
  --torch-tokens 512 --torch-dim 128 --torch-hidden 256 \
  --json-out report_torch_cpu.json
```

任意可用 CUDA GPU：

```
python deepseek_codesign_lab.py --torch --device cuda \
  --torch-tokens 2048 --torch-dim 256 --torch-hidden 512 \
  --warmup 5 --repeat 20 \
  --json-out report_torch_cuda.json
```

这个 benchmark 让 baseline 先计算所有 expert，再只选择 top-k 输出；优化版只计算被路由 token。两者使用相同权重、路由和 gate，并检查最大绝对差。

理论专家 FLOPs 比例为 `experts/topk` ，但实测 routed 版本未必更快。原因是 Python 循环、 `where` 、索引、 `index_add_` 和很多小 GEMM。生产 MoE 需要 token 重排、grouped GEMM、融合 Kernel 和通信协同。这正是 DeepGEMM/DeepEP 一类系统工作的价值。

### 常见卡适配

- Ampere RTX 3080/3090、A100：运行通用 FP32/BF16/TF32 PyTorch 实验；不要尝试冒充 Hopper FP8 Kernel。
- Ada Lovelace RTX 4090、L40/L40S：运行通用实验；RTX 4090 是 Ada，不是 Blackwell。
- Hopper H100/H200：可选验证 SM90 路径，但仍需满足 CUDA、PyTorch、NCCL、NVLink/RDMA 等完整要求。
- Blackwell RTX 5090：可运行通用 PyTorch 实验，但其消费级 SM120 不等于 B100/B200 的 SM100；当前 DeepSeek 官方 Kernel 支持矩阵必须逐项检查。
- Blackwell B100/B200/GB200：可在满足软件版本和拓扑时验证 SM100 专项路径。

### Level 2：官方 Kernel 与集群专项实验

截至 2026-08-05，官方仓库主分支的硬件要求会持续变化。以当前文档为准：

| 项目 | 当前公开要求摘要 | 常见消费卡回退 |
| --- | --- | --- |
| FlashMLA | 主要为 SM90/SM100，CUDA 12.8+；具体 prefill/decode 有独立矩阵 | 运行本课容量模型或 PyTorch SDPA，不声称等价 |
| DeepGEMM | SM90/SM100；CUDA 12.3+/12.9+，C++20 等 | 运行 PyTorch GEMM，记录 shape/dtype |
| DeepEP V2 | Hopper/SM90 或兼容 PTX，较新 PyTorch/NCCL，机内 NVLink、跨机 RDMA | 只运行路由/通信字节模型 |
| DualPipe | PyTorch 2.0+ 示例；真实模型需自定义 overlap 实现 | 运行本课流水线上限模型 |
| 3FS | 多 SSD、多节点 RDMA 的系统实验 | 本地文件系统实验不能等价 |

只读能力探测：

```
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv
nvidia-smi topo -m
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

若准备运行官方项目，应克隆时记录 commit，而不是永远追随 `main` ：

```
git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git
cd DeepGEMM
git rev-parse HEAD
sed -n '1,220p' README.md
```

在不满足支持矩阵时应输出 `SKIPPED` ，而不是悄悄切到另一条实现后继续宣称“DeepGEMM/DeepEP 实测”。

## 预期现象

### Level 0

- 原始路由在“热门专家”偏置下， `max_over_mean` 明显大于 1。
- 反馈 bias 通常降低 `max_over_mean` 和变异系数，但 `mean_selected_raw_score` 可能略降，体现负载和语义亲和的权衡。
- toy INT8 + block scale 的 payload 接近 FP16 的一半，但会产生非零误差；它不是 FP8 E4M3 的位级实现。
- MLA 容量模型给出 70,272 bytes/token，并相对于假设的完整 MHA KV 产生明显压缩。
- 当 compute 与 comm 接近时，理想 overlap 收益最大；一侧远大于另一侧时，只能隐藏较短的一侧。

### Level 1

- dense-all-experts 和 routed-topk 的输出应在浮点误差范围内一致。
- 理论 FLOPs 降低不保证等比例加速。小 batch 或 expert token 太少时，稀疏版本可能更慢。
- 增大 `torch-tokens` 后，每个专家 GEMM 变大，routed 版本更可能体现优势；显存峰值也会增加。
- 不同 GPU、PyTorch 和 CUDA 版本的绝对耗时不可直接横比。

## 结果分析

### 1\. 为什么负载不均会吃掉稀疏收益

假设 64 个专家平均各接收 256 个 token，但最忙专家收到 512 个。同步 MoE 层近似等待最忙专家， `max/mean=2` 意味着一部分设备先结束后空等。即使总 FLOPs 没变，降低最大值也能缩短 critical path。

### 2\. 为什么 top-k 越小不一定越好

较小 top-k 减少计算和 dispatch 字节，却可能降低模型容量利用、训练质量和专家 batch size。极小专家 batch 会让 GEMM 退化为许多低利用率操作。系统目标是质量约束下的 Goodput，不是把 K 调到 1。

### 3\. 为什么低精度需要细粒度 scale

若一个 block 里有极端 outlier，统一 scale 会让其他小值落入很少的量化等级。缩小 block 可降低误差，却增加 scale 数量与量化开销。DeepSeek 的 tile/block 设计是在数值精度、硬件布局和元数据之间取平衡。

### 4\. 为什么重叠比“减少通信总量”更微妙

通信总量不变，也可能通过重叠减少暴露在 critical path 上的时间；反之，日志显示网络传了很多字节，不代表这些字节全部增加 step time。诊断应同时报告总通信时间和 unhidden communication。

### 5\. 为什么模型结构决定基础设施形态

MLA 改变 KV Cache 大小与 Attention Kernel；MoE 决定需要 expert parallel 和 All-to-All；MTP 改变推理验证与调度；长推理又改变 KV 和 token budget。基础设施并不是模型确定后才添加的“外壳”。

## 优化前后对照

| 瓶颈 | 直接基线 | 协同优化 | 新代价/风险 |
| --- | --- | --- | --- |
| 671B 容量的计算 | 每 token 激活全部参数 | MoE 只激活部分专家 | 路由、All-to-All、小 GEMM |
| KV Cache 容量 | 保存完整每头 K/V | MLA 低秩 latent cache | 投影与专用 Kernel |
| BF16 计算/传输 | 2 字节主要 payload | FP8 混合精度与细粒度 scale | 误差、量化、布局、稳定性 |
| Pipeline bubble | 单向填充/排空 | DualPipe 双向 schedule | 内存和调度复杂度 |
| MoE 通信暴露 | dispatch→compute→combine 串行 | DeepEP/定制 Kernel 与重叠 | SM、stream、拓扑协同 |
| 专家热点 | 纯亲和路由 | 动态 bias、节点受限、冗余专家 | 可能改变路由质量，状态更复杂 |
| Prefill/decode 同构部署 | 一套并行配置 | 阶段解耦、不同 EP 与负载均衡 | KV 传输和控制面复杂度 |

## 常见错误与排查

### 1\. 把 671B 当作每 token 的计算量

应同时报告 total parameters 和 activated parameters。模型权重容量仍接近总参数口径，但每 token 专家计算接近激活口径；Attention、路由和通信不能简单按 37/671 线性缩放。

### 2\. 把约 557.6 万美元说成 R1 完整成本

回到技术报告的 GPU-hour 表格。该估算对应 V3 报告的最终训练运行，不覆盖完整研发成本，也不能直接归因给 R1。

### 3\. 认为 H800 是“HBM 低带宽消费卡”

H800 是 Hopper 数据中心产品。V3 公开材料强调的约束主要包括互联和大规模系统通信。分析具体瓶颈时使用实际规格和 profiler，不用笼统标签。

### 4\. 在 RTX 4090/5090 上直接安装 FlashMLA 或 DeepGEMM 后失败

RTX 4090 是 Ada；RTX 5090 虽属 Blackwell，但 compute capability/SM 目标与 B200 不同。官方仓库当前主要列出 SM90/SM100，不意味着所有 Blackwell 都支持。先检查 README 的精确矩阵和 `torch.cuda.get_device_capability()` 。

### 5\. toy INT8 误当作 FP8

INT8 对称量化与 FP8 E4M3/E5M2 的指数、尾数和动态范围完全不同。本课 Level 0 只教学“block scale + payload/误差”关系。

### 6\. 稀疏 PyTorch 版本反而更慢

这是合理结果。检查每专家 token 数、Kernel 数量、launch gap、索引和 scatter 时间。生产优化方向是 grouped GEMM、token sorting、融合、持久化调度，而不是宣称稀疏理论失效。

### 7\. 只比较吞吐，不检查模型质量

路由 bias、FP8、MTP 接受策略都可能影响质量或数值。性能实验必须包含输出一致性、训练稳定性和任务指标；否则只是 badput。

### 8\. 通信和计算在时间线上没有重叠

检查依赖、stream/event、buffer 生命周期、SM 竞争和网络进度机制。异步调用只代表 API 立即返回，不代表硬件并行。

## 面试题与答案

### 1\. DeepSeek-V3 为什么 671B 参数但每 token 只激活约 37B？

因为它使用 MoE。每个 MoE 层包含大量 routed experts，但 token 只选择 top-8，并经过 shared expert 等公共路径。总参数决定容量和权重存储，激活参数更接近单 token 计算。

### 2\. MoE 主要节省什么，又增加什么？

节省每 token 的 FFN 计算，增加路由、token 重排、专家负载不均、跨 GPU All-to-All 和小/不规则 GEMM 的复杂度。

### 3\. MLA 与 MQA/GQA 的共同目标是什么？

都试图减少 KV Cache 和 decode 内存流量，但参数化方式不同。MLA 学习低秩联合 latent 表示，并通过投影吸收/重构，而不是简单共享固定数量 KV heads。

### 4\. 辅助损失无关的负载均衡为什么有价值？

它把“平衡系统负载”的控制从主训练 loss 中部分解耦，通过路由 bias 调整选择，降低强 auxiliary loss 干扰模型学习的风险。仍需监控路由质量与稳定性。

### 5\. DualPipe 解决什么问题？

减少 pipeline bubble，并通过双向微批 schedule 让 forward/backward 和通信尽量重叠。它不是单纯增加 microbatch，也不是通用一键优化。

### 6\. 为什么 FP8 不等于精度减半、速度翻倍？

端到端还包含量化、scale、布局转换、高精度累加、通信、非 GEMM 算子和未命中 FP8 Kernel 的形状。速度与质量必须实测。

### 7\. 2.788M H800 GPU hours 如何换算训练时长？

若 2,048 张 GPU 始终满配，粗略墙钟时间为 $2.788M/2048\approx1361$ 小时，约 56.7 天。它是 GPU 时间除以设备数的理想折算，不包含所有调试和停机历史。

### 8\. 为什么 DeepSeek 不依赖 tensor parallel 作为主要训练方案？

V3 报告强调通过高效算子和内存设计避免 tensor parallel 带来的额外通信，采用 PP、EP、DP/ZeRO-1 等组合。具体并行选择与模型形状和集群拓扑共同决定。

### 9\. 为什么 decode 可能使用比 prefill 更大的 EP？

decode 每步 token 少且更受权重/KV 访问和延迟影响。扩大 EP 可让每 GPU 管更少专家、降低本地访问，但会增加通信，因此必须配合低延迟通信、batch overlap 与负载均衡。

### 10\. DeepSeek 案例最可迁移的经验是什么？

不是复制某个 H800 参数，而是约束驱动的协同设计：建立性能模型，减少计算/状态/字节，按拓扑放置，把不可避免通信重叠，并用正确性与 profiler 证据闭环。

## 课后练习

1. 把 `experts` 固定为 64，扫描 `topk=1/2/4/8/16` ，画通信 payload、路由不均和 PyTorch latency 三张图。
2. 修改热门 expert 偏置强度，研究 `max_over_mean` 与 balance bias 的关系；报告原始路由分数损失。
1. 把 block size 设为 16/32/128/512，比较量化元数据、MSE 和压缩比，解释 Pareto 前沿。
2. 用实际 GPU 运行 torch benchmark，把 Nsight Systems 中的 Kernel 数、launch gap 与每专家 token 数对齐。
1. 设计一个 node-limited router：专家分布在 8 个节点，每 token 最多访问 2 个节点；比较平均亲和分数和跨节点数。
2. 使用 KV 模型比较 $batch=1/8/32$ 、 $sequence=4K/32K/128K$ 的容量；说明何时显存而非 FLOPs 决定并发。
1. 阅读官方 DualPipe schedule，解释它与本课 `max(compute, comm)` 模型之间哪些部分不等价。
2. 选一台 RTX 3090、RTX 4090、RTX 5090 或数据中心 GPU，写出 FlashMLA/DeepGEMM/DeepEP 能力检测结论；不支持也算有效实验。

## Checklist

### 事实口径

- 已区分 671B 总参数与约 37B 激活参数。
- 已写清 256 routed experts、top-8 和 shared expert 的关系。
- 未把 V3 的 GPU 时间成本误称为 R1 完整研发成本。
- 未把 H800 错写成消费级 GPU或简单归因于 HBM 带宽。
- 已区分官方报告事实、生产时间窗观察和作者教学推导。

### 性能工程

- 已同时分析计算、KV 容量、通信、负载和 pipeline bubble。
- 已记录最忙 expert/rank，而不只看平均负载。
- 已报告总通信和未隐藏通信。
- FP8 实验包含 scale 粒度、量化开销和精度检查。
- 稀疏 MoE 实验包含正确性和小 GEMM/launch 开销。

### 硬件边界

- Ampere=RTX 3080/3090、A100。
- Ada Lovelace=RTX 4090、L40/L40S，未把 4090 写成 Blackwell。
- Hopper=H100/H200；H800 也属于 Hopper 数据中心产品。
- Blackwell=RTX 5090、B100/B200/GB200，但未把 SM120 和 SM100 功能视为相同。
- 无 NVLink/NVSwitch/RDMA/FP8 专用路径时只做回退，不伪造等价结果。

### 实验交付

- Level 0 在无 GPU 环境可完整运行。
- Level 1 自动报告框架、CUDA、GPU、Compute Capability 和显存。
- Level 2 运行前固定官方仓库 commit 并检查支持矩阵。
- 所有加速比只绑定本次环境、shape、dtype 和版本。
- 已保存 JSON、命令、正确性结果与失败/跳过原因。