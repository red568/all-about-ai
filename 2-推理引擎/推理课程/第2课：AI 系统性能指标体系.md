---
title: "第2课：AI 系统性能指标体系"
source: "https://everythingai.top/learn/ai-systems-performance-engineering-text/lesson-02"
author:
published:
created: 2026-09-26
description: "面向 AI 工程学习者的专业课程学习空间。"
tags:
  - "clippings"
---
**导航**

本课建立统一的 AI 系统性能指标语言，帮助你正确理解延迟、吞吐、利用率、Goodput 与 Roofline 之间的关系。

## 课程定位

性能工程最危险的情况不是“没有数据”，而是\*\*大家都拿着一个数字，却在讨论不同的东西\*\*。

两个团队都声称模型达到 `10,000 tokens/s` ，可能分别指：

- 团队 A：单个用户在单并发下的输出速度；
- 团队 B：100 个并发请求合计的系统输出吞吐；
- 团队 C：输入 Token 与输出 Token 相加后的总吞吐；
- 团队 D：离线批处理、不受延迟约束时的峰值吞吐；
- 团队 E：只统计 Decode，不包含排队、Prefill 和网络时间；
- 团队 F：生成长度只有 16，而另一个团队生成 512 个 Token。

这些数字都可能“计算正确”，但不能直接比较。

本课建立一套从用户体验、系统容量、训练效率到硬件上限的完整指标体系。重点不是背缩写，而是学会回答三个问题：

1. \*\*指标的分子和分母分别是什么？\*\*
2. \*\*测量窗口包含了什么，又排除了什么？\*\*
1. \*\*这个指标能证明什么，不能证明什么？\*\*

## 学习目标

完成本课后，你应该能够：

1. 区分平均延迟、P50、P95、P99，并解释为什么在线系统更关心尾延迟。
2. 区分 QPS、RPS、Samples/s、Sequences/s、Input/Output/Total Tokens/s。
1. 正确解释 LLM 的 TTFT、TPOT、ITL、E2E Latency 和每用户 Tokens/s。
2. 用 Little 定律连接并发数、吞吐和平均响应时间。
1. 从训练 Step Time 推导 Tokens/s，并估算 Dense Transformer 的 MFU。
2. 解释 MFU 与 HFU 的差异，特别是 Activation Recomputation 为什么会让 HFU 高于 MFU。
1. 根据 SLO、正确性和失败重试定义 Goodput，而不把 GPU-Util 当作有效产出。
2. 使用 Roofline 模型判断 Kernel 更可能受计算峰值还是内存带宽限制。
1. 运行一个纯 Python 指标实验；有 PyTorch 和 GPU 时，进一步测量矩阵乘法的实际 TFLOPs 与算术强度。

## 前置知识

- 已完成第 1 课，理解端到端关键路径和 Goodput 的基本直觉；
- 知道训练包含前向和反向，推理包含 Prefill 与逐 Token Decode；
- 会运行 Python；
- 可选：了解矩阵乘法 FLOPs 的基本计算方式。

## 核心直觉：没有上下文的性能数字几乎没有意义

假设一家快递公司说自己“速度是 1000 件/秒”。你至少还要问：

- 是整个仓库，还是每个工人？
- 是完成配送，还是只完成扫码？
- 包裹大小相同吗？
- 高峰期和空闲期一样吗？
- 超时、损坏和退件算不算？

AI 系统的性能数字也必须绑定上下文。一个完整指标可以写成：

```
Metric = 数值 + 单位 + 工作负载 + 负载模型 + 统计口径 + 正确性/SLO + 环境
```

例如：

这比单独写“820 tok/s”更有解释力，也更可复现。

## 指标全景图

```
用户体验
├── E2E Latency
├── TTFT：多久看到第一个 Token
├── TPOT / ITL：后续 Token 是否流畅
└── P50 / P95 / P99：普通用户与尾部用户体验

系统容量
├── QPS / RPS / Samples/s
├── Input Tokens/s
├── Output Tokens/s
├── Total Tokens/s
└── Concurrency 与队列长度

训练效率
├── Step Time
├── Samples/s / Tokens/s
├── MFU：模型所需 FLOPs 相对理论峰值
├── HFU：硬件实际执行 FLOPs 相对理论峰值
└── Scaling Efficiency

有效产出
├── SLO-compliant Goodput
├── Effective Training Time Ratio
├── 成本/百万 Token
└── Token/Joule

硬件瓶颈
├── Peak Compute
├── Memory Bandwidth
├── Arithmetic Intensity
└── Roofline 上限
```

这些指标不是互相替代，而是从不同高度观察同一系统。

## 一、Latency：用户等了多久

### 1\. 平均延迟为什么不够

对 N 个请求，平均延迟为：

$$
L_{avg}=\frac{1}{N}\sum_{i=1}^{N}L_i
$$

平均值容易被少数极端值拉高，也会掩盖尾部用户。例如 99 个请求耗时 100 ms，一个请求耗时 10 s，平均延迟约 199 ms。报告只写“平均 199 ms”看不出大多数用户很快，也看不出那个用户极慢。

因此在线系统通常同时报告：

- \*\*P50\*\*：50% 请求不超过此值，接近典型体验；
- \*\*P90/P95\*\*：观察较慢请求；
- \*\*P99/P99.9\*\*：观察尾延迟和 SLO 风险；
- \*\*Max\*\*：可辅助排障，但对样本量和偶然噪声极敏感。

百分位必须配合样本数。只有 100 个请求时，P99 几乎由最慢的一个样本决定；严谨的尾延迟评估需要足够长的测试时间和足够多的样本。

### 2\. LLM 请求的时间拆解

对流式生成请求：

```
客户端发出请求
    │ 排队、调度、网络、Prefill、首次采样
    ▼
第一个 Token ── gap1 ── Token2 ── gap2 ── ... ── 最后一个 Token
│<---- TTFT ---->│
│<--------------------- E2E Latency --------------------->│
```

### TTFT：Time to First Token

$$
TTFT=t_{first\ token}-t_{request\ sent}
$$

它通常包含客户端到服务端网络、排队、调度、Prompt 预处理、Prefill 和第一次 Decode。TTFT 对聊天系统“是否立即响应”非常重要。

### ITL：Inter-Token Latency

相邻流式 Token 事件的间隔：

$$
ITL_j=t_{j+1}-t_j
$$

它反映生成过程是否流畅。应报告 ITL 的分布，因为平均 30 ms 可能是每个 Token 都稳定在 30 ms，也可能是大多数 10 ms、偶尔卡顿 500 ms。

### TPOT：Time per Output Token

常见口径排除第一个 Token：

$$
TPOT_i=\frac{L_{e2e,i}-TTFT_i}{N_{out,i}-1}
$$

当服务端每次流式事件只返回一个 Token 时，一个请求的 TPOT 等于其 ITL 的平均值；但全局 TPOT 分布与所有 Token 间隔的 ITL 分布并不必然相同。某些服务端一次事件可能返回多个 Token，更要明确工具口径。

### 每用户输出 Tokens/s

对足够长的输出：

$$
TPS_{user}\approx\frac{1}{TPOT}
$$

若 TPOT 为 50 ms，则稳态约为 20 tok/s/user。这个值是单个用户看到的生成速度，不是整个服务集群的总吞吐。

### 3\. 何时 TTFT 高，何时 TPOT 高

| 现象 | 更可能的原因 | 优先检查 |
| --- | --- | --- |
| TTFT 高、TPOT 正常 | 排队、长 Prompt、Prefill、冷启动 | Scheduler、Prompt 长度、Prefill Batch、Prefix Cache |
| TTFT 正常、TPOT 高 | Decode 慢、KV Cache、显存带宽、批次过大 | Decode Batch、KV Cache、量化、带宽与 Kernel |
| TTFT 和 TPOT 都高 | 系统过载、模型/硬件不匹配 | 到达率、并发、资源、模型并行和服务配置 |
| P50 正常、P99 很差 | 长短请求干扰、队头阻塞、抖动 | 请求分布、调度策略、GC、共享资源和网络 |

## 二、Throughput、QPS 与 RPS：系统完成了多少工作

### 1\. Throughput

通用定义：

$$
Throughput=\frac{Completed\ Work}{Measurement\ Duration}
$$

关键在于 Completed Work 的单位：

- 图片任务：images/s；
- 语音任务：audio seconds/s；
- 训练：samples/s 或 tokens/s；
- LLM 推理：requests/s、input tokens/s、output tokens/s 或 total tokens/s。

### 2\. QPS、RPS 与 Samples/s

- \*\*RPS\*\*：每秒完成的请求数；
- \*\*QPS\*\*：每秒查询数；
- \*\*Samples/s\*\*：每秒完成的样本数。

在“一次请求等于一个查询，且查询只含一个样本”的简单 API 中，三者数值可能相同。但一个请求可能携带一个 Batch，一个 Query 也可能包含多个 Sample。报告必须定义单位，不能默认等价。

$$
RPS=\frac{N_{successful\ requests}}{T}
$$

失败请求是否计入分母、超时请求是否算完成，都应写清楚。生产容量通常只统计成功且满足质量要求的请求。

### 3\. LLM 的三种 Tokens/s

### 输入 Token 吞吐

$$
TPS_{input}=\frac{\sum_i N_{input,i}}{T}
$$

它主要反映 Prefill 工作量，但不同模型、Attention 实现和 Prefix Cache 命中率会改变实际成本。

### 输出 Token 吞吐

$$
TPS_{output}=\frac{\sum_i N_{output,i}}{T}
$$

这是服务端在整个测量窗口内完成的输出 Token 总数，常用于衡量 Decode 容量。

### 总 Token 吞吐

$$
TPS_{total}=\frac{\sum_i(N_{input,i}+N_{output,i})}{T}
$$

总 Tokens/s 不是输出 Tokens/s。若一个测试使用 4096 Token Prompt、只输出 16 Token，总 Tokens/s 会主要由 Prefill 输入贡献；另一个测试输入 128、输出 512，两者不能只看总数比较。

### 4\. 系统 Tokens/s 与每用户 Tokens/s 的矛盾

提高并发和 Continuous Batching 往往能提高系统总输出 Tokens/s，却可能降低每个用户的 Tokens/s 并增加 TPOT。原因是同一轮 Decode 要服务更多序列。

因此推理调优不是单目标最大化：

```
系统总吞吐 ↑  通常需要更高并发/更大批次
用户生成速度 ↑  通常需要更少竞争/更小批次
尾延迟 ↓      通常需要保留容量并控制排队
```

一个有效的容量数字应写成“在 P99 TTFT 和 P99 TPOT 均不超过 SLO 时的最大吞吐”。这就是 SLO 约束下的 Goodput 思想。

## 三、并发、队列与 Little 定律

稳定系统中，Little 定律为：

$$
L=\lambda W
$$

其中：

L：系统中的平均请求数，包括排队和处理中；

$\lambda$ ：平均到达/完成速率；

W：平均请求停留时间。

例如系统稳定完成 20 req/s，平均 E2E Latency 为 0.5 s，则系统平均约有：

$$
L=20\times0.5=10
$$

个请求处于排队或执行中。

Little 定律不告诉你怎样优化，却能做一致性检查：如果仪表盘显示 100 req/s、平均延迟 2 s，但平均在途请求只有 10，至少有一个统计口径不一致。

### 饱和点

当请求到达率低于服务能力，增加负载会提高吞吐，延迟变化不大；接近饱和后，吞吐增长变慢，队列和尾延迟迅速上升；超过稳定容量后，排队会不断积累。

```
低负载区：吞吐随到达率近线性增加，延迟平稳
拐点附近：设备利用提高，Batch 更有效，尾延迟开始上升
过载区：吞吐接近上限，延迟和超时急剧恶化
```

所以“压到系统不再变快”为止测到的是离线峰值，不一定是在线可用容量。

## 四、训练指标：Step Time、Tokens/s、MFU 与 HFU

### 1\. Step Time 与训练 Tokens/s

若全局 Batch Size 为 B\\\_{global}，每个样本序列长度为 S，Step Time 为 $T_{step}$ ：

$$
Training\ Tokens/s=\frac{B_{global}\times S}{T_{step}}
$$

全局 Batch 通常为：

$$
B_{global}=B_{micro}\times N_{data\ parallel}\times N_{grad\ accumulation}
$$

Tensor Parallel 或 Pipeline Parallel 切分的是同一模型/Batch，通常不直接乘进全局样本数。重复把所有 GPU 数量都乘进去，是常见统计错误。

### 2\. Dense Transformer 的模型 FLOPs 近似

一个常用的一阶估算是：Dense Transformer 训练每 Token 的模型 FLOPs 约为：

$$
F_{model/token}\approx6P
$$

P 为参数量。直觉上，前向传播约需 2P 次浮点操作，反向对权重和激活的梯度约再需 4P，合计约 6P。

这只是近似：

- Attention 的序列长度相关项可能不可忽略；
- Embedding、输出层和其他算子计数口径不同；
- MoE 不能直接使用总参数量，应按每 Token 实际激活专家与共享层计算；
- 参数共享、稀疏性和特殊架构会改变模型 FLOPs；
- 不同论文对一次 FMA 计 1 FLOP 还是 2 FLOPs 必须一致。

严谨比较应使用同一 FLOPs 公式，并公开公式。

### 3\. MFU：Model FLOPs Utilization

$$
MFU=\frac{F_{model/token}\times Tokens/s}{N_{device}\times Peak\ FLOPs/device}
$$

MFU 回答：\*\*按照模型算法完成这些有效 Token 所需的理论 FLOPs，占硬件对应精度理论峰值的多少？\*\*

它适合跨训练系统比较模型有效计算效率，但前提是：

- 模型 FLOPs 定义一致；
- Peak FLOPs 对应实际主要精度；
- 使用非稀疏还是稀疏峰值保持一致；
- 分母包含正确的设备数量；
- Tokens/s 为端到端稳态训练吞吐，包含数据、通信和优化器等时间。

### 4\. HFU：Hardware FLOPs Utilization

$$
HFU=\frac{Actual\ executed\ hardware\ FLOPs/s}{N_{device}\times Peak\ FLOPs/device}
$$

HFU 关注硬件实际执行了多少 FLOPs。它可能包含模型算法之外的额外计算，例如 Activation Recomputation。

如果训练为节省显存，在反向阶段重新执行部分前向计算，则：

- MFU 的分子仍按完成模型训练所需的“有效模型 FLOPs”计算；
- HFU 的分子包含重新计算的额外 FLOPs；
- 因此常见 $HFU>MFU$ 。

HFU 高不一定说明更高效。大量无效重算也能让硬件很忙。MFU 更接近模型有效计算，HFU 更接近硬件算力占用，两者结合才能定位问题。

### 5\. 为什么 MFU 不能替代 Profiler

MFU 低可能来自：

- GEMM 太小或形状不友好；
- HBM 带宽瓶颈；
- Pipeline Bubble；
- DataLoader 等待；
- 通信没有被重叠；
- Python/Kernel Launch 开销；
- Checkpoint、评估或日志被计入窗口；
- Peak FLOPs 分母选错。

MFU 只告诉你“离理论峰值有多远”，不告诉你为什么远。根因仍需时间线、硬件计数器和分层剖析。

## 五、Goodput：把正确性、SLO 和可靠性放回分子

### 1\. 推理 Goodput

定义一组条件：请求成功、输出通过质量要求、TTFT 不超过 SLO、TPOT 不超过 SLO。则：

$$
Inference\ Goodput=\frac{N_{correct\land successful\land SLO}}{T}
$$

也可以按 Token 定义：

$$
Token\ Goodput=\frac{Valid\ output\ tokens\ from\ SLO\ compliant\ requests}{T}
$$

如果系统原始吞吐 100 req/s，但只有 80 req/s 满足 P99 延迟目标，那么对业务更有价值的容量接近 80，而不是 100。

### 2\. 训练 Goodput

训练 Goodput 应排除：

- 因 NaN/Inf 或数据错误而无效的 Step；
- 故障后从旧 Checkpoint 重算的 Token；
- 节点预留、排队和重启时间；
- Straggler、通信气泡和输入等待；
- 没有推进目标训练进度的探测或失败运行。

可定义：

$$
Training\ Goodput=\frac{Net\ productive\ tokens}{Wall\ clock\ time}
$$

可靠性研究还使用 Effective Training Time Ratio 一类指标，把真正有效训练时间与总占用时间关联起来。它提醒我们：单次 Step 很快，但每隔几小时失败并回滚，最终完成训练仍可能很慢。

### 3\. Goodput 的前提是质量约束

把输出长度砍半、降低精度导致质量下降、直接丢弃慢请求，都可能提高表面吞吐。只有模型质量、任务正确性和 SLO 同时明确，Goodput 才不容易被“刷指标”。

## 六、Roofline：先判断应该优化计算还是数据搬运

### 1\. 算术强度

$$
AI=\frac{FLOPs}{Bytes\ transferred}
$$