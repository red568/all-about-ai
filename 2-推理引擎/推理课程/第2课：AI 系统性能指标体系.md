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

单位为 FLOPs/Byte。算术强度高，说明每搬运 1 Byte 数据能做很多计算；算术强度低，说明大量时间可能花在搬数据。

这里的 Bytes 应尽量是目标内存层级的实际流量。例如研究 HBM Roofline 时，应使用 HBM 与芯片之间的实际数据流量，而不是简单把张量文件大小相加。Cache 命中、数据复用和重复加载会改变真实流量。

### 2\. Roofline 上限

设理论计算峰值为 $P_{peak}$ ，内存带宽为 BW：

$$
P_{attainable}\le\min(P_{peak}, BW\times AI)
$$

- 当 $BW\times AI < P_{peak}$ ，Kernel 位于带宽斜坡，倾向于 Memory-Bound；
- 当 $BW\times AI\ge P_{peak}$ ，Kernel 进入计算平台区，倾向于 Compute-Bound。

### 3\. Ridge Point

计算峰值与带宽上限交点：

$$
AI_{ridge}=\frac{P_{peak}}{BW}
$$

若某 GPU 对应精度的非稀疏峰值为 100 TFLOP/s，HBM 带宽为 1 TB/s，则 Ridge Point 约为 100 FLOPs/Byte。算术强度明显低于 100 的 Kernel 很难仅靠增加计算单元利用率达到峰值，应优先减少访存或增加数据复用。

### 4\. Roofline 不是实测性能的保证

Roofline 给出上界，不包含所有现实因素：

- Kernel Launch 和同步；
- 指令依赖与流水线停顿；
- Occupancy 和寄存器压力；
- 非合并访存和 Cache 行浪费；
- 分支与 Warp Divergence；
- 通信、CPU 和框架开销；
- 不同内存层级的复杂性。

因此正确用法是：先用 Roofline 判断优化方向，再用 Nsight Compute 等工具验证真实瓶颈。

## 七、GPU-Util、Occupancy、MFU 为什么不是一回事

| 指标 | 观察对象 | 能说明什么 | 不能说明什么 |
| --- | --- | --- | --- |
| GPU-Util | 采样窗口是否有 Kernel 活动 | 是否存在明显空闲 | Kernel 是否高效、是否在做有效工作 |
| SM Active | SM 活跃时间/周期 | 计算单元是否活跃 | 活跃指令是否达到峰值吞吐 |
| Occupancy | 活跃 Warp 相对理论容量 | 是否有足够并发 Warp | 高 Occupancy 是否一定更快 |
| Tensor Core Util | Tensor Core 管线活动 | 是否使用矩阵加速路径 | 端到端 Goodput 是否高 |
| MFU | 模型有效 FLOPs/理论峰值 | 训练有效计算效率 | 具体瓶颈位置 |
| HFU | 实际硬件 FLOPs/理论峰值 | 硬件计算吞吐占比 | 额外 FLOPs 是否有业务价值 |
| Goodput | 满足约束的有效结果/时间 | 业务或训练真实产出 | 单个硬件单元的微观原因 |

看到 GPU-Util 100% 但 MFU 低并不矛盾：GPU 一直执行 Kernel，但 Kernel 可能受内存、形状、指令或同步限制，达不到对应精度的理论计算峰值。

## 八、如何设计一份可比较的性能报告

### 工作负载

- 模型名称、版本、参数量和 Dense/MoE；
- 精度与量化方式；
- 输入/输出长度分布，不只写平均值；
- Batch、并发、请求到达模型；
- 数据集、Tokenizer 和采样参数；
- 是否启用 Prefix Cache、Speculative Decoding、Continuous Batching。

### 系统环境

- GPU/CPU、设备数量、显存和互联；
- 驱动、CUDA、框架、推理引擎版本；
- 容器、功耗限制、时钟和共享资源；
- Tensor Parallel、Pipeline Parallel、Data Parallel 配置。

### 测量方法

- Warmup、稳定窗口和 Cooldown 是否排除；
- 请求是闭环并发还是开放环固定到达率；
- 样本数和测试时长；
- CUDA 是否正确同步计时；
- 成功、失败、取消和超时如何统计；
- P50/P95/P99 的计算方法。

### 结果

- E2E、TTFT、TPOT、ITL 分位数；
- RPS/QPS、输入/输出/总 Tokens/s；
- 每用户 Tokens/s 与系统 Tokens/s；
- 显存、功耗和成本；
- 正确性/质量；
- SLO Goodput；
- 原始 Trace 和完整运行命令。

## 完整可运行实验：从请求 Trace 到 MFU 与 Roofline

### 实验设计

本实验使用一个脚本完成三类练习：

1. \*\*Level 0：纯 Python 请求 Trace。\*\* 模拟固定到达率、多个服务 Worker、Prefill 和 Decode，计算 E2E、TTFT、TPOT、ITL、RPS、三种 Tokens/s 和 SLO Goodput。无需 GPU，也无需第三方库。
2. \*\*Level 1：训练 MFU/HFU 估算与 Roofline 计算器。\*\* 输入你真实测量的训练 Tokens/s、设备峰值、Kernel FLOPs 和流量。
1. \*\*Level 2：可选 PyTorch Matmul 实测。\*\* 自动选择 CPU/CUDA，正确同步计时，输出实际 TFLOP/s 和最低数据流量假设下的算术强度。

Trace 模拟不是某个真实推理引擎的性能预测器。它的目的，是让不同指标之间的数学关系可观察、可修改、可验证。

### 完整代码

保存为 `ai_metrics_lab.py` ：

```
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import platform
import random
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Iterable

@dataclass
class RequestTrace:
    request_id: int
    arrival_s: float
    input_tokens: int
    token_times_s: list[float]
    success: bool

    @property
    def output_tokens(self) -> int:
        return len(self.token_times_s)

    @property
    def first_token_s(self) -> float:
        return self.token_times_s[0]

    @property
    def finish_s(self) -> float:
        return self.token_times_s[-1]

    @property
    def e2e_ms(self) -> float:
        return (self.finish_s - self.arrival_s) * 1000.0

    @property
    def ttft_ms(self) -> float:
        return (self.first_token_s - self.arrival_s) * 1000.0

    @property
    def tpot_ms(self) -> float:
        if self.output_tokens <= 1:
            return 0.0
        return (
            (self.finish_s - self.first_token_s)
            * 1000.0
            / (self.output_tokens - 1)
        )

    @property
    def itl_ms(self) -> list[float]:
        return [
            (right - left) * 1000.0
            for left, right in zip(self.token_times_s, self.token_times_s[1:])
        ]

    @property
    def per_user_output_tps(self) -> float:
        if self.output_tokens <= 1 or self.finish_s <= self.first_token_s:
            return float("nan")
        return (self.output_tokens - 1) / (self.finish_s - self.first_token_s)

def percentile(values: Iterable[float], q: float) -> float:
    ordered = sorted(v for v in values if math.isfinite(v))
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

def summarize(values: Iterable[float]) -> dict[str, float]:
    cleaned = [v for v in values if math.isfinite(v)]
    if not cleaned:
        return {"mean": float("nan"), "p50": float("nan"),
                "p95": float("nan"), "p99": float("nan")}
    return {
        "mean": statistics.fmean(cleaned),
        "p50": percentile(cleaned, 0.50),
        "p95": percentile(cleaned, 0.95),
        "p99": percentile(cleaned, 0.99),
    }

def jittered_length(mean: int, jitter: float, rng: random.Random) -> int:
    low = max(1, round(mean * (1.0 - jitter)))
    high = max(low, round(mean * (1.0 + jitter)))
    return rng.randint(low, high)

def simulate_traces(args: argparse.Namespace) -> list[RequestTrace]:
    """A small queueing simulator, not a real model performance predictor."""
    rng = random.Random(args.seed)
    worker_available = [0.0 for _ in range(args.workers)]
    arrival_s = 0.0
    traces: list[RequestTrace] = []

    for request_id in range(args.requests):
        if request_id > 0:
            arrival_s += rng.expovariate(args.arrival_rps)

        input_tokens = jittered_length(
            args.prompt_tokens, args.length_jitter, rng
        )
        output_tokens = jittered_length(
            args.output_tokens, args.length_jitter, rng
        )

        worker = min(range(args.workers), key=worker_available.__getitem__)
        service_start_s = max(arrival_s, worker_available[worker])

        prefill_s = (
            args.base_prefill_ms / 1000.0
            + input_tokens / args.prefill_tokens_per_s
        )
        first_token_s = service_start_s + prefill_s

        # 为了观察 ITL 尾部，给每个 Token 间隔加入少量抖动。
        token_times = [first_token_s]
        for _ in range(1, output_tokens):
            jitter = max(0.1, rng.gauss(1.0, args.itl_jitter))
            token_times.append(
                token_times[-1] + args.decode_tpot_ms / 1000.0 * jitter
            )

        worker_available[worker] = token_times[-1]
        success = rng.random() >= args.failure_rate
        traces.append(
            RequestTrace(
                request_id=request_id,
                arrival_s=arrival_s,
                input_tokens=input_tokens,
                token_times_s=token_times,
                success=success,
            )
        )
    return traces

def analyze_traces(
    traces: list[RequestTrace], ttft_slo_ms: float, tpot_slo_ms: float
) -> dict:
    if not traces:
        raise ValueError("trace list is empty")

    window_start = min(t.arrival_s for t in traces)
    window_end = max(t.finish_s for t in traces)
    duration_s = window_end - window_start
    successful = [t for t in traces if t.success]
    good = [
        t for t in successful
        if t.ttft_ms <= ttft_slo_ms and t.tpot_ms <= tpot_slo_ms
    ]

    total_input = sum(t.input_tokens for t in successful)
    total_output = sum(t.output_tokens for t in successful)
    all_itls = [gap for t in successful for gap in t.itl_ms]

    metrics = {
        "attempted_requests": len(traces),
        "successful_requests": len(successful),
        "slo_compliant_requests": len(good),
        "duration_s": duration_s,
        "success_rate": len(successful) / len(traces),
        "request_throughput_rps": len(successful) / duration_s,
        "output_token_throughput_tps": total_output / duration_s,
        "input_token_throughput_tps": total_input / duration_s,
        "total_token_throughput_tps": (total_input + total_output) / duration_s,
        "goodput_rps": len(good) / duration_s,
        "goodput_ratio_of_attempted": len(good) / len(traces),
        "e2e_latency_ms": summarize(t.e2e_ms for t in successful),
        "ttft_ms": summarize(t.ttft_ms for t in successful),
        "tpot_ms_per_request": summarize(t.tpot_ms for t in successful),
        "itl_ms_per_token_gap": summarize(all_itls),
        "per_user_output_tps": summarize(
            t.per_user_output_tps for t in successful
        ),
    }
    return metrics

def print_distribution(name: str, stats: dict[str, float], unit: str) -> None:
    print(
        f"{name:24s} mean={stats['mean']:9.3f} {unit}  "
        f"p50={stats['p50']:9.3f}  p95={stats['p95']:9.3f}  "
        f"p99={stats['p99']:9.3f}"
    )

def print_trace_report(metrics: dict) -> None:
    print("\n=== Inference trace metrics ===")
    print(f"attempted requests       : {metrics['attempted_requests']}")
    print(f"successful requests      : {metrics['successful_requests']}")
    print(f"SLO-compliant requests   : {metrics['slo_compliant_requests']}")
    print(f"measurement duration     : {metrics['duration_s']:.3f} s")
    print(f"success rate             : {100*metrics['success_rate']:.2f}%")
    print(f"request throughput       : {metrics['request_throughput_rps']:.3f} req/s")
    print(f"input token throughput   : {metrics['input_token_throughput_tps']:.3f} tok/s")
    print(f"output token throughput  : {metrics['output_token_throughput_tps']:.3f} tok/s")
    print(f"total token throughput   : {metrics['total_token_throughput_tps']:.3f} tok/s")
    print(f"SLO goodput              : {metrics['goodput_rps']:.3f} req/s")
    print(f"goodput/attempted ratio  : {100*metrics['goodput_ratio_of_attempted']:.2f}%")
    print_distribution("E2E latency", metrics["e2e_latency_ms"], "ms")
    print_distribution("TTFT", metrics["ttft_ms"], "ms")
    print_distribution("TPOT per request", metrics["tpot_ms_per_request"], "ms")
    print_distribution("ITL per token gap", metrics["itl_ms_per_token_gap"], "ms")
    print_distribution("per-user output TPS", metrics["per_user_output_tps"], "tok/s")

def calculate_training_efficiency(args: argparse.Namespace) -> dict | None:
    if args.training_tokens_per_s <= 0 or args.peak_tflops <= 0:
        return None
    if args.exact_model_flops_per_token > 0:
        model_flops_per_token = args.exact_model_flops_per_token
        formula = "user supplied exact/analytical model FLOPs per token"
    else:
        model_flops_per_token = 6.0 * args.model_params_b * 1e9
        formula = "6P dense-Transformer approximation"

    useful_model_flops_per_s = (
        model_flops_per_token * args.training_tokens_per_s
    )
    cluster_peak_flops_per_s = (
        args.devices * args.peak_tflops * 1e12
    )
    mfu = useful_model_flops_per_s / cluster_peak_flops_per_s
    # 这是分析估算，不是硬件计数器实测 HFU。
    estimated_hfu = mfu * args.recompute_flops_factor
    return {
        "formula": formula,
        "model_flops_per_token": model_flops_per_token,
        "useful_model_tflops_per_s": useful_model_flops_per_s / 1e12,
        "cluster_peak_tflops": cluster_peak_flops_per_s / 1e12,
        "mfu": mfu,
        "estimated_hfu": estimated_hfu,
        "recompute_flops_factor": args.recompute_flops_factor,
    }

def print_training_efficiency(result: dict | None) -> None:
    print("\n=== Training MFU / HFU estimate ===")
    if result is None:
        print("Skipped. Set --training-tokens-per-s and --peak-tflops.")
        return
    print(f"model FLOPs formula       : {result['formula']}")
    print(f"model FLOPs/token         : {result['model_flops_per_token']:.4e}")
    print(f"useful model throughput   : {result['useful_model_tflops_per_s']:.3f} TFLOP/s")
    print(f"cluster theoretical peak  : {result['cluster_peak_tflops']:.3f} TFLOP/s")
    print(f"MFU                       : {100*result['mfu']:.2f}%")
    print(f"estimated HFU             : {100*result['estimated_hfu']:.2f}%")
    print(
        "HFU is analytical here. Real HFU needs profiler/counter data or a "
        "carefully validated executed-FLOPs model."
    )
    if result["mfu"] > 1.0 or result["estimated_hfu"] > 1.0:
        print("WARNING: utilization >100%; check peak precision, sparsity, FLOPs formula, and device count.")

def roofline(
    flops: float, bytes_moved: float, peak_tflops: float, bandwidth_gbps: float
) -> dict:
    if min(flops, bytes_moved, peak_tflops, bandwidth_gbps) <= 0:
        raise ValueError("all Roofline inputs must be positive")
    peak_flops_s = peak_tflops * 1e12
    bandwidth_bytes_s = bandwidth_gbps * 1e9
    arithmetic_intensity = flops / bytes_moved
    bandwidth_roof_flops_s = bandwidth_bytes_s * arithmetic_intensity
    attainable_flops_s = min(peak_flops_s, bandwidth_roof_flops_s)
    ridge_point = peak_flops_s / bandwidth_bytes_s
    return {
        "arithmetic_intensity_flops_per_byte": arithmetic_intensity,
        "ridge_point_flops_per_byte": ridge_point,
        "bandwidth_roof_tflops": bandwidth_roof_flops_s / 1e12,
        "attainable_tflops": attainable_flops_s / 1e12,
        "predicted_lower_bound_ms": flops / attainable_flops_s * 1000.0,
        "bound": "memory-bandwidth" if bandwidth_roof_flops_s < peak_flops_s else "compute",
    }

def print_roofline(result: dict | None, title: str = "Roofline estimate") -> None:
    print(f"\n=== {title} ===")
    if result is None:
        print("Skipped. Set peak, bandwidth, kernel FLOPs, and kernel bytes.")
        return
    print(f"arithmetic intensity      : {result['arithmetic_intensity_flops_per_byte']:.3f} FLOPs/Byte")
    print(f"ridge point               : {result['ridge_point_flops_per_byte']:.3f} FLOPs/Byte")
    print(f"bandwidth roof            : {result['bandwidth_roof_tflops']:.3f} TFLOP/s")
    print(f"attainable roof           : {result['attainable_tflops']:.3f} TFLOP/s")
    print(f"ideal time lower bound    : {result['predicted_lower_bound_ms']:.6f} ms")
    print(f"predicted primary bound   : {result['bound']}")

def optional_torch_matmul(args: argparse.Namespace) -> dict | None:
    if not args.torch_matmul:
        return None
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required only for --torch-matmul. Install torch or omit the flag."
        ) from exc

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    if device.type == "cpu" and dtype == torch.float16:
        print("WARNING: CPU float16 matmul may be unsupported or slower than float32.")

    if device.type == "cuda":
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        cc = torch.cuda.get_device_capability(index)
        device_description = {
            "name": props.name,
            "compute_capability": f"{cc[0]}.{cc[1]}",
            "vram_gib": props.total_memory / 2**30,
            "torch_cuda": torch.version.cuda,
        }
    else:
        device_description = {"name": platform.processor() or "CPU"}

    n = args.matmul_n
    generator = torch.Generator(device=device).manual_seed(args.seed)
    a = torch.randn((n, n), device=device, dtype=dtype, generator=generator)
    b = torch.randn((n, n), device=device, dtype=dtype, generator=generator)

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()

    for _ in range(args.matmul_warmup):
        _ = a @ b
    sync()
    start = time.perf_counter()
    for _ in range(args.matmul_iters):
        c = a @ b
    sync()
    elapsed_s = (time.perf_counter() - start) / args.matmul_iters
    # 防止某些编译/静态分析环境认为结果完全未使用。
    checksum = float(c[0, 0].float().cpu())

    flops = 2.0 * n**3
    element_size = a.element_size()
    # A、B 各读一次，C 写一次：这是理想最低流量，不是硬件计数器实测流量。
    minimum_bytes = 3.0 * n**2 * element_size
    achieved_tflops = flops / elapsed_s / 1e12
    result = {
        "device": str(device),
        "device_description": device_description,
        "dtype": args.dtype,
        "n": n,
        "mean_ms": elapsed_s * 1000.0,
        "achieved_tflops": achieved_tflops,
        "minimum_arithmetic_intensity": flops / minimum_bytes,
        "checksum": checksum,
    }
    if args.peak_tflops > 0:
        result["achieved_vs_peak"] = achieved_tflops / args.peak_tflops
    if args.memory_bandwidth_gbps > 0 and args.peak_tflops > 0:
        result["idealized_roofline"] = roofline(
            flops, minimum_bytes, args.peak_tflops, args.memory_bandwidth_gbps
        )
    return result

def print_matmul(result: dict | None) -> None:
    print("\n=== Optional PyTorch matmul ===")
    if result is None:
        print("Skipped. Add --torch-matmul to run it.")
        return
    print(f"device                     : {result['device']} {result['device_description']}")
    print(f"dtype / shape              : {result['dtype']} / {result['n']}x{result['n']}")
    print(f"mean latency               : {result['mean_ms']:.4f} ms")
    print(f"achieved throughput        : {result['achieved_tflops']:.4f} TFLOP/s")
    print(f"minimum arithmetic intensity: {result['minimum_arithmetic_intensity']:.3f} FLOPs/Byte")
    if "achieved_vs_peak" in result:
        print(f"achieved / supplied peak   : {100*result['achieved_vs_peak']:.2f}%")
    print(f"checksum                   : {result['checksum']:.6f}")
    if "idealized_roofline" in result:
        print_roofline(result["idealized_roofline"], "Idealized matmul Roofline")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI performance metric laboratory"
    )
    # Level 0: inference trace simulation
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--arrival-rps", type=float, default=4.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--length-jitter", type=float, default=0.20)
    parser.add_argument("--base-prefill-ms", type=float, default=8.0)
    parser.add_argument("--prefill-tokens-per-s", type=float, default=5000.0)
    parser.add_argument("--decode-tpot-ms", type=float, default=25.0)
    parser.add_argument("--itl-jitter", type=float, default=0.08)
    parser.add_argument("--failure-rate", type=float, default=0.01)
    parser.add_argument("--ttft-slo-ms", type=float, default=500.0)
    parser.add_argument("--tpot-slo-ms", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=2026)

    # Level 1: training MFU/HFU estimate
    parser.add_argument("--model-params-b", type=float, default=7.0)
    parser.add_argument("--training-tokens-per-s", type=float, default=0.0)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument(
        "--peak-tflops", type=float, default=0.0,
        help="Per-device non-sparse peak for the measured precision"
    )
    parser.add_argument(
        "--exact-model-flops-per-token", type=float, default=0.0
    )
    parser.add_argument(
        "--recompute-flops-factor", type=float, default=1.0,
        help="Executed FLOPs / useful model FLOPs; analytical estimate"
    )

    # Level 1/2: Roofline and optional real matmul
    parser.add_argument("--kernel-flops", type=float, default=0.0)
    parser.add_argument("--kernel-bytes", type=float, default=0.0)
    parser.add_argument("--memory-bandwidth-gbps", type=float, default=0.0)
    parser.add_argument("--torch-matmul", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--dtype", choices=["float32", "float16", "bfloat16"], default="float32"
    )
    parser.add_argument("--matmul-n", type=int, default=2048)
    parser.add_argument("--matmul-warmup", type=int, default=5)
    parser.add_argument("--matmul-iters", type=int, default=20)
    parser.add_argument("--json-out", type=str, default="")
    return parser.parse_args()

def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "requests": args.requests,
        "arrival_rps": args.arrival_rps,
        "workers": args.workers,
        "prompt_tokens": args.prompt_tokens,
        "output_tokens": args.output_tokens,
        "prefill_tokens_per_s": args.prefill_tokens_per_s,
        "devices": args.devices,
        "matmul_n": args.matmul_n,
        "matmul_iters": args.matmul_iters,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"these arguments must be positive: {invalid}")
    if not 0.0 <= args.failure_rate < 1.0:
        raise ValueError("--failure-rate must be in [0, 1)")
    if not 0.0 <= args.length_jitter < 1.0:
        raise ValueError("--length-jitter must be in [0, 1)")
    if args.recompute_flops_factor < 1.0:
        raise ValueError("--recompute-flops-factor should be >= 1.0")

def main() -> None:
    args = parse_args()
    validate_args(args)

    traces = simulate_traces(args)
    trace_metrics = analyze_traces(
        traces, args.ttft_slo_ms, args.tpot_slo_ms
    )
    print_trace_report(trace_metrics)

    training = calculate_training_efficiency(args)
    print_training_efficiency(training)

    kernel_roofline = None
    if (
        args.kernel_flops > 0
        and args.kernel_bytes > 0
        and args.peak_tflops > 0
        and args.memory_bandwidth_gbps > 0
    ):
        kernel_roofline = roofline(
            args.kernel_flops,
            args.kernel_bytes,
            args.peak_tflops,
            args.memory_bandwidth_gbps,
        )
    print_roofline(kernel_roofline)

    matmul = optional_torch_matmul(args)
    print_matmul(matmul)

    if args.json_out:
        report = {
            "arguments": vars(args),
            "trace_metrics": trace_metrics,
            "training_efficiency": training,
            "kernel_roofline": kernel_roofline,
            "matmul": matmul,
            "traces": [asdict(t) for t in traces],
        }
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print(f"\nJSON report written to: {args.json_out}")

if __name__ == "__main__":
    main()
```

### Level 0：纯 Python 运行

无需安装第三方依赖：

```
python3 ai_metrics_lab.py --json-out metrics_low_load.json
```

提高到达率，逐渐压过系统服务能力：

```
python3 ai_metrics_lab.py --arrival-rps 1  --workers 4
python3 ai_metrics_lab.py --arrival-rps 4  --workers 4
python3 ai_metrics_lab.py --arrival-rps 8  --workers 4
python3 ai_metrics_lab.py --arrival-rps 16 --workers 4
```

改变输入与输出长度：

```
# Prefill-heavy：长输入、短输出
python3 ai_metrics_lab.py \
  --prompt-tokens 4096 --output-tokens 32 --arrival-rps 2

# Decode-heavy：短输入、长输出
python3 ai_metrics_lab.py \
  --prompt-tokens 128 --output-tokens 512 --arrival-rps 2
```

改变 SLO，观察 Raw Throughput 不变而 Goodput 改变：

```
python3 ai_metrics_lab.py --ttft-slo-ms 1000 --tpot-slo-ms 60
python3 ai_metrics_lab.py --ttft-slo-ms 300  --tpot-slo-ms 30
python3 ai_metrics_lab.py --ttft-slo-ms 100  --tpot-slo-ms 20
```

### Level 1：用真实训练数据计算 MFU

先从训练日志获得端到端 `training_tokens_per_s` ，再从\*\*对应设备官方规格\*\*获取实际主要精度的\*\*非稀疏\*\*峰值。不要把 FP32、TF32、BF16、FP8、FP4 或稀疏峰值混用。

```
export TRAIN_TOKENS_PER_S=20000
export PER_DEVICE_PEAK_TFLOPS=100

python3 ai_metrics_lab.py \
  --model-params-b 7 \
  --training-tokens-per-s "$TRAIN_TOKENS_PER_S" \
  --devices 2 \
  --peak-tflops "$PER_DEVICE_PEAK_TFLOPS" \
  --recompute-flops-factor 1.20
```

上面的 `100` 只是演示命令格式，不代表任何具体 GPU。必须替换为你的设备、数据类型和稀疏口径对应的官方值。

对于注意力 FLOPs 不可忽略或 MoE 模型，先用你的模型结构计算更准确的 FLOPs/Token：

```
python3 ai_metrics_lab.py \
  --training-tokens-per-s 20000 \
  --devices 2 \
  --peak-tflops 100 \
  --exact-model-flops-per-token 4.8e10
```

### Level 1：Roofline 计算器

假设某 Kernel 做 $2\times10^{12}$ FLOPs，目标内存层级实际搬运 10^{11} Bytes；设备峰值和带宽仍需替换为真实官方值：

```
python3 ai_metrics_lab.py \
  --peak-tflops 100 \
  --memory-bandwidth-gbps 1000 \
  --kernel-flops 2e12 \
  --kernel-bytes 1e11
```

该例算术强度为 20 FLOPs/Byte，Ridge Point 为 100 FLOPs/Byte，因此 Roofline 会判断更倾向于内存带宽约束。

### Level 2：可选 PyTorch CPU/GPU Matmul

创建隔离环境：

```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch
```

如果需要 CUDA 专用 Wheel，请按运行当日的 PyTorch 官方安装选择器：https://pytorch.org/get-started/locally/生成命令，不要把某个 `cuXXX` 地址写死为所有环境通用方案。

CPU 路径：

```
python ai_metrics_lab.py \
  --torch-matmul --device cpu --dtype float32 --matmul-n 1024
```

通用 CUDA 路径：

```
python ai_metrics_lab.py \
  --torch-matmul --device cuda --dtype float16 --matmul-n 4096
```

如果同时提供正确的峰值和带宽，脚本会输出理想化 Roofline：

```
python ai_metrics_lab.py \
  --torch-matmul --device cuda --dtype float16 --matmul-n 4096 \
  --peak-tflops 100 --memory-bandwidth-gbps 1000
```

仍需再次强调：示例中的 `100 TFLOP/s` 和 `1000 GB/s` 是占位演示值，不是 RTX 3080、3090、4090、5090、A100、H100 或 B200 的统一规格。

## 预期现象

### 1\. 到达率低于服务能力

- Request Throughput 接近设定到达率；
- TTFT 主要由 Prefill 决定；
- P50/P99 差距较小；
- 大多数成功请求满足 SLO，Goodput 接近 Raw RPS。

### 2\. 接近饱和点

- 系统总吞吐继续增加，但增速变慢；
- 排队进入 TTFT，P95/P99 先明显上升；
- TPOT 变化可能不大，因为模拟器中单 Worker 的 Decode 速度固定；
- Goodput 可能在 Raw Throughput 仍增加时提前下降。

### 3\. 超过服务能力

- 完成吞吐接近上限；
- TTFT 和 E2E 尾延迟迅速上升；
- 更多请求违反 TTFT SLO；
- Raw RPS 看起来尚可，但 Goodput RPS 和 Goodput Ratio 下降。

这正是在线容量规划不能只追求峰值吞吐的原因。

### 4\. Prompt 变长

- TTFT 上升；
- Input Tokens/s 与 Total Tokens/s 的构成改变；
- Output Tokens/s 未必同比变化；
- 若只报告 Total Tokens/s，可能掩盖用户首 Token 等待恶化。

### 5\. 输出变长

- E2E Latency 上升；
- 单请求占用 Worker 更久；
- 同一到达率下更容易排队；
- `1 / TPOT` 仍接近单用户稳态 Tokens/s，但系统 RPS 会下降。

### 6\. MFU 超过 100%

这通常不是“突破物理极限”，而是输入口径错误：

- 分母用了错误精度的 Peak FLOPs；
- 把稀疏成绩与非稀疏峰值混用，或反过来；
- 设备数少算了；
- Tokens/s 重复乘了数据并行数；
- MoE 使用总参数量而不是激活 FLOPs；
- FMA 的 FLOPs 计数约定不一致。

## 结果分析方法：从指标组合推断瓶颈

| 指标组合 | 可能结论 | 下一步证据 |
| --- | --- | --- |
| GPU-Util 高、MFU 低、HBM 接近峰值 | Memory-Bound | Nsight Compute DRAM throughput、Roofline |
| GPU-Util 波动、时间线有空洞 | CPU/数据/Launch/同步瓶颈 | Nsight Systems、PyTorch Profiler |
| 单卡 MFU 高，多卡 MFU 快速下降 | 通信或并行气泡 | NCCL Trace、Scale Efficiency、各 Rank 时间 |
| HFU 明显高于 MFU | Activation Recomputation 或额外计算较多 | 重计算配置、Profiler FLOPs |
| 总 TPS 上升、每用户 TPS 下降 | 批处理提高容量但恶化体验 | 并发扫描、TPOT 与 SLO 曲线 |
| TTFT P99 高、TPOT 正常 | 排队或 Prefill 长尾 | Queue Time、输入长度分布、Prefill Batch |
| TTFT 正常、ITL P99 有尖峰 | Decode 调度、KV Cache 或系统抖动 | Token 时间线、Cache 利用、GC/抢占 |
| Raw RPS 高、Goodput 低 | 过载或质量/SLO 失败 | 失败码、SLO 分类、负载拐点 |
| Roofline 判定 Memory-Bound | 增加计算峰值收益有限 | 减少 Bytes、Fusion、Tiling、量化 |
| Roofline 判定 Compute-Bound | 计算单元是主要上限 | Tensor Core、精度、形状、Kernel 效率 |

## 常见错误与排查

### 错误 1：把系统 Tokens/s 当成单用户 Tokens/s

排查时必须同时报告：

- 并发数或到达率；
- System Output Tokens/s；
- Per-user Tokens/s 或 TPOT；
- TTFT 和 E2E 分位数。

### 错误 2：不同输入输出长度直接比较

固定 Prompt/Output 长度只是微基准；面向生产还要使用真实分布，至少给出长度 P50/P95/P99。长请求和短请求混合时，调度和队头阻塞会改变结果。

### 错误 3：只用闭环并发测在线服务

闭环模式是“上一个请求完成后客户端才发下一个”，客户端会自动降低到达率，可能掩盖排队崩溃。在线容量评估还应使用开放环固定请求率或 Poisson 到达，更接近真实流量。

### 错误 4：用平均延迟证明满足 P99 SLO

平均值不能推出尾部。直接计算足够样本上的 P99，并记录测试时长、样本量和置信要求。

### 错误 5：MFU 分母使用营销页最高数字

GPU 规格可能同时列出：

- 不同数据类型；
- Tensor Core 与普通 CUDA Core；
- 开启/关闭结构化稀疏；
- Boost 与非 Boost；
- 不同产品形态。

必须选择与工作负载实际算术路径一致的非稀疏或稀疏峰值，并在报告中注明。

### 错误 6：把 HFU 当作 MFU

如果用 Profiler 统计实际执行 FLOPs，其中包含重计算，再除以峰值得到的是更接近 HFU 的指标；若用 $6P\times Tokens/s$ ，得到的是近似 MFU。公式和名称不能互换。

### 错误 7：Roofline 的 Bytes 只按张量逻辑大小计算

真实 HBM 流量受 Cache、Tiling、重复加载、写回和中间张量影响。逻辑最小 Bytes 只能给理想化上界，严谨分析需要硬件计数器。

### 错误 8：优化后改变正确性或质量约束

量化、推测解码、Early Exit 或请求丢弃都可能提高吞吐。必须同时报告质量指标、接受率、失败率和 SLO Goodput。

### 错误 9：只测一个 Batch Size

应该扫描 Batch/并发/请求率，绘制吞吐-延迟曲线，找到 SLO 下的最佳点，而不是只展示离线最大值。

### 错误 10：忽略 Warmup、Cooldown 和 CUDA 异步

首次运行可能包含加载、编译和缓存建立；固定请求数测试的起止阶段也可能影响总吞吐。CUDA 计时还必须同步或使用 Event。测量窗口要明确。

## 优化前后对照：怎样证明优化真的有效

假设优化 Continuous Batching：

| 指标 | 优化前 | 优化后 | 结论条件 |
| --- | --- | --- | --- |
| System Output TPS | 800 | 1200 | 容量提高 50% |
| Per-user TPS | 28 | 22 | 单用户生成变慢，需要检查 SLO |
| P99 TTFT | 350 ms | 420 ms | 仍低于 500 ms SLO 才可接受 |
| P99 TPOT | 45 ms | 58 ms | 若 SLO 为 50 ms，则优化后不可接受 |
| Raw RPS | 5.0 | 6.8 | 仅说明完成请求更多 |
| Goodput RPS | 4.8 | 5.1 | 真正满足 SLO 的提升只有 6.25% |
| 质量指标 | 不变 | 不变 | 必须作为前提 |

如果只展示 System TPS，会声称提升 50%；加入用户体验与 SLO 后，业务有效收益可能只有 6.25%。这就是完整指标体系的价值。

## 硬件适配与架构专项说明

本课的指标定义不绑定某张 GPU，纯 Python Trace 在任意设备运行。可选 Matmul 实验适用于支持当前 PyTorch 的 CPU 或 CUDA GPU。

| 架构 | 示例设备 | 本课重点 | 专属能力边界 |
| --- | --- | --- | --- |
| Ampere | RTX 3080/3090、A100 | FP32/TF32/FP16/BF16 指标口径、常规 Roofline | A100 与消费卡的 HBM/GDDR、NVLink 等能力不同 |
| Ada Lovelace | RTX 4090、L40/L40S | 高吞吐推理、不同精度 Matmul | 4090 属于 Ada，不是 Blackwell |
| Hopper | H100/H200 | FP8、Transformer Engine、HBM Roofline | 需要对应软件栈和硬件，不能用 3080 等价模拟 |
| Blackwell | RTX 5090、B100/B200/GB200 | FP4/NVFP4、更新 Tensor Core 与互联系统 | 消费级 5090 与数据中心 B200/GB200 能力不相同 |

如果没有 H100/B200，仍可完成所有指标计算和 CPU/CUDA 基线；只是不能声称验证了 FP8/FP4、Transformer Engine 或 NVLink/NVSwitch 的真实硬件上限。

## 面试题与答案

### 1\. Throughput 与 Goodput 的区别是什么？

Throughput 统计单位时间完成的原始工作；Goodput 只统计正确、成功并满足质量/SLO 的有效工作。训练中还应扣除故障回滚和无效 Step。

### 2\. 为什么 1 / 平均 TPOT 不等于系统 Output Tokens/s？

`1/TPOT` 近似单用户稳态生成速度。系统吞吐是所有并发请求的输出 Token 总和除以墙钟时间，还受到批处理、并发、排队和测量窗口影响。

### 3\. TTFT 由哪些部分组成？

通常包含客户端网络、排队、调度、分词/预处理、Prefill、首次采样和返回第一个流式事件的时间。工具边界不同，需要明确客户端测量还是服务端测量。

### 4\. TPOT 和 ITL 有什么区别？

TPOT 常按单请求从第一到最后输出的生成阶段总时长除以后续 Token 数计算；ITL 是相邻流式事件之间的间隔。单 Token 流式返回时，单请求平均 ITL 与 TPOT接近，但全局分布和多 Token Chunk 情况可能不同。

### 5\. MFU 的常用公式是什么？

Dense Transformer 可近似使用：

$$
MFU\approx\frac{6P\times Tokens/s}{N_{device}\times Peak\ FLOPs/device}
$$

但 Attention、MoE、参数共享和 FLOPs 约定可能需要更精确公式。

### 6\. 为什么 Activation Checkpointing 会使 HFU 高于 MFU？

它在反向传播时重算部分前向，硬件实际执行的 FLOPs 增加；模型完成一个 Token 所需的有效算法 FLOPs定义不变。因此实际硬件 FLOPs 占比 HFU 可以高于有效模型 FLOPs 占比 MFU。

### 7\. MFU 低一定代表 Kernel 计算效率低吗？

不一定。数据等待、通信、Pipeline Bubble、Runtime、Checkpoint 和带宽瓶颈都会降低端到端 Tokens/s，从而降低 MFU。必须结合时间线和硬件指标定位。

### 8\. Roofline 如何判断 Memory-Bound？

计算 $AI=FLOPs/Bytes$ 和 $AI_{ridge}=Peak/BW$ 。若 $AI < AI_{ridge}$ ，带宽屋顶低于计算峰值，倾向 Memory-Bound；否则倾向 Compute-Bound。

### 9\. 为什么在线推理测试要使用固定到达率？

闭环并发会在服务变慢时自动减少发送速率，可能掩盖排队。固定到达率能观察接近饱和时队列与尾延迟如何恶化，更适合容量规划。

### 10\. 怎样公平比较两台 LLM 推理系统？

固定模型和质量、精度、Tokenizer、输入输出长度分布、采样配置、负载模型、并发/到达率和测量窗口；同时报告 TTFT/TPOT/ITL 分位数、输入/输出 Tokens/s、RPS、显存、功耗和 SLO Goodput。

### 11\. MoE 的 MFU 为什么容易算错？

因为总参数量不等于每 Token 激活参数量。直接使用 $6\times\text{总参数量}$ 会高估有效 FLOPs。应计算共享层、路由、激活专家和通信相关的实际模型 FLOPs，并公开口径。

### 12\. GPU-Util 100% 与 MFU 30% 是否矛盾？

不矛盾。GPU-Util 只表示采样窗口内存在 Kernel 活动；Kernel 可能受内存、指令、形状或同步限制，实际有效模型 FLOPs 只有峰值的 30%。

## 课后练习

### 练习 1：绘制吞吐-延迟-Goodput 曲线

将 `arrival-rps` 从 1 扫描到 20，记录：

- Raw RPS；
- Goodput RPS；
- P50/P99 TTFT；
- P99 TPOT；
- System Output TPS；
- Per-user TPS。

找到 Raw Throughput 峰值与 Goodput 峰值，解释它们为什么可能不是同一个点。

### 练习 2：验证 Little 定律

从 JSON Trace 计算每个时刻在途请求数的时间平均值，与 $RPS\times\text{平均 E2E 秒数}$ 比较。说明固定请求数测试的 Ramp-up/Ramp-down 为什么会产生偏差。

### 练习 3：为你的双 GPU 环境计算 MFU

从实际训练日志获得：

- 模型参数量与结构；
- Global Batch、Sequence Length、Step Time；
- 实际主要精度；
- 单卡对应精度的官方非稀疏峰值；
- Activation Recomputation 配置。

分别计算 Tokens/s、MFU 和分析估算 HFU，并列出所有假设。

### 练习 4：制造一个 120% MFU，再修正它

故意把稀疏/非稀疏峰值、设备数或 Tokens/s 口径设置错误，让 MFU 超过 100%。然后逐项排查，写出一份“指标口径事故复盘”。

### 练习 5：比较 Matmul Shape

在同一设备和精度下测试 `N=512/1024/2048/4096` ：

- 记录 Latency、TFLOP/s、Achieved/Peak；
- 解释为什么矩阵变大时设备利用通常先提高；
- 观察何时显存容量或运行时间成为限制。

### 练习 6：用真实 LLM 服务替换模拟 Trace

为 OpenAI 兼容流式接口编写客户端，记录请求发送、每个 SSE Chunk 到达和结束时间，复用本实验的 `analyze_traces()` 计算指标。注意 Token Chunk 可能包含多个 Token，需要用服务端 Tokenizer或返回 Usage 校准。

## 本课 Checklist

- 我不会在没有输入/输出长度和并发的情况下比较 Tokens/s。
- 我能区分系统 Output TPS 与 Per-user TPS。
- 我能解释 TTFT、TPOT、ITL 和 E2E Latency。
- 我会报告 P50/P95/P99，而不是只报告平均值。
- 我知道 QPS、RPS 和 Samples/s 何时相同、何时不同。
- 我能使用 Little 定律检查并发、吞吐和延迟是否自洽。
- 我能从 Global Batch、Sequence Length 和 Step Time 计算训练 Tokens/s。
- 我知道 6P 是 Dense Transformer 的近似，不适合盲套 MoE。
- 我能解释 MFU 与 HFU 的分子差异。
- 我不会混用数据类型、稀疏口径和设备数量的 Peak FLOPs。
- 我能用 SLO 和正确性定义推理 Goodput。
- 我知道故障回滚会降低训练 Goodput。
- 我会计算 Roofline 的算术强度和 Ridge Point。
- 我知道 Roofline 是上界，不是实测性能保证。
- 我不会把 GPU-Util、Occupancy、MFU 和 Goodput混为一谈。
- 我能为每个性能数字补齐工作负载、环境和统计口径。