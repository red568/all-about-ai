---
title: "第23课：LLM 推理系统架构"
source: "https://everythingai.top/learn/ai-systems-performance-engineering-text/lesson-23"
author:
published:
created: 2026-09-26
description: "面向 AI 工程学习者的专业课程学习空间。"
tags:
  - "clippings"
---
**导航**

本课拆解现代 LLM 推理系统的端到端架构，理解 Prefill、Decode、Batching、缓存与调度如何协同工作。

## 一、课程定位

大模型推理不是调用一次 `model.generate()` 那么简单。在线服务需要同时处理长短不同、到达时间不同、优先级不同的请求；还要在有限显存中保存模型权重和每个会话的 KV Cache，并在每个解码迭代重新决定哪些请求一起执行。

因此，LLM 推理系统的核心问题不是“单次矩阵乘法有多快”，而是：

本课建立端到端架构视角。后续课程会分别深入 Continuous Batching、KV Cache、推测解码、Prefill/Decode 解耦和 MoE；本课先回答它们在一套真实服务中处于什么位置、由谁管理、怎样互相影响。

## 二、学习目标

- 能画出一次 LLM 在线请求从接入到流式返回的完整链路。
- 区分 Gateway、Tokenizer、Admission Controller、Scheduler、KV Manager、Model Executor 和 Detokenizer 的职责。
- 理解 Prefill 与 Decode 的计算形态、性能指标和瓶颈差异。
- 掌握权重、KV Cache、Workspace、CUDA Graph Pool 和碎片的显存模型。
- 能用 TTFT、TPOT、ITL、Tokens/s、Goodput 和排队模型评价服务。
- 理解离线推理、在线推理、单体服务、分布式服务和解耦式服务的取舍。
- 完成无需 GPU 的容量规划实验，以及无需下载模型的 PyTorch Prefill/Decode 实验。
- 能为单卡、双卡和多节点环境选择数据并行、张量并行或请求路由策略。

## 三、前置知识

- Transformer Self-Attention、Causal Mask 和自回归生成。
- Latency、Throughput、P50/P99、TTFT、TPOT、Goodput。
- CUDA Stream、Kernel Launch、GPU Memory 与 Tensor Core。
- 基本的 HTTP/RPC、队列和并发概念。
- Python 与 PyTorch 基础。

## 四、核心直觉：推理系统是一台“有状态的 Token 工厂”

普通图像分类请求通常输入固定 Shape，执行一次前向计算后结束。LLM 请求不同：

1. Prompt 先经过一次 Prefill，生成第一个 Token 所需状态；
2. 系统保存所有层的 Key/Value；
1. 每生成一个新 Token，都要再次调度并执行一次 Decode；
2. 请求长度和结束时间事先并不完全确定；
1. 多个请求会在不同迭代加入或离开运行集合。

所以 LLM 服务不是“请求级 Batch”一次算完，而更像一个操作系统：

- 请求是进程；
- 每个 Decode Step 是一个时间片；
- KV Cache 是进程常驻内存；
- Scheduler 决定本轮运行谁；
- Token Budget 类似 CPU 时间与内存配额；
- Prefix Cache 类似可共享的只读页；
- 超时、取消和抢占决定无效工作能否及时停止。

优化对象由单个请求变成了整个在途请求集合。

## 五、端到端推理架构

### 5.1 一次请求的完整路径

```
客户端
  │ HTTP/gRPC，流式或非流式
  ▼
Gateway / API Server
  │ 鉴权、配额、路由、限流、参数校验
  ▼
Tokenizer / Input Processor
  │ 文本→Token IDs，Chat Template，多模态预处理
  ▼
Admission Controller + Request Queue
  │ 并发、Token、显存、Deadline 预算
  ▼
Scheduler / Batcher ───────────────┐
  │ 选择 Prefill/Decode 请求        │ 取消、优先级、抢占
  ▼                                │
KV Cache Manager ◄─────────────────┘
  │ 分配、映射、共享、驱逐 KV Block
  ▼
Model Executor
  │ 权重、Attention、MLP、Collective、采样 Kernel
  ▼
Sampler → Detokenizer → Stream Writer
  │ Token IDs→文本增量，停止条件，输出过滤
  ▼
客户端

旁路：Metrics / Trace / Logs / Billing / Health Check
```

该链路中，API Server 负责协议，Scheduler 负责时间，KV Manager 负责空间，Model Executor 负责计算。把所有职责都塞进一个同步 Python 循环，会让网络、日志、内存分配和 GPU 执行互相阻塞。

### 5.2 控制面与数据面

控制面负责低频决策：

- 模型版本、量化格式和副本数量；
- 路由、扩缩容、灰度与回滚；
- TP/PP/EP 拓扑和设备放置；
- SLO、租户配额和优先级规则。

数据面负责每请求、每迭代的高频工作：

- Tokenize、入队、组批；
- KV Block 分配和回收；
- Prefill/Decode 执行；
- Sampling、Detokenize 和流式发送。

高频数据面不应等待控制面 RPC，也不应在热路径加载模型、编译 Kernel 或初始化通信组。

### 5.3 Model Executor 内部

Model Executor 通常包含：

| 模块 | 职责 | 主要性能风险 |
| --- | --- | --- |
| Weight Loader | 加载、分片、量化权重 | 冷启动、峰值内存、格式转换 |
| Execution Plan | 选择图、Kernel 和 Shape Bucket | 重编译、图缓存膨胀 |
| Attention Backend | Prefill/Decode Attention | 长上下文、KV 读带宽 |
| KV Cache Manager | 分配和映射 KV Block | 碎片、驱逐、泄漏 |
| Distributed Executor | TP/PP/EP Collective | PCIe/NVLink/网络同步 |
| Sampler | Top-k、Top-p、温度、约束解码 | 小 Kernel、CPU 往返 |
| Output Processor | 停止条件、Detokenize、流式输出 | Python/GIL、网络背压 |

高性能框架的名称和实现会变化，但这些职责不会消失。

## 六、Prefill 与 Decode：同一个模型，两种工作负载

### 6.1 Prefill 阶段

Prefill 一次处理 Prompt 中的多个 Token，计算所有层的表示并建立 KV Cache。长 Prompt 可形成较大的矩阵乘法，通常有较高并行度；在常见密集 Transformer 上，较长 Prefill 往往更接近计算受限。

Prefill 的关键指标包括：

- 排队时间；
- Prompt Tokens/s；
- Prefill 时间；
- TTFT；
- 每请求占用的 KV 容量。

“Prefill 一定计算受限”不是定律。小 Batch、短 Prompt、低效 Kernel、CPU Tokenize 或频繁 Launch 都可能改变瓶颈。

### 6.2 Decode 阶段

Decode 每次通常只为每个活动序列生成一个 Token。每层都要读取权重和历史 KV，单步矩阵更“瘦”，算术强度通常低于 Prefill，因此常见瓶颈是：

- 权重和 KV Cache 的 HBM 带宽；
- 小 Kernel 与 Launch 开销；
- Scheduler 的每步开销；
- TP All-Reduce；
- 长上下文 Attention 的 KV 读取；
- 流式输出和同步。

Decode 的关键指标包括 TPOT、ITL、Generation Tokens/s 和每 Token 能耗。

### 6.3 为什么二者会互相干扰

长 Prefill 能占用大量计算时间；Decode 则需要频繁、低延迟地获得 GPU 时间片。如果把两者不加约束地放在同一队列，大 Prompt 可能让正在流式输出的请求出现 ITL 尖峰。

常见控制手段有：

- 限制每轮 Prefill Token Budget；
- Chunked Prefill，把长 Prompt 分段；
- Prefill/Decode 分别设优先级；
- 在资源足够时将两个阶段放到不同 Worker；
- 以 Deadline 或最大 ITL 约束调度。

## 七、推理延迟模型

### 7.1 TTFT

首 Token 延迟可拆为：

$$
T_{TTFT}=T_{gateway}+T_{tokenize}+T_{queue}+T_{schedule}+T_{prefill}+T_{sample}+T_{stream}
$$

只优化 `T_prefill` 不一定改善 TTFT。过载时， `T_queue` 往往远大于 GPU 执行时间。

### 7.2 TPOT、ITL 与端到端延迟

如果生成 $N_{out}$ 个 Token，近似有：

$$
T_{E2E}\approx T_{TTFT}+\sum_{i=2}^{N_{out}} ITL_i
$$

若使用平均每输出 Token 时间 TPOT：

$$
T_{E2E}\approx T_{TTFT}+(N_{out}-1)\cdot TPOT
$$

平均 TPOT 会掩盖停顿，因此在线聊天应同时报告 ITL 的 P50/P95/P99。

### 7.3 吞吐与 Goodput

$$
Throughput_{token}=\frac{N_{prompt}+N_{generated}}{T}
$$

但 Prompt Token 与 Generation Token 的成本不同，不宜只给一个总 Tokens/s。建议分开记录：

- Prompt Tokens/s；
- Generation Tokens/s；
- Requests/s；
- 满足 TTFT 与 TPOT SLO 的 Requests/s，即 Goodput。

设请求 i 是否满足正确性与 SLO 为 $I_i$ ：

$$
Goodput=\frac{\sum_i I_i}{T}
$$

把超时、取消后仍生成的 Token 计入吞吐，会虚高系统表现。

## 八、显存模型与 KV Cache

### 8.1 总显存预算

$$
M_{total}=M_{weights}+M_{KV}+M_{activation}+M_{workspace}+M_{graph}+M_{comm}+M_{fragment}+M_{reserve}
$$

其中权重通常较稳定；KV Cache 随在途 Token 数增长，是在线并发的主要动态容量；Workspace、CUDA Graph Pool、NCCL Buffer 和碎片决定“纸面可用显存”为何不能全部分给 KV。

### 8.2 权重容量

只考虑原始权重时：

$$
M_{weights}\approx N_{param}\cdot\frac{b_w}{8}
$$

7B 模型的 FP16 原始权重约 14 GB（十进制），4-bit 原始权重约 3.5 GB。实际加载量还包含 Scale、Zero Point、未量化层、Tensor 元数据和 Runtime Buffer，不能只按位宽做最终部署承诺。

### 8.3 KV Cache 每 Token 容量

对常见 Decoder-only Transformer，单个 Token 的 KV 近似为：

$$
M_{KV/token}=2\cdot L\cdot H_{kv}\cdot D_{head}\cdot\frac{b_{kv}}{8}
$$

- 2 表示 Key 和 Value；

L 是层数；

$H_{kv}$ 是 KV Head 数；

$D_{head}$ 是每个 Head 的维度；

$b_{kv}$ 是 KV 元素位数。

全部在途请求的 KV 近似为：

$$
M_{KV}=M_{KV/token}\cdot\sum_i S_i
$$

这里 $S_i$ 是请求当前已缓存 Token 数。GQA/MQA 通过减少 $H_{kv}$ 降低容量和带宽；它们不是运行时开关，而是模型结构的一部分。

### 8.4 连续内存与 Paged KV

为每个请求按最大长度预留连续 KV，简单但会产生内部碎片。Paged KV 将缓存拆成固定大小 Block，按请求实际增长分配，便于回收、共享 Prefix 和实现抢占。

Paged KV 并不意味着零开销：

- Block Table 需要管理；
- Block 大会增加内部碎片；
- Block 小会增加元数据和寻址开销；
- Prefix 只有完全匹配且语义安全时才能复用；
- 驱逐后可能需要重算或从主存/存储重新载入。

## 九、Scheduler：同时管理时间和空间

每轮 Scheduler 至少要回答四个问题：

1. 哪些新请求可以进入？
2. 本轮 Prefill 多少 Token？
1. 哪些 Decode 请求继续运行？
2. KV 不足时拒绝、等待、抢占还是驱逐谁？

### 9.1 双重预算

只限制 Batch Size 不够，因为 32 个短请求与 32 个长请求的成本差异巨大。实用 Scheduler 通常同时限制：

- 最大活动序列数；
- 每迭代最大 Token 数；
- 可用 KV Block；
- 最大 Prompt/Context 长度；
- Deadline、优先级和租户份额。

### 9.2 排队稳定性

到达率为 $\lambda$ ，有效服务率为 $\mu$ ：

$$
\rho=\frac{\lambda}{\mu}
$$

当 $\rho$ 长时间接近 1，尾延迟会迅速上升；当 $\rho>1$ ，队列必然增长。扩 Batch 可能提高 $\mu$ ，但如果同时让 TPOT 超过 SLO，Goodput 仍会下降。

### 9.3 背压与取消

无界队列不是容量。过载策略应明确：

- 拒绝并返回可重试状态；
- 降低最大输出长度或切换小模型；
- 路由到空闲副本；
- 对批处理请求延后执行；
- 用户断连后立即传播取消，释放 KV Block。

取消只停在 HTTP 层会产生“幽灵请求”：客户端已离开，GPU 仍在生成。

## 十、并行与部署形态

### 10.1 单 GPU

最简单、延迟最可控。若模型和目标 KV 容量能放入单卡，应先建立单卡基线。单卡没有 TP Collective，也更容易判断 Kernel、KV 和 Scheduler 的真实瓶颈。

### 10.2 数据并行副本

每张 GPU 保存完整模型，Router 按请求分发：

- 优点：没有请求内跨卡通信；故障隔离简单；吞吐易水平扩展。
- 缺点：每卡重复保存权重；单请求显存上限不变；Prefix/KV 在副本间迁移困难。

双 RTX 3080 20GB 环境中，若模型能放入单卡，两个独立副本通常比跨 PCIe 的 TP 更适合作为吞吐基线。

### 10.3 张量并行

模型层内切分到多张 GPU，可服务单卡放不下的模型，但每层会产生 Collective：

$$
T_{step}\approx T_{compute}+T_{collective}+T_{imbalance}+T_{sync}
$$

在没有 NVLink 的双卡或拓扑不理想时，Decode 的小粒度 All-Reduce 可能显著伤害 TPOT。RTX 3090 的特定双卡配置可有 NVLink；RTX 3080、RTX 4090、RTX 5090 消费卡不应默认存在可用 NVLink。必须用 `nvidia-smi topo -m` 和实测确认。

### 10.4 Pipeline、Expert 与解耦式并行

- Pipeline Parallel 适合模型跨设备分层，但在线小 Batch 容易产生 Bubble。
- Expert Parallel 用于 MoE，把专家分散在设备间，代价是 All-to-All 和负载不均。
- Prefill/Decode 解耦可针对两类负载独立扩缩，但增加 KV 传输、路由和故障处理复杂度。

并行的第一目标是“放得下”，第二目标才是“跑得快”。如果一个模型单卡可放下，不应仅凭 GPU 数量就启用 TP。

## 十一、Tokenizer、Sampling 与网络也在关键路径上

### 11.1 Tokenizer

Tokenizer 可能成为短请求、高 QPS 服务的 CPU 瓶颈。需要观察：

- 每请求 Tokenize/Detokenize 时间；
- 线程池是否受 GIL 或锁限制；
- Chat Template 是否重复处理；
- 超长文本是否在进入 GPU 前被限制；
- Token 数预算是否在排队前完成。

不要用字符数代替 Token 数做显存准入。

### 11.2 Sampling

Sampling 包含 Temperature、Top-k、Top-p、重复惩罚、停止词和约束解码。它通常是小算子，但每步执行，频繁 CPU 往返或全局同步会放大成本。

性能测试必须锁定 Sampling 配置。Greedy 与大 Vocabulary 上的复杂约束解码不是同一种负载。

### 11.3 流式输出

流式返回改善用户感知延迟，但会增加事件、序列化和网络写入次数。慢客户端必须有独立 Buffer 和背压，不能让一个阻塞 Socket 卡住整个 Decode Loop。

## 十二、瓶颈分析方法

### 12.1 先建立阶段时间线

为每个请求记录：

```
arrival
  → admission
  → queue_start / queue_end
  → tokenize_end
  → prefill_start / first_token
  → each_token_timestamp
  → finish / cancel
```

由此计算 Queue Time、TTFT、每个 ITL、E2E 和取消后的浪费。只有 GPU Kernel 时间无法回答“用户为什么慢”。

### 12.2 症状到证据

| 症状 | 优先检查 | 常见根因 |
| --- | --- | --- |
| TTFT 高、GPU 不忙 | Queue/Tokenize/CPU Timeline | 接入限速、CPU 瓶颈、组批等待 |
| TTFT 随 Prompt 急剧增长 | Prefill Kernel、Prompt 长度 | Attention/GEMM、Chunk Budget 不当 |
| TPOT 高、显存带宽高 | Decode Kernel、KV 读、权重读 | 内存受限、Batch 太小或上下文长 |
| ITL 偶发尖峰 | Prefill 插队、GC、同步、通信 | 长 Prefill 阻塞、Python Pause、TP 抖动 |
| 吞吐高但 P99 失败 | SLO Goodput、队列长度 | 过度组批、过载、无 Admission |
| OOM 随时间出现 | KV Block、取消、碎片 | KV 泄漏、请求未回收、Graph Pool 膨胀 |
| 双卡比单卡慢 | NCCL、 `topo -m` 、消息大小 | PCIe TP 通信、同步与负载不均 |
| GPU 利用率锯齿 | CPU/GPU Timeline | Tokenizer、Scheduler、Launch 或网络空洞 |

### 12.3 诊断顺序

1. 固定模型、精度、Prompt/Output 长度分布和 Sampling。
2. 分开测单请求、固定并发和开环到达率。
1. 记录阶段级指标，不只记录平均 E2E。
2. 检查 GPU 是否因 CPU、通信或显存不足而空闲。
1. 再下钻到 Nsight Systems、Nsight Compute 或框架 Trace。
2. 每次只改变一个旋钮，用端到端 Goodput 验证。

## 十三、实验环境与分层

### 13.1 Level 0：通用/CPU 回退

目的：不依赖 GPU，计算权重、KV、预留空间和理论并发，理解“模型能加载”与“服务能稳定并发”的差别。

### 13.2 Level 1：常见 NVIDIA GPU 基线

目的：自动检测 GPU、Compute Capability、显存、CUDA 与 PyTorch；用一个不下载权重的最小 Attention Block，实测 Prefill 与逐 Token Decode 的差异。

### 13.3 Level 2：架构专项可选实验

目的：在真实硬件上观察 Tensor Core、BF16/FP8、NVLink/NVSwitch 和多 GPU 拓扑。没有对应硬件时只做容量与流程回退，不能把模拟结果宣称为硬件实测。

## 十四、Level 0 实战：LLM 显存与并发容量规划器

将以下代码保存为 `llm_capacity_planner.py` 。

```
#!/usr/bin/env python3
import argparse
from dataclasses import dataclass

GIB = 1024 ** 3

@dataclass
class Plan:
    weight_gib: float
    kv_kib_per_token: float
    fixed_gib: float
    kv_budget_gib: float
    max_kv_tokens: int
    target_tokens: int
    target_kv_gib: float
    total_gib: float
    fits: bool
    max_concurrency_at_avg_context: int

def build_plan(args: argparse.Namespace) -> Plan:
    # 原始权重估算；overhead 用于覆盖量化元数据、未量化层和 Runtime Buffer。
    raw_weight_bytes = args.params_b * 1e9 * args.weight_bits / 8
    weight_bytes = raw_weight_bytes * (1 + args.weight_overhead)

    kv_bytes_per_token = (
        2
        * args.layers
        * args.kv_heads
        * args.head_dim
        * args.kv_bits
        / 8
    )

    usable_bytes = args.gpu_gib * GIB * args.utilization
    fixed_bytes = weight_bytes + args.runtime_gib * GIB
    kv_budget_bytes = max(0.0, usable_bytes - fixed_bytes)
    max_kv_tokens = int(kv_budget_bytes // kv_bytes_per_token)

    avg_context = args.avg_prompt + args.avg_output
    target_tokens = args.concurrency * avg_context
    target_kv_bytes = target_tokens * kv_bytes_per_token
    total_bytes = fixed_bytes + target_kv_bytes
    max_concurrency = max_kv_tokens // avg_context if avg_context > 0 else 0

    return Plan(
        weight_gib=weight_bytes / GIB,
        kv_kib_per_token=kv_bytes_per_token / 1024,
        fixed_gib=fixed_bytes / GIB,
        kv_budget_gib=kv_budget_bytes / GIB,
        max_kv_tokens=max_kv_tokens,
        target_tokens=target_tokens,
        target_kv_gib=target_kv_bytes / GIB,
        total_gib=total_bytes / GIB,
        fits=total_bytes <= usable_bytes,
        max_concurrency_at_avg_context=max_concurrency,
    )

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM inference memory capacity planner")
    p.add_argument("--params-b", type=float, default=7.0)
    p.add_argument("--weight-bits", type=float, default=4.0)
    p.add_argument("--weight-overhead", type=float, default=0.15)
    p.add_argument("--layers", type=int, default=32)
    p.add_argument("--kv-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--kv-bits", type=float, default=16.0)
    p.add_argument("--gpu-gib", type=float, default=20.0)
    p.add_argument("--utilization", type=float, default=0.85)
    p.add_argument("--runtime-gib", type=float, default=2.0)
    p.add_argument("--avg-prompt", type=int, default=1024)
    p.add_argument("--avg-output", type=int, default=256)
    p.add_argument("--concurrency", type=int, default=32)
    return p.parse_args()

def main() -> None:
    args = parse_args()
    if not (0 < args.utilization <= 1):
        raise ValueError("utilization must be in (0, 1]")
    if min(args.layers, args.kv_heads, args.head_dim) <= 0:
        raise ValueError("layers, kv_heads and head_dim must be positive")

    plan = build_plan(args)
    usable = args.gpu_gib * args.utilization
    print("=== LLM Inference Capacity Plan ===")
    print(f"GPU capacity              : {args.gpu_gib:.2f} GiB")
    print(f"Usable budget             : {usable:.2f} GiB")
    print(f"Estimated weights         : {plan.weight_gib:.2f} GiB")
    print(f"Weights + runtime fixed   : {plan.fixed_gib:.2f} GiB")
    print(f"KV cache per token        : {plan.kv_kib_per_token:.2f} KiB")
    print(f"KV budget                 : {plan.kv_budget_gib:.2f} GiB")
    print(f"Maximum resident KV tokens: {plan.max_kv_tokens:,}")
    print(f"Target resident tokens    : {plan.target_tokens:,}")
    print(f"Target KV cache           : {plan.target_kv_gib:.2f} GiB")
    print(f"Estimated total           : {plan.total_gib:.2f} GiB")
    print(f"Target fits budget        : {plan.fits}")
    print(
        "Max concurrency at average context "
        f"({args.avg_prompt + args.avg_output} tokens): "
        f"{plan.max_concurrency_at_avg_context}"
    )
    print("\nNote: this is a capacity estimate, not a performance guarantee.")

if __name__ == "__main__":
    main()
```

### 14.1 运行命令

```
python3 llm_capacity_planner.py

# 对照：同一 7B 模型改为 FP16 权重
python3 llm_capacity_planner.py --weight-bits 16 --weight-overhead 0.05

# 对照：将 KV Head 从 8 改为 32，模拟 MHA 相对 GQA 的容量变化
python3 llm_capacity_planner.py --kv-heads 32

# 无 GPU 机器也可模拟 24/48/80 GiB 卡
python3 llm_capacity_planner.py --gpu-gib 24 --concurrency 48
```

### 14.2 默认参数预期现象

默认配置模拟：7B、4-bit 权重、32 层、8 个 KV Head、Head Dim 128、FP16 KV、20 GiB 显存、85% 使用上限。你应看到：

- 权重估算约 3.75 GiB；
- KV 每 Token 约 128 KiB；
- 固定容量之外仍有一部分空间可用于 KV；
- 32 个平均 1280 Token 的在途请求能够通过该简化容量检查。

把权重改成 FP16 后，权重与 Runtime 固定容量接近预算上限，并发余量显著下降。把 KV Head 从 8 改成 32 后，每 Token KV 容量变为 4 倍。

### 14.3 结果边界

这个规划器故意不伪装成精确部署工具。真实系统还受到以下因素影响：

- Embedding/LM Head 是否共享；
- 量化 Scale、Zero Point 和部分 FP16 层；
- TP 后每 Rank 的权重与 KV 分片方式；
- CUDA Context、Kernel Workspace、Graph Pool、NCCL Buffer；
- KV Block 内部碎片和 Prefix 共享；
- Beam Search、LoRA、多模态 Encoder；
- Runtime 预留比例和版本实现。

正确用法是先算数量级，再用真实进程的显存指标和压力测试校准系数。

## 十五、Level 1 实战：最小 Prefill/Decode 执行器

该实验不下载模型，不依赖 Hugging Face。它使用一个最小 Causal Attention Block 展示：Prefill 一次处理整段 Prompt，Decode 每步追加 KV 并只查询最后一个 Token。

将以下代码保存为 `prefill_decode_bench.py` 。

```
#!/usr/bin/env python3
import argparse
import platform
import statistics
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

class ToyAttention(nn.Module):
    def __init__(self, hidden: int, heads: int) -> None:
        super().__init__()
        if hidden % heads != 0:
            raise ValueError("hidden must be divisible by heads")
        self.hidden = hidden
        self.heads = heads
        self.head_dim = hidden // heads
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        return x.view(batch, seq, self.heads, self.head_dim).transpose(1, 2)

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, heads, seq, dim = x.shape
        return x.transpose(1, 2).contiguous().view(batch, seq, heads * dim)

    def prefill(self, x: torch.Tensor):
        q = self.split_heads(self.q_proj(x))
        k = self.split_heads(self.k_proj(x))
        v = self.split_heads(self.v_proj(x))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(self.merge_heads(y)), (k, v)

    def decode(self, x: torch.Tensor, cache):
        old_k, old_v = cache
        q = self.split_heads(self.q_proj(x))
        new_k = self.split_heads(self.k_proj(x))
        new_v = self.split_heads(self.v_proj(x))
        # 教学用朴素追加。生产 Runtime 通常写入预分配的 Paged KV Block。
        k = torch.cat((old_k, new_k), dim=2)
        v = torch.cat((old_v, new_v), dim=2)
        # 当前查询位于序列末尾，可访问缓存中的全部历史位置。
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        return self.o_proj(self.merge_heads(y)), (k, v)

def percentile(values, q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]

def parse_args():
    p = argparse.ArgumentParser(description="Toy LLM prefill/decode benchmark")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--decode-steps", type=int, default=64)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        print("CUDA GPU not found. Run the Level 0 planner on this machine.")
        sys.exit(0)

    device = torch.device("cuda")
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU/PyTorch stack does not report BF16 support")

    prop = torch.cuda.get_device_properties(0)
    print("=== Environment ===")
    print(f"Python              : {platform.python_version()}")
    print(f"PyTorch             : {torch.__version__}")
    print(f"CUDA runtime        : {torch.version.cuda}")
    print(f"GPU                 : {prop.name}")
    print(f"Compute capability  : {prop.major}.{prop.minor}")
    print(f"VRAM                : {prop.total_memory / 1024**3:.2f} GiB")
    print(f"dtype               : {dtype}")

    torch.manual_seed(0)
    model = ToyAttention(args.hidden, args.heads).to(device=device, dtype=dtype).eval()
    prompt = torch.randn(
        args.batch, args.seq, args.hidden, device=device, dtype=dtype
    )

    with torch.inference_mode():
        for _ in range(args.warmup):
            _, warm_cache = model.prefill(prompt)
            token = torch.randn(
                args.batch, 1, args.hidden, device=device, dtype=dtype
            )
            _, _ = model.decode(token, warm_cache)
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output, cache = model.prefill(prompt)
        end.record()
        torch.cuda.synchronize()
        prefill_ms = start.elapsed_time(end)

        step_events = [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(args.decode_steps)
        ]
        token = output[:, -1:, :]
        host_start = time.perf_counter()
        for step_start, step_end in step_events:
            step_start.record()
            token, cache = model.decode(token, cache)
            step_end.record()
        torch.cuda.synchronize()
        host_ms = (time.perf_counter() - host_start) * 1000
        itl_ms = [start_event.elapsed_time(end_event) for start_event, end_event in step_events]

    generated_tokens = args.batch * args.decode_steps
    peak_gib = torch.cuda.max_memory_allocated() / 1024**3
    print("\n=== Results ===")
    print(f"Prompt shape        : batch={args.batch}, seq={args.seq}, hidden={args.hidden}")
    print(f"Prefill latency     : {prefill_ms:.3f} ms")
    print(f"Prompt throughput   : {args.batch * args.seq / (prefill_ms / 1000):,.1f} token/s")
    print(f"Decode host time    : {host_ms:.3f} ms")
    print(f"Mean TPOT           : {statistics.mean(itl_ms):.3f} ms/step")
    print(f"P50 ITL             : {percentile(itl_ms, 0.50):.3f} ms")
    print(f"P95 ITL             : {percentile(itl_ms, 0.95):.3f} ms")
    print(f"Generation throughput: {generated_tokens / (host_ms / 1000):,.1f} token/s")
    print(f"Final KV length     : {cache[0].shape[2]}")
    print(f"Peak allocated      : {peak_gib:.3f} GiB")
    print("\nThis toy block is for architecture study, not model-quality evaluation.")

if __name__ == "__main__":
    main()
```

### 15.1 隔离环境与安装

不要在系统 Python 中直接覆盖已有 CUDA 环境。PyTorch 安装命令与 CUDA Wheel 会变化，应从 PyTorch 官方安装选择器获取与你的驱动兼容的命令：

```
python3 -m venv .venv-llm-arch
source .venv-llm-arch/bin/activate
python -m pip install --upgrade pip

# 示例：先按官方安装选择器安装稳定版 PyTorch，再记录环境
python -m torch.utils.collect_env
```

### 15.2 运行命令

```
python prefill_decode_bench.py

# 改变 Prompt 长度，观察 Prefill 与 Decode
python prefill_decode_bench.py --seq 128
python prefill_decode_bench.py --seq 1024

# 改变活动序列数
python prefill_decode_bench.py --batch 1
python prefill_decode_bench.py --batch 8

# Ampere、Ada、Hopper、Blackwell 可在能力检测通过后对比 BF16
python prefill_decode_bench.py --dtype bf16
```

双 GPU 环境先分别运行单卡基线：

```
CUDA_VISIBLE_DEVICES=0 python prefill_decode_bench.py
CUDA_VISIBLE_DEVICES=1 python prefill_decode_bench.py
nvidia-smi topo -m
```

这个实验不使用两卡 TP，因为教学目标是先建立无通信的执行阶段基线。两张卡各自运行一个进程，可以检验副本吞吐；TP 必须另加 NCCL 通信并重新测 TPOT。

### 15.3 预期现象

- `Final KV length` 应等于 `seq + decode_steps` 。
- Prompt 变长后，Prefill 时间和 KV 峰值通常增加。
- Decode 每步只输入一个新 Token，但要读取更长的 KV 历史。
- 增大 Batch 可能提高总体 Generation Tokens/s，也可能提高单步 TPOT。
- 首次执行可能包含 Context、Allocator 和 Kernel 初始化，因此必须预热。
- 不同 GPU、PyTorch、CUDA、时钟、功耗和后台负载会得到不同数字；文中不提供“通用正确”的毫秒值。

### 15.4 结果分析

代码中的 `torch.cat` 是有意保留的朴素实现：每步都会创建更大的 K/V Tensor，不能代表生产级 Paged KV 性能。它帮助我们看清 Runtime 为什么需要预分配 Block Pool 和 Cache Manager。

实验得到的 Prefill Tokens/s 与 Generation Tokens/s不能直接当真实模型吞吐，因为它只有一个 Attention Block，没有 MLP、LayerNorm、Embedding、LM Head、跨层 KV、Sampling 和网络。不过执行形态是真实的：

- Prefill 的 Query 长度是整段 Prompt；
- Decode 的 Query 长度为 1；
- Decode 复用并扩展 KV；
- 流水中每一步都可能被调度、内存和 Launch 开销影响。

## 十六、Level 2：架构专项与真实服务

### 16.1 硬件能力边界

| 架构 | 示例 | 可选验证重点 | 不应默认 |
| --- | --- | --- | --- |
| Ampere | RTX 3080/3090、A100 | FP16/TF32/BF16、Flash/SDPA、A100 NVLink | 所有 Ampere 都有相同互联 |
| Ada Lovelace | RTX 4090、L40/L40S | FP16/BF16、较新 Tensor Core、单卡推理 | RTX 4090 是 Blackwell 或带 NVLink |
| Hopper | H100/H200 | FP8、Transformer Engine、NVLink/NVSwitch | 消费卡能等价模拟 FP8 数据路径 |
| Blackwell | RTX 5090、B100/B200/GB200 | 新低精度路径；数据中心型号的 NVLink/NVSwitch | RTX 5090 等同 GB200 系统 |

FP8、FP4、Transformer Engine、NVLink 和 NVSwitch 必须在真实支持的硬件与软件栈上验证。用 FP16 截断数值或用 PCIe 拷贝不能等价模拟专用低精度 Tensor Core 或 NVLink Fabric。

### 16.2 使用 vLLM 做真实服务基线

vLLM 当前官方 CLI 使用 `vllm serve` 。命令和参数随版本演进，先记录版本并查看本机帮助：

```
python -m venv .venv-vllm
source .venv-vllm/bin/activate

# 按 vLLM 官方安装文档选择与 CUDA/PyTorch 匹配的版本后：
vllm --version
vllm serve --help=max-num-seqs
vllm serve --help=max-num-batched-tokens

# MODEL 使用本地路径或你有权访问且显存可容纳的模型
MODEL=/path/to/model
vllm serve "$MODEL" \
  --dtype auto \
  --gpu-memory-utilization 0.85 \
  --port 8000
```

不要在生产机器上照抄最大并发。先用小模型验证 API，再逐步增加并发和长度；记录启动日志中的设备、Dtype、KV 容量和并行配置。

测试流式接口时，客户端要记录每个 Chunk 的到达时间，而不只记录最后完成时间。压测负载必须给出 Prompt/Output 长度分布、并发模型和 Sampling 参数。

### 16.3 真实系统的观测命令

```
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,power.draw --format=csv
nvidia-smi topo -m

# 系统级时间线；替换为实际服务启动命令
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --output=llm_server \
  vllm serve /path/to/model --dtype auto --port 8000
```

Nsight 的目标不是得到一张漂亮图，而是回答：

- GPU 空洞前 CPU 在做什么；
- Prefill 是否阻塞 Decode；
- 每 Token 有多少小 Kernel 和同步；
- TP 是否被 Collective 限制；
- H2D、网络和 Detokenize 是否进入关键路径。

### 16.4 框架选型不要只看单个榜单

vLLM、SGLang、TensorRT-LLM 等 Runtime 都在快速演进。评估时应固定同一模型、精度、长度分布、并发、Sampling、SLO 和硬件，并检查：

- 模型与量化格式支持；
- Attention/KV Backend；
- Continuous Batching 与优先级；
- Prefix Cache、LoRA、多模态和约束解码；
- TP/PP/EP 与多节点能力；
- 指标、Tracing、取消、健康检查和滚动升级；
- 部署复杂度、驱动/CUDA 约束和团队维护成本。

最快的离线吞吐配置不一定有最好的在线 P99，也不一定是最低总成本。

## 十七、优化前后对照

| 维度 | 朴素系统 | 性能工程化系统 |
| --- | --- | --- |
| 接入 | 无界接收、只按请求数限流 | 按 Token、KV、并发、Deadline 准入 |
| 调度 | 请求级静态 Batch | 迭代级调度、双重 Token/KV 预算 |
| Prefill/Decode | 同队列无约束竞争 | Token Budget、优先级、Chunk 或阶段隔离 |
| KV | 每请求按最大长度连续预留 | Paged Block、生命周期、复用与明确驱逐 |
| 内存 | 热路径频繁分配 | 预分配 Pool、Watermark 和碎片监控 |
| 并行 | 有几张卡就开几路 TP | 先单卡；按容量、拓扑和 SLO 选 DP/TP |
| 输出 | 每步全局同步、阻塞 Socket | 异步采样与流式 Buffer、慢客户端背压 |
| 观测 | 只有平均 QPS 和 GPU 利用率 | Queue、TTFT、ITL、KV、取消浪费和 Goodput |
| 过载 | 所有请求排队直到超时 | Admission、快速失败、降级与路由 |
| 验证 | 单 Prompt、闭环、只看均值 | 长度分布、开环负载、P99 与 SLO Goodput |

## 十八、常见错误与排查

### 18.1 “模型能加载，为什么一并发就 OOM”

模型加载只证明权重和初始化 Buffer 能放下。并发会增加 KV、临时 Workspace 和输出状态。用每 Token KV 模型估算，再用 KV Block 指标与真实峰值校准。

### 18.2 “GPU 利用率接近 100%，系统一定健康”

GPU 可能忙于超时请求、Padding、重算或低优先级 Prefill。检查 SLO Goodput、取消后生成 Token、Padding/Batch 组成和队列时间。

### 18.3 “Batch 越大，吞吐一定越高”

Batch 增大可能超过最佳 Kernel Shape、增加 KV 压力、降低时钟或使 TPOT 超标。以满足 SLO 的 Goodput 选择 Batch，而不是以最大显存占用选择。

### 18.4 “Decode 输入只有一个 Token，所以计算很少”

Query 虽短，但每层仍需读取权重和历史 KV。长上下文和低 Batch 时，内存流量、Launch 和通信可主导 TPOT。

### 18.5 “双卡应该比单卡快一倍”

如果启用 TP，每层通信会进入关键路径；如果是两个副本，单请求延迟不会减半。先明确目标是容量、单请求延迟还是总吞吐，再看拓扑和通信实测。

### 18.6 PyTorch 报 BF16 不支持

使用脚本自动检查。旧 GPU、旧驱动或 PyTorch 构建可能不支持该路径，回退 FP16/FP32。不要仅按 GPU 商品名推断软件栈可用性。

### 18.7 实验第一次特别慢

第一次包含 CUDA Context、Allocator、Kernel 加载和可能的编译。保留预热，但同时单独记录生产冷启动；不能把预热后的稳态数字冒充首次请求延迟。

### 18.8 ITL 周期性尖峰

检查长 Prefill 插入、Python GC、日志刷新、指标抓取、CPU 调度、CUDA 全局同步和 TP Collective。用请求 Token 时间戳与 Nsight Timeline 对齐。

### 18.9 请求取消后显存不下降

检查取消是否传到 Scheduler/Executor、KV Block 是否归还、输出协程是否仍持有引用、Allocator Reserved 与 Allocated 是否混淆。显存池保留不等于泄漏，但活动 KV Block 持续增长通常需要排查。

## 十九、性能优化闭环

### 19.1 建立基线

- 固定模型 Revision、Tokenizer、Dtype、量化、Sampling。
- 记录 GPU、Compute Capability、驱动、CUDA、框架版本和拓扑。
- 保存 Prompt/Output 长度直方图，而不只保存平均长度。
- 分别测冷启动、单请求、固定并发和目标到达率。

### 19.2 提出可证伪假设

示例：

- “TTFT 主要来自队列，而不是 Prefill。”
- “Decode 受 HBM/KV 读取限制，提高 Batch 可增加 Tokens/s。”
- “双 RTX 3080 上 TP 被 PCIe Collective 限制，双副本 Goodput 更高。”
- “长 Prefill 导致 ITL P99 尖峰，限制 Prefill Token Budget 可改善。”

### 19.3 一次改变一个变量

依次改变最大并发、Token Budget、KV 使用比例、Batch、Prompt Bucket、并行度或精度。优化前后必须使用相同请求集和到达过程。

### 19.4 正确性与质量门槛

验证：

- 相同随机种子与 Sampling 配置下输出是否符合预期；
- Stop/EOS、最大长度、取消与流式顺序是否正确；
- 量化后任务质量是否过线；
- Prefix/KV 复用是否存在跨租户或状态污染；
- OOM、Worker 重启和部分失败时请求语义是否正确。

### 19.5 端到端复测

最终报告至少包含：Requests/s、Prompt Tokens/s、Generation Tokens/s、TTFT P50/P99、ITL P50/P99、错误/拒绝率、SLO Goodput、显存峰值、功耗或成本。

## 二十、面试题与答案

### 20.1 为什么 LLM 服务需要迭代级调度？

因为每个请求会经历数量不确定的 Decode Step，完成时间不同。静态 Batch 必须等待最长序列，产生空转；迭代级调度可以在请求结束后立即移除，并让新请求进入下一轮。

### 20.2 Prefill 和 Decode 的主要差异是什么？

Prefill 对整段 Prompt 并行计算并建立 KV，长 Prompt 常有较高计算并行度；Decode 每步只生成少量 Token，却反复读取权重和历史 KV，通常更敏感于内存带宽、Launch、调度和通信。

### 20.3 TTFT 由哪些部分组成？

接入、Tokenize、排队、调度、Prefill、首轮 Sampling 和网络流式返回。GPU Prefill 只是其中一项。

### 20.4 KV Cache 为什么限制并发？

每个在途 Token 都要在每层保存 K/V。KV 容量与层数、KV Head 数、Head Dim、精度和所有活动上下文 Token 总数近似线性增长。

### 20.5 GQA 为什么能降低推理成本？

它让多组 Query Head 共享较少的 KV Head，减少 KV 每 Token 容量和 Decode 读取流量。代价与质量取舍由模型训练结构决定。

### 20.6 Paged KV 解决什么问题？

将 KV 拆成按需分配的 Block，降低按最大长度连续预留造成的碎片，支持请求动态增长、回收、Prefix 共享和更灵活的抢占。它仍有 Block 粒度、元数据和寻址成本。

### 20.7 为什么高 GPU 利用率不等于高 Goodput？

GPU 可能执行超时、取消、Padding 或错误质量的工作。Goodput 只统计满足正确性和 SLO 的有效请求。

### 20.8 双 GPU 什么时候选副本，什么时候选 TP？

模型单卡能放下且目标是总吞吐时优先测双副本；模型单卡放不下或必须降低单请求计算时间时考虑 TP，但要把 Collective、拓扑和同步成本计入 TPOT。

### 20.9 为什么不能只用平均 Prompt 长度压测？

真实长度分布会导致不同 Prefill 成本、KV 占用、Batch 组成和队头阻塞。平均值相同的两个分布可能有完全不同的 P99 和 OOM 风险。

### 20.10 什么是推理系统的背压？

当下游 GPU、KV 或输出链路饱和时，上游限制继续进入的工作，通过排队上限、拒绝、降级、路由或暂停读取，避免无界积压把所有请求拖入超时。

## 二十一、课后练习

1. 用 Level 0 规划器比较 7B 模型在 20、24、48、80 GiB 下的最大平均并发。
2. 固定显存，把 KV Head 从 32 改成 8，解释容量为何变化 4 倍。
1. 在 Level 1 中将 Prompt 从 128 增加到 2048，记录 Prefill、TPOT、峰值显存。
2. 将 `torch.cat` 改成预分配 KV Tensor 的原位写入，比较 Decode TPOT；验证输出 Shape 与缓存长度。
1. 设计一组长度分布：80% 短 Prompt、20% 长 Prompt；说明为什么平均长度压测会漏掉 ITL 尖峰。
2. 为双 RTX 3080 20GB 设计“两个副本”和“ $TP=2$ ”的公平测试，列出必须固定的变量。
1. 给一个聊天 SLO：TTFT $P99 < 2$ s、ITL $P99 < 80$ ms，写出 Admission 与过载策略。
2. 画出客户端取消后 Gateway、Scheduler、Executor、KV Manager 的状态转换。

## 二十二、课程 Checklist

- 能画出请求入口到流式输出的完整架构。
- 能区分 API Server、Scheduler、KV Manager 与 Model Executor。
- 能解释 Prefill 和 Decode 的计算与瓶颈差异。
- 能拆解 TTFT、TPOT、ITL 与端到端延迟。
- 能计算权重和 KV Cache 的数量级。
- 容量预算包含 Workspace、Graph、通信、碎片和安全余量。
- 准入同时考虑请求数、Token 数、KV Block 和 Deadline。
- 压测保留真实 Prompt/Output 长度分布。
- 分开报告 Prompt 与 Generation Tokens/s。
- 用 SLO Goodput 而不是 GPU 利用率作为最终目标。
- 单卡、双副本和 TP 采用同负载公平比较。
- 自动记录 GPU、Compute Capability、驱动、CUDA 和框架版本。
- RTX 4090 标记为 Ada，RTX 5090 标记为 Blackwell。
- FP8/FP4、Transformer Engine、NVLink/NVSwitch 只在真实支持环境验证。
- 用户取消能释放执行与 KV 资源。
- 优化后完成正确性、故障和端到端 SLO 回归。