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

```
Hidden States
      |
   Router / Top-K
      |
 Token Permute + Dispatch
      |
  All-to-All / Local Copy
      |
 Grouped Expert GEMM
      |
  All-to-All / Combine
      |
 Unpermute + Weighted Sum
```

MoE 的性能不是平均专家决定的，而经常由最拥挤专家、最慢 Rank 和最差链路决定。

## 系统与架构原理

### Router：决定模型质量，也决定系统负载

Router 通常输出每个 token 到各专家的 logits，再做 Top-K。推理时要保存专家 ID 和权重，Dispatch 按专家重排 token。

理想状态是专家专业化且负载可控。现实中，自然语言分布、Prompt 类型、批次大小和模型训练都会让某些专家更热门。训练阶段可使用辅助负载均衡损失、bias-based balancing 或其他策略；推理阶段还需要动态放置、冗余专家、路由约束和请求调度。

不能为了“看起来均匀”随意改 Router：修改专家选择可能改变模型输出。运行时的无损优化更常通过移动/复制专家映射、改变请求到 Rank 的分配，而不是替换模型定义的 Top-K。

### Capacity Factor、Padding 与 Token Drop

若一次共有 T 个 token，每个 token 选择 K 个专家，理想平均负载为：

$$
L_{avg}=\frac{TK}{E}
$$

每个专家容量常近似设置为：

$$
C=\left\lceil \gamma\frac{TK}{E}\right\rceil
$$

$\gamma$ 是 capacity factor。容量小会发生 token drop 或 reroute，可能影响质量；容量大则需要 Padding，浪费计算和显存。Dropless MoE 和 block-sparse/grouped kernel 尝试在不丢 token 的同时高效处理变长专家批次。

### Expert Parallel（EP）

EP 将完整专家分散到不同 GPU。假设 8 个专家、4 张 GPU，每张 GPU 可放 2 个专家。每个 Rank 先把非本地 token Dispatch 给目标 Rank，目标计算后再 Combine 回源 Rank。

优点：每张卡只存部分专家，总参数容量可扩展；本地专家权重完整，避免每个专家内部都做 TP。代价：每个 MoE 层通常出现两次全交换语义的数据移动。

### TP、EP 与混合 ETP

| 模式 | 权重放置 | token 通信 | 适用情况 |
| --- | --- | --- | --- |
| TP | 每个专家都切到多卡 | 所有 Rank 参与专家内计算 | 单专家太大或小规模部署 |
| EP | 不同 Rank 放不同完整专家 | token 在专家 Rank 间交换 | 专家多、单专家能放入一组卡 |
| ETP | 先按 EP 分专家，再对本地专家做 TP | Dispatch + 专家内部 Collective | 单专家仍很大或需混合扩展 |
| DP/Attention DP + EP | Attention 可复制/数据并行，专家跨 DP Rank 分布 | Attention 与 MoE 使用不同映射 | DeepSeek/Qwen 等大 MoE 服务 |

并行度越多不代表越快。EP 扩大后，每 Rank 的专家批量可能变小，All-to-All 占比上升；ETP 又会增加专家内部通信。选择必须结合 Batch、Top-K、专家宽度与拓扑。

### Dispatch 与 Combine

Dispatch 通常包含：

1. 计算每个目标专家 token 数；
2. Prefix Sum/Offset；
1. Permute，将同一专家 token 放到连续区间；
2. 按目标 Rank 做 All-to-All/All-to-All-V；
1. 目标 Rank 为本地专家建立分组描述符。

Combine 反向传回专家输出，按原 token ID 反排列，并用 Router 权重求和。这里容易出现很多“小但昂贵”的 kernel：计数、排序、Scatter、Gather、同步和跨流事件。高性能 Runtime 会融合 Router、Top-K、Permutation、量化和通信准备。

### Grouped GEMM 与小 Batch

同一 Rank 上不同专家的 token 数不同。如果对每个专家逐个启动 GEMM，Decode 小 Batch 时容易产生大量小矩阵和 Launch 开销。Grouped GEMM 把多个专家 GEMM 一次提交，Block-sparse kernel 则以块为单位调度不规则稀疏计算。

Prefill 和 Decode 的最优路径不同：

- Prefill：token 多，追求吞吐，可使用连续布局和高吞吐 All-to-All；
- Decode：每步 token 少，追求低延迟，需要更轻量的 Dispatch、固定/掩码布局、CUDA Graph 和低延迟通信。

同一套 MoE kernel 很难同时覆盖两种极端形状，这也是解耦推理与阶段专用 Runtime 的价值。

## 关键公式与性能模型

### 稀疏计算不等于稀疏显存

MoE 的每 token FLOPs 主要与激活专家数 K 有关，但权重显存近似与总专家数 E 有关：

$$
FLOPs_{token}\propto K\cdot FLOPs_{expert}
$$

$$
Memory_{weights}\propto E\cdot Params_{expert}
$$

因此，一个“总参数很大、激活参数较小”的模型可能计算接近中等 Dense 模型，却仍要求多卡保存权重。不能根据 Activated Parameters 推断单卡可加载。

### 负载不均衡

设第 e 个专家收到 $n_e$ 个 token：

$$
R_{imbalance}=\frac{\max_e n_e}{\frac{1}{E}\sum_e n_e}
$$

$R=1$ 表示完全均匀。若一轮必须等待最慢专家，则粗略计算效率上界：

$$
\eta_{balance}\approx\frac{1}{R_{imbalance}}
$$

例如最大专家是平均值的 2 倍，即使其他 Rank 很空闲，这一层也可能只获得约一半的理想并行效率。

### 通信量

隐藏维度为 H、元素字节数为 b。若远程路由比例为 $f_{remote}$ ，一次 Dispatch payload 近似：

$$
B_{dispatch}\approx TKHb\cdot f_{remote}
$$

Combine 还要传回同量级输出：

$$
B_{layer}\approx 2TKHb\cdot f_{remote}
$$

忽略元数据和协议开销时：

$$
T_{comm}\approx T_{dispatch}+T_{combine}\approx 2T_{setup}+\frac{B_{layer}}{BW_{effective}}
$$

小 Decode 消息更受 setup/同步影响；大 Prefill 更受带宽和拓扑影响。

### MoE 层延迟

串行近似：

$$
T_{MoE}=T_{route}+T_{permute}+T_{dispatch}+T_{expert}+T_{combine}+T_{unpermute}
$$

若实现通信计算重叠：

$$
T_{MoE}\approx T_{route}+T_{tail}+\max(T_{comm},T_{expert})
$$

这里的 `tail` 包括无法隐藏的启动、最后一个 chunk、负载不均衡和同步。仅看到 timeline 有重叠，不代表通信已完全隐藏。

### Goodput 与成本

生产系统应比较：

$$
Goodput=\frac{\text{满足 TTFT、TPOT、正确性 SLO 的请求}}{\text{时间}}
$$

并同时报告每请求成本、每输出 token 能耗、显存冗余和网络占用。MoE 可能降低激活计算，却因专家副本、空闲 Rank 和通信增加集群成本。

## DeepSeek 案例中的系统协同

DeepSeek-V3 是总参数 671B、每 token 激活 37B 参数的 MoE 模型。关键点不是单一技巧，而是组合：细粒度专家、共享专家、无辅助损失负载均衡、FP8、通信计算重叠、DeepEP、MTP 与 MLA。

需要避免两个常见误解：

- “只激活 37B”不代表只需保存 37B 权重；完整专家仍需分布在系统中；
- 官方特定 H800 集群的训练/推理结果不能直接套到 RTX 3080、4090 或任意以太网集群。

DeepEP 针对 EP 的 Dispatch/Combine 提供高吞吐与低延迟 All-to-All kernel，并支持低精度路径。当前 V2 实现及硬件/软件要求仍在快速变化，应以对应版本官方仓库为准；双 RTX 3080 不属于它的标准生产验证平台。

## 瓶颈分析方法

### 一：先分清模型瓶颈还是 Runtime 瓶颈

记录 Router 原始分布。如果少数专家天然过热，问题来自模型/数据分布或专家放置；如果路由均匀但耗时不均，检查 GPU 频率、拓扑、后台任务、专家大小和通信。

### 二：逐层记录专家负载

至少采集：

- 每层、每专家 token 数；
- max/mean、CV、空专家比例；
- dropped/rerouted token；
- 每 Rank 接收/发送字节；
- 最慢 Rank 与最慢专家；
- 专家副本命中率和迁移次数。

全模型平均负载可能掩盖某一层的热点。

### 三：拆解 MoE 时间线

```
router -> topk -> permutation
       -> dispatch_a2a -> grouped_gemm
       -> combine_a2a -> unpermute
```

用 NVTX/Nsight Systems 看通信与计算是否重叠；用 Nsight Compute 检查 Grouped GEMM 的 Tensor Core、带宽、Occupancy 和小矩阵效率；用 NCCL/通信库计数器看链路与拓扑。

### 四：区分 Prefill 与 Decode

分别测 Prompt 长度、Batch、并发和输出长度。只测 Prefill 大 Batch 得出的高带宽方案，可能在 Decode 小消息下延迟很差。

### 五：做正确性验证

- Router Top-K 与权重保持一致；
- Permute/Unpermute 恢复原 token 顺序；
- Dispatch/Combine 前后 shape、dtype、token ID 一致；
- Dropless/Drop/Reroute 策略符合模型定义；
- 专家重新映射或副本不会改变专家语义；
- 低精度通信与专家计算满足误差预算。

## 完整可运行实验

### Level 0：CPU Router 负载模拟

保存为 `moe_routing_sim.py` ：

```
#!/usr/bin/env python3
import argparse
import math
import random
import statistics

def route(tokens, experts, topk, skew, capacity_factor, penalty, seed):
    rng = random.Random(seed)
    loads = [0] * experts
    expected = tokens * topk / experts
    capacity = math.ceil(capacity_factor * expected)
    dropped = 0

    for _ in range(tokens):
        # 低编号专家更热门，用于模拟真实工作负载倾斜。
        raw = [rng.gauss(0, 1) - skew * e / max(1, experts - 1)
               for e in range(experts)]
        if penalty == 0.0:
            # 模型原始 Top-K：容量满时丢弃，不偷偷改选其他专家。
            selected = sorted(range(experts), key=lambda e: raw[e], reverse=True)[:topk]
            for e in selected:
                if loads[e] < capacity:
                    loads[e] += 1
                else:
                    dropped += 1
            continue

        selected = set()
        for _ in range(topk):
            candidates = []
            for e in range(experts):
                if e in selected or loads[e] >= capacity:
                    continue
                balanced_score = raw[e] - penalty * loads[e] / max(1.0, expected)
                candidates.append((balanced_score, e))
            if not candidates:
                dropped += 1
                continue
            _, e = max(candidates)
            selected.add(e)
            loads[e] += 1
    return loads, dropped, capacity

def summarize(name, loads, dropped, capacity):
    mean = statistics.mean(loads)
    cv = statistics.pstdev(loads) / mean if mean else 0.0
    ratio = max(loads) / mean if mean else 0.0
    efficiency = mean / max(loads) if max(loads) else 0.0
    print(f"\n[{name}]")
    print("loads=", loads)
    print(f"capacity={capacity} dropped_copies={dropped}")
    print(f"max/mean={ratio:.3f} CV={cv:.3f}")
    print(f"balance_efficiency_upper_bound={efficiency:.3f}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=4096)
    p.add_argument("--experts", type=int, default=16)
    p.add_argument("--topk", type=int, default=2)
    p.add_argument("--skew", type=float, default=1.5)
    p.add_argument("--capacity-factor", type=float, default=1.25)
    p.add_argument("--balance-penalty", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    if a.topk > a.experts:
        p.error("topk 不能大于 experts")

    base = route(a.tokens, a.experts, a.topk, a.skew,
                 a.capacity_factor, penalty=0.0, seed=a.seed)
    balanced = route(a.tokens, a.experts, a.topk, a.skew,
                     a.capacity_factor, penalty=a.balance_penalty, seed=a.seed)
    summarize("model_topk_with_capacity", *base)
    summarize("capacity_aware_simulation", *balanced)
    print("\n注意：第二组修改了路由分数，只用于系统容量学习，")
    print("不能直接替换真实模型 Router，否则可能改变输出质量。")

if __name__ == "__main__":
    main()
```

运行：

```
python moe_routing_sim.py
python moe_routing_sim.py --skew 0 --capacity-factor 1.1
python moe_routing_sim.py --skew 2.5 --capacity-factor 2.0
```

预期现象：倾斜增大时，原始 Top-K 更容易让热门专家达到容量并产生 dropped copies；模拟的负载惩罚会降低 max/mean，但它改变了专家选择，只能帮助理解“均衡与模型语义”的权衡，不是可直接部署的无损优化。

### Level 1：通用 PyTorch 分组专家微基准

这个实验比较“逐 token 启动专家计算”和“先按专家分组再做批量 GEMM”。它是 Runtime 机制实验，不是完整 MoE 模型。

保存为 `grouped_expert_benchmark.py` ：

```
#!/usr/bin/env python3
import argparse
import time
import torch

def arch_name(major, minor):
    if major == 8 and minor in (0, 6, 7):
        return "Ampere"
    if major == 8 and minor == 9:
        return "Ada Lovelace"
    if major == 9:
        return "Hopper"
    if major >= 10:
        return "Blackwell"
    return "Other/Older"

def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def bench(fn, device, warmup, iters):
    for _ in range(warmup):
        fn()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync(device)
    return (time.perf_counter() - t0) * 1000 / iters

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=512)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--experts", type=int, default=8)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    torch.manual_seed(a.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print("torch=", torch.__version__, "cuda_runtime=", torch.version.cuda)
    print("device=", device, "dtype=", dtype)
    if device.type == "cuda":
        prop = torch.cuda.get_device_properties(0)
        print("gpu=", prop.name, "cc=", f"{prop.major}.{prop.minor}",
              "arch=", arch_name(prop.major, prop.minor),
              "vram_GiB=", round(prop.total_memory / 2**30, 2))

    x = torch.randn(a.tokens, a.hidden, device=device, dtype=dtype)
    weights = torch.randn(a.experts, a.hidden, a.hidden,
                          device=device, dtype=dtype) / a.hidden**0.5
    expert_ids = torch.randint(0, a.experts, (a.tokens,), device=device)

    @torch.inference_mode()
    def token_loop():
        out = torch.empty_like(x)
        for t in range(a.tokens):
            out[t] = x[t] @ weights[expert_ids[t]]
        return out

    @torch.inference_mode()
    def grouped():
        out = torch.empty_like(x)
        for e in range(a.experts):
            idx = torch.where(expert_ids == e)[0]
            if idx.numel():
                out[idx] = x[idx] @ weights[e]
        return out

    ref = token_loop()
    opt = grouped()
    max_err = (ref - opt).abs().max().item()
    loop_ms = bench(token_loop, device, a.warmup, a.iters)
    grouped_ms = bench(grouped, device, a.warmup, a.iters)
    counts = torch.bincount(expert_ids, minlength=a.experts).cpu().tolist()
    print("tokens_per_expert=", counts)
    print(f"max_abs_error={max_err:.6g}")
    print(f"token_loop={loop_ms:.3f} ms")
    print(f"grouped={grouped_ms:.3f} ms")
    print(f"speedup={loop_ms/grouped_ms:.3f}x")
    print("结果仅代表本机形状；真实框架会使用融合 Grouped GEMM kernel。")

if __name__ == "__main__":
    main()
```

运行：

```
python grouped_expert_benchmark.py
python grouped_expert_benchmark.py --tokens 4096 --hidden 1024 --experts 16
```

CPU 或显存不足时保持默认值；GPU 上逐步增大参数。预期分组执行减少 Python/Kernel Launch，并把许多 GEMV/小 GEMM 合成较大的 GEMM。真实 Runtime 还应融合 Top-K、Permutation、Grouped GEMM 和 Combine，不能把这个微基准速度当作端到端 MoE 收益。

### Level 2：Expert Parallel 专项

### vLLM 标准后端功能路径

先核对安装版本：

```
vllm --version
vllm serve --help | grep -E 'expert-parallel|all2all|eplb'
```

当前 vLLM 使用 `--enable-expert-parallel` 与 `--all2all-backend` 。以下只展示配置形态，模型是否支持、显存是否足够和 DP/EP 语义应以当前版本文档为准：

```
CUDA_VISIBLE_DEVICES=0,1 vllm serve /path/to/supported-moe-model \
  --tensor-parallel-size 1 \
  --data-parallel-size 2 \
  --enable-expert-parallel \
  --all2all-backend allgather_reducescatter
```

双 RTX 3080 20GB 可先选择总权重能在两卡 EP 布局中容纳的小型 MoE checkpoint。注意：Attention/共享参数可能在 DP Rank 复制，不能简单用“模型总字节÷2”判断显存。

### DeepEP 高吞吐与低延迟路径

大规模数据中心部署可按 Prefill/Decode 分别选择高吞吐和低延迟后端：

```
# Prefill 倾向
--all2all-backend deepep_high_throughput

# Decode 倾向
--all2all-backend deepep_low_latency
```

DeepEP V2、NCCL、CUDA、GPU 架构和网络支持会变化，安装前必须阅读对应版本矩阵。H100/H200 属于 Hopper，B100/B200/GB200 属于 Blackwell；双 RTX 3080 的回退是标准 `allgather_reducescatter` /框架可用后端和 Level 1 微基准，不能宣称等价验证 DeepEP。

### EPLB 与冗余专家

运行时可根据窗口内负载重新映射专家，或为热门专家创建冗余副本。当前 vLLM 提供 EPLB 配置，但副本会占用显存：

```
vllm serve /path/to/supported-moe-model \
  --enable-expert-parallel \
  --enable-eplb \
  --eplb-config '{
    "window_size":1000,
    "step_interval":3000,
    "num_redundant_experts":2,
    "log_balancedness":true,
    "use_async":true
  }'
```

先在离线 Trace 上确定热点稳定性和副本显存成本。过于频繁地搬专家权重，可能让重平衡开销超过收益。

## 硬件适配与不可等价边界

| 架构 | 常见 GPU | 可验证内容 | 边界 |
| --- | --- | --- | --- |
| Ampere | RTX 3080/3090、A100 | 路由、Grouped GEMM、双卡标准通信；A100 数据中心互联 | RTX 3080 无 FP8 Tensor Core，双卡通常无 NVLink |
| Ada Lovelace | RTX 4090、L40/L40S | 更强单卡低精度/Grouped GEMM，按框架支持验证 | RTX 4090 不是 Blackwell，通常无 NVLink |
| Hopper | H100/H200 | FP8、Transformer Engine、DeepEP/高性能 EP、NVLink/NVSwitch | 消费卡不能模拟数据中心互联 |
| Blackwell | RTX 5090、B100/B200/GB200 | FP4/FP8、更新 Tensor Core、NVLink 5/NVSwitch 专项 | RTX 5090 是 Blackwell，但不等同 GB200 NVL 系统 |

FP8/FP4 是否加速还取决于模型权重格式、kernel、缩放粒度、框架版本和 GPU SKU。检测到架构名称不等于对应路径已经可用。

## 优化策略：从局部 Kernel 到全局 Runtime

### 路由与布局

- 保留模型 Top-K 语义，优化 Router/Top-K Fusion；
- 按历史热点重映射专家到 Rank；
- 将经常共同激活的专家放在高带宽域；
- 对稳定热门专家做受控复制；
- Prefix/租户/请求类型感知路由，提高专家局部性。

### 通信

- Prefill 用带宽型 Dispatch，Decode 用低延迟小消息路径；
- 采用 All-to-All-V，避免按最大专家负载 Padding；
- FP8/低精度 Dispatch 前先验证误差；
- 多 Chunk Pipeline 隐藏通信；
- 将高频 EP 放在 NVLink/NVSwitch 域，跨节点做分层交换；
- 绑定 GPU-NIC NUMA，避免 CPU bounce buffer。

### 计算

- Grouped GEMM/Block-sparse kernel；
- Token Permutation、激活与反排列融合；
- 小 Batch Persistent Kernel 或 CUDA Graph；
- 共享专家与路由专家并行/重叠；
- 按专家形状 Auto-tune tile 与并发；
- 量化专家权重时同时评估带宽、解量化和质量。

### 调度

- Continuous Batching 不只看请求，还要看专家负载；
- 避免把同一热门领域请求集中到一个迭代；
- 预测长输出与专家热点，做 Rank/副本选择；
- P/D 解耦后为两阶段选择不同 All-to-All backend；
- 用 SLO 与 Goodput 驱动扩缩容，而不是只看 GPU Util。

## 未来 AI Runtime：从静态执行器到自治控制系统

### 1\. Shape-aware Runtime

未来 Runtime 会按 Prompt 长度、Decode Batch、Top-K 分布、KV 命中和硬件拓扑动态选择 kernel、图和并行策略，而不是启动时固定一组参数。

### 2\. Stage-aware Runtime

Encode、Prefill、Decode、Draft、Verify、MoE Dispatch、Tool/Agent 执行会被视为不同阶段。每个阶段可拥有独立资源池、精度、并行度、缓存和 SLO。

### 3\. Data-movement-first Runtime

性能瓶颈越来越多地来自权重、KV、激活、专家 token 和多模态 embedding 的位置。Runtime 需要统一管理 HBM、DRAM、NVMe、远程存储与网络，把“数据在哪里”提升为一等调度信息。

### 4\. Topology-aware Runtime

同一节点内 NVLink、跨节点 RDMA、PCIe、NUMA 和共享 Switch 的代价不同。未来 Planner 会把 TP、EP、CP、KV 传输和专家副本联合映射到物理拓扑，而非只按 GPU 数量切分。

### 5\. Compiler-Runtime Co-design

Compiler 生成多组候选 kernel、CUDA Graph 和通信计划；Runtime 根据实时 shape 与拥塞选择版本。Triton、CUTLASS、DeepGEMM、DeepEP 和框架调度将不再是彼此独立的优化点。

### 6\. SLO-driven Autonomy

Runtime 持续观测 TTFT、TPOT、Goodput、能耗、KV 热点与专家负载，自动完成：

- P/D 池扩缩容；
- 专家复制、迁移和 EPLB；
- 动态 Speculative K；
- KV 分层、预取和淘汰；
- Batch/Chunk 调整；
- 故障迁移与降级路径；
- 低负载节能和频率控制。

真正成熟的 AI Runtime 更像数据库优化器、分布式操作系统和实时控制器的结合，而不只是一个 `model.forward()` 循环。

## 结果分析模板

```
模型、总参数/激活参数、Top-K：
MoE 层数、专家数、共享专家：
GPU、拓扑、EP/TP/DP/PP：
框架、All-to-All backend、dtype：
Prefill/Decode Batch 与长度：
每层 max/mean、CV、空专家率：
Dropped/Rerouted token：
Route/Permute/Dispatch/Expert/Combine ms：
Grouped GEMM TFLOPS/带宽/Occupancy：
通信有效带宽、最慢 Rank：
TTFT、TPOT、tokens/s、Goodput P50/P99：
专家副本显存与迁移开销：
正确性/误差结果：
```

## 常见错误与排查

### “MoE 激活参数小，所以单卡能放下”

错误。激活参数决定单 token 计算量，显存仍要容纳所有本地专家、共享层、Attention、KV 和工作区。先算总权重与实际分片。

### GPU 平均利用率高，但 TPOT 很差

平均值可能掩盖最慢 Rank。检查每 Rank token 数、All-to-All 等待、最大专家、P99 kernel 时间和同步点。

### 专家负载均匀，仍然很慢

可能是每专家 Batch 太小、Grouped GEMM 未生效、Permutation kernel 太多、通信 setup 占主导或 EP 过大。均衡只是必要条件之一。

### All-to-All 带宽低

检查消息是否太小、是否 Padding、GPU-NIC/NUMA、跨节点比例、backend、P2P、NVLink 域、网络拥塞和是否发生 Host staging。

### EPLB 开启后抖动更大

窗口太短、热点变化快、迁移频率太高或专家副本挤占 KV 空间。增大观测窗口、降低迁移频率，并把重平衡放到异步/低峰路径。

### 低精度后没有加速

可能仍由通信/调度主导，或 kernel 在解量化、缩放和格式转换中消耗收益。检查实际 kernel dtype、Tensor Core 指标和端到端时间。

### 双 RTX 3080 EP 比单卡量化慢

PCIe Dispatch/Combine 和小 Batch 开销可能超过专家分片收益。双卡实验可以证明机制，不保证生产加速；应与 TP、单卡量化和 CPU Offload 基线比较。

## 优化前后对照

| 维度 | 基础 MoE Runtime | 优化后 Runtime |
| --- | --- | --- |
| Router | 独立 Softmax/Top-K | 融合 Router/Top-K/计数 |
| Token 布局 | Padding、逐专家循环 | Dropless、Permutation Fusion、Grouped GEMM |
| 通信 | 通用 Collective | Prefill/Decode 专用 All-to-All、拓扑感知 |
| 负载 | 静态专家映射 | 观测、EPLB、冗余专家、异步迁移 |
| 并行 | 固定 TP/EP | TP+EP+DP/Attention DP+P/D 阶段组合 |
| 精度 | BF16/FP16 全链路 | 受控 FP8/FP4/量化通信与专家权重 |
| 调度目标 | GPU Util/tokens/s | TTFT+TPOT+Goodput+成本+能耗 |

## 面试题与答案

### 1\. MoE 为什么能扩大参数量而不同比例增加 FLOPs？

因为每个 token 只激活 Top-K 专家，计算量与 K 更相关，而总容量与专家总数 E 相关。但所有专家权重仍需存储，通信和路由成本也会增加。

### 2\. MoE 推理最主要的三个系统瓶颈是什么？

专家负载不均衡、Dispatch/Combine All-to-All 通信，以及每专家 Batch 太小导致的低效 GEMM/Launch。不同负载下三者占比不同。

### 3\. EP 与 TP 的区别是什么？

TP 把每个专家权重切到多卡，所有 Rank 共同计算；EP 把不同完整专家放到不同 Rank，并把 token 发送到专家所在位置。ETP 则组合两者。

### 4\. 为什么 max/mean 比平均 token 数更重要？

MoE 层通常要等待所有 Rank/专家完成，最拥挤专家决定尾部时间。平均负载看起来正常，也可能被单个热点拖慢。

### 5\. Capacity Factor 太大或太小分别有什么问题？

太小会 Drop/Reroute token，可能影响质量；太大需要更多 Padding、显存和计算。Dropless/变长 kernel 能减少这种二选一，但实现更复杂。

### 6\. 为什么 Decode 需要低延迟 All-to-All？

Decode 每轮 token 少，消息小，固定 setup、同步与 kernel launch 占比高；高峰值带宽不等于低小消息延迟。

### 7\. 共享专家的作用是什么？

共享专家始终激活，承载通用知识；路由专家更专注特定模式。系统上可把共享专家计算与 Dispatch/路由专家计算重叠，但会增加固定计算。

### 8\. 运行时如何无损地缓解热点专家？

保持 token 到专家语义不变，通过专家副本、专家到 Rank 的重新映射、请求调度和拓扑放置分散负载。若直接改变 Top-K，需要重新验证模型质量。

### 9\. MoE 一定比 Dense 快吗？

不一定。模型规模、Batch、Top-K、专家宽度、通信、负载均衡、kernel 和硬件决定结果。小 Batch、慢互联或实现不成熟时可能更慢。

### 10\. 未来 AI Runtime 的核心变化是什么？

从静态模型执行器变成数据、阶段、拓扑和 SLO 感知的自治控制系统，联合选择并行、kernel、缓存、通信、批处理、解耦和扩缩容策略。

## 课后练习

1. 计算 $T=8192,E=64,K=2,\gamma=1.25$ 时的平均负载与专家容量。
2. 用 Level 0 扫描 skew、capacity factor 和 penalty，画出 Drop 与 max/mean 曲线。
1. 修改 Level 0，让热门专家复制两份，并设计 token 到副本的无损分配策略。
2. 把 Level 1 扩展为 Top-2，加 Router 权重并验证 Combine 数值正确性。
1. 用 `torch.profiler` 或 Nsight Systems 比较逐 token 与分组版本的 kernel 数。
2. 在双 GPU 上实现 Top-1 专家 Dispatch，记录 PCIe P2P 带宽和端到端延迟。
1. 为 Prefill 与 Decode 分别设计 All-to-All benchmark shape，解释为何最优 backend 不同。
2. 设计一个 Runtime 控制器：输入 TTFT/TPOT、专家负载、KV 占用和网络拥塞，输出 P/D 比例、Batch、EPLB 和 Speculative K。

## Checklist

- 已区分总参数、激活参数、权重显存和单 token FLOPs；
- 已记录每层每专家负载，而非只看全局平均；
- 已计算 max/mean、CV、Drop/Reroute 和空专家率；
- 已拆分 Router、Permutation、Dispatch、GEMM、Combine；
- 已分别测 Prefill 与 Decode；
- 已确认 Grouped GEMM/融合 kernel 实际生效；
- 已测 All-to-All 有效带宽、小消息延迟和最慢 Rank；
- 已按物理拓扑规划 EP/TP/DP，而非只按 GPU 数；
- 已评估专家副本和 EPLB 的显存/迁移成本；
- 已验证路由、反排列、Combine 与低精度正确性；
- 已与 Dense、TP、量化和共置/解耦基线比较；
- RTX 4090 正确标为 Ada Lovelace；RTX 5090 正确标为 Blackwell；
- 双 RTX 3080 未被描述为 NVLink/DeepEP 数据中心环境；
- FP8/FP4/Transformer Engine/NVSwitch 路径均做能力检测；
- 所有性能结论都绑定模型、版本、硬件、拓扑和 workload；
- Runtime 优化以 TTFT、TPOT、Goodput、成本和正确性共同验收。