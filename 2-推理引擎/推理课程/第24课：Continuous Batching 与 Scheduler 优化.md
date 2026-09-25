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

## 七、调度的四个约束

### 7.1 Sequence Budget

`max_num_seqs` 限制一轮中可处理的最大活动序列数。它控制 Batch Width、调度元数据、CUDA Graph Shape 和部分 Workspace。

设置过小会让 GPU 吃不饱；设置过大可能增加 KV 压力、单步延迟、Graph Pool 和 Scheduler 成本。

### 7.2 Token Budget

`max_num_batched_tokens` 限制一轮最多处理多少输入 Token。对 Decode 请求，每个活动序列通常消耗一个 Token；Prefill 可能一次消耗数百或数千 Token。

一轮预算近似为：

$$
B_{token}=N_{decode}+\sum_j C^{prefill}_j
$$

其中 C^{prefill}\\\_j 是本轮为第 j 个 Prompt 处理的 Chunk 大小。

只看 Sequence 数会把“128 个 Decode Token”和“128 个 8K Prompt”误认为同样成本。

### 7.3 KV Budget

每个新 Prefill 和每个 Decode Token 都可能申请新 KV Block。调度前必须确认：

$$
KV_{used}+KV_{reserved}+KV_{new}\le KV_{capacity}
$$

否则要等待、拒绝、抢占、驱逐可重算状态或降低并发。先执行后发现 OOM，不是调度策略。

### 7.4 Time 与 SLO Budget

即使 Token 和 KV 容量允许，一轮太大也可能让 Decode ITL 超标。实际目标是：

$$
\max Goodput \quad\text{s.t.}\quad P99(TTFT)\le SLO_{TTFT},\;P99(ITL)\le SLO_{ITL}
$$

最大吞吐配置和最大 SLO Goodput 配置可能不同。

## 八、Prefill 与 Decode 的优先级冲突

### 8.1 Prefill 优先

先处理等待 Prompt，可降低新请求 TTFT，但大 Prompt 可能占满一轮 Token Budget，让已有会话的 Decode 暂停，形成 ITL 尖峰。

适合首 Token 极敏感、输出较短、负载可控的场景。

### 8.2 Decode 优先

先为所有 Running Decode 请求保留一个 Token，再把剩余预算给 Prefill。它保护流式体验和 ITL，却可能让新请求长时间等不到 Prefill。

适合聊天、语音交互和稳定流式输出。

### 8.3 Chunked Prefill

长 Prompt 不再一次占完整轮次，而是按剩余 Token Budget 切片：

$$
C_i=\min(PromptRemaining_i, BudgetRemaining)
$$

优点：

- Decode 与 Prefill 可以在相邻或同一调度周期共存；
- 长 Prompt 不易垄断 GPU；
- 预算更容易满足；
- ITL 尖峰通常更可控。

代价：

- Prefill 被拆成更多迭代，增加 Launch 和调度开销；
- 小 Chunk 的 GEMM 效率可能下降；
- 新请求 TTFT 可能增加；
- Partial Prefill 状态更复杂。

因此 Chunk Size 是 TTFT、ITL 和吞吐之间的旋钮，不存在跨硬件通用的最优值。

## 九、调度策略

### 9.1 FIFO

实现简单且容易解释，但长 Prompt 或超长输出会导致队头阻塞。

### 9.2 Shortest Remaining Processing Time

优先短任务可降低平均延迟，却可能让长请求饥饿；而且输出长度通常只是上限，不能准确预知。

### 9.3 Deadline-aware

按距离 SLO Deadline 的紧迫程度调度。需要可靠的成本估计，否则可能频繁插队并造成抖动。

### 9.4 Priority 与 Fair Share

在线、离线、付费等级或系统任务可设不同权重。但 Priority 不是无条件插队，必须有 Aging 或租户配额避免低优先级永久饥饿。

### 9.5 Cache-aware

优先选择可复用 Prefix/KV 的请求能减少 Prefill，但可能牺牲 FIFO 公平性。需要把节省的计算、等待增长和跨租户安全一起评估。

### 9.6 Capacity-aware

保守策略只启动保证不被 KV 容量中途打断的请求；激进策略尽量填满 GPU，容量不足时允许暂停或抢占。前者稳定，后者可能提高吞吐但增加重算与尾延迟。

## 十、性能模型

### 10.1 静态 Decode Padding Waste

Batch 内输出长度为 $O_i$ ：

$$
Waste_{static}=1-\frac{\sum_i O_i}{B\cdot\max_i O_i}
$$

Continuous Batching 用新请求补位，长期有效率更接近活动槽位率，但仍会受到队列不足、KV 不足和 Shape Bucket 约束。

### 10.2 每轮成本

简化模型：

$$
T_{iter}=T_{schedule}+T_{launch}+T_{prefill}(N_p)+T_{decode}(N_d,S)+T_{comm}+T_{sample}
$$

$N_p$ 是本轮 Prefill Token 数， $N_d$ 是 Decode 序列数，S 表示上下文长度分布。调度优化只有在节省的 GPU 空洞或 Padding 大于新增的 Scheduler、Pack 和 Block Table 开销时才有收益。

### 10.3 排队稳定性

若请求到达率为 $\lambda$ 、系统在当前长度分布下的有效服务率为 $\mu$ ：

$$
\rho=\frac{\lambda}{\mu}
$$

当 $\rho$ 接近 1，排队和 P99 会非线性上升。Continuous Batching 能提高 $\mu$ ，却不能让超过物理容量的系统稳定；仍需 Admission Control 和 Backpressure。

### 10.4 Little 定律在显存规划中的含义

$$
L=\lambda W
$$

平均在途请求数 L 随到达率和停留时间增长。在途请求越多，KV 占用越高；为了追求吞吐而拉长 W，可能反过来压缩可用并发并触发抢占。

## 十一、调度器数据结构与关键路径

### 11.1 请求状态

每个 Request 至少包含：

- Request ID、租户、优先级、Arrival 与 Deadline；
- Prompt Token 数、已 Prefill Token 数；
- 已生成 Token 数、最大输出长度、停止状态；
- KV Block Table 与 Prefix 复用信息；
- Sampling、LoRA、约束解码和多模态元数据；
- 取消、暂停、重算和流式输出状态。

### 11.2 Scheduler 热路径原则

- 避免每轮扫描无关的全部历史请求；
- 队列操作保持近似 O(1) 或 $O(\log N)$ ；
- 避免 Python 对每个 Token 做重型对象创建；
- 调度结果使用紧凑数组传递给 Executor；
- KV 预留与状态转换必须原子化；
- Metrics 不应持有全局锁或同步 GPU。

### 11.3 Packed Batch

将变长 Prompt Padding 到最大长度会浪费计算。生产 Runtime 常把有效 Token 紧凑排列，并用位置、序列边界和 Block Table 还原逻辑关系：

```
Padded : [A A A A][B B _ _][C C C _]
Packed : [A A A A B B C C C]
Meta   : offsets=[0,4,6,9]
```

Packed Input 提高有效 Token 密度，但要求 Attention Kernel 和调度元数据正确处理边界。

## 十二、瓶颈分析方法

### 12.1 必须观测的指标

请求级：

- Queue Time、TTFT、TPOT、ITL P50/P95/P99、E2E；
- 完成、拒绝、超时、取消和重试率；
- Prompt/Output 长度与优先级分布。

Scheduler 级：

- Waiting/Running/Paused 数；
- 每轮 Prefill Token、Decode Sequence、Token Budget 利用率；
- Batch Width 与有效 Token 密度；
- 调度耗时、Iteration 时间和空轮次；
- 抢占、重算、驱逐、Prefix 命中。

资源级：

- KV Block 使用率与碎片；
- GPU SM、Tensor Core、HBM 和功耗；
- CPU Tokenizer/Scheduler 利用率；
- TP/NCCL 和网络时间。

### 12.2 症状到根因

| 症状 | 证据 | 常见根因 |
| --- | --- | --- |
| TTFT 高、ITL 正常 | Waiting 增长、Prefill 少 | Decode 过度优先、Token Budget 太小 |
| TTFT 正常、ITL 尖峰 | 大 Prefill 与尖峰重合 | Prefill 未切片或 Chunk 太大 |
| GPU 低利用率、队列有请求 | 空轮次、CPU Gap | Scheduler/Tokenizer 慢、同步或 Pack 开销 |
| GPU 高利用率、Goodput 低 | SLO 失败、队列爆炸 | 并发过高、Batch 过大、无背压 |
| KV 满且频繁抢占 | Recompute/Swap 增长 | `max_num_seqs` 过高、长上下文混跑 |
| 短请求 P99 很差 | 长 Prompt 位于队首 | FIFO 队头阻塞、无长度隔离 |
| 吞吐随 Token Budget 增大后下降 | Iteration 变长、ITL 上升 | 超过最佳 Shape 或 SLO 限制 |

### 12.3 调优顺序

1. 固定模型、精度、Sampling 和真实长度分布。
2. 单请求测 Prefill 与 Decode 基线。
1. 固定 `max_num_seqs` ，扫描 Token Budget。
2. 固定 Token Budget，扫描最大并发。
1. 比较 Prefill 优先、Decode 优先与 Chunked Prefill。
2. 检查 KV 抢占、重算和取消回收。
1. 使用开环到达率找出 SLO Goodput 峰值。
2. 最后用 Nsight 确认 CPU、GPU、通信和空洞。

## 十三、实验分层

### 13.1 Level 0：CPU/通用模拟

用离散事件模型比较静态请求级 Batch 与 Decode 优先的 Continuous Batching。无需 GPU，可观察 TTFT、E2E、输出吞吐和静态 Padding Waste。

### 13.2 Level 1：常见 NVIDIA GPU

自动检测 GPU、Compute Capability、显存、CUDA 与 PyTorch；比较“每请求单独 Launch”与“每轮压紧活动序列”的 Decode 执行。

### 13.3 Level 2：真实 Runtime

使用 vLLM 的 `max_num_batched_tokens` 、 `max_num_seqs` 和 Chunked Prefill 做开环压测。参数随版本变化，先记录版本和 CLI 帮助。

## 十四、Level 0 实战：连续组批离散事件模拟器

将以下代码保存为 `continuous_batching_sim.py` 。

```
#!/usr/bin/env python3
import argparse
import copy
import math
import random
import statistics
from dataclasses import dataclass, field

@dataclass
class Request:
    rid: int
    arrival_ms: float
    prompt_tokens: int
    output_tokens: int
    prompt_remaining: int = 0
    generated: int = 0
    first_token_ms: float | None = None
    finish_ms: float | None = None
    token_times: list[float] = field(default_factory=list)

    def reset(self):
        self.prompt_remaining = self.prompt_tokens
        self.generated = 0
        self.first_token_ms = None
        self.finish_ms = None
        self.token_times = []
        return self

def percentile(values, q):
    values = sorted(values)
    index = min(len(values) - 1, max(0, round((len(values) - 1) * q)))
    return values[index]

def make_workload(args):
    rng = random.Random(args.seed)
    now = 0.0
    requests = []
    for rid in range(args.requests):
        if rid:
            now += rng.expovariate(args.request_rate / 1000.0)
        prompt = rng.randint(args.prompt_min, args.prompt_max)
        output = rng.randint(args.output_min, args.output_max)
        requests.append(Request(rid, now, prompt, output).reset())
    return requests

def static_batching(source, args):
    requests = copy.deepcopy(source)
    waiting, completed = [], []
    next_id, now = 0, 0.0
    total_slots = useful_slots = 0

    while len(completed) < len(requests):
        while next_id < len(requests) and requests[next_id].arrival_ms <= now:
            waiting.append(requests[next_id])
            next_id += 1

        if not waiting:
            now = max(now, requests[next_id].arrival_ms)
            continue

        batch = waiting[: args.max_seqs]
        del waiting[: len(batch)]

        # 静态 Prompt Batch 按最长 Prompt Padding。
        padded_prefill_tokens = len(batch) * max(r.prompt_tokens for r in batch)
        now += args.base_ms + padded_prefill_tokens * args.prefill_ms_per_token

        width = len(batch)
        max_output = max(r.output_tokens for r in batch)
        for step in range(max_output):
            now += args.base_ms + width * args.decode_ms_per_seq
            total_slots += width
            for r in batch:
                if step < r.output_tokens:
                    useful_slots += 1
                    r.generated += 1
                    r.token_times.append(now)
                    if r.first_token_ms is None:
                        r.first_token_ms = now
                    if r.generated == r.output_tokens:
                        r.finish_ms = now

            while next_id < len(requests) and requests[next_id].arrival_ms <= now:
                waiting.append(requests[next_id])
                next_id += 1

        completed.extend(batch)

    return completed, now, useful_slots / total_slots

def continuous_batching(source, args):
    requests = copy.deepcopy(source)
    waiting, running, completed = [], [], []
    next_id, now = 0, 0.0

    while len(completed) < len(requests):
        while next_id < len(requests) and requests[next_id].arrival_ms <= now:
            waiting.append(requests[next_id])
            next_id += 1

        if not running and not waiting:
            now = max(now, requests[next_id].arrival_ms)
            continue

        token_budget = args.token_budget
        decode_reqs = [r for r in running if r.prompt_remaining == 0]
        decode_reqs = decode_reqs[:token_budget]
        token_budget -= len(decode_reqs)

        # 先推进已有的 Partial Prefill，再接收新请求。
        prefill_plan = []
        for r in [x for x in running if x.prompt_remaining > 0]:
            if token_budget <= 0:
                break
            take = min(r.prompt_remaining, token_budget)
            prefill_plan.append((r, take))
            token_budget -= take

        while waiting and len(running) < args.max_seqs and token_budget > 0:
            r = waiting[0]
            if not args.chunked_prefill and r.prompt_remaining > token_budget:
                break
            waiting.pop(0)
            running.append(r)
            take = min(r.prompt_remaining, token_budget)
            prefill_plan.append((r, take))
            token_budget -= take

        prefill_tokens = sum(take for _, take in prefill_plan)
        if not decode_reqs and prefill_tokens == 0:
            # 容量配置不可能推进请求，防止静默死循环。
            raise RuntimeError("scheduler made no progress; increase token budget")

        now += (
            args.base_ms
            + prefill_tokens * args.prefill_ms_per_token
            + len(decode_reqs) * args.decode_ms_per_seq
        )

        for r, take in prefill_plan:
            r.prompt_remaining -= take

        for r in decode_reqs:
            r.generated += 1
            r.token_times.append(now)
            if r.first_token_ms is None:
                r.first_token_ms = now
            if r.generated == r.output_tokens:
                r.finish_ms = now
                completed.append(r)

        finished_ids = {r.rid for r in completed}
        running = [r for r in running if r.rid not in finished_ids]

    useful = sum(r.output_tokens for r in completed)
    return completed, now, 1.0 if useful else 0.0

def summarize(name, requests, makespan, slot_efficiency):
    ttft = [r.first_token_ms - r.arrival_ms for r in requests]
    e2e = [r.finish_ms - r.arrival_ms for r in requests]
    itl = []
    for r in requests:
        itl.extend(b - a for a, b in zip(r.token_times, r.token_times[1:]))
    outputs = sum(r.output_tokens for r in requests)
    duration_s = makespan / 1000.0
    print(f"\n=== {name} ===")
    print(f"Makespan              : {makespan:.2f} ms")
    print(f"Request throughput    : {len(requests) / duration_s:.2f} req/s")
    print(f"Output throughput     : {outputs / duration_s:.2f} token/s")
    print(f"TTFT P50 / P99        : {percentile(ttft, .50):.2f} / {percentile(ttft, .99):.2f} ms")
    print(f"E2E  P50 / P99        : {percentile(e2e, .50):.2f} / {percentile(e2e, .99):.2f} ms")
    print(f"ITL  P50 / P99        : {percentile(itl, .50):.2f} / {percentile(itl, .99):.2f} ms")
    print(f"Decode slot efficiency: {slot_efficiency * 100:.2f}%")

def parse_args():
    p = argparse.ArgumentParser(description="Static vs continuous batching simulator")
    p.add_argument("--requests", type=int, default=120)
    p.add_argument("--request-rate", type=float, default=300.0)
    p.add_argument("--prompt-min", type=int, default=32)
    p.add_argument("--prompt-max", type=int, default=512)
    p.add_argument("--output-min", type=int, default=8)
    p.add_argument("--output-max", type=int, default=96)
    p.add_argument("--max-seqs", type=int, default=16)
    p.add_argument("--token-budget", type=int, default=512)
    p.add_argument("--base-ms", type=float, default=0.12)
    p.add_argument("--prefill-ms-per-token", type=float, default=0.003)
    p.add_argument("--decode-ms-per-seq", type=float, default=0.055)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--chunked-prefill", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()

def main():
    args = parse_args()
    if args.token_budget < args.max_seqs:
        raise ValueError("token_budget should be >= max_seqs")
    workload = make_workload(args)
    static_result = static_batching(workload, args)
    continuous_result = continuous_batching(workload, args)
    summarize("Static request-level batching", *static_result)
    summarize("Continuous iteration-level batching", *continuous_result)
    print("\nModel boundary: numbers are synthetic; compare policies, not hardware speed.")

if __name__ == "__main__":
    main()
```

### 14.1 运行命令

```
python3 continuous_batching_sim.py

# 扫描 Token Budget
python3 continuous_batching_sim.py --token-budget 128
python3 continuous_batching_sim.py --token-budget 512
python3 continuous_batching_sim.py --token-budget 2048

# 增加长度离散度，观察静态 Padding
python3 continuous_batching_sim.py --output-min 4 --output-max 256

# 提高到达率，观察排队与 P99
python3 continuous_batching_sim.py --request-rate 800
```

### 14.2 预期现象

- 静态 Batch 的 Decode Slot Efficiency 小于 100%，输出长度越离散，浪费越明显。
- Continuous Batching 的模拟有效槽位率为 100%，因为完成请求立即退出且不执行 Padding。
- Continuous Batching 通常提高 Output Tokens/s，并改善短请求的 E2E。
- Token Budget 太小会把长 Prompt 切得很碎，TTFT 可能上升。
- 到达率超过模拟服务能力后，两种策略的 P99 都会变差；Continuous Batching 不能突破物理容量。

### 14.3 结果边界

该模型只用于解释调度关系，没有模拟真实 GEMM 非线性、KV 带宽、上下文长度对 Decode 的影响、CUDA Graph、TP 通信和 Cache 命中。模拟中的 100% 槽位效率不代表真实 GPU 达到 100% 利用率。

## 十五、Level 1 实战：GPU Decode 动态压紧

以下实验让多个不同输出长度的“请求”执行同一个小型 Decoder。Baseline 每个请求单独 Launch；Continuous 版本每轮把仍活跃的请求压成一个 Batch，完成请求立即退出。

将代码保存为 `gpu_continuous_decode.py` 。

```
#!/usr/bin/env python3
import argparse
import platform
import random
import statistics
import sys

import torch
import torch.nn as nn

class TinyDecoder(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden * 4, bias=False),
            nn.GELU(),
            nn.Linear(hidden * 4, hidden, bias=False),
            nn.LayerNorm(hidden),
        )

    def forward(self, x):
        return self.net(x)

def time_cuda(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end)

def baseline(model, initial, lengths):
    outputs = []
    calls = 0
    for state, steps in zip(initial, lengths):
        x = state.unsqueeze(0)
        for _ in range(steps):
            x = model(x)
            calls += 1
        outputs.append(x.squeeze(0))
    return torch.stack(outputs), calls

def continuous(model, initial, lengths):
    states = [x.clone() for x in initial]
    remaining = list(lengths)
    calls = 0
    while any(x > 0 for x in remaining):
        active = [i for i, value in enumerate(remaining) if value > 0]
        batch = torch.stack([states[i] for i in active])
        batch_out = model(batch)
        calls += 1
        for row, request_id in enumerate(active):
            states[request_id] = batch_out[row]
            remaining[request_id] -= 1
    return torch.stack(states), calls

def parse_args():
    p = argparse.ArgumentParser(description="GPU continuous decode compaction")
    p.add_argument("--requests", type=int, default=32)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--min-steps", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=64)
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()

def main():
    args = parse_args()
    if not torch.cuda.is_available():
        print("CUDA GPU not found. Run the Level 0 simulator on this machine.")
        sys.exit(0)

    dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[args.dtype]
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is not supported by this GPU/PyTorch stack")

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    prop = torch.cuda.get_device_properties(0)
    lengths = [rng.randint(args.min_steps, args.max_steps) for _ in range(args.requests)]
    initial = torch.randn(args.requests, args.hidden, device=device, dtype=dtype)
    model = TinyDecoder(args.hidden).to(device=device, dtype=dtype).eval()

    print("=== Environment ===")
    print(f"Python             : {platform.python_version()}")
    print(f"PyTorch            : {torch.__version__}")
    print(f"CUDA runtime       : {torch.version.cuda}")
    print(f"GPU                : {prop.name}")
    print(f"Compute capability : {prop.major}.{prop.minor}")
    print(f"VRAM               : {prop.total_memory / 1024**3:.2f} GiB")
    print(f"dtype              : {dtype}")
    print(f"output steps       : min={min(lengths)}, mean={statistics.mean(lengths):.1f}, max={max(lengths)}")

    with torch.inference_mode():
        # 两条路径各预热一次，不计入结果。
        baseline(model, initial[:2], [2, 3])
        continuous(model, initial[:2], [2, 3])
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        (baseline_out, baseline_calls), baseline_ms = time_cuda(
            lambda: baseline(model, initial, lengths)
        )
        baseline_peak = torch.cuda.max_memory_allocated()

        torch.cuda.reset_peak_memory_stats()
        (continuous_out, continuous_calls), continuous_ms = time_cuda(
            lambda: continuous(model, initial, lengths)
        )
        continuous_peak = torch.cuda.max_memory_allocated()

    max_error = (baseline_out.float() - continuous_out.float()).abs().max().item()
    total_tokens = sum(lengths)
    print("\n=== Results ===")
    print(f"Useful decode tokens       : {total_tokens}")
    print(f"Baseline model calls       : {baseline_calls}")
    print(f"Continuous model calls     : {continuous_calls}")
    print(f"Baseline latency           : {baseline_ms:.3f} ms")
    print(f"Continuous latency         : {continuous_ms:.3f} ms")
    print(f"Baseline throughput        : {total_tokens / (baseline_ms / 1000):,.1f} token/s")
    print(f"Continuous throughput      : {total_tokens / (continuous_ms / 1000):,.1f} token/s")
    print(f"Speedup                    : {baseline_ms / continuous_ms:.2f}x")
    print(f"Max absolute output error  : {max_error:.6f}")
    print(f"Peak allocated baseline    : {baseline_peak / 1024**3:.3f} GiB")
    print(f"Peak allocated continuous  : {continuous_peak / 1024**3:.3f} GiB")
    print("\nThis is a launch/compaction microbenchmark, not a full LLM server.")

if __name__ == "__main__":
    main()
```

### 15.1 安装与运行

```
python3 -m venv .venv-batching
source .venv-batching/bin/activate
python -m pip install --upgrade pip

# 按 PyTorch 官方安装选择器安装稳定版、与驱动匹配的 Wheel 后：
python -m torch.utils.collect_env
python gpu_continuous_decode.py

python gpu_continuous_decode.py --requests 8
python gpu_continuous_decode.py --requests 64 --min-steps 4 --max-steps 128
python gpu_continuous_decode.py --dtype bf16
```

双 GPU 分别建立单卡基线：

```
CUDA_VISIBLE_DEVICES=0 python gpu_continuous_decode.py
CUDA_VISIBLE_DEVICES=1 python gpu_continuous_decode.py
nvidia-smi topo -m
```

### 15.2 预期现象

- Baseline 调用次数等于全部输出步数之和。
- Continuous 调用次数等于最长请求的步数，因为同一轮处理多个活动请求。
- 常见 GPU 上 Continuous 版本通常显著减少小 Kernel Launch，并提高总 Token 吞吐。
- 请求很少、Hidden 很小或 CPU 调度很慢时， `stack` 和 Python 更新开销可能削弱收益。
- 两条路径的结果应接近；不同 Batch Shape 的 GEMM 可能产生低精度舍入差异。

该代码没有 KV Cache、Attention、Sampling 和到达队列，所以不能把 Speedup 当成 vLLM 或 TensorRT-LLM 的收益。它只证明“把活动请求压紧到每轮 Batch”为什么能摊薄 Launch 和提高矩阵规模。

## 十六、Level 2：真实 vLLM 调优实验

### 16.1 硬件边界

| 架构 | 示例 | 本课可验证 | 专属边界 |
| --- | --- | --- | --- |
| Ampere | RTX 3080/3090、A100 | Continuous Batching、FP16/BF16、单/双副本 | NVLink 仅特定型号与形态 |
| Ada Lovelace | RTX 4090、L40/L40S | 高吞吐单卡/副本、FP16/BF16 | RTX 4090 无 NVLink，且不是 Blackwell |
| Hopper | H100/H200 | FP8、Transformer Engine、NVLink/NVSwitch | 需真实数据中心软件栈 |
| Blackwell | RTX 5090、B100/B200/GB200 | 新低精度与更大规模服务 | RTX 5090 不等价于 GB200 Fabric |

### 16.2 启动与参数检查

```
python -m venv .venv-vllm-batching
source .venv-vllm-batching/bin/activate

# 按 vLLM 官方安装文档选择与本机 CUDA/PyTorch 匹配的版本。
vllm --version
vllm serve --help=max-num-batched-tokens
vllm serve --help=max-num-seqs
vllm serve --help=chunked

MODEL=/path/to/model
vllm serve "$MODEL" \
  --dtype auto \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 2048 \
  --port 8000
```

当前 vLLM V1 在可用场景默认启用 Chunked Prefill，但版本、模型类型和参数行为会变化。应以本机 `--help` 和启动日志为准，不依赖旧博客中的默认值。

### 16.3 开环压测

```
vllm bench serve \
  --model "$MODEL" \
  --host 127.0.0.1 \
  --port 8000 \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 256 \
  --random-range-ratio 0.5 \
  --num-prompts 500 \
  --request-rate 4 \
  --max-concurrency 32 \
  --percentile-metrics ttft,tpot,itl \
  --save-result \
  --save-detailed
```

建立至少以下实验矩阵：

| 实验 | `max_num_seqs` | Token Budget | 观察重点 | | ------------------------------------ | -- | ----- | -------------- | | A | 16 | 1024 | 低并发、ITL 基线 | | B | 32 | 2048 | 平衡配置 | | C | 64 | 4096 | 吞吐、KV 和 P99 | | D | 64 | 8192+ | TTFT 与 GPU 饱和点 |

不要预设更大的 Budget 一定更好。对小模型和大 GPU，更大矩阵可能提高吞吐；对聊天 SLO，它也可能拉长单轮时间。每次重启服务、记录启动配置，并用同一随机种子与请求集比较。

### 16.4 Nsight 观测

```
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --output=continuous_batching \
  vllm serve /path/to/model \
    --dtype auto \
    --max-num-seqs 32 \
    --max-num-batched-tokens 2048
```

在 Timeline 中检查：

- Scheduler 是否造成 GPU 空洞；
- Prefill 大 Kernel 是否与 ITL 尖峰对应；
- Decode Batch Width 是否随请求完成动态变化；
- 小 Batch 是否产生大量小 Kernel；
- TP 配置下 Collective 是否进入每轮关键路径。

## 十七、取消、抢占与过载

### 17.1 取消

用户断开后，取消信号必须从 Gateway 传到 Scheduler 与 KV Manager。下一安全点应停止调度该请求、回收 KV Block，并避免继续 Sampling 和流式写入。

衡量指标：

$$
CancelledWaste=\frac{Tokens_{after\ cancel}}{Tokens_{all}}
$$

### 17.2 抢占

KV 不足时可：

- 直接拒绝新请求；
- 暂停低优先级请求；
- 丢弃其 KV，稍后重算；
- Offload KV 到 Host/Storage；
- 缩短上下文或切换服务等级。

抢占不是免费操作。重算节省当前显存，却增加未来 GPU 工作和尾延迟。

### 17.3 Backpressure

当 Waiting、KV 水位或预测 Deadline 超出预算时，应拒绝、降级或路由，而不是继续无界排队。快速失败能保护已准入请求的 Goodput。

## 十八、优化前后对照

| 维度 | 朴素调度 | 优化后 |
| --- | --- | --- |
| Batch 生命周期 | 一批请求一起开始、一起结束 | 每轮动态进入、完成即退出 |
| 预算 | 只限制请求数 | Sequence、Token、KV、Deadline 联合预算 |
| Prompt | 整段 Prefill 抢占一轮 | Chunked Prefill 与 Decode 配额 |
| 长短请求 | 同队列 FIFO | 长度、优先级、公平与 Aging |
| KV | 执行时临时申请 | 调度前预留、Watermark 与回收 |
| 取消 | API 丢弃响应，GPU 继续 | 取消贯穿 Scheduler/Executor/KV |
| 过载 | 无界排队 | Admission、Backpressure、降级和路由 |
| 指标 | 平均 QPS、GPU 利用率 | TTFT/ITL P99、Token Budget、KV、Goodput |
| 调优 | 追求最大 Batch | 在 SLO 下最大化有效吞吐 |

## 十九、常见错误与排查

### 19.1 把 Dynamic Batching 当 Continuous Batching

检查 Batch 是否只在请求开始前形成，还是每个 Decode 迭代都能补位。只有短等待窗口组批，不等于 In-flight Batching。

### 19.2 只调 max\_num\_seqs

请求数无法表示 Prompt 成本。同时观察 Token Budget、KV Block 和 Context 长度。

### 19.3 Token Budget 越大越好

大 Budget 可能提高 GEMM 效率和 TTFT，也可能阻塞 Decode、增大 Workspace 并恶化 ITL。用扫描实验找 SLO Goodput 峰值。

### 19.4 Decode 永久优先

持续高 Decode 负载可能让新 Prefill 饥饿。加入 Prefill 最低配额、Deadline、Aging 或阶段隔离。

### 19.5 Chunk 太小

Prompt 被拆成过多小块，Kernel 变小、Launch 和调度开销增加。观察 Prefill Tokens/s、迭代数和 TTFT。

### 19.6 KV 满时盲目提高并发

这会增加抢占和重算，导致吞吐看似不变而 P99 爆炸。检查 KV Block 使用、Preemption、Recompute 和拒绝率。

### 19.7 用闭环客户端掩盖过载

闭环测试要等上一个响应后再发请求，服务器变慢时客户端自动降速。使用固定到达率的开环测试才能看到队列崩溃点。

### 19.8 只看平均长度

长尾 Prompt/Output 决定队头阻塞、KV 峰值和 P99。保存分布和请求级结果。

### 19.9 GPU 微基准输出误差

不同 Batch Shape 会选择不同 GEMM/归约路径，FP16/BF16 结果可能有小差异。先用 FP32 对照，再根据任务质量设容差；不要要求逐 Bit 一致。

## 二十、面试题与答案

### 20.1 Continuous Batching 的核心收益是什么？

把调度粒度降到推理迭代，使完成或取消的请求立即退出，新请求补入空槽，从而减少输出 Padding、队头阻塞和小 Batch 空洞。

### 20.2 它与 Dynamic Batching 的差别是什么？

Dynamic Batching 主要在执行前等待并合并请求；Continuous Batching 在自回归生成的每轮重新选择活动请求。

### 20.3 为什么要同时限制 Sequence 和 Token？

Sequence 数描述 Batch Width，Token 数描述本轮真实工作量。Decode 请求通常每轮一个 Token，而 Prefill 可有数千 Token，仅限制请求数无法控制成本。

### 20.4 Chunked Prefill 如何影响 TTFT 与 ITL？

切片能避免长 Prefill 长时间阻塞 Decode，通常改善 ITL；但 Prompt 需要更多轮才能完成，Chunk 太小时 TTFT 和 Prefill 效率可能变差。

### 20.5 Decode 优先为什么可能导致饥饿？

如果活动 Decode 持续占满 Token Budget，新 Prompt 永远没有 Prefill 机会。需要保留 Prefill 配额、Aging、Deadline 或独立资源池。

### 20.6 为什么 Continuous Batching 依赖 Paged KV？

严格说并非逻辑上必须，但动态加入、退出和不同长度请求需要灵活分配 KV。Block 化管理能减少连续预留碎片，并支持回收、共享和抢占。

### 20.7 什么配置决定最大吞吐？

没有单一配置。模型、精度、GPU、长度分布、到达率、KV 容量、Token Budget、最大序列数、并行和 SLO 共同决定。

### 20.8 为什么开环压测更适合找容量？

它按外部到达率发请求，不会因服务变慢自动降速，因此能显示 Queue 增长、P99 拐点和稳定容量边界。

### 20.9 抢占一定提高利用率吗？

不一定。抢占释放当前 KV，却可能带来重算、Offload、Cache Miss 和尾延迟。应比较抢占节省与恢复成本。

### 20.10 如何判断 Scheduler 成了 CPU 瓶颈？

队列有工作但 GPU 出现规律空洞，CPU Timeline 显示调度、Pack 或 Python 对象处理占据间隙；调度耗时随在途请求数显著增长。

## 二十一、课后练习

1. 在 Level 0 中扫描输出长度离散度，画出静态 Decode Slot Efficiency。
2. 给模拟器加入 5% 的长 Prompt，并比较 FIFO 与短 Prompt 优先的 P99。
1. 给连续调度加入 Prefill 最低配额，验证 Decode 优先是否仍会饿死新请求。
2. 模拟用户取消，统计 Cancelled Waste 与释放的 KV Token。
1. 在 GPU 实验中记录每轮活动 Batch Width，解释为什么后期吞吐下降。
2. 把 `torch.stack` 改为预分配 Buffer，比较 CPU 与 GPU Timeline。
1. 对真实 vLLM 服务扫描 4 组 Token Budget，报告 TTFT、ITL、输出吞吐和 Goodput。
2. 为双 RTX 3080 20GB 设计双副本路由，说明如何避免一张卡队列爆满、另一张卡空闲。

## 二十二、课程 Checklist

- 能区分 Static、Dynamic 和 Continuous Batching。
- 能解释请求级与迭代级调度差异。
- 调度同时使用 Sequence、Token、KV 和 SLO 预算。
- 能解释 Prefill 优先与 Decode 优先的取舍。
- 能说明 Chunked Prefill 的收益和额外开销。
- 有公平性、Aging、Priority 和租户隔离策略。
- 用户取消能传到 Executor 并回收 KV。
- KV 满时有拒绝、抢占或降级策略，而不是等待 OOM。
- 使用开环到达率和真实长度分布压测。
- 同时报告 TTFT、ITL、E2E、Tokens/s 和 Goodput。
- 记录每轮 Batch Width、Token Budget 和 KV 水位。
- 先验证单卡，再根据容量和拓扑选择多卡方案。
- 自动记录 GPU、Compute Capability、驱动、CUDA 和框架版本。
- RTX 4090 标记为 Ada，RTX 5090 标记为 Blackwell。
- FP8/FP4、Transformer Engine、NVLink/NVSwitch 只在真实支持环境验证。
- 优化前后保持模型、精度、Sampling、请求集和到达过程一致。