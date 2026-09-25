---
title: "第22课：AI Runtime 优化思想"
source: "https://everythingai.top/learn/ai-systems-performance-engineering-text/lesson-22"
author:
published:
created: 2026-09-26
description: "面向 AI 工程学习者的专业课程学习空间。"
tags:
  - "clippings"
---
**导航**

本课从调度、内存、执行图与设备协同角度理解 AI Runtime，建立系统级优化方法。

## 一、课程定位

高性能 Kernel 并不等于高性能系统。一个算子可以达到很高的 Tensor Core 利用率，但用户请求仍可能在队列里等待；GPU 可以拥有很高的瞬时利用率，但大量工作可能是 Padding、重复计算或最终超时的无效请求。

AI Runtime 是模型与硬件之间的“现场总指挥”。它接收工作，管理状态和内存，选择执行计划，把请求组合成 Batch，把任务投递到设备，并在超载时进行限流、降级和回退。

本课不局限于某个推理框架，而是建立一套通用于训练 Runtime、推理 Runtime、在线服务和本地多 GPU 系统的优化思想：

## 二、学习目标

- 能画出 AI Runtime 的控制面、数据面和执行面。
- 理解 Scheduler、Batcher、Memory Manager、Executor、Compiler Cache 的职责。
- 掌握 Hot Path/Cold Path、异步流水线、Shape Bucket 和内存复用思想。
- 用排队模型解释吞吐、尾延迟、并发和背压之间的关系。
- 建立端到端性能预算，而不是只优化 GPU Kernel。
- 能设计 Admission Control、超时、取消、优先级和降级策略。
- 完成 CPU 调度模拟和通用 NVIDIA GPU Runtime 实验。
- 能判断下一步应优化 Scheduler、Batch、内存、Compiler、Kernel、通信还是拓扑。

## 三、前置知识

- Latency、Throughput、Goodput、Little 定律与 P50/P99。
- CUDA Stream、异步执行、Kernel Launch 和 CUDA Graph。
- `torch.compile` 、Shape 专门化和重编译。
- GPU Memory、Pinned Memory、内存池与数据搬运。
- 分布式通信、NCCL 与计算通信重叠。

## 四、核心直觉：Runtime 优化的是“流”，不是单个点

把 AI 系统想象成机场：

- 请求是乘客；
- Scheduler 是塔台；
- Batch 是同一班飞机；
- GPU 是跑道与飞机；
- KV Cache/Activation/Workspace 是登机口和行李位；
- CUDA Stream 是不同作业通道；
- Graph/Compiled Plan 是预先批准的固定航线；
- Backpressure 是限制进入候机楼的人数。

只把飞机发动机调快，并不能解决登机口拥堵、跑道冲突和乘客错过转机。Runtime 优化要关注请求从进入到完成的整条路径：

```
接入 → 排队 → 预处理 → 调度/组批 → H2D → 计算/通信
    → 后处理 → D2H/流式输出 → 计量与回收
```

任何一段变慢，都会通过队列放大成尾延迟。

## 五、AI Runtime 的分层架构

### 5.1 控制面、数据面和执行面

```
┌──────────────────────── 控制面 ────────────────────────┐
│ 模型版本、配置、路由、配额、扩缩容、健康检查、回滚      │
└────────────────────────────────────────────────────────┘
                           │
┌──────────────────────── 数据面 ────────────────────────┐
│ Gateway → Queue → Admission → Scheduler → Batcher      │
│                   ↕ 状态/KV/缓存 ↕                      │
└────────────────────────────────────────────────────────┘
                           │
┌──────────────────────── 执行面 ────────────────────────┐
│ Executor → Compiler/Plan Cache → Memory Pool            │
│          → CUDA Stream/Graph → Kernel/Library/NCCL      │
└────────────────────────────────────────────────────────┘
                           │
                   GPU / CPU / 网络 / 存储
```

控制面变化频率低，负责“应该运行什么”；数据面高频处理请求，负责“何时运行谁”；执行面贴近设备，负责“如何运行得快”。把三者混在一个同步线程中，会让配置、日志或网络抖动进入计算热路径。

### 5.2 Runtime 的核心组件

| 组件 | 主要职责 | 常见失败模式 |
| --- | --- | --- |
| Admission Controller | 限制并发和显存承诺 | 无界接收导致 OOM 和 P99 爆炸 |
| Scheduler | 选择下一批工作 | 饥饿、队头阻塞、不公平 |
| Batcher | 合并兼容请求 | 等待过久或 Padding 浪费 |
| Executor | 驱动设备执行 | 同步过多、Launch 碎片 |
| Memory Manager | 分配、复用、驱逐 | 碎片、地址不稳定、错误驱逐 |
| Plan/Graph Cache | 复用编译和捕获结果 | Shape 爆炸、缓存污染 |
| State Manager | 管理 KV、会话、随机状态 | 泄漏、状态串扰、恢复错误 |
| Telemetry | 观测阶段、队列和资源 | 只有平均值，无法解释 P99 |

## 六、第一原则：先定义有效工作

Runtime 最终优化目标应是 Goodput，而不是单一的 GPU Utilization：

$$
Goodput=\frac{\text{满足正确性、SLO 和业务约束的完成工作}}{\text{时间或资源成本}}
$$

以下工作可能提高 Throughput 却不提高 Goodput：

- 大 Batch 导致大量请求超过 P99 SLO；
- 取消的请求仍继续生成几十个 Token；
- Padding 占用了大部分 FLOPs；
- 低精度提高 Tokens/s，却使质量不达标；
- 过载时不断重试，制造重试风暴；
- GPU 100% 忙于预热或低优先级任务，在线请求却超时。

Runtime 的指标应至少同时包含：吞吐、P50/P95/P99、超时率、拒绝率、取消浪费、有效 Token/s、显存水位和成本。

## 七、第二原则：拆分 Cold Path 与 Hot Path

### 7.1 Cold Path

Cold Path 可以承受较高延迟，负责：

- 加载权重、初始化 CUDA Context；
- 编译、Autotune、CUDA Graph Capture；
- 建立 NCCL Communicator；
- 分配大块内存池；
- 构建 Shape/Plan Cache；
- 健康检查和模型切换。

### 7.2 Hot Path

Hot Path 每请求或每 Token 执行，应尽可能：

- 不分配大对象；
- 不做文件 I/O、日志格式化或远程 RPC；
- 不触发编译、模型加载和算法搜索；
- 不做非必要 `.item()` 和全局同步；
- 不持有粗粒度锁；
- 使用预分配 Buffer、缓存 Plan 和异步队列。

判断方法很简单：如果某件事能在请求到来前完成，就不应放进每请求路径。

## 八、第三原则：调度是性能功能，不只是排队

### 8.1 FIFO 不是万能策略

FIFO 简单公平，但长请求可能造成队头阻塞。Runtime 需要根据业务选择：

- FIFO：简单、可预测；
- 优先级：区分在线、批处理和系统任务；
- Deadline-aware：优先处理接近 SLO 的请求；
- Shortest-job-first：降低平均延迟，但需防止长请求饥饿；
- Fair-share：限制租户或用户长期占用；
- Cache-aware：优先命中同一 Prefix、KV 或已编译 Shape；
- Topology-aware：把任务放到数据和通信路径更近的设备。

### 8.2 Admission Control 比事后排队更重要

当到达率长期超过服务率，无论 Scheduler 多聪明，队列都会增长。设到达率为 $\lambda$ ，每个 Worker 服务率为 $\mu$ ，Worker 数为 c：

$$
\rho=\frac{\lambda}{c\mu}
$$

当 $\rho$ 接近 1，排队延迟会非线性上升；当 $\rho>1$ ，系统无法稳定。Admission Control 应在显存、并发、Deadline 或队列长度达到预算时拒绝、降级或转移请求。

“尽量不拒绝”常常会变成“所有人都超时”。快速失败对 Goodput 更友好。

### 8.3 取消必须贯穿执行链

用户断开连接后，Gateway、Scheduler、Batcher 和 Executor 都要识别取消。若只在 API 层丢弃响应，GPU 仍继续计算，取消请求会吞噬有效请求资源。

## 九、第四原则：Batching 是吞吐与等待的交换

### 9.1 批处理收益模型

设固定调度/Launch 开销为 a，单样本增量计算成本为 b，Batch 大小为 B：

$$
T(B)=a+bB+\epsilon(B)
$$

单样本平均执行成本为：

$$
\bar{T}(B)=\frac{a}{B}+b+\frac{\epsilon(B)}{B}
$$

增大 B 能摊薄固定成本，但会增加排队等待、显存、Padding 和长尾干扰。最优 Batch 不是显存能装下的最大值，而是满足 SLO 时 Goodput 最高的值。

### 9.2 Dynamic Batching

Dynamic Batcher 在短时间窗口内收集兼容请求。核心旋钮是：

- `max_batch_size` ：最大批量；
- `max_wait` ：允许首个请求等待多久；
- Shape/Dtype/Model Version 兼容规则；
- Priority 与 Deadline；
- 最大队列和超时动作。

如果流量低，等待满 Batch 会伤害延迟；如果流量高，Batch 往往自然形成，无需额外延迟。

### 9.3 Shape Bucket 与 Padding

混合长度请求直接组批会产生 Padding：

$$
Padding\ Waste=1-\frac{\sum_i L_i}{B\cdot \max_i L_i}
$$

按长度 Bucket 可以降低浪费，也有利于 `torch.compile` 、CUDA Graph 和 TensorRT Plan 复用。但 Bucket 太细会减少可合批请求并扩大 Cache。

## 十、第五原则：异步不是“不要同步”，而是显式依赖

一个高效 Runtime 会让 CPU 预处理、H2D、GPU 计算、通信、D2H 和后处理形成流水线：

```
时间 →
CPU :  prep N+1 ─ prep N+2 ─ prep N+3
H2D :      copy N+1 ─ copy N+2 ─ copy N+3
GPU : run N ───── run N+1 ───── run N+2
D2H :          out N ───── out N+1
```

实现手段包括 Pinned Memory、 `non_blocking=True` 、CUDA Stream、Event、异步 NCCL 和独立线程/进程。关键不是删除所有同步，而是把同步缩小到真实数据依赖：

- Producer 完成后 Consumer 才能读取；
- Buffer 重用前必须确认前一次使用结束；
- 输出返回前必须确认相应请求的设备工作完成；
- 不相关请求不应被全局 `cudaDeviceSynchronize()` 一起阻塞。

## 十一、第六原则：内存是 Runtime 的一等资源

### 11.1 容量规划

Runtime 显存近似由以下部分构成：

$$
M=M_{weights}+M_{state/cache}+M_{activation}+M_{workspace}+M_{graph}+M_{fragmentation}+M_{reserve}
$$

不能把 `free_memory / request_memory` 直接当最大并发，因为 Workspace、Graph Pool、临时通信 Buffer 和碎片会随执行计划变化。

### 11.2 预分配与对象生命周期

热路径频繁分配会引入锁、碎片和地址不稳定。常用方式：

- 大块预分配后切分；
- Size-class Pool；
- Paged/Block 管理；
- Ring Buffer 或双 Buffer；
- 请求完成或取消时明确归还；
- Watermark 触发 Admission 或 Eviction。

### 11.3 Eviction 不是免费空间

驱逐 KV、Activation 或编译缓存会带来重算、重新加载或重新编译。合理策略应比较：

$$
Eviction\ Cost=Reload+Recompute+Tail\ Risk
$$

不能只看“释放了多少 GB”。

## 十二、第七原则：专门化与安全回退并存

Runtime 可以为高频路径建立：

- 编译后的计算图；
- CUDA Graph；
- TensorRT/后端 Engine；
- 高频 Batch/Shape 的 Kernel 配置；
- 常用 Prefix 或 KV Cache；
- 特定 GPU 架构的低精度路径。

但专门化必然增加状态空间。每个优化路径都需要：

1. Capability 检测；
2. 正确性门槛；
1. Cache Key；
2. 生命周期和淘汰；
1. 不支持输入的通用回退；
2. 版本升级后的重新验证。

一个只有 Fast Path、没有 Fallback 的 Runtime 不是生产系统。

## 十三、第八原则：调优对象是关键路径

端到端请求延迟可以拆为：

$$
L=L_{queue}+L_{pre}+L_{batch}+L_{H2D}+L_{dispatch}+L_{compute}+L_{comm}+L_{post}+L_{return}
$$

优化某阶段 x 的收益上限受 Amdahl 定律约束：

$$
S=\frac{1}{(1-p_x)+p_x/s_x}
$$

若 Queue 占 P99 的 60%，把 Kernel 加速 2 倍不会解决主要问题。若 GPU 只有 20% 利用率但 Queue 很长，可能是 CPU 调度、锁、输入搬运或内存 Admission 在阻塞，而不是 GPU 不够快。

## 十四、瓶颈分析方法

### 14.1 分层证据

| 层级 | 关键证据 | 常见工具 |
| --- | --- | --- |
| 请求 | Arrival、Queue、Deadline、Cancel | 服务 Trace、Histogram |
| Scheduler | Batch Size、等待、拒绝、公平性 | Runtime Metrics |
| CPU | Thread、Lock、GIL、预处理、Launch | CPU Profile、Nsight Systems |
| GPU | Kernel、空洞、Memcpy、Stream | Nsight Systems/Compute |
| 内存 | Pool、碎片、Cache、Eviction | PyTorch Snapshot、Runtime Metrics |
| 多卡 | Rank 等待、Collective、拓扑 | NCCL 日志、Nsight、拓扑图 |

### 14.2 Runtime 诊断顺序

```
SLO/Goodput 不达标
  │
  ├─ Queue 是否增长？
  │    ├─ 是：到达率、服务率、Admission、Batch、Worker
  │    └─ 否
  │
  ├─ CPU 是否喂不满 GPU？
  │    ├─ 是：锁、同步、预处理、Launch、H2D
  │    └─ 否
  │
  ├─ GPU 是否有长空洞或低效 Kernel？
  │    ├─ 空洞：Graph、Batch、Stream、通信重叠
  │    └─ 长热点：Fusion、精度、Kernel、并行策略
  │
  └─ Memory/Network 是否限制并发？
       ├─ Memory：Pool、KV、碎片、驱逐
       └─ Network：拓扑、Payload、重叠、解耦
```

### 14.3 不要只看平均值

Runtime 优化至少保留：

- Arrival Rate 与完成率；
- Queue Time P50/P95/P99；
- Service Time P50/P95/P99；
- Batch Size 分布；
- Inflight 和 Queue Depth；
- 超时、拒绝、取消和重试；
- 显存 Pool、碎片和驱逐；
- 按 Model、Shape、Tenant、GPU、Rank 分组的指标。

## 十五、Level 0：通用 CPU 调度模拟实验

下面的纯 Python 离散事件模拟器比较“立即执行”和“等待组批”，无需 GPU 或第三方库。它不是具体推理框架实现，但能验证 Batch、Queue 和 SLO 的基本关系。

保存为 `runtime_scheduler_sim.py` ：

```
#!/usr/bin/env python3
import argparse
import math
import random
import statistics

def percentile(values, q):
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)

def make_arrivals(count, rate, seed):
    rng = random.Random(seed)
    now = 0.0
    arrivals = []
    for _ in range(count):
        now += rng.expovariate(rate) * 1000.0
        arrivals.append(now)
    return arrivals

def simulate(arrivals, max_batch, max_wait_ms, fixed_ms, per_item_ms, slo_ms):
    pending = []
    latencies = []
    now = 0.0
    index = 0
    batches = []

    while index < len(arrivals) or pending:
        if not pending and index < len(arrivals):
            now = max(now, arrivals[index])
        # 先接收 Server 忙碌期间已经到达的请求。
        while index < len(arrivals) and arrivals[index] <= now:
            pending.append(arrivals[index])
            index += 1

        # 若尚未满批，最多等到队首请求的 max_wait 截止时间。
        deadline = max(now, pending[0] + max_wait_ms)
        while (index < len(arrivals) and len(pending) < max_batch
               and arrivals[index] <= deadline):
            pending.append(arrivals[index])
            index += 1

        if len(pending) >= max_batch:
            start = max(now, pending[max_batch - 1])
        else:
            start = deadline

        batch = pending[:max_batch]
        del pending[:max_batch]
        service = fixed_ms + per_item_ms * len(batch)
        now = start + service
        batches.append(len(batch))
        latencies.extend(now - arrival for arrival in batch)

    elapsed_s = (now - arrivals[0]) / 1000.0
    return {
        "throughput": len(arrivals) / elapsed_s,
        "p50": percentile(latencies, 0.50),
        "p95": percentile(latencies, 0.95),
        "p99": percentile(latencies, 0.99),
        "goodput": sum(x <= slo_ms for x in latencies) / elapsed_s,
        "slo_rate": sum(x <= slo_ms for x in latencies) / len(latencies),
        "avg_batch": statistics.mean(batches),
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--requests", type=int, default=5000)
    p.add_argument("--arrival-rps", type=float, default=300.0)
    p.add_argument("--max-batch", type=int, default=16)
    p.add_argument("--max-wait-ms", type=float, default=2.0)
    p.add_argument("--fixed-ms", type=float, default=1.0)
    p.add_argument("--per-item-ms", type=float, default=0.15)
    p.add_argument("--slo-ms", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    if min(args.requests, args.max_batch) < 1:
        raise SystemExit("requests/max-batch 必须为正整数")
    if min(args.arrival_rps, args.fixed_ms, args.per_item_ms, args.slo_ms) <= 0:
        raise SystemExit("速率和时间参数必须为正数")
    if args.max_wait_ms < 0:
        raise SystemExit("max-wait-ms 不能为负数")

    arrivals = make_arrivals(args.requests, args.arrival_rps, args.seed)
    immediate = simulate(arrivals, 1, 0.0, args.fixed_ms,
                         args.per_item_ms, args.slo_ms)
    batched = simulate(arrivals, args.max_batch, args.max_wait_ms,
                       args.fixed_ms, args.per_item_ms, args.slo_ms)

    print("mode throughput_rps p50_ms p95_ms p99_ms goodput_rps slo_rate avg_batch")
    for name, result in [("immediate", immediate), ("batched", batched)]:
        print(f"{name:9s} {result['throughput']:14.2f} "
              f"{result['p50']:7.2f} {result['p95']:7.2f} "
              f"{result['p99']:7.2f} {result['goodput']:11.2f} "
              f"{result['slo_rate']:8.3f} {result['avg_batch']:9.2f}")

if __name__ == "__main__":
    main()
```

运行命令：

```
python3 runtime_scheduler_sim.py
python3 runtime_scheduler_sim.py --arrival-rps 50 --max-wait-ms 5
python3 runtime_scheduler_sim.py --arrival-rps 900 --max-batch 32 --slo-ms 20
```

### 15.1 预期现象

- 中高负载时，组批能摊薄固定开销，提高 Throughput。
- 低流量时，额外等待可能让 P50/P99 变差，Batch 也未必能装满。
- 高于系统服务能力时，队列增长，P99 和 SLO 达标率迅速恶化。
- 最大 Throughput 配置不一定拥有最高 Goodput。

### 15.2 模拟边界

该程序采用简化单 Worker、线性 Batch 服务模型，不模拟 GPU 并发、KV Cache、动态 Token 数和真实框架开销。它用于建立直觉，不应替代真实压测。

## 十六、Level 1：通用 NVIDIA GPU Runtime 实验

这个 PyTorch 实验比较三种执行路径：每请求同步、异步逐请求、Micro-batching。输入预先放在 GPU 上，以突出 Runtime Dispatch 和 Batch 的影响。

保存为 `runtime_pipeline_lab.py` ：

```
#!/usr/bin/env python3
import argparse
import statistics
import time

import torch
from torch import nn

class TinyModel(nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend([nn.Linear(width, width), nn.GELU()])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def bench(fn, repeats):
    samples = []
    output = None
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        output = fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples), output

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--requests", type=int, default=512)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--microbatch", type=int, default=32)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeats", type=int, default=5)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("需要可用的 NVIDIA CUDA GPU；无 GPU 请运行 Level 0")
    if min(args.requests, args.width, args.depth,
           args.microbatch, args.warmup, args.repeats) < 1:
        raise SystemExit("所有参数必须为正数")

    device = torch.device("cuda")
    prop = torch.cuda.get_device_properties(device)
    print(f"torch={torch.__version__} cuda={torch.version.cuda}")
    print(f"gpu={prop.name} capability={prop.major}.{prop.minor} "
          f"vram_gib={prop.total_memory/2**30:.2f}")

    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    model = TinyModel(args.width, args.depth).eval().to(device)
    inputs = torch.randn(args.requests, args.width, device=device)

    with torch.inference_mode():
        for _ in range(args.warmup):
            model(inputs[:args.microbatch])
        torch.cuda.synchronize()

        def sync_each_request():
            last = None
            for i in range(args.requests):
                last = model(inputs[i:i+1])
                torch.cuda.synchronize()
            return last

        def async_each_request():
            last = None
            for i in range(args.requests):
                last = model(inputs[i:i+1])
            return last

        def microbatched():
            last = None
            for start in range(0, args.requests, args.microbatch):
                last = model(inputs[start:start+args.microbatch])
            return last

        sync_ms, sync_out = bench(sync_each_request, args.repeats)
        async_ms, async_out = bench(async_each_request, args.repeats)
        batch_ms, batch_out = bench(microbatched, args.repeats)

        torch.testing.assert_close(sync_out, async_out, rtol=1e-4, atol=1e-4)
        tail = args.requests % args.microbatch or args.microbatch
        reference_batch = model(inputs[-tail:])
        torch.cuda.synchronize()
        torch.testing.assert_close(batch_out, reference_batch,
                                   rtol=1e-4, atol=1e-4)

    def rps(ms):
        return args.requests / (ms / 1000.0)

    print(f"sync_each_ms={sync_ms:.3f} throughput_rps={rps(sync_ms):.2f}")
    print(f"async_each_ms={async_ms:.3f} throughput_rps={rps(async_ms):.2f}")
    print(f"microbatch_ms={batch_ms:.3f} throughput_rps={rps(batch_ms):.2f}")
    print(f"async_speedup={sync_ms/async_ms:.3f}x")
    print(f"batch_speedup={sync_ms/batch_ms:.3f}x")
    print(f"peak_allocated_mib={torch.cuda.max_memory_allocated()/2**20:.1f}")
    print(f"peak_reserved_mib={torch.cuda.max_memory_reserved()/2**20:.1f}")

if __name__ == "__main__":
    main()
```

### 16.1 安装与运行

```
python3 -m venv .venv-runtime
source .venv-runtime/bin/activate
python -m pip install --upgrade pip
# 从 PyTorch 官方安装页选择与驱动匹配的稳定 CUDA 构建
```

通用 GPU：

```
nvidia-smi
python runtime_pipeline_lab.py
python runtime_pipeline_lab.py --microbatch 1
python runtime_pipeline_lab.py --microbatch 8
python runtime_pipeline_lab.py --microbatch 32
python runtime_pipeline_lab.py --microbatch 128
```

双 RTX 3080 分别验证：

```
CUDA_VISIBLE_DEVICES=0 python runtime_pipeline_lab.py --microbatch 32
CUDA_VISIBLE_DEVICES=1 python runtime_pipeline_lab.py --microbatch 32
```

### 16.2 预期现象

- 每请求 `synchronize()` 通常最慢，因为完全破坏 CPU/GPU 异步流水。
- 异步逐请求减少同步，但仍有大量 Python、Dispatcher 和 Kernel Launch。
- Micro-batching 通常能显著提高 Throughput，尤其是 $Batch=1$ 时 GPU 吃不满的模型。
- Microbatch 继续增大后收益会递减，甚至因显存、Cache、排队或形状不理想而下降。
- 本脚本测试的是 GPU 内已有输入；真实服务还必须测 H2D、Tokenizer、Queue 和响应返回。

### 16.3 结果分析模板

| 路径 | 总时间 | Requests/s | 峰值显存 | 解释 | | -------------------------- | -- | -- | -- | ---------- | | 每请求同步 | 实测 | 实测 | 实测 | 同步基线 | | 异步逐请求 | 实测 | 实测 | 实测 | 仅消除高频同步 | | Microbatch | 实测 | 实测 | 实测 | 摊薄调度并提高并行度 |

这些数值只属于本机环境，不应写成所有 RTX 3080、4090、H100 或 5090 的固定结论。

## 十七、Level 2：架构专项与系统实验

### 17.1 硬件映射

| 架构 | 示例 GPU | Runtime 可验证重点 |
| --- | --- | --- |
| Ampere | RTX 3080/3090、A100 | TF32/BF16、Graph、Batch、异步 Pipeline |
| Ada Lovelace | RTX 4090、L40/L40S | 通用推理 Runtime 与更大 Batch |
| Hopper | H100/H200 | FP8、Transformer Engine、数据中心多实例 |
| Blackwell | RTX 5090、B100/B200/GB200 | FP4、新 Tensor Core 与机架级 Runtime |

RTX 4090 属于 Ada Lovelace；RTX 5090 属于 Blackwell。

### 17.2 通用 Profile

```
nsys profile --trace=cuda,nvtx,osrt -o runtime_pipeline \
  python runtime_pipeline_lab.py --microbatch 32 --repeats 3
```

重点观察：

- CPU CUDA API 是否连续占用；
- Kernel 之间是否有空洞；
- Batch 增大后 GEMM 是否变长、Kernel 数是否下降；
- 是否存在每请求同步；
- 内存分配是否进入 Hot Path。

### 17.3 双 3080 的扩展实验

双 RTX 3080 通常通过 PCIe 连接。可做两种对比：

1. 一个模型副本/卡，由上层 Scheduler 做 Data Parallel 请求路由；
2. 一个模型跨两卡做 Tensor Parallel，支付每层通信成本。

模型单卡可放下时，在线请求副本化往往更简单，吞吐扩展也更直接；跨卡切分是否更好取决于模型容量、Batch、PCIe、通信比例和 SLO，不能仅凭 GPU 数量判断。

### 17.4 专属能力边界

- H100/H200 的 FP8 需要硬件和软件栈真实支持，3080 不能等价模拟。
- B100/B200/GB200 的 FP4、NVLink/NVSwitch 机架能力需真实 Blackwell 平台。
- RTX 5090 虽属于 Blackwell，但不能等同于 B200/GB200 的数据中心互联和可靠性特性。
- MPS、MIG、NVLink、GPUDirect、Transformer Engine 均需先做 Capability 检测。

## 十八、Runtime 优化优先级

建议按收益、风险和维护成本分层：

### Level A：低侵入

- 正确计时和阶段指标；
- 删除高频同步；
- `inference_mode` 、Pinned Memory、异步 Copy；
- 调整 Batch、Queue、Worker；
- 预热、进程常驻、预分配。

### Level B：执行计划

- `torch.compile` ；
- Shape Bucket；
- CUDA Graph；
- Fused Operator；
- KV/Workspace Pool；
- 计算通信重叠。

### Level C：系统重构

- Continuous Batching；
- Prefill/Decode 解耦；
- Paged KV Cache；
- 分布式 Executor；
- 自定义 Kernel/Runtime；
- 拓扑感知路由和远端内存。

在 Level A 已满足目标时，不必直接进入 Level C。可维护性、回退能力和性能回归成本也是 Runtime 的一部分。

## 十九、优化前后对照

| 低效 Runtime | 优化后的 Runtime | 价值 |
| --- | --- | --- |
| 无界请求队列 | Admission + Queue Limit + Deadline | 防止过载雪崩 |
| 每请求独立执行 | SLO 约束下动态组批 | 摊薄固定开销 |
| 任意 Shape | 有限 Bucket + 通用回退 | 复用 Plan/Graph |
| 每请求分配 | 预分配 Pool + 明确归还 | 降低碎片和抖动 |
| 全局同步 | Event/Stream 表达局部依赖 | 提升并发与重叠 |
| 取消仅丢响应 | Executor 传播取消 | 减少无效计算 |
| 只看 GPU 利用率 | Goodput + Queue + P99 + 成本 | 优化业务结果 |
| 快慢路径混在一起 | Cold/Hot Path 隔离 | 稳定尾延迟 |
| 只有 Fast Path | Capability 检测和安全回退 | 生产可靠性 |

## 二十、常见错误与排查

### 20.1 GPU 利用率高但 P99 很差

可能是 Batch 等待过长、长短请求互相干扰、队列过载或取消请求仍执行。按 Queue、Batch、Service 分解延迟，不要继续盲目加 Batch。

### 20.2 Throughput 高但 Goodput 低

检查超时、拒绝、Padding、无效 Token、低精度质量和重试风暴。吞吐分子必须是满足业务约束的完成工作。

### 20.3 增加 Worker 反而变慢

多个 Worker 可能争抢 CPU、GPU Context、显存和 PCIe；模型副本还会挤压 Batch/Cache。逐步增加实例，并同时看单实例 Batch、上下文切换和显存。

### 20.4 Microbatch 越大越慢

可能触发显存压力、不同 Kernel 路径、Cache Thrash、更多 Padding 或排队。扫描多个 Batch，建立 Latency-Throughput-Goodput 曲线。

### 20.5 显存还有很多却拒绝请求

Runtime 可能为 Workspace、Graph、通信或碎片保留 Safety Margin。检查承诺模型和峰值，而非只看瞬时 `nvidia-smi` Free Memory。

### 20.6 异步后出现错误结果

通常是 Buffer 在前一次执行完成前被重用、缺少 Stream Event，或输出生命周期错误。异步优化必须显式表达数据依赖。

### 20.7 编译缓存不断增长

检查 Cache Key 是否包含无价值的 Python 状态，Shape 是否未 Bucket，版本是否重复。设置容量、命中率、淘汰和通用回退指标。

### 20.8 平均延迟稳定但偶发尖峰

排查首次编译、Graph Capture、Allocator 扩容、GC、模型切换、Checkpoint、热降频、后台 Profile 和 NUMA 漂移。Cold Path 事件应有独立标记。

## 二十一、面试题与答案

### 题1：AI Runtime 与深度学习框架有什么区别？

框架提供算子、Autograd、编译等编程能力；Runtime 负责在真实请求和资源约束下管理队列、调度、Batch、状态、内存、执行、回退和观测。两者可以重叠，但关注尺度不同。

### 题2：为什么 GPU Utilization 不是最终目标？

它不区分有效计算、Padding、超时请求和重算。Goodput 才衡量满足正确性和 SLO 的有效工作。

### 题3：Dynamic Batching 的关键权衡是什么？

Batch 能摊薄固定成本并提高 GPU 并行度，但会增加排队、显存和 Padding。最优点由流量分布和 SLO 决定。

### 题4：为什么利用率接近 100% 时 P99 会快速恶化？

系统缺少吸收到达抖动的余量，服务时间稍有波动就会积累队列。排队延迟在高利用区通常非线性增长。

### 题5：Admission Control 的价值是什么？

在系统过载前限制 Inflight、显存承诺或队列，选择拒绝、降级或转移，避免所有请求一起超时或 OOM。

### 题6：Hot Path 应避免哪些操作？

模型加载、编译、Autotune、文件/网络 I/O、日志格式化、大对象分配、粗粒度锁和非必要设备同步。

### 题7：Shape Bucket 有什么收益和代价？

它降低 Padding，并提高编译 Plan、CUDA Graph 和 Kernel 配置复用；代价是队列被拆分、Cache 数量增加，Bucket 设计不当还会降低合批率。

### 题8：异步执行最常见的正确性风险是什么？

Producer 尚未完成就读取，或 Buffer 仍被使用就重写。应使用 Stream/Event 和生命周期所有权表达依赖。

### 题9：Runtime 内存预算包括哪些部分？

权重、状态/KV、Activation、Workspace、Graph/编译缓存、通信 Buffer、碎片和安全余量，不能只计算权重加 KV。

### 题10：如何确定下一个优化点？

分解端到端关键路径，比较 Queue、CPU、H2D、GPU Kernel、通信、内存和后处理证据；优先优化占比最高且可改善的阶段。

## 二十二、课后练习

1. 用 Level 0 扫描 Arrival Rate 50～1000 RPS，画出 P99 和 Goodput 曲线。
2. 修改模拟器，加入两级优先队列并防止低优先级饥饿。
1. 加入请求取消，统计取消后继续执行产生的浪费。
2. 扫描 Level 1 的 Microbatch 1/2/4/8/16/32/64/128。
1. 给 GPU 实验加入 CPU Pinned Memory 与非阻塞 H2D。
2. 使用 Nsight Systems 标记 Queue、Batch、H2D、Compute、Return 阶段。
1. 为双 RTX 3080 设计副本化路由，比较 Round-Robin 与最短队列。
2. 写出一个 Runtime Capacity Plan：SLO、峰值流量、并发、Batch、显存和 Safety Margin。

## 二十三、Checklist

### 目标与指标

- 定义正确性、SLO、Goodput 和成本目标。
- 分开 Queue Time、Service Time 和端到端时间。
- 保存 P50/P95/P99、超时、拒绝、取消和重试。
- 指标可按模型、Shape、租户、GPU 和版本切分。

### 调度与容量

- Arrival Rate、Service Rate、Inflight 和 Queue Depth 可观测。
- Admission、队列上限、Deadline 和降级策略明确。
- Batch/等待窗口通过实测选择，不按显存最大化。
- 取消从接入层传播到 Executor。
- 优先级策略具备公平性和防饥饿机制。

### 执行与内存

- Cold Path 和 Hot Path 已隔离。
- 热路径无非必要分配、I/O、日志和全局同步。
- Buffer 生命周期和 Stream/Event 依赖正确。
- Shape/Plan/Graph Cache 有容量、命中率和淘汰策略。
- 显存预算包含 Workspace、Pool、碎片和安全余量。

### 硬件与验证

- 自动记录 GPU、Compute Capability、驱动、CUDA 和框架版本。
- CPU、常见 GPU、专项硬件都有明确实验边界。
- 4090 标记为 Ada，5090 标记为 Blackwell。
- FP8/FP4、NVLink/NVSwitch 只在真实支持硬件上验证。
- 优化前后保持输入、正确性和负载分布一致。
- 最终收益由无 Profile 的端到端压测确认。