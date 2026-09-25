---
title: "第12课：GPU 性能分析方法"
source: "https://everythingai.top/learn/ai-systems-performance-engineering-text/lesson-12"
author:
published:
created: 2026-09-26
description: "面向 AI 工程学习者的专业课程学习空间。"
tags:
  - "clippings"
---
**导航**

本课建立从基准测量到 Nsight 分析的标准流程，把“感觉慢”转化为可验证的瓶颈假设。

## 课程定位

性能优化最危险的状态，不是不会写 CUDA，而是拿到一个“GPU 利用率 95%”就开始改 Kernel。这个数字可能只表示采样窗口内 GPU 经常有任务在运行，并不等于 Tensor Core 接近峰值、不等于显存带宽已打满，更不等于业务 Goodput 达标。

本课建立一套可重复的 GPU 性能分析流程：先定义目标和基线，再用时间线确认时间花在哪里，最后对少数热点 Kernel 做深度分析。核心工具链是：

```
业务指标 / SLO
      ↓
稳定、可复现的基准
      ↓
nvidia-smi / DCGM：长期趋势与粗粒度遥测
      ↓
Nsight Systems：端到端时间线，回答“时间去哪了”
      ↓
Nsight Compute：Kernel 计数器，回答“为什么慢”
      ↓
提出假设 → 只改一个变量 → 正确性与端到端复测
```

## 学习目标

完成本课后，你能够：

1. 建立带版本、输入、预热、重复次数和噪声统计的性能基线。
2. 正确区分 Host Wall Time、CUDA Event Time 与 Profiler Timeline。
1. 用 Nsight Systems 识别 CPU Launch、GPU Idle、Memcpy、同步、通信和 Kernel 热点。
2. 用 Nsight Compute 分析 Speed of Light、Roofline、Memory、Warp、Occupancy 和 Scheduler 指标。
1. 解释为什么 GPU Util、Occupancy、单个 Counter 都不能独立证明性能良好。
2. 使用 NVTX 把业务阶段映射到 CUDA 时间线。
1. 使用 Amdahl 定律判断某个局部优化是否值得做。
2. 避免异步计时、冷启动、Profiler Replay、频率波动和错误输入造成的伪结论。
1. 在 CPU、常见 NVIDIA GPU 和架构专项环境中执行分层实验。

## 前置知识

- 理解 Latency、Throughput、Goodput、P50/P99、MFU/HFU 与 Roofline。
- 理解 CUDA Stream、Kernel Launch、Event、同步和异步执行。
- 理解 SM、Warp、Global Memory、Shared Memory 与 Tensor Core。
- Level 0 只要求 Python 3.9+；Level 1 需要 CUDA 可用的 PyTorch。

## 核心直觉：Profiler 是显微镜，不是判决书

Profiler 提供证据，但证据必须放在正确的问题里解释。

假设一个训练 Step 需要 100 ms，其中：

- 数据输入 20 ms；
- CPU 调度和 Kernel Launch 15 ms；
- GPU Kernel 50 ms；
- AllReduce 与同步 15 ms。

如果只盯着最慢的 10 ms Kernel，并把它优化 2 倍，Step 也只是从 100 ms 降到 95 ms。反过来，如果许多小 Kernel 之间存在 15 ms CPU Gap，Kernel Fusion、CUDA Graph 或减少 Python Dispatch 可能比手写单个 Kernel 更有效。

因此分析顺序必须是：

1. 业务是否真的慢；
2. 慢在哪个阶段；
1. 哪些阶段位于 Critical Path；
2. 热点 Kernel 为什么慢；
1. 修改后端到端指标是否改善。

## 性能分析的科学方法

### 第一步：定义目标

训练系统常见目标：

- Step Time、Samples/s、Tokens/s；
- MFU、Goodput；
- 每次有效训练更新的成本；
- Checkpoint 或故障导致的吞吐损失。

推理系统常见目标：

- TTFT、TPOT、ITL、E2E Latency；
- Request/s、Output Tokens/s；
- 满足 P99 SLO 的 Goodput；
- 单位 Token 成本。

不要用“GPU Util 越高越好”代替业务目标。一个死循环 Kernel 可以制造很高的 GPU Util，却没有任何 Goodput。

### 第二步：固定实验条件

最低限度记录：

| 类别 | 需要记录的内容 |
| --- | --- |
| 软件 | OS、Driver、CUDA、cuDNN/NCCL、框架与依赖版本 |
| 硬件 | GPU 型号、数量、Compute Capability、显存、互联与 CPU/NUMA |
| 工作负载 | 模型、精度、Batch、Shape、序列长度、输入分布 |
| 运行状态 | 功耗/温度、时钟、后台任务、MIG、容器与 CPU Affinity |
| 测量协议 | 预热次数、正式重复次数、同步点、统计方法 |

性能对比必须保证正确性和输入等价。若“优化版”少算了一部分、改变精度或偷偷减小 Batch，速度提升没有意义。

### 第三步：先建立端到端基线

推荐至少输出：

- Median；
- P95/P99；
- 最小值与最大值；
- Coefficient of Variation，简称 CV；
- 正确性误差；
- 吞吐或业务 Goodput。

若重复时间为 $t_1$,$t_2$,$\ldots$,$t_n$ ，均值为 $\mu$ ，标准差为 $\sigma$ ：

$$
CV=\frac{\sigma}{\mu}
$$

CV 很高说明实验不稳定。此时先控制环境、增加单次工作量或延长测量时间，不要急着宣称 3% 的提升。

### 第四步：从系统时间线下钻

先用 Nsight Systems 看：

- CPU 是否及时提交工作；
- CUDA API 是否被同步调用阻塞；
- Kernel 之间是否有空洞；
- H2D/D2H 是否与计算重叠；
- 多 Stream 是否真正并行；
- NCCL 是否位于 Critical Path；
- 哪些 NVTX 阶段占时最多；
- 哪些 Kernel 值得进一步分析。

再用 Nsight Compute 分析少数代表性 Kernel。不要一开始就对整个训练任务采集 `--set full` 。

### 第五步：形成可证伪假设

好的假设包含证据、机制和预测：

坏的假设只有工具提示：

Occupancy 低可能由寄存器、Shared Memory、Block 数、尾部波次或算法本身导致；更高 Occupancy 还可能降低每线程资源并拖慢 Kernel。

### 第六步：一次只改一个主要变量

同时改 Batch、精度、DataLoader、Kernel 和编译选项，速度即使提升也无法归因。工程上可以最终组合多项优化，但验证阶段应保留可解释的增量对照。

## 端到端时间模型

### 串行近似

在完全串行的简化系统中：

$$
T_{step}=T_{input}+T_{cpu}+T_{h2d}+T_{kernel}+T_{comm}+T_{sync}+T_{io}
$$

实际系统通常存在重叠，所以不能机械相加。真正决定端到端时间的是依赖图上的 Critical Path：

$$
T_{step}=\max_{p\in Paths}\sum_{i\in p}T_i
$$

这解释了两个常见现象：

- 某个 Memcpy 占用 5 ms，但完全隐藏在 20 ms Kernel 后面，单独优化它不一定缩短 Step；
- 某个 1 ms 同步点让后续所有工作等待，它可能比 5 ms 的非关键任务更值得优化。

### Amdahl 定律

若可优化部分占原总时间比例 p，局部加速倍数为 s，整体理论加速为：

$$
Speedup_{total}=\frac{1}{(1-p)+\frac{p}{s}}
$$

例如热点只占 10%，即使把它变成零成本，整体也最多加速：

$$
\frac{1}{1-0.1}=1.11\times
$$

先算上限，再决定是否投入开发时间。

## Kernel 性能模型

### 有效带宽

若 Kernel 读写总数据量为 B\\\_{useful}，执行时间为 t：

$$
BW_{effective}=\frac{B_{useful}}{t}
$$

对 `out = x + y * z` 的 FP32 元素操作，至少读取 `x/y/z` 并写回 `out` ：

$$
B_{useful}=N\times(3+1)\times4\ bytes
$$

基线若先产生临时张量，再执行加法，会多一次写临时结果和一次读临时结果。Fusion 的收益往往来自减少 Kernel Launch 和 Global Memory Round Trip，而不是增加 FLOPs。

有效带宽应与同一机器、相近访问模式下测得的稳定带宽比较，而不是直接除以规格表中的理想峰值。

### Arithmetic Intensity 与 Roofline

$$
AI=\frac{FLOPs}{Bytes_{DRAM}}
$$

$$
P\leq\min(P_{peak},AI\times BW_{memory})
$$

若测量点贴近 Memory Roof，继续减少算术指令通常帮助有限，应减少数据流量、改善访问模式或提高复用；若贴近 Compute Roof，则应关注 Tensor Core、指令吞吐、流水线和数值精度。

### Occupancy 与延迟隐藏

$$
Occupancy=\frac{Active\ Warps\ per\ SM}{Maximum\ Warps\ per\ SM}
$$

Occupancy 是“可驻留 Warp 数比例”，不是“算力利用率”。它的重要价值是帮助隐藏 Memory/Execution Latency。较低 Occupancy 可能是瓶颈，但较高 Occupancy 不保证更快：

- Kernel 可能有足够的 Instruction-Level Parallelism；
- 增加 Block 可能导致寄存器压缩与 Spill；
- 更多 Warp 可能争抢 Cache、Shared Memory 或 Memory Bandwidth；
- Tail Effect 可能使平均 Occupancy 看似低。

## 工具分工：每把尺子测不同问题

| 工具 | 主要回答 | 不应被误用为 |
| --- | --- | --- |
| 应用日志/压测器 | SLO、Goodput、业务吞吐 | Kernel 根因分析 |
| `nvidia-smi` | 温度、功耗、显存、粗粒度活动 | MFU、精确 Kernel 利用率 |
| DCGM | 集群长期遥测、健康与作业统计 | 单条指令级分析 |
| CUDA Event | 某 Stream 上 GPU 工作耗时 | 端到端 Host Latency |
| `torch.utils.benchmark` | 有预热、重复与同步的微基准 | 完整分布式时间线 |
| NVTX | 给时间线添加业务语义 | 自动给出优化答案 |
| Nsight Systems | CPU/GPU/通信的端到端时间线 | 所有 Kernel 计数器深挖 |
| Nsight Compute | 单个 Kernel 的硬件计数器和分析 | 长任务全量追踪 |
| Compute Sanitizer | 越界、竞态、未初始化和同步错误 | 性能测量工具 |

### 为什么 nvidia-smi 的 GPU Util 不等于 MFU

采样型 Utilization 通常反映一个时间窗口内 Graphics/Compute Engine 是否忙。它不能告诉你：

- SM 是否主要在等待 Memory；
- Tensor Core 是否被使用；
- 有多少运算对业务有效；
- Kernel 是否在做冗余工作；
- P99 是否满足 SLO。

因此它适合发现“GPU 长时间完全空闲”“显存异常增长”“温度导致降频”等现象，不适合单独证明优化成功。

### Nsight Systems：先找时间去哪了

重点观察：

1. \*\*GPU Row\*\*：是否有大段空白；Kernel、Memcpy 与 NCCL 是否重叠。
2. \*\*CUDA API Row\*\*： `cudaDeviceSynchronize` 、 `cudaStreamSynchronize` 、Allocation 是否频繁阻塞。
1. \*\*CPU Thread Row\*\*：Python、DataLoader、锁、文件 I/O 与调度是否拖慢提交。
2. \*\*NVTX Row\*\*：Forward、Backward、Optimizer、Input、Checkpoint 等阶段占比。
1. \*\*CUDA HW/Memory\*\*：不同版本和硬件可显示更细的吞吐与利用信息。

一个常用诊断树：

```
GPU 有明显 Idle Gap？
├─ 是：Gap 时 CPU 在做什么？
│  ├─ 数据/I/O：优化数据管线、预取、Pinned Memory
│  ├─ Python/Launch：融合、编译、CUDA Graph、减少 Dispatch
│  ├─ 同步：消除不必要的 .item()、synchronize、D2H
│  └─ 通信等待：检查慢 Rank、拓扑、Bucket 与重叠
└─ 否：GPU 持续忙
   ├─ 少数大 Kernel：用 Nsight Compute 下钻
   ├─ 大量小 Kernel：检查 Launch Bound 与 Fusion
   └─ Memcpy/通信占主导：检查带宽、拓扑和 Critical Path
```

### Nsight Compute：解释单个 Kernel 为什么慢

推荐从较小的 Section Set 开始，逐步关注：

- \*\*Speed Of Light\*\*：SM 与 Memory 吞吐相对上限；
- \*\*Roofline\*\*：算术强度与 Compute/Memory Roof；
- \*\*Memory Workload Analysis\*\*：DRAM/L2/L1/Shared 流量、合并与命中；
- \*\*Scheduler / Warp State\*\*：Eligible Warp、Issue、主要 Stall 原因；
- \*\*Occupancy\*\*：寄存器、Shared Memory、Block/SM 限制；
- \*\*Source / SASS\*\*：高代价指令与源码对应关系。

不要把某一个 Stall 百分比直接当根因。Stall 指标需要结合 Scheduler 状态、并发 Warp、吞吐和源码解释；某类 Stall 高，有时只是另一个瓶颈的结果。

## 正确计时：异步系统最容易骗你

### 错误示例

```
start = time.perf_counter()
out = torch.mm(a, b)          # 异步提交
elapsed = time.perf_counter() - start
```

这里可能主要测到 Python 调用和 Kernel Launch，而不是 GPU 完成时间。

### Host Wall Time

```
torch.cuda.synchronize()
start = time.perf_counter()
out = torch.mm(a, b)
torch.cuda.synchronize()
elapsed = time.perf_counter() - start
```

它包含 Host Dispatch、GPU 执行和同步开销，适合端到端调用延迟。

### CUDA Event Time

```
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
out = torch.mm(a, b)
end.record()
end.synchronize()
elapsed_ms = start.elapsed_time(end)
```

Event 在 Stream 时间线上记录，适合测 GPU 工作区间。两种时间回答的问题不同，不能混为一谈。

### 预热不可省略

首次运行可能包含：

- CUDA Context 创建；
- Module/Kernel Loading；
- JIT 编译；
- cuBLAS/cuDNN Heuristic 与 Workspace 建立；
- Memory Pool 扩容；
- Page Fault 与 Cache 冷启动。

Cold Start 与 Steady State 都可能是业务指标，但必须分别测量和命名。

## Level 0：无 GPU 的可运行性能分析实验

本实验展示五件事：正确性、预热、重复测量、阶段分解和 Amdahl 上限。基线逐元素计算多项式之和；优化版使用等价闭式公式。它不是为了证明“Python 一定慢”，而是演示如何把猜测变成证据。

将下面代码保存为 `profile_cpu_lab.py` ：

```
#!/usr/bin/env python3
import argparse
import json
import math
import platform
import statistics
import time

def percentile(values, q):
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)

def summarize(samples_s):
    mean = statistics.fmean(samples_s)
    stdev = statistics.pstdev(samples_s)
    return {
        "median_ms": statistics.median(samples_s) * 1e3,
        "p95_ms": percentile(samples_s, 0.95) * 1e3,
        "min_ms": min(samples_s) * 1e3,
        "max_ms": max(samples_s) * 1e3,
        "cv_percent": 100.0 * stdev / mean if mean else 0.0,
    }

def baseline_once(n):
    t0 = time.perf_counter()
    values = list(range(n))
    t1 = time.perf_counter()

    total = 0
    for x in values:
        total += x * x + 3 * x + 7
    t2 = time.perf_counter()

    # 模拟真实流水线中的结果封装，而不是用 sleep 制造瓶颈。
    result = int(total)
    t3 = time.perf_counter()
    phases = {
        "input_ms": (t1 - t0) * 1e3,
        "compute_ms": (t2 - t1) * 1e3,
        "finalize_ms": (t3 - t2) * 1e3,
    }
    return result, phases

def optimized_once(n):
    t0 = time.perf_counter()
    # sum(x) = n(n-1)/2
    # sum(x^2) = n(n-1)(2n-1)/6
    sum_x = n * (n - 1) // 2
    sum_x2 = n * (n - 1) * (2 * n - 1) // 6
    result = sum_x2 + 3 * sum_x + 7 * n
    t1 = time.perf_counter()
    return result, {"closed_form_ms": (t1 - t0) * 1e3}

def run_many(fn, n, repeat):
    samples = []
    phase_totals = {}
    result = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result, phases = fn(n)
        samples.append(time.perf_counter() - t0)
        for key, value in phases.items():
            phase_totals[key] = phase_totals.get(key, 0.0) + value
    phase_mean = {key: value / repeat for key, value in phase_totals.items()}
    return result, summarize(samples), phase_mean

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300_000)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=9)
    args = parser.parse_args()
    if args.n <= 0 or args.repeat < 3 or args.warmup < 0:
        raise SystemExit("要求 n > 0、repeat >= 3、warmup >= 0")

    expected, _ = optimized_once(args.n)
    actual, _ = baseline_once(args.n)
    if actual != expected:
        raise AssertionError(f"结果不一致：baseline={actual}, optimized={expected}")

    for _ in range(args.warmup):
        baseline_once(args.n)
        optimized_once(args.n)

    base_result, base_stats, base_phases = run_many(
        baseline_once, args.n, args.repeat
    )
    opt_result, opt_stats, opt_phases = run_many(
        optimized_once, args.n, args.repeat
    )
    assert base_result == opt_result

    base_total = sum(base_phases.values())
    compute_share = base_phases["compute_ms"] / base_total
    amdahl_if_compute_10x = 1.0 / ((1.0 - compute_share) + compute_share / 10.0)
    measured_speedup = base_stats["median_ms"] / opt_stats["median_ms"]

    report = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "config": vars(args),
        "correctness": True,
        "baseline": {"stats": base_stats, "phase_mean_ms": base_phases},
        "optimized": {"stats": opt_stats, "phase_mean_ms": opt_phases},
        "analysis": {
            "baseline_compute_share": compute_share,
            "amdahl_speedup_if_compute_10x": amdahl_if_compute_10x,
            "measured_algorithmic_speedup": measured_speedup,
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
```

### 运行命令

```
python3 profile_cpu_lab.py --n 300000 --warmup 2 --repeat 9
```

如需保存可比较的基线：

```
mkdir -p reports
python3 profile_cpu_lab.py --n 300000 --warmup 3 --repeat 15 \
  | tee reports/cpu_profile_baseline.json
```

### 预期现象

- `correctness` 为 `true` ；
- 基线的 `compute_ms` 通常占主导；
- 闭式公式的 Median 显著更低；
- 不同 CPU、Python 版本和系统负载会得到不同数值；
- 若 CV 较高，应增加 `n` 或重复次数，并检查后台任务。

### 结果分析

这个实验故意展示“算法变化”而不是微优化。Profiler 先证明逐元素循环是热点，数学推导再消除整个循环。真实 GPU 优化同理：有时应改善 Kernel，有时应融合 Kernel，有时应改变算法或数据布局。

`amdahl_speedup_if_compute_10x` 是在其他阶段不变的假设下得到的上限；实际闭式公式也消除了输入列表构造，所以测得速度可能超过这个局部假设。这里恰好提醒我们：Amdahl 的 p 必须与真实修改边界一致。

## Level 1：通用 NVIDIA GPU Profiling 实验

本实验比较两个等价表达式：

```
baseline:  tmp = y * z; out = x + tmp
optimized: out = addcmul(x, y, z)
```

在 PyTorch Eager 中，基线通常产生两个 Pointwise Kernel 和一个中间张量， `addcmul` 通常可减少 Kernel 与中间显存流量。具体 Kernel 名称、实现和提升随 PyTorch、CUDA、Shape 与 GPU 而变化，必须现场测量。

### 隔离环境

```
python3 -m venv .venv-profile
source .venv-profile/bin/activate
python -m pip install --upgrade pip
# 按 PyTorch 官方安装页选择与驱动匹配的 CUDA Wheel；已有可用环境可跳过。
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

将下面代码保存为 `gpu_profile_lab.py` ：

```
#!/usr/bin/env python3
import argparse
import json
import statistics
import time
from contextlib import contextmanager

import torch

@contextmanager
def nvtx_range(name):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()

def baseline(x, y, z):
    tmp = y * z
    return x + tmp

def optimized(x, y, z):
    return torch.addcmul(x, y, z)

def event_benchmark(name, fn, x, y, z, warmup, repeat):
    with nvtx_range(f"{name}_warmup"):
        for _ in range(warmup):
            out = fn(x, y, z)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record()
    with nvtx_range(f"{name}_measure"):
        for _ in range(repeat):
            out = fn(x, y, z)
    end.record()
    end.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1e3 / repeat
    event_ms = start.elapsed_time(end) / repeat
    return out, {"cuda_event_ms": event_ms, "host_wall_ms": wall_ms}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=4_194_304)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument(
        "--mode", choices=("baseline", "optimized", "both"), default="both"
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("未检测到 CUDA GPU；请先完成 Level 0")
    if args.n <= 0 or args.warmup < 0 or args.repeat <= 0:
        raise SystemExit("参数必须为正数，warmup 可为 0")

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)

    x = torch.randn(args.n, device=device, dtype=torch.float32)
    y = torch.randn_like(x)
    z = torch.randn_like(x)

    # 正确性验证放在正式测量区间外。
    ref = baseline(x, y, z)
    candidate = optimized(x, y, z)
    max_abs_error = (ref - candidate).abs().max().item()
    if not torch.allclose(ref, candidate, rtol=1e-5, atol=1e-6):
        raise AssertionError(f"结果不一致，max_abs_error={max_abs_error}")
    del ref, candidate
    torch.cuda.synchronize()

    results = {}
    if args.mode in ("baseline", "both"):
        out, results["baseline"] = event_benchmark(
            "baseline", baseline, x, y, z, args.warmup, args.repeat
        )
        del out
    if args.mode in ("optimized", "both"):
        out, results["optimized"] = event_benchmark(
            "optimized", optimized, x, y, z, args.warmup, args.repeat
        )
        del out

    if args.mode == "both":
        results["speedup_by_cuda_event"] = (
            results["baseline"]["cuda_event_ms"]
            / results["optimized"]["cuda_event_ms"]
        )

    report = {
        "environment": {
            "torch": torch.__version__,
            "cuda_runtime_reported_by_torch": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "compute_capability": [props.major, props.minor],
            "total_memory_gib": props.total_memory / 2**30,
        },
        "config": vars(args),
        "correctness": {"allclose": True, "max_abs_error": max_abs_error},
        "results": results,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
```

### 直接运行

```
python gpu_profile_lab.py --n 4194304 --warmup 10 --repeat 50 --mode both
```

双 RTX 3080 环境中可分别固定设备运行，避免两个进程误抢同一张卡：

```
CUDA_VISIBLE_DEVICES=0 python gpu_profile_lab.py --mode both
CUDA_VISIBLE_DEVICES=1 python gpu_profile_lab.py --mode both
```

单进程只使用一张 GPU 时，不需要双卡、NVLink 或 NCCL。

### 用 Nsight Systems 看时间线

先确认工具版本：

```
nsys --version
mkdir -p reports
```

采集 CUDA、NVTX 和必要的 OS Runtime 事件，并关闭 CPU Sampling/Context Switch 以降低本实验的额外开销：

```
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  -o reports/gpu_profile \
  python gpu_profile_lab.py --n 4194304 --warmup 5 --repeat 20 --mode both
```

输出统计摘要：

```
nsys stats reports/gpu_profile.nsys-rep
```

用 Nsight Systems GUI 打开 `reports/gpu_profile.nsys-rep` ，比较 `baseline_measure` 与 `optimized_measure` NVTX Range 中的 Kernel 数、持续时间和空洞。

### 用 Nsight Compute 下钻优化版 Kernel

先查看本机可用 Section Set：

```
ncu --version
ncu --list-sets
```

只采集 `optimized_measure` NVTX Push/Pop Range 内第一个匹配 Kernel：

```
ncu \
  --nvtx \
  --nvtx-include "optimized_measure/" \
  --set full \
  --launch-count 1 \
  --force-overwrite \
  -o reports/optimized_kernel \
  python gpu_profile_lab.py --n 4194304 --warmup 5 --repeat 1 --mode optimized
```

`--set full` 可能需要多次 Replay，开销远高于正常执行；它用于实验，不用于生产压测。若只需要快速筛查，先使用当前版本默认 Set，或按 `ncu --list-sections` 选择更少的 Section。

### 预期现象

- 环境信息会自动显示 GPU、Compute Capability、显存、PyTorch 与 CUDA Runtime；
- `allclose` 为 `true` ；
- 基线 Range 通常包含更多 Pointwise Kernel；
- 优化版通常减少一个中间张量的显存往返；
- CUDA Event 与 Host Wall 结果接近但不完全相同；
- 小 Shape 更可能由 Launch Overhead 主导，大 Shape 更可能由 Memory Bandwidth 主导；
- 速度比不是固定常数，甚至在某些版本和 Shape 上可能不提升。

### 结果分析

在 Nsight Systems 中，先确认“少一个 Kernel”和“Range 更短”是否同时成立。若 Kernel 数减少但端到端几乎不变，可能是：

- 输入太大，仍受 DRAM Bandwidth 支配；
- 这一段只占整个应用很小比例；
- 其他同步或 CPU Gap 位于 Critical Path；
- 测量噪声大于收益。

在 Nsight Compute 中，检查优化版是否接近 Memory Throughput 上限、每元素的 DRAM Byte 是否符合预期、Warp 是否有足够的 Eligible Warp。不要因为 Occupancy 未达 100% 就继续盲目调 Block Size。

## Level 2：架构专项分析与边界

不同架构支持的 Counter、Section、硬件单元和工具版本不完全相同。先用以下命令按本机查询，不要把别人的 Metric 名称硬编码进自动化：

```
nvidia-smi --query-gpu=name,driver_version,memory.total,pstate,temperature.gpu,power.draw --format=csv
ncu --query-metrics
ncu --list-sets
ncu --list-sections
```

| 架构 | 常见 GPU | 通用分析重点 | 专项能力与边界 |
| --- | --- | --- | --- |
| Ampere | RTX 3080/3090、A100 | SM、DRAM/L2、Tensor Core、 `cp.async` | RTX 3080/3090 可做 `sm_86` 分析；A100 的 HBM/NVLink/MIG 不应套到消费卡 |
| Ada Lovelace | RTX 4090、L40/L40S | 更高频率、L2、FP8 路径与推理 Kernel | RTX 4090 是 Ada 且无 NVLink；不能写成 Blackwell |
| Hopper | H100/H200 | TMA、FP8、Transformer Engine、NVLink 4 | 必须在真实 Hopper 上验证 TMA/FP8 Counter；Ampere 模拟不等价 |
| Blackwell | RTX 5090、B100/B200/GB200 | 新 Tensor Core、FP4、更新的 Trace/Counter | RTX 5090 是消费级 Blackwell且无数据中心 NVLink；GB200 Fabric 结论不能外推到 5090 |

### 双 RTX 3080 的推荐路径

1. 每张卡单独采集同一工作负载，检查是否存在温度、频率或后台任务差异；
2. 使用 `CUDA_VISIBLE_DEVICES` 固定设备；
1. 单卡分析不需要两张卡同时运行；
2. 到通信课程再用 Nsight Systems 的 NCCL Trace 对比双卡 Critical Path；
1. RTX 3080 没有 H100 TMA、FP8 Transformer Engine 或 GB200 NVLink Fabric，不能用软件标志“模拟验证”这些能力。

### Tool 与目标 GPU 的版本匹配

较新的 GPU 往往需要较新的 Nsight Systems/Compute 才能识别全部 Counter；较新的工具也可能不再支持非常旧的 GPU。分析报告必须记录 Tool 版本。若团队成员使用不同版本，先对 Section 和 Metric 含义做映射，再比较结果。

## 瓶颈分析方法

### 症状一：GPU 利用率呈锯齿，时间线有周期性空洞

检查顺序：

1. 用 NVTX 对齐 Input、Forward、Backward、Optimizer、Checkpoint；
2. 查看空洞期间 CPU 是否在 DataLoader、日志、GC 或 Checkpoint；
1. 检查 `.item()` 、`.cpu()` 、 `print(tensor)` 等隐式同步；
2. 检查 H2D 是否使用 Pinned Memory 和异步拷贝；
1. 只在证据指向输入管线后再增加 Worker。

### 症状二：GPU 持续繁忙，但吞吐低

可能原因：

- Kernel 在做冗余计算；
- 算法复杂度过高；
- 低算术强度 Kernel 持续占用 DRAM；
- 精度没有启用 Tensor Core 路径；
- Shape 不利于底层 Library；
- 大量 Spill 或低效访问。

此时应从热点 Kernel、Roofline、Memory Byte、Instruction Mix 和业务有效工作量下钻，而不是继续追求更高 GPU Util。

### 症状三：许多很短的 Kernel

证据：Nsight Systems 中 CUDA API 与 Kernel 数量巨大，Kernel 间存在 Launch Gap，单个 Kernel 只有数微秒到数十微秒。

候选优化：

- Kernel Fusion；
- `torch.compile` ；
- Triton/自定义融合算子；
- CUDA Graph；
- 批量化小请求。

这些主题将在后续课程分别展开。本课只负责先证明 Launch Bound 是否存在。

### 症状四：一个大 Kernel 很慢

使用 Nsight Compute：

1. 看 Compute 与 Memory 吞吐；
2. 放到 Roofline 判断方向；
1. 检查 DRAM/L2/L1/Shared 数据流量；
2. 检查 Eligible Warp、Issue 与主要 Stall；
1. 检查 Occupancy 的限制资源；
2. 对照 Source/SASS 定位具体指令；
1. 改一个参数后复测 Kernel 与端到端。

### 症状五：多 GPU 中一个 Rank 拖慢全部任务

Collective 的完成时间由最慢 Rank 决定。应把各 Rank 的 NVTX、CUDA 和 NCCL 时间线对齐，检查：

- Rank 0 是否承担额外日志、评估或数据聚合；
- NUMA/PCIe/NIC 亲和性是否不一致；
- 输入 Shape、Token 数或 Expert Load 是否失衡；
- 某张 GPU 是否降频、报错或被其他进程共享；
- 通信是否真正与计算重叠。

## Profiler 自身会改变被观察对象

### Nsight Systems 开销

Trace API 越多、调用栈越深、采样越密，报告越大、扰动越高。先采集短而有代表性的区间；生产长任务使用 Capture Range、延迟启动或窄 Trace 配置。

### Nsight Compute Replay

硬件 Counter 无法总在一次执行中全部采集。Nsight Compute 可能 Replay Kernel，并保存/恢复可访问 Memory。后果包括：

- 运行时间显著放大；
- Cache 状态可能与自然执行不同；
- 非确定性或有外部副作用的 Kernel 需要谨慎；
- 采集过多 Kernel 会非常慢。

因此只筛选代表性 Kernel 和 Range，并把 Profiler 报告视为诊断数据，而不是正常吞吐基准。

### Counter 权限

Linux 驱动可能限制普通用户访问 GPU Performance Counter。出现权限错误时，应由管理员按组织安全策略配置；不要用不受控的特权容器绕过策略。

### DCGM 与 Nsight 冲突

持续采集 DCGM Profiling Counter 可能与 Nsight 工具竞争硬件计数器。若报告提示资源占用，应在维护窗口暂停对应 DCGM Profiling 采集，完成后恢复；基础健康遥测与 Profiling Counter 要区分。

## 正确性优先：Compute Sanitizer

性能数据建立在程序正确的前提上。CUDA Toolkit 提供：

```
compute-sanitizer --tool memcheck ./app
compute-sanitizer --tool racecheck ./app
compute-sanitizer --tool initcheck ./app
compute-sanitizer --tool synccheck ./app
```

- `memcheck` ：越界、Misaligned Access、Leak 和硬件异常；
- `racecheck` ：Shared Memory Hazard；
- `initcheck` ：未初始化的 Device Memory 访问；
- `synccheck` ：错误的同步原语使用。

它们会显著拖慢程序，是正确性工具，不应用其耗时评价性能。官方建议通常先跑 `memcheck` ，再分析 Race 或 Sync 问题。

## 常见错误与排查

### 错误一：没有同步就用 Python Timer 测 GPU

\*\*现象\*\*：结果小得离谱，增大矩阵几乎不变。

\*\*排查\*\*：使用 CUDA Event；或在 Host Wall 测量前后 `torch.cuda.synchronize()` 。

### 错误二：把第一次运行当稳定性能

\*\*现象\*\*：第一次比后续慢很多。

\*\*排查\*\*：把 Cold Start 与 Steady State 分开；预热后再重复测量。

### 错误三：只报告平均值

\*\*现象\*\*：平均值提升，但 P99 恶化；或少数异常值扭曲结论。

\*\*排查\*\*：同时报告 Median、Tail、Range、CV 和样本数。

### 错误四：把 Occupancy 当 KPI

\*\*现象\*\*：Occupancy 提高，Kernel 反而变慢。

\*\*排查\*\*：检查寄存器 Spill、Cache、Memory Throughput、Eligible Warp 和端到端时间。

### 错误五：对所有 Kernel 使用 ncu --set full

\*\*现象\*\*：任务运行极慢、报告巨大，甚至看似卡住。

\*\*排查\*\*：先用 Nsight Systems 找热点，再用 NVTX、Kernel Filter 和 `--launch-count` 限制范围。

### 错误六：Profiler 下速度比与正常运行不同

\*\*原因\*\*：Trace、Counter Replay、Cache Reset、序列化与文件写入改变执行。

\*\*排查\*\*：Profiler 用于解释机制，最终速度用无 Profiler 的稳定基准复测。

### 错误七：ncu 无权限访问 Counter

\*\*排查\*\*：记录完整错误、Driver 和 Tool 版本，由管理员配置 Performance Counter 权限；云环境还要确认实例与虚拟化限制。

### 错误八：Nsight Systems 找不到 NVTX Range

\*\*排查\*\*：确认代码实际执行到 `range_push/pop` ；Trace 中包含 `nvtx` ；Push/Pop 成对；目标进程在采集范围内。

### 错误九：优化结果不一致

\*\*排查\*\*：先比较误差与容差；检查精度、Reduction Order、随机种子、Shape、NaN/Inf 和输入是否相同。数值不等价的结果不能直接比较速度。

### 错误十：测试时 GPU 正在降频

\*\*排查\*\*：记录温度、功耗、P-State、Clock 和其他进程；保证散热稳定，并在相近条件下交错运行 A/B，而不是上午测 A、下午测 B。

## 优化前后对照模板

| 项目 | 优化前 | 优化后 | 判断标准 |
| --- | --- | --- | --- |
| 正确性 | Reference 结果 | 误差在容限内 | 必须通过 |
| 端到端 Median/P99 | 基准分布 | 同协议复测 | SLO/Goodput 改善 |
| GPU Idle Gap | 时间线证据 | Gap 缩短 | Critical Path 改善 |
| Kernel 数 | 原数量 | 融合后数量 | 少不等于必然快 |
| Hot Kernel 时间 | 原时间 | 新时间 | 无 Profiler 复测 |
| DRAM Byte/吞吐 | Counter | Counter | 与假设一致 |
| Occupancy/Spill | 诊断数据 | 诊断数据 | 不作为单独 KPI |
| 功耗与温度 | 原状态 | 新状态 | 条件可比 |
| 工程复杂度 | 原实现 | 新维护成本 | 收益覆盖成本 |

一次合格的性能结论应写成：

不要把某次 RTX 3080、RTX 4090、H100 或 RTX 5090 的数值写成所有 GPU 的通用结论。

## 面试题与答案

### 1\. 为什么 GPU Util 100% 仍可能性能很差？

GPU Util 常表示采样窗口内引擎处于活动状态，不说明执行的是有效工作，也不说明 SM、Tensor Core 或 Memory 达到有效上限。Memory Stall、冗余 Kernel、低效算法甚至死循环都可能产生高 Util。

### 2\. Nsight Systems 和 Nsight Compute 的核心区别是什么？

Nsight Systems 做系统级时间线，找 CPU、GPU、Memcpy、同步和通信之间的时间关系；Nsight Compute 深挖单个 Kernel 的硬件计数器和指令行为。先 Systems，后 Compute。

### 3\. 为什么 CUDA Kernel 计时需要同步？

Kernel Launch 对 Host 通常异步。不同步的 Host Timer 可能只测到提交时间。CUDA Event 在 Stream 上记录；Host Wall 测量则需在边界同步。

### 4\. Occupancy 越高越好吗？

不是。足够的 Active Warp 有助于隐藏延迟，但更高 Occupancy 可能压缩寄存器、引起 Spill 或增加资源竞争。最终以 Kernel 和端到端时间判断。

### 5\. 如何判断 Kernel 是 Compute Bound 还是 Memory Bound？

结合 Arithmetic Intensity、Roofline、Compute/Memory Throughput、DRAM/L2 流量与指令吞吐。单看一个百分比不够；还要确认实际工作量和测得的稳定硬件上限。

### 6\. 为什么 Nsight Compute 会很慢？

许多硬件 Counter 不能一次采完，工具需要 Replay Kernel； `full` Set、多个 Kernel、大 Memory Footprint 和首次 Context 配置都会增加开销。

### 7\. Amdahl 定律在优化中有什么作用？

它用热点占比和局部加速估算整体上限，防止花大量时间优化只占 1% 的路径。并行和重叠系统还要结合 Critical Path 分析。

### 8\. 如何证明 Kernel Fusion 的收益机制？

正确性通过后，用 Systems 证明 Kernel Launch 和 Gap 减少，用 Compute/模型证明中间 Memory Traffic 下降，再用无 Profiler 的端到端基准确认速度提升。

### 9\. Compute Sanitizer 与 Profiler 有什么关系？

Sanitizer 先验证 Memory、Race、Initialization 和 Synchronization 正确性；Profiler 再分析性能。Sanitizer 的耗时不能作为性能结果。

### 10\. 为什么要使用 NVTX？

底层 Kernel 名称不一定能直接映射到业务阶段。NVTX 把 Input、Forward、Decode、AllReduce 等语义标注到时间线上，也可用于 Nsight Compute 的范围筛选。

## 课后练习

1. 修改 Level 0 的 `n` ，绘制 Median 和 CV 随问题规模的变化；解释小规模为什么更受 Timer Overhead 影响。
2. 给 CPU 实验加入一个固定成本的 JSON 序列化阶段，用 Amdahl 定律预测只优化计算的上限。
1. 在一张可用 NVIDIA GPU 上运行 Level 1，记录 GPU、版本、Shape、Event Time、Wall Time 和速度比。
2. 用 Nsight Systems 截取两个 NVTX Range，数出 Kernel 数并定位 CPU/GPU Gap。
1. 用 Nsight Compute 的默认 Set 与 `full` Set 分别采集同一 Kernel，比较报告时间和 Section 数量。
2. 把 `n` 从 2^{16} 增大到 2^{26}，观察优化收益从 Launch Bound 向 Memory Bound 的变化。
1. 人为在 PyTorch 循环中加入 `.item()` ，用时间线证明隐式 D2H 同步的影响，然后移除。
2. 对双 GPU 的同一工作负载分别采集报告，分析性能差异是频率、温度、其他进程还是软件配置造成。

## Checklist

### 基线与正确性

- 明确业务 KPI、SLO 和输入分布。
- 固定模型、Shape、Batch、精度和随机性。
- 记录 Driver、CUDA、Framework、Tool 与硬件版本。
- 区分 Cold Start 与 Steady State。
- 设置预热、重复次数与同步边界。
- 报告 Median、Tail、CV 和样本数。
- 优化前后正确性通过。

### 时间线分析

- 先用 Nsight Systems 找 Critical Path。
- 用 NVTX 标记业务阶段。
- 检查 GPU Idle、Launch Gap、Memcpy、Sync 和 NCCL。
- 检查 CPU、DataLoader、I/O 与日志。
- 只把少数代表性 Kernel 交给 Nsight Compute。

### Kernel 分析

- 结合 Roofline 判断 Compute/Memory 方向。
- 检查 DRAM/L2/L1/Shared 流量。
- 检查 Scheduler、Eligible Warp 和 Stall。
- 检查 Occupancy 限制与 Register Spill。
- 不把高 Occupancy 或高 GPU Util 当作最终 KPI。
- 记录 Profiler Replay 和权限边界。

### 架构边界

- RTX 3080/3090 与 A100 正确归入 Ampere。
- RTX 4090/L40/L40S 正确归入 Ada Lovelace。
- H100/H200 正确归入 Hopper。
- RTX 5090、B100/B200/GB200 正确归入 Blackwell。
- 没有用 Ampere 伪装验证 Hopper TMA 或 Blackwell 专属能力。
- 性能数字只描述本次环境，不外推为通用结论。

### 最终验证

- 每个优化都有“证据—机制—预测”。
- 一次只改变一个主要变量。
- 用无 Profiler 的同协议基准复测端到端结果。
- 结合 Amdahl 和工程维护成本判断是否值得上线。
- 把报告、命令、配置和结论保存为可复现实验记录。