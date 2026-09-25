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

### 4.2 CPU 时间不等于 GPU 执行时间

CUDA launch 通常异步。CPU 可能只花几十微秒提交一个运行数毫秒的 Kernel。普通 Python 计时若不在边界同步，测到的主要是提交时间：

```
torch.cuda.synchronize()
start = time.perf_counter()
run()
torch.cuda.synchronize()
elapsed = time.perf_counter() - start
```

微基准优先使用 `torch.utils.benchmark.Timer` 或 CUDA Event；端到端基准必须明确同步边界。

### 4.3 Count 是重要信号

某个算子单次只占 5 微秒，但每 step 调用 10,000 次，总开销仍可能巨大。高 Count 常提示：

- Python 循环没有向量化。
- Pointwise 算子没有 Fusion。
- 输入被切成大量小块。
- 日志、`.item()` 或小 Tensor 创建太频繁。

### 4.4 Shape、Stack 和 Memory 都有代价

`record_shapes=True` 、 `with_stack=True` 、 `profile_memory=True` 能提供强证据，但会增加开销；shape 记录还可能延长 Tensor 生命周期。先用轻配置找到可疑窗口，再对 2～5 个 step 开启重配置。

## 五、诊断阶梯：从轻到重

### Level A：稳定基准

固定模型、数据 shape、dtype、batch、线程数、随机种子和设备状态。记录 warmup 后的多次结果、中位数与分位数。

### Level B：快速入口

CPU 或混合负载可先用：

```
python -m torch.utils.bottleneck your_script.py --your-args
```

它同时给出 Python cProfile 与 Autograd Profiler 摘要，但 CUDA 异步会让普通 CPU Profile 产生误导，因此只用于第一轮筛查。

### Level C：算子聚合

用 `torch.profiler.profile` 收集 CPU/设备 Activity，查看 `key_averages()` ：

- CPU-bound：按 `self_cpu_time_total` 排序。
- GPU-bound：按对应设备 self time 排序。
- 调用碎片：关注 Count。
- shape 特化：按 input shape 分组。

### Level D：时间线

导出 Chrome Trace，用 Perfetto UI 或兼容查看器观察：

- CPU launch 与 GPU 执行之间是否有空洞。
- Memcpy 是否串行。
- 前向、反向、optimizer 是否清晰。
- 是否频繁 `cudaDeviceSynchronize` 。
- 多 Stream 是否真正重叠。

### Level E：显存快照

PyTorch Memory Snapshot 能查看 caching allocator 的 Segment、Block、分配/释放历史和 OOM 时状态。它只看 PyTorch allocator 管理的内存，NCCL 或其他 CUDA 库直接分配的显存可能不可见。

### Level F：系统与 Kernel

- Nsight Systems：CPU、CUDA Runtime、Memcpy、NCCL、NVTX 全局时间线。
- Nsight Compute：单个 Kernel 的 Roofline、吞吐、Occupancy、Warp 与 Memory 指标。
- DCGM / `nvidia-smi dmon` ：长时间温度、功耗、时钟和利用率。

## 六、常见性能形态

### 6.1 CPU launch-bound

现象：GPU 时间线上大量短 Kernel，中间有空洞；CPU 线程持续忙于 Python/Dispatcher；GPU 利用率锯齿。

候选优化：向量化、算子融合、 `torch.compile` 、CUDA Graph、减少 Python 控制流。第 20～21 课会深入这些方法。

### 6.2 数据管线饥饿

现象：step 开头长时间没有 GPU 活动； `enumerate(DataLoader)` 或预处理范围很长；CPU/磁盘忙。

候选优化：增加合理 workers、Pinned Memory、批量 H2D、 `non_blocking=True` 、缓存、异步预取。必须区分 DataLoader 慢与 H2D 慢。

### 6.3 隐式同步

常见触发：

- 在热路径调用 `.item()` 。
- 把 CUDA Tensor 打印或转为 NumPy。
- 频繁查询需要设备结果的属性。
- 错误的计时同步。

现象：CUDA Runtime 线程出现同步调用，CPU 等待 GPU，流水线被切断。

### 6.4 算子碎片化

现象：大量 `aten::add` 、 `mul` 、 `copy_` 、小 reduction；单个很快，总数极多。

候选优化：Fusion、in-place（先确认 Autograd 正确性）、布局调整、编译器捕获。优化前要确认不是 Profile 本身制造了额外开销。

### 6.5 显存碎片或泄漏

- `allocated` 持续上涨：可能有 Tensor 被 Python 容器、闭包或计算图引用。
- `reserved` 高但 `allocated` 低：可能是缓存池正常行为，也可能有不可复用碎片。
- `nvidia-smi` 高于 PyTorch：可能包含 CUDA Context、NCCL、第三方库或其他进程。

`empty_cache()` 只能释放完全空闲的缓存 Segment，不会释放仍被 Tensor 引用的内存，也通常不是训练循环里的性能优化。

### 6.6 分布式慢 rank

平均 rank 没有意义，Collective 会等待最慢 rank。每个 rank 单独保存 trace，用相同 step ID 对齐，寻找数据、日志、checkpoint 或拓扑差异。

## 七、关键性能模型

### 7.1 Profiler 扰动

设原始耗时为 $T$ ，采集事件、栈、shape 和写 trace 的开销为 $(O_p)$ ：

$$
T_{profile}=T+O_p
$$

需要优化的是原始 $T$ ，不是带着完整 Profiler 的 $(T_{profile})$ 。因此性能结论必须回到无 Profiler 基准复验。

### 7.2 GPU 空闲比例

对一个 step：

$$
IdleRatio=1-\frac{T_{GPU\ busy}}{T_{step}}
$$

多 Stream 有重叠时不能简单把所有 Kernel duration 相加；应对时间轴区间求并集。

### 7.3 尾延迟与方差

$$
CV=\frac{\sigma}{\mu}
$$

均值相近但 CV 或 P99 变差，通常意味着动态 shape、GC、数据 I/O、后台进程、温度/时钟或周期性 checkpoint。性能工程不能只报最优单次结果。

## 八、三级实验

## Level 0：纯 Python Trace 摘要诊断

这个实验在没有 PyTorch、没有 GPU 时也能运行。它不模拟 CUDA，只训练“如何把现象转成可验证假设”。

保存为 `profile_summary_analyzer.py` ：

```
#!/usr/bin/env python3
import argparse
import json
import statistics

SAMPLE = [
    {"step_ms": 120, "gpu_busy_ms": 70, "input_ms": 32,
     "launch_gap_ms": 12, "sync_ms": 3, "memory_mb": 5200},
    {"step_ms": 124, "gpu_busy_ms": 71, "input_ms": 35,
     "launch_gap_ms": 13, "sync_ms": 3, "memory_mb": 5310},
    {"step_ms": 130, "gpu_busy_ms": 72, "input_ms": 39,
     "launch_gap_ms": 14, "sync_ms": 4, "memory_mb": 5430},
    {"step_ms": 138, "gpu_busy_ms": 73, "input_ms": 44,
     "launch_gap_ms": 15, "sync_ms": 5, "memory_mb": 5560},
    {"step_ms": 146, "gpu_busy_ms": 74, "input_ms": 49,
     "launch_gap_ms": 16, "sync_ms": 6, "memory_mb": 5700},
]

def percentile(values, q):
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac

def slope(values):
    n = len(values)
    x_mean = (n - 1) / 2
    y_mean = statistics.mean(values)
    num = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(values))
    den = sum((i - x_mean) ** 2 for i in range(n)) or 1
    return num / den

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", help="JSON 文件，内容为 step 记录数组")
    args = p.parse_args()
    rows = json.load(open(args.input, encoding="utf-8")) if args.input else SAMPLE
    if not rows:
        raise SystemExit("没有记录")

    steps = [r["step_ms"] for r in rows]
    idle = [1 - r["gpu_busy_ms"] / r["step_ms"] for r in rows]
    mem = [r["memory_mb"] for r in rows]
    report = {
        "step_p50_ms": round(percentile(steps, 0.50), 2),
        "step_p95_ms": round(percentile(steps, 0.95), 2),
        "mean_gpu_idle_ratio": round(statistics.mean(idle), 3),
        "input_mean_ms": round(statistics.mean(r["input_ms"] for r in rows), 2),
        "launch_gap_mean_ms": round(statistics.mean(r["launch_gap_ms"] for r in rows), 2),
        "sync_mean_ms": round(statistics.mean(r["sync_ms"] for r in rows), 2),
        "memory_growth_mb_per_step": round(slope(mem), 2),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    hypotheses = []
    if report["mean_gpu_idle_ratio"] > 0.25:
        hypotheses.append("GPU 空闲较多：先检查输入管线和 CPU launch 空洞")
    if report["input_mean_ms"] > 0.20 * report["step_p50_ms"]:
        hypotheses.append("输入阶段占比高：用 record_function 拆 DataLoader/H2D/预处理")
    if report["launch_gap_mean_ms"] > 0.08 * report["step_p50_ms"]:
        hypotheses.append("launch gap 偏高：检查小算子、Python 循环与同步")
    if report["memory_growth_mb_per_step"] > 5:
        hypotheses.append("显存随 step 上涨：检查 Tensor/计算图引用并抓 Memory Snapshot")
    print("\n优先验证的假设：")
    for i, text in enumerate(hypotheses or ["未触发规则，扩大采样窗口"], 1):
        print(f"{i}. {text}")

if __name__ == "__main__":
    main()
```

运行：

```
python3 profile_summary_analyzer.py
```

预期现象：脚本指出 GPU 空闲、输入占比、launch gap 和显存增长风险。规则只生成假设，不能直接宣判根因；下一步要用 PyTorch Profiler 或 Memory Snapshot 验证。

## Level 1：CPU/GPU 通用 PyTorch Profiler 实验

### 8.1 隔离环境

```
python3 -m venv .venv-profiler
source .venv-profiler/bin/activate
python -m pip install --upgrade pip
python -m pip install "torch>=2.8,<2.14"
```

### 8.2 完整可运行代码

保存为 `pytorch_profile_lab.py` ：

```
#!/usr/bin/env python3
import argparse
import contextlib
import json
import os
import time

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile, record_function

class MLP(nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers += [nn.Linear(width, width), nn.GELU()]
        layers += [nn.Linear(width, width)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

@contextlib.contextmanager
def phase(name, use_nvtx):
    if use_nvtx:
        torch.cuda.nvtx.range_push(name)
    try:
        with record_function(name):
            yield
    finally:
        if use_nvtx:
            torch.cuda.nvtx.range_pop()

def preprocess(x, mode):
    if mode == "baseline":
        # 故意保留 Python 循环和许多小 Tensor 操作。
        return torch.stack([(row - row.mean()) / (row.std() + 1e-5) for row in x])
    return (x - x.mean(dim=1, keepdim=True)) / (
        x.std(dim=1, keepdim=True) + 1e-5
    )

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["baseline", "optimized"], required=True)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--with-stack", action="store_true")
    p.add_argument("--memory-snapshot", action="store_true")
    p.add_argument("--out", default="profiler_artifacts")
    return p.parse_args()

def main():
    args = parse_args()
    has_cuda = torch.cuda.is_available()
    device = "cuda" if args.device == "auto" and has_cuda else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not has_cuda:
        raise SystemExit("请求 CUDA，但 torch.cuda.is_available() 为 False")

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(2026)
    use_cuda = device == "cuda"
    if use_cuda:
        p = torch.cuda.get_device_properties(0)
        print({"torch": torch.__version__, "cuda": torch.version.cuda,
               "gpu": p.name, "cc": f"{p.major}.{p.minor}",
               "vram_gib": round(p.total_memory / 1024**3, 2)})
    else:
        print({"torch": torch.__version__, "device": "cpu",
               "threads": torch.get_num_threads()})

    model = MLP(args.width, args.depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    host_batch = torch.randn(args.batch, args.width)

    if args.memory_snapshot and use_cuda:
        torch.cuda.memory._record_memory_history(max_entries=100000)

    def train_step():
        optimizer.zero_grad(set_to_none=True)
        with phase("input/preprocess", use_cuda):
            x = preprocess(host_batch, args.mode)
        with phase("input/h2d", use_cuda):
            x = x.to(device, non_blocking=use_cuda)
        with phase("model/forward", use_cuda):
            y = model(x)
            loss = y.float().square().mean()
        with phase("model/backward", use_cuda):
            loss.backward()
        with phase("optimizer/step", use_cuda):
            optimizer.step()
        return loss.detach()

    # 无 Profiler 预热，隔离 CUDA 初始化与优化器状态首次创建。
    for _ in range(3):
        train_step()
    if use_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # 先做低扰动端到端计时。
    start = time.perf_counter()
    for _ in range(args.steps):
        train_step()
    if use_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    benchmark = {
        "mode": args.mode,
        "device": device,
        "step_ms": elapsed * 1000 / args.steps,
        "samples_per_s": args.batch * args.steps / elapsed,
    }
    if use_cuda:
        benchmark.update({
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        })
    print("BENCHMARK", json.dumps(benchmark, ensure_ascii=False))

    activities = [ProfilerActivity.CPU]
    if use_cuda:
        activities.append(ProfilerActivity.CUDA)

    def save_trace(prof):
        path = os.path.join(args.out, f"{args.mode}_step_{prof.step_num}.json")
        prof.export_chrome_trace(path)
        print("TRACE", path)

    schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)
    with profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=save_trace,
        record_shapes=True,
        profile_memory=True,
        with_stack=args.with_stack,
        with_flops=True,
    ) as prof:
        # wait + warmup + active = 5；多跑一步确保 cycle 结束并导出。
        for _ in range(6):
            train_step()
            prof.step()

    sort_key = "self_cuda_time_total" if use_cuda else "self_cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_key, row_limit=20))

    if args.memory_snapshot and use_cuda:
        path = os.path.join(args.out, f"{args.mode}_memory_snapshot.pickle")
        torch.cuda.memory._dump_snapshot(path)
        print("MEMORY_SNAPSHOT", path)

if __name__ == "__main__":
    main()
```

### 8.3 运行命令

CPU 回退：

```
python pytorch_profile_lab.py --mode baseline --device cpu --width 512
python pytorch_profile_lab.py --mode optimized --device cpu --width 512
```

自动选择 NVIDIA GPU：

```
python pytorch_profile_lab.py --mode baseline --device auto
python pytorch_profile_lab.py --mode optimized --device auto
```

只在缩小窗口后开启 stack 和显存历史：

```
python pytorch_profile_lab.py --mode baseline --device cuda \
  --with-stack --memory-snapshot --steps 10
```

查看输出：

- 把 `profiler_artifacts/*.json` 拖入 Perfetto UI 或兼容 Trace Viewer。
- 把 `*_memory_snapshot.pickle` 拖入 PyTorch Memory Visualizer。
- 快照可能包含调用栈、路径和张量信息，外发前按组织安全规范处理。

### 8.4 预期现象

- Baseline 的 `input/preprocess` 更长，并出现更多小算子调用。
- Optimized 用按维度归约和广播替代 Python 行循环，CPU launch 数减少。
- GPU 上端到端加速幅度取决于预处理占比；模型 GEMM 很重时，Amdahl 定律会限制收益。
- Profile 运行时间会比无 Profile Benchmark 慢，不能拿两者直接算优化加速比。
- 具体数值只属于该次环境。

## Level 2：NVIDIA GPU 与多卡专项

### 8.5 支持矩阵

| 平台 | PyTorch Profiler | Nsight Systems/Compute | 架构专项注意 | | -------------------------------------------------- | -- | --------- | ---------------------------------- | | Ampere RTX 3080/3090、A100 | 支持 | 支持 | 3080 双卡通常走 PCIe；A100 视 PCIe/SXM 拓扑 | | Ada RTX 4090、L40/L40S | 支持 | 支持 | 4090 是 Ada 且无 NVLink | | Hopper H100/H200 | 支持 | 支持 | 可另测 Transformer Engine/FP8，但不是通用基线 | | Blackwell RTX 5090、B100/B200/GB200 | 支持 | 需匹配新版本工具链 | 消费卡与 NVLink/NVSwitch 数据中心系统不可等价 |

Profiler 方法本身跨架构通用，Kernel 名称、可用 dtype、性能计数器和工具版本并不通用。

### 8.6 Nsight Systems 交叉验证

脚本已用 NVTX 标出阶段。先用本机帮助确认参数：

```
nsys profile --help
nsys profile --trace=cuda,nvtx,osrt,cublas \
  --output=pytorch_systems --force-overwrite=true \
  python pytorch_profile_lab.py --mode baseline --device cuda --steps 10
```

检查：

1. `input/preprocess` 是否阻塞后续 GPU 提交。
2. H2D 是否集中为大传输，还是大量小传输。
1. CPU launch 是否持续领先 GPU。
2. 是否出现同步 API 和长 GPU 空洞。
1. PyTorch Profiler 的热点是否与系统时间线一致。

### 8.7 Nsight Compute 深入 Kernel

只有先用时间线找到真正影响端到端的 Kernel，才进入 Nsight Compute：

```
ncu --set basic --target-processes all \
  python pytorch_profile_lab.py --mode optimized --device cuda --steps 3
```

全面指标采集会多次 replay Kernel、显著改变运行时间。用 Kernel 名称或 NVTX 范围过滤，避免对整个训练脚本全量采集。

### 8.8 双卡与分布式

多 rank 应分别输出 trace：

```
traces/rank_0/...
traces/rank_1/...
```

每个 rank 使用相同 step ID 和 NVTX 名称。对齐后观察最慢 rank 到达 All-Reduce/All-Gather 的时间，而不是只打开 rank 0。没有 NVLink/NVSwitch 的双 RTX 3080 只能验证 PCIe/NCCL 场景，不能模拟数据中心 Fabric。

## 九、如何从 Trace 得出结论

### 9.1 先找 Critical Path

Trace 中事件很多，但真正影响 step 的是 Critical Path。重叠在其他工作的事件即使累计时间很高，也可能不是暴露延迟。

### 9.2 再形成可证伪假设

不好的结论：

好的假设：

### 9.3 最后无 Profile 复验

优化前后保持：

- 相同输入 shape、dtype、batch 和模型。
- 相同 warmup、测量窗口和同步边界。
- 至少多次重复，报告中位数和离散程度。
- 正确性阈值一致。

## 十、优化前后对照

| 场景 | 错误做法 | 正确做法 |
| --- | --- | --- |
| CUDA 计时 | 只包 `perf_counter` | 边界同步或使用正确计时器 |
| 第一次迭代慢 | 当作稳定性能 | 预热后再测 |
| Profiler 很重 | 全程开 stack/shape/memory | 用 schedule 只抓短窗口 |
| Top 算子很贵 | 直接重写 Kernel | 先确认是否在 Critical Path |
| reserved 很大 | 每步 `empty_cache()` | 查 allocated、快照和复用状态 |
| GPU 利用率低 | 盲目增 batch | 时间线区分输入、launch、同步、通信 |
| 多卡慢 | 只看 rank 0 | 对齐所有 rank 并找 straggler |
| 优化后更快 | 用 Profile 时间证明 | 回到无 Profile 基准复验 |

## 十一、常见错误与排查

### 11.1 Trace 文件为空

使用非默认 schedule 时，每个 iteration 后必须调用 `prof.step()` ；循环长度还要覆盖 wait、warmup、active 和保存边界。

### 11.2 看不到 CUDA Activity

检查：

```
print(torch.cuda.is_available())
print(torch.version.cuda)
```

并确认 activities 包含 `ProfilerActivity.CUDA` 。驱动、PyTorch wheel 与 CUPTI 环境必须匹配。

### 11.3 Profile 后 OOM

shape、stack、memory history 会保留额外元数据， `record_shapes` 还可能延长 Tensor 引用。缩短 active steps、关闭不需要的选项，并限制 Memory History entries。

### 11.4 CUDA 时间看起来比 step 还长

多个 Stream 的事件可能重叠，聚合表的时间不是端到端串行和；父子事件也不能重复相加。回到时间线看区间并集。

### 11.5.item() 排名不高但程序很慢

真正代价可能显示为同步 Runtime 调用或等待，而不是 `.item()` 的 Python 自身时间。用 Trace 查看它前后的 CPU/GPU Flow。

### 11.6 nvidia-smi 显存大于 PyTorch

差值可能来自 CUDA Context、NCCL、cuBLAS workspace、扩展库或其他进程。PyTorch Memory Snapshot 不保证看见非 PyTorch allocator 分配。

### 11.7 Baseline 与 Optimized loss 不同

性能比较先失效。检查归一化维度、默认 `std` 估计方式、dtype、随机数和 optimizer 状态。只有正确性等价后才讨论速度。

### 11.8 Profile 每次结果差异很大

固定 shape 与种子，增加 warmup；检查 CPU 频率、GPU 温度/功耗、后台任务、DataLoader 随机 I/O 和动态编译。Profile 窗口要取稳定区间。

## 十二、面试题与答案

### 题1：为什么普通 Python 计时会低估 CUDA 耗时？

CUDA Kernel 异步提交，CPU 计时可能在 GPU 完成前就结束。需要在边界同步、使用 CUDA Event，或使用能正确处理设备异步的基准工具。

### 题2：Self CPU 与 CPU Total 有什么区别？

Self 只包含事件自身时间，Total 包含所有子事件。找叶子热点多看 Self，分析模块整体成本看 Total。

### 题3：为什么 Profiler 表不能直接相加得到 step time？

存在父子包含关系、多个 Stream 重叠、CPU/GPU 并行和异步 Flow，直接相加会重复计算或忽略重叠。

### 题4：allocated 与 reserved 的区别是什么？

allocated 是活跃 Tensor 占用；reserved 是 PyTorch caching allocator 从 CUDA 获得并管理的 Segment，总量可包含可复用空闲块。二者差值不等于泄漏。

### 题5：GPU 利用率低通常有哪些原因？

输入饥饿、Python launch-bound、小算子碎片、同步、通信等待、batch 太小、CPU/磁盘慢、动态 shape 或周期性 checkpoint。利用率只是症状。

### 题6：什么时候使用 with\_stack=True？

在算子热点已确定但不知道源代码调用位置时，对很短 active 窗口使用；不应默认覆盖整个长训练任务。

### 题7：PyTorch Profiler、Nsight Systems、Nsight Compute 如何分工？

PyTorch Profiler关联 Module/ATen 与设备活动；Nsight Systems看全系统时间线和重叠；Nsight Compute深挖单 Kernel 的硬件执行效率。

### 题8：如何证明一次优化有效？

保持工作负载与正确性一致，用无 Profile、多次重复的基准证明端到端改善，并保存 Profile 证据解释为什么改善；同时报告显存与尾延迟副作用。

### 题9：为什么多卡 Profile 必须看所有 rank？

Collective 由最慢 rank 决定。rank 0 正常不代表其他 rank 没有数据、I/O、拓扑或日志瓶颈。

### 题10：什么是观测者效应？

采集本身会增加事件记录、栈回溯、内存元数据和 Trace 写出开销，使程序行为偏离原始状态，所以结论必须用低扰动基准复验。

## 十三、课后练习

1. 修改 Level 1，使 Baseline 每 step 调用一次 `.item()` ，在 Trace 中找到同步证据。
2. 把 20 个 Pointwise 操作改成一个等价表达式，比较算子 Count 与端到端时间。
1. 分别打开 shape、stack、memory，量化三类采集对 Profile 时间和显存的影响。
2. 构造一个把 loss Tensor 持续 append 到列表的泄漏，并用 Memory Snapshot 定位。
1. 为 DataLoader、H2D、forward、backward、optimizer、checkpoint 添加统一阶段标签。
2. 在双卡训练上保存 rank 0/1 Trace，找出 Collective 的最晚到达者。
1. 选择一个热点 Kernel，用 Nsight Compute 判断是 Compute-bound 还是 Memory-bound。
2. 建立一份性能回归 JSON：环境、命令、P50/P95、吞吐、峰值显存、正确性摘要。

## 十四、Checklist

### 测量

- 固定模型、shape、dtype、batch、种子和线程数。
- 完成 warmup，CUDA 计时边界正确同步。
- 重复测量并报告中位数与离散程度。
- Profile 前先有无 Profile 基线。

### Profile

- 使用短 schedule，逐项开启 shape、stack、memory。
- 每个 iteration 调用 prof.step()。
- 用 record\\\_function 标记业务阶段。
- 区分 Self/Total、CPU/Device、Count/Duration。
- 从时间线识别 Critical Path，不直接相加聚合时间。

### 显存

- 同时记录 allocated、reserved 与峰值。
- Memory Snapshot 限制 entries 并只抓问题窗口。
- 明确快照看不到的第三方/非 PyTorch 分配。
- 不把 empty\\\_cache() 当作泄漏修复。

### 多卡与硬件

- 所有 rank 使用统一 step ID 并分别保存 Trace。
- 检查 GPU 拓扑、时钟、功耗和温度。
- 4090 标记为 Ada，5090 标记为 Blackwell。
- NVLink/NVSwitch、FP8/FP4 结果不向不支持硬件外推。

### 结论

- 每个结论都能指出具体证据和可证伪假设。
- 每次只改变一个主要变量。
- 正确性、吞吐、尾延迟和显存一起比较。
- 最终收益由无 Profile 基准确认。