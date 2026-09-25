---
title: "项目1：GPU 性能分析实验"
source: "https://everythingai.top/learn/ai-systems-performance-engineering-text/project-01"
author:
published:
created: 2026-09-26
description: "面向 AI 工程学习者的专业课程学习空间。"
tags:
  - "clippings"
---
**导航**

本项目使用真实分析流程定位 GPU 性能瓶颈，并输出可复现的测量结果与优化报告。

## 项目定位

这个项目把前 28 课里最重要的一条方法论真正跑通：\*\*不要凭 GPU 利用率猜瓶颈，要用可复现基准、系统时间线和 Kernel 指标建立证据链。\*\*

你将从一个同时包含“小算子启动开销、显存流量和矩阵乘法”的 PyTorch 工作负载出发，依次完成：

1. 固化环境与工作负载；
2. 建立无 Profiler 的墙钟基线；
1. 用 PyTorch Profiler 找到高开销算子；
2. 用 Nsight Systems 判断 CPU、CUDA API、Memcpy 和 GPU Kernel 的时间关系；
1. 用 Nsight Compute 深入一个 Kernel，判断它受计算、带宽、延迟还是资源限制；
2. 只修改一个变量，验证正确性并复测；
1. 输出一份别人可以复现、审阅和继续优化的性能报告。

本项目不要求特定 GPU。没有 NVIDIA GPU 时可以完整完成 Level 0，并在 CPU 上完成 Level 1 的方法训练；有常见 NVIDIA GPU 时完成 CUDA 路径；Nsight Compute 硬件计数器不可用时，保留 PyTorch Profiler 与 Nsight Systems 证据，不伪造 Kernel 级结论。

## 学习目标

完成项目后，你应该能够：

- 区分基准测试、系统级 Profiling 与 Kernel 级 Profiling；
- 正确测量异步 CUDA 工作负载，不把 CPU 提交时间当成 GPU 执行时间；
- 根据时间线识别 Launch-bound、Memory-bound、Compute-bound 和同步等待；
- 使用算术强度与 Roofline 为优化方向设定上界；
- 用 PyTorch Profiler、NVTX、Nsight Systems 和 Nsight Compute 形成逐层下钻的证据链；
- 避免“同时改很多变量”“只跑一次”“只看平均值”等常见实验错误；
- 交付包含环境、原始数据、报告、正确性检查和结论边界的性能分析包。

## 前置知识

- 能运行 Python 与基础 Bash 命令；
- 理解 Latency、Throughput、P50/P95/P99；
- 理解 GPU Kernel、CUDA Stream、异步执行和显存层级；
- 理解算术强度、Roofline、Occupancy 的基本含义；
- 可选：已安装 PyTorch、CUDA Toolkit、Nsight Systems、Nsight Compute。

## 最终交付物

建议建立如下目录：

```
gpu-profiling-project/
├── roofline_planner.py
├── gpu_profiling_lab.py
├── reports/
│   ├── environment.txt
│   ├── roofline.json
│   ├── benchmark.json
│   ├── pytorch_trace.json
│   ├── nsys_baseline.nsys-rep
│   └── ncu_matmul.ncu-rep
└── analysis.md
```

并在 `analysis.md` 中回答四个问题：

1. 当前最主要的瓶颈是什么？
2. 哪些证据支持这个判断？
1. 修改了哪个变量，为什么？
2. 提升是否通过正确性、稳定性与跨尺寸复测？

## 核心直觉：Profiler 是显微镜，不是秒表

性能工程应按由粗到细的顺序推进：

```
业务指标异常
  ↓
可复现墙钟基准：问题真的存在吗？
  ↓
系统时间线：时间花在 CPU、I/O、通信还是 GPU？
  ↓
框架算子：哪个算子或阶段最贵？
  ↓
Kernel 指标：为什么这个 Kernel 慢？
  ↓
单变量优化 → 正确性检查 → 复测 → 回归门禁
```

Profiler 会增加开销，因此不能用带完整 Profiler 的一次运行替代正常基准。正确做法是：

- 用轻量墙钟基准确认问题和收益；
- 用 Profiler 解释原因；
- 用不带 Profiler 的重复实验确认最终收益。

## 性能模型

### 端到端延迟

对一组重复测量值排序后，使用中位数和尾延迟：

$$
P_q = \operatorname{percentile}(t_1,t_2,\ldots,t_n,q)
$$

小型实验至少报告 P50 与 P95。平均值容易被初始化、缓存填充、频率变化和偶发系统抖动影响。

### 加速比

$$
S = \frac{T_{baseline}}{T_{optimized}}
$$

`S > 1` 表示优化后更快。只有在输入、精度、输出语义和测量策略相同时，这个比值才有意义。

### 有效带宽

若一次算子最少读写 `B` 字节，耗时为 `t` ：

$$
BW_{effective}=\frac{B}{t}
$$

它是算法层面的有效带宽，不等于 GPU 规格表峰值，也不等于 Nsight Compute 观察到的全部物理事务。

### 算术强度与 Roofline

$$
AI=\frac{FLOPs}{Bytes}
$$

$$
P_{attainable}\leq \min(P_{peak}, BW_{peak}\times AI)
$$

其中 `AI` 是每搬运一个字节完成多少次浮点运算。屋脊转折点为：

$$
AI_{ridge}=\frac{P_{peak}}{BW_{peak}}
$$

- `AI < AI_ridge` ：更可能受内存带宽限制；
- `AI > AI_ridge` ：更可能受计算峰值限制；
- 远低于两条屋脊：还可能受 Launch、依赖链、分支、Occupancy、同步或输入流水线限制。

Roofline 是方向模型，不是瓶颈判决器。最终判断要结合时间线与硬件指标。

## 瓶颈诊断矩阵

| 现象 | 优先查看 | 可能原因 | 首个实验 |
| --- | --- | --- | --- |
| GPU 时间线有大量短 Kernel 与空隙 | Nsight Systems | Python/Launch-bound、同步过多 | 合并小算子或减少循环 |
| GPU 长时间空闲，CPU 线程繁忙 | PyTorch Profiler、OSRT | 数据准备、日志、锁、线程调度 | 关闭日志并隔离数据阶段 |
| 单个 Kernel 很长，DRAM 吞吐高 | Nsight Compute Memory | Memory-bound | 减少中间张量与内存流量 |
| 单个 Kernel 很长，计算管线高 | Nsight Compute SpeedOfLight | Compute-bound | 精度、Tensor Core、算法与 Tiling |
| Occupancy 低但吞吐不低 | Occupancy + Stall | 可能是合理资源交换 | 不要只为提高 Occupancy 改代码 |
| 每轮都出现同步锯齿 | CUDA API、NVTX | `.item()` 、隐式同步、阻塞拷贝 | 删除或降低同步频率 |
| 多卡只有一个 Rank 落后 | 各 Rank 时间线 | 负载不均、拓扑、I/O、日志 | 分 Rank 采集并对齐 NVTX |

## 实验前的测量纪律

### 冻结工作负载

每次比较必须保持一致：

- 输入尺寸、数据类型和随机种子；
- PyTorch、CUDA、驱动和依赖版本；
- GPU 型号、GPU 数量和可见设备；
- Warmup 次数、正式迭代次数与统计方法；
- 精度策略、编译开关与同步位置；
- 正确性容差。

如果同时更换 GPU、PyTorch、精度和算法，只能把结果标为探索性观察，不能把收益归因给某一个优化。

### 记录环境

```
mkdir -p gpu-profiling-project/reports
cd gpu-profiling-project

{
  date -Iseconds
  uname -a
  lscpu | sed -n '1,25p'
  free -h
  command -v nvidia-smi >/dev/null && nvidia-smi -L
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version,memory.total,pci.bus_id,pstate,temperature.gpu,power.limit --format=csv
  command -v nvcc >/dev/null && nvcc --version
  command -v nsys >/dev/null && nsys --version
  command -v ncu >/dev/null && ncu --version
  python --version
} 2>&1 | tee reports/environment.txt
```

不要在报告中只写“RTX 3080”或“H100”。驱动、软件版本、功耗状态、温度和输入形状都可能改变结果。

## Level 0：无 GPU 的 Roofline 与诊断训练

下面的脚本只依赖 Python 标准库。它根据 FLOPs、最小数据搬运量、设备峰值计算能力和内存带宽计算理论上界，并明确给出“模型推断”而不是“实测结论”。

将下面代码保存为 `roofline_planner.py` ：

```
#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

def analyze(flops, bytes_moved, peak_tflops, bandwidth_gbps):
    if min(flops, bytes_moved, peak_tflops, bandwidth_gbps) <= 0:
        raise ValueError("所有参数必须大于 0")

    peak_flops = peak_tflops * 1e12
    bandwidth = bandwidth_gbps * 1e9
    arithmetic_intensity = flops / bytes_moved
    ridge_point = peak_flops / bandwidth
    memory_roof = bandwidth * arithmetic_intensity
    attainable = min(peak_flops, memory_roof)
    lower_bound_seconds = flops / attainable
    bound = "memory-model-bound" if arithmetic_intensity < ridge_point else "compute-model-bound"

    return {
        "flops": flops,
        "bytes_moved": bytes_moved,
        "peak_tflops": peak_tflops,
        "bandwidth_gbps": bandwidth_gbps,
        "arithmetic_intensity_flop_per_byte": arithmetic_intensity,
        "ridge_point_flop_per_byte": ridge_point,
        "roofline_attainable_gflops": attainable / 1e9,
        "roofline_time_lower_bound_ms": lower_bound_seconds * 1e3,
        "model_classification": bound,
        "warning": "这是理论方向模型；真实瓶颈需由基准与 Profiler 验证。",
    }

def main():
    parser = argparse.ArgumentParser(description="通用 Roofline 规划器")
    parser.add_argument("--flops", type=float, default=2e9)
    parser.add_argument("--bytes", dest="bytes_moved", type=float, default=1e9)
    parser.add_argument("--peak-tflops", type=float, default=20.0)
    parser.add_argument("--bandwidth-gbps", type=float, default=500.0)
    parser.add_argument("--output", default="reports/roofline.json")
    args = parser.parse_args()

    result = analyze(
        args.flops,
        args.bytes_moved,
        args.peak_tflops,
        args.bandwidth_gbps,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

if __name__ == "__main__":
    main()
```

### 运行命令

```
python roofline_planner.py \
  --flops 2000000000 \
  --bytes 1000000000 \
  --peak-tflops 20 \
  --bandwidth-gbps 500

python roofline_planner.py \
  --flops 2000000000000 \
  --bytes 1000000000 \
  --peak-tflops 20 \
  --bandwidth-gbps 500 \
  --output reports/roofline_compute_case.json
```

### 预期现象与结果分析

第一组的算术强度较低，模型会指向内存屋脊；第二组的算术强度较高，模型会指向计算屋脊。这里的峰值参数只是教学输入，不代表任何具体 GPU。正式分析时应填入当前精度、当前设备与当前功耗条件下对应的官方或实测上限。

如果一个实际 Kernel 的性能远低于 Roofline 两条屋脊，先检查 Launch、依赖、分支、Occupancy 和同步，不要直接断言“显存带宽不够”。

## Level 1：通用 PyTorch CPU/CUDA 性能实验

### 隔离环境

```
python -m venv .venv-project1
source .venv-project1/bin/activate
python -m pip install --upgrade pip
python -m pip install "torch>=2.5,<3"
```

如果默认安装包与本机 CUDA 不匹配，请使用 PyTorch 官方安装选择器给出的命令。没有 NVIDIA GPU 时可以安装 CPU wheel；不要为了运行本实验替换生产机驱动。

下面代码为本课程原创实验，不直接复制配套仓库。它自动检测 CPU/CUDA、GPU 型号、Compute Capability、显存与软件版本，并包含三类工作负载：

- `launch` ：许多微小算子，对比一次向量化表达；
- `memory` ：产生中间张量的逐元素链，对比减少分配的原地版本；
- `matmul` ：用于观察高算术强度矩阵乘法。

将代码保存为 `gpu_profiling_lab.py` ：

```
#!/usr/bin/env python3
import argparse
import contextlib
import json
import math
import platform
import statistics
import time
from pathlib import Path

import torch

def percentile(values, q):
    data = sorted(values)
    if not data:
        return float("nan")
    position = (len(data) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return data[lower]
    weight = position - lower
    return data[lower] * (1 - weight) + data[upper] * weight

def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def architecture_name(capability):
    major, minor = capability
    if (major, minor) == (8, 9):
        return "Ada Lovelace"
    if major == 8:
        return "Ampere"
    if major == 9:
        return "Hopper"
    if major >= 10:
        return "Blackwell-or-newer"
    return f"pre-Ampere-or-other-sm_{major}{minor}"

def environment(device):
    info = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "pytorch_cuda_runtime": torch.version.cuda,
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        capability = torch.cuda.get_device_capability(device)
        info.update({
            "gpu_name": props.name,
            "compute_capability": f"{capability[0]}.{capability[1]}",
            "architecture_family": architecture_name(capability),
            "vram_gib": round(props.total_memory / 2**30, 2),
            "gpu_count": torch.cuda.device_count(),
        })
    return info

@contextlib.contextmanager
def nvtx_range(name, device):
    enabled = device.type == "cuda"
    if enabled:
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if enabled:
            torch.cuda.nvtx.range_pop()

def timed(fn, device, warmup, iterations):
    with torch.inference_mode():
        for _ in range(warmup):
            output = fn()
        synchronize(device)

        samples_ms = []
        for _ in range(iterations):
            start = time.perf_counter_ns()
            output = fn()
            synchronize(device)
            samples_ms.append((time.perf_counter_ns() - start) / 1e6)

    return output, {
        "p50_ms": statistics.median(samples_ms),
        "p95_ms": percentile(samples_ms, 0.95),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "iterations": iterations,
    }

def check_close(reference, candidate):
    torch.testing.assert_close(reference, candidate, rtol=2e-4, atol=2e-4)
    return True

def build_workloads(args, device):
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    x = torch.randn(args.elements, device=device, generator=generator)
    a = torch.randn((args.matrix, args.matrix), device=device, generator=generator)
    b = torch.randn((args.matrix, args.matrix), device=device, generator=generator)

    def launch_baseline():
        y = x.clone()
        for _ in range(args.steps):
            y = y + 0.001
        return y

    def launch_optimized():
        return x + args.steps * 0.001

    def memory_baseline():
        return torch.relu(x * 1.1 + 0.3).square()

    def memory_optimized():
        y = x.clone()
        return y.mul_(1.1).add_(0.3).relu_().square_()

    def matmul():
        return a @ b

    return {
        "launch": (launch_baseline, launch_optimized),
        "memory": (memory_baseline, memory_optimized),
        "matmul": (matmul, None),
    }

def profile_once(name, fn, device, output_path):
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
    ) as profiler:
        with torch.profiler.record_function(name):
            with nvtx_range(name, device):
                output = fn()
                synchronize(device)

    profiler.export_chrome_trace(str(output_path))
    sort_key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    print(profiler.key_averages().table(sort_by=sort_key, row_limit=15))
    return output

def capture_one_matmul(fn, device):
    if device.type != "cuda":
        raise RuntimeError("--capture-kernel 需要 CUDA 设备")
    with torch.inference_mode():
        for _ in range(5):
            output = fn()
        synchronize(device)
        torch.cuda.cudart().cudaProfilerStart()
        with nvtx_range("capture_matmul", device):
            output = fn()
            synchronize(device)
        torch.cuda.cudart().cudaProfilerStop()
    print(float(output.flatten()[0]))

def main():
    parser = argparse.ArgumentParser(description="通用 GPU 性能分析综合实验")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--mode", choices=["all", "launch", "memory", "matmul"], default="all")
    parser.add_argument("--elements", type=int, default=1_000_000)
    parser.add_argument("--matrix", type=int, default=512)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--capture-kernel", action="store_true")
    parser.add_argument("--output", default="reports/benchmark.json")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但 PyTorch 未检测到可用 CUDA GPU")
    device = torch.device(
        "cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu"
    )
    if min(args.elements, args.matrix, args.steps, args.warmup, args.iters) <= 0:
        raise ValueError("尺寸、步骤与迭代参数必须大于 0")

    workloads = build_workloads(args, device)
    if args.capture_kernel:
        capture_one_matmul(workloads["matmul"][0], device)
        return

    selected = list(workloads) if args.mode == "all" else [args.mode]
    report = {
        "environment": environment(device),
        "config": vars(args),
        "results": {},
    }
    print(json.dumps(report["environment"], indent=2, ensure_ascii=False))

    for name in selected:
        baseline, optimized = workloads[name]
        with nvtx_range(f"{name}_baseline_benchmark", device):
            baseline_output, baseline_stats = timed(
                baseline, device, args.warmup, args.iters
            )
        item = {"baseline": baseline_stats}

        if optimized is not None:
            with nvtx_range(f"{name}_optimized_benchmark", device):
                optimized_output, optimized_stats = timed(
                    optimized, device, args.warmup, args.iters
                )
            item["optimized"] = optimized_stats
            item["correct"] = check_close(baseline_output, optimized_output)
            item["speedup_p50"] = baseline_stats["p50_ms"] / optimized_stats["p50_ms"]

        report["results"][name] = item
        print(name, json.dumps(item, indent=2, ensure_ascii=False))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.profile:
        profile_name = selected[0]
        trace_path = output_path.parent / "pytorch_trace.json"
        profile_once(
            f"{profile_name}_baseline_profile",
            workloads[profile_name][0],
            device,
            trace_path,
        )
        print(f"PyTorch trace: {trace_path}")

if __name__ == "__main__":
    main()
```

### 运行命令

CPU 或自动设备：

```
python gpu_profiling_lab.py --device auto --mode all
python gpu_profiling_lab.py --device cpu --mode launch --profile \
  --output reports/cpu_benchmark.json
```

NVIDIA GPU：

```
CUDA_VISIBLE_DEVICES=0 python gpu_profiling_lab.py \
  --device cuda --mode all --output reports/gpu_benchmark.json

CUDA_VISIBLE_DEVICES=0 python gpu_profiling_lab.py \
  --device cuda --mode memory --profile \
  --output reports/gpu_memory_benchmark.json
```

### 预期现象

- `launch` 优化版通常显著减少算子与 Kernel 启动次数；输入较小时，收益往往比输入较大时更明显。
- `memory` 优化版减少中间张量分配，但原地链仍可能包含多个 Kernel；它不等价于编译器生成的真正融合 Kernel。
- `matmul` 的性能受尺寸、精度、库版本、Tensor Core 可用性和频率状态影响。
- CPU 与 GPU 的绝对延迟不可直接比较；它们主要用于练习同一套测量与诊断方法。
- 第一次运行可能包含 CUDA Context、库初始化和缓存建立开销，因此必须 Warmup。

### 如何读 PyTorch Profiler

优先查看：

1. `self_cpu_time_total` ：Python/框架侧谁占用 CPU；
2. `self_cuda_time_total` ：哪个算子对应的 CUDA Kernel 最耗时；
1. 调用次数：是否存在大量微小算子；
2. 输入形状：慢算子是否只在特定尺寸出现；
1. 内存分配：是否频繁创建与释放中间张量。

`record_shapes` 、 `profile_memory` 、 `with_stack` 都可能增加开销。诊断完成后应关闭它们，再跑纯墙钟基准。

## Level 2：NVIDIA Nsight 专项实验

### 支持边界

| 架构 | 常见 GPU | 本项目可做 | 注意事项 |
| --- | --- | --- | --- |
| Ampere | RTX 3080/3090、A100 | Systems、Compute、TF32/FP16 对照 | 消费卡可能受性能计数器权限限制 |
| Ada Lovelace | RTX 4090、L40/L40S | Systems、Compute、Kernel 对照 | RTX 4090 不是 Blackwell |
| Hopper | H100/H200 | Systems、Compute、可选 FP8 | FP8 结论不能由低代 GPU 等价模拟 |
| Blackwell | RTX 5090、B100/B200/GB200 | Systems、Compute、可选 FP4/新 Tensor Core | 数据中心与消费卡能力并不相同 |

Nsight Systems 主要回答“何时发生、各阶段如何重叠”；Nsight Compute 主要回答“一个 CUDA Kernel 为什么慢”。前者适合先扫全局，后者采集硬件计数器时会重放 Kernel，开销可能很大，不应直接包住完整训练任务。

### Nsight Systems：先看全局时间线

```
mkdir -p reports

nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  --output=reports/nsys_baseline \
  python gpu_profiling_lab.py --device cuda --mode all --warmup 5 --iters 10

nsys stats reports/nsys_baseline.nsys-rep \
  | tee reports/nsys_baseline_stats.txt
```

打开 `.nsys-rep` 后检查：

- NVTX 范围是否清楚分隔 baseline 与 optimized；
- CUDA API 调用与 GPU Kernel 之间是否存在长空隙；
- 是否出现大量短 Kernel；
- `cudaDeviceSynchronize` 、 `cudaStreamSynchronize` 是否过多；
- Host-to-Device/Device-to-Host 拷贝是否与计算重叠；
- GPU 空闲时 CPU 在执行什么。

如果系统限制 CPU Sampling，不需要直接使用 `sudo` 。本命令已关闭 Sampling 与 Context Switch 采集，仍可分析 CUDA/NVTX 时间线。

### Nsight Compute：只抓一个矩阵乘 Kernel

脚本的 `--capture-kernel` 会先完成分配和 Warmup，再通过 CUDA Profiler API 只开放一个采集窗口：

```
ncu \
  --profile-from-start off \
  --set full \
  --target-processes all \
  --force-overwrite \
  --export reports/ncu_matmul \
  python gpu_profiling_lab.py \
    --device cuda --mode matmul --matrix 1024 --capture-kernel
```

若 `--set full` 太慢，可先使用：

```
ncu \
  --profile-from-start off \
  --section SpeedOfLight \
  --section MemoryWorkloadAnalysis \
  --section Occupancy \
  --section LaunchStats \
  --force-overwrite \
  --export reports/ncu_matmul_focused \
  python gpu_profiling_lab.py \
    --device cuda --mode matmul --matrix 1024 --capture-kernel
```

重点回答：

- SM 与 Memory 吞吐哪一侧更接近上限？
- DRAM、L2、L1/TEX 哪一级压力最大？
- 活跃 Warp、寄存器、Shared Memory、Block 尺寸如何限制 Occupancy？
- Warp Stall 的主要类别是什么？
- 当前 Kernel 是否使用预期的矩阵计算路径？

不要把“Occupancy 不是 100%”自动判定为问题。高寄存器或 Shared Memory 使用可能换来更少的访存和更高吞吐，最终仍以实际时间与瓶颈指标为准。

### 性能计数器权限不足

如果出现 `ERR_NVGPUCTRPERM` 或类似错误：

1. 记录错误与当前驱动、容器、虚拟化环境；
2. 在有权限的机器上让管理员按 NVIDIA 官方安全策略开放计数器；
1. 无法开放时，使用 Nsight Systems、PyTorch Profiler 和墙钟基准完成项目；
2. 不要声称已经验证 DRAM 吞吐、Warp Stall 或 Occupancy 原因。

## 优化前后对照

把实际结果填入下表，不要照抄示例数字：

| 工作负载 | Baseline P50 | Optimized P50 | P95 | 加速比 | 正确性 | 主要证据 | | ------------------------------------------------- | -- | --- | -- | --- | ----- | ------------------------ | | launch | 待测 | 待测 | 待测 | 待测 | 通过/失败 | Kernel 数、GPU 空隙 | | memory | 待测 | 待测 | 待测 | 待测 | 通过/失败 | 分配、DRAM/L2、Kernel 数 | | matmul | 待测 | 不适用 | 待测 | 不适用 | 不适用 | SM/Memory Roof、Tensor 路径 |

一份合格结论应类似：

## 完整结果分析流程

### 第一步：确认基准可信

- 是否 Warmup？
- CUDA 测量是否同步？
- 是否报告了多次测量的分布？
- 基线与优化是否使用同一输入？
- 温度、P-State、后台任务是否异常？

### 第二步：确定最大时间贡献者

先找占总时间最多的阶段。优化占比 1% 的算子，即使把它完全消除，端到端收益也不会超过约 1%。

### 第三步：提出可证伪假设

坏假设：

好假设：

### 第四步：一次只改一个变量

保持输入、精度、迭代和环境不变，只修改循环表达式。若同时启用 `torch.compile` 、FP16、CUDA Graph 和更大 Batch，就无法确定收益来源。

### 第五步：验证收益边界

至少改变两组输入尺寸重新运行：

```
python gpu_profiling_lab.py --device auto --mode launch --elements 10000
python gpu_profiling_lab.py --device auto --mode launch --elements 1000000
python gpu_profiling_lab.py --device auto --mode launch --elements 10000000
```

小输入常受 Launch 影响，大输入可能逐渐转为内存带宽限制。优化不一定在所有尺寸都保持同一加速比。

## 常见错误与排查

### 错误 1：CUDA 计时没有同步

症状：耗时异常小，只测到 CPU 提交 Kernel 的时间。

排查：在计时边界使用 CUDA Event，或像本项目一样在每次样本结束时 `torch.cuda.synchronize()` 。同步会改变流水方式，因此端到端服务测量还应使用真实并发负载工具。

### 错误 2：把第一次运行纳入统计

症状：P95 极高、重复运行差异大。

排查：增加 Warmup，检查 CUDA Context、cuBLAS、编译缓存、内存池和功耗状态是否已稳定。

### 错误 3：Profiler 打开太多选项

症状：报告巨大、运行极慢、结果与正常基准差异明显。

排查：先用 Systems 做短时间线，再用 Compute 抓一个 Kernel；只有需要定位源码时才打开调用栈与更多指标。

### 错误 4：优化后结果不一致

症状：代码更快，但 `assert_close` 失败。

排查：确认运算次序、数据类型、溢出、随机性与容差。性能收益不能覆盖语义错误。

### 错误 5：GPU 利用率高就认为没有问题

症状： `nvidia-smi` 显示 100%，但吞吐远低于预期。

排查：利用率表示采样窗口内 GPU 在忙，不表示 SM、Tensor Core 或显存带宽有效饱和。继续查看 Roofline、Kernel 指标与业务 Goodput。

### 错误 6：在容器中看不到 GPU 或计数器

排查顺序：

```
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
ls -l /dev/nvidia* 2>/dev/null
ncu --version
```

确认宿主机驱动、NVIDIA Container Toolkit、设备挂载和安全策略。不要在不理解风险时修改生产机权限。

### 错误 7：直接套用别人的峰值数字

不同架构、形态、功耗、精度和软件版本的峰值不可混用。RTX 4090 属于 Ada Lovelace；RTX 5090 属于 Blackwell。A100、H100、B200 的数据中心能力也不能直接代表同代消费卡。

## 架构专项观察建议

### Ampere：RTX 3080/3090、A100

- 用 FP32 矩阵乘法观察 TF32 策略对性能和误差的影响；
- 检查矩阵尺寸是否适合 Tensor Core 路径；
- 双 RTX 3080 通常以 PCIe 通信，不能把它当作 NVLink/NVSwitch 环境。

### Ada Lovelace：RTX 4090、L40/L40S

- 与 Ampere 对比时保持精度、形状和软件版本一致；
- RTX 4090 没有 NVLink，双卡实验要单独测 PCIe 与拓扑；
- 不要把 Ada 结果标成 Blackwell。

### Hopper：H100/H200

- 可选验证 FP8、Transformer Engine 与更高带宽路径；
- 必须同时报告精度校验与实际 Kernel 路径；
- 没有 Hopper 时只能学习方法，不能用 FP16 模拟出等价 FP8 硬件结论。

### Blackwell：RTX 5090、B100/B200/GB200

- 可选验证 FP4、Block Scaling、新 Tensor Core 或 TMA 路径；
- 先检测 Compute Capability 和库支持，避免代码默默回退；
- RTX 5090 与 B200/GB200 的互联、显存和数据中心特性不同，结论必须分开。

## 面试题与答案

### 1\. 为什么先用 Nsight Systems，再用 Nsight Compute？

\*\*答：\*\* Systems 用低得多的分析成本给出 CPU、CUDA API、Memcpy、Kernel、NVTX 和空闲区间的全局时间关系，先确定问题是否真的在某个 GPU Kernel。Compute 会采集并可能重放 Kernel，适合对已经锁定的少数 Kernel 做硬件级分析。反过来容易在非瓶颈 Kernel 上浪费时间。

### 2\. 为什么 nvidia-smi 的 GPU 利用率不能证明程序高效？

\*\*答：\*\* 它主要表示采样窗口内 GPU 是否有工作，不表示工作是否使用了足够的 SM、Tensor Core、显存带宽，也不表示业务 Goodput 达标。高利用率下仍可能有低 Occupancy、低 Warp 效率、冗余计算或低价值工作。

### 3\. CUDA 工作负载为什么需要特殊计时？

\*\*答：\*\* CUDA Kernel 通常异步提交。普通 CPU 计时可能只覆盖提交开销。需要 CUDA Event 或在边界显式同步，才能覆盖设备执行；同时要明确同步本身是否属于业务路径。

### 4\. Occupancy 越高越好吗？

\*\*答：\*\* 不一定。高 Occupancy 有助于隐藏延迟，但增加寄存器或 Shared Memory 可能减少访存并提高单 Warp 效率。最优点取决于吞吐、Stall、内存流量和实际延迟，不能只追求 100%。

### 5\. 如何证明一个优化是有效的？

\*\*答：\*\* 固定工作负载和环境，一次只改一个变量；验证输出正确；重复测量 P50/P95；保存基线与优化的原始数据；用 Profiler 证据解释原因；跨尺寸复测；最后在不带 Profiler 的真实路径确认端到端收益。

### 6\. Roofline 判断为 Memory-bound，下一步一定是提高 Occupancy 吗？

\*\*答：\*\* 不一定。首先应减少数据搬运、提高复用、融合 Kernel、改布局或提高合并访问效率。Occupancy 只影响隐藏延迟的能力，不能消除不必要的字节流量。

### 7\. Profiler 下加速，正常运行却不加速，可能是什么原因？

\*\*答：\*\* Profiler 改变了调度和开销比例；采集区间过短；优化只减少了 Profiler 记录事件；正常运行受其他阶段限制；或基准方差大。应回到不带 Profiler 的重复墙钟基准，并检查端到端占比。

### 8\. 为什么要保存原始报告，而不仅是截图？

\*\*答：\*\* 原始报告包含可检索指标、时间线、环境和上下文，便于复核、二次分析和版本比较。截图只适合沟通局部现象，不能承担完整证据链。

## 课后练习

1. 把 `launch` 的 `steps` 分别设为 1、10、50、200，画出 Kernel 数、P50 和加速比的关系。
2. 把 `memory` 的元素数量从 `10^4` 扫到 `10^7` ，判断瓶颈何时从 Launch 逐渐转向内存流量。
1. 在支持 CUDA 的设备上比较矩阵尺寸 256、512、1024、2048，记录 P50 与 Nsight Compute 的 SM/Memory 吞吐。
2. 在同一 GPU 上比较 FP32 与 FP16/BF16 矩阵乘法，必须同时记录正确性误差和实际 Kernel 路径。
1. 人为在循环内加入 `.item()` ，使用 Nsight Systems 找到同步锯齿，再删除并复测。
2. 为 `benchmark.json` 增加 Git Commit、主机名和依赖快照字段。
1. 为一个真实训练或推理脚本加入 NVTX：数据加载、前向、反向、优化器、通信、保存检查点。
2. 编写回归规则：若 P50 退化超过 5% 且重复两轮仍成立，则阻止合并；同时说明如何处理测量噪声。

## 项目验收 Checklist

### 环境与可复现性

- 已保存操作系统、CPU、GPU、驱动、CUDA、PyTorch 和 Profiler 版本。
- 已固定输入、随机种子、精度、Warmup 和迭代策略。
- 已记录 GPU 温度、功耗状态或说明无法获取。
- 未把一台机器的绝对性能写成所有 GPU 的通用结论。

### 测量正确性

- CUDA 墙钟测量处理了异步执行。
- 报告包含 P50、P95、最小值、最大值和迭代次数。
- 基线与优化使用相同输入与输出语义。
- 优化结果通过 torch.testing.assert\\\_close。
- 最终收益由关闭 Profiler 的基准确认。

### 证据链

- Level 0 Roofline 规划器已运行并保存 JSON。
- Level 1 CPU 或 CUDA 基准已运行并保存 JSON。
- 已生成并阅读 PyTorch Profiler Trace。
- 有 NVIDIA GPU 时已生成 Nsight Systems 时间线，或记录无法执行的原因。
- 有计数器权限时已对一个 Kernel 运行 Nsight Compute，或明确标注未验证。
- 结论同时引用墙钟、时间线和 Kernel/算子证据。

### 优化与复盘

- 每次实验只改变一个主要变量。
- 已解释最主要瓶颈，而不是罗列所有指标。
- 已跨至少三个输入尺寸复测。
- 已记录优化无效或退化的实验。
- 已写明结论适用范围与不可等价模拟边界。
- 已整理 reports/，他人可按命令复现。