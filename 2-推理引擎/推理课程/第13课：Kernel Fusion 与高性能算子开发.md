---
title: "第13课：Kernel Fusion 与高性能算子开发"
source: "https://everythingai.top/learn/ai-systems-performance-engineering-text/lesson-13"
author:
published:
created: 2026-09-26
description: "面向 AI 工程学习者的专业课程学习空间。"
tags:
  - "clippings"
---
**导航**

本课从中间张量与内存流量出发理解 Kernel Fusion，并掌握高性能自定义算子的设计与验证方法。

## 课程定位

很多 AI Kernel 的算术并不复杂：乘法、加法、激活、缩放、Dropout、Mask、Residual。问题在于，如果每一步都成为独立 Kernel，中间结果就要反复写入 Global Memory、再读回来，同时 CPU 还要不断提交 Kernel Launch。

Kernel Fusion 的核心并不是“把代码写进同一个函数”，而是让 Producer 产生的数据尽量停留在 Register 或 Shared Memory 中，直接交给 Consumer，而不落到显存中的中间 Tensor。

本课从可计算的显存流量模型出发，讲清 Fusion 为什么有效、什么时候会适得其反，以及如何完成一个真正可测量的 CUDA 融合算子。第 14 课才会系统学习 Triton；本课使用原生 CUDA C++ 和 PyTorch Extension，建立底层直觉。

## 学习目标

完成本课后，你能够：

1. 区分 Launch Fusion、Memory-Traffic Fusion、Epilogue Fusion 与 Algorithmic Fusion。
2. 计算 Elementwise 算子链融合前后的理论 Global Memory Traffic。
1. 建立包含 Launch、Memory、Compute 和同步开销的性能模型。
2. 判断一组算子是否适合 Vertical Fusion 或 Horizontal Fusion。
1. 识别 Fusion 导致的 Register Pressure、Occupancy 下降、Spill 和并行度损失。
2. 用纯 Python 实验观察中间列表带来的时间和峰值内存成本。
1. 编写并运行 PyTorch CUDA Extension，对比三个 Kernel 与一个融合 Kernel。
2. 用 Nsight Systems 验证 Kernel 数，用 Nsight Compute 验证 DRAM Traffic 和 Occupancy。
1. 建立高性能自定义算子的正确性、数值、性能和可维护性验收流程。

## 前置知识

- 理解 CUDA Grid、Block、Thread、Warp、Stream 和异步 Launch。
- 理解 Global Memory、Register、Shared Memory 与 Memory Coalescing。
- 理解 Effective Bandwidth、Arithmetic Intensity、Roofline 和 Amdahl 定律。
- 会使用 CUDA Event、Nsight Systems 与 Nsight Compute。
- Level 0 仅需 Python；Level 1 需要 CUDA Toolkit、PyTorch 和 C++ 编译器。

## 核心直觉：中间 Tensor 是一张昂贵的“交接单”

考虑下面的表达式：

```
tmp1 = x * y
tmp2 = tmp1 + bias
out = relu(tmp2)
```

从数学上看，它只是：

$$
out_i=\max(x_i y_i+bias,0)
$$

但在 Eager 执行中，它可能形成三个独立 Kernel：

```
Kernel 1: 读 x、y → 写 tmp1
Kernel 2: 读 tmp1 → 写 tmp2
Kernel 3: 读 tmp2 → 写 out
```

`tmp1` 和 `tmp2` 不是最终结果，却完整走了两次 Global Memory Store 和两次后续 Load。融合后：

```
Fused Kernel: 读 x、y → 在 Register 中乘、加、ReLU → 写 out
```

每个线程加载自己的 `x_i` 、 `y_i` ，中间值停留在 Register，最终只写一次 `out_i` 。这就是最典型的 Producer-Consumer Fusion。

## Fusion 的两大收益

### 收益一：减少 Kernel Launch

设一次 Launch 的有效开销为 $t_{launch}$ ，原来有 K 个 Kernel，融合后有 F 个：

$$
\Delta T_{launch}\approx(K-F)t_{launch}
$$

真实 Launch Cost 受 CPU、Driver、Python/Framework、CUDA Graph、Profiler 和系统负载影响，不能把某个固定微秒数写成通用常量。

对大 GEMM，Launch Cost 可能几乎可以忽略；对数百个只有几微秒的小 Pointwise Kernel，它可能成为主瓶颈。

### 收益二：减少 Global Memory Round Trip

以 FP32 的 `relu(x * y + bias)` 为例，不考虑 Cache，按有用字节估算：

| 阶段 | 读取 | 写入 | 每元素流量 |
| --- | --- | --- | --- |
| Multiply | `x,y` ：8 B | `tmp1` ：4 B | 12 B |
| Bias Add | `tmp1` ：4 B | `tmp2` ：4 B | 8 B |
| ReLU | `tmp2` ：4 B | `out` ：4 B | 8 B |
| 未融合合计 |  |  | 28 B |
| 融合 Kernel | `x,y` ：8 B | `out` ：4 B | 12 B |

理想流量缩减比：

$$
R_{traffic}=\frac{28}{12}\approx2.33
$$

这不是“必然加速 2.33 倍”。Cache 可能吸收部分中间流量；Kernel 还可能受 Launch、Compute、Register、Occupancy 和其他系统开销影响。

## 综合性能模型

一个便于推理的近似模型：

$$
T\approx N_{launch}t_{launch}+\max\left(\frac{Bytes}{BW_{effective}},\frac{FLOPs}{P_{effective}}\right)+T_{sync}+T_{other}
$$

Fusion 主要改变三个量：

$N_{launch}$ 减少；

Bytes 减少；

$P_{effective}$ 可能上升，也可能因为 Register Pressure、Occupancy 或指令依赖而下降。

因此 Fusion 的正确判断不是“能不能合”，而是：

## Fusion 的主要类型

### Vertical Fusion：上下游融合

把 Producer 与 Consumer 合并：

```
Bias → Activation → Dropout
Residual Add → LayerNorm 前处理
Dequantize → GEMM 输入转换
GEMM → Bias → Activation Epilogue
```

收益来自消除中间 Tensor，最适合一对一或规则广播的 Pointwise 链。

### Horizontal Fusion：同类小任务融合

若很多小 Tensor 分别执行相同更新，可以把它们打包为一个 Multi-Tensor Kernel：

```
参数 1 的 Adam 更新 ┐
参数 2 的 Adam 更新 ├─→ 一个 Multi-Tensor Apply Kernel
参数 3 的 Adam 更新 ┘
```

它重点减少 Launch 数，也能改善小 Tensor 的并行度。Fused Optimizer 常采用这种思路。

### Epilogue Fusion

矩阵乘核心计算得到 Accumulator 后，不立即把它写回 Global Memory，而是在 Epilogue 中完成：

$$
D=Activation(\alpha AB+\beta C+bias)
$$

Bias、Scale、Activation、Residual 或 Quantization 可以直接消费 Accumulator。cuBLASLt、CUTLASS 等库提供多种 Epilogue 能力；优先复用成熟库，不要为了“自定义”而重写 GEMM Mainloop。

### Reduction Fusion

例如将统计量计算与归一化、Loss 与 Reduction、Softmax 的若干阶段融合。它比 Pointwise Fusion 难，因为需要：

- Warp/Block Reduction；
- 跨线程同步；
- 数值稳定算法；
- 可能的多阶段全局归约。

无法在单个 Block 内完成的 Global Reduction，未必适合强行塞进一个普通 Kernel。

### Algorithmic Fusion

FlashAttention 不只是把几个表达式写到同一 Kernel，而是通过 Tiling 和 Online Softmax 避免物化完整 Attention Matrix。它改变了 I/O 复杂度与数据流，属于比 Pointwise Fusion 更深的算法级融合。

### Persistent Kernel 与 Megakernel

Persistent Kernel 让 Thread Block 长时间驻留，循环从 Work Queue 取任务；Megakernel 则把更多执行阶段放在一个大型 Kernel 中。它们可以减少 Host Launch 和全局同步，但调度、负载均衡、Register/Shared Memory、死锁和可维护性都更复杂。

## 什么样的算子最适合融合

### 适合

- 连续的 Elementwise 操作；
- 中间 Tensor 只被下游消费一次；
- 相同或易广播的 Shape；
- 每步 Arithmetic Intensity 很低；
- Kernel 很短，Launch Overhead 明显；
- GEMM/Conv 后的 Bias、Activation、Scale 等 Epilogue；
- 多个结构相同的小 Tensor 更新。

### 谨慎

- 中间结果会被多个 Consumer 复用；
- 上下游拥有完全不同的最优 Grid/Tile；
- 一个阶段是大 Reduction 或需要全局同步；
- 融合后每线程寄存器激增；
- 分支路径复杂，Warp Divergence 加重；
- 融合会破坏原本可以并行运行的独立 Kernel；
- 为了消除存储而大量重复计算；
- 不同精度和 Reduction Order 造成数值变化；
- 动态 Shape 让编译缓存持续失效。

## 为什么“融合得越多”不一定越快

### Register Pressure

多个阶段的 Live Value 同时存在，编译器需要更多 Register。若超过物理限制：

- Active Block/Warp 数下降；
- Occupancy 降低；
- 发生 Local Memory Spill；
- Spill 又引入新的显存流量。

原来省掉的中间 Tensor 流量，可能被 Spill 抵消。

### 最优并行策略冲突

Pointwise Kernel 可能希望大量独立 Thread；Reduction 需要协同；GEMM 需要 Tile 和 Tensor Core Pipeline。强行融合会让其中至少一个阶段使用不合适的映射。

### 失去并发

两个互不依赖的小 Kernel 原本可以在不同 Stream 并发。把它们合成一个资源占用很高的 Kernel，不一定更快。

### Code Size 与编译成本

大量 Shape、Dtype、Layout 和 Feature 组合会产生变体爆炸：

$$
Variants=Shapes\times Dtypes\times Layouts\times Architectures\times Features
$$

JIT 编译、Binary Size、Cache、测试矩阵和回归维护都会变贵。

## 高性能算子开发流程

### 1\. 写清算子契约

- 输入/输出 Shape、Dtype、Device 和 Layout；
- Contiguous、Stride、Alignment 要求；
- Broadcast 规则；
- In-place、Alias 与 Mutation 语义；
- Forward、Backward 与 Higher-order Gradient；
- 确定性、随机数和数值容差；
- 支持的架构与回退路径。

### 2\. 建立可信 Reference

Reference 优先由现有 PyTorch/NumPy 操作组合而成，追求易读和正确，而不是快。测试随机值、零、极值、NaN/Inf、奇数长度、非对齐 Tail 和空 Tensor。

### 3\. 建立性能模型

先数清：

- FLOPs；
- 最少输入/输出 Byte；
- 未融合中间 Byte；
- Kernel 数；
- 预期是 Launch、Memory 还是 Compute Bound。

### 4\. 写最小正确 Kernel

先实现单 Dtype、Contiguous、规则 Shape，确保边界检查和正确性，再逐步加入 Vectorized Load、Mixed Precision、不同 Layout、Backward 和 Auto-tuning。

### 5\. Profile 而不是猜

- Nsight Systems：Kernel 数、Launch Gap、NVTX Range；
- Nsight Compute：DRAM Byte、Memory Throughput、Register、Occupancy、Stall、Source/SASS；
- 无 Profiler Benchmark：最终端到端速度与 Tail。

### 6\. 集成 Framework

教学实验可用 `torch.utils.cpp_extension.load_inline` 快速 JIT。生产自定义算子还要正确注册 Operator Schema、Fake/Meta Kernel、Autograd 和编译器可见性，不能只暴露一个随意的 PyBind 函数。

### 7\. 建立回归矩阵

正确性与性能都要覆盖：

- 多 Shape、多 Batch、多 Sequence Length；
- FP32、FP16、BF16 或目标低精度；
- Ampere、Ada、Hopper、Blackwell 的实际支持范围；
- Cold Compile 与 Warm Execution；
- 单 Kernel Microbenchmark 与端到端模型。

## Level 0：CPU 单遍融合实验

这个实验不需要 GPU。基线对 Python List 做三次遍历并构造两个中间列表；融合版只做一次遍历并直接写最终输出。

将下面代码保存为 `fusion_cpu_lab.py` ：

```
#!/usr/bin/env python3
import argparse
import gc
import json
import math
import platform
import statistics
import time
import tracemalloc

def baseline(x, y, bias):
    tmp1 = [a * b for a, b in zip(x, y)]
    tmp2 = [v + bias for v in tmp1]
    out = [v if v > 0.0 else 0.0 for v in tmp2]
    return out

def fused(x, y, bias):
    return [
        (value if value > 0.0 else 0.0)
        for a, b in zip(x, y)
        for value in (a * b + bias,)
    ]

def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - position) + ordered[hi] * (position - lo)

def benchmark(fn, x, y, bias, warmup, repeat):
    for _ in range(warmup):
        fn(x, y, bias)
    gc.collect()

    samples = []
    checksum = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn(x, y, bias)
        samples.append(time.perf_counter() - t0)
        checksum = sum(out[:: max(1, len(out) // 1000)])
    mean = statistics.fmean(samples)
    return {
        "median_ms": statistics.median(samples) * 1e3,
        "p95_ms": percentile(samples, 0.95) * 1e3,
        "cv_percent": statistics.pstdev(samples) / mean * 100.0,
        "checksum_sample": checksum,
    }

def measure_peak(fn, x, y, bias):
    gc.collect()
    tracemalloc.start()
    out = fn(x, y, bias)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {"peak_mib": peak / 2**20, "output_length": len(out)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300_000)
    parser.add_argument("--bias", type=float, default=-0.25)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=7)
    args = parser.parse_args()
    if args.n <= 0 or args.warmup < 0 or args.repeat < 3:
        raise SystemExit("要求 n > 0、warmup >= 0、repeat >= 3")

    x = [((i % 1009) - 504) / 503.0 for i in range(args.n)]
    y = [((i % 997) - 498) / 499.0 for i in range(args.n)]

    ref = baseline(x, y, args.bias)
    opt = fused(x, y, args.bias)
    max_abs_error = max(abs(a - b) for a, b in zip(ref, opt))
    if max_abs_error != 0.0:
        raise AssertionError(f"结果不一致：max_abs_error={max_abs_error}")
    del ref, opt

    base_stats = benchmark(
        baseline, x, y, args.bias, args.warmup, args.repeat
    )
    fused_stats = benchmark(
        fused, x, y, args.bias, args.warmup, args.repeat
    )
    base_mem = measure_peak(baseline, x, y, args.bias)
    fused_mem = measure_peak(fused, x, y, args.bias)

    report = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "config": vars(args),
        "correctness": {"exact": True, "max_abs_error": max_abs_error},
        "baseline": {"timing": base_stats, "memory": base_mem},
        "fused": {"timing": fused_stats, "memory": fused_mem},
        "speedup": base_stats["median_ms"] / fused_stats["median_ms"],
        "peak_memory_reduction": base_mem["peak_mib"] / fused_mem["peak_mib"],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
```

### 运行命令

```
python3 fusion_cpu_lab.py --n 300000 --warmup 2 --repeat 7
```

### 预期现象

- `correctness.exact` 为 `true` ；
- 融合版通常更快；
- 融合版 `peak_mib` 通常明显更低；
- 具体速度随 Python 版本、CPU、GC 和后台负载变化；
- `tracemalloc` 只用于比较 Python Allocation，不代表 CUDA 显存。

### 结果分析

基线需要同时持有 `tmp1` 、 `tmp2` 和 `out` ；融合版主要只持有 `out` 。这与 GPU Fusion 的数据流直觉一致，但 CPU Python 实验不是 GPU Kernel 的等价模拟：它不包含 Warp、Register、CUDA Launch、Coalescing 或显存带宽。

## Level 1：三个 CUDA Kernel 融合为一个

### 环境检查

```
python3 -m venv .venv-fusion
source .venv-fusion/bin/activate
python -m pip install --upgrade pip ninja
# 按 PyTorch 官方安装页选择与 Driver 匹配的 CUDA 版本。
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
nvcc --version
c++ --version
```

PyTorch Wheel 自带的 CUDA Runtime 与系统 `nvcc` 是两个概念。JIT 编译 Extension 时，系统 CUDA Toolkit、编译器和 PyTorch 构建版本必须兼容。

将下面代码保存为 `fusion_cuda_extension.py` ：

```
#!/usr/bin/env python3
import argparse
import json
import os
import time

import torch
from torch.utils.cpp_extension import load_inline

CPP_SOURCE = r"""
torch::Tensor baseline_cuda(torch::Tensor x, torch::Tensor y, double bias);
torch::Tensor fused_cuda(torch::Tensor x, torch::Tensor y, double bias);
"""

CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void multiply_kernel(
    const float* x, const float* y, float* out, int64_t n) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] * y[i];
}

__global__ void bias_kernel(
    const float* x, float bias, float* out, int64_t n) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] + bias;
}

__global__ void relu_kernel(const float* x, float* out, int64_t n) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

__global__ void fused_kernel(
    const float* x, const float* y, float bias, float* out, int64_t n) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        float value = x[i] * y[i] + bias;
        out[i] = value > 0.0f ? value : 0.0f;
    }
}

void check_inputs(const torch::Tensor& x, const torch::Tensor& y) {
    TORCH_CHECK(x.is_cuda() && y.is_cuda(), "x and y must be CUDA tensors");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(y.scalar_type() == torch::kFloat32, "y must be float32");
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous(), "tensors must be contiguous");
    TORCH_CHECK(x.sizes() == y.sizes(), "x and y must have the same shape");
    TORCH_CHECK(x.get_device() == y.get_device(), "x and y must be on one GPU");
}

torch::Tensor baseline_cuda(torch::Tensor x, torch::Tensor y, double bias) {
    check_inputs(x, y);
    auto tmp1 = torch::empty_like(x);
    auto tmp2 = torch::empty_like(x);
    auto out = torch::empty_like(x);
    const int threads = 256;
    const int64_t n = x.numel();
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    multiply_kernel<<<blocks, threads, 0, stream>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), tmp1.data_ptr<float>(), n);
    bias_kernel<<<blocks, threads, 0, stream>>>(
        tmp1.data_ptr<float>(), static_cast<float>(bias), tmp2.data_ptr<float>(), n);
    relu_kernel<<<blocks, threads, 0, stream>>>(
        tmp2.data_ptr<float>(), out.data_ptr<float>(), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor fused_cuda(torch::Tensor x, torch::Tensor y, double bias) {
    check_inputs(x, y);
    auto out = torch::empty_like(x);
    const int threads = 256;
    const int64_t n = x.numel();
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    fused_kernel<<<blocks, threads, 0, stream>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), static_cast<float>(bias),
        out.data_ptr<float>(), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""

def load_extension(verbose):
    return load_inline(
        name="fusion_lab_ext_v1",
        cpp_sources=CPP_SOURCE,
        cuda_sources=CUDA_SOURCE,
        functions=["baseline_cuda", "fused_cuda"],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=verbose,
    )

def benchmark(fn, x, y, bias, warmup, repeat):
    for _ in range(warmup):
        out = fn(x, y, bias)
        del out
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record()
    for _ in range(repeat):
        out = fn(x, y, bias)
        del out
    end.record()
    end.synchronize()
    return {
        "cuda_event_ms": start.elapsed_time(end) / repeat,
        "host_wall_ms": (time.perf_counter() - wall_start) * 1e3 / repeat,
    }

def peak_allocation(fn, x, y, bias):
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = fn(x, y, bias)
    torch.cuda.synchronize()
    peak_delta = torch.cuda.max_memory_allocated() - base
    del out
    return peak_delta

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=16_777_216)
    parser.add_argument("--bias", type=float, default=-0.25)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--verbose-build", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("未检测到 CUDA GPU；请运行 Level 0")
    if args.n <= 0 or args.warmup < 0 or args.repeat <= 0:
        raise SystemExit("参数无效")

    ext = load_extension(args.verbose_build)
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    x = torch.randn(args.n, device=device, dtype=torch.float32)
    y = torch.randn_like(x)

    reference = torch.relu(x * y + args.bias)
    base_out = ext.baseline_cuda(x, y, args.bias)
    fused_out = ext.fused_cuda(x, y, args.bias)
    base_error = (reference - base_out).abs().max().item()
    fused_error = (reference - fused_out).abs().max().item()
    if not torch.allclose(reference, base_out, rtol=1e-5, atol=1e-6):
        raise AssertionError(f"baseline mismatch: {base_error}")
    if not torch.allclose(reference, fused_out, rtol=1e-5, atol=1e-6):
        raise AssertionError(f"fused mismatch: {fused_error}")
    del reference, base_out, fused_out
    torch.cuda.synchronize()

    baseline_stats = benchmark(
        ext.baseline_cuda, x, y, args.bias, args.warmup, args.repeat
    )
    fused_stats = benchmark(
        ext.fused_cuda, x, y, args.bias, args.warmup, args.repeat
    )
    baseline_peak = peak_allocation(ext.baseline_cuda, x, y, args.bias)
    fused_peak = peak_allocation(ext.fused_cuda, x, y, args.bias)

    bytes_unfused = args.n * 28
    bytes_fused = args.n * 12
    report = {
        "environment": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "compute_capability": [props.major, props.minor],
            "total_memory_gib": props.total_memory / 2**30,
        },
        "config": vars(args),
        "correctness": {
            "baseline_max_abs_error": base_error,
            "fused_max_abs_error": fused_error,
        },
        "model": {
            "unfused_useful_bytes": bytes_unfused,
            "fused_useful_bytes": bytes_fused,
            "theoretical_traffic_reduction": bytes_unfused / bytes_fused,
        },
        "baseline": {
            "timing": baseline_stats,
            "peak_incremental_allocation_mib": baseline_peak / 2**20,
        },
        "fused": {
            "timing": fused_stats,
            "peak_incremental_allocation_mib": fused_peak / 2**20,
        },
        "speedup_by_cuda_event": (
            baseline_stats["cuda_event_ms"] / fused_stats["cuda_event_ms"]
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
```

### 通用运行命令

```
export MAX_JOBS=4
python fusion_cuda_extension.py --n 16777216 --warmup 10 --repeat 50
```

第一次运行需要 JIT 编译，不能把编译时间混入 Steady-State Kernel Benchmark。PyTorch 会缓存 Extension；修改源码后名称不变时，应留意缓存是否被正确重建。

双 RTX 3080 20GB 可这样运行：

```
CUDA_VISIBLE_DEVICES=0 python fusion_cuda_extension.py --n 16777216
CUDA_VISIBLE_DEVICES=1 python fusion_cuda_extension.py --n 16777216
```

这不是双 GPU Kernel；每次进程只在一张卡上建立基线，适合比较两张卡的温度、频率和环境稳定性。

### 预期现象

- 自动输出 GPU 型号、Compute Capability、显存和 PyTorch/CUDA 版本；
- Baseline 和 Fused 都通过正确性检查；
- Baseline 在时间线中有 Multiply、Bias、ReLU 三个 Kernel；
- Fused 只有一个 Kernel；
- Fused 的增量峰值显存更低；
- 大数组下通常看到 Fused Effective Bandwidth 和耗时更好；
- 实测 Speedup 不会严格等于理论流量缩减比 2.33。

### 结果分析

Baseline 的两个中间 Tensor 各占：

$$
Memory_{tmp}=2N\times4\ bytes
$$

当 $N=2^{24}$ 时，两个中间 FP32 Tensor 合计约 128 MiB。PyTorch Caching Allocator 可能保留已释放的 Reserved Memory，所以应比较 `memory_allocated` /Peak，而不是只看 `nvidia-smi` 的进程显存。

融合 Kernel 的算术强度仍很低：约 2 个主要 FLOP、12 Byte 有用流量，通常仍是 Memory Bound。它的价值是更接近一次必要的读写下限。

## 用 Profiler 证明 Fusion 机制

### Nsight Systems：证明 Kernel 数减少

```
mkdir -p reports
nsys profile \
  --trace=cuda,osrt \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  -o reports/fusion_timeline \
  python fusion_cuda_extension.py --n 4194304 --warmup 2 --repeat 5
```

```
nsys stats reports/fusion_timeline.nsys-rep
```

重点比较 `multiply_kernel` 、 `bias_kernel` 、 `relu_kernel` 和 `fused_kernel` 的调用次数、总时间、平均时间，以及 Kernel 之间是否存在 CPU Launch Gap。

### Nsight Compute：证明流量与资源变化

先用 Kernel Name 过滤单个融合 Kernel：

```
ncu \
  --kernel-name regex:.*fused_kernel.* \
  --launch-count 1 \
  --set full \
  --force-overwrite \
  -o reports/fused_kernel \
  python fusion_cuda_extension.py --n 4194304 --warmup 1 --repeat 1
```

再分别采集三个 Baseline Kernel，比较：

- DRAM/L2 Read/Write Byte；
- Memory Throughput；
- Registers Per Thread；
- Achieved Occupancy；
- Eligible Warps 与主要 Stall；
- Source/SASS 中的 Load、Store 和算术指令。

Profiler 会带来 Replay 和显著开销；最终 Speedup 必须用无 Profiler 的 CUDA Event 结果确认。

## Level 2：架构专项方向与边界

Level 1 的基础融合 Kernel 使用普通 FP32 CUDA Core，各代 NVIDIA GPU 均可学习。Level 2 关注的是如何把 Fusion 放入不同架构的高性能数据流。

| 架构 | 常见 GPU | 可重点研究 | 不可伪装验证的边界 |
| --- | --- | --- | --- |
| Ampere | RTX 3080/3090、A100 | `sm_86/sm_80` 、Vector Load、 `cp.async` 、GEMM Epilogue | RTX 3080/3090 不能代表 A100 HBM、NVLink 或 MIG |
| Ada Lovelace | RTX 4090、L40/L40S | `sm_89` 、更大 L2、推理 Epilogue、FP8 软件栈 | RTX 4090 是 Ada，不是 Blackwell，且无 NVLink |
| Hopper | H100/H200 | TMA、WGMMA、Warp Specialization、FP8、Persistent Kernel | 必须在 Hopper 验证 TMA/WGMMA 与 Transformer Engine |
| Blackwell | RTX 5090、B100/B200/GB200 | 新 Tensor Core/FP4、CUTLASS 新 Epilogue、Cluster/Fabric 数据流 | RTX 5090 不能代表 B200/GB200 的 HBM、NVLink 与机架级互联 |

### Ampere 双 RTX 3080 的可执行方向

- 编译目标自动检测到 Compute Capability 8.6；
- 对比标量与 `float4` Vectorized Fused Kernel；
- 用 Nsight Compute 检查 Register 和 DRAM Traffic；
- 把 Fused Elementwise 放到不同 Stream/Shape 中测试；
- 可研究 `cp.async` 的 Global-to-Shared Pipeline，但本课的纯 Pointwise Kernel不需要 Shared Memory，硬加 `cp.async` 反而可能更慢。

### Hopper/Blackwell 可选方向

- 将 Bias/Activation/Scale 放入 Tensor Core GEMM Epilogue；
- 使用 CUTLASS 的 Epilogue Fusion 能力组合 Broadcast、Activation、Auxiliary Output；
- 在真实 H100/H200 上研究 TMA 与 Warp-Specialized Producer-Consumer；
- 在真实 B100/B200/GB200 或相应开发平台验证新的 FP4/Block Scaling 路径。

没有对应硬件时，可以学习数据流和接口，但不能把 Ampere 上的普通 CUDA Kernel 当成 TMA、WGMMA 或 Blackwell FP4 的等价实验。

## PyTorch 自动融合与自定义算子的边界

如果计算能由标准 PyTorch Operator 表达，优先保留 Python 组合并让编译器尝试自动 Fusion。自定义算子适合：

- 调用框架不理解的 CUDA/C++ Library；
- 需要特殊数据布局或算法；
- 自动编译无法生成满意代码；
- 需要稳定封装已有高性能 Kernel。

生产集成不能止步于 `load_inline` 。一个可组合的 PyTorch Custom Operator 还需要考虑：

- Operator Schema；
- Mutation/Aliasing Contract；
- Fake/Meta Kernel；
- Autograd Formula；
- `torch.compile` 、 `torch.export` 、 `vmap` 等子系统；
- Wheel/ABI/Architecture Packaging；
- CPU 或 Unsupported Shape 回退。

若操作本身可由内置 PyTorch 算子表达，官方建议通常先写普通 Python 函数，而不是无必要地注册 Opaque Custom Op。

## 数值与 Autograd

### 融合可能改变运算顺序

编译器可能把乘加变成 FMA：

$$
round(round(xy)+b)\neq round(xy+b)
$$

两者都可能符合浮点语义，却有细微误差。 `--use_fast_math` 还可能改变某些运算的精度和特殊值行为。测试应使用合理 `rtol/atol` ，同时单独覆盖 NaN、Inf、Subnormal 和边界值。

### Backward 也需要性能设计

Forward 很快但 Backward 退化，训练总体未必提升。自定义训练算子必须：

- 推导并验证梯度；
- 使用 `gradcheck` 时采用适合的 Double Precision Reference；
- 检查 Saved Tensor 显存；
- 决定是保存中间值还是 Backward 重算；
- Profile 整个 Forward+Backward Step。

本课实验是 Forward-only 教学算子，不应直接放入训练图中宣称支持 Autograd。

## 常见错误与排查

### 1\. Fused Kernel 反而更慢

检查：

- 问题规模是否太小或太大；
- Register 数和 Occupancy 是否恶化；
- 是否发生 Local Memory Spill；
- 分支是否加剧 Warp Divergence；
- Baseline 中间数据是否被 L2 Cache 命中；
- 上下游是否有不匹配的线程映射；
- Benchmark 是否包含 JIT Compile。

### 2\. load\_inline 找不到 CUDA

检查：

```
which nvcc
echo "$CUDA_HOME"
python -c "from torch.utils.cpp_extension import CUDA_HOME; print(CUDA_HOME)"
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

PyTorch 能运行 CUDA，不代表系统一定安装了用于编译 Extension 的 `nvcc` 。

### 3\. 编译器版本不兼容

症状可能是 Unsupported GNU Version、Header Error 或 Link Error。按当前 CUDA Toolkit 和 PyTorch 官方支持矩阵选择 Host Compiler；不要随意用 `-allow-unsupported-compiler` 掩盖生产问题。

### 4\. 编译耗尽 CPU/RAM

Ninja 默认并行可能较高：

```
export MAX_JOBS=2
```

减少并发后重试，并确认磁盘空间充足。

### 5\. 结果出现微小误差

本实验启用了 `--use_fast_math` ，Fusion/FMA 可能改变舍入。检查误差是否在契约容限内；需要严格行为时移除 Fast Math 并复测。

### 6\. nvidia-smi 看不到显存下降

PyTorch Caching Allocator 会保留 Reserved Memory。比较 `torch.cuda.max_memory_allocated()` 、Profiler 和 OOM Headroom，不要只看进程保留显存。

### 7\. Kernel 数少了但端到端没提升

可能该链只占总时间很小，或优化不在 Critical Path。用 Amdahl 定律和完整模型 Step 复测。

### 8\. 只支持 Contiguous FP32

这是本课为了保持代码可读的明确边界。生产实现应加入 Dtype Dispatch、Stride/Layout、Alignment、Tail、Empty Tensor 和非支持输入回退，而不是默默给出错误结果。

### 9\. 自定义算子破坏 torch.compile

裸 PyBind 调用对编译器可能不透明。按 PyTorch Custom Operator 机制注册 Schema、Fake Kernel 与 Autograd；同时测试 Eager 与 Compile 路径。

### 10\. 为追求 Fusion 重写成熟 GEMM

优先检查 cuBLASLt、CUTLASS、cuDNN 或框架现有 Fused Operator。只为一个 Bias/ReLU 重写 Tensor Core GEMM Mainloop，维护成本通常不划算。

## 优化前后对照

| 维度 | 未融合 | 融合后 | 验证工具 |
| --- | --- | --- | --- |
| Kernel 数 | 3 | 1 | Nsight Systems |
| 理论有用流量 | 28 B/元素 | 12 B/元素 | 数据流模型 |
| 中间 Tensor | `tmp1,tmp2` | 无 | Allocator/Profiler |
| Launch Gap | 可能存在两段额外 Gap | 减少 | Nsight Systems |
| Register Live Value | 较少 | 可能增多 | Nsight Compute |
| Occupancy | 各 Kernel 独立 | 可能下降 | Nsight Compute |
| 正确性 | PyTorch Reference | 容差内一致 | `allclose` /专项测试 |
| 最终效果 | 基线 Median/P99 | 同协议复测 | CUDA Event/端到端压测 |

## 面试题与答案

### 1\. Kernel Fusion 为什么能加速？

它减少 Kernel Launch，并让中间值留在 Register/Shared Memory，避免把中间 Tensor 写到 Global Memory再读回。对 Launch-Bound 或 Memory-Bound 的算子链最有效。

### 2\. 为什么理论流量减少 2.33 倍，不一定加速 2.33 倍？

Cache、Launch、Compute、Register、Occupancy、同步和系统其他阶段都会影响实际时间。理论比值只是 Memory Traffic 上限模型。

### 3\. Vertical Fusion 和 Horizontal Fusion 有什么区别？

Vertical Fusion 合并上下游 Producer/Consumer，消除中间 Tensor；Horizontal Fusion 合并多个同类小任务，重点摊薄 Launch 并提高并行度。

### 4\. 什么是 Epilogue Fusion？

在 GEMM/Conv 的 Accumulator 写回 Global Memory 前，直接完成 Bias、Scale、Activation、Residual 或 Quantization，避免额外读写和 Kernel。

### 5\. Fusion 为什么会降低 Occupancy？

融合后 Live Value 和逻辑增加，Register/Shared Memory 占用上升，导致每个 SM 可驻留的 Block/Warp 数下降；严重时还会 Spill。

### 6\. 怎样证明 Fusion 的收益来自减少 Memory Traffic？

先建 Byte 模型，再用 Nsight Systems 证明 Kernel 数，用 Nsight Compute 比较 DRAM/L2 Byte 和吞吐，最后用无 Profiler Benchmark 确认端到端提升。

### 7\. 为什么 Reduction Fusion 比 Pointwise Fusion 难？

Reduction 需要线程协同、同步、稳定求和和可能的跨 Block 全局归约；不同阶段的并行映射不一定兼容。

### 8\. 自定义 CUDA 算子为什么需要 Fake/Meta Kernel？

编译器和 Export 系统需要在不执行真实数据计算时推导 Shape、Dtype、Device 和 Alias 行为。缺少这些契约会影响图捕获和编译组合。

### 9\. 什么情况下不该写自定义 Kernel？

已有 Library/Fused Op 可满足需求、Compiler 能自动生成高质量代码、热点占比太低，或维护成本超过收益时，不应自定义。

### 10\. FlashAttention 只是普通 Kernel Fusion 吗？

不是。它通过 Tiling 与 Online Softmax 改变 Attention 的数据流，避免物化大型中间矩阵，属于 I/O-aware 的算法级融合。

## 课后练习

1. 把 Level 0 改为 `sigmoid(x*y+bias)` ，比较融合前后时间和峰值内存。
2. 根据 FP16 元素宽度重新计算 28 B 与 12 B 模型，并讨论转换和累加精度。
1. 给 CUDA Extension 增加 `float4` Vectorized 路径，处理 Alignment 和 Tail。
2. 去掉 `--use_fast_math` ，比较误差、SASS 和性能。
1. 用 Nsight Systems 截图证明 Baseline 三个 Kernel 与 Fused 一个 Kernel。
2. 用 Nsight Compute 比较四个 Kernel 的 Register、DRAM Byte 和 Occupancy。
1. 将 `threads` 分别改为 128、256、512，建立 Shape×Block Size Benchmark 表。
2. 增加一个被两个 Consumer 复用的中间 Tensor，分析“存一次”与“重算两次”的权衡。
1. 研究 PyTorch Custom Operator 注册流程，为教学 Kernel 补充 Schema、Fake Kernel 和 CPU 回退。
2. 在真实模型中找一段 Pointwise 链，用 Amdahl 定律预测端到端收益，再验证预测。

## Checklist

### 候选选择

- 已用 Profiler 证明算子链是热点或 Launch-Bound。
- 中间 Tensor 主要由单个下游消费。
- 已计算融合前后的 FLOPs、Byte 和 Kernel 数。
- 已用 Amdahl 判断端到端收益上限。
- 已检查现成 Library/Fused Op 和 Compiler 路径。

### 正确性

- 有清晰、可信的 Reference。
- 覆盖 Tail、奇数 Shape、零、极值、NaN/Inf。
- 明确 Dtype、Layout、Contiguous 与 Alignment 契约。
- 明确 Fast Math、FMA 和 Reduction Order 的数值边界。
- 训练算子已验证 Forward、Backward 和 Saved Tensor。

### 性能

- JIT Compile 时间没有混入 Steady-State Benchmark。
- 使用 CUDA Event 与足够的预热/重复。
- Nsight Systems 证明 Kernel 数和 Launch Gap 变化。
- Nsight Compute 证明 DRAM Traffic、Register 和 Occupancy 变化。
- 最终用无 Profiler 的端到端负载复测。

### 工程化

- Custom Operator 的 Schema、Alias、Fake/Meta 和 Autograd 完整。
- 有 Unsupported Shape/Dtype/Architecture 回退。
- 控制编译变体和 Binary/JIT Cache。
- 建立多 Shape、多 GPU 架构的回归测试。
- 收益能够覆盖长期维护成本。

### 硬件边界

- Ampere 正确包含 RTX 3080/3090、A100。
- Ada 正确包含 RTX 4090、L40/L40S。
- Hopper 正确包含 H100/H200。
- Blackwell 正确包含 RTX 5090、B100/B200/GB200。
- 没有把 RTX 4090 写成 Blackwell。
- 没有用 RTX 3080 模拟验证 TMA、WGMMA、FP8 Transformer Engine 或 Blackwell FP4。
- 性能数字只代表实际测试环境，不外推为所有 GPU 的结论。