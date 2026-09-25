---
title: "第14课：Triton Kernel 开发"
source: "https://everythingai.top/learn/ai-systems-performance-engineering-text/lesson-14"
author:
published:
created: 2026-09-26
description: "面向 AI 工程学习者的专业课程学习空间。"
tags:
  - "clippings"
---
**导航**

本课使用 Triton 将算子意图映射为高效 GPU Kernel，掌握开发、调优和正确性验证的完整流程。

## 课程定位

CUDA 把 GPU 暴露为 Thread、Warp、Block 和 Shared Memory；Triton 则把最常见的并行数据块提升为语言中的一等对象。开发者不再逐个编写 Thread 的标量逻辑，而是让每个 Triton Program Instance 处理一块数据，再由编译器把块操作 Lower 到目标 GPU。

这并不意味着 Triton 会自动把任何 Python 写法变快。你仍然需要理解数据布局、访存、Tiling、Mask、Reduction、Register、Shared Memory、Occupancy 和 Roofline。Triton降低的是“表达高性能数据流”的门槛，而不是取消性能工程。

本课从 `program_id + arange + mask` 建立 Triton 的核心心智模型，完成一个可自动调优的融合算子，并讲清 Triton、CUDA C++、Library Kernel 和 `torch.compile` 之间的选型边界。

## 学习目标

完成本课后，你能够：

1. 解释 Triton Program Instance、Launch Grid、Block/Tile 与 CUDA Thread Block 的关系和区别。
2. 使用 `@triton.jit` 、 `tl.program_id` 、 `tl.arange` 、 `tl.load` 、 `tl.store` 和 Mask。
1. 区分 Runtime Argument 与 `tl.constexpr` Meta-parameter。
2. 设计 1D、2D Grid，并完成规则 Pointer Arithmetic。
1. 使用 `triton.Config` 与 `@triton.autotune` 搜索 Block Size 和 `num_warps` 。
2. 计算 Triton Kernel 的有效显存流量、算术强度与吞吐上限。
1. 编写并验证融合的 Multiply + Bias + ReLU Kernel。
2. 使用 Interpreter、断言、Compute Sanitizer、Nsight Systems/Compute 排查问题。
1. 判断何时使用 Triton、CUDA C++、CUTLASS/cuBLAS 或框架内置算子。
2. 正确区分 Ampere、Ada、Hopper 与 Blackwell 的能力边界。

## 前置知识

- 熟悉 Python、PyTorch Tensor 与基本 Benchmark。
- 理解 CUDA Grid、Block、Warp、Stream 和异步执行。
- 理解 Memory Coalescing、Shared Memory、Tensor Core 和 Occupancy。
- 理解 Kernel Fusion、Roofline 与 Nsight 分析方法。
- Level 0 只要求 Python；Level 1 需要 Linux、支持的 NVIDIA GPU、PyTorch 与 Triton。

## 核心直觉：你写的是“一块数据怎么处理”

### CUDA 的常见表达

CUDA Elementwise Kernel 常让一个 Thread 处理一个元素：

```
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) out[i] = x[i] + y[i];
```

### Triton 的表达

Triton 让一个 Program Instance 处理 `BLOCK_SIZE` 个元素：

```
pid = tl.program_id(0)
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offsets < n_elements
x = tl.load(x_ptr + offsets, mask=mask)
y = tl.load(y_ptr + offsets, mask=mask)
tl.store(out_ptr + offsets, x + y, mask=mask)
```

`offsets` 、 `x` 、 `y` 都是一个 Block Tensor，不是普通 Python List，也不是一个 CUDA Thread 的单值。编译器负责把这些块操作映射到 Warp、Vector Instruction、Register、Shared Memory 或 Tensor Core 数据流。

### Program Instance 不等于一个 CUDA Thread

最重要的纠正：

在 1D 向量例子中：

```
N = 1000, BLOCK_SIZE = 256
Grid = ceil(1000 / 256) = 4 个 Program Instance

Program 0 → offsets   0...255
Program 1 → offsets 256...511
Program 2 → offsets 512...767
Program 3 → offsets 768...1023，其中 1000...1023 被 Mask
```

## Triton 编译与执行链路

一个典型调用经历：

```
Python Wrapper
   ↓ 传入 Tensor、Shape、Stride、Meta-parameter
Triton Frontend / @triton.jit
   ↓
Triton IR / GPU-specific Lowering
   ↓
LLVM/PTX 或目标后端代码
   ↓
Driver 加载并缓存编译产物
   ↓
异步 Kernel Launch
```

第一次遇到某个新签名或新 Meta-parameter 组合，可能发生 JIT Compile；后续命中缓存才是稳定执行。Benchmark 必须把编译与 Auto-tuning 时间从 Steady-State Kernel Time 中分离。

## Triton 程序的五个核心组件

### 1\. @triton.jit

它把受 Triton 语言约束的 Python 函数编译为 GPU Kernel。Kernel Body 不是普通 Python Runtime；只能使用 Triton 支持的语义、类型和控制流。

### 2\. Launch Grid

Grid 决定启动多少 Program Instance：

```
grid = lambda meta: (
    triton.cdiv(n_elements, meta["BLOCK_SIZE"]),
)
```

Grid 最多可以有三个 Axis； `tl.program_id(axis)` 读取对应 Axis 上的 Program ID。

### 3\. Block Tensor

`tl.arange(0, BLOCK_SIZE)` 构造一组连续 Offset。通过 `[:, None]` 和 `[None, :]` 可以构造 2D Pointer Block：

```
rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
ptrs = base + rows[:, None] * stride_m + cols[None, :] * stride_n
```

### 4\. Masked Load/Store

当 Shape 不是 Block Size 的整倍数时：

```
mask = offsets < n_elements
value = tl.load(ptr + offsets, mask=mask, other=0.0)
tl.store(out + offsets, value, mask=mask)
```

Mask 是正确处理 Tail 的基础。对于 Reduction， `other` 必须选择不会破坏运算的单位元，例如 Sum 用 0，Max 常用负无穷。

### 5\. Meta-parameter

```
BLOCK_SIZE: tl.constexpr
```

`tl.constexpr` 在编译期已知，可决定 Block Shape、静态分支和 Unroll。普通参数如 `n_elements` 通常在运行期传入。

不同 `BLOCK_SIZE` 、 `num_warps` 、 `num_stages` 会形成不同编译变体。它们能带来性能，也会增加编译和缓存成本。

## 从 1D 到 2D：Pointer Arithmetic

若矩阵 X 的行、列 Stride 为 $s_m$,$s_n$ ：

$$
address(X_{i,j})=base+i\times s_m+j\times s_n
$$

Triton 处理一个二维 Tile：

```
offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
ptrs = x_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
tile = tl.load(ptrs, mask=mask, other=0.0)
```

这里的 Pointer Block Shape 是 `[BLOCK_M, BLOCK_N]` 。Layout、Stride 与 Program Order 会共同决定 Global Memory Coalescing 和 L2 Reuse。

## Block Size、Warp 与 Pipeline

### BLOCK\_SIZE

更大的 Block 可以：

- 摊薄 Program 管理开销；
- 增加每个 Program 的工作量；
- 给 Reduction 提供更大的范围。

但也会：

- 增加 Register Live Value；
- 降低 Occupancy；
- 增加无效 Tail Lane；
- 让单个 Program 过重。

### num\_warps

`num_warps=4` 表示一个 Program Instance 通常由 4 个 Warp，即 128 个线程协同。更复杂的 Tile/Reduction 可能受益于 8 Warp；简单 Pointwise Kernel 并非 Warp 越多越好。

### num\_stages

它主要控制循环 Software Pipelining 的 Stage 数，对矩阵乘这类分块加载/计算循环更重要。更多 Stage 可以增加重叠，也会使用更多片上资源。

### num\_ctas

当前 Triton `Config` 可表达 Cluster 中的 CTA 数，但该能力要求相应硬件与后端支持，官方接口标注为 SM90+。RTX 3080 不能通过设置 `num_ctas` 模拟 Hopper Thread Block Cluster。

## Auto-tuning：让配置适配真实 Shape 与 GPU

```
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def kernel(..., n_elements, BLOCK_SIZE: tl.constexpr):
    ...
```

当 Tuning Key 改变，Auto-tuner 会试运行多个配置并选择更快者。需要理解三个风险：

1. \*\*调优成本\*\*：新 Shape 太多会反复 Benchmark 和编译；
2. \*\*副作用\*\*：Auto-tuner 会多次执行 Kernel，In-place/Atomic 更新可能被重复应用；
1. \*\*过拟合\*\*：在空闲机器上选出的配置，不一定代表生产多租户环境。

对于会修改输入的 Kernel，应使用 `reset_to_zero` 、 `restore_value` 或 Hook 保证每个候选配置看到等价状态。可以设置：

```
TRITON_PRINT_AUTOTUNING=1 python your_kernel.py
```

查看调优完成后的最佳配置和耗时。不要把 Auto-tuning 时间算入线上请求延迟。

## 性能模型

### Program 数与 Tail 效率

对 1D 数据：

$$
N_{program}=\left\lceil\frac{N}{B}\right\rceil
$$

$$
\eta_{tail}=\frac{N}{N_{program}\times B}
$$

当 N 很小却使用巨大 Block 时，Tail Lane 大量被 Mask；但 Block 太小也会增加 Program 数和调度开销。

### 融合 Pointwise 的显存模型

`out = relu(x * y + bias)` 的 FP32 融合 Kernel：

$$
Bytes_{useful}=N\times(2\ reads+1\ write)\times4=12N
$$

约 2 个主要 FLOP，算术强度：

$$
AI\approx\frac{2}{12}=0.167\ FLOP/Byte
$$

它通常是 Memory Bound。优化重点是连续访问、一次读写、足够并行度和合理 Block，而不是增加复杂算术技巧。

### 有效带宽

$$
BW_{effective}=\frac{12N}{T}
$$

要与相同 GPU、相近数据规模和访问模式下的稳定测量比较，而不是把结果直接当作显存规格表利用率。

## Level 0：用纯 Python 模拟 Triton Grid 与 Mask

主线 Triton 官方后端当前仍以 GPU 为目标，CPU 后端处于开发中。Level 0 不伪装成真实 Triton 性能实验，只模拟 Program Instance、Block 和 Tail Mask 的语义。

将下面代码保存为 `triton_grid_sim.py` ：

```
#!/usr/bin/env python3
import argparse
import json
import math

def cdiv(x, y):
    return (x + y - 1) // y

def simulated_add(x, y, block_size):
    if len(x) != len(y):
        raise ValueError("x and y length mismatch")
    n = len(x)
    grid = cdiv(n, block_size)
    out = [None] * n
    records = []

    for pid in range(grid):
        block_start = pid * block_size
        offsets = [block_start + lane for lane in range(block_size)]
        mask = [offset < n for offset in offsets]
        active_offsets = []
        for offset, active in zip(offsets, mask):
            if active:
                out[offset] = x[offset] + y[offset]
                active_offsets.append(offset)
        records.append({
            "program_id": pid,
            "block_start": block_start,
            "active_count": sum(mask),
            "masked_count": block_size - sum(mask),
            "active_range": (
                [active_offsets[0], active_offsets[-1]]
                if active_offsets else None
            ),
        })
    return out, records

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=37)
    parser.add_argument("--block-size", type=int, default=16)
    args = parser.parse_args()
    if args.n <= 0 or args.block_size <= 0:
        raise SystemExit("n and block-size must be positive")

    x = [float(i) for i in range(args.n)]
    y = [float(2 * i + 1) for i in range(args.n)]
    out, records = simulated_add(x, y, args.block_size)
    reference = [a + b for a, b in zip(x, y)]
    if out != reference:
        raise AssertionError("simulation result mismatch")

    grid = cdiv(args.n, args.block_size)
    total_lanes = grid * args.block_size
    report = {
        "config": vars(args),
        "grid_size": grid,
        "total_logical_lanes": total_lanes,
        "active_lanes": args.n,
        "masked_lanes": total_lanes - args.n,
        "tail_efficiency": args.n / total_lanes,
        "programs": records,
        "correctness": True,
        "output_head": out[: min(8, len(out))],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
```

### 运行命令

```
python3 triton_grid_sim.py --n 37 --block-size 16
python3 triton_grid_sim.py --n 1000 --block-size 256
```

### 预期现象

- `n=37, block=16` 时启动 3 个 Program；
- 最后一个 Program 只有 5 个 Active Lane，11 个 Lane 被 Mask；
- `correctness` 为 `true` ；
- 修改 Block Size 会改变 Grid 数与 Tail Efficiency。

### 结果分析

模拟器帮助理解 Coverage 和边界，但不包含 Warp Mapping、访存事务、编译器向量化、Register、Shared Memory 或真实并行执行。因此它只能作为语义回退实验，不能推导 GPU 性能。

## Level 1：通用 NVIDIA GPU Triton 融合算子

### 支持与安装

当前 Triton 官方兼容说明以 Linux 为支持平台，NVIDIA 主线支持 Compute Capability 8.0+。因此：

- RTX 3080/3090（8.6）与 A100（8.0）满足基础要求；
- RTX 4090/L40/L40S、H100/H200 以及受当前发行版支持的 Blackwell GPU 可运行相应后端；
- V100（7.0）不属于当前主线声明的 NVIDIA 支持范围，应改用 CUDA C++、框架算子或与旧环境匹配的方案。

创建隔离环境：

```
python3 -m venv .venv-triton
source .venv-triton/bin/activate
python -m pip install --upgrade pip
```

优先按 PyTorch 官方安装选择器安装匹配的 PyTorch，再检查其中的 Triton：

```
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -c "import triton; print(triton.__version__)"
```

若环境没有 Triton，官方稳定发行版可以：

```
python -m pip install triton
```

不要在已有 `torch.compile` 环境里盲目升级单独的 Triton，PyTorch 与 Triton 组合可能存在版本约束。基本 Triton JIT 不要求系统安装 `nvcc` ，这与上一课的 CUDA C++ Extension 不同。

### 完整代码

将下面代码保存为 `triton_fusion_lab.py` ：

```
#!/usr/bin/env python3
import argparse
import json
import time

import torch
import triton
import triton.language as tl

AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
]

@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["n_elements"])
@triton.jit
def fused_mul_bias_relu_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    bias,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    value = x * y + bias
    output = tl.where(value > 0.0, value, 0.0)
    tl.store(out_ptr + offsets, output, mask=mask)

def triton_fused(x, y, bias):
    if not x.is_cuda or not y.is_cuda:
        raise ValueError("x and y must be CUDA tensors")
    if x.dtype != torch.float32 or y.dtype != torch.float32:
        raise ValueError("this lab supports float32 only")
    if not x.is_contiguous() or not y.is_contiguous():
        raise ValueError("this lab requires contiguous tensors")
    if x.shape != y.shape or x.device != y.device:
        raise ValueError("shape and device must match")

    out = torch.empty_like(x)
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    fused_mul_bias_relu_kernel[grid](x, y, out, n_elements, bias)
    return out

def torch_reference(x, y, bias):
    return torch.relu(x * y + bias)

def benchmark(fn, warmup, repeat):
    for _ in range(warmup):
        out = fn()
        del out
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record()
    for _ in range(repeat):
        out = fn()
        del out
    end.record()
    end.synchronize()
    return {
        "cuda_event_ms": start.elapsed_time(end) / repeat,
        "host_wall_ms": (time.perf_counter() - wall_start) * 1e3 / repeat,
    }

def peak_allocation(fn):
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = fn()
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
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("未检测到 CUDA GPU；请先运行 Level 0")
    if args.n <= 0 or args.warmup < 0 or args.repeat <= 0:
        raise SystemExit("invalid arguments")

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    x = torch.randn(args.n, device=device, dtype=torch.float32)
    y = torch.randn_like(x)

    reference = torch_reference(x, y, args.bias)
    candidate = triton_fused(x, y, args.bias)
    max_abs_error = (reference - candidate).abs().max().item()
    if not torch.allclose(reference, candidate, rtol=1e-5, atol=1e-6):
        raise AssertionError(f"Triton mismatch: {max_abs_error}")
    del reference, candidate
    torch.cuda.synchronize()

    torch_fn = lambda: torch_reference(x, y, args.bias)
    triton_fn = lambda: triton_fused(x, y, args.bias)

    # 上面的正确性调用已经触发 JIT/Autotune；正式计时不含首次编译。
    torch_stats = benchmark(torch_fn, args.warmup, args.repeat)
    triton_stats = benchmark(triton_fn, args.warmup, args.repeat)
    torch_peak = peak_allocation(torch_fn)
    triton_peak = peak_allocation(triton_fn)

    useful_bytes = args.n * 12
    triton_seconds = triton_stats["cuda_event_ms"] / 1e3
    report = {
        "environment": {
            "torch": torch.__version__,
            "triton": triton.__version__,
            "torch_cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "compute_capability": [props.major, props.minor],
            "total_memory_gib": props.total_memory / 2**30,
        },
        "config": vars(args),
        "correctness": {"allclose": True, "max_abs_error": max_abs_error},
        "torch_eager": {
            "timing": torch_stats,
            "peak_incremental_allocation_mib": torch_peak / 2**20,
        },
        "triton_fused": {
            "timing": triton_stats,
            "peak_incremental_allocation_mib": triton_peak / 2**20,
            "modeled_effective_gbps": useful_bytes / triton_seconds / 1e9,
        },
        "speedup_by_cuda_event": (
            torch_stats["cuda_event_ms"] / triton_stats["cuda_event_ms"]
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
```

### 运行命令

```
TRITON_PRINT_AUTOTUNING=1 \
python triton_fusion_lab.py --n 16777216 --warmup 10 --repeat 50
```

双 RTX 3080 20GB 可分别运行：

```
CUDA_VISIBLE_DEVICES=0 python triton_fusion_lab.py --n 16777216
CUDA_VISIBLE_DEVICES=1 python triton_fusion_lab.py --n 16777216
```

### 预期现象

- 输出 Triton、PyTorch、CUDA、GPU、Compute Capability 与显存信息；
- `correctness.allclose` 为 `true` ；
- 第一次新 Shape 运行会出现 JIT/Auto-tuning 延迟；
- 缓存命中后的 Triton Kernel 是一次融合 Launch；
- Eager Reference 可能生成多个 Kernel 和中间 Tensor；
- Triton 增量峰值显存通常更低；
- Speedup 随 GPU、Shape、版本、频率和 PyTorch 实现变化，不保证大于 1。

### 结果分析

本实验把上一课的 CUDA C++ 融合算子改写为 Triton。两者追求同样的数据流：两次读、一次写，中间值留在片上。区别在于：

- CUDA C++ 显式控制 Thread、Block 和低层 API；
- Triton 以 Program/Block Tensor 表达数据块，编译器负责更多映射；
- Triton Auto-tuner 可以按 Shape 和 GPU搜索配置；
- CUDA C++ 在特殊指令、复杂同步和成熟 Library 集成上仍有更细控制。

## 进阶：Reduction 与 Softmax 的 Triton 思维

Softmax：

$$
y_i=\frac{e^{x_i-\max(x)}}{\sum_j e^{x_j-\max(x)}}
$$

朴素框架实现可能物化 Max、Shift、Exp、Sum、Divide 等中间 Tensor。Triton 的典型策略是让一个 Program 处理一行：

```
row = tl.load(row_ptrs, mask=mask, other=-float("inf"))
row = row - tl.max(row, axis=0)
numerator = tl.exp(row)
denominator = tl.sum(numerator, axis=0)
output = numerator / denominator
tl.store(out_ptrs, output, mask=mask)
```

核心难点：

- 一行能否装入片上资源；
- `BLOCK_SIZE` 常取不小于列数的 2 次幂；
- Padding Lane 的 `other` 必须是负无穷；
- 行太宽时 Register/Shared Memory 会爆炸，需要分块或多阶段算法；
- Triton `tl.exp` 采用适合 GPU 的快速近似路径，需验证数值容差。

这说明 Triton 的价值不只是语法短，而是能自然表达“整行加载→片上 Reduction→一次写回”的融合数据流。

## 进阶：Tiled Matrix Multiplication

对 A\[M,K\]B\[K,N\]：

1. 每个 Program 负责 C 的一个 `[BLOCK_M,BLOCK_N]` Tile；
2. 沿 K 维循环加载 A/B Tile；
1. 用 `tl.dot` 累加到 FP32 Accumulator；
2. 在 Accumulator 写回前融合 Bias/Activation；
1. 用 Program Reordering 提升 L2 Reuse；
2. Auto-tune `BLOCK_M/N/K` 、 `GROUP_M` 、 `num_warps/stages` 。

伪代码：

```
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for k in range(0, tl.cdiv(K, BLOCK_K)):
    a = tl.load(a_ptrs, mask=a_mask, other=0.0)
    b = tl.load(b_ptrs, mask=b_mask, other=0.0)
    acc = tl.dot(a, b, acc)
    a_ptrs += BLOCK_K * stride_ak
    b_ptrs += BLOCK_K * stride_bk

# Epilogue Fusion
acc = tl.where(acc > 0.0, acc, 0.01 * acc)
tl.store(c_ptrs, acc, mask=c_mask)
```

对 FP32 `tl.dot` ，当前接口的目标精度选项会影响 Tensor Core 路径；默认行为可能使用 TF32。不能拿不同 `input_precision` 的结果只比速度、不比误差。

对于标准 GEMM，优先使用 cuBLAS/cuBLASLt 或框架 Library；Triton 更适合需要特殊 Layout、融合 Epilogue、稀疏/分组模式或 Library 无法直接表达的工作负载。

## 调试方法

### Interpreter

```
TRITON_INTERPRET=1 python triton_fusion_lab.py \
  --n 1024 --warmup 0 --repeat 1
```

Interpreter 在 CPU 上顺序模拟每个 Program Instance，可用普通 `print` 和 `pdb` 检查中间 Block。已知边界包括：不直接支持 BF16 操作，以及部分 Indirect Memory Access Pattern 不支持。Interpreter 用于正确性定位，不代表 GPU 性能。

### 编译期与运行期断言

Triton 提供：

- `static_print` 、 `static_assert` ：编译期；
- `device_print` 、 `device_assert` ：运行期；
- `device_assert` 需要设置 `TRITON_DEBUG=1` 才执行。

生产 Kernel 不应依赖打印调试；调试完成后关闭高开销输出。

### Compute Sanitizer

```
compute-sanitizer --tool memcheck \
  python triton_fusion_lab.py --n 100003 --warmup 0 --repeat 1
```

使用不规则 `n` 更容易暴露 Tail Mask 和越界问题。Sanitizer 会显著拖慢运行，不能把其耗时当作性能结果。

## Profiling

### Nsight Systems

```
mkdir -p reports
nsys profile \
  --trace=cuda,osrt \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  -o reports/triton_timeline \
  python triton_fusion_lab.py --n 4194304 --warmup 2 --repeat 5
```

确认：

- 首次 JIT/Auto-tuning 与稳定执行区间被分开；
- Eager 路径与 Triton 路径的 Kernel 数；
- Kernel 之间是否存在 Launch Gap；
- Triton Kernel 是否位于 Critical Path。

### Nsight Compute

```
ncu \
  --kernel-name regex:.*fused_mul_bias_relu.* \
  --launch-count 1 \
  --set full \
  --force-overwrite \
  -o reports/triton_fused \
  python triton_fusion_lab.py --n 4194304 --warmup 1 --repeat 1
```

检查：

- DRAM/L2 Read/Write Byte 是否接近数据流预期；
- Memory Throughput；
- Register Per Thread 与 Occupancy；
- Eligible Warp、Stall 与 Instruction；
- Auto-tuner 选出的 Block/Warps 是否合理。

`--set full` 可能多次 Replay，最终速度仍需无 Profiler Benchmark。

## Level 2：架构专项与不可等价边界

| 架构 | 常见 GPU | Triton 重点 | 边界 |
| --- | --- | --- | --- |
| Ampere | RTX 3080/3090、A100 | CC 8.x、 `tl.dot` 、Tensor Core、Software Pipeline | RTX 3080/3090 不具备 A100 HBM/NVLink/MIG；没有 TMA |
| Ada Lovelace | RTX 4090、L40/L40S | CC 8.9、更大 L2、推理融合与 FP8 软件路径 | RTX 4090 属于 Ada 且无 NVLink，不是 Blackwell |
| Hopper | H100/H200 | SM90、TMA Tensor Descriptor、Warp Specialization、Cluster、FP8 | 必须在真实 Hopper 验证 TMA/WGMMA/Transformer Engine |
| Blackwell | RTX 5090、B100/B200/GB200 | 新 Tensor Core、Block-scaled Matmul、FP4、更新后端 | RTX 5090 不能代表 B200/GB200 的 HBM、NVLink/Fabric |

### 双 RTX 3080 的建议实验

1. 使用当前稳定 Triton 与匹配 PyTorch；
2. 对 `n=2^12...2^27` 扫描 Pointwise Kernel；
1. 记录 Auto-tuner 在不同 Shape 选出的 Block；
2. 用 Nsight Compute 分析 `sm_86` 的 Register、Occupancy 与 DRAM；
1. 编写 FP16 `tl.dot` Matmul，与 `torch.mm` 比较正确性和速度；
2. 不尝试用 Tensor Descriptor 宣称验证 TMA。

### Hopper 的真实 TMA 路径

当前 Triton 提供 Tensor Descriptor 接口；在具有 TMA 支持的 NVIDIA GPU 上，Descriptor Load/Store 可由 TMA Hardware 支撑。它对 Shape、Stride、Alignment 和 Block Shape 有约束，需要目标 GPU、Driver 与 Triton 版本共同支持。

RTX 3080 上可以学习 Descriptor 的抽象，但不能得到 Hopper TMA 的等价性能或 Counter。

### Blackwell 路径

Triton 官方教程已包含 Persistent Matmul 和 Block-scaled Matrix Multiplication 等方向。FP4/Block Scaling 依赖真实硬件、具体数据格式和后端版本；RTX 5090 虽然属于 Blackwell，也不能代替 B100/B200/GB200 的数据中心互联与 HBM 实验。

## Triton、CUDA C++、Library 与编译器怎么选

| 方案 | 适合场景 | 主要代价 |
| --- | --- | --- |
| 框架内置/Library | 标准 GEMM、Conv、Attention、Norm | 定制空间有限 |
| `torch.compile` | 标准 PyTorch Graph 的自动融合 | 编译、Graph Break、动态 Shape |
| Triton | 特殊融合、Layout、Pointwise/Reduction、定制 Matmul | 要理解数据流与目标后端 |
| CUDA C++ | 特殊指令、复杂同步、极致控制、成熟发布 | 开发与维护成本高 |
| CUTLASS/cuBLASLt | Tensor Core GEMM 与 Epilogue 定制 | Template/API 学习成本 |

推荐顺序：

1. 先检查框架与 Vendor Library；
2. 尝试编译器自动融合；
1. 热点且无法满足时用 Triton；
2. Triton 表达不足或需要最底层控制时再用 CUDA C++/PTX。

## 常见错误与排查

### 1\. ModuleNotFoundError: triton

确认虚拟环境、Python 与安装位置：

```
which python
python -m pip show triton
python -c "import triton; print(triton.__version__)"
```

### 2\. GPU 不在支持范围

```
python -c "import torch; print(torch.cuda.get_device_name(), torch.cuda.get_device_capability())"
```

当前主线声明 NVIDIA CC 8.0+。V100 为 CC 7.0，不应强行套用本课主线环境。

### 3\. Mask 遗漏导致越界

用不是 Block Size 整倍数的 Shape 测试，例如 `100003` ；Interpreter 与 Compute Sanitizer 共同验证。

### 4\. tl.arange 或 Block Shape 编译失败

许多 Triton Block 操作要求静态且适合编译器的 Shape，常使用 2 次幂 Block。确认参数标记为 `tl.constexpr` ，并控制 Block 不要超过目标约束。

### 5\. 第一次运行特别慢

这是 JIT Compile/Auto-tuning 的典型现象。区分 Cold Compile、Warm Cache 和 Steady-State；生产部署考虑预热代表性 Shape。

### 6\. Auto-tune 把输出累加多次

Auto-tuner 会执行所有候选。In-place、Atomic、随机或有副作用 Kernel 必须用 Reset/Restore/Hook，或设计幂等 Benchmark。

### 7\. 配置越多反而系统越慢

候选过多和 Key 过细会造成编译、Benchmark 与缓存爆炸。先用性能模型剪枝，再保留少量有意义配置。

### 8\. Triton 比 PyTorch 慢

可能：

- PyTorch 已调用高度优化的 Library/Fused Kernel；
- Shape 太小，Wrapper/Allocation 主导；
- Block/Warps 不合适；
- Register/Occupancy 恶化；
- 访存不合并；
- Reference 与 Triton 的精度或工作量不同。

### 9\. Interpreter 正确，GPU 错误

检查 Race、Atomic、同步假设、Dtype、Fast Math、Backend 差异和 Interpreter 不支持的模式。继续使用 Sanitizer 与不规则 Shape。

### 10\. FP32 Matmul 结果差异大

检查 `tl.dot` 的 `input_precision` 。TF32、IEEE 和其他精度路径速度与误差不同；必须显式记录。

### 11\. OOM 或 Register 爆炸

减小 Block/Tile、 `num_warps` 或 `num_stages` ，检查 Reduction 行宽和 Live Value。不要把整张超宽矩阵行硬塞进一个 Program。

### 12\. 新 GPU 无法编译

新架构需要 Driver、PyTorch 和 Triton 后端共同支持。先核对当前发行版，不要只因 CUDA Driver 能识别 GPU 就认定 Triton Wheel 包含目标代码生成。

## 优化前后对照

| 维度 | PyTorch Eager 链 | Triton Fused | 验证方式 |
| --- | --- | --- | --- |
| 表达 | 多个 Tensor Operator | 一个 Program/Block 数据流 | 代码与 IR 思维 |
| Kernel 数 | 可能多个 | 1 | Nsight Systems |
| 中间 Tensor | 乘法、Bias 中间值 | Register 中间值 | Allocator/Profiler |
| 有用显存流量 | 高于最小值 | 模型约 12N Byte | Nsight Compute |
| 首次成本 | Operator 初始化 | JIT + Auto-tuning | Cold/Warm 分测 |
| Shape 适配 | Library/Framework | Config + Tuning Key | Shape Sweep |
| 数值 | 框架语义 | 近似函数/精度需声明 | `allclose` /边界测试 |
| 维护 | 简单 | 自定义 Kernel 测试矩阵 | 工程评审 |

## 面试题与答案

### 1\. Triton Program Instance 与 CUDA Thread 有什么区别？

Program Instance 通常处理一个 Block/Tile，由一个或多个 Warp 协同执行；CUDA Thread 通常执行标量线程逻辑。Triton 编译器负责把 Block Tensor 操作映射到底层线程与指令。

### 2\. tl.constexpr 有什么作用？

它声明编译期 Meta-parameter，可决定 Block Shape、静态分支和 Unroll。不同值通常生成不同 Kernel 变体。

### 3\. 为什么需要 Mask？

Grid 的最后一个 Block 往往超出真实 Shape。Masked Load/Store 阻止越界，并用合适 `other` 保持 Reduction 语义。

### 4\. num\_warps 越大越好吗？

不是。更多 Warp 可能提高协同和并行度，也会增加资源占用；简单 Pointwise Kernel 可能没有收益。必须按 Shape 和架构测量。

### 5\. Auto-tune 的最大陷阱是什么？

它会多次执行 Kernel。对 In-place、Atomic 或有副作用操作，候选之间状态可能不等价；另一个陷阱是 Tuning Key 过细导致延迟和缓存爆炸。

### 6\. Triton 为什么适合 Kernel Fusion？

它能在一个 Program 中加载 Block、完成 Elementwise/Reduction/ `tl.dot` 和 Epilogue，只写回最终结果，表达数据复用比逐 Thread CUDA 更紧凑。

### 7\. Triton 能替代 cuBLAS 吗？

标准 GEMM 优先 Vendor Library。Triton 适合特殊 Layout、融合 Epilogue、分组/稀疏或 Library 无法覆盖的 Shape；是否更快必须实测。

### 8\. Triton Kernel 如何调试？

先用 Reference 与不规则 Shape；再用 `TRITON_INTERPRET=1` 、Static/Device Assert、Compute Sanitizer；最后用 Nsight 定位性能。Interpreter 不能代表 GPU 性能。

### 9\. Triton 的第一次调用为什么慢？

新签名和 Meta-parameter 组合可能触发 JIT Compile，Auto-tune 还会 Benchmark 多个配置。生产应预热，并分开报告 Cold 与 Warm。

### 10\. 为什么 RTX 3080 不能验证 TMA？

RTX 3080 是 Ampere CC 8.6，没有 Hopper 的 TMA Hardware。软件接口或语义模拟不能产生真实 TMA 数据通路、吞吐与 Counter。

## 课后练习

1. 修改 Level 0 的 `n` 与 Block，绘制 Tail Efficiency 曲线。
2. 给 Triton 融合算子加入上限裁剪 `min(max(v,0),6)` ，验证仍只有一个 Kernel。
1. 加入 `BLOCK_SIZE=128/2048` 候选，观察不同 Shape 的最佳配置。
2. 将 Auto-tune Key 从完整 `n_elements` 改成 Shape Bucket，比较调优次数与性能。
1. 实现 Vector Add，对比理论 12N Byte 与 Nsight Compute 实测流量。
2. 实现 Row-wise Softmax，使用负无穷填充 Mask Lane，测试奇数列数。
1. 实现 FP16 Tiled Matmul，显式记录 `BLOCK_M/N/K` 、Warps、Stages 和误差。
2. 在 Matmul Epilogue 中融合 Bias + Leaky ReLU，对比中间 Tensor 和 Kernel 数。
1. 用 Interpreter 和 `pdb` 单步观察一个 Program 的 Offset、Mask 和 Value。
2. 在双 RTX 3080 上分别做 Shape Sweep，解释最佳配置差异是否来自温度、频率或噪声。

## Checklist

### 语义与正确性

- 明确每个 Program Instance 负责的数据 Tile。
- Grid 覆盖所有元素且没有重复写冲突。
- Tail Load/Store 使用正确 Mask。
- Reduction 的 other 是正确单位元。
- Shape、Stride、Dtype、Contiguous 与 Alignment 契约明确。
- 使用不规则 Shape、极值、NaN/Inf 和随机数据测试。

### 性能设计

- 已计算 FLOPs、最少 Byte、AI 和理论瓶颈。
- Global Memory 访问连续或可合并。
- Block/Tile 不会造成严重 Tail Waste。
- Auto-tune 候选经过模型剪枝。
- 检查 Register、Shared Memory、Occupancy 与 Spill。
- JIT/Auto-tuning 未混入 Steady-State Benchmark。

### 调试与分析

- 有 PyTorch/NumPy Reference。
- Interpreter 仅用于语义调试，不用于性能结论。
- 使用 Assert、Sanitizer 验证边界与 Race。
- Nsight Systems 验证 Kernel 数与 Launch Gap。
- Nsight Compute 验证 Memory、Register、Occupancy 和指令。
- 最终以无 Profiler 的端到端指标判断。

### 工程化

- 控制 JIT Variant、Auto-tune Key 和 Cache 规模。
- 对有副作用 Kernel 使用 Reset/Restore/Hook。
- 记录 PyTorch、Triton、Driver、GPU 和目标精度。
- 有 Unsupported Shape/Dtype/Hardware 回退。
- 比较现有 Library 与 torch.compile，避免重复造轮子。

### 架构边界

- Ampere 正确包含 RTX 3080/3090、A100。
- Ada 正确包含 RTX 4090、L40/L40S。
- Hopper 正确包含 H100/H200。
- Blackwell 正确包含 RTX 5090、B100/B200/GB200。
- 没有把 RTX 4090 写成 Blackwell。
- 没有用 Ampere 模拟验证 Hopper TMA/WGMMA。
- 没有用 RTX 5090 外推 GB200 NVLink/Fabric 能力。
- 性能数字只描述实际环境，不作为通用结论。