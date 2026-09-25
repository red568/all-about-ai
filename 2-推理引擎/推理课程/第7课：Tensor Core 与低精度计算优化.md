---
title: "第7课：Tensor Core 与低精度计算优化"
source: "https://everythingai.top/learn/ai-systems-performance-engineering-text/lesson-07"
author:
published:
created: 2026-09-26
description: "面向 AI 工程学习者的专业课程学习空间。"
tags:
  - "clippings"
---
**导航**

本课讲清 Tensor Core 与低精度计算的工作机制，帮助你在速度、显存和数值稳定性之间取得平衡。

## 课程定位

低精度是现代 AI 性能工程中最容易“看起来懂了、实际用错了”的领域。把 FP32 Tensor 转成 FP16 只是一种数据类型转换；只有算子、形状、布局、硬件和软件栈共同满足条件，计算才可能进入 Tensor Core 高吞吐路径。即使 Kernel 变快，数值溢出、额外量化、数据搬运或其他非矩阵算子也可能让端到端性能没有提升。

本课从矩阵乘累加的硬件直觉出发，建立 TF32、FP16、BF16、FP8、FP4、INT8/INT4 的统一理解，讲清输入精度、累加精度、存储精度和缩放粒度的区别。实战部分不绑定某一型号：CPU 可以完成数值模拟，常见 NVIDIA GPU 可以自动检测并测试 FP32/TF32/FP16/BF16，Ada/Hopper/Blackwell 可按能力选择 FP8，Blackwell 再选择 NVFP4。

## 学习目标

完成本课后，你能够：

1. 解释 Tensor Core 与普通 CUDA Core 的分工，以及 MMA 的基本语义。
2. 区分存储 dtype、输入计算 dtype、乘法精度和累加精度。
1. 理解 FP32、TF32、FP16、BF16、FP8 E4M3/E5M2、NVFP4 与 INT8/INT4 的精度和动态范围权衡。
2. 解释 Automatic Mixed Precision、Loss Scaling、Amax 与缩放因子的作用。
1. 识别“转换成低精度但没有提速”的形状、布局、算子和端到端原因。
2. 在 PyTorch 中正确设置 TF32、Autocast 与 GradScaler，并兼容新旧稳定版本 API。
1. 为 Ampere、Ada、Hopper、Blackwell 建立正确的能力矩阵与回退路径。
2. 同时验证速度、显存、误差、溢出和模型质量，不把峰值 FLOPS 当成业务结论。

## 前置知识

- 理解 GPU 的 SM、Warp、Register、Shared Memory 与 HBM/GDDR。
- 会计算 GEMM 的 FLOP 数：2MNK。
- 理解 Roofline、Latency、Throughput 和有效带宽。
- 会运行 Python；Level 1 需要可选 PyTorch，Level 2 需要对应 NVIDIA GPU。

## 核心直觉：Tensor Core 是“矩阵块流水线”

普通标量 FMA 可以理解为：

$$
d=a\times b+c
$$

Tensor Core 面向矩阵块执行 MMA（Matrix Multiply-Accumulate）：

$$
D=A\times B+C
$$

它不是“一条指令完成任意大小矩阵”。大 GEMM 会被库或 Kernel 切成层级 Tile：

```
完整 GEMM
└── Thread Block / Cluster Tile
    └── Warp / Warpgroup Tile
        └── Tensor Core MMA Tile
```

高吞吐来自以下共同条件：

- Tile 形状与硬件 MMA 形状匹配；
- A、B 数据按要求布局、对齐并及时搬到片上；
- 有足够工作填满 SM；
- 多级流水隐藏 Global/Shared Memory 延迟；
- 累加器和缩放不会成为额外瓶颈；
- 输出精度满足模型要求。

因此，“Tensor Core 更快”的完整表述应是：

## Tensor Core 的系统原理

### 矩阵乘法为什么适合专用单元

对 $A\in\mathbb{R}^{M\times K}$ 、 $B\in\mathbb{R}^{K\times N}$ ：

$$
C_{ij}=\sum_{k=1}^{K}A_{ik}B_{kj}
$$

总计算量近似为：

$$
FLOPs=2MNK
$$

当 Tile 中的 A、B 被重复使用时，GEMM 具有较高算术强度。Tensor Core 将固定形状的小矩阵乘累加映射到专用数据通路，Warp 或 Warpgroup 负责提交分片、持有累加器并协调搬运。

### Tensor Core 不负责所有算子

容易从 Tensor Core 获益的工作：

- GEMM、BMM、Linear；
- 卷积的部分实现；
- Attention 中的 QK 和 PV 矩阵乘；
- MoE Expert MLP 的 grouped GEMM；
- 某些结构化稀疏矩阵乘。

不一定直接由 Tensor Core 加速的工作：

- Softmax、LayerNorm/RMSNorm；
- Sampling、Top-k、Tokenizer；
- 很小的逐元素 Kernel；
- 不规则 Gather/Scatter；
- CPU 调度、通信、I/O。

如果一次训练 Step 中 GEMM 只占 50%，即使 GEMM 理想加速 4 倍，Amdahl 上界也只有：

$$
S=\frac{1}{(1-0.5)+0.5/4}=1.6
$$

### 输入、累加与输出可以是不同精度

低精度计算常见模式为：

```
低精度 A × 低精度 B → 较高精度 Accumulator → 指定输出 dtype
```

例如 FP16/BF16 输入可以使用 FP32 累加器。这样既提高乘法吞吐，又降低长点积累计误差。不要只看 Tensor 的存储 dtype；库实际选择的 MMA 指令和 accumulation mode 才决定计算路径。

## 数值格式全景

### 常见格式

| 格式 | 典型构成 | 直觉 | 常见用途 |
| --- | --- | --- | --- |
| FP32 | 1 sign + 8 exponent + 23 fraction | 范围和精度都较好 | 主权重、敏感归约、基准 |
| TF32 | 1 + 8 + 10，用于计算路径 | FP32 范围、接近 FP16 尾数精度 | Ampere+ 的 FP32 GEMM/Conv 加速 |
| BF16 | 1 + 8 + 7 | 范围接近 FP32，精度较粗 | 现代训练主力半精度 |
| FP16 | 1 + 5 + 10 | 精度较 BF16高、范围较小 | 训练、推理、消费卡通用路径 |
| FP8 E4M3 | 1 + 4 + 3 | 精度较好、范围较小 | 常用于前向/激活等场景 |
| FP8 E5M2 | 1 + 5 + 2 | 范围更大、精度更低 | 常用于梯度等大范围场景 |
| NVFP4 | 4-bit 值 + 分层 scale | 极低存储与吞吐路径 | Blackwell 低精度训练/推理 |
| INT8/INT4 | 整数 + scale/zero point | 规则、便于量化 | 推理权重/激活量化 |

TF32 不是常规模型存储格式。PyTorch 中 Tensor 仍是 `float32` ，cuBLAS/cuDNN 等后端可在受支持操作内部用 TF32 读取较少的有效尾数位并在更高精度中累加。

### 动态范围与精度是两条轴

浮点数可抽象为：

$$
x=(-1)^s\times(1.m)\times 2^{e-bias}
$$

- exponent bits 决定主要动态范围；
- fraction/mantissa bits 决定相邻可表示数之间的精细程度。

BF16 保留 8 位 exponent，所以比 FP16 更不容易溢出，但只有 7 位 fraction，单次数值舍入更粗。FP16 有 10 位 fraction，但 exponent 只有 5 位，最大有限值为 65504，更容易出现 overflow/underflow。

所以“BF16 精度低于 FP16”只描述尾数分辨率；在训练中，BF16 的大范围往往带来更稳定的整体行为。

### FP8 不是把 Tensor 直接.to(float8) 就结束

FP8 的有限范围和较少有效位要求缩放：

$$
x_{fp8}=cast_{fp8}(x\times s)
$$

$$
\hat{x}=cast_{high}(x_{fp8})/s
$$

缩放策略需要决定：

- 用历史 Amax 还是当前 Amax；
- 每 Tensor、每通道还是每 Block 一个 scale；
- scale 多久更新一次；
- 分布式训练时 Amax 是否跨 Rank 归约；
- 前向、反向是否使用不同 FP8 格式。

NVIDIA Transformer Engine 的作用不是“所有层自动降到最低精度”，而是为受支持 Transformer 模块管理低精度配方、Amax、scale、量化缓存与高性能 Kernel。哪些操作进入 FP8/NVFP4，仍受 recipe、形状、框架版本和硬件支持约束。

### NVFP4 不是普通 INT4

NVFP4 使用浮点值与分层缩放表达数据。它与常见的对称 INT4 权重量化不同：

- 表示集合不同；
- scale 结构不同；
- 硬件 MMA 路径不同；
- 校准、训练和精度行为不同；
- 文件中写着“4-bit”并不能证明能走 NVFP4 Tensor Core。

没有 Blackwell 与对应软件栈时，可以模拟 4-bit 量化误差，但不能声称验证了 NVFP4 吞吐。

## 混合精度训练

### Autocast：按算子选择 dtype

Automatic Mixed Precision 不是把整个模型无脑转换成半精度。Autocast 根据算子策略选择：

- GEMM/Conv 等适合低精度的算子使用 FP16/BF16；
- 某些归约、损失和敏感算子保留 FP32；
- 输出 dtype 按算子规则传播。

典型训练结构：

```
with torch.autocast(device_type="cuda", dtype=torch.float16):
    output = model(inputs)
    loss = loss_fn(output, target)

# backward 不建议额外包在 autocast 中
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Loss Scaling：解决 FP16 梯度下溢

小梯度在 FP16 中可能舍入为 0。若把 Loss 乘以尺度 S：

$$
L'=S\cdot L
$$

则反向梯度也被放大：

$$
\nabla L'=S\cdot\nabla L
$$

优化器更新前再除以 S，数学上恢复原梯度。动态 GradScaler 在检测到 Inf/NaN 时跳过更新并减小 scale，稳定时逐渐增大。

BF16 动态范围接近 FP32，通常不需要 GradScaler；但是否使用应由训练栈和数值行为决定，而不是只按习惯复制代码。

### Master Weight 与 Optimizer State

混合精度训练可能仍维护 FP32 主权重或 FP32 Optimizer State。因此，FP16/BF16 前向并不意味着训练显存严格减半：

$$
M_{train}\approx M_{model\ copy}+M_{master}+M_{grad}+M_{optimizer}+M_{activation}+M_{workspace}
$$

AMP 常显著减少激活和部分算子流量，但总节省取决于优化器和框架实现。

## Tensor Core 性能模型

### 计算时间与转换时间

一次低精度算子的端到端时间可近似拆成：

$$
T_{low}=T_{cast}+T_{scale}+T_{layout}+T_{mma}+T_{epilogue}+T_{sync}
$$

若输入原本是 FP32，而矩阵很小， `cast + scale` 可能比节省的 MMA 时间还大。更理想的系统让数据在多个算子间保持低精度，并融合 scale、bias、activation 等 Epilogue。

### 可达 Tensor Core 利用率

定义粗略的 Tensor Core MFU：

$$
MFU_{tc}=\frac{2MNK/t}{P_{peak,dtype}}
$$

必须确保分母与实际 dtype、稀疏性口径一致：

- dense 与 2:4 sparse 峰值不能混用；
- boost clock 与持续功耗状态不能混用；
- FP8、NVFP4、FP16 峰值不能混用；
- 单 GPU 与整机/整柜数字不能混用。

### Shape Efficiency

假设实现需要把维度补齐到 Tile 倍数 q：

$$
M'=\lceil M/q\rceil q,\quad N'=\lceil N/q\rceil q,\quad K'=\lceil K/q\rceil q
$$

有用计算比例为：

$$
\eta_{shape}=\frac{MNK}{M'N'K'}
$$

小而不规则的矩阵可能产生大量 padding 或尾块，使 Tensor Core 峰值难以转化为吞吐。实际库支持多种 Tile， `q=8/16` 只是教学模型，不是所有硬件的固定要求。

### 端到端加速上界

若低精度可加速部分占比为 p，该部分加速 s：

$$
S_{end}=\frac{1}{(1-p)+p/s}
$$

再考虑精度失败重跑、通信、I/O 和 Batch 变化，实际 Goodput 可能更低。最终目标应是满足质量/SLO 的 tokens/s 或 samples/s，而不是单个 GEMM 的 TFLOPS。

## 架构能力矩阵

### 不同代际的正确映射

| 架构 | 常见产品 | 典型 CC | 本课通用能力定位 |
| --- | --- | --- | --- |
| Ampere | A100、RTX 3080/3090 | 8.0 / 8.6 | TF32、BF16、FP16、INT8/INT4 Tensor Core；无 FP8 Tensor Core |
| Ada Lovelace | RTX 4090、L40/L40S | 8.9 | 第四代 Tensor Core，增加 FP8 输入能力 |
| Hopper | H100/H200 | 9.0 | FP8 Transformer Engine、第四代 Tensor Core、数据中心训练路径 |
| Blackwell 数据中心 | B100/B200/GB200 | 10.x | 第五代 Tensor Core、FP8/FP6/FP4，TE 支持 MXFP8/NVFP4 |
| Blackwell GeForce | RTX 5090 | 12.x | Blackwell 消费级目标，支持格式集合不等于 B200 软件能力完全相同 |

注意：

- RTX 4090 是 Ada，不是 Blackwell。
- Ada 的 FP8 硬件输入能力不代表任意 PyTorch 运算都会自动使用 FP8。
- Hopper 的代表性 FP8 训练路径不等于支持 Blackwell NVFP4。
- RTX 5090 与 B200 都是 Blackwell，但应分别编译、检测并验证。
- Ampere 上安装 Transformer Engine 可以使用其 FP16/BF16 优化组件，但不能产生真实 FP8 Tensor Core 执行。

### 形状与布局要求

不同 cuBLASLt、CUTLASS、Transformer Engine 与框架版本对形状有不同约束。常见经验是：

- GEMM 的 M/N/K 尽量是 8 或 16 的倍数；
- FP8 Transformer Engine Linear 常要求相关维度可被 16 整除；
- 指针对齐和 Leading Dimension 会影响向量化 load；
- Batch 太小或 K 太小可能无法填满 GPU；
- Grouped GEMM 需要同时考虑每组大小和负载均衡。

这些是选型起点，不是“不是 16 倍数就绝对不用 Tensor Core”的定律。现代库可处理尾块，只是效率可能下降。

## 瓶颈分析方法

### 第一步：确认实际执行路径

检查：

- 输入、权重、输出和 accumulator dtype；
- GPU 名称与 Compute Capability；
- PyTorch、CUDA、cuBLAS/cuDNN/TE 版本；
- Matmul 是否由 cuBLAS/cuBLASLt、CUTLASS、Triton 或自定义 Kernel 执行；
- Nsight Compute 中 Tensor Pipe 是否活跃；
- 是否发生隐式 cast、layout transform 或 fallback。

### 第二步：同时测速度与误差

至少记录：

- Median/P50 与 P99 延迟；
- samples/s 或 tokens/s；
- 有效 TFLOPS；
- 峰值显存；
- 相对误差、最大误差；
- Inf/NaN 比例；
- 最终模型指标，如 Loss、PPL、Accuracy、Reward。

### 第三步：区分五类失败

| 现象 | 常见原因 | 优先动作 |
| --- | --- | --- |
| dtype 已变但速度不变 | 算子太小、未走 Tensor Core | 放大 Shape，查 Kernel 与 Tensor Pipe |
| Kernel 快但 Step 不快 | 非 GEMM、通信、I/O 占比高 | 端到端时间线 + Amdahl |
| FP16 出现 NaN | 溢出、Loss Scale、敏感算子 | GradScaler、BF16、保留 FP32 |
| BF16 稳定但误差偏大 | 尾数更短、累计/归约敏感 | 高精度累加、局部 FP32 |
| FP8/FP4 没收益 | scale/layout 开销或 fallback | 融合量化、增大 GEMM、查 recipe |

### 第四步：建立精度门禁

优化不能只通过速度测试。建议建立：

```
单算子误差 → 小模型收敛 → 固定评测集 → 长时间稳定性 → 线上 SLO/质量
```

如果低精度让失败率增加，Goodput 可能下降，即使单次成功请求更快。

## 完整可运行实验：格式、误差、Tensor Core 与 AMP

### 环境准备

Level 0 仅需 Python 3：

```
python3 --version
```

Level 1 可选 PyTorch：

```
python3 -m venv .venv-tensor-core
source .venv-tensor-core/bin/activate
python -m pip install --upgrade pip
# 从 https://pytorch.org/get-started/locally/ 选择与本机驱动匹配的稳定版
python -m pip install torch
```

### 保存脚本

将以下内容保存为 `tensor_core_lab.py` ：

```
#!/usr/bin/env python3
import argparse
import json
import math
import random
import statistics
import struct
import time

def float_to_bits(x):
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]

def bits_to_float(x):
    return struct.unpack("<f", struct.pack("<I", x & 0xFFFFFFFF))[0]

def round_fp32_fraction(x, keep_fraction_bits):
    """模拟保留 FP32 exponent，只减少 fraction bits；使用近似 RNE 位舍入。"""
    bits = float_to_bits(x)
    exponent = (bits >> 23) & 0xFF
    if exponent == 0xFF or keep_fraction_bits >= 23:
        return x
    drop = 23 - keep_fraction_bits
    # round-to-nearest-even：bias 加上保留结果最低位。
    lsb = (bits >> drop) & 1
    rounding_bias = (1 << (drop - 1)) - 1 + lsb
    rounded = bits + rounding_bias
    mask = ~((1 << drop) - 1) & 0xFFFFFFFF
    return bits_to_float(rounded & mask)

def to_tf32(x):
    return round_fp32_fraction(x, 10)

def to_bf16(x):
    return round_fp32_fraction(x, 7)

def to_fp16(x):
    try:
        return struct.unpack("<e", struct.pack("<e", float(x)))[0]
    except OverflowError:
        return math.copysign(math.inf, x)

def quantize_int(values, bits, symmetric=True):
    if symmetric:
        qmax = 2 ** (bits - 1) - 1
        qmin = -qmax
        amax = max(abs(v) for v in values) or 1.0
        scale = amax / qmax
        q = [max(qmin, min(qmax, round(v / scale))) for v in values]
        dq = [v * scale for v in q]
        zero_point = 0
    else:
        qmin, qmax = 0, 2**bits - 1
        vmin, vmax = min(values), max(values)
        scale = (vmax - vmin) / max(qmax - qmin, 1) or 1.0
        zero_point = round(qmin - vmin / scale)
        q = [max(qmin, min(qmax, round(v / scale) + zero_point))
             for v in values]
        dq = [(v - zero_point) * scale for v in q]
    return q, dq, scale, zero_point

def error_stats(reference, actual):
    finite_pairs = [(a, b) for a, b in zip(reference, actual)
                    if math.isfinite(a) and math.isfinite(b)]
    abs_err = [abs(a - b) for a, b in finite_pairs]
    signal = math.sqrt(sum(a * a for a, _ in finite_pairs) /
                       max(len(finite_pairs), 1))
    rmse = math.sqrt(sum(x * x for x in abs_err) /
                     max(len(abs_err), 1))
    return {
        "max_abs_finite": max(abs_err, default=0.0),
        "mean_abs_finite": statistics.fmean(abs_err) if abs_err else 0.0,
        "rmse": rmse,
        "relative_rmse": rmse / max(signal, 1e-30),
        "finite_pair_count": len(finite_pairs),
        "inf_count": sum(math.isinf(x) for x in actual),
        "zero_count": sum(x == 0.0 for x in actual),
    }

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def level0(samples, seed, tile_multiple):
    rng = random.Random(seed)
    # 对数跨度用于同时观察大数、小数、溢出与下溢。
    values = []
    for _ in range(samples):
        sign = -1.0 if rng.random() < 0.5 else 1.0
        exponent = rng.uniform(-8.0, 5.0)
        values.append(sign * (10.0 ** exponent))

    formats = {
        "tf32_sim": [to_tf32(x) for x in values],
        "bf16_sim": [to_bf16(x) for x in values],
        "fp16_python": [to_fp16(x) for x in values],
    }

    small = [rng.gauss(0.0, 1.0) for _ in range(4096)]
    _, int8_deq, int8_scale, _ = quantize_int(small, 8)
    _, int4_deq, int4_scale, _ = quantize_int(small, 4)

    a = [rng.gauss(0.0, 1.0) for _ in range(4096)]
    b = [rng.gauss(0.0, 1.0) for _ in range(4096)]
    dot_ref = dot(a, b)
    dots = {
        "fp32_python": dot_ref,
        "tf32_inputs_fp64_acc": dot([to_tf32(x) for x in a],
                                     [to_tf32(x) for x in b]),
        "bf16_inputs_fp64_acc": dot([to_bf16(x) for x in a],
                                     [to_bf16(x) for x in b]),
        "fp16_inputs_fp64_acc": dot([to_fp16(x) for x in a],
                                     [to_fp16(x) for x in b]),
    }

    shapes = []
    for m, n, k in [(127, 127, 127), (128, 128, 128),
                    (511, 1025, 4095), (512, 1024, 4096)]:
        mp = math.ceil(m / tile_multiple) * tile_multiple
        np = math.ceil(n / tile_multiple) * tile_multiple
        kp = math.ceil(k / tile_multiple) * tile_multiple
        shapes.append({
            "shape": [m, n, k],
            "padded": [mp, np, kp],
            "useful_compute_ratio": (m * n * k) / (mp * np * kp),
        })

    return {
        "format_error": {name: error_stats(values, result)
                         for name, result in formats.items()},
        "integer_quantization": {
            "int8": {"scale": int8_scale,
                     "error": error_stats(small, int8_deq)},
            "int4": {"scale": int4_scale,
                     "error": error_stats(small, int4_deq)},
        },
        "dot_product": {
            name: {"value": value,
                   "relative_error": abs(value - dot_ref) /
                                     max(abs(dot_ref), 1e-30)}
            for name, value in dots.items()
        },
        "shape_padding_model": shapes,
        "warning": "数值模拟不等于对应 Tensor Core 硬件执行",
    }

def set_tf32(torch, enabled):
    """兼容 PyTorch 新 fp32_precision API 与旧 allow_tf32 API。"""
    mode = "tf32" if enabled else "ieee"
    try:
        torch.backends.cuda.matmul.fp32_precision = mode
        return "fp32_precision"
    except (AttributeError, RuntimeError, TypeError):
        torch.backends.cuda.matmul.allow_tf32 = enabled
        return "allow_tf32"

def cuda_median_ms(torch, fn, repeat, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)

def cpu_median_ms(fn, repeat, warmup=2):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)

def torch_level(matrix_n, repeat, train_steps, hidden, batch):
    try:
        import torch
    except ImportError:
        return {"available": False, "reason": "未安装 PyTorch"}

    use_cuda = torch.cuda.is_available() and torch.version.cuda is not None
    device = torch.device("cuda" if use_cuda else "cpu")
    output = {
        "available": True,
        "torch_version": torch.__version__,
        "device": str(device),
    }
    if use_cuda:
        name = torch.cuda.get_device_name(0)
        cc = torch.cuda.get_device_capability(0)
        output.update({
            "gpu": name,
            "compute_capability": f"{cc[0]}.{cc[1]}",
            "cuda_runtime": torch.version.cuda,
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            "capability_notes": {
                "tf32": cc[0] >= 8,
                "fp8_tensor_core": cc == (8, 9) or cc[0] >= 9,
                "blackwell_low_precision": cc[0] >= 10,
            },
        })

    torch.manual_seed(7)
    a32 = torch.randn((matrix_n, matrix_n), device=device,
                      dtype=torch.float32)
    b32 = torch.randn((matrix_n, matrix_n), device=device,
                      dtype=torch.float32)

    timer = (lambda fn: cuda_median_ms(torch, fn, repeat)) \
        if use_cuda else (lambda fn: cpu_median_ms(fn, repeat))

    # IEEE FP32 reference；CUDA 上显式关闭 TF32。
    if use_cuda:
        set_tf32(torch, False)
    reference = torch.mm(a32, b32)
    if use_cuda:
        torch.cuda.synchronize()

    results = []

    def add_result(label, a, b, configure=None):
        if configure:
            api = configure()
        else:
            api = None
        try:
            ms = timer(lambda: torch.mm(a, b))
            result = torch.mm(a, b).float()
            if use_cuda:
                torch.cuda.synchronize()
            diff = result - reference
            rel_l2 = (torch.linalg.vector_norm(diff) /
                      torch.linalg.vector_norm(reference)).item()
            tflops = (2 * matrix_n**3) / (ms / 1000) / 1e12
            results.append({
                "mode": label,
                "input_dtype": str(a.dtype),
                "median_ms": ms,
                "effective_tflops": tflops,
                "relative_l2_error_vs_ieee_fp32": rel_l2,
                "control_api": api,
            })
        except RuntimeError as exc:
            results.append({"mode": label, "skipped": True,
                            "reason": str(exc).splitlines()[0]})

    if use_cuda:
        add_result("fp32_ieee", a32, b32,
                   lambda: set_tf32(torch, False))
        add_result("fp32_tf32_allowed", a32, b32,
                   lambda: set_tf32(torch, True))
    else:
        add_result("fp32_cpu", a32, b32)

    add_result("fp16", a32.to(torch.float16), b32.to(torch.float16))
    if not use_cuda or torch.cuda.is_bf16_supported():
        add_result("bf16", a32.to(torch.bfloat16),
                   b32.to(torch.bfloat16))
    else:
        results.append({"mode": "bf16", "skipped": True,
                        "reason": "当前 CUDA 设备/栈未报告 BF16 支持"})
    output["matmul"] = {"n": matrix_n, "results": results}

    # 最小训练实验：相同初始权重，比较 FP32 与 AMP 的 Step 时间和稳定性。
    if train_steps > 0:
        import copy
        model_base = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden * 2),
            torch.nn.GELU(),
            torch.nn.Linear(hidden * 2, hidden),
        ).to(device)
        x = torch.randn((batch, hidden), device=device)
        target = torch.randn((batch, hidden), device=device)

        def run_train(use_amp, amp_dtype):
            model = copy.deepcopy(model_base)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
            scaler_enabled = use_cuda and use_amp and amp_dtype == torch.float16
            try:
                scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
            except TypeError:
                scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
            times, losses = [], []
            for _ in range(train_steps):
                if use_cuda:
                    torch.cuda.synchronize()
                start = time.perf_counter()
                opt.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=use_amp,
                ):
                    pred = model(x)
                    loss = torch.nn.functional.mse_loss(pred, target)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                if use_cuda:
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)
                losses.append(float(loss.detach()))
            return {
                "median_step_ms": statistics.median(times[2:] or times),
                "first_loss": losses[0],
                "last_loss": losses[-1],
                "finite": all(math.isfinite(x) for x in losses),
                "grad_scaler_enabled": scaler_enabled,
            }

        train = {"fp32": run_train(False, torch.float32)}
        if use_cuda:
            preferred = (torch.bfloat16 if torch.cuda.is_bf16_supported()
                         else torch.float16)
            train[str(preferred)] = run_train(True, preferred)
        else:
            # CPU autocast 仅作为流程回退，不代表 Tensor Core。
            train["cpu_bfloat16_autocast"] = run_train(True, torch.bfloat16)
        output["training"] = {
            "steps": train_steps,
            "batch": batch,
            "hidden": hidden,
            "runs": train,
        }
    return output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tile-multiple", type=int, default=16)
    parser.add_argument("--torch", action="store_true")
    parser.add_argument("--matrix-n", type=int, default=1024)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--train-steps", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=256)
    args = parser.parse_args()

    output = {"level0": level0(args.samples, args.seed,
                               args.tile_multiple)}
    if args.torch:
        output["level1"] = torch_level(
            args.matrix_n, args.repeat, args.train_steps,
            args.hidden, args.batch
        )
    print(json.dumps(output, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
```

### Level 0：CPU 数值与 Tile 模型

运行：

```
python3 tensor_core_lab.py > tensor_core_level0.json
python3 -m json.tool tensor_core_level0.json | less
```

比较不同 Tile 倍数：

```
for q in 8 16 32; do
  python3 tensor_core_lab.py --tile-multiple "$q" \
    > "tensor_core_tile_${q}.json"
done
```

### 预期现象

- TF32 模拟保留 FP32 的 exponent 范围，因此不应像 FP16 那样在较大数值处大量溢出。
- BF16 模拟的动态范围接近 FP32，但尾数更短，常见相对误差高于 TF32/FP16 输入模拟。
- FP16 在超过可表示范围时出现 Inf，在非常小的值处可能变成 0。
- INT4 的量化误差通常明显高于 INT8；scale 由本次样本的 Amax 决定。
- 127³ 补到 128³ 的浪费很小；511×1025×4095 补齐的浪费与每个维度余数共同相关。
- 模拟点积使用 Python 高精度累加，故只隔离输入舍入误差，不代表真实 Tensor Core 累加细节。

### 结果分析

Level 0 的目的不是模拟 Tensor Core 速度，而是把三件事拆开：

1. 输入表示误差；
2. 累加精度；
1. Tile padding 效率。

真实 Kernel 还包含 FMA 舍入、缩放、subnormal 处理、布局和流水。没有对应 GPU 时，只能完成数值方法实验，不能声称完成硬件性能验证。

### Level 1：通用 PyTorch Matmul 与 AMP

先检测设备：

```
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total \
  --format=csv,noheader 2>/dev/null || true

python3 - <<'PY'
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda runtime:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
        print("cc:", torch.cuda.get_device_capability(0))
        print("bf16:", torch.cuda.is_bf16_supported())
except ImportError:
    print("PyTorch not installed")
PY
```

快速 Matmul：

```
python3 tensor_core_lab.py --torch \
  --matrix-n 1024 --repeat 10 \
  > tensor_core_torch.json
```

显存充足时扩大 GEMM，并增加最小训练实验：

```
python3 tensor_core_lab.py --torch \
  --matrix-n 4096 --repeat 20 \
  --train-steps 30 --hidden 2048 --batch 512 \
  > tensor_core_cuda_large.json
```

CPU 回退应缩小规模：

```
python3 tensor_core_lab.py --torch \
  --matrix-n 512 --repeat 5 \
  --train-steps 5 --hidden 256 --batch 32 \
  > tensor_core_cpu.json
```

### 预期现象

- CUDA 设备自动输出名称、CC、CUDA Runtime、BF16 与低精度能力提示。
- Ampere 及更新 GPU 上，足够大的 TF32 Matmul 通常比 IEEE FP32 更快，但误差略高。
- FP16/BF16 的结果误差和速度不同；具体排名由 GPU、形状、时钟和库算法决定。
- 小矩阵可能没有明显加速，因为 Launch、转换和尾块占比较高。
- BF16 AMP 通常不启用 GradScaler；FP16 AMP 实验会启用动态 GradScaler。
- CPU BF16 Autocast 只验证框架流程，不能证明 NVIDIA Tensor Core 性能。

### 为什么性能不能照抄

脚本输出的 TFLOPS 是：

$$
TFLOPS=\frac{2N^3}{t\times10^{12}}
$$

它只代表本次环境、Shape 与库算法。GPU 可能因为功耗、温度、MIG、后台任务或首次编译改变结果。课程不提供“3090 一定是某个 TFLOPS”之类的通用结论。

### 误差基准

脚本将 CUDA TF32 关闭后的 FP32 Matmul 作为 Reference，再计算其他模式的相对 L2 误差：

$$
Error_{rel}=\frac{\lVert C_{low}-C_{fp32} \rVert_2}{\lVert C_{fp32} \rVert_2}
$$

这仍不是数学真值。严谨数值测试可对较小矩阵使用 CPU FP64 或高精度库作为 Reference，再覆盖真实模型分布和极端值。

### Level 2：FP8、Transformer Engine 与 NVFP4

### 能力门禁

```
python3 - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("没有 CUDA GPU：跳过 Level 2")
name = torch.cuda.get_device_name(0)
cc = torch.cuda.get_device_capability(0)
print({"gpu": name, "cc": cc})
if cc in [(8, 0), (8, 6), (8, 7)]:
    print("Ampere：测试 TF32/BF16/FP16；不能验证 FP8 Tensor Core")
elif cc == (8, 9):
    print("Ada：可检查 FP8 软件路径；不支持 Blackwell NVFP4")
elif cc[0] == 9:
    print("Hopper：可检查 FP8 Transformer Engine；不支持 NVFP4")
elif cc[0] >= 10:
    print("Blackwell：可检查 FP8/MXFP8/NVFP4，但仍需核对具体产品与软件版本")
else:
    print("不在本课能力表：仅运行框架报告支持的路径")
PY
```

### 隔离环境安装

Transformer Engine 对驱动、CUDA、PyTorch 和 GPU 有明确组合要求。不要把配套仓库偏 Blackwell/CUDA 13 的环境原样套到 RTX 3080。先根据当前官方安装说明选择匹配版本：

```
python3 -m venv .venv-te
source .venv-te/bin/activate
python -m pip install --upgrade pip
# 先安装匹配当前驱动的稳定 PyTorch，再安装 TE
python -m pip install torch
python -m pip install 'transformer_engine[pytorch]'
```

### FP8 最小实验

将下列内容保存为 `te_low_precision_probe.py` ：

```
import json
import torch

if not torch.cuda.is_available():
    raise SystemExit("需要 NVIDIA CUDA GPU")

cc = torch.cuda.get_device_capability(0)
name = torch.cuda.get_device_name(0)
if not (cc == (8, 9) or cc[0] >= 9):
    raise SystemExit(f"{name}, CC {cc}: 无 FP8 Tensor Core，使用 BF16/FP16 回退")

import transformer_engine.pytorch as te
from transformer_engine.common import recipe

torch.manual_seed(7)
in_features = 768       # 维度可被 16 整除
out_features = 3072
tokens = 2048

layer = te.Linear(in_features, out_features, bias=True).cuda()
x = torch.randn(tokens, in_features, device="cuda")

if cc[0] >= 10:
    recipes = {
        "fp8_delayed_scaling": recipe.DelayedScaling(
            fp8_format=recipe.Format.HYBRID
        ),
        "nvfp4": recipe.NVFP4BlockScaling(),
    }
else:
    recipes = {
        "fp8_delayed_scaling": recipe.DelayedScaling(
            fp8_format=recipe.Format.HYBRID
        )
    }

with torch.no_grad():
    reference = layer(x).float()
    output = {"gpu": name, "cc": cc}
    for label, lowp_recipe in recipes.items():
        # 预热并让 recipe 状态建立起来
        for _ in range(5):
            with te.autocast(enabled=True, recipe=lowp_recipe):
                y = layer(x)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(20):
            with te.autocast(enabled=True, recipe=lowp_recipe):
                y = layer(x)
        end.record()
        end.synchronize()
        rel = (torch.linalg.vector_norm(y.float() - reference) /
               torch.linalg.vector_norm(reference)).item()
        output[label] = {
            "mean_ms": start.elapsed_time(end) / 20,
            "relative_l2_error": rel,
            "finite": bool(torch.isfinite(y).all()),
        }

print(json.dumps(output, ensure_ascii=False, indent=2))
```

运行：

```
python3 te_low_precision_probe.py
```

### Level 2 预期与边界

- Ada/Hopper/Blackwell 可进入 FP8 候选路径，但是否成功取决于 TE、PyTorch、CUDA、驱动和 Shape。
- NVFP4 分支只在 Blackwell 候选设备启用；运行时失败应记录并回退 FP8/BF16，而不是伪造结果。
- FP8/NVFP4 的第一次调用包含状态、Kernel 选择或编译成本，因此要预热。
- 单层 Relative L2 合格不等于模型质量合格；还需校准、收敛与任务评测。
- RTX 4090 的 FP8 测试不等价于 H100 的训练系统，也不等价于 B200 的 NVFP4。
- 双 RTX 3080 20GB 可以完整运行 Level 0、TF32/FP16/BF16 与 AMP；FP8/FP4 仅保留能力探测和回退说明。

## 优化前后对照

| 维度 | 优化前 | 优化后 | 必须验证 |
| --- | --- | --- | --- |
| FP32 GEMM | 强制 IEEE FP32 | 容差允许时启用 TF32 | 误差、Tensor Pipe、端到端 |
| 训练前向 | 全部 FP32 | Autocast 按算子选择 FP16/BF16 | Loss、显存、Step time |
| FP16 梯度 | 直接 backward | 动态 GradScaler | Inf/NaN、跳步、收敛 |
| 精度选择 | 全模型统一 dtype | 敏感算子 FP32，GEMM 低精度 | 模型质量门禁 |
| FP8 | 手工 cast | TE recipe 管理 Amax/scale | 实际 Kernel、误差、吞吐 |
| FP4 | 把任意 4-bit 当 NVFP4 | Blackwell + NVFP4 recipe | CC、软件栈、质量 |
| Shape | 任意小而碎的 GEMM | 合批、padding、grouped GEMM | 有用计算比、P99 |
| 数据流 | 每层反复 cast/scale | 保持低精度并融合 Epilogue | 转换流量、Kernel 数 |
| 性能结论 | 对比宣传峰值 | 同口径实测 MFU/Goodput | dtype、稀疏口径、功耗 |

## 常见错误与排查

### 错误 1：Tensor 是 FP16，所以一定使用 Tensor Core

\*\*原因\*\*：算子、Shape、布局或库算法可能走普通管线。

\*\*排查\*\*：用 Profiler 确认 Kernel，再用 Nsight Compute 查看 Tensor Pipe 和指令路径。

### 错误 2：TF32 是一种 19-bit 存储格式

\*\*原因\*\*：混淆内部计算路径与 Tensor 存储 dtype。

\*\*排查\*\*：PyTorch Tensor 仍是 FP32；分别记录输入存储和后端 `fp32_precision` 设置。

### 错误 3：BF16 一定比 FP16 精确

\*\*原因\*\*：BF16 范围更大，但 fraction 更短。

\*\*排查\*\*：分别测试溢出率、相对误差和模型收敛，不用一个“精度高低”概括两条轴。

### 错误 4：BF16 训练照搬 FP16 GradScaler

\*\*原因\*\*：BF16 通常不需要用 Loss Scaling 解决 FP16 式范围问题。

\*\*排查\*\*：按 AMP dtype 启用 scaler，并观察梯度和框架建议。

### 错误 5：低精度峰值翻倍，模型一定翻倍

\*\*原因\*\*：非 GEMM、内存、通信、转换和 Shape 效率限制端到端收益。

\*\*排查\*\*：用 Amdahl 计算上界，测完整 Step/请求而不是单 GEMM。

### 错误 6：打开 TF32 后结果完全不变

\*\*可能原因\*\*：矩阵太小、未使用 cuBLAS 路径、当前 PyTorch 默认/设置不同，或数据恰好对舍入不敏感。

\*\*排查\*\*：输出设置 API、扩大 Shape、使用随机宽动态范围输入、比较相对误差并检查 Kernel。

### 错误 7：FP8 只要 dtype 支持就能训练

\*\*原因\*\*：缺少 scale、Amax、配方、形状约束和高精度主状态。

\*\*排查\*\*：使用成熟 TE/框架路径，记录 recipe、版本与校准状态。

### 错误 8：RTX 4090 是 Blackwell，可以跑 NVFP4

\*\*纠正\*\*：RTX 4090 是 Ada Lovelace，CC 8.9；NVFP4 是 Blackwell 相关路径。

### 错误 9：RTX 5090 与 B200 的所有 Kernel 通用

\*\*原因\*\*：同属 Blackwell 不代表 CC、资源和架构专用目标完全相同。

\*\*排查\*\*：分别构建目标，运行时能力检测，并保留 FP16/BF16/FP8 回退。

### 错误 10：AMP 出现 NaN 就降低 Learning Rate

\*\*原因\*\*：问题可能是 FP16 溢出、Scaler、敏感归约或非法输入。

\*\*排查\*\*：先定位第一个非有限 Tensor，检查 scaler 状态，尝试 BF16/局部 FP32，再调整优化超参数。

### 错误 11：CPU 回退结果证明 Tensor Core 提速

\*\*原因\*\*：CPU 的 BF16/FP16 实现、向量单元和 Kernel 完全不同。

\*\*排查\*\*：CPU 只用于数值与 API 流程，硬件吞吐结论必须在目标 GPU 上测量。

### 错误 12：实验 OOM

`N=4096` 的多个矩阵和输出会占用显存，训练 MLP 还包含优化器状态。资源不足时：

```
python3 tensor_core_lab.py --torch \
  --matrix-n 1024 --repeat 5 \
  --train-steps 10 --hidden 512 --batch 64
```

## 面试题与答案

### 1\. Tensor Core 与 CUDA Core 的主要区别是什么？

Tensor Core 面向固定形状矩阵乘累加，支持多种低精度输入和较高精度累加；普通 CUDA Core 执行更通用的标量/向量指令。一个 AI Kernel 往往同时使用两类管线。

### 2\. 为什么低精度可能同时提高计算和内存性能？

更少的位数让同样带宽搬运更多元素，也让专用矩阵单元每周期执行更多乘法。但 scale、cast、布局和误差控制会增加成本，收益不是自动的。

### 3\. TF32 和 FP32 有什么关系？

TF32 使用 FP32 的 exponent 范围和约 10 位 fraction 精度作为 Tensor Core 输入路径，通常以 FP32 累加。Tensor 在内存中仍可存为 FP32，它不是常规模型存储 dtype。

### 4\. BF16 与 FP16 如何选择？

BF16 范围大、训练通常更稳定；FP16 fraction 更长，但范围小，训练常需 Loss Scaling。最终还应考虑 GPU 支持、Kernel 性能、模型收敛和推理生态。

### 5\. Loss Scaling 为什么不改变理想梯度？

Loss 和梯度先乘 S，优化器更新前再除 S，数学上抵消。它让小梯度在 FP16 可表示范围内参与计算，同时需要检测放大后溢出。

### 6\. FP8 E4M3 与 E5M2 的权衡是什么？

E4M3 有更多 fraction 位，精度较好但范围较小；E5M2 有更多 exponent 位，范围更大但精度更低。训练配方可在前向和反向选择不同格式。

### 7\. 为什么 FP8 需要 scale？

FP8 范围和精度有限，原始分布可能溢出或大量挤在少数字段。Scale 把局部 Amax 映射到可表示范围，提高有效利用率；粒度越细，精度通常越好，但 metadata 与计算成本越高。

### 8\. NVFP4 与 INT4 是否相同？

不同。NVFP4 使用浮点值和分层缩放，并对应 Blackwell Tensor Core 路径；INT4 是整数编码配合 scale/zero point。两者表示集合、Kernel 和精度行为都不同。

### 9\. 怎样确认 GEMM 使用了 Tensor Core？

先确认 dtype/Shape/库 Kernel，再通过 Nsight Compute 查看 Tensor Pipe、相关 MMA 指令和 Speed of Light 指标。只看 API 参数或 GPU Utilization 不够。

### 10\. 为什么 Shape 影响 Tensor Core 利用率？

硬件按 Tile 处理矩阵。不规则或很小的 M/N/K 会产生尾块、padding、负载不足和更多调度开销，使有用计算占比下降。

### 11\. 混合精度训练为什么显存不一定减半？

训练可能仍保留 FP32 主权重、梯度、Optimizer State 和 Workspace。低精度主要减少部分模型副本、激活和算子流量，具体取决于实现。

### 12\. 低精度优化的最终验收指标是什么？

满足质量与稳定性门禁后的端到端 Goodput，例如达到目标 P99/精度的 tokens/s 或 samples/s，同时记录成本、功耗和失败率。

## 课后练习

1. 修改 Level 0 的输入指数范围，找出 FP16 开始大量 Inf 和 0 的区间。
2. 分别用高斯、均匀、长尾分布比较 INT8/INT4 scale 与误差。
1. 把 INT4 量化从 per-tensor 改为每 32 个值一组，比较误差和 scale metadata。
2. 为 $M=1,N=4096,K=4096$ 与 $M=1024,N=4096,K=4096$ 计算 Shape padding 和算术强度差异。
1. 在 RTX 3080/3090 上比较 IEEE FP32、TF32、FP16、BF16 Matmul，并记录 CC/版本/Shape。
2. 扫描 `matrix_n={127,128,255,256,511,512,1024,2048}` ，画出时间和有效 TFLOPS。
1. 在最小训练实验中强制 FP16、关闭 GradScaler，构造溢出并定位第一个非有限值。
2. 比较 BF16 与 FP16 AMP 的 Step time、峰值显存和 Loss 曲线。
1. 用 PyTorch Profiler 统计训练 Step 中 GEMM 占比，再计算低精度理论端到端上界。
2. 若有 Ada/Hopper/Blackwell，运行 TE FP8；若有 Blackwell，再运行 NVFP4，并完成质量门禁表。

## Checklist

### 概念

- 能解释 MMA： $D=A\times$ B+C。
- 能区分存储、乘法、累加和输出精度。
- 知道 TF32 是 FP32 操作的内部计算路径，不是常规存储 dtype。
- 能解释 BF16 与 FP16 的范围/精度权衡。
- 能解释 FP8 Amax、scale 和 recipe。
- 不把 NVFP4 与普通 INT4 混为一谈。

### 性能

- 会计算 GEMM 的 2MNK FLOPs 与有效 TFLOPS。
- 会计算 Shape padding 的有用工作比例。
- 会用 Amdahl 估算端到端加速上界。
- 会检查隐式 cast、layout transform 和 fallback。
- 不使用错误的 dtype/稀疏峰值作为 MFU 分母。

### 数值稳定性

- 同时记录相对误差、Inf/NaN、Loss 和最终任务质量。
- FP16 训练正确使用动态 GradScaler。
- BF16 不机械照搬 FP16 Loss Scaling。
- 对敏感归约和损失保留更高精度候选路径。
- 低精度优化具有可自动执行的质量回归门禁。

### 架构边界

- Ampere：RTX 3080/3090、A100；无 FP8 Tensor Core。
- Ada：RTX 4090、L40/L40S；不是 Blackwell。
- Hopper：H100/H200；可验证 FP8 Transformer Engine。
- Blackwell：RTX 5090、B100/B200/GB200；可检查 FP4/FP6，但产品目标需区分。
- 无对应硬件时只做数值模拟和回退，不伪造 Tensor Core 实测。

### 实验

- 已运行 Level 0，并保存 JSON。
- 有 PyTorch 时已运行通用 Matmul。
- 有 CUDA 时已比较 IEEE FP32 与 TF32。
- 已完成至少一种 FP16/BF16 AMP 训练测试。
- 性能结果包含 GPU、CC、驱动/CUDA、PyTorch、Shape 和 dtype。