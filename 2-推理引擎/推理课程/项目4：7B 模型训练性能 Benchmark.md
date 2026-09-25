---
title: "项目4：7B 模型训练性能 Benchmark"
source: "https://everythingai.top/learn/ai-systems-performance-engineering-text/project-04"
author:
published:
created: 2026-09-26
description: "面向 AI 工程学习者的专业课程学习空间。"
tags:
  - "clippings"
---
**导航**

本项目为 7B 模型训练构建标准 Benchmark，形成可比较的吞吐、显存、MFU 与稳定性报告。

## 项目定位

本项目建立一套可审计的 7B 级自回归模型训练 Benchmark。重点不是下载某个品牌模型后跑出一个数字，而是把模型规模、Token 工作量、训练状态内存、并行策略、MFU、吞吐和稳定性放进同一张证据表。

项目采用两条路径：

- 通用路径：用纯 Python 容量规划器和小型 GPT 代理模型验证方法，CPU 或任意 PyTorch 设备可执行；
- 真实 7B 路径：在容量足够的多 GPU 环境中使用同一模型结构和报告口径，配合 FSDP、Megatron Core 或其他经过验证的分布式栈。

双 RTX 3080 20GB 可以完成代理实验和并行策略测量，但不能把“2×20GB 总显存”简单视为可训练 7B 的 40GB 连续显存。7B 全参数 AdamW 训练还需要参数、梯度、主权重、优化器状态、激活、通信缓冲区与碎片空间；即使完全分片，双卡通常仍不够。课程明确提供回退实验，不伪造 7B 实测。

## 学习目标

- 从参数量和精度估算训练静态状态内存；
- 从 Token 数、参数量和时间估算模型 FLOPs、Tokens/s 与 MFU；
- 区分理论 FLOPs、硬件峰值、实际训练 FLOPs 与有效训练吞吐；
- 设计具有热身、稳态、正确性和尾部抖动的 Benchmark；
- 理解 DDP、FSDP、TP、PP 对内存和通信的不同影响；
- 在消费级 GPU 上用缩小模型验证方法，在数据中心集群上扩展到真实 7B；
- 避免跨框架、跨精度、跨序列长度直接比较没有上下文的数字。

## 前置知识

- 第 15、17、18、19 课；
- Transformer、Causal Attention、梯度累积与混合精度；
- PyTorch 2.x；真实多 GPU 实验需要 `torchrun` 、NCCL 与匹配的 CUDA 环境；
- 真实 7B 实验建议使用成熟分布式训练栈，并锁定提交版本。

## 项目交付物

```
7b-training-benchmark/
├── capacity_planner.py
├── gpt_proxy_benchmark.py
├── reports/
│   ├── environment.json
│   ├── capacity.json
│   ├── proxy_single_gpu.json
│   ├── proxy_multi_gpu.json
│   └── benchmark_report.md
└── checkpoints/
```

## 核心直觉：先证明“放得下”，再讨论“跑得快”

混合精度 AdamW 的粗略静态状态可以按每参数 16 字节估算：

$$
M_{state}\approx P\times(2_{param}+2_{grad}+4_{master}+8_{Adam})
$$

对 70 亿参数：

$$
7\times10^9\times16\ Bytes\approx112\ GB
$$

这还没有包含激活、临时张量、通信 Bucket、CUDA Context 和碎片。DDP 在每个 Rank 复制完整状态；理想完全分片只把可分片部分除以数据并行数，实际还会有峰值 All-Gather、未分片参数和缓冲区。

因此容量判断应保留安全余量：

$$
M_{required}\le M_{device}\times U_{safe}
$$

其中 U\\\_{safe} 通常不能写成 100%，而应由框架、模型和测量决定。

## 训练计算模型

Decoder-only Transformer 预训练常用近似：

$$
FLOPs_{step}\approx6PT
$$

P 为参数量，T 为一次参数更新处理的 Token 数。这个近似用于容量规划和跨运行口径统一，不包含所有注意力二次项、Embedding、重计算和通信。

若一步耗时 t：

$$
Tokens/s=\frac{T}{t}
$$

$$
Achieved\ FLOPs=\frac{6PT}{t}
$$

$$
MFU=\frac{Achieved\ FLOPs}{N_{GPU}\times Peak\ FLOPs_{per\ GPU}}
$$

峰值必须与实际精度和稀疏条件一致。不能用 FP8/稀疏峰值计算 BF16 Dense 训练 MFU，也不能把消费卡的 Boost 理论值当作持续功耗下的实测上限。

## Level 0：7B 容量与性能规划器

保存为 `capacity_planner.py` ：

```
#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

GIB = 1024 ** 3

def gib(value):
    return round(value / GIB, 3)

def main():
    parser = argparse.ArgumentParser(description="7B 训练容量与 MFU 规划器")
    parser.add_argument("--params", type=float, default=7e9)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--gpu-memory-gib", type=float, default=20.0)
    parser.add_argument("--safe-utilization", type=float, default=0.85)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--data-parallel", type=int, default=2)
    parser.add_argument("--step-seconds", type=float, default=1.0)
    parser.add_argument("--peak-tflops-per-gpu", type=float)
    parser.add_argument("--output", default="reports/capacity.json")
    args = parser.parse_args()

    positive = [args.params, args.gpus, args.gpu_memory_gib, args.micro_batch,
                args.sequence_length, args.grad_accum, args.data_parallel,
                args.step_seconds]
    if any(value <= 0 for value in positive):
        raise ValueError("所有规模、时间与峰值参数必须大于 0")
    if not 0 < args.safe_utilization <= 1:
        raise ValueError("safe-utilization 必须位于 (0, 1]")
    if args.peak_tflops_per_gpu is not None and args.peak_tflops_per_gpu <= 0:
        raise ValueError("peak-tflops-per-gpu 必须大于 0")

    param_bytes = args.params * 2
    grad_bytes = args.params * 2
    master_bytes = args.params * 4
    adam_bytes = args.params * 8
    total_state = param_bytes + grad_bytes + master_bytes + adam_bytes
    ddp_per_gpu = total_state
    ideal_fsdp_per_gpu = total_state / args.data_parallel
    safe_capacity = args.gpu_memory_gib * GIB * args.safe_utilization

    tokens_per_step = (args.micro_batch * args.sequence_length *
                       args.grad_accum * args.data_parallel)
    flops_per_step = 6 * args.params * tokens_per_step
    achieved_tflops = flops_per_step / args.step_seconds / 1e12
    cluster_peak_tflops = (
        args.gpus * args.peak_tflops_per_gpu
        if args.peak_tflops_per_gpu is not None else None
    )
    mfu = (
        achieved_tflops / cluster_peak_tflops
        if cluster_peak_tflops is not None else None
    )

    result = {
        "parameters": args.params,
        "training_state_gib": {
            "parameters_bf16": gib(param_bytes),
            "gradients_bf16": gib(grad_bytes),
            "master_weights_fp32": gib(master_bytes),
            "adam_moments_fp32": gib(adam_bytes),
            "total_before_activations": gib(total_state),
        },
        "per_gpu_estimate_gib": {
            "ddp": gib(ddp_per_gpu),
            "ideal_full_shard": gib(ideal_fsdp_per_gpu),
            "safe_capacity": round(args.gpu_memory_gib * args.safe_utilization, 3),
        },
        "fits_before_activations": {
            "ddp": ddp_per_gpu <= safe_capacity,
            "ideal_full_shard": ideal_fsdp_per_gpu <= safe_capacity,
        },
        "workload": {
            "tokens_per_step": tokens_per_step,
            "tokens_per_second": tokens_per_step / args.step_seconds,
            "approx_flops_per_step": flops_per_step,
            "achieved_tflops": achieved_tflops,
            "cluster_peak_tflops": cluster_peak_tflops,
            "mfu": mfu,
        },
        "warning": (
            "静态状态估算不含激活、通信缓冲、临时张量、CUDA Context、"
            "碎片和峰值 All-Gather；MFU 依赖正确的同精度 Dense 峰值。"
        ),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

if __name__ == "__main__":
    main()
```

### 运行命令

双 20GB GPU 的容量检查：

```
python capacity_planner.py \
  --params 7e9 --gpus 2 --data-parallel 2 --gpu-memory-gib 20 \
  --micro-batch 1 --sequence-length 2048 --grad-accum 16 \
  --step-seconds 10 \
  --output reports/capacity_2x20g.json
```

未提供峰值时，规划器把 `cluster_peak_tflops` 与 `mfu` 输出为 `null` ，避免制造无意义数字。请从对应 GPU 官方规格选择与实际训练精度一致的 Dense 峰值，再加入 `--peak-tflops-per-gpu` 。

### 预期现象

7B 混合精度 AdamW 静态状态约为 104 GiB（十进制 112 GB）。理想二路完全分片仍约 52 GiB/卡，尚未计算激活，因此双 20GB 卡不会被判定为可行。这个结论来自容量模型，不是 OOM 实测；CPU Offload、8-bit Optimizer、LoRA 等会改变问题定义，必须单列结果。

## Level 1：规模可调的 GPT 代理 Benchmark

下面的模型保留 Causal Attention、RMSNorm、SwiGLU 和语言模型损失。默认配置适合 CPU 或常见 GPU； `--preset 7b` 只在经过容量规划的环境中启用。

保存为 `gpt_proxy_benchmark.py` ：

```
#!/usr/bin/env python3
import argparse
import json
import math
import os
import platform
import statistics
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True)
        return (x * torch.rsqrt(norm + self.eps).to(x.dtype)) * self.weight

class Block(nn.Module):
    def __init__(self, hidden, heads, intermediate):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden 必须能被 heads 整除")
        self.hidden = hidden
        self.heads = heads
        self.head_dim = hidden // heads
        self.attn_norm = RMSNorm(hidden)
        self.qkv = nn.Linear(hidden, 3 * hidden, bias=False)
        self.proj = nn.Linear(hidden, hidden, bias=False)
        self.ffn_norm = RMSNorm(hidden)
        self.gate = nn.Linear(hidden, intermediate, bias=False)
        self.up = nn.Linear(hidden, intermediate, bias=False)
        self.down = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        batch, seq, _ = x.shape
        h = self.attn_norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        def shape(tensor):
            return tensor.view(batch, seq, self.heads, self.head_dim).transpose(1, 2)
        q, k, v = shape(q), shape(k), shape(v)
        attention = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attention = attention.transpose(1, 2).contiguous().view(batch, seq, self.hidden)
        x = x + self.proj(attention)
        h = self.ffn_norm(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))

class GPT(nn.Module):
    def __init__(self, vocab, layers, hidden, heads, intermediate):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.blocks = nn.ModuleList([
            Block(hidden, heads, intermediate) for _ in range(layers)
        ])
        self.norm = RMSNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, tokens, targets):
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.norm(x))
        return F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))

def percentile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return values[low]
    return values[low] * (high - position) + values[high] * (position - low)

def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def main():
    parser = argparse.ArgumentParser(description="GPT 训练代理 Benchmark")
    parser.add_argument("--preset", choices=["proxy", "7b"], default="proxy")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--output", default="reports/proxy.json")
    args = parser.parse_args()

    if args.preset == "7b":
        config = dict(vocab=32000, layers=32, hidden=4096, heads=32, intermediate=11008)
    else:
        config = dict(vocab=2048, layers=4, hidden=256, heads=8, intermediate=768)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但当前不可用")
    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(2026)

    model = GPT(**config).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    if args.compile:
        model = torch.compile(model)

    amp_enabled = args.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)
    times = []
    losses = []
    total_steps = args.warmup + args.steps
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for step in range(total_steps):
        sync(device)
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for _ in range(args.grad_accum):
            tokens = torch.randint(
                0, config["vocab"],
                (args.batch_size, args.sequence_length + 1),
                device=device,
            )
            inputs, targets = tokens[:, :-1], tokens[:, 1:]
            context = (
                torch.autocast("cuda", dtype=amp_dtype)
                if amp_enabled else nullcontext()
            )
            with context:
                loss = model(inputs, targets) / args.grad_accum
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accumulated_loss += float(loss.detach().cpu())
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        sync(device)
        elapsed = time.perf_counter() - start
        if step >= args.warmup:
            times.append(elapsed)
            losses.append(accumulated_loss)

    tokens_per_step = args.batch_size * args.sequence_length * args.grad_accum
    p50 = percentile(times, 0.50)
    approx_flops = 6 * parameter_count * tokens_per_step
    result = {
        "preset": args.preset,
        "config": config,
        "parameters": parameter_count,
        "device": str(device),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "grad_accum": args.grad_accum,
        "tokens_per_step": tokens_per_step,
        "p50_step_seconds": p50,
        "p95_step_seconds": percentile(times, 0.95),
        "tokens_per_second": tokens_per_step / p50,
        "approx_achieved_tflops": approx_flops / p50 / 1e12,
        "final_loss": losses[-1],
        "amp": amp_enabled,
        "compile": args.compile,
        "peak_memory_mib": (
            torch.cuda.max_memory_allocated(device) / 2**20
            if device.type == "cuda" else None
        ),
        "host": platform.platform(),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

if __name__ == "__main__":
    main()
```

### 运行命令

CPU 回退：

```
python gpt_proxy_benchmark.py --device cpu --preset proxy \
  --batch-size 1 --sequence-length 64 --warmup 1 --steps 3 \
  --output reports/proxy_cpu.json
```

常见 NVIDIA GPU：

```
python gpt_proxy_benchmark.py --device cuda --preset proxy --amp \
  --batch-size 4 --sequence-length 512 --warmup 5 --steps 20 \
  --output reports/proxy_gpu.json

python gpt_proxy_benchmark.py --device cuda --preset proxy --amp --compile \
  --batch-size 4 --sequence-length 512 --warmup 10 --steps 20 \
  --output reports/proxy_gpu_compile.json
```

默认脚本是单进程 Benchmark。不要仅把它放进 `torchrun` 就宣称获得 DDP/FSDP；多进程训练必须显式初始化 Process Group、包装模型、使用分布式采样，并按全局 Token 数计算吞吐。

## Level 2：真实 7B 扩展方案

### 并行策略选择

| 条件 | 首选起点 | 主要代价 |
| --- | --- | --- |
| 单卡能容纳全部状态 | 单 GPU | 无通信，但容量受限 |
| 每卡能容纳完整状态 | DDP | All-Reduce，状态复制 |
| 完整状态放不下 | FSDP/ZeRO | All-Gather、Reduce-Scatter、峰值内存 |
| 单层参数也难容纳 | TP | 高频张量通信 |
| 层数多、跨节点扩展 | PP | Pipeline Bubble、调度复杂 |
| 大规模集群 | DP+TP+PP+SP/CP | 配置与拓扑耦合 |

真实 7B Benchmark 推荐在锁定版本的 PyTorch FSDP2 或 Megatron Core 中运行。课程不复制不断变化的全部启动参数；无论框架如何，必须固定并记录：

- 模型参数量与配置；
- Sequence Length、Micro Batch、Gradient Accumulation、Global Batch；
- 精度、Activation Checkpoint、Optimizer 与分片策略；
- DP/TP/PP/CP 数量及物理拓扑；
- 热身步、测量步、Checkpoint/Validation 是否计入；
- Tokens/s、Step P50/P95、MFU、峰值显存、通信占比与失败步。

### 双 RTX 3080 20GB 的有效回退

1. 使用代理配置扫描 Sequence Length 和 Batch；
2. 用 DDP 测 All-Reduce 暴露程度；
1. 用 FSDP 包装代理模型验证分片与 Checkpoint 流程；
2. 把代理模型参数量、每步 Token 和测得 TFLOP/s写入报告；
1. 只把容量规划器用于外推，不把外推值写成 7B 实测。

RTX 3080/3090 属于 Ampere。RTX 4090 属于 Ada Lovelace；RTX 5090 属于 Blackwell。H100/H200 才属于 Hopper。不同架构的 BF16/FP8、互联、显存容量和峰值不同，不能直接迁移 MFU。

## Benchmark 方法

### 受控变量

- 同一模型代码和初始化；
- 同一 Token 数和 Global Batch；
- 同一精度与损失；
- 同一功耗、时钟策略和 GPU 拓扑；
- 同一编译器、驱动、CUDA、NCCL、PyTorch/训练栈版本；
- 同一是否包含数据、日志、Checkpoint 和验证的口径。

### 运行阶段

```
环境记录 → 容量模型 → 正确性 Smoke Test → 热身
→ 稳态 20～100 步 → Profiler 窗口 → 重复运行
→ 中位数/尾部统计 → 回归对比 → 保存证据
```

### 结果表

| 运行 | Params | GPUs | DP/TP/PP | Seq | Global Tokens/Step | P50 s | Tokens/s | MFU | Peak GiB | 有效步率 | | -------------------------------------------------------------------------------- | -- | -- | ----- | -- | -- | -- | -- | -- | -- | -- | | Proxy-A | 实测 | 1 | 1/1/1 | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 | | Proxy-B | 实测 | 2 | 2/1/1 | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 | | 7B | 实测 | 集群 | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 |

## 瓶颈分析

- OOM：先拆分静态状态、激活、通信峰值与碎片，不要只减 Batch；
- MFU 低且 GPU 空洞多：检查数据、CPU Launch、同步和 Pipeline Bubble；
- GEMM 时间长但利用率低：检查 Shape、精度、Tensor Core 对齐与并行切分；
- 通信占比高：检查拓扑、Bucket、重叠、TP/PP 切分与慢 Rank；
- P95 远高于 P50：检查热降频、日志、Checkpoint、数据尾部和网络抖动；
- Scale-out 变慢：计算强扩展效率，确认每卡工作量是否过小。

强扩展效率：

$$
E_N=\frac{Throughput_N}{N\times Throughput_1}
$$

`Throughput_1` 必须来自能运行同一工作负载的单卡基线；若 7B 单卡根本放不下，不应伪造单卡效率。

## 常见错误与排查

### 1\. 把模型权重大小当训练总显存

权重只是训练状态的一部分。必须加入梯度、主权重、优化器、激活、通信缓冲与峰值分片操作。

### 2\. MFU 大于 100%

通常是 Token、参数、GPU 数、时间单位或峰值精度口径错误。检查是否把稀疏/FP8 峰值用于 BF16 Dense。

### 3\. 代理模型很快，所以外推 7B 也很快

代理模型用于验证流程。大模型会改变 GEMM Shape、通信、激活、缓存、并发和编译行为，不能线性外推实测吞吐。

### 4\. 用峰值显存之和判断可行

多卡显存不是默认共享池。DDP 复制状态；FSDP/TP 也有临时 All-Gather 和非分片部分。

### 5\. 只测三步

首步可能包含初始化、Allocator、JIT/Compile 和通信连接。应独立报告冷启动，热身后再统计稳态。

### 6\. 只报告 Samples/s

语言模型的 Sample 可包含不同 Token 数。应优先报告 Tokens/s，并记录序列长度、Padding 与有效 Token。

### 7\. Checkpoint 导致尾延迟却被忽略

明确区分“纯训练 Step”与“生产端到端”口径。生产 Goodput 应包含定期 Checkpoint、验证和故障恢复的影响。

## 优化前后对照

| 维度 | 基线 | 优化后 | 验证证据 |
| --- | --- | --- | --- |
| 精度 | FP32/BF16 固定 | 合法低精度 | Loss、NaN、吞吐 |
| 激活 | 全保存 | Checkpoint | 显存、重计算时间 |
| 状态 | 复制 | FSDP/ZeRO | 每卡峰值、通信 |
| 计算 | Eager | Compile/Fusion | 冷启动、稳态、Graph Break |
| 通信 | 暴露 | Bucket/Overlap | Trace、Collective 占比 |
| 数据 | 同步供给 | Prefetch/Pinned | GPU 空洞、P95 |

## 面试题与答案

### 1\. 7B BF16 权重约 14GB，为什么 20GB GPU 不能直接全参数训练？

训练还需要梯度、FP32 主权重、Adam 一阶/二阶状态、激活和临时缓冲，静态状态就可能超过 100GB。

### 2\. FSDP 为什么不是简单地把总显存除以 GPU 数？

运行时存在参数 All-Gather、Reduce-Scatter、未分片对象、激活、通信缓冲和碎片，峰值取决于 Wrap、Prefetch 与执行顺序。

### 3\. MFU 的分子是什么？

通常用模型参数、每步 Token 和训练计算近似得到实际每秒模型 FLOPs；必须写明所用公式和适用边界。

### 4\. 为什么 Tokens/s 比 Samples/s 更适合 LLM？

不同 Sample 可能包含不同长度和 Padding，Token 更接近实际计算工作量。

### 5\. DDP、FSDP 与 TP 的核心差异？

DDP 复制模型并同步梯度；FSDP 分片训练状态并在执行时收集参数；TP 把单层张量计算切到多卡并进行高频通信。

### 6\. 如何公平比较两个 7B 训练框架？

固定模型、Token、Global Batch、精度、优化器、Checkpoint/日志口径、硬件与版本，并同时比较正确性、吞吐、P95、显存和稳定性。

## 课后练习

1. 用规划器比较 DDP、2/4/8 路理想完全分片的静态状态。
2. 扫描代理模型 Sequence Length，观察注意力成本与显存曲线。
1. 扫描梯度累积，在 Global Tokens/Step 不变时比较 P50。
2. 给代理模型加入 Activation Checkpoint，测量显存—时间交换。
1. 在双 GPU 上实现 DDP 版本，报告全局 Tokens/s 和扩展效率。
2. 用 FSDP 包装每个 `Block` ，记录峰值内存和 Collective 时间。
1. 把 Checkpoint 写盘纳入每 N 步端到端 Goodput。
2. 分别用 BF16 与 FP16 AMP，比较数值稳定性和吞吐。

## 项目验收 Checklist

- 已记录模型真实参数量，而不是仅使用“7B”标签；
- 已完成静态训练状态与安全显存估算；
- 已明确激活、缓冲、碎片与峰值分片操作未包含在粗估中；
- 已固定 Global Tokens/Step、精度和优化器；
- 已区分冷启动、热身和稳态；
- 已报告 P50/P95、Tokens/s、Loss、峰值显存和有效步率；
- MFU 使用与实际精度一致的 Dense 峰值；
- CPU/代理实验没有被冒充为真实 7B 实测；
- 双消费卡不可行时采用代理回退，没有伪造共享显存；
- 真实多 GPU 运行记录了 DP/TP/PP/CP 和拓扑；
- 保存了环境、命令、配置、Profiler 与原始 JSON。