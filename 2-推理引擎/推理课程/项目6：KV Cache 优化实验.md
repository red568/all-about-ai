---
title: "项目6：KV Cache 优化实验"
source: "https://everythingai.top/learn/ai-systems-performance-engineering-text/project-06"
author:
published:
created: 2026-09-26
description: "面向 AI 工程学习者的专业课程学习空间。"
tags:
  - "clippings"
---
**导航**

本项目通过可运行实验分析 KV Cache 的容量、带宽与调度瓶颈，并验证分页和复用策略。

## 项目定位

本项目把 KV Cache 从一个公式变成可测量、可调度、可回归的系统资源。你将完成容量估算、连续分配与分页分配对照、Block Size 扫描、共享前缀命中实验，以及真实服务中的 KV 容量、抢占、TTFT 和吞吐分析。

```
模型结构 → 每 Token KV 字节 → 理论 Token 容量
→ 分页与碎片模拟 → 服务端 KV 配置
→ 重复前缀负载 → 命中/未命中对照
→ 并发与上下文扫描 → SLO、抢占、显存回归
```

Level 0 纯 Python 可运行。Level 1 适配常见 NVIDIA GPU 上的 vLLM。Level 2 的 KV 量化、跨 GPU/NVLink、Host Offload 与 Disaggregated KV Transfer 必须在真实支持的硬件和框架版本上验证。

## 学习目标

- 从层数、KV Head、Head Dimension、精度计算每 Token KV 字节；
- 区分理论容量、可用容量、内部碎片、外部碎片和运行峰值；
- 理解 Paged KV Cache 如何把请求生命周期映射到固定大小 Block；
- 解释 MHA、GQA、MQA 对 KV 容量的影响；
- 构造有命中和无命中的 Prefix Cache A/B 实验；
- 用服务日志识别 Preemption、Recompute、Swap、OOM 与低命中率；
- 同时优化容量、TTFT、TPOT、吞吐和正确性，而不是只追求显存占满。

## 前置知识

- 第 23～25 课与项目 5；
- Transformer Attention、Continuous Batching、流式推理指标；
- Python 3.10+；真实服务实验需要 vLLM 和可用模型。

## 项目交付物

```
kv-cache-project/
├── kv_cache_lab.py
├── prefix_cache_benchmark.py
├── reports/
│   ├── capacity.json
│   ├── fragmentation.json
│   ├── prefix_off.json
│   ├── prefix_on.json
│   └── analysis.md
└── logs/
    ├── server_prefix_off.log
    └── server_prefix_on.log
```

## 核心直觉：KV Cache 是“按活跃 Token 计费”的显存池

对 Decoder-only Transformer，单 Token 的 KV Cache 粗略字节数：

$$
Bytes_{token}=2\times L\times H_{kv}\times D_{head}\times B_{dtype}
$$

- 2：Key 与 Value；

L：层数；

$H_{kv}$ ：KV Head 数；

$D_{head}$ ：每个 Head 的维度；

B\\\_{dtype}：每元素字节数。

请求长度为 S，并发为 C：

$$
M_{KV}\approx C\times S\times Bytes_{token}
$$

这个公式没有包含 Block 元数据、Padding、Allocator、CUDA Graph、工作区和碎片。真实可用 Token 容量应由服务启动日志与实测共同确认。

GQA/MQA 通过减少 $H_{kv}$ 显著降低 KV Cache，但不会把 Query Head 的计算量按同样比例消除。

## 分页与碎片模型

连续预留若按最大上下文 $S_{max}$ 为每个请求分配：

$$
Waste_{contiguous}=\sum_i(S_{max}-S_i)
$$

分页分配 Block Size 为 K：

$$
Reserved_i=\left\lceil\frac{S_i}{K}\right\rceil K
$$

$$
Waste_{paged}=\sum_i(Reserved_i-S_i)
$$

较小 Block 减少内部碎片，但会增加 Block Table、调度和 Kernel 元数据成本。最佳 Block Size 是系统实验结果，不由碎片公式单独决定。

## Level 0：容量与分页模拟器

保存为 `kv_cache_lab.py` ：

```
#!/usr/bin/env python3
import argparse
import json
import math
import random
import statistics
from pathlib import Path

def percentile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return values[low]
    return values[low] * (high - position) + values[high] * (position - low)

def capacity(args):
    bytes_per_token = 2 * args.layers * args.kv_heads * args.head_dim * args.dtype_bytes
    usable_bytes = args.kv_memory_gib * (1024 ** 3) * args.safe_utilization
    token_capacity = math.floor(usable_bytes / bytes_per_token)
    request_capacity = math.floor(token_capacity / args.tokens_per_request)
    return {
        "bytes_per_token": bytes_per_token,
        "kib_per_token": bytes_per_token / 1024,
        "usable_kv_gib": args.kv_memory_gib * args.safe_utilization,
        "theoretical_token_capacity": token_capacity,
        "theoretical_request_capacity": request_capacity,
        "warning": "不含 Block 元数据、工作区、图捕获、Allocator 和运行峰值。",
    }

def fragmentation(args):
    rng = random.Random(args.seed)
    lengths = [rng.randint(args.min_tokens, args.max_tokens) for _ in range(args.requests)]
    contiguous_reserved = args.requests * args.max_tokens
    paged_reserved = sum(math.ceil(length / args.block_size) * args.block_size
                         for length in lengths)
    used = sum(lengths)
    return {
        "requests": args.requests,
        "block_size": args.block_size,
        "used_tokens": used,
        "contiguous_reserved_tokens": contiguous_reserved,
        "paged_reserved_tokens": paged_reserved,
        "contiguous_utilization": used / contiguous_reserved,
        "paged_utilization": used / paged_reserved,
        "paged_internal_fragmentation_tokens": paged_reserved - used,
        "length_p50": percentile(lengths, 0.50),
        "length_p95": percentile(lengths, 0.95),
        "mean_length": statistics.mean(lengths),
        "warning": "模拟器不包含真实请求生命周期、Prefix 共享、抢占或 Kernel 成本。",
    }

def main():
    parser = argparse.ArgumentParser(description="KV Cache 容量与分页实验")
    parser.add_argument("--mode", choices=["capacity", "fragmentation"], required=True)
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype-bytes", type=float, default=2)
    parser.add_argument("--kv-memory-gib", type=float, default=8)
    parser.add_argument("--safe-utilization", type=float, default=0.90)
    parser.add_argument("--tokens-per-request", type=int, default=4096)
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--min-tokens", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default="reports/kv_cache.json")
    args = parser.parse_args()
    numeric = [args.layers, args.kv_heads, args.head_dim, args.dtype_bytes,
               args.kv_memory_gib, args.tokens_per_request, args.requests,
               args.min_tokens, args.max_tokens, args.block_size]
    if any(value <= 0 for value in numeric):
        raise ValueError("容量与负载参数必须大于 0")
    if args.min_tokens > args.max_tokens:
        raise ValueError("min-tokens 不能大于 max-tokens")
    if not 0 < args.safe_utilization <= 1:
        raise ValueError("safe-utilization 必须位于 (0, 1]")

    result = capacity(args) if args.mode == "capacity" else fragmentation(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

if __name__ == "__main__":
    main()
```

### 运行容量实验

```
mkdir -p reports
python kv_cache_lab.py --mode capacity \
  --layers 32 --kv-heads 8 --head-dim 128 --dtype-bytes 2 \
  --kv-memory-gib 8 --tokens-per-request 4096 \
  --output reports/capacity.json
```

将 `kv-heads` 从 32 改为 8、再改为 1，对比 MHA、GQA、MQA 的理论容量。参数必须来自实际模型配置，不能从模型名称猜测。

### 运行碎片实验

```
for block in 8 16 32 64; do
  python kv_cache_lab.py --mode fragmentation \
    --requests 1000 --min-tokens 32 --max-tokens 4096 \
    --block-size "$block" \
    --output "reports/fragmentation_block_${block}.json"
done
```

### 预期现象

分页预留比按最大长度连续预留更接近实际使用量；Block 越小，模拟内部碎片通常越低。但模拟器没有计入元数据和 Kernel 成本，不能据此宣布最小 Block 一定最快。

## Level 1：真实 vLLM KV Cache 实验

### 环境记录

沿用项目 5 的隔离环境，并保存：

```
mkdir -p reports logs
{
  date -Iseconds
  nvidia-smi -L
  nvidia-smi topo -m
  python --version
  vllm --version
  vllm serve --help=all | rg "cache|block|swap|offload|max-model-len"
} 2>&1 | tee reports/environment.txt
```

### 基线服务

```
export MODEL_ID="your-org/your-model"
vllm serve "$MODEL_ID" \
  --host 127.0.0.1 --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  2>&1 | tee logs/server_prefix_off.log
```

记录启动日志中的 KV Cache 容量。不要从总显存减权重后直接宣称剩余全部属于 KV；Runtime 还需要工作区、CUDA Graph 和临时张量。

## Prefix Cache A/B 客户端

下面脚本生成“共享长前缀”和“随机前缀”两组请求。为了避免把网络并发混入命中实验，默认串行发送并比较热身后的端到端延迟。更完整的 TTFT/TPOT 应复用项目 5 的流式客户端。

保存为 `prefix_cache_benchmark.py` ：

```
#!/usr/bin/env python3
import argparse
import json
import math
import random
import statistics
import time
import urllib.request
from pathlib import Path

def percentile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return values[low]
    return values[low] * (high - position) + values[high] * (position - low)

def request(url, model, prompt, max_tokens, timeout):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = json.load(response)
    elapsed = time.perf_counter() - start
    usage = body.get("usage") or {}
    return elapsed, usage.get("prompt_tokens"), usage.get("completion_tokens")

def main():
    parser = argparse.ArgumentParser(description="Prefix Cache A/B Benchmark")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--prefix-words", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default="reports/prefix.json")
    args = parser.parse_args()
    if min(args.requests, args.prefix_words, args.max_tokens, args.timeout) <= 0:
        raise ValueError("负载参数必须大于 0")

    rng = random.Random(args.seed)
    shared_prefix = "统一系统知识：AI 性能工程需要可复现测量。 " * args.prefix_words
    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    shared_times, random_times = [], []
    token_rows = []
    for index in range(args.requests):
        suffix = f"请求 {index}：解释 KV Cache 的作用。"
        elapsed, prompt_tokens, completion_tokens = request(
            url, args.model, shared_prefix + suffix, args.max_tokens, args.timeout
        )
        shared_times.append(elapsed)
        token_rows.append((prompt_tokens, completion_tokens))

        nonce = " ".join(str(rng.getrandbits(64)) for _ in range(args.prefix_words))
        elapsed, _, _ = request(
            url, args.model, nonce + suffix, args.max_tokens, args.timeout
        )
        random_times.append(elapsed)

    result = {
        "requests_per_group": args.requests,
        "shared_prefix_latency_s": {
            "p50": percentile(shared_times, 0.50),
            "p95": percentile(shared_times, 0.95),
            "mean": statistics.mean(shared_times),
        },
        "random_prefix_latency_s": {
            "p50": percentile(random_times, 0.50),
            "p95": percentile(random_times, 0.95),
            "mean": statistics.mean(random_times),
        },
        "usage_samples": token_rows,
        "warning": "文本 Word 数不等于 Token 数；正式报告以服务 Usage/模型 Tokenizer 为准。",
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

if __name__ == "__main__":
    main()
```

### A/B 实验步骤

1. 关闭 Prefix Cache 启动服务，热身后运行：

```
export SERVED_MODEL="your-served-model-name"
python prefix_cache_benchmark.py --model "$SERVED_MODEL" \
  --output reports/prefix_off.json
```

1. 停止服务，使用当前版本帮助确认 Prefix Caching 参数名，然后开启：

```
vllm serve --help=all | rg "prefix"
vllm serve "$MODEL_ID" \
  --host 127.0.0.1 --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --enable-prefix-caching \
  2>&1 | tee logs/server_prefix_on.log
```

1. 使用相同模型、Prompt、请求顺序和输出长度复测：

```
python prefix_cache_benchmark.py --model "$SERVED_MODEL" \
  --output reports/prefix_on.json
```

首次共享 Prefix 请求用于填充缓存，不应与热命中请求混在一个结论中。正式实验应分别报告 Cold、Warm Hit 与 Random Miss。

## 并发与上下文容量扫描

复用项目 5 的流式 Benchmark，构造：

| 负载 | 输入 | 输出 | 目标 |
| --- | --- | --- | --- |
| Decode-heavy | 128 | 1024 | TPOT、KV 增长 |
| Prefill-heavy | 4096 | 64 | TTFT、Prefix 命中 |
| Long-context | 8192+ | 128 | 容量、抢占、OOM |
| Mixed | 分布 | 分布 | 生产 Goodput |

每种负载扫描并发，记录 Server 日志、GPU 显存、TTFT P95、TPOT P95、Token/s、成功率、Preemption 和 Cache Hit 指标。

## Level 2：专属实验与边界

### KV Cache 低精度

当前 vLLM 版本、模型与 GPU 若支持 KV Cache 量化，可做 FP8 KV A/B。必须先查看当前帮助与官方支持矩阵，验证模型质量和数值稳定性。FP8 KV 减少字节数不代表端到端必然加速；量化/反量化、Kernel 与带宽瓶颈都会影响结果。

Ampere RTX 3080/3090 不应被宣称为 Hopper FP8 实验；Ada RTX 4090 也不属于 Blackwell。H100/H200 属于 Hopper，RTX 5090/B100/B200/GB200 属于 Blackwell。FP4 KV 或计算只有在真实支持的硬件、框架和 Kernel 上才可验证。

### CPU Offload / Swap

Host KV 或 Swap 可能扩大容量，但引入 PCIe/NUMA/页锁定与尾延迟。实验必须报告：

- Host Memory 与 NUMA 绑定；
- PCIe/NVLink/NVSwitch 拓扑；
- 命中与换入换出字节；
- TTFT/TPOT P95/P99；
- CPU 内存压力和 OOM 风险。

CPU 模拟无法等价验证真实传输带宽和尾延迟。

### 多 GPU KV

Tensor Parallel 通常让每卡保存相应分片 KV，但具体布局取决于引擎与 Attention 实现。只有具备 NVLink/NVSwitch 的系统才能验证其互联收益；双 RTX 3080 一般以 PCIe 为基线。

## 结果分析

### 容量与服务指标表

| 配置 | KV dtype | Block | Prefix | 并发 | TTFT P95 | TPOT P95 | tok/s | Peak GiB | Preemptions | | ------------------------------------------------------------------------- | -- | -- | --- | -- | -- | -- | -- | -- | -- | | Baseline | 实测 | 实测 | Off | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 | | Paged Tune | 实测 | 实测 | Off | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 | | Prefix On | 实测 | 实测 | On | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 |

### 结论规则

- 理论容量大于实测：差值来自工作区、图捕获、元数据、碎片与安全余量；
- Prefix Warm Hit 只改善重复长前缀：这是预期边界，不是随机负载回归；
- 更小 Block 碎片更少但吞吐下降：元数据/调度/Kernel 成本可能变高；
- 并发升高后抢占出现：已经越过稳定容量点，最大吞吐未必满足 SLO；
- 低精度 KV 容量上升但质量或 TPOT 变差：必须按产品目标决策，不能只看显存。

## 常见错误与排查

### 1\. KV 公式中的 Head 数用错

使用 KV Head 数，而不是 Query Head 数。GQA/MQA 两者不同，应读取模型配置。

### 2\. 把 Max Context 当每个请求实际使用

容量规划可用上界，但真实碎片和平均容量依请求长度分布。必须报告 P50/P95/P99 长度。

### 3\. Prefix A/B 的请求不一致

除 Prefix Cache 开关外，模型、Prompt、输出、顺序、并发和服务配置必须一致。

### 4\. 把第一条请求当命中

缓存需要填充。Cold Fill、Warm Hit 与 Miss 应分开统计。

### 5\. 只看 nvidia-smi 已用显存

总已用显存不能区分权重、KV、工作区和图池。结合引擎日志、Profiler 和容量公式。

### 6\. Offload 后平均延迟正常就宣布成功

换入换出通常影响尾部。必须看 P95/P99、Preemption、PCIe 和 Host 内存压力。

### 7\. 用随机 Prompt 测缓存

随机 Prefix 不可复用。应构造业务真实共享 System Prompt、文档前缀或多轮会话。

## 优化前后对照

| 层次 | 基线 | 优化 | 验证 |
| --- | --- | --- | --- |
| 模型 | MHA/GQA 固定 | 合理 KV Head | 质量与容量 |
| 分配 | 最大长度预留 | Paged Block | 碎片与吞吐 |
| 复用 | 重算 Prefill | Prefix Cache | Cold/Hit/Miss |
| 精度 | BF16/FP16 KV | 支持的低精度 KV | 质量、容量、TPOT |
| 容量 | 激进并发 | SLO 驱动并发 | Preemption/Goodput |
| 层级 | GPU Only | 合法 Offload | PCIe、NUMA、P99 |

## 面试题与答案

### 1\. KV Cache 每 Token 字节数由什么决定？

层数、KV Head 数、Head Dimension、K/V 两份张量和元素精度。

### 2\. PagedAttention 主要解决什么问题？

它用固定大小 Block 按需管理非连续 KV，减少按最大长度预留造成的浪费，并支持更灵活的请求生命周期与共享。

### 3\. Block 越小越好吗？

不一定。内部碎片降低，但 Block Table、调度和 Kernel 元数据成本可能上升。

### 4\. Prefix Cache 什么时候收益最大？

共享前缀长、命中率高且 Prefill 成本占比较大时。随机短 Prompt 收益有限。

### 5\. 为什么 GQA 能减少 KV 显存？

多个 Query Head 共享较少的 KV Head，缓存的 K/V Head 数下降。

### 6\. KV Offload 的主要风险？

跨 PCIe/网络传输增加带宽压力和尾延迟，还受 NUMA、Pinned Memory、Host 容量和故障恢复影响。

## 课后练习

1. 对同一隐藏维度扫描 KV Head 数，计算每 Token 字节与并发容量。
2. 用真实请求长度分布替换均匀随机分布，比较 Block Size。
1. 分开报告 Prefix Cold、Warm Hit、Random Miss 的 TTFT。
2. 扫描最大模型长度，记录引擎 KV Token Capacity。
1. 制造容量压力，找到第一次 Preemption 前的最大稳定并发。
2. 若硬件支持，比较 BF16 与 FP8 KV 的容量、质量和 TPOT。
1. 加入 Host Offload，记录 PCIe 与 P99。
2. 把 KV 容量门禁加入服务发布 CI。

## 项目验收 Checklist

- KV 公式使用实际层数、KV Head、Head Dimension 与 dtype；
- 理论容量与引擎实测容量都已保存；
- 已扫描 Block Size 并同时看碎片与性能；
- 已构造 Prefix Cold/Hit/Miss 三组负载；
- 已报告 TTFT、TPOT、Token/s、显存、命中与 Preemption；
- 已扫描上下文与并发，找到稳定容量边界；
- Offload 实验包含 PCIe/NUMA 与尾延迟；
- CPU 模拟没有被写成 GPU 或 vLLM 实测；
- FP8/FP4 只在真实支持的硬件与工具链上验证；
- 所有结论注明模型、版本、长度分布、并发和硬件。