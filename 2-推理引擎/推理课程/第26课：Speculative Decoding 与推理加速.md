---
title: "第26课：Speculative Decoding 与推理加速"
source: "https://everythingai.top/learn/ai-systems-performance-engineering-text/lesson-26"
author:
published:
created: 2026-09-26
description: "面向 AI 工程学习者的专业课程学习空间。"
tags:
  - "clippings"
---
**导航**

本课讲清 Speculative Decoding 的验证与接受机制，掌握在保证输出质量前提下加速推理的方法。

## 课程定位

大模型 Decode 阶段有一个近乎反直觉的特点：每一步只生成一个 token，矩阵规模很小，却要把整套模型权重从显存读一遍。GPU 往往不是算不动，而是每次只干了一点活就必须等待下一步。

Speculative Decoding（推测解码）不改变目标模型，而是让一个更便宜的“草稿器”先猜若干 token，再让目标模型一次并行验证。猜得准，目标模型一次前向就提交多个 token；猜错，只丢弃错误后缀并按目标模型修正。它优化的是串行轮数，不是简单地把模型量化或缩小。

本课从正确性、性能模型、系统实现到可运行实验，回答四个问题：

1. 为什么一次验证多个 token 可能比逐 token 解码快？
2. 为什么“草稿猜错”仍能保持目标模型分布？
1. 何时推测解码会加速，何时反而变慢？
2. 如何在消费级 GPU、数据中心 GPU 和真实推理框架中测量它？

## 学习目标

完成本课后，你应能：

- 区分草稿、验证、接受、拒绝、修正和 KV Cache 回滚；
- 推导平均每轮提交 token 数和粗略加速比；
- 理解严格拒绝采样为何能保持目标分布；
- 比较 Draft/Target、N-gram、Medusa、EAGLE、MTP 等路线；
- 用 acceptance rate、accepted length、draft/verify latency 和 TPOT 判断瓶颈；
- 在 CPU 上做容量模拟，在任意 PyTorch CPU/CUDA 环境做验证批处理实验；
- 避免把论文中的单点速度或某款 GPU 结果写成普遍结论。

## 前置知识

- 自回归生成、logits、softmax、temperature、top-p；
- Prefill 与 Decode 的区别；
- KV Cache、Continuous Batching；
- GPU kernel launch、显存带宽和小 batch 利用率。

## 核心直觉：把时间上的串行改成一次空间并行

普通 Decode 生成 4 个 token，需要目标模型连续运行 4 次：

```
Target(x) -> t1 -> Target(x,t1) -> t2 -> Target(...) -> t3 -> Target(...) -> t4
```

推测解码先让便宜的草稿器提出 `d1,d2,d3,d4` ，目标模型用因果掩码一次计算这 4 个位置的分布：

```
Draft:  d1 -> d2 -> d3 -> d4
Target: [并行验证 d1,d2,d3,d4]
Result: 接受最长正确前缀 + 一个目标模型修正 token
```

目标模型仍然要读权重，但一次读权重服务了多个候选位置。小 batch Decode 原本常受显存带宽和串行依赖限制，验证阶段把更多算术工作塞进同一次前向，可能提高算术强度与 GPU 利用率。

关键不是“猜得越多越好”，而是：

## 算法原理

### Greedy 场景

若温度为 0，目标模型每个位置取 argmax。草稿 token 与目标 argmax 相等就接受，从左到右遇到第一次不相等即停止；错误位置由目标模型 token 替换，后续草稿全部丢弃。

例如草稿为 `A B X Y` ，目标验证结果为 `A B C ...`，本轮提交 `A B C` 。虽然 `X Y` 被计算过，但输出与普通 greedy 解码一致。

### Sampling 场景：不能只比较 token 是否相等

设目标分布为 p(x)，草稿分布为 q(x)。从 q 采样候选 x 后，以

$$
a(x)=\min\left(1,\frac{p(x)}{q(x)}\right)
$$

的概率接受。若拒绝，则从修正分布采样：

$$
p'(x)=\frac{\max(0,p(x)-q(x))}{\sum_y \max(0,p(y)-q(y))}
$$

这个“接受—修正”过程使最终样本仍来自目标分布 p。因此，理论上的 lossless 指“输出分布不变”，不是“固定随机种子下每次文本逐字相同”。浮点舍入、批形变化、并行采样和随机数消费顺序仍可能导致单次输出不同。

工程上要特别检查：

- 草稿和目标使用相同 tokenizer 与词表映射；
- temperature、top-p、top-k 等参数在验证器中一致；
- 拒绝后使用修正分布，而不是随便重采样；
- greedy equality 与 sampling convergence 分开测试；
- 质量检查不能只看几条人工样例。

### 一轮状态变化

一轮典型执行过程如下：

1. 调度器为请求预留最多 K 个草稿 token 的 KV 空间；
2. 草稿器生成线性链或候选树；
1. 目标模型在一次前向中验证候选；
2. 采样器得到最长可接受前缀；
1. 提交接受 token 与一个目标修正 token；
2. 释放或回滚被拒绝后缀占用的 KV block；
1. 进入下一轮，直到 EOS 或长度上限。

Paged KV Cache 中的“回滚”通常是调整逻辑长度、引用计数或空闲块标记，不应复制整段 KV；若实现发生大规模显存搬运，回滚会成为新瓶颈。

## 关键性能模型

### 平均每轮提交多少 token

假设每个草稿 token 独立地以固定概率 $\alpha$ 被接受，最多草拟 K 个。每轮提交“接受前缀 + 一个目标 token”。在这一简化假设下：

$$
E[N]=1+\alpha+\alpha^2+\cdots+\alpha^K=\frac{1-\alpha^{K+1}}{1-\alpha}
$$

当 $\alpha=1$ 时， $E[N]=K+1$ 。真实系统的接受事件并不独立，EOS、候选树、变长草稿和采样策略也会改变结果，所以公式用于容量判断，不替代实测。

### 加速比近似

记：

$t_T$ ：普通目标模型生成一个 token 的时间；

$t_D$ ：草稿器生成一个 token 的时间；

$t_V(K)$ ：目标模型验证 K 个候选的时间；

$t_O$ ：调度、采样、KV 管理等额外开销。

则近似加速比为：

$$
S\approx\frac{E[N]\cdot t_T}{K\cdot t_D+t_V(K)+t_O}
$$

只有 $S>1$ 才值得开启。它揭示三条重要规律：

- 草稿器即使很准，若太慢，也不会加速；

K 增大会增加潜在提交数，也会增加草稿和验证成本；

- 大 batch 下目标 GPU 已经饱和， $t_V(K)$ 可能接近线性增长，收益会消失。

### 服务指标

推测解码最常改善 TPOT/ITL，不一定改善 TTFT。应同时记录：

| 指标 | 含义 | 诊断价值 |
| --- | --- | --- |
| Acceptance rate | 被接受草稿 token / 草稿 token | 草稿与目标匹配度 |
| Mean accepted length | 每轮平均接受前缀长度 | 串行轮数减少程度 |
| Draft latency | 每轮草稿耗时 | 草稿是否过重 |
| Verify latency | 目标批量验证耗时 | 验证是否真正并行获益 |
| Rejected tokens | 被丢弃的草稿工作量 | 过深草稿的浪费 |
| TPOT / ITL | 输出 token 间延迟 | 交互体验 |
| Tokens/s | 每秒输出 token | 单请求或整体吞吐 |
| P50/P99 | 延迟分位数 | 尾延迟和调度抖动 |
| GPU 利用率/带宽 | 计算与内存状态 | 是否仍为低利用率 Decode |

## 主要技术路线

### Draft/Target 双模型

一个较小且 tokenizer 兼容的模型自回归提出 K 个候选，目标模型统一验证。它通用、容易理解，但多占一份权重与 KV Cache；草稿模型若跨 GPU 放置，还会引入 PCIe/NVLink 同步。

### N-gram、Prompt Lookup 与 Suffix

从 prompt、已生成文本或共享后缀结构中寻找重复片段，不加载神经草稿模型。代码生成、文档复述、结构化模板常有较高命中率；开放式创作命中率可能较低。优点是显存和草稿计算开销小，适合作为消费级 GPU 的第一条实验路径。

### Medusa

在目标模型上增加多个解码头，并行预测不同未来位置，组成候选树，再用 tree attention 验证。它减少独立草稿模型成本，但需要匹配目标模型训练额外头，树宽、树深和验证成本必须联合调优。

### EAGLE

EAGLE 系列在特征空间而非只在 token 空间外推未来状态。后续版本使用动态树或训练期测试等方法改善候选质量。它通常比完整小模型草稿更轻，但依赖与目标模型匹配的 EAGLE checkpoint 和运行时实现。

### MTP（Multi-Token Prediction）

目标模型原生带有未来 token 预测模块时，可以把这些模块作为草稿器。DeepSeek V3/R1 是典型案例。MTP 与模型训练、隐藏状态、KV 管理紧密耦合，不能给任意模型加一个命令行参数就获得同等能力。

### Self-Speculative 与层跳过

同一模型用部分层、早退或蒸馏分支先草拟，再用完整层验证，可减少额外权重，但需要处理共享 KV、隐藏状态和图捕获；草稿与验证之间的依赖可能限制并行。

## 瓶颈分析方法

### 第一步：确认基线真的适合推测解码

优先选择低到中 QPS、小 batch、Decode 占主导、目标模型内存带宽受限的负载。若服务已经通过 Continuous Batching 把 GPU 填满，推测解码可能只是扩大每轮 token 数而没有减少总 GPU 时间。

### 第二步：分解一轮时间

不要只测总 tokens/s。至少对以下区间加 NVTX 或 profiler 标记：

```
schedule -> draft -> prepare_verify -> target_verify -> reject_sample
         -> kv_commit_or_rewind -> stream_output
```

若 `draft` 占比高，缩小草稿模型或用 N-gram；若 `target_verify` 随 K 近线性增长，减小 K 或检查 attention/kernel；若 `prepare/rewind` 高，检查 Python 调度、动态 shape、block 管理和同步。

### 第三步：画 acceptance 与 K 的二维表

分别在真实 prompt 集、不同 temperature、不同输出类型和不同并发下测量 $K=1,2,4,6,8$ 。最优 K 是工作负载属性，不是模型常量。

### 第四步：检查正确性

- Greedy：与普通解码逐 token 相等；
- Sampling：大量样本的 token 频率/统计检验接近目标分布；
- EOS、停止词、结构化输出、logprobs、LoRA 和量化路径分别测试；
- 被拒绝 token 的 KV 不得污染后续输出。

## 完整实战

### 环境准备

Level 0 仅需 Python 3.9+。Level 1 使用隔离环境安装 PyTorch；CUDA wheel 必须按 PyTorch 官方安装页选择，不要把系统 CUDA 版本直接当作 wheel 标签。

```
python3 -m venv .venv-specdecode
source .venv-specdecode/bin/activate
python -m pip install --upgrade pip
# CPU 通用路径
python -m pip install "torch>=2.3"

python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("GPU:", p.name)
    print("Compute Capability:", f"{p.major}.{p.minor}")
    print("VRAM GiB:", round(p.total_memory / 2**30, 2))
PY
```

### Level 0：CPU 蒙特卡洛容量模拟器

将下面代码保存为 `speculative_decoding_sim.py` ：

```
#!/usr/bin/env python3
import argparse
import random

def expected_tokens(alpha: float, k: int) -> float:
    if abs(alpha - 1.0) < 1e-12:
        return k + 1.0
    return (1.0 - alpha ** (k + 1)) / (1.0 - alpha)

def run(tokens, k, alpha, target_ms, draft_ms, verify_base_ms,
        verify_per_token_ms, overhead_ms, seed):
    rng = random.Random(seed)
    emitted = rounds = accepted = proposed = rejected = 0
    spec_ms = 0.0

    while emitted < tokens:
        rounds += 1
        a = 0
        for _ in range(k):
            proposed += 1
            if rng.random() < alpha:
                a += 1
                accepted += 1
            else:
                break

        # 接受前缀后，总有一个来自目标分布的 token 可提交；末尾截断。
        committed = min(a + 1, tokens - emitted)
        emitted += committed
        rejected += k - a
        spec_ms += (k * draft_ms + verify_base_ms
                    + k * verify_per_token_ms + overhead_ms)

    base_ms = tokens * target_ms
    empirical_n = tokens / rounds
    theoretical_n = expected_tokens(alpha, k)
    print(f"tokens={tokens} k={k} alpha={alpha:.3f}")
    print(f"rounds={rounds} empirical_tokens/round={empirical_n:.3f}")
    print(f"theoretical_tokens/round={theoretical_n:.3f}")
    print(f"accepted/proposed={accepted}/{proposed} ({accepted/max(1, proposed):.3f})")
    print(f"rejected_or_unused_slots={rejected}")
    print(f"baseline_ms={base_ms:.2f} speculative_ms={spec_ms:.2f}")
    print(f"estimated_speedup={base_ms/spec_ms:.3f}x")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=4096)
    p.add_argument("--draft-k", type=int, default=4)
    p.add_argument("--acceptance", type=float, default=0.75)
    p.add_argument("--target-ms", type=float, default=8.0)
    p.add_argument("--draft-ms", type=float, default=0.45)
    p.add_argument("--verify-base-ms", type=float, default=8.5)
    p.add_argument("--verify-per-token-ms", type=float, default=0.25)
    p.add_argument("--overhead-ms", type=float, default=0.35)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    if not 0.0 <= a.acceptance <= 1.0 or a.draft_k < 1:
        p.error("acceptance 必须在 [0,1]，draft-k 必须 >= 1")
    run(a.tokens, a.draft_k, a.acceptance, a.target_ms, a.draft_ms,
        a.verify_base_ms, a.verify_per_token_ms, a.overhead_ms, a.seed)
```

运行三组对照：

```
python speculative_decoding_sim.py --draft-k 4 --acceptance 0.80
python speculative_decoding_sim.py --draft-k 8 --acceptance 0.80
python speculative_decoding_sim.py --draft-k 4 --acceptance 0.35
```

预期现象：高接受率下，单轮提交 token 数增加；但 K 从 4 增到 8 并不保证继续加速，因为草稿与验证成本也增加。低接受率时，推测路径可能慢于基线。模拟数字是输入参数推导的容量结果，不代表任何具体 GPU。

### Level 1：通用 PyTorch 验证批处理微基准

这个实验不下载 LLM，而是测量系统核心机制：“目标网络运行 K 次”与“把 K 个位置合成一次验证”之间的差异。它是 kernel/调度微基准，不声称生成语义等价。

保存为 `verify_batch_benchmark.py` ：

```
#!/usr/bin/env python3
import argparse
import time
import torch
from torch import nn

def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def bench(fn, device, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync(device)
    return (time.perf_counter() - t0) * 1000 / iters

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--draft-k", type=int, default=4)
    p.add_argument("--acceptance", type=float, default=0.75)
    p.add_argument("--iters", type=int, default=50)
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print("device=", device, "torch=", torch.__version__)
    if device.type == "cuda":
        prop = torch.cuda.get_device_properties(0)
        print("gpu=", prop.name, "cc=", f"{prop.major}.{prop.minor}",
              "vram_GiB=", round(prop.total_memory / 2**30, 2))

    target = nn.Sequential(
        nn.Linear(a.hidden, 4 * a.hidden, bias=False), nn.GELU(),
        nn.Linear(4 * a.hidden, a.hidden, bias=False)
    ).to(device=device, dtype=dtype).eval()
    draft = nn.Sequential(
        nn.Linear(a.hidden, a.hidden // 2, bias=False), nn.ReLU(),
        nn.Linear(a.hidden // 2, a.hidden, bias=False)
    ).to(device=device, dtype=dtype).eval()

    x = torch.randn(a.batch, a.hidden, device=device, dtype=dtype)
    xk = torch.randn(a.batch * a.draft_k, a.hidden,
                     device=device, dtype=dtype)

    @torch.inference_mode()
    def target_sequential():
        y = x
        for _ in range(a.draft_k):
            y = target(y)
        return y

    @torch.inference_mode()
    def target_verify_batch():
        return target(xk)

    @torch.inference_mode()
    def draft_sequential():
        y = x
        for _ in range(a.draft_k):
            y = draft(y)
        return y

    seq_ms = bench(target_sequential, device, iters=a.iters)
    verify_ms = bench(target_verify_batch, device, iters=a.iters)
    draft_ms = bench(draft_sequential, device, iters=a.iters)
    alpha = a.acceptance
    expected = ((1 - alpha ** (a.draft_k + 1)) / (1 - alpha)
                if alpha < 1 else a.draft_k + 1)
    one_target_ms = seq_ms / a.draft_k
    estimated = expected * one_target_ms / (draft_ms + verify_ms)

    print(f"K sequential target calls: {seq_ms:.3f} ms")
    print(f"one batched target verify: {verify_ms:.3f} ms")
    print(f"K draft calls: {draft_ms:.3f} ms")
    print(f"expected committed/round: {expected:.3f}")
    print(f"model-only estimated speedup: {estimated:.3f}x")
    print("注意：未计入 attention、KV、采样、调度和回滚开销。")

if __name__ == "__main__":
    main()
```

运行命令：

```
python verify_batch_benchmark.py --batch 1 --hidden 2048 --draft-k 4
python verify_batch_benchmark.py --batch 16 --hidden 2048 --draft-k 4
python verify_batch_benchmark.py --batch 1 --hidden 2048 --draft-k 8
```

显存不足时使用 `--hidden 1024` 。预期现象是小 batch 时一次批量验证通常比 K 次目标调用更有效；batch 增大后，目标调用本身已有较高利用率，优势可能缩小。结果必须连同 GPU 型号、Compute Capability、PyTorch/CUDA 版本和参数保存。

### Level 2：真实框架专项实验

### vLLM：先从无草稿模型的 N-gram 开始

vLLM 的当前接口使用 JSON 形式的 `--speculative-config` 。下面命令只示范配置结构；模型、显存利用率和最大长度应按本机调整：

```
vllm serve /path/to/target-model \
  --speculative-config '{
    "method":"ngram",
    "num_speculative_tokens":4,
    "prompt_lookup_min":2,
    "prompt_lookup_max":5
  }'
```

双模型路径：

```
vllm serve /path/to/target-model \
  --speculative-config '{
    "method":"draft_model",
    "model":"/path/to/tokenizer-compatible-draft-model",
    "num_speculative_tokens":4,
    "draft_tensor_parallel_size":1
  }'
```

先用 `vllm --version` 和对应版本文档核对字段。不要把旧版 `--speculative-model` 示例与新版 `--speculative-config` 混用。基线与优化必须使用相同 prompt、采样参数、并发和输出长度，并记录 acceptance 与 P50/P99 TPOT。

### TensorRT-LLM：能力随版本变化

TensorRT-LLM 的 LLM API 可配置 Draft/Target、NGram、EAGLE 3、MTP 等路线。例如：

```
from tensorrt_llm import LLM
from tensorrt_llm.llmapi import NGramDecodingConfig

config = NGramDecodingConfig(
    max_draft_len=3,
    max_matching_ngram_size=4,
    is_public_pool=False,
)
llm = LLM("/path/to/target-model", speculative_config=config)
```

具体类名和支持矩阵以安装版本文档为准。MTP 需要模型原生支持，EAGLE 需要匹配 checkpoint；不能把 N-gram 的成功运行视为 MTP/EAGLE 的等价验证。

## 硬件适配与边界

| 层级 | 硬件 | 建议实验 | 不能据此声称 |
| --- | --- | --- | --- |
| Level 0 | 任意 CPU | 容量模型、接受率与 K 扫描 | 真实 GPU 加速比 |
| Level 1 | 任意 PyTorch CPU/CUDA | 连续目标调用 vs 批量验证 | 完整 LLM 端到端收益 |
| Ampere | RTX 3080/3090、A100 | FP16/BF16 能力检测后做小 batch 验证；3090 可容纳更大草稿 | 3080 双卡具备 NVLink |
| Ada | RTX 4090、L40/L40S | 单卡目标+轻量草稿或 N-gram | RTX 4090 属于 Blackwell |
| Hopper | H100/H200 | 数据中心框架、FP8 模型、CUDA Graph 专项 | 消费卡可等价模拟 Hopper |
| Blackwell | RTX 5090、B100/B200/GB200 | 新低精度与专用 kernel 的框架实测 | RTX 5090 等同 B200/GB200 |

双 RTX 3080 20GB 上，优先把目标与轻量草稿放在同一张卡，或先使用 N-gram。把草稿放到第二张卡会经过 PCIe 传输和跨设备同步，只有端到端实测证明收益后才保留。消费级 3080 之间通常没有 NVLink，不能套用 A100/H100 NVLink 结论。

## 结果分析模板

建议每组实验保存：

```
模型/量化：
GPU/CC/显存：
框架/PyTorch/CUDA：
Prompt 数据集：
并发与 batch：
采样参数：
K / 草稿方法：
Acceptance rate：
Mean accepted length：
Draft / Verify / Rewind ms：
Baseline vs Spec TTFT：
Baseline vs Spec TPOT：
Baseline vs Spec tokens/s：
P50/P99：
正确性检查：
```

如果 TPOT 下降而 TTFT 上升，这是合理现象：初始化草稿器或图捕获增加首 token 成本。若平均接受长度很高但仍变慢，重点看草稿器耗时、跨 GPU 同步、动态 shape、验证 kernel 和 Python 调度，而不是继续提高接受率。

## 常见错误与排查

### 开启后反而变慢

可能原因：batch 已很大、草稿模型过重、K 过大、接受率低、验证 kernel 未优化或 CPU 调度占比高。先分解一轮耗时，再分别扫描并发与 K。

### 草稿接受率接近零

检查 tokenizer、词表大小、special token、chat template、模型家族、量化方式和采样参数。仅仅“参数量更小”不代表它是合格草稿模型。

### 显存突然不足

草稿权重、草稿 KV、目标验证的扩张 token、CUDA Graph 静态 buffer 都会额外占显存。降低 `num_speculative_tokens` 、最大 batch、最大长度或使用无模型 N-gram。

### 输出与基线不一致

Greedy 模式先做逐 token equality。Sampling 模式不要只比较单次字符串；核对拒绝采样实现、随机数、logits processor、停止条件和浮点精度，并做分布统计。

### P99 变差但平均值变好

不同请求接受长度不一致会造成批内分歧，固定长度 CUDA Graph 还可能填充草稿槽位。检查请求分组、动态 speculation、长尾 prompt 和 scheduler 公平性。

### 第二张 GPU 没有带来收益

草稿结果必须及时送回目标 GPU，PCIe 同步可能超过省下的目标轮次。测量 P2P 可用性、链路带宽、同步时间；不要仅看两张卡的平均利用率。

## 优化前后对照

| 维度 | 普通自回归 Decode | 推测解码 |
| --- | --- | --- |
| 目标模型轮次 | 每 token 一次 | 每轮可能提交多个 token |
| 目标验证形状 | 小 batch、单位置 | 多候选位置或候选树 |
| 额外计算 | 少 | 草稿、拒绝后缀、采样 |
| 额外显存 | 基线 KV | 草稿权重/KV、验证 buffer |
| 正确性 | 直接来自目标模型 | 正确拒绝采样可保持目标分布 |
| 典型收益区 | 通用 | 低/中 QPS、内存受限 Decode |
| 主要风险 | 串行慢 | 接受率低或编排成本超过收益 |

## 面试题与答案

### 1\. 推测解码为什么不会降低模型质量？

因为草稿只提出候选，最终由目标模型验证。Greedy 下只接受与目标 argmax 一致的前缀；Sampling 下用接受概率和修正分布保证最终仍服从目标分布。若实现省略修正采样，则不再具有这项保证。

### 2\. Acceptance rate 越高，加速一定越大吗？

不一定。加速取决于每轮提交量与草稿、验证、调度、KV 回滚总成本之比。一个很大但很准的草稿模型可能比目标多跑几次更贵。

### 3\. 为什么它更适合小 batch？

小 batch Decode 往往读权重多、计算少，GPU 利用率低。批量验证多个候选可以提高一次权重读取对应的计算量。大 batch 已能填满 GPU 时，扩大验证长度可能只增加工作量。

### 4\. K 应如何选择？

用真实 workload 扫描 K，同时看 accepted length、draft latency、verify latency 和 P99。低接受率选小 K；高接受率且验证扩张便宜时可增大。生产系统可按请求动态调整。

### 5\. N-gram 与双模型草稿如何取舍？

N-gram 无额外模型、成本低，适合重复文本；双模型对开放式生成更通用，但占显存且有草稿计算。先按工作负载重复度、显存和延迟预算做选择，再 A/B 测试。

### 6\. KV Cache 为什么需要回滚？

目标验证会为草稿位置产生 KV，但第一次拒绝后的后缀不属于最终序列。运行时必须缩短逻辑长度或释放对应 block，否则错误状态会污染后续生成并浪费显存。

### 7\. Medusa、EAGLE、MTP 的共同目标是什么？

都是用比完整目标模型逐 token 执行更便宜的方式提出多个未来候选，再由目标模型验证。差别在候选来自额外解码头、特征预测还是模型原生多 token 模块，以及候选树和运行时集成方式。

### 8\. 如何证明生产加速不是测试幻觉？

固定模型、数据集、输出长度和采样参数，预热后测 P50/P99 TTFT、TPOT、tokens/s；分解 draft/verify/rewind；做正确性测试；覆盖不同并发并报告完整环境。单条 prompt 或单个平均值不足以证明。

## 课后练习

1. 用 Level 0 扫描 $\alpha\in\{0.2,0.5,0.8,0.95\}$ 和 $K\in\{1,2,4,8\}$ ，找出每个接受率下的最佳 K。
2. 把模拟器改为位置相关接受率，例如第 i 个候选的接受率为 $\alpha^i$ ，比较固定概率模型的误差。
1. 在 Level 1 中扫描 batch 1、4、16、64，画出批量验证收益随 batch 的变化。
2. 给 Level 1 加 NVTX 标记，用 Nsight Systems 区分 draft、verify 和同步。
1. 用任意小词表构造 p,q，实现拒绝采样，运行十万次并比较经验分布与 p。
2. 在重复性文本与开放式文本上测试 N-gram，比较 acceptance、TPOT 和 P99。
1. 设计一个动态 K 控制器：根据最近 32 轮接受长度和服务队列深度调整草稿深度。

## Checklist

- 已确认负载是低/中 QPS、Decode 主导或内存受限；
- 基线与优化使用相同模型、prompt、采样和输出长度；
- 已记录 GPU、Compute Capability、框架、CUDA 和精度；
- 已测 acceptance rate 与 mean accepted length；
- 已拆分 draft、verify、sampling、KV rewind 时间；
- 已同时比较 TTFT、TPOT、tokens/s 和 P50/P99；
- 已扫描 K，没有把更深草稿默认当作更快；
- 已验证 tokenizer、chat template 和 special token 兼容；
- Greedy 已做逐 token equality；Sampling 已做分布检验；
- 已检查显存增量、KV 预留和 CUDA Graph buffer；
- 多 GPU 路径已测链路与同步成本；
- 专属 MTP/EAGLE/Hopper/Blackwell 能力未被消费卡模拟冒充；
- 所有性能数字只作为本次环境测量结论。