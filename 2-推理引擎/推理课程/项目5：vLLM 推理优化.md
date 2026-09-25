---
title: "项目5：vLLM 推理优化"
source: "https://everythingai.top/learn/ai-systems-performance-engineering-text/project-05"
author:
published:
created: 2026-09-26
description: "面向 AI 工程学习者的专业课程学习空间。"
tags:
  - "clippings"
---
**导航**

本项目围绕 vLLM 推理服务完成压测、瓶颈分析与参数调优，优化吞吐、延迟和并发能力。

## 项目定位

本项目把 vLLM 从“能启动模型”推进到“能用 SLO 驱动调优”。你将搭建 OpenAI 兼容服务，构造固定输入/输出长度的并发负载，测量 TTFT、TPOT、端到端延迟、请求吞吐、输出 Token 吞吐与成功率，然后只改变一个调度或内存参数完成优化闭环。

```
容量估算 → 服务冷启动 → 单并发正确性
→ 并发扫描 → 找到饱和拐点 → 调整 Scheduler/KV/Prefix/并行
→ 重复 Benchmark → SLO 回归门禁 → 保存配置与原始结果
```

Level 0 使用纯 Python 模拟排队与 Prefill/Decode 竞争，不需要 GPU。Level 1 面向常见 NVIDIA GPU 并自动记录环境。Level 2 才验证 Tensor Parallel、FP8/FP4、NVLink 或架构专属 Kernel；没有对应硬件时不宣称完成这些特性验证。

## 学习目标

- 区分 TTFT、TPOT/ITL、E2E Latency、QPS、Tokens/s 与 Goodput；
- 理解 Continuous Batching、Chunked Prefill、KV Cache 与 Scheduler 的耦合；
- 使用 OpenAI 兼容流式接口测量首 Token 与生成阶段；
- 扫描并发而不是只测最大吞吐；
- 根据 SLO 找到最大可接受负载；
- 正确调节 `max_num_batched_tokens` 、 `max_num_seqs` 、模型长度和显存利用率；
- 识别 OOM、抢占、排队、Tokenizer、网络与客户端瓶颈；
- 生成可重复的服务配置、环境快照和回归报告。

## 前置知识

- 第 23～25 课；
- Linux、Python、HTTP/SSE 与基本 GPU 监控；
- 可下载且许可允许使用的 Hugging Face 模型；
- vLLM、驱动、CUDA 与 PyTorch 必须使用兼容组合，安装前锁定目标版本。

## 项目交付物

```
vllm-serving-project/
├── scheduler_sim.py
├── openai_stream_benchmark.py
├── run_sweep.sh
├── reports/
│   ├── environment.txt
│   ├── server_command.txt
│   ├── baseline_c1.json
│   ├── baseline_c8.json
│   ├── optimized_c8.json
│   └── analysis.md
└── logs/
    └── server.log
```

## 核心直觉：最优点在吞吐与排队之间

若并发从 1 增加到 C，GPU Batch 变大，通常会提高计算效率；但请求会在 Scheduler 和 Batch 中等待，TTFT 与 TPOT 也可能变差。

$$
Latency=T_{queue}+T_{prefill}+T_{decode}+T_{network}
$$

$$
TTFT=T_{queue}+T_{prefill}+T_{first\ decode}+T_{network,first}
$$

生成 N 个输出 Token 时，可用近似：

$$
TPOT\approx\frac{E2E-TTFT}{N-1}
$$

当 $N\le1$ 时 TPOT 无定义，Benchmark 必须避免除零。

吞吐最大点不一定满足交互 SLO。定义：

$$
Goodput=\frac{N_{TTFT\le S_{TTFT}\ \land\ TPOT\le S_{TPOT}}}{T_{wall}}
$$

本项目最终选择 Goodput 最大且错误率、P95/P99 满足约束的配置。

## Level 0：Continuous Batching 排队模拟

保存为 `scheduler_sim.py` ：

```
#!/usr/bin/env python3
import argparse
import heapq
import json
import random
import statistics

def percentile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    low = int(position)
    high = min(low + 1, len(values) - 1)
    weight = position - low
    return values[low] * (1 - weight) + values[high] * weight

def simulate(requests, arrival_rate, max_sequences, token_budget,
             prefill_tokens, output_tokens, tick_ms, seed):
    rng = random.Random(seed)
    arrivals = []
    now = 0.0
    for request_id in range(requests):
        now += rng.expovariate(arrival_rate) * 1000.0
        heapq.heappush(arrivals, (now, request_id))

    waiting = []
    running = []
    completed = []
    clock = 0.0
    while len(completed) < requests:
        while arrivals and arrivals[0][0] <= clock:
            arrival, request_id = heapq.heappop(arrivals)
            waiting.append({
                "id": request_id,
                "arrival": arrival,
                "prefill_left": prefill_tokens,
                "decode_left": output_tokens,
                "first_token": None,
            })
        while waiting and len(running) < max_sequences:
            running.append(waiting.pop(0))

        budget = token_budget
        for request in list(running):
            if budget <= 0:
                break
            if request["prefill_left"] > 0:
                consumed = min(request["prefill_left"], budget)
                request["prefill_left"] -= consumed
                budget -= consumed
            elif request["decode_left"] > 0 and budget > 0:
                request["decode_left"] -= 1
                budget -= 1
                if request["first_token"] is None:
                    request["first_token"] = clock + tick_ms

        clock += tick_ms
        for request in list(running):
            if request["decode_left"] == 0:
                completed.append({
                    "ttft_ms": request["first_token"] - request["arrival"],
                    "e2e_ms": clock - request["arrival"],
                })
                running.remove(request)

        if not running and not waiting and arrivals and arrivals[0][0] > clock:
            clock = arrivals[0][0]

    ttft = [item["ttft_ms"] for item in completed]
    e2e = [item["e2e_ms"] for item in completed]
    return {
        "requests": requests,
        "wall_seconds": clock / 1000.0,
        "request_throughput": requests / (clock / 1000.0),
        "output_token_throughput": requests * output_tokens / (clock / 1000.0),
        "ttft_p50_ms": percentile(ttft, 0.50),
        "ttft_p95_ms": percentile(ttft, 0.95),
        "e2e_p50_ms": percentile(e2e, 0.50),
        "e2e_p95_ms": percentile(e2e, 0.95),
        "mean_ttft_ms": statistics.mean(ttft),
        "warning": "离散事件模型用于理解趋势，不是 vLLM 或 GPU 性能预测。",
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--arrival-rate", type=float, default=8.0)
    parser.add_argument("--max-sequences", type=int, default=16)
    parser.add_argument("--token-budget", type=int, default=512)
    parser.add_argument("--prefill-tokens", type=int, default=256)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--tick-ms", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if min(args.requests, args.arrival_rate, args.max_sequences,
           args.token_budget, args.prefill_tokens, args.output_tokens,
           args.tick_ms) <= 0:
        raise ValueError("所有负载参数必须大于 0")
    print(json.dumps(simulate(**vars(args)), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
```

### 运行命令

```
python scheduler_sim.py --arrival-rate 4 --max-sequences 8 --token-budget 256
python scheduler_sim.py --arrival-rate 16 --max-sequences 8 --token-budget 256
python scheduler_sim.py --arrival-rate 16 --max-sequences 32 --token-budget 1024
```

### 预期现象

当到达速率超过服务能力，排队快速增加，P95 TTFT 比吞吐更早恶化。增加并发容量或 Token Budget 可能提高吞吐，也可能让单个 Tick 更昂贵；这个简化模型没有 GPU Kernel、KV 容量、Prefix Cache 或抢占，因此只能用于形成假设。

## Level 1：安装与启动 vLLM

### 环境隔离

```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip uv
uv pip install "vllm==0.23.0"
python -m pip freeze > reports/requirements-lock.txt
vllm --version
vllm serve --help=all > reports/vllm-serve-help.txt
```

版本示例用于复现实验快照，不代表永远推荐。若官方稳定版、驱动或 GPU 支持矩阵已变化，应新建环境测试，不在原环境原地升级。

### 自动记录环境

```
mkdir -p reports logs
{
  date -Iseconds
  uname -a
  nvidia-smi -L
  nvidia-smi topo -m
  nvidia-smi --query-gpu=name,driver_version,memory.total,pci.bus_id,pstate,power.limit --format=csv
  python --version
  vllm --version
} 2>&1 | tee reports/environment.txt
```

### 启动基线服务

用一个当前机器能容纳、且许可允许使用的模型替换 `MODEL_ID` ：

```
export MODEL_ID="your-org/your-model"
vllm serve "$MODEL_ID" \
  --host 127.0.0.1 --port 8000 \
  --dtype auto \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  2>&1 | tee logs/server.log
```

另开终端验证：

```
curl -s http://127.0.0.1:8000/v1/models | python -m json.tool
```

如果模型超出单卡容量，不要盲目把显存利用率设为 1。应选择更小模型、合法量化版本、Tensor Parallel 或容量更大的 GPU，并记录改变后的问题定义。

## 完整流式负载生成器

脚本只依赖 Python 标准库，通过 SSE 流记录首个内容事件和完成时间。若服务返回 Usage，就使用精确 Completion Token；否则退化为内容事件数，并在结果中明确标记估算。

保存为 `openai_stream_benchmark.py` ：

```
#!/usr/bin/env python3
import argparse
import concurrent.futures
import json
import math
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return values[low]
    return values[low] * (high - position) + values[high] * (position - low)

def one_request(url, model, prompt, max_tokens, timeout, request_id):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    start = time.perf_counter()
    first_content = None
    content_events = 0
    completion_tokens = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                usage = event.get("usage")
                if usage and usage.get("completion_tokens") is not None:
                    completion_tokens = int(usage["completion_tokens"])
                choices = event.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    if delta.get("content"):
                        content_events += 1
                        if first_content is None:
                            first_content = time.perf_counter()
        end = time.perf_counter()
        if first_content is None:
            raise RuntimeError("响应完成但没有收到内容 Token")
        output_tokens = completion_tokens or content_events
        e2e = end - start
        ttft = first_content - start
        tpot = (e2e - ttft) / (output_tokens - 1) if output_tokens > 1 else None
        return {
            "id": request_id,
            "ok": True,
            "ttft_s": ttft,
            "e2e_s": e2e,
            "tpot_s": tpot,
            "output_tokens": output_tokens,
            "token_count_source": "usage" if completion_tokens is not None else "content_events",
        }
    except (urllib.error.URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as error:
        return {"id": request_id, "ok": False, "error": repr(error)}

def main():
    parser = argparse.ArgumentParser(description="OpenAI SSE 推理 Benchmark")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--prompt-words", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--ttft-slo", type=float, default=1.0)
    parser.add_argument("--tpot-slo", type=float, default=0.05)
    parser.add_argument("--output", default="reports/benchmark.json")
    args = parser.parse_args()
    if min(args.requests, args.concurrency, args.max_tokens,
           args.prompt_words, args.timeout) <= 0:
        raise ValueError("负载参数必须大于 0")

    prompt = "请用简洁语言解释 AI 系统性能工程。 " * args.prompt_words
    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    wall_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(one_request, url, args.model, prompt, args.max_tokens,
                        args.timeout, request_id)
            for request_id in range(args.requests)
        ]
        rows = [future.result() for future in concurrent.futures.as_completed(futures)]
    wall = time.perf_counter() - wall_start

    successful = [row for row in rows if row["ok"]]
    ttft = [row["ttft_s"] for row in successful]
    e2e = [row["e2e_s"] for row in successful]
    tpot = [row["tpot_s"] for row in successful if row["tpot_s"] is not None]
    output_tokens = sum(row["output_tokens"] for row in successful)
    good = [
        row for row in successful
        if row["ttft_s"] <= args.ttft_slo
        and row["tpot_s"] is not None
        and row["tpot_s"] <= args.tpot_slo
    ]
    result = {
        "model": args.model,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "successful": len(successful),
        "success_rate": len(successful) / args.requests,
        "wall_seconds": wall,
        "request_throughput": len(successful) / wall,
        "output_token_throughput": output_tokens / wall,
        "goodput_requests_per_second": len(good) / wall,
        "ttft_s": {"p50": percentile(ttft, 0.50), "p95": percentile(ttft, 0.95),
                   "p99": percentile(ttft, 0.99)},
        "tpot_s": {"p50": percentile(tpot, 0.50), "p95": percentile(tpot, 0.95),
                   "p99": percentile(tpot, 0.99)},
        "e2e_s": {"p50": percentile(e2e, 0.50), "p95": percentile(e2e, 0.95),
                  "p99": percentile(e2e, 0.99)},
        "token_count_sources": sorted(set(
            row["token_count_source"] for row in successful
        )),
        "errors": [row for row in rows if not row["ok"]],
        "raw": sorted(rows, key=lambda row: row["id"]),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

if __name__ == "__main__":
    main()
```

### 单并发正确性

先从 `/v1/models` 返回结果中取得真实 Served Model Name：

```
export SERVED_MODEL="your-served-model-name"
python openai_stream_benchmark.py --model "$SERVED_MODEL" \
  --requests 3 --concurrency 1 --max-tokens 32 \
  --output reports/smoke.json
```

### 并发扫描

保存为 `run_sweep.sh` ：

```
#!/usr/bin/env bash
set -euo pipefail
: "${SERVED_MODEL:?请先 export SERVED_MODEL=...}"
mkdir -p reports
for concurrency in 1 2 4 8 16 32; do
  python openai_stream_benchmark.py \
    --model "$SERVED_MODEL" \
    --requests 100 \
    --concurrency "$concurrency" \
    --max-tokens 128 \
    --prompt-words 64 \
    --ttft-slo 1.0 \
    --tpot-slo 0.05 \
    --output "reports/c${concurrency}.json"
done
```

```
chmod +x run_sweep.sh
./run_sweep.sh
```

## 使用 vLLM 官方 Benchmark 交叉验证

当前版本应先查看帮助，确认参数名：

```
vllm bench serve --help=all | tee reports/vllm-bench-help.txt
```

典型随机负载命令：

```
vllm bench serve \
  --backend openai-chat \
  --base-url http://127.0.0.1:8000 \
  --endpoint /v1/chat/completions \
  --model "$SERVED_MODEL" \
  --dataset-name random \
  --random-input-len 512 \
  --random-output-len 128 \
  --num-prompts 200 \
  --max-concurrency 8 \
  --save-result
```

自写客户端用于理解指标和回归门禁；官方 Benchmark 用于与 vLLM 当前实现口径对齐。两者结果不一致时，先检查 Prompt Tokenizer、输出 Token、请求到达模型、Endpoint 和是否忽略 EOS。

## 参数调优顺序

### 1\. 固定模型长度

`max_model_len` 会影响可预留的 KV 容量。不要把模型声明的最大 Context 无条件作为服务配置；根据真实输入、输出与 SLO 选择。

### 2\. 扫描调度容量

先用当前版本帮助确认参数：

```
vllm serve --help=all | rg "max-num-batched-tokens|max-num-seqs|chunked-prefill"
```

常见方向：

- 增大 `max_num_batched_tokens` ：可能提高 Prefill 吞吐，但增加单批执行时间与 KV 压力；
- 增大 `max_num_seqs` ：可能提高并发，但增加调度和 KV 占用；
- Chunked Prefill：帮助长 Prefill 与 Decode 共享调度，但最优 Token Budget 依负载而变；
- Prefix Caching：只对可复用 Prefix 有意义，随机 Prompt 不应期待收益。

### 3\. 观察容量事件

同时采集：

```
nvidia-smi dmon -s pucvmet -d 1 -o DT > reports/gpu_dmon.csv
```

检查 Server 日志中的 KV Cache、Preemption、OOM、CUDA Graph、Engine Restart 与错误请求。显存利用率高不等于 Goodput 高。

### 4\. 一次只改一个参数

| 实验 | 改变量 | 其他条件 |
| --- | --- | --- |
| A | 基线 | 固定 |
| B | Batched Token Budget | 与 A 相同 |
| C | Max Sequences | 与最佳 B 相同 |
| D | Prefix Caching | 使用重复 Prefix 负载 |
| E | Tensor Parallel | 模型、Token、并发相同 |

## Level 2：架构专项与多 GPU

| 架构 | 示例 | 可选验证 |
| --- | --- | --- |
| Ampere | RTX 3080/3090、A100 | FP16/BF16（按卡能力）、Tensor Parallel；RTX 3090 特定型号可有 NVLink |
| Ada Lovelace | RTX 4090、L40/L40S | FP16/BF16、PCIe 多卡；RTX 4090 无 NVLink |
| Hopper | H100/H200 | FP8、Transformer Engine、NVLink/NVSwitch 数据中心场景 |
| Blackwell | RTX 5090、B100/B200/GB200 | FP8/FP4 与新 Kernel，必须匹配工具链 |

多 GPU 启动示例：

```
CUDA_VISIBLE_DEVICES=0,1 vllm serve "$MODEL_ID" \
  --tensor-parallel-size 2 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85
```

Tensor Parallel 是否更快取决于模型是否放得下、矩阵规模与互联。双 RTX 3080 通常走 PCIe，不应写成 NVLink 实验。只有实际具备 NVLink/NVSwitch 的系统才能验证其收益。

量化不是无损的通用性能开关。必须同时验证模型质量、支持的 Kernel、显存、TTFT、TPOT 和吞吐；Hopper/Blackwell 的 FP8/FP4 结果不能迁移到 Ampere/Ada。

## 结果分析

### SLO 表

| 配置 | 并发 | TTFT P95 | TPOT P95 | E2E P95 | Output tok/s | Goodput req/s | 成功率 | | --------------------------------------------------------------- | - | -- | -- | -- | -- | -- | -- | | Baseline | 1 | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 | | Baseline | 8 | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 | | Optimized | 8 | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 |

### 典型现象

- 并发上升、Token 吞吐上升、TTFT 恶化：Batch 收益与排队同时发生；
- TTFT 好但 TPOT 差：Decode 批次过重、抢占或内存压力；
- GPU 利用率低且 TTFT 高：客户端到达不足、CPU/Tokenizer、模型加载或调度瓶颈；
- 吞吐突然下降且日志出现 Preemption：KV 容量不足或并发/长度配置过高；
- Prefix Caching 无收益：前缀不相同、缓存命中低或 Prefill 占比小；
- $TP=2$ 比 $TP=1$ 慢：模型本来能单卡运行，通信成本超过分片收益。

## 常见错误与排查

### 1\. Served Model Name 不一致

先请求 `/v1/models` ，客户端的 `model` 必须匹配服务端名称。

### 2\. TTFT 等于完整响应时间

客户端没有真正读取流，或服务未返回 SSE。确认 `stream=true` ，以第一个非空内容事件计 TTFT。

### 3\. TPOT 计算错误

TPOT 分母通常是首 Token 后的 Token 间隔数，即 N-1。只有一个输出 Token 时应为 `null` 。

### 4\. Prompt 长度不受控

“重复若干单词”不等于固定 Token。正式结果应使用目标模型 Tokenizer 构造精确长度，并保存 Token 数分布。

### 5\. 客户端先饱和

观察客户端 CPU、连接、文件描述符与网络。服务端 GPU 空闲而客户端吞吐封顶时，应分离 Load Generator。

### 6\. 把 OOM 解决为显存利用率 1.0

更高利用率减少运行余量，可能增加抢占或 OOM。先降低模型长度、并发或选择合适并行/量化方案。

### 7\. 只测离线吞吐

离线吞吐不包含真实到达、排队和 SLO。在线服务必须测 TTFT、TPOT、尾延迟与 Goodput。

### 8\. 随机负载验证 Prefix Cache

随机 Prefix 命中率接近零。应构造共享 System Prompt 或固定长前缀的成对实验。

## 优化前后对照

| 层次 | 基线问题 | 优化方向 | 证据 |
| --- | --- | --- | --- |
| 请求 | 长度不可控 | 固定 Token 分布 | 输入/输出直方图 |
| 调度 | 并发过低/过高 | 扫描拐点 | TTFT/TPOT/吞吐 |
| Prefill | 长请求阻塞 Decode | Chunk/Budget 调整 | 时间线与 P95 |
| KV | 抢占/OOM | 长度、并发、缓存配置 | 日志与显存 |
| 计算 | Kernel/精度不合适 | 合法量化/架构 Kernel | 质量与性能 |
| 多卡 | 通信暴露 | TP 与拓扑匹配 | 单卡/双卡对照 |

## 面试题与答案

### 1\. Continuous Batching 为什么能提高吞吐？

它在每个调度迭代动态加入新请求、移除完成请求，使 Decode Batch 保持较高占用，减少静态 Batch 的空槽。

### 2\. TTFT 与 TPOT 分别受什么影响？

TTFT 主要受排队和 Prefill 影响；TPOT 反映生成阶段每个后续 Token 的节奏，受 Decode Batch、KV、内存带宽、抢占与调度影响。

### 3\. 为什么最大吞吐配置不一定是生产最优？

最大吞吐可能伴随不可接受的 TTFT/TPOT 尾延迟。生产应在 SLO 约束下最大化 Goodput。

### 4\. gpu\_memory\_utilization 越高越好吗？

不是。它会影响可用于 KV Cache 的容量，但过高会压缩运行余量，增加 OOM 或不稳定风险。

### 5\. Prefix Cache 什么时候有效？

多个请求共享可缓存的 Token Prefix 且 Prefill 成本显著时有效；随机、短或低重复 Prefix 的收益有限。

### 6\. 为什么 Tensor Parallel 可能降低单请求性能？

每层计算被切分后需要跨卡通信；若单卡已能高效运行，通信和同步成本可能超过并行收益。

## 课后练习

1. 扫描并发 1～64，画 TTFT P95、TPOT P95、Token/s 和 Goodput 曲线。
2. 构造短输入长输出、长输入短输出、长输入长输出三种负载。
1. 扫描 `max_num_batched_tokens` ，找到交互 SLO 下最优值。
2. 构造 80% 共享 Prefix 的请求，验证 Prefix Cache 命中收益。
1. 比较单卡和双卡 TP；分析通信与模型容量的权衡。
2. 故意降低 KV 容量，观察 Preemption、P95 和吞吐变化。
1. 让客户端与服务端分机，区分网络和服务时间。
2. 用 vLLM 官方 Benchmark 复测自写客户端，并解释口径差异。

## 项目验收 Checklist

- 已锁定 vLLM、PyTorch、CUDA、驱动和模型版本；
- 已记录服务完整启动命令和 /v1/models；
- 已固定输入/输出 Token 分布与请求到达模型；
- 已测 TTFT、TPOT、E2E、QPS、Token/s、成功率与 Goodput；
- 已扫描并发并找到 SLO 饱和拐点；
- 每次只修改一个 Scheduler/KV/并行参数；
- 已检查服务日志中的 OOM、Preemption 和 Engine 错误；
- 已排除客户端、Tokenizer 与网络瓶颈；
- CPU 模拟未被写成 vLLM/GPU 实测；
- NVLink、FP8、FP4 只在真实支持硬件上验证；
- 所有性能结论都附带模型、长度、并发、硬件和 SLO 边界。