---
title: "项目8：AI Systems Performance Engineering 综合项目"
source: "https://everythingai.top/learn/ai-systems-performance-engineering-text/project-08"
author:
published:
created: 2026-09-26
description: "面向 AI 工程学习者的专业课程学习空间。"
tags:
  - "clippings"
---
**导航**

本项目综合运用性能建模、GPU、分布式、编译器与推理系统知识，交付完整的 AI Systems Performance Engineering 优化方案。

## 项目定位

这是整套课程的最终工程验收。你需要选择一个真实训练或推理工作负载，从业务目标出发完成一次可复现的性能工程闭环：定义 SLO 与成本目标、建立可信基线、定位主瓶颈、提出可证伪假设、实施最小改动、验证正确性与收益、设计发布回滚，并交付完整证据包。

```
业务目标
  → 可测 SLO / Goodput / 成本
  → 冻结工作负载与环境
  → 正确性基线
  → Profile 找主瓶颈
  → 假设与实验矩阵
  → 优化实现
  → 性能 + 正确性 + 稳定性门禁
  → 灰度、观测、回滚
  → 复盘与知识沉淀
```

本项目不要求每位读者拥有相同 GPU。Level 0 使用纯 Python 完成方法和门禁；Level 1 在任意可用 NVIDIA GPU 上选择训练或推理工作负载；Level 2 仅在真实支持环境验证 NVLink/NVSwitch、RDMA、FP8/FP4、Transformer Engine、Disaggregated Inference 等专项能力。

## 学习目标

- 把性能问题翻译为业务 SLO、Goodput 和成本；
- 建立环境、数据、模型、精度和工作量受控的基线；
- 用系统、框架、Kernel 和分布式证据定位瓶颈；
- 用 Roofline、排队论、通信模型与 Amdahl 定律建立预测；
- 设计单变量、可重复、带正确性门禁的实验；
- 同时报告 P50/P95/P99、吞吐、显存、错误率、能耗和成本；
- 区分资料事实、模型推导、模拟结果和硬件实测；
- 形成可发布、可回滚、可持续防回归的工程交付。

## 前置知识

- 完成全部 28 课与前 7 个综合项目；
- 能使用至少一种 Profiler；
- 能阅读训练或推理日志；
- 对目标系统拥有合法的测试权限、模型/数据许可和隔离测试环境。

## 项目选题

从以下方向选择一个，不改变统一交付标准：

| 方向 | 示例目标 | 核心指标 |
| --- | --- | --- |
| 单 GPU 训练 | 提升稳态训练吞吐 | Step P95、Samples/s、显存、Loss |
| 多 GPU 训练 | 提升扩展效率 | Tokens/s、MFU、通信占比、慢 Rank |
| CUDA Kernel | 消除热点 | Kernel P50/P95、带宽、正确性 |
| 在线推理 | SLO 下最大 Goodput | TTFT、TPOT、P99、Token/s |
| KV Cache | 提升稳定并发容量 | KV Token、Preemption、P95 |
| P/D 解耦 | 降低干扰/成本 | Transfer、TTFT/TPOT、Goodput |
| MoE 推理 | 优化路由与 All-to-All | Expert Load、通信、Token/s |

## 最终交付物

```
capstone/
├── README.md
├── experiment.yaml
├── capstone_runner.py
├── baseline/
│   ├── environment.txt
│   ├── metrics.json
│   └── traces/
├── candidate/
│   ├── environment.txt
│   ├── metrics.json
│   └── traces/
├── reports/
│   ├── hypothesis_register.md
│   ├── comparison.md
│   ├── correctness.md
│   ├── cost_model.md
│   └── rollback.md
└── artifacts/
    └── commands.txt
```

## 课程总模型

### 端到端延迟

$$
T_{end}=T_{queue}+T_{data}+T_{compute}+T_{memory}+T_{communication}+T_{runtime}+T_{sync}
$$

### Goodput

$$
Goodput=Throughput\times SuccessRate\times SLOPassRate
$$

### Roofline

$$
Performance\le\min(PeakCompute,\ Bandwidth\times ArithmeticIntensity)
$$

### 通信

$$
T_{comm}\approx\alpha N_{messages}+\frac{Bytes}{BW_{effective}}
$$

### Amdahl

$$
S_{total}=\frac{1}{(1-p)+\frac{p}{S_{optimized}}}
$$

### 成本

$$
CostPerWork=\frac{InfrastructureCost+EnergyCost+FailureCost}{ValidWork}
$$

这些模型用于产生可验证预测。任何模型都必须与真实 Profile 对照，不能把公式结果直接写成硬件实测。

## 阶段一：项目章程

项目开始前写清：

```
project: example
owner: team
workload:
  type: training-or-inference
  model_revision: immutable-id
  dataset_revision: immutable-id
  precision: bf16
  input_distribution: fixed-and-versioned
slo:
  primary: "p95_step_ms <= target"
  secondary: "peak_memory_gib <= target"
correctness:
  metric: "loss-delta-or-output-match"
  threshold: 0.0
budget:
  max_gpu_hours: 0
  deadline: YYYY-MM-DD
```

必须有一个 Primary Metric 和至少两个 Guardrail。没有 Guardrail 的“更快”可能来自精度下降、丢请求、减少 Token、改变 Batch 或禁用必要功能。

## 阶段二：冻结工作负载与环境

### 环境快照

```
mkdir -p baseline candidate reports artifacts
{
  date -Iseconds
  uname -a
  lscpu
  free -h
  python --version
  nvidia-smi -L 2>/dev/null || true
  nvidia-smi topo -m 2>/dev/null || true
  nvidia-smi --query-gpu=name,driver_version,memory.total,pci.bus_id,pstate,power.limit --format=csv 2>/dev/null || true
  nvcc --version 2>/dev/null || true
  nsys --version 2>/dev/null || true
  ncu --version 2>/dev/null || true
  python -m pip freeze 2>/dev/null || true
} 2>&1 | tee baseline/environment.txt
```

还应保存 Git Commit、容器 Digest、模型 Revision、Tokenizer、数据 Hash、完整命令、环境变量、功耗/时钟策略和物理拓扑。

### 受控变量

- 相同请求/Token/样本工作量；
- 相同有效 Batch、序列长度和精度；
- 相同随机种子或可比较的数据分布；
- 相同正确性标准；
- 相同是否包含数据、日志、Checkpoint 和网络的计时边界；
- 相同机器、隔离和功耗条件；
- 只改变实验声明的变量。

## 阶段三：可信基线

基线至少包含：

1. Smoke Test：能运行并得到正确结果；
2. Warmup：排除初始化、Allocator、编译和连接建立；
1. 稳态：足够多的重复 Step/Request；
2. Tail：P95/P99 与错误率；
1. 容量：峰值显存、Host Memory、KV/Activation；
2. Profile：至少一个代表性窗口；
1. 重复：相同条件下 3～5 次独立运行。

性能数字必须保存原始样本，而不只保存平均值。

## 阶段四：瓶颈定位

### 从外到内

```
业务 SLO / Goodput / 成本
  → 服务排队、数据、Checkpoint、故障
  → 框架 Op、Runtime、Allocator、编译
  → GPU Kernel、Memory、Occupancy、Tensor Core
  → 多 GPU Collective、拓扑、慢 Rank
  → 硬件功耗、温度、ECC、PCIe/RDMA
```

### 工具选择

| 问题 | 工具 |
| --- | --- |
| CPU/GPU 时间线、Launch、空洞 | Nsight Systems |
| Kernel、Memory、Roofline | Nsight Compute |
| PyTorch Op、Shape、Memory | PyTorch Profiler |
| NCCL/多 Rank | NCCL 日志、nccl-tests、Trace |
| vLLM 服务 | 官方 Bench、服务日志、Prometheus |
| 正确性 | Unit/Golden/Convergence/Quality Eval |

不要把所有 Profiler 同时打开。采集本身有开销，应使用短窗口并保留无 Profiler 的最终 Benchmark。

## 阶段五：假设登记

保存为 `reports/hypothesis_register.md` ：

| ID | 证据 | 假设 | 改动 | 预测 | Guardrail | 结果 | | ---------------------------- | ---------------- | ----------------- | --------------- | --------- | ------- | -- | | H1 | GPU 空洞 20% | 数据供给不足 | Worker/Prefetch | P50 -10% | Loss/内存 | 待测 | | H2 | Kernel Launch 密集 | Fusion 可减少开销 | Compile/Fusion | P50 -8% | 数值误差 | 待测 | | H3 | NCCL 暴露 | Bucket/Overlap 不佳 | 调整重叠 | Step -12% | 峰值显存 | 待测 |

每个假设必须可证伪，并写出“如果没有收益，下一步检查什么”。

## Level 0：完整性能实验与回归门禁

下面的纯 Python 程序提供一个可运行的参考实验：Baseline 对列表做多次中间分配，Candidate 将缩放和激活融合到一次遍历；两者完成相同的 RMS Normalization + SiLU 工作。程序执行热身、重复计时、正确性、P50/P95、吞吐、Goodput、Amdahl 端到端预测和自动门禁。

保存为 `capstone_runner.py` ：

```
#!/usr/bin/env python3
import argparse
import json
import math
import operator
import platform
import random
import statistics
import time
from pathlib import Path

def silu(value):
    return value / (1.0 + math.exp(-max(-60.0, min(60.0, value))))

def baseline(values, epsilon):
    squares = [value * value for value in values]
    mean_square = sum(squares) / len(squares)
    inverse_rms = 1.0 / math.sqrt(mean_square + epsilon)
    normalized = [value * inverse_rms for value in values]
    return [silu(value) for value in normalized]

def candidate(values, epsilon):
    mean_square = sum(map(operator.mul, values, values)) / len(values)
    inverse_rms = 1.0 / math.sqrt(mean_square + epsilon)
    return [silu(value * inverse_rms) for value in values]

def percentile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return values[low]
    return values[low] * (high - position) + values[high] * (position - low)

def benchmark(function, values, epsilon, warmup, repeats):
    for _ in range(warmup):
        function(values, epsilon)
    samples = []
    output = None
    for _ in range(repeats):
        start = time.perf_counter()
        output = function(values, epsilon)
        samples.append(time.perf_counter() - start)
    return output, samples

def summarize(samples, elements):
    p50 = percentile(samples, 0.50)
    return {
        "p50_ms": p50 * 1000,
        "p95_ms": percentile(samples, 0.95) * 1000,
        "mean_ms": statistics.mean(samples) * 1000,
        "elements_per_second_at_p50": elements / p50,
        "raw_seconds": samples,
    }

def main():
    parser = argparse.ArgumentParser(description="AI 性能工程结课实验")
    parser.add_argument("--elements", type=int, default=200000)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--max-relative-error", type=float, default=1e-12)
    parser.add_argument("--max-regression", type=float, default=0.05)
    parser.add_argument("--target-speedup", type=float, default=1.0)
    parser.add_argument("--hotspot-fraction", type=float, default=0.60)
    parser.add_argument("--output", default="reports/capstone.json")
    args = parser.parse_args()
    if min(args.elements, args.repeats, args.epsilon, args.target_speedup) <= 0:
        raise ValueError("规模、重复、epsilon 和目标加速必须大于 0")
    if args.warmup < 0 or not 0 < args.hotspot_fraction <= 1:
        raise ValueError("warmup 必须非负，hotspot-fraction 必须位于 (0, 1]")

    rng = random.Random(2026)
    values = [rng.uniform(-3.0, 3.0) for _ in range(args.elements)]
    baseline_output, baseline_samples = benchmark(
        baseline, values, args.epsilon, args.warmup, args.repeats
    )
    candidate_output, candidate_samples = benchmark(
        candidate, values, args.epsilon, args.warmup, args.repeats
    )

    max_abs = max(abs(left - right)
                  for left, right in zip(baseline_output, candidate_output))
    denominator = max(max(abs(value) for value in baseline_output), 1e-30)
    max_relative = max_abs / denominator
    baseline_summary = summarize(baseline_samples, args.elements)
    candidate_summary = summarize(candidate_samples, args.elements)
    speedup = baseline_summary["p50_ms"] / candidate_summary["p50_ms"]
    predicted_end_to_end_speedup = 1.0 / (
        (1 - args.hotspot_fraction) + args.hotspot_fraction / speedup
    )
    correctness_pass = max_relative <= args.max_relative_error
    performance_pass = (
        candidate_summary["p50_ms"]
        <= baseline_summary["p50_ms"] * (1 + args.max_regression)
        and speedup >= args.target_speedup
    )

    result = {
        "workload": {
            "elements": args.elements,
            "operation": "RMSNorm + SiLU",
            "seed": 2026,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "correctness": {
            "max_absolute_error": max_abs,
            "max_relative_error": max_relative,
            "threshold": args.max_relative_error,
            "pass": correctness_pass,
        },
        "speedup_p50": speedup,
        "hotspot_fraction": args.hotspot_fraction,
        "amdahl_predicted_end_to_end_speedup": predicted_end_to_end_speedup,
        "gate": {
            "performance_pass": performance_pass,
            "correctness_pass": correctness_pass,
            "pass": performance_pass and correctness_pass,
        },
        "warning": "这是 CPU 教学负载；结果不能外推到 GPU Kernel 或真实模型。",
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if not result["gate"]["pass"]:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
```

### 运行命令

```
mkdir -p reports
python capstone_runner.py \
  --elements 200000 --warmup 3 --repeats 20 \
  --target-speedup 1.0 --max-regression 0.05 \
  --output reports/capstone.json
```

如果当前 CPU、Python 版本或频率抖动导致 Candidate 未达到目标，程序会以状态码 2 退出。这不是脚本失败，而是门禁按设计拒绝没有证据的优化。先增加重复、隔离噪声并查看原始样本；不要降低阈值只为获得 PASS。

### 预期现象

Candidate 少创建一个中间列表，通常减少分配和内存流量，但 `math.fsum` 与 Python 解释器成本可能让它在部分环境中没有优势。这正是性能工程的核心：代码看起来更“优化”不等于实测更快。

## Level 1：替换为真实 GPU 工作负载

保留 `capstone_runner.py` 的报告和门禁结构，把 `baseline()` 、 `candidate()` 替换为目标工作负载，并遵守：

- CUDA 用 Event 或同步边界计时；
- PyTorch 训练同时测 Loss/Gradient/Convergence；
- 推理同时测 TTFT/TPOT/输出一致性；
- 多 GPU 同时测全局吞吐、慢 Rank 和通信；
- Kernel 同时跑 Compute Sanitizer、误差与 Nsight Compute；
- 每个结果保存硬件、版本、Shape、精度和原始样本。

### GPU 能力检测

```
python - <<'PY'
try:
    import torch
except ImportError:
    print("PyTorch 未安装，使用 Level 0")
else:
    print("torch", torch.__version__)
    print("cuda runtime", torch.version.cuda)
    print("cuda available", torch.cuda.is_available())
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            print(index, torch.cuda.get_device_name(index),
                  torch.cuda.get_device_capability(index),
                  round(torch.cuda.get_device_properties(index).total_memory / 2**30, 2))
PY
```

## Level 2：架构专项

| 架构 | 例子 | 合法专项 |
| --- | --- | --- |
| Ampere | RTX 3080/3090、A100 | TF32、FP16/BF16（按卡）、Ampere Async Copy |
| Ada Lovelace | RTX 4090、L40/L40S | FP16/BF16、PCIe 推理与编译器优化 |
| Hopper | H100/H200 | FP8、Transformer Engine、TMA、NVLink/NVSwitch |
| Blackwell | RTX 5090、B100/B200/GB200 | FP8/FP4、新 Tensor Core/Fabric，依工具链 |

RTX 4090 不是 Blackwell。消费卡与数据中心 GPU 即使属于相近代际，也可能在显存、互联、ECC、虚拟化和低精度能力上不同。

专项实验必须有能力检测与跳过逻辑。例如：没有 NVLink 时测 PCIe 基线并标记 `SKIPPED_NVLINK` ，而不是把 Host Staging 写成 NVLink 模拟。

## 阶段六：实验矩阵

### 单变量实验

| Run | 改变量 | 工作量 | 正确性 | P50/P95 | 容量 | 结论 | | ------------------------------ | ---- | -- | --------- | -- | -- | ----- | | B0 | 无 | 固定 | Pass | 实测 | 实测 | 基线 | | C1 | 变量 A | 相同 | Pass/Fail | 实测 | 实测 | 接受/拒绝 | | C2 | 变量 B | 相同 | Pass/Fail | 实测 | 实测 | 接受/拒绝 |

### 交互实验

只有单变量结果明确后，才组合 AMP + Compile、Batch + Scheduler、FSDP + Checkpoint 或 TP + KV 配置。组合收益不能简单相乘，可能存在重复优化和资源冲突。

## 阶段七：正确性与稳定性

### 正确性门禁

训练：

- Loss/Gradient 有限；
- 固定小数据上的趋势一致；
- 精度误差在阈值内；
- 完整收敛或质量代理不退化。

推理：

- 固定 Prompt 的 Greedy 输出或 Logit 差异；
- Token 数、Stop、EOS 与结构化输出一致；
- 量化前后质量集；
- 流式顺序、取消和重试正确。

系统：

- OOM/Timeout/错误率；
- 30～120 分钟 Soak；
- Worker/网络/磁盘故障注入；
- 热降频、功耗、ECC 和内存泄漏。

### 统计规则

- 保存原始样本；
- 报告中位数与尾部；
- 独立运行至少 3 次；
- 对收益小于环境噪声的改动不下结论；
- 不删除“难看”的离群点，先解释来源；
- 预先定义门禁，不在看结果后改阈值。

## 阶段八：成本与收益

### 训练

$$
CostPerBillionTokens=\frac{GPUHourPrice\times GPUCount\times Hours}{Tokens/10^9}
$$

### 推理

$$
CostPerMillionOutputTokens=\frac{HourlyFleetCost}{OutputTokensPerHour/10^6}
$$

必须按有效 Token 计算：失败、重试、SLO Miss 与空闲容量都计入成本。若优化需要更多工程维护、专属硬件或可靠性风险，报告中也应列入。

### 投资回收

$$
Payback=\frac{EngineeringCost}{MonthlyInfrastructureSavings}
$$

没有稳定工作量或收益小于噪声时，不应给出虚假的精确回收期。

## 阶段九：发布、回滚与持续回归

### 发布计划

1. 离线正确性与性能门禁；
2. Shadow Traffic；
1. 1% Canary；
2. 分阶段扩大；
1. 观察一个完整业务周期；
2. 固化默认配置。

### 自动回滚条件

- Primary SLO 连续超阈值；
- 错误/OOM/NaN 上升；
- Goodput 或成本退化；
- 输出质量门禁失败；
- GPU/Host Memory 持续增长；
- 新配置无法在目标架构安全启动。

### 回归层级

| 频率 | 测试 |
| --- | --- |
| 每次提交 | 小规模正确性、语法、Smoke |
| 每日 | 固定机器 Microbenchmark |
| 每周 | 端到端、并发扫描、Profiler |
| 发布前 | 质量、Soak、故障注入、成本 |

## 结果报告模板

### 1\. 结论

一句话写清在什么硬件、版本、工作负载和 SLO 下，Candidate 相对 Baseline 改善了什么。

### 2\. 证据

| 指标 | Baseline | Candidate | 变化 | 门禁 | | ---------------------------- | -- | -- | -- | --------- | | Primary | 实测 | 实测 | 计算 | Pass/Fail | | P95/P99 | 实测 | 实测 | 计算 | Pass/Fail | | Goodput | 实测 | 实测 | 计算 | Pass/Fail | | Peak Memory | 实测 | 实测 | 计算 | Pass/Fail | | Quality/Loss | 实测 | 实测 | 计算 | Pass/Fail | | Cost/Work | 实测 | 实测 | 计算 | Pass/Fail |

### 3\. 原因

用 Profile 说明收益来自哪里；将“观察到”与“推断”分开。

### 4\. 边界

列出未测试架构、长度、并发、故障和版本。不要把一次环境结果写成所有 GPU 的通用结论。

### 5\. 决策

接受、拒绝、继续实验或仅对特定架构启用，并附回滚条件。

## 常见错误与排查

### 1\. 从 GPU 利用率直接下结论

利用率是现象，不是业务目标。结合 Goodput、SLO、Kernel、Memory、通信和成本。

### 2\. 优化时改变工作量

检查 Token、Batch、Sequence、精度、请求成功数、Drop Last、Padding、Stop 和数据过滤。

### 3\. 只报告最佳一次

保存全部独立运行与原始样本，报告中位数和尾部。最佳值可作上界观察，不能作默认结论。

### 4\. Profiler 数字当最终性能

Profiler 会扰动程序。用它找原因，用关闭 Profiler 的受控 Benchmark 给结论。

### 5\. 正确性门禁最后才补

正确性应先于性能。没有 Golden/Convergence/Quality 标准，任何加速都可能无效。

### 6\. 一次打开所有优化

先做单变量，再组合。否则无法归因，也无法在回归时快速回滚。

### 7\. 使用错误架构假设

根据设备 ID、Compute Capability、互联和官方支持矩阵检测，不根据商品名称模糊归类。

### 8\. 把模拟写成硬件结果

模拟用于设计和边界推导；硬件结论必须有环境、命令、日志、原始测量与工具证据。

## 优化前后对照

| 维度 | 低质量做法 | 工程化做法 |
| --- | --- | --- |
| 目标 | “GPU 更快” | SLO/Goodput/成本 |
| 基线 | 单次运行 | 冻结环境、多次稳态 |
| 诊断 | 猜参数 | Profile + 模型 |
| 改动 | 多开关 | 可证伪单变量 |
| 正确性 | 能运行 | 数值/质量/收敛门禁 |
| 性能 | 平均值 | P50/P95/P99、原始样本 |
| 范围 | 宣称通用 | 明确硬件/版本/Shape |
| 发布 | 直接替换 | Canary、观测、回滚 |
| 沉淀 | 截图 | 配置、脚本、Trace、报告 |

## 面试题与答案

### 1\. 性能工程第一步是什么？

定义业务目标、工作量和正确性边界，再建立可信基线；不是先改代码或开优化开关。

### 2\. 如何证明某优化真的有效？

受控变量下重复测量，收益超过噪声，正确性与稳定性门禁通过，并用 Profile 解释原因。

### 3\. Goodput 为什么比峰值吞吐重要？

Goodput 只计成功且满足 SLO 的有效工作，能反映错误、重试、排队和尾延迟。

### 4\. Roofline 如何指导优化？

它用算术强度判断工作负载更可能受计算峰值还是带宽上限约束，帮助选择减少字节、增加复用或提高计算效率。

### 5\. 为什么 Kernel 加速没有转化为端到端加速？

热点占比有限，或加速后瓶颈转移到数据、通信、Runtime、排队等其他阶段，受 Amdahl 定律限制。

### 6\. 如何处理 3% 收益但环境波动 5%？

不能下结论。提高隔离、增加独立重复、固定功耗与输入，使用统计区间；仍无法区分就拒绝该收益声明。

### 7\. 为什么发布计划是性能项目的一部分？

性能优化可能改变数值、内存、并发、故障和架构兼容性。没有 Canary、监控与回滚，就不是可上线工程。

### 8\. 何时应停止继续优化？

达到 SLO/成本目标、剩余热点收益低于工程成本、改动风险过高或测量噪声大于潜在收益时。

## 课后练习

1. 选择一个真实训练或推理任务，完成项目章程。
2. 建立 Baseline 并保存三次独立运行原始样本。
1. 用两个不同层级的 Profiler 交叉定位同一热点。
2. 为三个候选优化写假设、预测和失败标准。
1. 实施一个低风险优化，并用门禁接受或拒绝。
2. 把硬件/版本/Shape 支持矩阵写进自动检测。
1. 计算优化前后 Cost/Valid Token 或 Cost/Training Token。
2. 设计 Canary、自动回滚与每周回归任务。

## 项目验收 Checklist

- 有明确 Primary Metric、SLO、Guardrail 与预算；
- 工作量、模型、数据、精度、Batch 和计时边界已冻结；
- 环境、版本、拓扑、功耗和完整命令已保存；
- Baseline 有热身、稳态、尾部和独立重复；
- 至少一个系统/框架工具和一个 Kernel/通信工具提供证据；
- 假设可证伪，且每次实验只改变声明变量；
- 正确性、质量或收敛门禁先于性能结论；
- 报告 P50/P95/P99、吞吐、Goodput、容量和错误；
- 模拟、推导与硬件实测清楚区分；
- 架构专项有能力检测、跳过与不可等价边界；
- 已完成成本、发布、观测和回滚设计；
- 证据包可由另一位工程师复现。

## 28 课总索引

1. 第1课：AI Systems Performance Engineering概述
2. 第2课：AI系统性能指标体系
1. 第3课：现代AI系统全栈架构
2. 第4课：DeepSeek案例——软件优化如何突破硬件限制
1. 第5课：GPU架构深度解析
2. 第6课：GPU Memory体系优化
1. 第7课：Tensor Core与低精度计算优化
2. 第8课：NVLink、NVSwitch与AI集群互联
1. 第9课：Linux、Docker、Kubernetes GPU性能优化
2. 第10课：CUDA编程模型
1. 第11课：GPU Memory访问优化
2. 第12课：GPU性能分析方法
1. 第13课：Kernel Fusion与高性能算子开发
2. 第14课：Triton Kernel开发
1. 第15课：大模型分布式训练体系
2. 第16课：NCCL深度优化
1. 第17课：GPU通信性能优化
2. 第18课：FSDP与超大模型训练优化
1. 第19课：PyTorch性能分析体系
2. 第20课：torch.compile深度解析
1. 第21课：CUDA Graph与Runtime优化
2. 第22课：AI Runtime优化思想
1. 第23课：LLM推理系统架构
2. 第24课：Continuous Batching与Scheduler优化
1. 第25课：KV Cache性能优化
2. 第26课：Speculative Decoding与推理加速
1. 第27课：Disaggregated Inference架构
2. 第28课：MoE推理与未来AI Runtime

## 8 个综合项目总索引

1. 项目1：GPU性能分析实验
2. 项目2：CUDA Kernel优化项目
1. 项目3：PyTorch训练性能优化
2. 项目4：7B模型训练性能Benchmark
1. 项目7：Disaggregated Inference系统设计
2. 项目8：AI Systems Performance Engineering综合项目