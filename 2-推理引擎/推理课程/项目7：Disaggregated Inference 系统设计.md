---
title: "项目7：Disaggregated Inference 系统设计"
source: "https://everythingai.top/learn/ai-systems-performance-engineering-text/project-07"
author:
published:
created: 2026-09-26
description: "面向 AI 工程学习者的专业课程学习空间。"
tags:
  - "clippings"
---
**导航**

本项目设计一套 Prefill 与 Decode 分离的推理系统，完成资源规划、通信分析与高可用方案。

## 项目定位

本项目要求你设计一个可落地的 Prefill/Decode 解耦推理系统。最终产物不是一张概念图，而是一份带容量模型、KV 传输预算、路由规则、故障语义、SLO、观测指标、回退路径和验证计划的系统设计。

```
请求入口
  → Prefill Router → Prefill Worker Pool
  → KV Transfer / KV Metadata Commit
  → Decode Router → Decode Worker Pool
  → Streaming Response
```

Prefill 通常是计算密集的大矩阵阶段，Decode 通常是内存带宽敏感的小步迭代。解耦允许两类资源独立扩缩和调度，但引入 KV 传输、跨池排队、一致性、故障恢复和更多网络跳数。只有收益超过这些成本时，解耦才成立。

Level 0 使用纯 Python 容量规划器，任何机器可运行。Level 1 使用单机或 PCIe 多 GPU 验证调度和传输成本。Level 2 才验证 GPUDirect RDMA、NIXL、NVLink/NVSwitch 或跨节点高速网络。

## 学习目标

- 从 Prefill、KV Transfer、Decode 三阶段建立延迟和吞吐模型；
- 使用 Erlang-C 估计排队等待和稳定性边界；
- 计算 KV 传输字节、链路占用和最小 Decode 容量；
- 设计 Request ID、KV Handle、幂等、超时和重试语义；
- 设计 P/D 独立扩缩、拓扑感知路由和过载保护；
- 区分模拟、单机验证与真实 RDMA/NVLink 实测；
- 用端到端 Goodput 决定是否采用解耦，而不是只比较阶段峰值。

## 前置知识

- 第 23、25、27 课与项目 5、6；
- 排队论基础、HTTP/gRPC、KV Cache、GPU 拓扑与分布式系统；
- 可选 vLLM、NVIDIA Dynamo/NIXL 或等价的受支持框架。

## 项目交付物

```
disaggregated-inference-design/
├── disagg_capacity_planner.py
├── architecture.md
├── api_contract.md
├── failure_matrix.md
├── deployment/
│   ├── prefill.yaml
│   ├── decode.yaml
│   └── router.yaml
└── reports/
    ├── capacity.json
    ├── transfer_sweep.json
    └── validation_report.md
```

## 核心直觉：把两种不同形态的工作独立排队

Monolithic Worker 同时处理 Prefill 与 Decode。长 Prefill 可能干扰正在生成的请求；Decode 又可能让大 GEMM 难以形成高效 Batch。解耦后：

$$
T_{TTFT}=T_{queue,p}+T_{prefill}+T_{transfer}+T_{queue,d}+T_{first\ decode}
$$

$$
T_{E2E}=T_{TTFT}+(N_{out}-1)\times TPOT
$$

收益来自阶段隔离、独立 Batch、异构硬件与独立扩缩；成本来自 KV 传输、额外排队、网络、元数据和故障处理。

解耦必要条件不是“Prefill 和 Decode 不同”，而是：

$$
Gain_{isolation}+Gain_{scaling}>Cost_{transfer}+Cost_{queue}+Cost_{operation}
$$

## KV 传输模型

Prefill 产生的 KV 字节：

$$
Bytes_{KV}=S_{input}\times2\times L\times H_{kv}\times D_{head}\times B_{dtype}
$$

链路有效带宽为 $BW_{effective}$ ，固定开销为 $T_{fixed}$ ：

$$
T_{transfer}=T_{fixed}+\frac{Bytes_{KV}}{BW_{effective}}
$$

协议 Header、序列化、注册内存、NUMA、PCIe Switch、网络拥塞和小消息效率都会让有效带宽低于标称带宽。

链路长期字节速率：

$$
BW_{required}=ArrivalRate\times Bytes_{KV}
$$

仅看单请求传输时间不足以判断系统是否稳定，还必须检查总吞吐和突发流量。

## 排队稳定性

Prefill 平均服务时间 $S_p$ 、副本数 $c_p$ 、到达率 $\lambda$ ：

$$
\rho_p=\frac{\lambda S_p}{c_p}
$$

Decode 同理。若 $\rho\ge1$ ，稳态队列没有有限均值。生产系统不应长期运行在 100% 理论利用率，应为流量突发、尾部请求和故障副本保留余量。

## Level 0：完整容量规划器

下面的脚本计算 KV 传输、阶段利用率、Erlang-C 平均排队、TTFT/E2E 近似和链路占用。它用于方案筛选，不替代离散事件模拟与真实集群压测。

保存为 `disagg_capacity_planner.py` ：

```
#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

def erlang_c_wait(arrival_rate, service_seconds, servers):
    offered = arrival_rate * service_seconds
    rho = offered / servers
    if rho >= 1:
        return math.inf, rho
    series = sum(offered ** n / math.factorial(n) for n in range(servers))
    tail = offered ** servers / math.factorial(servers) / (1 - rho)
    probability_wait = tail / (series + tail)
    wait_seconds = probability_wait * service_seconds / (servers * (1 - rho))
    return wait_seconds, rho

def main():
    parser = argparse.ArgumentParser(description="P/D 解耦容量规划器")
    parser.add_argument("--arrival-rate", type=float, default=4.0)
    parser.add_argument("--prefill-workers", type=int, default=2)
    parser.add_argument("--decode-workers", type=int, default=4)
    parser.add_argument("--prefill-ms", type=float, default=200.0)
    parser.add_argument("--decode-ms-per-token", type=float, default=20.0)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--input-tokens", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype-bytes", type=float, default=2.0)
    parser.add_argument("--link-gbps", type=float, default=100.0)
    parser.add_argument("--link-efficiency", type=float, default=0.70)
    parser.add_argument("--fixed-transfer-ms", type=float, default=0.2)
    parser.add_argument("--output", default="reports/capacity.json")
    args = parser.parse_args()

    positive = [args.arrival_rate, args.prefill_workers, args.decode_workers,
                args.prefill_ms, args.decode_ms_per_token, args.output_tokens,
                args.input_tokens, args.layers, args.kv_heads, args.head_dim,
                args.dtype_bytes, args.link_gbps]
    if any(value <= 0 for value in positive):
        raise ValueError("容量参数必须大于 0")
    if not 0 < args.link_efficiency <= 1:
        raise ValueError("link-efficiency 必须位于 (0, 1]")

    kv_bytes_per_token = 2 * args.layers * args.kv_heads * args.head_dim * args.dtype_bytes
    kv_bytes = args.input_tokens * kv_bytes_per_token
    effective_bytes_per_second = args.link_gbps * 1e9 / 8 * args.link_efficiency
    transfer_seconds = args.fixed_transfer_ms / 1000 + kv_bytes / effective_bytes_per_second
    required_link_bytes_per_second = args.arrival_rate * kv_bytes
    link_utilization = required_link_bytes_per_second / effective_bytes_per_second

    prefill_service = args.prefill_ms / 1000
    decode_service = args.decode_ms_per_token / 1000 * args.output_tokens
    prefill_wait, prefill_rho = erlang_c_wait(
        args.arrival_rate, prefill_service, args.prefill_workers
    )
    decode_wait, decode_rho = erlang_c_wait(
        args.arrival_rate, decode_service, args.decode_workers
    )
    stable = all(value < 1 for value in (prefill_rho, decode_rho, link_utilization))

    if stable:
        ttft = (prefill_wait + prefill_service + transfer_seconds +
                decode_wait + args.decode_ms_per_token / 1000)
        e2e = ttft + (args.output_tokens - 1) * args.decode_ms_per_token / 1000
    else:
        ttft = None
        e2e = None

    result = {
        "stable": stable,
        "kv": {
            "bytes_per_token": kv_bytes_per_token,
            "mib_per_request": kv_bytes / 2**20,
            "required_link_gbps": required_link_bytes_per_second * 8 / 1e9,
            "effective_link_gbps": args.link_gbps * args.link_efficiency,
            "link_utilization": link_utilization,
            "transfer_ms": transfer_seconds * 1000,
        },
        "prefill": {
            "workers": args.prefill_workers,
            "utilization": prefill_rho,
            "mean_queue_ms": None if math.isinf(prefill_wait) else prefill_wait * 1000,
        },
        "decode": {
            "workers": args.decode_workers,
            "utilization": decode_rho,
            "mean_queue_ms": None if math.isinf(decode_wait) else decode_wait * 1000,
        },
        "latency_estimate": {
            "ttft_ms": None if ttft is None else ttft * 1000,
            "e2e_ms": None if e2e is None else e2e * 1000,
        },
        "warning": (
            "Erlang-C 假设泊松到达、指数服务时间与同质 Worker；"
            "未包含 Batch、抢占、长尾、KV 元数据和故障。"
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

```
mkdir -p reports
python disagg_capacity_planner.py \
  --arrival-rate 4 --prefill-workers 2 --decode-workers 16 \
  --prefill-ms 200 --decode-ms-per-token 20 --output-tokens 128 \
  --input-tokens 2048 --link-gbps 100 --link-efficiency 0.70 \
  --output reports/capacity.json
```

扫描到达率、P/D 副本数和链路：

```
for rate in 1 2 4 8 16; do
  python disagg_capacity_planner.py --arrival-rate "$rate" \
    --prefill-workers 2 --decode-workers 16 \
    --output "reports/rate_${rate}.json"
done

for link in 16 32 64 100 200 400; do
  python disagg_capacity_planner.py --link-gbps "$link" \
    --output "reports/link_${link}.json"
done
```

### 预期现象

Decode 服务时间是“每 Token 时间 × 输出 Token”，因此长输出可能需要明显更多 Decode 副本。若任何阶段或链路利用率达到 1，规划器把稳态延迟输出为 `null` ，避免在不稳定队列上制造有限平均延迟。

## 系统架构设计

### 控制面与数据面

```
Control Plane
├─ Worker Registry / Health / Capability
├─ Model Revision / KV Layout Version
├─ Autoscaler / Placement / Topology
└─ Config Rollout / Drain / Rollback

Data Plane
Client → Gateway → Prefill Router → P Worker
                                  ├─ KV Data → Transfer Fabric → D Worker
                                  └─ KV Commit → Metadata Service/Router
Client ← Streaming Gateway ← Decode Router ← D Worker
```

KV 数据路径与元数据路径应分开设计。大块 KV 不应绕经普通控制面数据库；控制面只保存 Handle、位置、版本、长度、校验和、租约和状态。

### Request/KV 状态机

```
RECEIVED
  → PREFILL_ASSIGNED
  → PREFILL_RUNNING
  → KV_TRANSFERRING
  → KV_COMMITTED
  → DECODE_ASSIGNED
  → STREAMING
  → COMPLETED

任意阶段 → CANCELLED / TIMED_OUT / FAILED
```

Decode Worker 只有在 KV 完整写入、校验通过并提交元数据后才能读取。先暴露 Handle、后完成传输会产生部分 KV 可见性问题。

## API 契约

### Prefill 请求

```
{
  "request_id": "uuid",
  "model_revision": "immutable-hash",
  "input_token_ids": [1, 2, 3],
  "kv_layout_version": "v1",
  "deadline_unix_ms": 0,
  "preferred_decode_pool": "pool-a"
}
```

### KV Commit

```
{
  "request_id": "uuid",
  "kv_handle": "opaque-handle",
  "model_revision": "immutable-hash",
  "token_count": 2048,
  "dtype": "bf16",
  "bytes": 268435456,
  "checksum": "...",
  "locations": ["worker-d-17"],
  "lease_expiry_unix_ms": 0,
  "status": "COMMITTED"
}
```

`request_id` 必须幂等。重试 Prefill 时，新 Attempt 不能让旧 KV Commit 覆盖新状态；可使用 `(request_id, attempt_id, generation)` 做条件提交。

## 路由策略

### Prefill Router

评分可综合：

$$
Score_p=w_1Q_p+w_2EstimatedPrefill+w_3MemoryPressure+w_4TransferDistance
$$

### Decode Router

优先级顺序：

1. 已持有目标 KV 的 Decode Worker；
2. 同 NVLink/NVSwitch 域或同节点；
1. 同机架 RDMA；
2. 需要跨域传输但仍满足 Deadline；
1. 拒绝、降级或回退 Monolithic。

路由不能只看队列长度。一个短队列但 KV 很远的 Worker，端到端可能更慢。

## 扩缩容

Prefill 与 Decode 独立扩缩：

- Prefill：队列 Token、Prefill GPU 时间、TTFT Budget；
- Decode：Active Sequences、Output Token/s、TPOT、KV 占用；
- Transfer：排队字节、有效带宽、P95 传输时间；
- 全局：Goodput、错误率、Deadline Miss、成本/Token。

扩容 Decode Worker 后并不会自动获得现有 KV。需要 Drain、复制、迁移或让新 Worker 只接新请求。

## 故障矩阵

| 故障 | 可见症状 | 正确处理 |
| --- | --- | --- |
| Prefill Worker 崩溃 | 无 Commit | 同 Attempt 超时，幂等重试 |
| 传输中断 | KV 不完整 | 不提交 Handle，清理临时 Buffer |
| Commit 后 D Worker 崩溃 | 流中断 | 有副本则重路由；否则重做 Prefill |
| Router 重启 | 状态丢失风险 | 状态外置或从 Worker/租约恢复 |
| Client 取消 | 无人消费 KV | 传播取消并释放 KV |
| Model Revision 不同 | 错误输出/崩溃 | 强校验并拒绝路由 |
| 网络分区 | 超时、积压 | 熔断、Deadline、回退/拒绝 |
| 慢 Worker | P99 恶化 | Deadline-aware 路由与 Drain |

“自动重试所有失败”可能放大负载。重试必须受 Deadline、Budget、Attempt 和幂等约束。

## Level 1：单机与常见 GPU 验证

没有高速互联时仍可验证：

1. 在 CPU 内存中生成等大小 KV Buffer；
2. 测量本机内存拷贝和 PCIe GPU↔Host/GPU↔GPU；
1. 用独立队列模拟 P/D Worker；
2. 注入延迟、失败、取消和重复 Commit；
1. 用项目 5 的流式压测比较 Monolithic 与模拟 Disaggregated 路由。

环境记录：

```
mkdir -p reports
{
  date -Iseconds
  nvidia-smi -L
  nvidia-smi topo -m
  nvidia-smi nvlink -s 2>/dev/null || true
  ip -br link
  rdma link 2>/dev/null || true
} 2>&1 | tee reports/environment.txt
```

双 RTX 3080 通常以 PCIe 为跨 GPU 路径。这个实验能验证慢链路回退和路由策略，不能等价模拟 NVLink、NVSwitch 或 GPUDirect RDMA。

## Level 2：真实框架与高速互联

可选择当前受支持的 vLLM Disaggregated Serving、NVIDIA Dynamo/NIXL 或等价实现。由于 CLI、Connector 和部署拓扑持续变化，必须：

1. 锁定框架 Release/Commit；
2. 保存官方示例原始配置；
1. 运行框架自带 Smoke Test；
2. 确认 KV Layout、Model Revision、Dtype 与 Connector 一致；
1. 保存完整启动命令和日志；
2. 用相同模型/负载对比 Monolithic。

### 硬件边界

| 路径 | 可验证内容 | 不可外推 |
| --- | --- | --- |
| 单机 CPU Copy | 协议、状态机 | GPUDirect 带宽 |
| 双消费 GPU PCIe | GPU P2P/Host Staging | NVSwitch 全互联 |
| RTX 3090 特定 NVLink | 单机 NVLink 对照 | 数据中心 NVSwitch/RDMA |
| H100/H200 NVLink/NVSwitch | Hopper 数据中心路径 | Blackwell NVLink 5 |
| B100/B200/GB200 | Blackwell Fabric | 消费级 RTX 5090 等价能力 |

RTX 4090 是 Ada Lovelace，不是 Blackwell；RTX 5090 属于 Blackwell，但消费卡不等于 GB200 NVL 系统。

## 验证计划

### 功能验证

- Token 与 Monolithic 基线一致；
- Model Revision/Dtype/Layout 不一致被拒绝；
- Client Cancel 能释放 P/D/KV；
- 重复请求不会重复 Commit；
- 部分传输永不对 Decode 可见；
- Worker Drain 不丢正在生成的请求。

### 性能验证

| 负载 | Monolithic | Disaggregated | 关注指标 | | --------------------------------- | -- | --- | ------------- | | 短输入长输出 | 基线 | P/D | TPOT、D 利用率 | | 长输入短输出 | 基线 | P/D | TTFT、Transfer | | 混合长度 | 基线 | P/D | 隔离、Goodput | | 突发 | 基线 | P/D | 队列、P99、拒绝 | | Worker 故障 | 基线 | P/D | 恢复时间、重复工作 |

### 观测指标

- `prefill_queue_tokens` 、 `prefill_duration` ；
- `kv_transfer_bytes` 、 `kv_transfer_duration` 、 `kv_transfer_failures` ；
- `decode_active_sequences` 、 `decode_tpot` ；
- `kv_bytes_in_use` 、 `kv_lease_expired` ；
- `request_attempts` 、 `deadline_miss` 、 `cancel_propagation_ms` ；
- TTFT/TPOT/E2E P50/P95/P99、Goodput 与成本/Token。

每个 Trace 应以 Request ID 串联 Gateway、P Router、P Worker、Transfer、D Router 与 D Worker。

## 结果分析

### 设计决策表

| 决策 | 候选 | 选择证据 |
| --- | --- | --- |
| P:D 比例 | 1:1、1:2、1:4 | 阶段利用率与 SLO |
| KV 路径 | Local、P2P、RDMA、Host | P95 传输与容量 |
| KV 副本 | 1、2 | 可用性与额外带宽 |
| 路由 | 最短队列、拓扑、Deadline | Goodput 与 P99 |
| 回退 | 重做 Prefill、Monolithic、拒绝 | Deadline 与成本 |

### 采用门槛

只有同时满足以下条件才建议上线：

- 相同模型质量和输出语义；
- Goodput 或成本/Token 有稳定收益；
- TTFT/TPOT P95/P99 满足目标；
- KV Transfer 不成为新瓶颈；
- 单 Worker/单链路故障可恢复；
- 运维复杂度、容量余量和回滚路径可接受。

## 常见错误与排查

### 1\. 只比较 Prefill 和 Decode Kernel

系统收益必须包含两次排队、KV 传输、路由、网络和错误重试。

### 2\. KV 字节估算使用 Query Head

应使用 KV Head。GQA/MQA 模型的 Query Head 更大，会严重高估传输。

### 3\. Decode Worker 先看到未完成 KV

采用临时 Buffer + 校验 + 原子 Commit，只有 COMMITTED 状态可读。

### 4\. P/D 扩缩使用同一个 GPU 利用率指标

Prefill 和 Decode 形态不同，应使用阶段队列、Token、TTFT/TPOT 与 KV 容量分别扩缩。

### 5\. 重试导致重复输出

流式响应一旦部分发送，重试语义复杂。使用 Request/Attempt、输出序号、幂等 Commit 和明确的断流策略。

### 6\. 链路标称带宽替代有效带宽

测量实际 KV Size、并发、协议与拓扑下的 P50/P95/P99，不使用单次大块 Copy 峰值代替生产流量。

### 7\. 模拟结果被写成 RDMA 实测

CPU/PCIe 模拟只能验证逻辑和低速边界。NIXL、GPUDirect RDMA、NVSwitch 必须在真实环境验证。

## 优化前后对照

| 维度 | Monolithic | Disaggregated | 风险 |
| --- | --- | --- | --- |
| 资源 | P/D 共享 | 独立池 | 碎片化容量 |
| 调度 | 单队列 | 两阶段队列 | 额外排队 |
| KV | 本地 | 跨 Worker | 传输与一致性 |
| 扩缩 | 统一 | P/D 独立 | KV 迁移 |
| 故障 | 单 Worker | 跨阶段 | 重试放大 |
| 观测 | 单 Trace | 跨服务 Trace | 关联复杂 |

## 面试题与答案

### 1\. P/D 解耦的主要收益是什么？

阶段隔离、独立 Batch/调度、资源独立扩缩和异构硬件匹配。

### 2\. 最大新增成本是什么？

KV 跨 Worker 传输，以及由此产生的排队、一致性、元数据、故障恢复和运维复杂度。

### 3\. 如何估算 KV 传输时间？

用 KV 字节除以实测有效带宽，再加固定协议/注册/调度延迟；不能直接使用标称链路峰值。

### 4\. 为什么 Decode 可能需要更多副本？

每个请求 Decode 持续多个 Token Step，服务占用时间约为输出 Token 数乘 TPOT，长输出会显著增加容量需求。

### 5\. KV Commit 为什么需要幂等？

Prefill 或网络重试可能重复发送；幂等和 Generation 可防止旧 Attempt 覆盖新 KV 或重复消费。

### 6\. 什么时候不应采用解耦？

模型/负载小、KV 传输昂贵、阶段隔离收益有限、SLO 未改善或系统复杂度与可靠性成本超过收益时。

## 课后练习

1. 扫描输入 512～16K Token，画 KV 传输与 TTFT 曲线。
2. 扫描输出长度，计算最小 Decode 副本数。
1. 在规划器中加入 30% 长尾服务时间，比较均值与 P99。
2. 设计 KV 双副本策略，计算额外带宽与故障恢复收益。
1. 实现 Attempt/Generation 条件 Commit 的状态机测试。
2. 模拟一个 Decode Worker 故障，比较重做 Prefill与 KV 副本恢复。
1. 设计 Topology-aware Router，并解释何时最短队列不是最佳。
2. 用相同负载完成 Monolithic 与 Disaggregated Goodput/成本对照。

## 项目验收 Checklist

- 已计算 KV 字节、单请求传输和总链路占用；
- 已计算 P/D 阶段利用率与稳定性；
- 已设计 Request/Attempt/KV Handle/Generation；
- KV 使用临时写入、校验和原子 Commit；
- 已定义取消、超时、重试、Drain 和回退；
- P/D/Transfer 使用独立扩缩指标；
- Router 同时考虑队列、Deadline、KV 位置与拓扑；
- 已设计跨阶段 Trace 和核心指标；
- 已与 Monolithic 做同负载端到端对照；
- CPU/PCIe 模拟没有被写成 NVLink/RDMA 实测；
- 只有真实硬件验证 NIXL、GPUDirect、NVSwitch；
- 设计包含发布、回滚和故障演练。