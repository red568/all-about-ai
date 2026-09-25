---
title: "第27课：Disaggregated Inference 架构"
source: "https://everythingai.top/learn/ai-systems-performance-engineering-text/lesson-27"
author:
published:
created: 2026-09-26
description: "面向 AI 工程学习者的专业课程学习空间。"
tags:
  - "clippings"
---
**导航**

本课解析 Prefill 与 Decode 分离架构，理解资源解耦、网络传输和调度带来的收益与代价。

## 课程定位

一条 LLM 请求看似只是“输入一段文本，再逐字输出”，但 GPU 实际经历的是两种性格完全不同的工作：

- Prefill：一次处理大量输入 token，矩阵较大，通常更偏计算密集，主要决定 TTFT；
- Decode：每轮只处理新 token，却反复读取整套模型权重，通常更偏显存带宽和串行延迟，主要决定 TPOT/ITL。

把两种负载放在同一批 GPU 上，部署简单，但长 Prompt 的 Prefill 可能打断正在流式输出的 Decode；同一套张量并行、批处理和扩缩容策略也必须同时迁就两个阶段。

Disaggregated Inference（解耦式推理）把 Prefill 与 Decode 放到独立 Worker 池：P 节点计算 Prompt 并生成 KV Cache，随后把 KV 状态传给 D 节点继续生成。它不是“多买一倍 GPU”这么简单，而是把计算、显存、网络、KV 生命周期和调度器重新组合成一套流水线。

本课要建立一个判断标准：

## 学习目标

完成本课后，你应能：

- 解释 Prefill 与 Decode 为什么适合不同资源和并行策略；
- 画出 Router、P Pool、KV Data Plane、D Pool 和控制面的完整架构；
- 计算 KV Cache 大小、传输时间和链路带宽下限；
- 用 TTFT、TPOT、Goodput、队列时间和 KV 传输时间定位瓶颈；
- 设计 $xP\times yD$ 的容量配比、路由、背压与故障恢复；
- 在纯 CPU 环境运行队列模拟，在常见 NVIDIA GPU 上测量 KV 搬运；
- 明确 PCIe、NVLink、RDMA/NIXL 的验证边界，不把模拟结果冒充集群实测。

## 前置知识

- LLM Prefill、Decode、KV Cache 与 PagedAttention；
- TTFT、TPOT/ITL、吞吐、P50/P99 与 Goodput；
- Tensor Parallel、Data Parallel、Continuous Batching；
- PCIe、NVLink、RDMA、GPUDirect RDMA 的基本概念。

## 核心直觉：让两种流水线分别做到擅长的事

### 共置式推理

```
Request -> [同一组 GPU]
           Prefill + Decode + Prefill + Decode ...
```

优点是没有跨实例 KV 传输，模型只部署一组，故障和调度逻辑简单。缺点是阶段互相干扰：一条长 Prompt 进入批次后，可能拉长其他请求下一 token 的等待时间；为了保 TPOT，只能限制 Prefill 或切成小块，又可能损失 TTFT 和 Prefill 吞吐。

### Prefill/Decode 解耦

```
控制面：注册、路由、负载、租约、故障恢复
                                           |
Client -> Gateway/Router -> Prefill Pool --+--> KV Data Plane --> Decode Pool -> Stream
                           计算 Prompt             传状态          逐 token 生成
```

解耦后可以分别做：

- P Pool：偏大批、较强计算、较高张量并行，优化 TTFT 与输入 tokens/s；
- D Pool：偏连续批处理、较大 KV 容量、较高显存带宽，优化 TPOT 与输出 tokens/s；
- Router：同时看 P 队列、D 可用 KV block、前缀命中、拓扑和 SLO；
- Planner：独立调整 P/D 数量，甚至选用不同硬件；
- Data Plane：异步搬运 KV，不让 HTTP 控制路径承载大张量。

解耦不是要求所有请求都走远程 Prefill。短 Prompt、低负载或链路拥塞时，直接在 D 节点本地 Prefill 可能更快，这称为 conditional disaggregation 或 mixed pool。

## 系统与架构原理

### 请求的完整生命周期

1. Gateway 解析请求、采样参数、模型/Adapter、上下文长度和 SLO；
2. Router 选择一个 P Worker，并预选或预约 D Worker；
1. P Worker 完成 Prefill，产生首 token logits 和分层 KV block；
2. P Worker/控制面发布 KV 元数据：请求 ID、block ID、shape、dtype、位置和目标地址；
1. 数据面把 KV 从 P 侧 HBM 发送到 D 侧 HBM，或写入共享 KV Store；
2. D Worker 确认 KV 完整可见，接管请求并开始连续 Decode；
1. 流式 token 经 Gateway 返回；
2. 请求结束、取消或超时后，两侧释放 KV、租约和元数据。

### 控制面与数据面必须分离

控制面传递的是“小而关键”的元数据：请求归属、Worker 负载、KV 地址、传输状态和故障信息。数据面传递的是数百 MB 甚至数 GB 的 KV 张量。

如果让 Python HTTP 服务转发 KV，通常会出现 HBM→CPU、序列化、内核网络栈、CPU→HBM 的多次复制。生产系统更常用 NCCL P2P、UCX、RDMA、GPUDirect RDMA、NIXL 或 Mooncake Transfer Engine 等路径，并通过异步事件通知完成。

NIXL 的定位是面向推理的点到点数据移动抽象，可覆盖 GPU/CPU 内存与存储后端；NCCL 更擅长 GPU 集体通信。两者不是简单替代关系。

### KV Cache 不只是一个 Tensor

D Worker 要正确继续 Decode，至少要获得并校验：

- 每层 K/V 数据与有效 token 数；
- block/page 布局、dtype、head 数与 head dimension；
- RoPE 位置、position ids、滑动窗口状态；
- tokenizer、模型权重版本、量化格式和 Adapter/LoRA 标识；
- 请求采样状态、随机数状态、停止条件；
- Tensor Parallel rank 与 KV shard 的映射。

若 P 与 D 使用不同 TP 度，KV 分片通常需要重排或重分片。异构硬件可带来成本优势，但也会增加 dtype、kernel、拓扑和容量规划复杂度。

### Push、Pull 与共享 KV Store

| 模式 | 工作方式 | 优点 | 风险 |
| --- | --- | --- | --- |
| Push | P 主动把 KV 推到已选 D | 延迟直接、易做流水 | 必须提前选 D，目标背压复杂 |
| Pull | D 获得元数据后拉取 KV | D 可按容量决定时机 | Prefill 完成后可能多等一次协调 |
| Shared Store | P 写入分布式 KV 池，D 按 key 读取 | 易复用、迁移和多级缓存 | 元数据、一致性、存储和网络更复杂 |

长上下文、多轮对话和前缀复用场景更适合 KV-centric 架构；单轮短请求未必能摊薄额外编排成本。

## 关键公式与性能模型

### KV Cache 容量

对标准 Transformer，单个请求每 token 的 KV 字节数近似为：

$$
B_{token}=2\cdot L\cdot H_{kv}\cdot D_h\cdot b
$$

其中 2 表示 K 和 V，L 是层数， $H_{kv}$ 是 KV head 数， $D_h$ 是 head dimension，b 是每元素字节数。长度为 S 的 Prompt：

$$
B_{KV}=S\cdot B_{token}
$$

GQA/MQA 通过减少 $H_{kv}$ 显著降低 KV 大小。实际系统还存在 block 对齐、元数据、碎片、临时 buffer 和复制冗余。

### 传输时间与带宽下限

$$
T_{xfer}\approx T_{setup}+\frac{B_{KV}}{BW_{effective}}+T_{sync}
$$

若要求 KV 传输不超过 TTFT 预算中的 $T_{budget}$ ，链路有效带宽至少应满足：

$$
BW_{required}\ge\frac{B_{KV}}{T_{budget}-T_{setup}-T_{sync}}
$$

标称 PCIe/NVLink/RDMA 带宽不能直接代入；必须测有效 payload 带宽。NUMA 跨 Socket、PCIe Switch 竞争、NIC-GPU 拓扑、消息粒度和注册成本都会降低结果。

### 端到端延迟

解耦路径近似：

$$
TTFT_{disagg}=Q_P+T_P+T_{xfer}+Q_D+T_{first_decode}
$$

$$
E2E=TTFT_{disagg}+N_{out}\cdot TPOT_D
$$

共置路径没有 $T_{xfer}$ ，但 Q 与 TPOT 会受到 Prefill/Decode 混部干扰。解耦的目标不是让每一项都变小，而是让更多请求同时满足 TTFT 与 TPOT SLO。

### Goodput

$$
Goodput=\frac{\text{同时满足 TTFT、TPOT 与正确性 SLO 的请求数}}{\text{时间}}
$$

只看总 tokens/s 会掩盖尾延迟。P Pool 高吞吐但 D Pool 堵塞时，系统仍会产生大量迟到请求；反之亦然。

### P/D 资源配比

设到达率为 $\lambda$ ，平均输入、输出长度分别为 $E[S_{in}]$ 、 $E[S_{out}]$ ，单个 P/D Worker 的有效 token 能力分别为 C\\\_P、C\\\_D：

$$
N_P\gtrsim\frac{\lambda E[S_{in}]}{C_P\rho_P},\qquad N_D\gtrsim\frac{\lambda E[S_{out}]}{C_D\rho_D}
$$

$\rho$ 是为突发和尾延迟保留余量后的目标利用率。P/D 比例会随请求长度分布变化，不能只按平均值一次性固定。

## 瓶颈分析方法

### 先回答：为什么要解耦？

满足下列至少一项，才值得进入原型阶段：

- 共置服务中长 Prefill 明显抬高 Decode P99 ITL；
- TTFT 与 TPOT 需要不同 TP/批处理配置；
- 输入/输出长度比例波动大，需要独立扩缩容；
- 想用异构硬件分别承载计算密集与带宽密集阶段；
- 长上下文 KV 需要跨实例复用、迁移或分层存储。

若单 GPU/单机小流量已经满足 SLO，解耦通常只会增加故障面。

### 分段测量

至少打点：

```
gateway_queue
prefill_queue -> prefill_compute
kv_metadata -> kv_register -> kv_transfer -> kv_ready
decode_queue -> first_decode -> decode_iterations
stream_flush -> request_finish
```

应记录 P50/P95/P99，而不只记录均值。KV Transfer 还要记录 size、setup、有效带宽、重试、目标拓扑和是否与计算重叠。

### 识别四种典型瓶颈

1. P 饱和：Prefill queue 上升，TTFT 变差，D 利用率下降；
2. D 饱和：Prefill 很快但 KV 等待消费，TPOT/P99 和显存压力上升；
1. 网络饱和：P/D 计算都有空闲，KV transfer latency 随并发陡升；
2. 控制面饱和：小请求也慢，元数据、路由、Python/RPC 或锁成为热点。

## 完整可运行实验

### Level 0：CPU 离散事件容量模拟

这个模拟器比较“相同总 Worker 数下的共置串行模型”和“独立 P/D 池”。它用于建立容量直觉，不模拟真实 GPU batching、attention kernel 或网络协议。

保存为 `disaggregated_queue_sim.py` ：

```
#!/usr/bin/env python3
import argparse
import heapq
import random
import statistics

def percentile(values, q):
    values = sorted(values)
    i = min(len(values) - 1, int((len(values) - 1) * q))
    return values[i]

def make_requests(n, qps, seed):
    rng = random.Random(seed)
    now = 0.0
    reqs = []
    for i in range(n):
        now += rng.expovariate(qps) * 1000.0
        # 混合短/长 Prompt 与短/长输出，单位为 token。
        prompt = rng.choice([128, 256, 512, 2048, 4096])
        output = rng.choice([32, 64, 128, 256])
        reqs.append((i, now, prompt, output))
    return reqs

def run_monolithic(reqs, workers, p_ms_per_token, d_ms_per_token):
    heap = [0.0] * workers
    heapq.heapify(heap)
    ttft, tpot, e2e = [], [], []
    for _, arrival, pin, pout in reqs:
        free = heapq.heappop(heap)
        start = max(arrival, free)
        p_ms = pin * p_ms_per_token
        first = start + p_ms + d_ms_per_token
        finish = first + max(0, pout - 1) * d_ms_per_token
        heapq.heappush(heap, finish)
        ttft.append(first - arrival)
        tpot.append(d_ms_per_token)
        e2e.append(finish - arrival)
    return ttft, tpot, e2e

def run_disagg(reqs, p_workers, d_workers, p_ms_per_token,
               d_ms_per_token, xfer_base_ms, gbps, kv_bytes_per_token):
    p_heap = [0.0] * p_workers
    d_heap = [0.0] * d_workers
    heapq.heapify(p_heap)
    heapq.heapify(d_heap)
    ttft, tpot, e2e, xfers = [], [], [], []

    for _, arrival, pin, pout in reqs:
        p_free = heapq.heappop(p_heap)
        p_start = max(arrival, p_free)
        p_done = p_start + pin * p_ms_per_token
        heapq.heappush(p_heap, p_done)

        kv_bytes = pin * kv_bytes_per_token
        xfer_ms = xfer_base_ms + kv_bytes * 8 / (gbps * 1e9) * 1000
        xfers.append(xfer_ms)
        kv_ready = p_done + xfer_ms

        d_free = heapq.heappop(d_heap)
        d_start = max(kv_ready, d_free)
        first = d_start + d_ms_per_token
        finish = first + max(0, pout - 1) * d_ms_per_token
        heapq.heappush(d_heap, finish)

        ttft.append(first - arrival)
        tpot.append(d_ms_per_token)
        e2e.append(finish - arrival)
    return ttft, tpot, e2e, xfers

def report(name, ttft, tpot, e2e, ttft_slo, tpot_slo):
    good = sum(a <= ttft_slo and b <= tpot_slo for a, b in zip(ttft, tpot))
    print(f"\n[{name}]")
    print(f"TTFT p50/p99: {percentile(ttft,.50):.1f}/{percentile(ttft,.99):.1f} ms")
    print(f"TPOT p50/p99: {percentile(tpot,.50):.1f}/{percentile(tpot,.99):.1f} ms")
    print(f"E2E  p50/p99: {percentile(e2e,.50):.1f}/{percentile(e2e,.99):.1f} ms")
    print(f"Goodput requests: {good}/{len(ttft)} ({good/len(ttft):.1%})")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--requests", type=int, default=500)
    p.add_argument("--qps", type=float, default=3.0)
    p.add_argument("--mono-workers", type=int, default=4)
    p.add_argument("--p-workers", type=int, default=1)
    p.add_argument("--d-workers", type=int, default=3)
    p.add_argument("--prefill-ms-per-token", type=float, default=0.04)
    p.add_argument("--decode-ms-per-token", type=float, default=8.0)
    p.add_argument("--p-specialization-speedup", type=float, default=2.0)
    p.add_argument("--d-specialization-speedup", type=float, default=2.0)
    p.add_argument("--xfer-base-ms", type=float, default=0.3)
    p.add_argument("--link-gbps", type=float, default=100.0)
    p.add_argument("--kv-bytes-per-token", type=int, default=131072)
    p.add_argument("--ttft-slo-ms", type=float, default=500.0)
    p.add_argument("--tpot-slo-ms", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    reqs = make_requests(a.requests, a.qps, a.seed)

    mono = run_monolithic(reqs, a.mono_workers,
                          a.prefill_ms_per_token, a.decode_ms_per_token)
    dis = run_disagg(reqs, a.p_workers, a.d_workers,
                     a.prefill_ms_per_token / a.p_specialization_speedup,
                     a.decode_ms_per_token / a.d_specialization_speedup,
                     a.xfer_base_ms, a.link_gbps, a.kv_bytes_per_token)

    report("monolithic", *mono, a.ttft_slo_ms, a.tpot_slo_ms)
    report("disaggregated", *dis[:3], a.ttft_slo_ms, a.tpot_slo_ms)
    print(f"KV transfer mean/p99: {statistics.mean(dis[3]):.2f}/"
          f"{percentile(dis[3], .99):.2f} ms")
    print("注意：这是教学排队模型，不是硬件性能结论。")

if __name__ == "__main__":
    main()
```

运行：

```
python disaggregated_queue_sim.py
python disaggregated_queue_sim.py --qps 6 --p-workers 1 --d-workers 3
python disaggregated_queue_sim.py --link-gbps 10 --kv-bytes-per-token 524288
```

预期现象：默认输入假设阶段专用配置让 P、D 各获得 2 倍服务效率，因此解耦路径可能提高 Goodput；这只是用于观察架构收益的显式假设，不是硬件事实。将两个 `specialization-speedup` 改为 1，能看到只有传输开销、没有阶段优化时的结果。若 D Worker 太少，KV 会在 D 队列前积压；若有效链路从 100 Gbps 降到 10 Gbps 且 KV 很大，传输可能吞掉收益。

### Level 1：通用 PyTorch KV 搬运基准

保存为 `kv_transfer_benchmark.py` 。脚本会自动检测 CPU、单 GPU、双 GPU和 P2P 能力：

```
#!/usr/bin/env python3
import argparse
import time
import torch

def sync_all():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mib", type=int, default=256)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    a = p.parse_args()
    n = a.mib * 1024 * 1024 // 2  # FP16 元素数

    print("torch=", torch.__version__, "cuda=", torch.version.cuda)
    print("gpu_count=", torch.cuda.device_count())
    if torch.cuda.device_count() >= 2:
        src = torch.empty(n, dtype=torch.float16, device="cuda:0").normal_()
        dst = torch.empty_like(src, device="cuda:1")
        print("src=", torch.cuda.get_device_name(0))
        print("dst=", torch.cuda.get_device_name(1))
        print("peer_access=", torch.cuda.can_device_access_peer(0, 1))

        def copy_once():
            with torch.cuda.device(1):
                dst.copy_(src, non_blocking=True)

        route = "GPU0 -> GPU1"
    elif torch.cuda.device_count() == 1:
        src = torch.empty(n, dtype=torch.float16, device="cuda:0").normal_()
        host = torch.empty(n, dtype=torch.float16, pin_memory=True)
        dst = torch.empty_like(src)
        print("gpu=", torch.cuda.get_device_name(0))

        def copy_once():
            host.copy_(src, non_blocking=True)
            dst.copy_(host, non_blocking=True)

        route = "GPU -> pinned CPU -> GPU round trip"
    else:
        src = torch.empty(n, dtype=torch.float16).normal_()
        dst = torch.empty_like(src)

        def copy_once():
            dst.copy_(src)

        route = "CPU memory copy"

    for _ in range(a.warmup):
        copy_once()
    sync_all()
    t0 = time.perf_counter()
    for _ in range(a.iters):
        copy_once()
    sync_all()
    elapsed = time.perf_counter() - t0
    bytes_per_iter = a.mib * 1024 * 1024
    if torch.cuda.device_count() == 1:
        bytes_per_iter *= 2
    gib_s = bytes_per_iter * a.iters / elapsed / 2**30
    print(f"route={route}")
    print(f"payload={a.mib} MiB iterations={a.iters}")
    print(f"mean_latency={elapsed/a.iters*1000:.3f} ms")
    print(f"effective_bandwidth={gib_s:.3f} GiB/s")
    print("该结果只代表当前路径与并发条件，不等于链路标称带宽。")

if __name__ == "__main__":
    main()
```

运行：

```
python kv_transfer_benchmark.py --mib 64
python kv_transfer_benchmark.py --mib 256
nvidia-smi topo -m
```

双 RTX 3080 20GB 通常通过 PCIe 通信，不应假设存在 NVLink。若 `peer_access=False` ，路径可能经主机内存或受到拓扑限制；即使为 `True` ，也要以有效带宽和延迟为准。

### Level 2：双 GPU 功能实验与生产级路径

### vLLM 两实例实验

vLLM 的解耦接口仍在快速演进。使用安装版本对应的官方示例，不要混用旧版 `P2pNcclConnector` 、新版 `NixlConnector` 或代理参数：

```
python -m vllm.entrypoints.cli.main --version 2>/dev/null || vllm --version
git clone --depth 1 https://github.com/vllm-project/vllm.git vllm-src
find vllm-src/examples -iname '*disagg*' -o -iname '*prefill*'
```

当前常见形态是两个相同模型实例：

```
# P Worker：KV producer
CUDA_VISIBLE_DEVICES=0 vllm serve /path/to/model \
  --port 8100 --max-model-len 4096 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'

# D Worker：KV consumer
CUDA_VISIBLE_DEVICES=1 vllm serve /path/to/model \
  --port 8200 --max-model-len 4096 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}'
```

随后按该版本仓库中的 `disagg_proxy_demo.py` 或启动脚本连接 P/D 实例。这里故意不复制一份固定代理代码，因为 KV 参数、Push/Pull 模式和脚本路径会随版本改变；版本匹配比“命令看起来完整”更重要。

双 3080 功能实验建议选择能在每张 20GB 卡上各自完整加载的小模型。P 和 D 都需要模型权重，解耦不会自动把一个超大模型拆成两半。

### NIXL/RDMA 专项

NIXL 面向 Linux，真实跨节点高性能路径需要正确的 UCX、NIC、RDMA、GDRCopy/GPUDirect 与 NUMA 拓扑。消费级双卡可以验证 API 与 PCIe 功能，但不能等价模拟多节点 RDMA。

```
python -m venv .venv-nixl
source .venv-nixl/bin/activate
python -m pip install --upgrade pip
# 安装命令与 CUDA 12/13 包名以 NIXL 当前发布说明为准
python -m pip install nixl
python -c "import nixl; print('NIXL import OK')"
```

生产测试还应运行项目自带的 `nixlbench` 或 KV benchmark，并记录消息大小、并发、GPU/NIC 亲和性、RDMA/TCP 后端和是否启用 GPUDirect。

## 硬件适配与边界

| 架构 | 常见 GPU | 合适实验 | 边界 |
| --- | --- | --- | --- |
| Ampere | RTX 3080/3090、A100 | PCIe P2P、A100 NVLink/RDMA 集群 | 3080 双卡通常无 NVLink |
| Ada Lovelace | RTX 4090、L40/L40S | 单机 PCIe、数据中心 L40S 多卡 | RTX 4090 不是 Blackwell |
| Hopper | H100/H200 | NVLink 4、NVSwitch、InfiniBand/RDMA | 消费卡不能等价模拟 |
| Blackwell | RTX 5090、B100/B200/GB200 | 新一代数据中心 NVLink/NVSwitch 与大规模解耦 | RTX 5090 不等同 GB200 NVL 系统 |

无专属硬件时，可以学习队列、容量、KV 大小和控制面设计；无法验证的部分必须明确写成“模型估算”或“可选专项”。

## 调度、弹性与故障设计

### 路由目标

Router 不应只做 Round Robin。至少考虑：

- P/D 队列长度与预测完成时间；
- D Worker 可用 KV blocks；
- Prompt 前缀命中与 KV 所在位置；
- P→D 拓扑和当前链路拥塞；
- 模型、量化、LoRA/Adapter 与采样兼容性；
- 请求优先级、TTFT/TPOT SLO；
- 长输出导致的 D 侧驻留时间。

### 背压

P 比 D 快时，Prefill 完成并不代表系统有能力接管 Decode。需要给 D 预留容量、限制在途 KV、延迟启动 Prefill或短路到共置节点。否则 P Pool 会制造大量占用显存但无人消费的 KV。

### 故障恢复

- P 在传输前失败：请求可重跑 Prefill；
- 传输中失败：以 request/transfer ID 做幂等重试，不能让 D 使用半份 KV；
- D 在 Decode 中失败：若无远程 KV 副本或 checkpoint，通常要重新 Prefill；
- Gateway 断连/取消：及时回收 P、D 和 KV Store 中的资源；
- 模型滚动升级：版本不一致的 P/D 不得配对。

## 结果分析模板

```
模型、精度、TP/PP：
P/D GPU 与数量：
输入/输出长度分布：
到达率与并发：
链路与拓扑：
KV bytes/request：
Prefill queue/compute P50/P99：
KV setup/transfer/sync P50/P99：
Decode queue、TTFT、TPOT P50/P99：
有效传输带宽：
Goodput（TTFT+TPOT SLO）：
共置基线：
故障/重试/回收结果：
```

## 常见错误与排查

### 只看 GPU 利用率，发现两边都不高

可能是 KV 传输或控制面串行。查看时间线是否出现 P 完成后等待注册、D 收到元数据后等待数据、CPU 同步或阻塞 RPC。

### TTFT 变差但 TPOT 变好

KV 传输进入 TTFT 路径，而 Decode 不再被 Prefill 打断。需要通过层级流水、提前选择 D、提高链路有效带宽或短 Prompt 本地 Prefill来权衡，而不是只盯一个指标。

### P Pool 空闲、D Pool 爆满

输入/输出长度比例或到达分布与规划不符。增加 D、降低 D batch 驻留、设置在途 KV 上限，或启用混合 Worker 承接峰值。

### KV 传输带宽远低于标称值

检查 GPU-NIC/PCIe NUMA 拓扑、P2P、IOMMU、RDMA backend、内存注册、消息粒度、并发流、PCIe Switch 竞争和是否发生 CPU bounce buffer。

### 输出错误或崩溃

核对模型/LoRA 版本、block size、dtype、KV head、position/RoPE、TP shard 映射、完整性事件和 request ID。不要把“网络传完”当成“D 侧 kernel 已可安全读取”。

### 双 3080 比单卡更慢

每张卡都加载模型，PCIe 搬运 KV，还增加代理和调度开销；小流量没有阶段干扰可消除时，这是正常结果。该实验的价值是验证架构，不保证性价比。

## 优化前后对照

| 维度 | 共置式推理 | 解耦式推理 |
| --- | --- | --- |
| Prefill/Decode | 同池混合 | 独立 Worker 池 |
| TTFT/TPOT 调优 | 相互牵制 | 可分别规划 |
| KV 传输 | 无跨实例传输 | 新增关键数据面 |
| 扩缩容 | 整体扩缩 | P/D 独立扩缩 |
| 硬件选择 | 通常同构 | 可阶段感知异构 |
| 故障面 | 较小 | Router、传输、状态一致性增加 |
| 适用场景 | 单机、小规模、负载稳定 | 大规模、长上下文、SLO 严格、负载不对称 |

## 面试题与答案

### 1\. 为什么 Prefill 计算密集、Decode 更偏带宽密集？

Prefill 能并行处理多个 Prompt token，矩阵维度大，容易提高 Tensor Core 利用率；Decode 每次只新增少量 token，却要读取模型权重，算术强度较低且有跨 token 串行依赖。

### 2\. 解耦推理最大的新增成本是什么？

把 Prefill 产生的 KV Cache 连同完整元数据可靠地交给 Decode。成本包括传输、同步、显存预留、控制面、失败重试和可能的 KV 重分片。

### 3\. 为什么不能只看 tokens/s？

高吞吐可能来自大批处理，却让 TTFT 或 TPOT 尾延迟超标。解耦系统通常以同时满足 TTFT、TPOT 和正确性 SLO 的 Goodput 作为核心目标。

### 4\. 如何估算 KV 传输是否可接受？

先按层数、KV heads、head dimension、dtype 和 Prompt 长度算 B\\\_{KV}，再用实测有效带宽与 setup/sync 估算 $T_{xfer}$ ，把它放入 TTFT预算，并在并发下验证 P99。

### 5\. P/D 比例为什么会变化？

它由到达率、输入/输出长度分布、P/D 吞吐、目标利用率和 SLO 共同决定。对话短输入长输出需要更多 D，文档总结长输入短输出需要更多 P。

### 6\. NIXL 与 NCCL 的区别是什么？

NCCL 主要提供 GPU 集体通信和点到点能力；NIXL 面向推理数据移动，抽象 GPU/CPU 内存及存储后端和不同传输插件。具体系统可以组合使用，不能简单说谁替代谁。

### 7\. 解耦是否一定需要 RDMA？

功能上不一定，单机 PCIe、共享内存、TCP 也能实现；高吞吐跨节点长上下文场景通常需要更高效的数据路径。是否需要由 KV 大小、SLO 和实测带宽决定。

### 8\. 什么是 conditional disaggregation？

Router 按 Prompt 长度、队列、缓存命中或网络状态决定请求走远程 Prefill还是在 Decode/混合节点本地 Prefill，用于避免短请求的传输固定成本或缓解池间失衡。

### 9\. 为什么 P 与 D 必须加载同一模型？

D 要基于 P 计算出的 KV 继续执行，模型结构、权重语义、位置编码和 Adapter 不一致会使状态失效。可以使用不同硬件和不同并行布局，但必须有明确兼容与重分片方案。

### 10\. 解耦失败时最安全的回退是什么？

停止使用不完整 KV，在健康的共置或 P/D 路径重新执行 Prefill；通过幂等 request ID 和租约清理旧状态。静默接着错误 KV Decode 是不可接受的。

## 课后练习

1. 用公式计算一个 32 层、8 个 KV head、head dimension 128、BF16 模型在 4K/32K Prompt 下的 KV 大小。
2. 在 Level 0 中扫描 1P3D、2P2D、3P1D，分别测试“长输入短输出”和“短输入长输出”。
1. 把模拟器加入链路共享：多个传输并发时按带宽均分，观察 P99。
2. 在双 GPU 上测 16、64、256、1024 MiB 传输，解释小消息为何受 setup 影响。
1. 使用 Nsight Systems 标记 Prefill、KV copy、Decode，验证能否真正重叠。
2. 设计带前缀命中的 Router 评分函数，同时考虑缓存命中、队列和拓扑。
1. 设计 P 完成但 D 失败时的状态机，列出所有幂等键和资源回收点。
2. 为双 RTX 3080 写一份“何时退回共置式”的判定规则。

## Checklist

- 已用共置基线证明存在阶段干扰或资源耦合；
- 已记录输入/输出长度分布，而非只看平均值；
- 已计算每请求 KV 大小和峰值在途 KV；
- 已测有效传输带宽、setup、sync 和 P99；
- 已分别测 P queue/compute、KV transfer、D queue/decode；
- 已用 TTFT、TPOT 和 Goodput 共同评估；
- 已为 P/D 独立规划 TP、batch、并发和扩缩容；
- 已设计背压，避免 P 制造无人消费的 KV；
- 已校验模型、Adapter、dtype、block 和 TP shard 兼容；
- 已覆盖取消、超时、P/D/网络失败和幂等清理；
- 已测试短 Prompt 本地 Prefill或混合池回退；
- 双 3080 未被写成 NVLink 集群；
- Hopper/Blackwell/RDMA 专属能力均明确标为可选；
- 性能数字仅作为对应硬件、版本和 workload 的测量结果。