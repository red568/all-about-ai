---
title: "第16课：NCCL 深度优化"
source: "https://everythingai.top/learn/ai-systems-performance-engineering-text/lesson-16"
author:
published:
created: 2026-09-26
description: "面向 AI 工程学习者的专业课程学习空间。"
tags:
  - "clippings"
---
**导航**

本课深入 NCCL 集合通信的执行路径与拓扑选择，掌握通信瓶颈定位和参数调优方法。

## 课程定位

多 GPU 训练中，计算结果只有被正确、及时地送到其他 GPU，新增算力才真正有用。NCCL 是 NVIDIA GPU 集群中最常见的 Collective 通信库，但“使用了 NCCL”不等于“通信已经最优”：同一个 All-Reduce，可能走 NVLink、PCIe P2P、共享内存、InfiniBand/RoCE 或 TCP Socket；可能选择 Ring、Tree、NVLS 或其他算法；还可能因为错误网卡、容器共享内存、PCIe ACS、慢 Rank 和错误的 Collective 顺序而降速或挂死。

本课从通信语义开始，建立 NCCL 的性能模型和诊断顺序。重点不是背环境变量，而是学会先建立默认基线，再用拓扑、消息大小、日志和 Profiler 形成假设，最后一次只改变一个变量。第 15 课解决“模型怎样分”，本课解决“分开后怎样高效交换数据”。

## 学习目标

完成本课后，你应当能够：

- 区分 All-Reduce、Reduce-Scatter、All-Gather、Broadcast、All-to-All 和 Send/Recv；
- 解释 NCCL Communicator、Rank、Channel、CUDA Stream、Algorithm 和 Protocol 的关系；
- 用延迟-带宽模型估算 Ring 与 Tree 的适用区间；
- 正确理解 `algbw` 、 `busbw` 、消息大小和链路峰值之间的差异；
- 判断通信走的是 NVLink、PCIe P2P、SHM、RDMA 还是 Socket；
- 使用 PyTorch 与 `nccl-tests` 建立可重复的通信基线；
- 用日志、拓扑、RAS 和对照实验定位初始化失败、Hang、带宽低与长尾；
- 知道哪些环境变量适合诊断，哪些不应固化为“万能优化参数”。

## 前置知识

- 熟悉第 8 课的 NVLink、NVSwitch、PCIe 与集群互联；
- 理解第 15 课的 DP、TP、PP、FSDP/ZeRO 和 EP；
- 会使用 `torchrun` 启动一 GPU 一进程程序；
- 理解带宽、延迟、P50/P99、同步、异步和 CUDA Stream；
- 了解 Linux 网络接口、容器和 NUMA 的基本概念。

## 核心直觉：NCCL 是“通信执行器”，不是一条固定链路

把 NCCL 想成一个物流调度系统：

- \*\*Collective\*\* 决定所有货物最终要到哪里；
- \*\*Algorithm\*\* 决定车辆按环、树或分层网络怎样走；
- \*\*Protocol\*\* 决定每次装卸的粒度和控制开销；
- \*\*Channel/CTA\*\* 决定并行使用多少条运输流水线；
- \*\*Transport\*\* 决定底层走 NVLink、PCIe、共享内存、RDMA 还是 Socket；
- \*\*Topology\*\* 决定哪些路径近、哪些路径会绕远或争用；
- \*\*CUDA Stream\*\* 决定通信与计算何时发生、能否重叠。

因此，看到 `backend=nccl` 只能证明框架选择了 NCCL 后端，不能证明：

- GPU 间一定使用 NVLink；
- 跨节点一定使用 GPUDirect RDMA；
- Ring 一定优于 Tree；
- 通信一定与反向计算重叠；
- 当前带宽已经接近硬件上限。

性能工程的正确顺序是：

```
语义正确 → 拓扑正确 → Transport 正确 → 基线稳定
        → 找消息区间 → 验证算法/协议假设 → 端到端验证
```

## NCCL 架构与执行路径

### Communicator 与 Rank

NCCL Communicator 描述一组参与通信的 GPU Rank。每个 Rank 在相同 Communicator 上，必须以兼容的顺序进入 Collective。某 Rank 调用了 All-Reduce，而另一个 Rank 同时调用 Broadcast，常见结果不是报错，而是所有人相互等待。

Communicator 初始化通常包含：

1. 交换 Rank 与唯一 ID；
2. 枚举 GPU、PCIe、CPU NUMA、NIC 和 NVLink/NVSwitch 拓扑；
1. 选择 Transport、Algorithm、Protocol 和 Channel；
2. 建立 P2P、共享内存、网络与内部缓冲。

初始化可能明显慢于一次 Collective，因此不要在每个训练 Step 中销毁并重建 Communicator。

### 单机 Transport

单机通信常见路径从优到退化大致包括：

1. GPU 之间通过 NVLink/NVSwitch 直接传输；
2. GPU 通过 PCIe Peer-to-Peer 访问；
1. P2P 不可用时，通过主机共享内存 SHM 中转；
2. 特殊情况下使用网络 Transport 绕行。

实际选择由拓扑、驱动、IOMMU/ACS、容器可见性和 NCCL 版本共同决定。RTX 3080/4090/5090 等消费卡是否有 P2P、能否跨 PCIe Root Complex，必须实机检测，不能只按架构名称推断。

### 多节点 Transport

跨节点时常见两条路径：

- \*\*RDMA 路径\*\*：GPU Memory 与 RDMA NIC 直接交换，条件满足时使用 GPUDirect RDMA；
- \*\*Socket 路径\*\*：经过 TCP/IP Socket，兼容性强，但通常占用更多 CPU，延迟和有效带宽受网络栈影响。

InfiniBand、RoCE、以太网、NIC 数量、Rail 设计、GPU-NIC PCIe 亲和性都会改变结果。“有 400Gb/s 网卡”并不等于每个 GPU 都能得到 400Gb/s；多张 GPU 可能共享一个 NIC 或 PCIe 上行链路。

### NCCL 与 CUDA Stream

NCCL 通信调用关联到 CUDA Stream。调用返回通常表示操作已经入队，而不是数据传输已经完成。真正完成要遵循 CUDA Stream 语义，通过事件、Stream 同步或依赖关系确认。

在 PyTorch 中， `async_op=True` 返回 Work Handle，也不表示可以立即在无依赖的 Stream 上安全使用结果。优化通信重叠时必须同时验证：

- 通信是否真的与计算在 Timeline 上重叠；
- 消费结果的计算是否正确等待通信完成；
- 并发是否争用 SM、Copy Engine、HBM 或链路，导致两者都变慢。

## Collective 语义：先选对操作，再谈调参

### All-Reduce

每个 Rank 提供一份输入，所有输入被归约，每个 Rank 都得到完整结果。

```
Rank0: A    Rank1: B    Rank2: C    Rank3: D
                    All-Reduce SUM
Result: A+B+C+D on every rank
```

典型用途是 DDP 梯度同步。

### Reduce-Scatter

先归约所有 Rank 的输入，再把结果分片，每个 Rank 只保留一片。它可以看作 All-Reduce 的前半段，常见于 FSDP/ZeRO 梯度分片和部分张量并行模式。

### All-Gather

每个 Rank 提供一片，所有 Rank 收集成完整张量。它常见于 FSDP 参数按需聚合、张量并行激活聚合和分片结果重建。

经典恒等关系为：

$$
AllReduce = ReduceScatter + AllGather
$$

这不是说框架一定真的调用两个独立 API，而是说明数据运动结构可以如此分解。

### Broadcast 与 Reduce

- Broadcast：一个 Root 把同一数据发送给所有 Rank；
- Reduce：所有 Rank 归约，但结果只保留在 Root。

它们适合参数广播、初始化和中心化汇总，但大规模同步训练通常更关注 All-Reduce 或分片 Collective。

### All-to-All

每个 Rank 都给每个其他 Rank 发送不同分片。MoE Token Dispatch/Combine 是典型场景。All-to-All 对双向带宽、网络拥塞和负载均衡非常敏感；平均消息量相同，最忙 Rank 仍可能决定 Step 尾延迟。

### Send/Recv

点对点通信适合流水线并行相邻 Stage、邻居交换和自定义稀疏模式。多次 Send/Recv 可用 NCCL Group 语义合并调度，但所有 Rank 仍要保证一致的操作顺序。

## Algorithm、Protocol 与 Channel

### Ring：大消息优先追求带宽

环形 All-Reduce 通常拆为 Reduce-Scatter 和 All-Gather。设 Rank 数为 N，每 Rank 输入大小为 S Bytes，忽略流水化细节，每个 Rank 的总收发量约为：

$$
V_{ring}=2\frac{N-1}{N}S
$$

简化时间模型：

$$
T_{ring}\approx 2(N-1)\alpha + 2\frac{N-1}{N}\frac{S}{B}
$$

Ring 的特点是每条链路上的大消息带宽利用通常较好，但步骤数随 Rank 数线性增加，小消息时启动延迟占比高。

### Tree：小消息和大规模 Rank 更关注延迟

树形归约与广播的通信轮数大致随 $\log_2N$ 增长：

$$
T_{tree}\approx 2\lceil\log_2N\rceil\alpha + C_{tree}\frac{S}{B}
$$

C\\\_{tree} 取决于具体实现、拓扑和分块方式。Tree 不等于任何消息都更快：它减少通信轮数，但链路利用、节点度数和数据分块可能让大消息表现不如 Ring。

### CollNet、NVLS 与 NVLSTree

- CollNet 系列面向支持相应网络插件与拓扑的分层 Collective；
- NVLS 使用支持 NVLink SHARP 的 NVSwitch 域进行 Collective Offload；
- NVLSTree 将 NVLS 与跨域树形路径结合，适合相应的大系统。

NVLS 不是“所有带 NVLink 的 GPU 都支持”。它依赖支持该能力的 NVSwitch 系统，官方文档将其与 Hopper 及后续、第三代 NVSwitch 系统联系起来。双 RTX 3080、RTX 4090、RTX 5090 不能通过环境变量模拟 NVLS。

### LL、LL128 与 Simple Protocol

- LL：面向低延迟的小消息路径；
- LL128：在支持的平台上平衡低延迟与带宽；
- Simple：通常适合大消息和高带宽传输。

NCCL 会根据版本、平台、拓扑和消息大小自动选择。官方文档明确不鼓励把 `NCCL_PROTO` 当成通用优化旋钮；尤其不能在不支持的平台强制 LL128，否则存在数据损坏风险。正确用法是诊断某协议是否异常或做受控 A/B，然后恢复自动选择。

### Channel / CTA

NCCL 把 Collective 分成多个 Channel 并行推进，GPU 上由通信 Kernel 的 CTA 执行。Channel 太少可能吃不满链路；太多会占用更多 SM、寄存器、内部 Buffer，并与模型计算争用。

现代 NCCL 提供 `NCCL_MIN_CTAS` 、 `NCCL_MAX_CTAS` 等控制范围；旧资料中的 Channel/Ring 变量可能仍为兼容项，但不应跨版本复制。默认自动调优应是第一基线，只有 Profiler 证明通信 Kernel 并行度或 SM 争用存在问题时才做 A/B。

## 性能指标与模型

### 延迟、算法带宽与总线带宽

`nccl-tests` 常报告：

- `time` ：Collective 完成时间；
- `algbw` ：按算法输入数据量除以时间得到的带宽；
- `busbw` ：根据 Collective 数据运动模式对 `algbw` 做归一化，便于与物理总线能力比较。

对 All-Reduce，常用关系为：

$$
BusBW=AlgBW\times 2\frac{N-1}{N}
$$

`busbw` 仍不是“某一根线的实测物理速率”：它是算法归一化指标。多链路、双向传输、协议开销、拓扑路径和 Rank 数都会影响如何解读。比较结果时必须同时报告操作、Rank 数、消息大小、in-place/out-of-place、数据类型和拓扑。

### 小消息看延迟，大消息看平台带宽

设固定启动成本为 $\alpha$ ，消息大小为 S，有效带宽为 B：

$$
T(S)\approx \alpha+\frac{S}{B}
$$

当 $S\ll \alpha$ B，增加原始链路带宽帮助有限，应该减少 Collective 次数、融合小消息或优化调度；当 $S\gg \alpha$ B，路径、协议、Channel、NIC/PCIe 带宽和拓扑成为重点。

### 端到端通信暴露时间

通信优化的目标不是让 NCCL Kernel 本身最短，而是减少 Step 关键路径：

$$
T_{exposed}=T_{comm}-T_{overlapped}
$$

将一个 20ms All-Reduce 优化到 18ms，未必比把 20ms 通信中的 15ms 隐藏到反向计算后面更有价值。反过来，如果通信与计算争用资源，Timeline 看似重叠也可能使 Step 更慢。

## 瓶颈诊断顺序

### 第一层：正确性与 Hang

1. 每个 Rank 的 Collective 类型、张量元素数、数据类型和调用顺序是否一致；
2. 是否有某个 Rank 在数据加载、异常、OOM 或条件分支中没有进入 Collective；
1. `torchrun` 的 Rank、World Size、Master 地址与端口是否一致；
2. 防火墙和网络接口是否允许所有节点互通；
1. GPU 是否正确绑定到 Local Rank。

不要一开始就随机设置几十个 NCCL 变量。Hang 的首要嫌疑通常是语义或连通性，而非 Algorithm 不够快。

### 第二层：初始化失败

检查：

- `NCCL_DEBUG=WARN` 或 `INFO` 的第一条错误；
- `/sys` 是否在容器中正确暴露，NCCL 依赖它识别 PCI 拓扑；
- `/dev/shm` 与 pinned memory 限制；
- Driver、CUDA Runtime、NCCL 与框架组合；
- RDMA 设备、驱动、固件和权限；
- 错误网卡、不可达的 Docker/VPN/虚拟接口。

Docker 运行多进程通信时常用：

```
docker run --gpus all --ipc=host --ulimit memlock=-1 ...
```

也可使用足够大的 `--shm-size` 。这不是性能魔法，而是避免共享内存资源不足。

### 第三层：单机链路与拓扑

```
nvidia-smi topo -m
nvidia-smi topo -p2p r
nvidia-smi topo -p2p w
lspci -tv
numactl --hardware
```

重点关注：

- GPU 间是 NVLink、同一 PCIe Switch、同一 Root Complex 还是跨 NUMA；
- NIC 与 GPU 是否靠近同一 NUMA/PCIe 根；
- P2P 是否实际可读写；
- 虚拟化或 ACS 是否让 P2P 绕行 Root Complex；
- 两张 GPU 与 NIC 是否共享有限的 PCIe 上行链路。

### 第四层：消息大小扫描

不要只测 1GB 大消息。真实训练同时包含小控制消息、中等 Bucket、FSDP 参数块和大梯度。应按指数区间扫描，例如 8B 到 512MiB，画出 Latency、AlgBW 和 BusBW 曲线，找到：

- 固定延迟区；
- 带宽爬升区；
- 平台区；
- 大消息异常下降区。

### 第五层：多节点链路

先在 NCCL 外验证网络：

```
ip -br addr
ibv_devinfo
ib_write_bw <peer-host>
```

没有 RDMA 硬件时，使用 `iperf3` 验证 TCP 吞吐。低层网络都不稳定时，继续调 NCCL Algorithm 没有意义。

### 第六层：端到端重叠与慢 Rank

使用 Nsight Systems 或 PyTorch Profiler 观察：

- NCCL Kernel 是否及时启动；
- Bucket 是否过碎或过大；
- 通信 Stream 是否与计算重叠；
- 某 Rank 是否长期晚到；
- GPU 降频、CPU 数据线程、网络重传或 MoE 负载是否造成长尾。

## 完整实验

### Level 0：纯 Python Ring/Tree 成本模型

保存为 `nccl_cost_model.py` ：

```
#!/usr/bin/env python3
import argparse
import math

def ring_allreduce_us(size_bytes, ranks, alpha_us, bandwidth_gbs):
    steps = 2 * (ranks - 1)
    traffic = 2 * (ranks - 1) / ranks * size_bytes
    transfer_us = traffic / (bandwidth_gbs * 1e9) * 1e6
    return steps * alpha_us + transfer_us

def tree_allreduce_us(size_bytes, ranks, alpha_us, bandwidth_gbs,
                      tree_bandwidth_factor):
    levels = math.ceil(math.log2(ranks)) if ranks > 1 else 0
    # 教学模型：上下树各 levels 轮；有效带宽由 factor 表示。
    transfer_us = 2 * size_bytes / (
        bandwidth_gbs * tree_bandwidth_factor * 1e9
    ) * 1e6
    return 2 * levels * alpha_us + transfer_us

def parse_sizes(text):
    values = []
    for item in text.split(","):
        value = float(item.strip())
        if value <= 0:
            raise ValueError("消息大小必须为正数")
        values.append(value)
    return values

def main():
    p = argparse.ArgumentParser(description="NCCL All-Reduce 教学成本模型")
    p.add_argument("--ranks", type=int, default=8)
    p.add_argument("--bandwidth-gbs", type=float, default=25.0,
                   help="假设的单向有效链路带宽 GB/s")
    p.add_argument("--alpha-us", type=float, default=4.0,
                   help="每个通信阶段的启动延迟 us")
    p.add_argument("--tree-bandwidth-factor", type=float, default=0.65)
    p.add_argument("--sizes-mib", default="0.001,0.01,0.1,1,16,64,256")
    args = p.parse_args()

    if args.ranks < 2:
        raise ValueError("至少需要 2 个 Rank")
    if args.bandwidth_gbs <= 0 or args.alpha_us < 0:
        raise ValueError("带宽必须为正，延迟不能为负")
    if not 0 < args.tree_bandwidth_factor <= 1:
        raise ValueError("tree-bandwidth-factor 必须位于 (0, 1]")

    sizes = parse_sizes(args.sizes_mib)
    print(f"ranks={args.ranks} bandwidth={args.bandwidth_gbs}GB/s "
          f"alpha={args.alpha_us}us")
    print(f"{'MiB':>10} {'Ring_us':>12} {'Tree_us':>12} {'ModelPick':>12}")
    crossover = None
    for size_mib in sizes:
        size_bytes = size_mib * 1024 ** 2
        ring = ring_allreduce_us(
            size_bytes, args.ranks, args.alpha_us, args.bandwidth_gbs
        )
        tree = tree_allreduce_us(
            size_bytes, args.ranks, args.alpha_us,
            args.bandwidth_gbs, args.tree_bandwidth_factor
        )
        pick = "Ring" if ring < tree else "Tree"
        if pick == "Ring" and crossover is None:
            crossover = size_mib
        print(f"{size_mib:10.3f} {ring:12.3f} {tree:12.3f} {pick:>12}")

    print("\n注意：这是理解 alpha/B 权衡的教学模型，不代表 NCCL 选型器。")
    print("模型中的首次 Ring 优势消息大小:", crossover)

if __name__ == "__main__":
    main()
```

运行：

```
python nccl_cost_model.py --ranks 8 --bandwidth-gbs 25 --alpha-us 4
python nccl_cost_model.py --ranks 64 --bandwidth-gbs 25 --alpha-us 4
```

预期现象：小消息区 Tree 受益于较少的通信轮次；消息增大后 Ring 的带宽项可能占优；Rank 数增加时 Ring 的启动阶段线性增加，交叉点发生变化。这个模型不能替代 NCCL 的真实选择器，因为没有包含多 Channel、分层拓扑、不同 Protocol、网络 Offload 和链路并行。

### Level 1：PyTorch CPU/GPU Collective Benchmark

保存为 `collective_bench.py` ：

```
#!/usr/bin/env python3
import argparse
import json
import math
import os
import statistics
import time

import torch
import torch.distributed as dist

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--op", choices=["all_reduce", "all_gather", "reduce_scatter"],
                   default="all_reduce")
    p.add_argument("--sizes-mib", default="0.001,0.01,0.1,1,16,64")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    return p.parse_args()

def percentile(values, q):
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[index]

def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def run_collective(op, rank, world, numel, dtype, device):
    value = float(rank + 1)
    if op == "all_reduce":
        tensor = torch.full((numel,), value, dtype=dtype, device=device)
        dist.all_reduce(tensor)
        expected = world * (world + 1) / 2
        ok = torch.isclose(tensor[0].float(), torch.tensor(expected, device=device))
        return bool(ok.item())

    if op == "all_gather":
        source = torch.full((numel,), value, dtype=dtype, device=device)
        output = torch.empty(world * numel, dtype=dtype, device=device)
        dist.all_gather_into_tensor(output, source)
        checks = [
            torch.isclose(output[i * numel].float(),
                          torch.tensor(float(i + 1), device=device))
            for i in range(world)
        ]
        return all(bool(x.item()) for x in checks)

    source = torch.full((world * numel,), value, dtype=dtype, device=device)
    output = torch.empty(numel, dtype=dtype, device=device)
    dist.reduce_scatter_tensor(output, source)
    expected = world * (world + 1) / 2
    ok = torch.isclose(output[0].float(), torch.tensor(expected, device=device))
    return bool(ok.item())

def allocate_buffers(op, rank, world, numel, dtype, device):
    value = float(rank + 1)
    if op == "all_reduce":
        source = torch.full((numel,), value, dtype=dtype, device=device)
        return source, None
    if op == "all_gather":
        source = torch.full((numel,), value, dtype=dtype, device=device)
        output = torch.empty(world * numel, dtype=dtype, device=device)
        return source, output
    source = torch.full((world * numel,), value, dtype=dtype, device=device)
    output = torch.empty(numel, dtype=dtype, device=device)
    return source, output

def call_collective(op, source, output):
    if op == "all_reduce":
        dist.all_reduce(source)
    elif op == "all_gather":
        dist.all_gather_into_tensor(output, source)
    else:
        dist.reduce_scatter_tensor(output, source)

def main():
    args = parse_args()
    use_cuda = torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"
    dist.init_process_group(backend=backend)
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("CPU 回退请使用 float32")
    if backend == "gloo" and args.op == "reduce_scatter":
        raise ValueError("部分 Gloo 版本不支持 reduce_scatter_tensor；请用 GPU/NCCL")

    results = []
    element_size = torch.empty((), dtype=dtype).element_size()
    for size_mib in [float(x) for x in args.sizes_mib.split(",")]:
        numel = max(1, int(size_mib * 1024 ** 2 / element_size))

        # 先单独验证正确性，不把验证开销计入性能。
        if not run_collective(args.op, rank, world, numel, dtype, device):
            raise RuntimeError(f"rank {rank}: {args.op} correctness failed")
        sync(device)
        dist.barrier()

        source, output = allocate_buffers(
            args.op, rank, world, numel, dtype, device
        )
        for _ in range(args.warmup):
            if args.op == "all_reduce":
                source.fill_(rank + 1)
            call_collective(args.op, source, output)
        sync(device)
        dist.barrier()

        local_times_ms = []
        for _ in range(args.iters):
            if args.op == "all_reduce":
                source.fill_(rank + 1)
            sync(device)
            start = time.perf_counter()
            call_collective(args.op, source, output)
            sync(device)
            local_times_ms.append((time.perf_counter() - start) * 1000)

        # 同步作业应使用每次迭代最慢 Rank 的时间。
        times = torch.tensor(local_times_ms, dtype=torch.float32, device=device)
        dist.all_reduce(times, op=dist.ReduceOp.MAX)
        max_rank_times = times.cpu().tolist()

        payload_bytes = numel * element_size
        p50_ms = statistics.median(max_rank_times)
        p95_ms = percentile(max_rank_times, 0.95)
        algbw_gbs = payload_bytes / (p50_ms / 1000) / 1e9
        factor = (2 * (world - 1) / world
                  if args.op == "all_reduce" else (world - 1) / world)
        results.append({
            "size_mib_per_rank": payload_bytes / 1024 ** 2,
            "p50_ms_max_rank": p50_ms,
            "p95_ms_max_rank": p95_ms,
            "algbw_GBps": algbw_gbs,
            "nccl_tests_like_busbw_GBps": algbw_gbs * factor,
        })

    if rank == 0:
        nccl_version = None
        if use_cuda and hasattr(torch.cuda, "nccl"):
            try:
                nccl_version = torch.cuda.nccl.version()
            except Exception:
                pass
        print(json.dumps({
            "backend": backend,
            "world_size": world,
            "device": str(device),
            "torch_version": torch.__version__,
            "nccl_version": nccl_version,
            "op": args.op,
            "dtype": args.dtype,
            "results": results,
        }, indent=2))

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

### CPU 回退

```
python3 -m venv .venv-nccl
source .venv-nccl/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.4,<3"

torchrun --standalone --nproc-per-node=2 collective_bench.py \
  --op all_reduce --sizes-mib 0.001,0.01,0.1,1,16 --iters 20
```

CPU/Gloo 实验不能验证 NCCL、NVLink 或 GPUDirect RDMA，但可以学习 Rank 同步、Collective 正确性、消息大小曲线和最慢 Rank 计时方法。

### 通用 NVIDIA GPU

使用 PyTorch 官方安装选择器安装与驱动兼容的 CUDA Wheel，然后运行：

```
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  collective_bench.py --op all_reduce \
  --sizes-mib 0.001,0.01,0.1,1,16,64,256 --iters 50 --dtype float32

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  collective_bench.py --op all_gather \
  --sizes-mib 0.001,0.01,0.1,1,16,64 --iters 50 --dtype float32

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  collective_bench.py --op reduce_scatter \
  --sizes-mib 0.001,0.01,0.1,1,16,64 --iters 50 --dtype float32
```

预期现象：

- 极小消息的延迟相对稳定，带宽数字很低；
- 消息增大后带宽快速爬升，随后进入平台区；
- All-Gather 和 Reduce-Scatter 的内存需求与 All-Reduce 不同；
- 双 RTX 3080 通常走 PCIe，不应期待 NVLink 级结果；
- 实测曲线可能有抖动，应报告多次运行中位数和 P95，而不是只保留最好的一次。

### Level 2：nccl-tests、多节点与架构专项

### 构建 nccl-tests

`nccl-tests` 是 NVIDIA 官方维护的 NCCL 正确性与性能测试项目，采用 BSD-3-Clause 许可证。需要 CUDA Toolkit 和可用的 NCCL 开发头文件/库；只有 PyTorch Wheel 内置 Runtime 时，可能仍缺少编译所需 Header。

```
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make -j CUDA_HOME=/usr/local/cuda NCCL_HOME=/usr
```

双 GPU 单机扫描：

```
./build/all_reduce_perf -b 8 -e 512M -f 2 -g 2
./build/all_gather_perf -b 8 -e 256M -f 2 -g 2
./build/reduce_scatter_perf -b 8 -e 256M -f 2 -g 2
```

参数含义：

- `-b` ：最小消息大小；
- `-e` ：最大消息大小；
- `-f 2` ：每次翻倍；
- `-g 2` ：单进程管理两张 GPU。

不要在显存较小的 GPU 上盲目设置数 GiB 的最大消息。All-Gather 等操作还需要额外输出 Buffer，应从小区间逐步放大。

### 多节点 MPI

构建 MPI 版本：

```
make clean
make -j MPI=1 MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi \
  CUDA_HOME=/usr/local/cuda NCCL_HOME=/usr NAME_SUFFIX=_mpi
```

两台机器、每台 8 个 Rank 的示例：

```
mpirun -np 16 -N 8 --host node0:8,node1:8 \
  ./build/all_reduce_perf_mpi -b 8 -e 1G -f 2 -g 1
```

所有节点需要一致或兼容的二进制、Driver、CUDA/NCCL、网络库和设备权限。最终结果必须记录 Host 列表、Rank 映射、NIC、Rail、消息大小和 NCCL 版本。

### 安全的环境变量实验方法

先保存自动选择基线：

```
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,NET \
NCCL_DEBUG_FILE=/tmp/nccl.%h.%p.log \
./build/all_reduce_perf -b 8 -e 512M -f 2 -g 2
```

日志文件名必须包含 `%h` 和 `%p` ，否则多进程可能互相覆盖。日志会增加输出与少量开销，最终 Benchmark 应恢复较低日志级别。

在同一环境中做单变量 A/B：

```
NCCL_ALGO=Ring ./build/all_reduce_perf -b 8 -e 512M -f 2 -g 2
NCCL_ALGO=Tree ./build/all_reduce_perf -b 8 -e 512M -f 2 -g 2
NCCL_PROTO=Simple ./build/all_reduce_perf -b 8 -e 512M -f 2 -g 2
```

这些命令用于理解当前机器的选型边界，不应直接写进所有生产作业。不要在未知平台强制 `LL128` 。

Transport 诊断：

```
# 只用于对照：关闭 P2P 后若明显变慢，说明原路径受益于 P2P。
NCCL_P2P_DISABLE=1 ./build/all_reduce_perf -b 1M -e 512M -f 2 -g 2

# 多节点只用于判断 RDMA 与 Socket 差异，不是推荐生产配置。
NCCL_IB_DISABLE=1 mpirun ... ./build/all_reduce_perf_mpi ...

# 显式选择可达 Socket 网卡；名称按实机修改。
NCCL_SOCKET_IFNAME='=eth0' mpirun ... ./build/all_reduce_perf_mpi ...
```

选择 RDMA HCA 时可使用 `NCCL_IB_HCA` ，但设备名、端口与 `=` 精确匹配语义随官方文档配置，必须从 `ibv_devices` 、 `ibdev2netdev` 和 NCCL 日志确认，不能照抄别人的 `mlx5_0` 。

### NCCL RAS

NCCL 2.24 起提供 RAS 子系统，默认启用，用于在作业运行时查询 Communicator 和 Rank 健康状态。默认本地监听地址通常为 `localhost:28028` 。在长时间运行或疑似 Hang 时可查询：

```
ncclras -v
# 没有 ncclras 客户端时，可使用文本协议：
echo 'verbose status' | nc localhost 28028
```

如果一台节点同时运行多个独立作业，可为每个作业设置不同 `NCCL_RAS_ADDR` 。把 RAS 暴露到外部接口存在安全影响，生产环境应限制访问。

### 架构支持边界

| 架构 | 示例 | 可做实验 | 不可外推 |
| --- | --- | --- | --- |
| Ampere | RTX 3080/3090、A100 | PCIe/NVLink P2P、NCCL Ring/Tree、单/多节点 | 双 3080 无法验证 NVLS |
| Ada Lovelace | RTX 4090、L40/L40S | PCIe NCCL、数据中心网络（视系统） | RTX 4090 无 NVLink，且不是 Blackwell |
| Hopper | H100/H200 | SXM NVLink/NVSwitch、NVLS、RDMA | 需对应服务器和 Fabric，不是所有 H100 形态都相同 |
| Blackwell | RTX 5090、B100/B200/GB200 | PCIe 或数据中心 NVLink/Fabric，视具体产品 | RTX 5090 不能代表 B200/GB200 NVLS/MNNVL |

NVLS、MNNVL、SHARP 网络 Offload 依赖不同硬件、Fabric Manager、网络插件和部署条件。它们不能互相当作同义词，也不能由消费卡通过环境变量等价模拟。

## Early Release 变量纠错

主 PDF 是 2025 年 Early Release，其中列出的部分 NCCL 参数、默认值和 SHARP 开关已经不适合作为 2026 年通用配置：

- `NCCL_NCCL_MAX_RINGS` 不是应继续传播的当前通用变量；
- `NCCL_SHARP_ENABLE` 、 `NCCL_SHARP_CAPABLE` 不能当作所有 NCCL 安装都支持的通用开关；网络 SHARP 通常涉及外部网络插件与集群部署；
- 把 `NCCL_NTHREADS` 描述成“每 GPU CPU 线程数”是不准确的，它与 NCCL GPU Kernel 线程配置有关；
- `NCCL_BUFFSIZE` 、旧 Channel/Ring 变量和阈值不能脱离版本、拓扑和消息区间直接给“推荐值”；
- 当前官方文档建议 `NCCL_ALGO` 、 `NCCL_PROTO` 默认不设置，由 NCCL 自动选择；强制配置只用于经过测量的场景或 Bug 规避。

课程以生成当日官方 NCCL 文档为准。生产集群升级 NCCL 后，应重新跑消息扫描与端到端回归，而不是继承旧版本的全部环境变量。

## 优化前后对照

| 场景 | 优化前 | 优化后 | 验证方式 |
| --- | --- | --- | --- |
| 初始化 Hang | 随机调 Algorithm | 核对 Rank、接口、端口、SHM、日志 | 所有 Rank 完成初始化 |
| 单机带宽低 | 只看 GPU 型号 | 检查 P2P、PCIe Root、ACS 与拓扑 | `nccl-tests` 曲线、拓扑日志 |
| 多节点误走 Socket | 假设 RDMA 自动成功 | 验证 HCA、GDR、网卡与低层 RDMA | NCCL NET 日志、 `ib_write_bw` |
| 小消息延迟高 | 强制更大 Buffer | 融合消息、减少 Collective、验证 Tree | P50/P95、Collective 次数 |
| 大消息吃不满 | 只调 Protocol | 检查链路、NIC/PCIe 争用、Algorithm | BusBW 平台区 |
| 通信暴露高 | 等全部反向结束再同步 | Bucket 化并与计算重叠 | Timeline、Step time |
| 多作业偶发 Hang | 只保存报错尾部 | 使用独立日志、RAS、网络计数器 | 问题 Rank 与时间点 |
| 配置不可迁移 | 固化一组“神奇变量” | 默认基线 + 版本化 A/B 记录 | 升级回归结果 |

## 常见错误与排查

### 1\. NCCL INFO 很多就表示出错

不是。 `INFO` 是详细诊断级别。先寻找 `WARN` 、第一处失败和所有 Rank 是否一致，不要把正常拓扑打印当成错误。

### 2\. All-Reduce 结果错误或程序 Hang

检查每个 Rank 的调用顺序、元素数、数据类型和 Communicator。某 Rank 少执行一次 Collective，就会让其他 Rank 永久等待。

### 3\. nvidia-smi topo -m 显示 NVLink，但 NCCL 没走预期路径

检查容器 `/sys` 、P2P 权限、驱动、MIG、虚拟化、ACS 和 NCCL 日志。拓扑存在不代表应用路径一定可用。

### 4\. /dev/shm 报错

增加容器共享内存或使用 `--ipc=host` ，并检查 memlock。不要通过关闭 SHM 掩盖资源不足，除非正在做诊断对照。

### 5\. 多节点选择了 Docker/VPN 网卡

用 `ip -br addr` 确认接口，从所有节点互相测试可达性，再使用 `NCCL_SOCKET_IFNAME` 精确选择或排除接口。错误接口常表现为初始化慢或 Hang。

### 6\. NCCL\_IB\_DISABLE=1 反而更快

这通常意味着 RDMA 路径配置异常，而不是 Socket 天生更优。检查 GDR、HCA/NIC 亲和性、MTU、交换机、PFC/ECN、固件和错误计数器。

### 7\. 强制 Ring 在一个消息区间更快，就永久固定 Ring

真实模型包含多种消息大小和多种 Collective。固定 Ring 可能优化大 Bucket，却伤害小消息。要用真实训练端到端验证，并在 NCCL 升级后重新测试。

### 8\. BusBW 超过一条 PCIe/NVLink 标称值就是测试错了

BusBW 是 Collective 归一化指标，可能反映多链路和双向并发，不能直接等同单根链路速率。先核对公式、Rank 数和拓扑，再判断异常。

### 9\. async\_op=True 后立刻读取结果

异步只表示调用可以提前返回。必须建立正确依赖或等待 Work 完成，否则会得到未完成的数据或隐式同步，性能与正确性都可能出问题。

### 10\. 只测通信 Microbenchmark，不测训练

Microbenchmark 用于隔离通信上限；最终目标仍是训练 Step、MFU、Goodput 和 Time-to-Quality。通信 Kernel 更快但与计算争用加剧时，端到端可能变慢。

## 面试题与答案

### 1\. NCCL 与 MPI 有什么区别？

NCCL 面向 NVIDIA GPU 的 Collective 与点对点通信，理解 GPU 拓扑并以 CUDA Stream 异步执行；MPI 是更通用的进程通信标准与运行生态。多节点 `nccl-tests` 常用 MPI 启动进程，但实际 GPU Collective 仍由 NCCL 完成。

### 2\. 为什么 Ring All-Reduce 对大消息友好？

每个 Rank 的数据量趋近 2(N-1)S/N，链路可流水化，带宽利用高；代价是约 2(N-1) 个通信阶段，小消息容易被启动延迟支配。

### 3\. All-Reduce 为什么可以分成 Reduce-Scatter 和 All-Gather？

前者先完成归约并让每个 Rank 持有一片结果，后者再交换各片，使所有 Rank 得到完整归约结果。这也解释了 FSDP 为什么常直接使用分片 Collective。

### 4\. Algorithm 和 Protocol 有什么区别？

Algorithm 决定 Rank 与链路之间的整体通信拓扑，例如 Ring、Tree、NVLS；Protocol 决定链路上传输和同步的数据粒度与机制，例如 LL、LL128、Simple。

### 5\. 为什么不能默认强制 LL128？

平台支持和数据正确性有约束。官方文档警告，在不支持的平台启用 LL128 可能造成数据损坏；应让 NCCL 自动选择，除非在受控环境验证并用于特定故障规避。

### 6\. algbw 与 busbw 的区别是什么？

AlgBW 是算法输入数据量除以时间；BusBW 根据 Collective 的理论数据运动模式归一化。All-Reduce 常乘 2(N-1)/N。BusBW 更适合跨 Rank 数比较，但仍不等于单链路物理速率。

### 7\. 如何证明 NCCL 使用了 RDMA，而不是 Socket？

结合 NCCL `INIT/NET/GRAPH` 日志、HCA 与 GDR 信息、网卡计数器和 `ib_write_bw` 。再用 `NCCL_IB_DISABLE=1` 做受控对照，但不能仅凭环境中安装了 OFED 就下结论。

### 8\. NCCL Hang 最常见的应用原因是什么？

Rank 之间 Collective 调用顺序或参数不一致，或某 Rank 因异常、数据长尾、OOM 没有进入 Collective。应先查语义和慢 Rank，再查 Algorithm。

### 9\. 通信重叠为什么可能没有收益？

通信 Kernel 也会占用 SM、HBM、Copy/Network 资源；如果与计算争用，重叠只是在 Timeline 上并发，Step 仍可能变慢。必须比较端到端时间和资源利用率。

### 10\. 双 RTX 3080 能完成哪些 NCCL 实验？

可以验证双进程 NCCL、PCIe P2P/SHM 路径、Ring/Tree A/B、消息曲线、日志和计算重叠；不能验证 NVSwitch/NVLS、InfiniBand/RoCE 或 GB200 多节点 NVLink Fabric。

## 课后练习

1. 修改 Level 0 模型的 Rank 数、延迟和带宽，画出 Ring/Tree 交叉点变化。
2. 在 CPU/Gloo 上扫描 1KiB～16MiB，解释为什么带宽从低到高再平台化。
1. 在双 GPU 上运行 `all_reduce` 、 `all_gather` 、 `reduce_scatter` ，比较相同输入分片大小的 P50/P95。
2. 用 `nvidia-smi topo -m` 画出本机 GPU、CPU NUMA 与 NIC 拓扑，并预测 NCCL 路径。
1. 用 `NCCL_P2P_DISABLE=1` 做单变量 A/B，解释差异；实验后恢复默认。
2. 分别强制 Ring、Tree，只比较相同消息区间，并说明为什么不能直接固化配置。
1. 使用 Nsight Systems 检查一次 DDP 反向传播中 NCCL Kernel 与计算 Kernel 是否重叠。
2. 设计一个 Hang 排查表，要求从语义、进程、容器、单机拓扑、网络、RAS 六层逐步排除。

## Checklist

### 语义与正确性

- 每个 Rank 使用相同 Collective 顺序和兼容张量形状。
- 能区分 All-Reduce、Reduce-Scatter、All-Gather、All-to-All。
- 异步通信有正确的 Stream/Work 依赖。
- Benchmark 在计时前后正确同步。
- 正确性验证不混入性能计时。

### 性能模型

- 会计算 Ring All-Reduce 每 Rank 通信量。
- 能解释 Ring 与 Tree 的延迟-带宽权衡。
- 能区分 AlgBW、BusBW 与物理链路带宽。
- 同时报告消息大小、Rank 数、类型、拓扑和版本。
- 最终用端到端 Step 时间验证通信优化。

### 拓扑与 Transport

- 已检查 nvidia-smi topo -m、P2P 和 NUMA。
- 容器正确暴露 /sys、GPU、SHM 与 memlock。
- 多节点已验证 Socket/RDMA 基础连通性。
- GPU、NIC 与 PCIe Root 的映射符合预期。
- 没有把安装 RDMA 驱动当作已使用 GDR 的证明。

### 调参与诊断

- 先保存 NCCL 自动选择基线。
- 每次只改变一个变量并保留完整结果。
- 不在未知平台强制 LL128。
- 不把关闭 P2P/IB/SHM 的诊断变量当成生产优化。
- NCCL 日志文件名对每个 Host/进程唯一。
- NCCL 2.24+ Hang 可使用 RAS 辅助定位。

### 架构边界

- Ampere 正确包含 RTX 3080/3090、A100。
- Ada Lovelace 正确包含 RTX 4090、L40/L40S。
- Hopper 正确包含 H100/H200。
- Blackwell 正确包含 RTX 5090、B100/B200/GB200。
- 没有把 RTX 4090 写成 Blackwell。
- 没有用双 RTX 3080/RTX 5090 模拟 NVLS、SHARP 或 MNNVL。
- 没有用 RTX 5090 外推 GB200 Fabric 能力。