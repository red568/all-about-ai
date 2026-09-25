---
title: "第8课：NVLink、NVSwitch 与 AI 集群互联"
source: "https://everythingai.top/learn/ai-systems-performance-engineering-text/lesson-08"
author:
published:
created: 2026-09-26
description: "面向 AI 工程学习者的专业课程学习空间。"
tags:
  - "clippings"
---
**导航**

本课解析 NVLink、NVSwitch 与多机网络的互联路径，建立大规模 AI 集群通信性能的整体认识。

## 课程定位

单卡性能回答“一个 GPU 能算多快”，互联性能回答“很多 GPU 能否像一个系统那样工作”。当模型、参数、梯度、激活或 KV Cache 必须跨设备移动时，瓶颈往往从 Tensor Core 转移到 PCIe、NVLink、NVSwitch、NUMA 和节点网络。

本课建立从单机双卡到机架级 GPU 域的统一模型。重点不是背诵某代 NVLink 的峰值，而是学会回答四个工程问题：数据要经过哪条路径、每步移动多少字节、路径的有效带宽与固定延迟是多少、通信能否与计算重叠。

## 学习目标

完成本课后，你能够：

1. 区分 PCIe、NVLink、NVSwitch、NVLink-C2C 与 InfiniBand/RoCE 的角色。
2. 正确解读 `nvidia-smi topo -m` 的 `PIX/PXB/PHB/NODE/SYS/NV#` 。
1. 使用延迟—带宽模型估算 P2P、Ring AllReduce 和分层 AllReduce 时间。
2. 区分 CUDA P2P 可访问、实际传输路径与 NVLink 连接，避免把三者混为一谈。
1. 使用 PyTorch 和 `nccl-tests` 测量 GPU 间有效带宽、延迟与正确性。
2. 为数据并行、张量并行、流水并行和 MoE Expert Parallel 设计拓扑感知的 Rank 映射。
1. 识别 NUMA、PCIe Root Complex、ACS/IOMMU、主机中转和网络超售造成的瓶颈。
2. 为没有 NVLink/NVSwitch 的机器提供可执行基线，同时明确模拟不能替代硬件验证。

## 前置知识

- 理解 GPU、CPU、显存和主存的基本关系。
- 理解 Latency、Throughput、Goodput 与 Roofline。
- 知道分布式训练中的 Rank、AllReduce、AllGather、ReduceScatter。
- 会运行 Python 和 Bash；Level 1 需要 PyTorch，Level 2 需要多块 NVIDIA GPU。

## 核心直觉：多 GPU 性能由最窄的数据通道决定

把 AI 集群想成城市交通系统：

- Tensor Core 是工厂产能；
- HBM/GDDR 是厂内传送带；
- NVLink 是 GPU 之间的高速专线；
- NVSwitch 是连接多条高速专线的交换枢纽；
- PCIe 是兼容范围广的通用道路；
- InfiniBand/RoCE/Ethernet 是跨主机、跨机架网络。

增加 GPU 数量相当于增加工厂。若原料与半成品仍要经过一座窄桥，总产出不会线性增加。

因此，多 GPU 优化的第一原则是：

## 互联架构原理

### PCIe：通用 I/O 骨干

PCIe 连接 GPU、CPU、NIC、NVMe 与 PCIe Switch。两块 GPU 即使都标为 PCIe x16，它们之间的路径也可能不同：

```
较短路径：GPU 0 ─ PCIe Switch ─ GPU 1

较长路径：GPU 0 ─ Root Complex ─ CPU/NUMA Interconnect
                                  └─ Root Complex ─ GPU 1
```

路径越长，通常固定延迟越高，且更容易与网卡、存储或其他 GPU 争用共享上行链路。PCIe 标称带宽是链路能力，不等于 P2P Payload 带宽；编码、协议、事务大小、方向、拓扑和软件路径都会造成差异。

### CUDA P2P 不等于 NVLink

需要区分三件事：

1. \*\*Peer Access 能力\*\*：一个 GPU 是否可以访问另一个 GPU 的显存地址。
2. \*\*传输路径\*\*：数据经过 NVLink、PCIe P2P，还是回退到主机内存中转。
1. \*\*有效性能\*\*：给定消息大小和方向下的实际 GB/s 与微秒延迟。

`torch.cuda.can_device_access_peer(0, 1)` 返回 `True` ，只说明 CUDA P2P 能力可用；它不证明存在 NVLink。反过来，硬件有 NVLink 也不保证软件、驱动、容器权限和拓扑配置都正确。

### NVLink：GPU 间的高速 Scale-Up 链路

NVLink 是 NVIDIA 的高速互联。它常用于 GPU-GPU，也存在 Grace CPU 与 GPU 间的 NVLink-C2C。二者不能简单当成同一种“显卡桥”：

- 离散 GPU NVLink 关注 GPU 间高带宽传输；
- NVLink-C2C 是封装/模块级 CPU-GPU 芯粒互联，具有不同的一致性与系统语义；
- 支持与带宽取决于 GPU SKU、板卡形态和整机设计。

NVLink 也不自动把多块 GPU 变成对任意程序透明的统一共享内存。应用仍需使用 CUDA、NCCL、NVSHMEM、框架通信算子或支持 Fabric 的内存机制。

### NVSwitch：构建多 GPU NVLink 域

点对点 NVLink 桥适合少量 GPU。NVSwitch 作为专用交换芯片，把更多 GPU 组织成高带宽 NVLink Fabric：

```
GPU 0 ─┐
GPU 1 ─┤
GPU 2 ─┼─ NVSwitch Fabric ─ 任意目标 GPU
 ...   │
GPU N ─┘
```

NVSwitch 的价值是提高连接度、双向带宽和可预测性，并支持整机级拓扑设计。它不能消除：

- Collective 的必要数据量；
- Kernel 启动和协议延迟；
- 热点、路由和并发流量争用；
- 跨 NVLink 域后的 InfiniBand/RoCE 瓶颈；
- Rank 放置、消息切分和同步不当。

### Scale-Up 与 Scale-Out

AI 集群通常具有层级结构：

```
GPU ─ HBM/GDDR
 └─ PCIe/NVLink ─ 单节点或单 NVLink 域（Scale-Up）
      └─ NIC + InfiniBand/RoCE/Ethernet ─ 多节点（Scale-Out）
```

Scale-Up 让少量或数十块 GPU 低延迟协同；Scale-Out 让系统扩展到更多节点。跨层级 Collective 通常需要分层算法，不能把所有 Rank 当作同质全互联。

## 产品能力矩阵：必须按 SKU 与整机判断

| 架构 | 示例产品 | 互联定位 | 课程实验结论 |
| --- | --- | --- | --- |
| Ampere | RTX 3080 | PCIe，无 NVLink | 双卡做 PCIe/P2P 基线 |
| Ampere | RTX 3090/3090 Ti | 可选两卡 NVLink Bridge | 必须安装桥并由拓扑确认 |
| Ampere | A100 | PCIe/SXM；部分系统有桥或 NVSwitch | 按 PCIe、SXM、HGX 形态区分 |
| Ada Lovelace | RTX 4090 | PCIe，无 NVLink | 不能写成 Blackwell，也不能假设专用互联 |
| Ada Lovelace | L40/L40S | PCIe，无 NVLink | 适合 PCIe 与网络基线 |
| Hopper | H100/H200 | PCIe、SXM、NVL/HGX 等形态 | NVLink 4 能力取决于形态与系统 |
| Blackwell | RTX 5090 | PCIe Gen5，无 NVLink | 消费级 Blackwell 不等于 NVLink 5 |
| Blackwell | B100/B200/GB200 | NVLink 5；NVL/HGX 系统可含 NVSwitch | 作为数据中心专项实验 |

官方规格中，H100 SXM 的 NVLink 总带宽标为 900 GB/s，H100 NVL 标为 600 GB/s；GB200 NVL72 的 72-GPU NVLink 域标为 130 TB/s 聚合通信带宽、单 GPU NVLink 5 为 1.8 TB/s 量级。这些是特定产品配置的双向或聚合规格，不应与一次单向 P2P Copy 的有效带宽直接比较。

对双 RTX 3080 20GB 环境，正确预期是：两卡通过 PCIe 路径通信，没有可安装的 NVLink Bridge。是否支持直接 P2P、走 `PIX/PXB/PHB/SYS` 中哪条路径，必须由实际主板、CPU、BIOS、驱动与 `nvidia-smi topo -m` 决定。

## 拓扑识别

### nvidia-smi topo -m 图例

| 标记 | 含义 | 性能直觉 |
| --- | --- | --- |
| `X` | 当前设备自身 | 非通信路径 |
| `NV#` | 经过由 # 条 NVLink 组成的连接 | 常见高速 GPU P2P |
| `PIX` | 至多经过一个 PCIe Bridge/Switch | PCIe 中较短路径 |
| `PXB` | 经过多个 PCIe Bridge，不经过 Host Bridge | 路径更长、可能共享上行 |
| `PHB` | 经过 PCIe Host Bridge/CPU Root Complex | 常见跨 Root 路径 |
| `NODE` | 同 NUMA Node 内跨 PCIe Host Bridge | 比简单 PIX/PXB 更复杂 |
| `SYS` | 还经过 NUMA 节点间 CPU 互联 | 通常是最需警惕的单机路径 |

常用检查命令：

```
nvidia-smi --query-gpu=index,name,compute_cap,pci.bus_id,memory.total --format=csv
nvidia-smi topo -m
nvidia-smi topo -mp
nvidia-smi nvlink --status
lspci -tv
numactl -H
```

说明：部分驱动版本使用 `nvidia-smi nvlink -s` ；不支持 NVLink 的产品会报告无活动链路或不支持。 `topo -mp` 排除 NVLink、只展示 PCI 路径，适合确认 Fabric 下方的回退路径。

### 拓扑检查五问

1. GPU 对之间是 `NV#` 、 `PIX` 、 `PXB` 、 `PHB` 还是 `SYS` ？
2. GPU 与目标 NIC 是否处于同一 NUMA/PCIe 局部域？
1. P2P 能力矩阵是否对称且实际可用？
2. 链路宽度、代际和速率是否降级？
1. 容器、虚拟机、IOMMU/ACS 是否改变了直通或 P2P 路径？

## 关键性能模型

### 延迟—带宽模型

传输一条大小为 S 字节的消息，可先用：

$$
T_{msg}=\alpha+\frac{S}{B_{eff}}
$$

$\alpha$ ：软件、协议、调度与链路的固定延迟；

B\\\_{eff}：该消息大小和方向下的有效带宽；

S：实际 Payload 字节数。

小消息中 $\alpha$ 主导，大消息中 S/B\\\_{eff} 主导。用 1 GiB Copy 测出的 GB/s 不能回答 4 KiB 控制消息的延迟。

### Ring AllReduce

Ring AllReduce 可分解为 ReduceScatter 与 AllGather。若有 N 个 Rank，每个 Rank 的逻辑消息大小为 S，理想 Ring 中每个 Rank 的通信量为：

$$
V_{ring}=2\frac{N-1}{N}S
$$

简化时间模型：

$$
T_{ring}\approx2(N-1)\alpha+2\frac{N-1}{N}\frac{S}{B_{eff}}
$$

Ring 对大消息带宽效率高，但步数随 N 增长。树或递归倍增算法对小消息通常更有延迟优势，实际 NCCL 会结合消息大小、拓扑、协议和版本选算法。

### algbw 与 busbw

`nccl-tests` 的 `algbw` 表示算法观察到的 Payload 吞吐。为便于跨 Collective 比较，AllReduce 的近似总线带宽归一化为：

$$
busbw=algbw\times2\frac{N-1}{N}
$$

不要把 `busbw` 当成某一条物理链路的单向实测，也不要把它直接与厂商“全双工聚合带宽”数字比较。

### 通信占比与扩展效率

若一次 Step 的单卡计算时间为 $T_c$ ，多卡通信时间为 $T_{comm}$ ，可重叠部分为 $T_{overlap}$ ：

$$
T_{step}\approx T_c+T_{comm}-T_{overlap}
$$

相对理想计算的效率近似为：

$$
\eta\approx\frac{T_c}{T_c+T_{comm}-T_{overlap}}
$$

互联优化不只是增大带宽；增大 Bucket、减少同步、计算通信重叠、调整并行策略都在改变这个分母。

### 分层 AllReduce

设每节点 G 块 GPU，共 M 个节点。常见思路是：

1. 节点内用 NVLink/NVSwitch 或本地 PCIe 归约；
2. 节点间只让代表流或分片通过 NIC；
1. 节点内再广播/AllGather。

简化模型可写为：

$$
T_{hier}\approx T_{intra-reduce}+T_{inter-allreduce}+T_{intra-broadcast}
$$

真实实现可能使用多 Rail、多 Channel、ReduceScatter/AllGather 和并发流，模型用于定位层级瓶颈，不用于替代基准测试。

## Collective 与工作负载的关系

| 并行方式 | 典型通信 | 对互联的敏感点 |
| --- | --- | --- |
| DDP | 梯度 AllReduce | 大消息带宽、计算通信重叠 |
| FSDP/ZeRO | 参数 AllGather、梯度 ReduceScatter | Bucket、预取、峰值带宽与显存 |
| Tensor Parallel | 每层 AllReduce/AllGather | 极度依赖低延迟 Scale-Up |
| Pipeline Parallel | Stage 间激活 P2P | 相邻 Rank 映射、Microbatch Bubble |
| Context/Sequence Parallel | Attention 分片通信 | 消息规模与序列长度 |
| MoE Expert Parallel | AllToAll | 路由不均衡、双向带宽、尾延迟 |
| 解耦推理 | KV Cache 迁移 | 点对点大对象、网络与内存带宽 |

拓扑映射原则：通信最密集的 Rank 优先放在最快、最稳定的局部域中。例如张量并行组优先留在同一 NVSwitch 域；数据并行更适合跨节点，因为其梯度通信更容易做大 Bucket 和重叠。这是经验起点，不是固定规则。

## 瓶颈分析方法

### 第一步：确认问题确实是通信

观察：

- GPU Kernel 之间是否有长时间 NCCL/P2P 区间；
- Step Time 是否随 Rank 数快速恶化；
- 通信时间是否随消息大小近似线性增长；
- GPU 利用率下降时，NIC/NVLink/PCIe 是否繁忙；
- 是否存在某个 Rank 明显更慢造成全局等待。

### 第二步：把通信拆成四层

```
应用：并行策略、Bucket、同步频率、负载均衡
Collective：Ring/Tree、Channel、Protocol、Rank Mapping
传输：P2P、SHM、NET、GPUDirect RDMA、主机中转
硬件：NVLink/NVSwitch、PCIe、NUMA、NIC、交换网络
```

只调 NCCL 环境变量而不确认上层消息模式，通常无法解决根因。

### 第三步：做消息大小扫描

至少测三个区间：

- 小消息：暴露启动与同步延迟；
- 中消息：暴露算法切换和分块效果；
- 大消息：逼近持续带宽与共享链路上限。

必须保留 Size、Latency、 `algbw` 、 `busbw` 、错误数、拓扑、版本和 GPU 时钟/功耗状态。

### 第四步：做 A/B 隔离

- 单 GPU 与多 GPU；
- 同 NUMA 与跨 NUMA；
- 相邻 GPU 对与最远 GPU 对；
- P2P 与显式 Host-Staged；
- 单进程与多进程；
- 节点内与节点间；
- 默认 NCCL 与仅用于诊断的传输禁用选项。

禁用某条路径只能用于定位，不能把“绕开硬件”直接当作生产优化。

## 完整可运行实验：模型、拓扑与双 GPU Copy

本实验是一份独立脚本：

- Level 0：纯 Python，任何机器都能运行通信模型；
- Level 1：若安装 PyTorch，自动检测 GPU、Compute Capability、P2P 和拓扑，并测试双 GPU Copy；
- Level 2：另用 `nccl-tests` 验证真实 Collective。

### 环境准备

```
python3 -m venv .venv-interconnect
source .venv-interconnect/bin/activate
python -m pip install --upgrade pip

# Level 0 无第三方依赖
# Level 1 请从 PyTorch 官方安装页选择与本机驱动匹配的稳定版本
```

### 完整代码：interconnect\_lab.py

```
#!/usr/bin/env python3
"""通用 GPU 互联实验：Level 0 模型 + Level 1 可选 PyTorch 实测。"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass

@dataclass
class Link:
    name: str
    bandwidth_gbs: float
    latency_us: float

def transfer_ms(size_mib: float, link: Link, steps: int = 1,
                volume_factor: float = 1.0) -> float:
    """十进制 GB/s + 二进制 MiB；返回教学模型时间，不代表硬件实测。"""
    size_bytes = size_mib * 1024**2 * volume_factor
    return steps * link.latency_us / 1000.0 + size_bytes / (link.bandwidth_gbs * 1e9) * 1000.0

def ring_allreduce_ms(size_mib: float, ranks: int, link: Link) -> float:
    if ranks < 2:
        return 0.0
    steps = 2 * (ranks - 1)
    volume = 2 * (ranks - 1) / ranks
    # 延迟按步数计算，Payload 总量按 volume 计算。
    return steps * link.latency_us / 1000.0 + transfer_ms(
        size_mib, link, steps=0, volume_factor=volume
    )

def recursive_doubling_ms(size_mib: float, ranks: int, link: Link) -> float:
    """小消息教学近似；非 2 次幂 Rank 与 NCCL 实现会有不同路径。"""
    if ranks < 2:
        return 0.0
    steps = math.ceil(math.log2(ranks))
    return transfer_ms(size_mib, link, steps=steps, volume_factor=steps)

def hierarchical_ms(size_mib: float, gpus_per_node: int, nodes: int,
                    intra: Link, inter: Link) -> float:
    if nodes < 2:
        return ring_allreduce_ms(size_mib, gpus_per_node, intra)
    local_steps = math.ceil(math.log2(max(gpus_per_node, 1)))
    local_reduce = transfer_ms(size_mib, intra, local_steps, local_steps)
    inter_reduce = ring_allreduce_ms(size_mib, nodes, inter)
    local_broadcast = transfer_ms(size_mib, intra, local_steps, local_steps)
    return local_reduce + inter_reduce + local_broadcast

def run_level0(args: argparse.Namespace) -> dict:
    intra = Link("intra", args.intra_gbs, args.intra_latency_us)
    inter = Link("inter", args.inter_gbs, args.inter_latency_us)
    rows = []
    for size in args.sizes_mib:
        ring = ring_allreduce_ms(size, args.ranks, intra)
        tree = recursive_doubling_ms(size, args.ranks, intra)
        hier = hierarchical_ms(size, args.gpus_per_node, args.nodes, intra, inter)
        for algorithm, comm_ms in (
            ("ring", ring),
            ("recursive_doubling_teaching_model", tree),
            ("hierarchical_teaching_model", hier),
        ):
            efficiency = args.compute_ms / (args.compute_ms + comm_ms)
            rows.append({
                "size_mib": size,
                "algorithm": algorithm,
                "comm_ms": round(comm_ms, 4),
                "compute_ms": args.compute_ms,
                "efficiency_without_overlap": round(efficiency, 4),
            })
    return {
        "warning": "All bandwidth/latency inputs are user assumptions, not measured hardware specs.",
        "intra_link": asdict(intra),
        "inter_link": asdict(inter),
        "ranks": args.ranks,
        "gpus_per_node": args.gpus_per_node,
        "nodes": args.nodes,
        "results": rows,
    }

def command_output(argv: list[str]) -> str:
    if not shutil.which(argv[0]):
        return f"{argv[0]} not found"
    result = subprocess.run(argv, capture_output=True, text=True, timeout=15,
                            check=False)
    text = (result.stdout or result.stderr).strip()
    return text if text else f"exit_code={result.returncode}"

def sync_pair(torch, a: int, b: int) -> None:
    torch.cuda.synchronize(a)
    torch.cuda.synchronize(b)

def benchmark_cross_device_copy(torch, src_device: int, dst_device: int,
                                size_mib: int, warmup: int, repeat: int) -> dict:
    elements = size_mib * 1024**2 // 4
    with torch.cuda.device(src_device):
        src = torch.arange(elements, dtype=torch.float32, device=src_device)
    with torch.cuda.device(dst_device):
        dst = torch.empty(elements, dtype=torch.float32, device=dst_device)

    for _ in range(warmup):
        dst.copy_(src, non_blocking=True)
    sync_pair(torch, src_device, dst_device)

    start = time.perf_counter()
    for _ in range(repeat):
        dst.copy_(src, non_blocking=True)
    sync_pair(torch, src_device, dst_device)
    elapsed_s = (time.perf_counter() - start) / repeat

    # 抽样检查，避免把完整 Tensor 拷回 CPU 干扰计时。
    sample_idx = torch.tensor([0, elements // 2, elements - 1], device=dst_device)
    got = dst[sample_idx].cpu()
    expected = torch.tensor([0, elements // 2, elements - 1], dtype=torch.float32)
    correct = bool(torch.equal(got, expected))
    gib_per_s = (size_mib / 1024.0) / elapsed_s
    return {
        "size_mib": size_mib,
        "latency_ms": round(elapsed_s * 1000, 4),
        "effective_gib_s": round(gib_per_s, 3),
        "sample_correct": correct,
        "label": "cross-device copy; path must be interpreted with topology/P2P data",
    }

def benchmark_host_staged(torch, src_device: int, dst_device: int,
                          size_mib: int, warmup: int, repeat: int) -> dict:
    elements = size_mib * 1024**2 // 4
    with torch.cuda.device(src_device):
        src = torch.ones(elements, dtype=torch.float32, device=src_device)
    host = torch.empty(elements, dtype=torch.float32, pin_memory=True)
    with torch.cuda.device(dst_device):
        dst = torch.empty(elements, dtype=torch.float32, device=dst_device)

    def one_copy() -> None:
        host.copy_(src, non_blocking=True)
        torch.cuda.synchronize(src_device)
        dst.copy_(host, non_blocking=True)
        torch.cuda.synchronize(dst_device)

    for _ in range(warmup):
        one_copy()
    start = time.perf_counter()
    for _ in range(repeat):
        one_copy()
    elapsed_s = (time.perf_counter() - start) / repeat
    correct = float(dst[0].item()) == 1.0 and float(dst[-1].item()) == 1.0
    gib_per_s = (size_mib / 1024.0) / elapsed_s
    return {
        "size_mib": size_mib,
        "latency_ms": round(elapsed_s * 1000, 4),
        "effective_payload_gib_s": round(gib_per_s, 3),
        "sample_correct": correct,
        "label": "explicit GPU->pinned-host->GPU baseline; two transfer legs",
    }

def run_level1(args: argparse.Namespace) -> dict:
    try:
        import torch
    except ImportError:
        return {"skipped": "PyTorch not installed; Level 0 remains fully usable."}

    result = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "topology": command_output(["nvidia-smi", "topo", "-m"]),
        "nvlink_status": command_output(["nvidia-smi", "nvlink", "--status"]),
    }
    if not torch.cuda.is_available():
        result["skipped"] = "CUDA unavailable."
        return result

    count = torch.cuda.device_count()
    result["gpus"] = [
        {
            "index": i,
            "name": torch.cuda.get_device_name(i),
            "compute_capability": list(torch.cuda.get_device_capability(i)),
            "memory_gib": round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 2),
        }
        for i in range(count)
    ]
    result["peer_access_matrix"] = [
        [True if i == j else bool(torch.cuda.can_device_access_peer(i, j))
         for j in range(count)]
        for i in range(count)
    ]
    if count < 2:
        result["skipped"] = "Need at least two visible CUDA GPUs for copy benchmark."
        return result

    result["copy_0_to_1"] = []
    result["host_staged_0_to_1"] = []
    for size in args.gpu_sizes_mib:
        try:
            result["copy_0_to_1"].append(
                benchmark_cross_device_copy(torch, 0, 1, size, args.warmup, args.repeat)
            )
            result["host_staged_0_to_1"].append(
                benchmark_host_staged(torch, 0, 1, size, args.warmup, args.repeat)
            )
        except RuntimeError as exc:
            result.setdefault("errors", []).append({"size_mib": size, "error": str(exc)})
            break
    return result

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", choices=("0", "1", "all"), default="all")
    parser.add_argument("--sizes-mib", type=float, nargs="+", default=[0.004, 1, 64, 1024])
    parser.add_argument("--ranks", type=int, default=8)
    parser.add_argument("--gpus-per-node", type=int, default=4)
    parser.add_argument("--nodes", type=int, default=2)
    parser.add_argument("--intra-gbs", type=float, default=25.0)
    parser.add_argument("--intra-latency-us", type=float, default=8.0)
    parser.add_argument("--inter-gbs", type=float, default=12.5)
    parser.add_argument("--inter-latency-us", type=float, default=20.0)
    parser.add_argument("--compute-ms", type=float, default=50.0)
    parser.add_argument("--gpu-sizes-mib", type=int, nargs="+", default=[4, 64, 256])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    output = {}
    if args.level in ("0", "all"):
        output["level0"] = run_level0(args)
    if args.level in ("1", "all"):
        output["level1"] = run_level1(args)
    print(json.dumps(output, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
```

### Level 0：CPU/通用回退实验

把代码保存为 `interconnect_lab.py` 后运行：

```
python interconnect_lab.py --level 0
```

自定义两级链路：

```
python interconnect_lab.py --level 0 \
  --ranks 8 --gpus-per-node 4 --nodes 2 \
  --intra-gbs 25 --intra-latency-us 8 \
  --inter-gbs 12.5 --inter-latency-us 20 \
  --compute-ms 50
```

这些默认数字只是教学输入，不代表 RTX 3080、PCIe、NVLink 或网络的官方性能。正确做法是把实测结果回填为参数。

### 预期现象

- 几 KiB 小消息中，低步数模型常更有优势；
- 消息增大后，带宽项主导，Ring 的通信量优势更明显；
- 降低 `inter-gbs` 会让分层模型迅速受节点间网络限制；
- `comm_ms` 增大时， `efficiency_without_overlap` 下降；
- 增大 `compute-ms` 会提高表面效率，但不代表通信本身变快。

### 结果分析

模型用于回答“哪个变量最敏感”，而不是预测 NCCL 到小数点后的时间。真实系统还包含并发 Channel、协议、缓存、拓扑路由、GPU Clock、主机噪声和计算重叠。

### Level 1：常见 NVIDIA GPU 自动检测与双卡实测

```
python interconnect_lab.py --level 1 --gpu-sizes-mib 4 64 256
```

脚本会输出：

- PyTorch、CUDA、GPU 名称、Compute Capability、显存；
- `nvidia-smi topo -m` ；
- NVLink 状态；
- P2P 能力矩阵；
- GPU 0 到 GPU 1 的 Cross-Device Copy；
- 显式 GPU→Pinned Host→GPU 回退基线。

### 双 RTX 3080 20GB 的预期

- 拓扑不会显示 `NV#` ；
- 可能显示 `PIX/PXB/PHB/SYS` ，取决于主板和 CPU；
- P2P 可能可用，也可能因平台、IOMMU/ACS 或虚拟化限制不可用；
- Cross-Device Copy 通常优于显式 Host-Staged，但不是所有平台的必然结论；
- 256 MiB 测试约为每卡额外占用数百 MiB，20GB 显存足够；若系统已有负载，可改为 `4 32 128` 。

### 计时边界

脚本使用设备同步与墙钟测量端到端完成时间，避免只测异步 Launch。它适合教学 A/B，不替代 CUDA Event、Nsight Systems、CUDA Samples 或专业互联基准。 `effective_gib_s` 是单向 Payload 除以时间，不是全双工聚合带宽。

### Level 2：NVLink/NVSwitch/NCCL 专项实验

Level 2 需要两块及以上 NVIDIA GPU、CUDA Toolkit、NCCL 开发包和编译工具。使用 NVIDIA 官方 `nccl-tests` ：

```
git clone --depth 1 https://github.com/NVIDIA/nccl-tests.git
make -C nccl-tests -j CUDA_HOME=/usr/local/cuda

NCCL_DEBUG=INFO \
NCCL_DEBUG_SUBSYS=INIT,GRAPH \
./nccl-tests/build/all_reduce_perf -b 8 -e 1G -f 2 -g 2
```

若 CUDA/NCCL 不在默认目录：

```
make -C nccl-tests -j \
  CUDA_HOME=/opt/cuda \
  NCCL_HOME=/opt/nccl
```

多节点才需要 MPI 构建：

```
make -C nccl-tests -j MPI=1 \
  MPI_HOME=/path/to/mpi \
  CUDA_HOME=/usr/local/cuda \
  NCCL_HOME=/usr

mpirun -np 4 -N 2 \
  ./nccl-tests/build/all_reduce_perf_mpi -b 8 -e 1G -f 2 -g 1
```

`-g 2` 表示每个线程使用两块 GPU；容器中先检查 `CUDA_VISIBLE_DEVICES` 。测试同时给出 Out-of-place/In-place 时间、 `algbw` 、 `busbw` 和错误数，只有错误数为 0 的结果才可用于性能结论。

### Level 2 支持边界

| 环境 | 可验证内容 | 不能验证的内容 |
| --- | --- | --- |
| 双 RTX 3080 | PCIe/P2P、NCCL 两卡基线 | NVLink/NVSwitch |
| 双 RTX 3090 + Bridge | 两卡 NVLink 路径 | 多 GPU NVSwitch Fabric |
| RTX 4090/L40S/RTX 5090 | PCIe 与网络路径 | NVLink Bridge |
| A100/H100/H200 HGX | NVLink/NVSwitch、NCCL Collective | 不能外推到 GB200 NVLink 5 |
| B100/B200/GB200 NVL | NVLink 5/NVSwitch 与新系统能力 | 不能用消费卡等价模拟 |

没有对应硬件时，Level 0 可以模拟带宽变化对算法的影响，但不能验证 NVLink/NVSwitch 的协议、路由、并发、SHARP 或真实吞吐。

## 优化方法

### 1\. 拓扑感知 Rank 放置

把高频、细粒度通信组放在更快的域：

- Tensor Parallel 优先同一 NVSwitch/NVLink 域；
- Pipeline 相邻 Stage 优先使用短路径；
- MoE 高频互访 Expert 避免跨慢速域；
- GPU 与 NIC 尽量保持 PCIe/NUMA 局部性。

### 2\. 减少总字节与同步次数

- 使用 ReduceScatter + AllGather 代替不必要的全量副本；
- 梯度累积降低同步频率；
- 通信压缩或低精度通信必须同时验证精度；
- 避免重复 Layout 转换和隐式 Host Copy；
- 合并过小消息，但防止 Bucket 大到无法重叠。

### 3\. 计算通信重叠

理想重叠条件：

- 计算与通信使用合适的 CUDA Stream；
- 依赖事件正确；
- 通信不会争用计算必需的 HBM、Copy Engine 或 SM 资源；
- Bucket 足够早地产生；
- 最后一个 Bucket 不形成长尾。

“Timeline 中同时出现”不等于完全隐藏，必须看 Step Time 是否实际下降。

### 4\. 选择合适的 Collective 与消息粒度

- 小消息优先减少启动和步数；
- 大消息优先提高持续带宽；
- 异构层级使用 Hierarchical Collective；
- AllToAll 还要关注流量偏斜与尾部 Rank；
- 不手工固定算法前，先对默认自适应选择做完整扫描。

### 5\. 保证 NUMA 与 GPUDirect 路径

- 将数据加载线程与 GPU/NIC 绑定到合理 CPU/NUMA 节点；
- 检查 NIC 与 GPU 的 `topo -m` 距离；
- 验证 GPUDirect RDMA 是否实际启用；
- 避免 Pinned Memory 全部落在远端 NUMA；
- 虚拟化、容器与安全策略改变路径时重新基准测试。

## 优化前后对照

| 维度 | 常见低效方案 | 优化方向 | 验证指标 |
| --- | --- | --- | --- |
| Rank 映射 | TP Rank 跨 `SYS` 或跨节点 | TP 留在快速局部域 | TP Collective P50/P99 |
| 消息粒度 | 大量几 KiB 同步 | 合并 Bucket、减少 Launch | 小消息 Latency、Step Time |
| 数据路径 | 隐式 Host-Staged | 验证 P2P/GPUDirect | P2P GB/s、Host 流量 |
| Collective | 所有规模固定 Ring | 按规模/拓扑选算法 | `algbw/busbw` 扫描 |
| 重叠 | 计算结束后统一通信 | Bucket Ready 即通信 | Exposed Comm Time |
| NUMA | CPU、GPU、NIC 随机放置 | 绑定本地 CPU/Memory/NIC | 带宽、尾延迟、抖动 |
| 扩展 | 只看平均 GPU 利用率 | 看最慢 Rank 与 Goodput | Step P99、有效样本/s |

任何优化都要同时检查正确性。Collective 参数不一致、Rank 到设备映射错误或异步依赖遗漏，可能表现为 Hang、数据错误或“异常快”的无效结果。

## 常见错误与排查

### 错误 1：看到 P2P=True 就宣布启用了 NVLink

排查：同时检查 `nvidia-smi topo -m` 、NVLink 状态与带宽实测。PCIe P2P 同样可以返回 True。

### 错误 2：用 RTX 4090/5090 做 NVLink 实验

两者官方产品规格的 NVLink 支持均为 No。RTX 4090 是 Ada，RTX 5090 是 Blackwell，但架构代际不等于产品拥有数据中心互联特性。

### 错误 3：把理论双向聚合带宽与单向 Copy 比较

先统一：单向/双向、每 GPU/全系统、十进制 GB/s/二进制 GiB/s、Payload/链路、单流/多流。

### 错误 4：AllReduce Hang

检查：

- 所有 Rank 是否以相同 Count、dtype、顺序调用 Collective；
- Rank 是否映射到唯一 GPU；
- 防火墙、接口选择与进程启动是否一致；
- 是否某 Rank OOM 或先发生 CUDA Error；
- 先用小规模、单节点、 `NCCL_DEBUG=INFO` 复现。

### 错误 5：跨卡 Copy 比 Host-Staged 还慢

可能原因：P2P 未启用、跨 `SYS` 、ACS/IOMMU、链路降级、GPU 降频、其他设备争用、消息太小、同步方式不当。先看拓扑和 P2P，再做大小扫描。

### 错误 6：nvidia-smi topo -m 在容器中不可用

检查驱动工具是否挂载、容器是否使用 NVIDIA Container Runtime、设备权限是否完整。容器中的 PyTorch 可见 GPU，不代表所有管理命令和 NVML 能力都可见。

### 错误 7：安装了 RTX 3090 Bridge 但看不到 NV#

检查 Bridge 型号、插槽间距、两卡型号、物理安装、驱动、主板布局和供电；以拓扑输出为准，不以“产品支持”推断“当前已连接”。

### 错误 8：NCCL busbw 高于某条链路标称值，认为测量错误

`busbw` 是 Collective 的归一化指标，可能包含多个并行链路与 Ring 通信因子，不是单物理链路计数器。结合 `algbw` 、Rank 数和拓扑解释。

### 错误 9：跨节点带宽低就只调 NCCL\_ALGO

先检查 GPU-NIC 亲和性、RDMA、MTU、Rail、交换网络超售、ECN/PFC、CPU/IRQ 与进程映射。算法只是链路之上的一层。

## 结果复盘模板

每次互联实验至少记录：

```
日期与机器：
GPU/数量/板卡形态：
CPU/NUMA：
Driver/CUDA/PyTorch/NCCL：
拓扑矩阵：
NVLink 状态：
GPU-NIC 亲和性：
P2P 能力矩阵：
测试工具与 Commit/版本：
消息大小/方向/并发数：
Latency/algbw/busbw/错误数：
功耗/时钟/温度：
优化变量：
优化前后 Step Time 与 Goodput：
正确性检查：
```

如果没有记录拓扑、版本和消息大小，“带宽为 80 GB/s”几乎不可复现。

## 面试题与答案

### 1\. NVLink 和 NVSwitch 有什么区别？

NVLink 是端点间高速链路协议与物理连接；NVSwitch 是连接多条 NVLink、构建多 GPU Fabric 的交换芯片/系统。少量 GPU 可以点对点连接，更多 GPU 需要交换结构提高连接度。

### 2\. can\_device\_access\_peer=True 能证明有 NVLink 吗？

不能。它只证明 CUDA P2P 访问能力。PCIe P2P 也可为 True；必须结合拓扑和实测判断路径。

### 3\. 为什么 Ring AllReduce 适合大消息？

每 Rank 数据量约为 2(N-1)S/N，可把大消息切块并让多条链路持续流水。但它有 2(N-1) 个逻辑步骤，小消息容易被固定延迟主导。

### 4\. algbw 与 busbw 的区别？

`algbw` 是应用 Payload 大小除以算法耗时； `busbw` 根据 Collective 通信量做归一化。AllReduce 常乘 2(N-1)/N。

### 5\. 为什么 Tensor Parallel 更偏好 NVLink/NVSwitch？

TP 常在每个 Transformer Layer 内发生 Collective，频率高、可隐藏窗口小，对低延迟和高带宽非常敏感。跨慢速网络会把每层同步累积成明显长尾。

### 6\. NVSwitch 是否让多 GPU 等价于一块大 GPU？

不是。它提供高速 Fabric，但显存地址、同步、数据放置和计算调度仍由软件管理。不同 API 可能提供更统一的访问语义，也不代表通信成本消失。

### 7\. 双 RTX 3080 如何做本课实验？

用 `nvidia-smi topo -m` 确认 PCIe 路径，用 PyTorch 测 P2P 能力和 Cross-Device Copy，再用 `nccl-tests` 测两卡 AllReduce。不能验证 NVLink/NVSwitch，但能完整学习测量方法。

### 8\. 为什么大消息带宽正常，训练仍然扩展差？

训练可能由许多小 Bucket、同步长尾、Rank 不均衡、计算通信无法重叠或跨 NUMA 数据加载拖慢。1 GiB 单流峰值不能代表真实消息分布。

### 9\. 如何判断通信是否被计算隐藏？

看 Timeline 中通信与计算重叠后，Step Critical Path 上仍暴露多少通信时间；再通过禁用/改变重叠的 A/B 实验验证 Step Time，而不是只看时间轴颜色重合。

### 10\. GPU-NIC 拓扑为什么重要？

跨节点数据先从 GPU 到 NIC。若两者跨 CPU Socket/NUMA，流量会增加 PCIe 与 CPU 互联跳数，降低 GPUDirect RDMA 效率并增加尾延迟。

## 课后练习

1. 在自己的机器运行 `nvidia-smi topo -m` ，画出 GPU、CPU、NIC 的实际路径。
2. 修改 Level 0 的带宽和延迟，找出 8 Rank 下 Ring 与小消息模型的交点。
1. 测量 4、64、256 MiB Cross-Device Copy，解释为什么 GB/s 随消息大小变化。
2. 在双卡机器上运行 `all_reduce_perf` ，手算一个 Size 对应的 `busbw/algbw` 比值。
1. 比较 GPU 0→1 与 1→0，判断方向是否对称并解释误差。
2. 设计 2 节点 × 8 GPU 的 $TP=8$ 、 $DP=2$ Rank 映射，说明为什么这样放置。
1. 用 Nsight Systems 标出一次 DDP Step 中暴露的通信时间，提出两种优化。
2. 写一份“理论规格与实测不可直接比较”的单位规范。

## Checklist

### 概念

- 能区分 PCIe、NVLink、NVSwitch、NVLink-C2C 与节点网络。
- 知道 P2P 能力不等于 NVLink。
- 能解释 Scale-Up 与 Scale-Out。
- 不把 NVSwitch 描述成“通信免费”或透明统一显存。

### 拓扑

- 已保存 nvidia-smi topo -m。
- 能解释 NV#/PIX/PXB/PHB/NODE/SYS。
- 已确认 GPU-GPU 与 GPU-NIC NUMA 局部性。
- 已检查 P2P 能力、链路状态和可能的回退路径。

### 测量

- 小、中、大消息均已扫描。
- 区分单向、双向、聚合、Payload 与归一化带宽。
- 基准包含 Warmup、同步和正确性检查。
- 已记录 Driver、CUDA、NCCL、PyTorch 与工具版本。
- 性能数字只作为本次环境观察，不外推到所有 GPU。

### 优化

- 高频通信组优先放在快速局部域。
- 已检查消息合并、Bucket 与同步频率。
- 已测量暴露通信时间，而不是只看重叠图形。
- 已检查 NUMA、NIC、RDMA 与共享 PCIe 上行。
- 优化前后同时验证 Step Time、Goodput 与正确性。

### 硬件边界

- RTX 3080 按 PCIe 基线处理。
- RTX 3090 只有安装正确 Bridge 并显示 NV# 才算 NVLink 环境。
- RTX 4090 是 Ada 且无 NVLink。
- RTX 5090 是 Blackwell 消费卡且无 NVLink。
- H100/H200、B100/B200/GB200 按具体 SKU 与整机形态描述。
- 没有用软件模拟冒充 NVLink、NVSwitch、SHARP 或 RDMA 实测。