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