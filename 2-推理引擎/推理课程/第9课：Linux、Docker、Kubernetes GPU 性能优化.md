---
title: "第9课：Linux、Docker、Kubernetes GPU 性能优化"
source: "https://everythingai.top/learn/ai-systems-performance-engineering-text/lesson-09"
author:
published:
created: 2026-09-26
description: "面向 AI 工程学习者的专业课程学习空间。"
tags:
  - "clippings"
---
**导航**

本课从 Linux、容器与 Kubernetes 全链路定位 GPU 之外的系统瓶颈，让训练和推理环境稳定高效。

## 课程定位

GPU 利用率低，不一定是 GPU Kernel 慢。数据可能卡在磁盘、CPU 解码、NUMA 远端内存、Pinned Memory、容器共享内存、CPU CFS 配额或 Kubernetes 错误调度上。此时继续优化 CUDA Kernel，往往等于给堵车中的跑车换发动机。

本课建立从 Linux 主机到容器再到 Kubernetes Pod 的性能诊断链。课程不会提供一份“复制粘贴后永久修改所有 sysctl”的激进脚本，而是采用可复现的工程方法：先采集基线，再只改一个变量，记录回滚方法，最后用端到端 Goodput 验证。

## 学习目标

完成本课后，你能够：

1. 解释 Linux 调度、NUMA、Page Cache、Swap、THP、Pinned Memory 如何影响 GPU。
2. 区分 NVIDIA Driver、CUDA Driver API、CUDA Runtime、Toolkit 与容器内用户态库。
1. 正确配置和验证 NVIDIA Container Toolkit，而不是把完整驱动打进镜像。
2. 使用 cgroup v2 指标定位 CPU Throttling、Memory Pressure 与 OOM。
1. 解释 Docker 的 CPU/Memory/ `/dev/shm` /IPC/NUMA 配置对训练与推理的影响。
2. 正确配置 Kubernetes GPU Resource、QoS、CPU Manager 与 Topology Manager。
1. 区分整卡、Time-Slicing、MPS 与 MIG 的共享和隔离语义。
2. 在 CPU-only、通用 NVIDIA GPU 和数据中心专项环境中完成分层实验。

## 前置知识

- 会使用 Linux Shell、Python、Docker；Level 2 需要可选 Kubernetes。
- 理解 GPU Memory、PCIe、NVLink、NUMA 与 Pinned Memory。
- 理解 Latency、Throughput、P50/P99、Goodput。
- 了解 Pod、Container、Request、Limit 的基本概念。

## 核心直觉：GPU 是流水线末端的消费者

训练或推理数据路径可抽象为：

```
Storage / Network
        ↓
Page Cache / Filesystem
        ↓
CPU Read → Decode → Tokenize → Collate
        ↓
Pinned Host Memory
        ↓ PCIe / NVLink-C2C
GPU Memory
        ↓
GPU Kernels
```

系统吞吐受最慢阶段限制：

$$
R_{pipeline}\leq \min(R_{storage},R_{decode},R_{collate},R_{H2D},R_{GPU})
$$

若 GPU 每个 Batch 计算 40 ms，数据准备却要 55 ms，那么 GPU 再快 50% 也不会让端到端吞吐提高 50%。真正要优化的是 55 ms 的上游阶段，或者用预取把它与 GPU 计算重叠。