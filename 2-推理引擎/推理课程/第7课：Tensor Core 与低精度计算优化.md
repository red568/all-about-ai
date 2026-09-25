---
title: "第7课：Tensor Core 与低精度计算优化"
source: "https://everythingai.top/learn/ai-systems-performance-engineering-text/lesson-07"
author:
published:
created: 2026-09-26
description: "面向 AI 工程学习者的专业课程学习空间。"
tags:
  - "clippings"
---
**导航**

本课讲清 Tensor Core 与低精度计算的工作机制，帮助你在速度、显存和数值稳定性之间取得平衡。

## 课程定位

低精度是现代 AI 性能工程中最容易“看起来懂了、实际用错了”的领域。把 FP32 Tensor 转成 FP16 只是一种数据类型转换；只有算子、形状、布局、硬件和软件栈共同满足条件，计算才可能进入 Tensor Core 高吞吐路径。即使 Kernel 变快，数值溢出、额外量化、数据搬运或其他非矩阵算子也可能让端到端性能没有提升。

本课从矩阵乘累加的硬件直觉出发，建立 TF32、FP16、BF16、FP8、FP4、INT8/INT4 的统一理解，讲清输入精度、累加精度、存储精度和缩放粒度的区别。实战部分不绑定某一型号：CPU 可以完成数值模拟，常见 NVIDIA GPU 可以自动检测并测试 FP32/TF32/FP16/BF16，Ada/Hopper/Blackwell 可按能力选择 FP8，Blackwell 再选择 NVFP4。

## 学习目标

完成本课后，你能够：

1. 解释 Tensor Core 与普通 CUDA Core 的分工，以及 MMA 的基本语义。
2. 区分存储 dtype、输入计算 dtype、乘法精度和累加精度。
1. 理解 FP32、TF32、FP16、BF16、FP8 E4M3/E5M2、NVFP4 与 INT8/INT4 的精度和动态范围权衡。
2. 解释 Automatic Mixed Precision、Loss Scaling、Amax 与缩放因子的作用。
1. 识别“转换成低精度但没有提速”的形状、布局、算子和端到端原因。
2. 在 PyTorch 中正确设置 TF32、Autocast 与 GradScaler，并兼容新旧稳定版本 API。
1. 为 Ampere、Ada、Hopper、Blackwell 建立正确的能力矩阵与回退路径。
2. 同时验证速度、显存、误差、溢出和模型质量，不把峰值 FLOPS 当成业务结论。

## 前置知识

- 理解 GPU 的 SM、Warp、Register、Shared Memory 与 HBM/GDDR。
- 会计算 GEMM 的 FLOP 数：2MNK。
- 理解 Roofline、Latency、Throughput 和有效带宽。
- 会运行 Python；Level 1 需要可选 PyTorch，Level 2 需要对应 NVIDIA GPU。

## 核心直觉：Tensor Core 是“矩阵块流水线”

普通标量 FMA 可以理解为：

$$
d=a\times b+c
$$

Tensor Core 面向矩阵块执行 MMA（Matrix Multiply-Accumulate）：

$$
D=A\times B+C
$$

它不是“一条指令完成任意大小矩阵”。大 GEMM 会被库或 Kernel 切成层级 Tile：

```
完整 GEMM
└── Thread Block / Cluster Tile
    └── Warp / Warpgroup Tile
        └── Tensor Core MMA Tile
```

高吞吐来自以下共同条件：

- Tile 形状与硬件 MMA 形状匹配；
- A、B 数据按要求布局、对齐并及时搬到片上；
- 有足够工作填满 SM；
- 多级流水隐藏 Global/Shared Memory 延迟；
- 累加器和缩放不会成为额外瓶颈；
- 输出精度满足模型要求。

因此，“Tensor Core 更快”的完整表述应是：

## Tensor Core 的系统原理

### 矩阵乘法为什么适合专用单元

对 $A\in\mathbb{R}^{M\times K}$ 、 $B\in\mathbb{R}^{K\times N}$ ：

$$
C_{ij}=\sum_{k=1}^{K}A_{ik}B_{kj}
$$

总计算量近似为：

$$
FLOPs=2MNK
$$

当 Tile 中的 A、B 被重复使用时，GEMM 具有较高算术强度。Tensor Core 将固定形状的小矩阵乘累加映射到专用数据通路，Warp 或 Warpgroup 负责提交分片、持有累加器并协调搬运。

### Tensor Core 不负责所有算子

容易从 Tensor Core 获益的工作：

- GEMM、BMM、Linear；
- 卷积的部分实现；
- Attention 中的 QK 和 PV 矩阵乘；
- MoE Expert MLP 的 grouped GEMM；
- 某些结构化稀疏矩阵乘。

不一定直接由 Tensor Core 加速的工作：

如果一次训练 Step 中 GEMM 只占 50%，即使 GEMM 理想加速 4 倍，Amdahl 上界也只有：

$$
S=\frac{1}{(1-0.5)+0.5/4}=1.6
$$