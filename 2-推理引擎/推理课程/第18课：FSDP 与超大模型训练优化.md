---
title: "第18课：FSDP 与超大模型训练优化"
source: "https://everythingai.top/learn/ai-systems-performance-engineering-text/lesson-18"
author:
published:
created: 2026-09-26
description: "面向 AI 工程学习者的专业课程学习空间。"
tags:
  - "clippings"
---
**导航**

本课掌握 FSDP 的参数分片、通信与显存模型，构建可扩展的超大模型训练方案。

## 课程定位

DDP 解决的是“同一个模型如何用更多 GPU 更快训练”，FSDP 解决的是“单卡根本放不下的模型如何训练”。它不只是把权重切开，而是把参数、梯度和优化器状态都切到不同 rank，并在计算某一层前临时拼回所需参数。

这一课不把 FSDP 当成一个开关，而把它当成一套显存与通信调度系统。学完后，你应能回答三个工程问题：为什么分片后仍会 OOM、为什么 FSDP 可能比 DDP 慢，以及如何为自己的模型选择分片粒度、混合精度、激活重计算和 CPU Offload。

截至 2026-08-05，PyTorch 官方主线已经把基于 DTensor、逐参数分片的 `fully_shard` 称为 FSDP2；旧的 `FullyShardedDataParallel` 通常称为 FSDP1。本文的新项目优先采用 FSDP2，同时讲清 FSDP1 配置的对应关系。

## 学习目标

完成本课后，你将能够：

1. 从训练状态的组成推导 DDP、ZeRO 与 FSDP 的单卡显存。
2. 解释 FSDP 的 All-Gather、Reduce-Scatter、Reshard 生命周期。
1. 区分 FSDP1 的 FlatParameter 与 FSDP2 的 DTensor 逐参数分片。
2. 选择合理的 Transformer Block 分片边界。
1. 正确组合混合精度、激活重计算、梯度累积和 CPU Offload。
2. 用峰值显存、step time、tokens/s 和通信暴露时间评价优化，而不是只看 `nvidia-smi` 。
1. 在双 RTX 3080 或其他双卡环境中完成 DDP/FSDP2 对照实验。

## 前置知识

- 理解参数、梯度、优化器状态和激活值。
- 了解 DDP、All-Reduce、All-Gather、Reduce-Scatter。
- 能使用 `torchrun` 启动多进程训练。
- 建议先完成第 15～17 课。

## 一、核心直觉：分片保存，按需借用

假设有四张 GPU 和四层模型。DDP 的做法是每张 GPU 都放完整四层，只给每张卡不同数据：

```
GPU0: L0 L1 L2 L3 + 全部梯度 + 全部优化器状态
GPU1: L0 L1 L2 L3 + 全部梯度 + 全部优化器状态
GPU2: L0 L1 L2 L3 + 全部梯度 + 全部优化器状态
GPU3: L0 L1 L2 L3 + 全部梯度 + 全部优化器状态
```

FSDP 的静止状态更像仓库分区：

```
GPU0: 每个参数的第 0 片 + 对应梯度/优化器状态
GPU1: 每个参数的第 1 片 + 对应梯度/优化器状态
GPU2: 每个参数的第 2 片 + 对应梯度/优化器状态
GPU3: 每个参数的第 3 片 + 对应梯度/优化器状态
```

计算某个 Block 前，各 rank 通过 All-Gather 临时取得这个 Block 的完整参数；完成计算后释放完整参数，只保留自己的分片。反向得到完整梯度后，再通过 Reduce-Scatter 完成求和并只留下本 rank 的梯度片。

一句话概括：

## 二、训练显存到底花在哪里

设模型有 $P$ 个参数，每个参数的常驻参数、梯度和优化器状态分别占 $(b_p)$ 、 $(b_g)$ 、 $(b_o)$ 字节。暂不计激活和临时缓冲区：

$$
M_{state}=P(b_p+b_g+b_o)
$$

一个常见的保守规划是：FP32 主参数 4 字节、FP32 梯度 4 字节、Adam 一阶与二阶状态共 8 字节，因此：

$$
M_{state}\approx 16P\ \text{bytes}
$$

这不是所有训练栈的固定常数。低精度参数、低精度梯度、额外 master weight、8-bit optimizer 都会改变它。工程上应把字节数做成输入，而不是背“每参数 16 字节”。

### 2.1 DDP

每个 rank 都保留全部模型状态：

$$
M_{DDP}\approx M_{state}+M_{act}+M_{temp}
$$

增加 GPU 数量不会降低单卡模型状态显存。

### 2.2 ZeRO 与 FSDP 的理想常驻状态

若数据并行规模为 $N$ ：

| 方案 | 参数 | 梯度 | 优化器状态 | | -------------------- | -- | -- | -- | | DDP / ZeRO-0 | 复制 | 复制 | 复制 | | ZeRO-1 | 复制 | 复制 | 分片 | | ZeRO-2 | 复制 | 分片 | 分片 | | ZeRO-3 / FULL\\\_SHARD | 分片 | 分片 | 分片 |

完全分片的理想常驻模型状态为：

$$
M_{sharded}\approx \frac{M_{state}}{N}
$$

但真实峰值还要加：

$$
M_{peak}\approx \frac{M_{state}}{N}+M_{active\ group}+M_{prefetch}+M_{act}+M_{comm}+M_{fragment}
$$

其中 `active group` 是当前临时 All-Gather 出来的完整参数组， `prefetch` 是为了隐藏通信而提前拉取的下一组参数。于是出现一个重要结论：

## 三、一次 FSDP2 迭代发生了什么

以一个 Transformer Block 为一个分片组， `reshard_after_forward=True` 时：

```
前向：参数分片 --All-Gather--> 完整 Block 参数 --计算--> 释放完整参数
反向：参数分片 --All-Gather--> 完整 Block 参数 --计算梯度
      完整梯度 --Reduce-Scatter--> 本 rank 梯度分片 --释放完整参数
优化：本 rank 只更新自己的参数分片和优化器状态分片
```

### 3.1 为什么要两次 All-Gather

前向结束后立刻释放完整参数，能获得最低峰值显存，但反向计算该层时必须重新 All-Gather。若不在前向后 Reshard，就能少一次参数 All-Gather，却要更久地保留完整参数。

这是一条典型的空间—通信交换：

| 选择 | 峰值显存 | 参数通信 | 适合场景 | | ------------------ | - | -- | ------------ | | 前向后立即 Reshard | 低 | 较高 | 模型逼近显存上限 | | 前向后保留完整参数 | 高 | 较低 | 模型较小、通信成为主瓶颈 |

### 3.2 通信量近似

Ring All-Gather 或 Reduce-Scatter 对一个总大小为 $S$ 的张量，每 rank 传输量可粗略写成：

$$
V_{AG}\approx V_{RS}\approx \frac{N-1}{N}S
$$

若每步对全部参数执行两次 All-Gather 和一次 Reduce-Scatter：

$$
V_{FSDP}\approx 3\frac{N-1}{N}P b_{comm}
$$

DDP 对梯度执行一次 Ring All-Reduce，可视为 Reduce-Scatter 加 All-Gather：

$$
V_{DDP}\approx 2\frac{N-1}{N}P b_g
$$

这解释了为什么“模型明明能放进单卡”时，FSDP 未必比 DDP 快。FSDP 的首要价值是容量；吞吐需要靠合理分组、低精度通信和计算通信重叠争取。

### 3.3 暴露通信时间

设一个 step 的计算时间为 $(T_c)$ ，通信总时间为 $(T_m)$ ，其中成功隐藏在计算后的部分为 $(T_h)$ ：

$$
T_{step}=T_c+T_m-T_h+T_{other}
$$

优化目标不是让 $(T_m)$ 消失，而是让 $(T_h)$ 尽量接近 $(T_m)$ 。Nsight Systems 时间线上应看到下一层 All-Gather 与当前层计算重叠、当前层 Reduce-Scatter 与前一层反向计算重叠。

## 四、FSDP1 与 FSDP2

### 4.1 FSDP1

FSDP1 用 `FullyShardedDataParallel` 包装 Module，通常把一组参数展平为 FlatParameter 再切片。常见配置包括：

- `ShardingStrategy.FULL_SHARD`
- `MixedPrecision`
- `auto_wrap_policy`
- `BackwardPrefetch`
- `CPUOffload`
- `limit_all_gathers`

它仍大量存在于生产代码中，阅读旧项目必须会，但新项目不应无理由继续绑定旧接口。

### 4.2 FSDP2

FSDP2 使用 `fully_shard(module)` 原地改变 Module，并把参数表示为沿第 0 维切分的 DTensor。它不再依赖 FlatParameter，参数全名保持稳定，更容易与其他 DTensor 并行方式组合。

最重要的接口变化不是名字，而是“通信分组显式化”：

```
# 正确方向：从叶子层到底层根模块，bottom-up
for block in model.blocks:
    fully_shard(block, mesh=mesh)
fully_shard(model, mesh=mesh)
```

每次 `fully_shard` 对应一个通信组。若只对根模型调用一次，所有参数可能形成一个巨大组：前向开始前做一次大 All-Gather，反向结束后做一次大 Reduce-Scatter，几乎没有层间重叠空间。

## 五、分片粒度：最关键的性能旋钮

设每个分片组参数量为 $G$ ，通信启动延迟为 (alpha)，有效带宽为 $B$ ，则每次 Collective 可近似为：

$$
T_{group}\approx \alpha + \frac{G}{B}
$$

组太小：Collective 数量多，启动延迟和调度开销累积；组太大：峰值显存高，通信晚启动，难以与逐层计算重叠。

Transformer 的第一选择通常是“一层 Block 一个组”，然后依据 Profile 调整：

- Block 很小时，可把相邻 Block 合并。
- 单个 Block 很大时，可继续按 Attention / MLP 分组，但要防止 Collective 过碎。
- Embedding 与 LM Head 是否权重共享，需要验证分片与 checkpoint 行为。
- 只包根模块通常只适合教学或极小模型，不是高性能默认方案。

## 六、超大模型的组合优化

### 6.1 混合精度

FSDP2 的 `MixedPrecisionPolicy` 可以分别控制临时完整参数和梯度归约的 dtype。常见策略是：

- 常驻分片参数与优化器状态保持 FP32。
- All-Gather 后的计算参数用 BF16。
- Reduce-Scatter 使用 BF16 以减小通信，或 FP32 以优先稳定性。

Ampere（RTX 3080/3090、A100）、Ada（RTX 4090、L40/L40S）、Hopper（H100/H200）和 Blackwell（RTX 5090、B100/B200/GB200）都可运行 BF16 路径，但最终仍应通过 `torch.cuda.is_bf16_supported()` 检测。FP8/FP4 不是本课通用基线，且不能用双 3080 做等价验证。

### 6.2 激活重计算

FSDP 主要减少模型状态，激活显存仍与 batch、序列长度、隐藏维度和层数相关。对长序列模型，激活可能重新成为第一瓶颈。

Activation Checkpointing 只保存部分边界激活，反向时重跑前向：

$$
M_{act}\downarrow,\qquad FLOPs\uparrow
$$

判断它是否值得，不要只看峰值显存下降，还要看降低显存后能否增加 micro-batch、序列长度或减少梯度累积，从而提高最终 Goodput。

### 6.3 梯度累积

全局 batch 为：

$$
B_{global}=B_{micro}\times N_{DP}\times K_{accum}
$$

前 (K-1) 个 micro-batch 不应进行完整梯度同步，最后一个再同步。否则只是把通信重复了 $K$ 次。FSDP2 可通过梯度同步控制接口实现；接口在不同 PyTorch 小版本可能变化，生产代码要对照所用版本文档。

### 6.4 CPU Offload

CPU Offload 把参数分片、梯度和优化器状态移到主机内存。它能救 OOM，但 PCIe 传输和 CPU optimizer step 可能显著降低吞吐。

使用前检查：

1. 每个进程的 CPU 内存总量是否足够。
2. 多进程锁页内存是否把主机内存耗尽。
1. GPU 是否在等待 H2D/D2H。
2. NUMA 绑定是否让 GPU 访问远端 CPU 内存。

双 RTX 3080 的优先顺序通常是：BF16 → FSDP2 → 激活重计算 → 调小 micro-batch → 最后才考虑 CPU Offload。

### 6.5 Hybrid Sharding

多节点时，可在节点内做 FULL\\\_SHARD、节点间复制同一份 shard。这样高频 All-Gather/Reduce-Scatter 留在 NVLink/NVSwitch 或节点内 PCIe 域，跨节点只承担副本间归约。

双卡单机不需要 Hybrid Sharding。它适合“节点内互联快、节点间网络相对慢”的集群，而不是 GPU 越多就一定启用。

### 6.6 分布式 Checkpoint

不要在每个 rank 上先聚合完整模型再 `torch.save` 。大模型可能在保存时造成 GPU OOM、CPU OOM 或 rank 0 长时间停顿。FSDP2 的 sharded state dict 以 DTensor 表示，优先使用 PyTorch Distributed Checkpoint 保存分片状态，并在恢复时支持改变 world size 的重分片。

Checkpoint 需要单独压测：记录保存时长、恢复时长、峰值 CPU 内存、存储带宽和训练暂停时间。能成功保存不等于能在故障窗口内恢复。

## 七、瓶颈分析方法

### 7.1 先建立四组指标

| 维度 | 指标 | 典型问题 |
| --- | --- | --- |
| 容量 | peak allocated/reserved、OOM 点 | 分片组或激活过大 |
| 速度 | step time、tokens/s、samples/s | 通信或重计算过重 |
| 通信 | AG/RS 时长、暴露尾部、带宽 | 拓扑差、粒度不当 |
| 正确性 | loss、梯度范数、checkpoint 恢复 | dtype、累积或状态保存错误 |

### 7.2 诊断决策树

```
是否 OOM？
├─ 是
│  ├─ 模型初始化就 OOM：meta 初始化 / 分片前避免整模落卡
│  ├─ forward 峰值 OOM：减小分片组、开启 reshard、减 batch
│  ├─ backward 峰值 OOM：激活重计算、减少预取、查梯度累积
│  └─ optimizer.step OOM：确认优化器状态已分片，查临时 foreach buffer
└─ 否
   ├─ 比 DDP 慢：模型是否本就适合 DDP？检查 AG/RS 暴露时间
   ├─ GPU 有空洞：分片过大、CPU 发射晚、网络慢或 straggler
   ├─ reserved 远大于 allocated：碎片/预取/动态 shape
   └─ loss 异常：检查 reduction dtype、梯度缩放和累积除数
```

### 7.3 最小实验法

每次只改一个变量，并固定随机种子、模型、输入 shape、warmup 和测量步数：

1. DDP 基线。
2. FSDP2，只改并行策略。
1. FSDP2 + 按 Block 分组。
2. 再分别测试 BF16、Checkpoint、梯度累积、Offload。
1. 每次同时保存吞吐、峰值显存和 loss。

## 八、三级实验

## Level 0：无 GPU 显存规划器

这个实验不模拟真实 Collective，也不能替代 GPU 测量；它用于在申请机器前排除明显不可能的配置。

保存为 `fsdp_memory_planner.py` ：

```
#!/usr/bin/env python3
import argparse

GB = 1024 ** 3

def gib(x):
    return x / GB

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--params-b", type=float, default=7.0,
                   help="参数量，单位十亿")
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument("--param-bytes", type=float, default=4)
    p.add_argument("--grad-bytes", type=float, default=4)
    p.add_argument("--optim-bytes", type=float, default=8)
    p.add_argument("--comm-param-bytes", type=float, default=2)
    p.add_argument("--largest-group-mb", type=float, default=512)
    p.add_argument("--activation-gb", type=float, default=4)
    p.add_argument("--prefetch-groups", type=int, default=1)
    p.add_argument("--fragment-ratio", type=float, default=0.10)
    args = p.parse_args()

    if args.world_size < 1:
        raise SystemExit("world-size 必须 >= 1")

    params = args.params_b * 1e9
    state = params * (args.param_bytes + args.grad_bytes + args.optim_bytes)
    ddp = state + args.activation_gb * GB

    group = args.largest_group_mb * 1024 ** 2
    # 当前组 + 提前拉取组；真实框架还会有通信缓冲和 allocator 碎片。
    fsdp_raw = (
        state / args.world_size
        + group * args.comm_param_bytes / args.param_bytes
          * (1 + args.prefetch_groups)
        + args.activation_gb * GB
    )
    fsdp = fsdp_raw * (1 + args.fragment_ratio)

    n = args.world_size
    ring = (n - 1) / n if n > 1 else 0
    ddp_comm = 2 * ring * params * args.grad_bytes
    fsdp_comm = 3 * ring * params * args.comm_param_bytes

    print(f"参数量: {args.params_b:.2f}B, world_size={n}")
    print(f"完整模型状态: {gib(state):.2f} GiB")
    print(f"DDP 单卡粗估: {gib(ddp):.2f} GiB")
    print(f"FSDP 单卡峰值粗估: {gib(fsdp):.2f} GiB")
    print(f"DDP 每 rank/step 通信粗估: {gib(ddp_comm):.2f} GiB")
    print(f"FSDP 每 rank/step 通信粗估: {gib(fsdp_comm):.2f} GiB")
    print("边界：不含 CUDA context、临时算子、真实预取生命周期和 checkpoint I/O。")

if __name__ == "__main__":
    main()
```

运行：

```
python3 fsdp_memory_planner.py --params-b 7 --world-size 2 \
  --largest-group-mb 512 --activation-gb 4

python3 fsdp_memory_planner.py --params-b 7 --world-size 8 \
  --largest-group-mb 256 --activation-gb 4
```

预期现象：world size 增大时模型状态分片部分近似按 (1/N) 降低，但激活和完整参数组不随 $N$ 同比例下降；FSDP 通信粗估可能高于 DDP。

## Level 1：双卡 DDP 与 FSDP2 对照

### 8.1 环境准备

使用隔离环境；PyTorch 与 CUDA 安装方式会变化，GPU 用户优先从 PyTorch 官方安装选择器取得与驱动兼容的命令：

```
python3 -m venv .venv-fsdp
source .venv-fsdp/bin/activate
python -m pip install --upgrade pip
python -m pip install "torch>=2.8,<2.14"

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(i, p.name, "CC", f"{p.major}.{p.minor}",
              "VRAM GiB", round(p.total_memory / 1024**3, 2))
PY

nvidia-smi topo -m
```

### 8.2 完整训练基准

保存为 `fsdp2_benchmark.py` ：

```
#!/usr/bin/env python3
import argparse
import contextlib
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.nn.parallel import DistributedDataParallel as DDP

class Block(nn.Module):
    def __init__(self, hidden, expansion=4):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, hidden * expansion, bias=False)
        self.fc2 = nn.Linear(hidden * expansion, hidden, bias=False)

    def forward(self, x):
        h = self.norm(x)
        h = torch.nn.functional.gelu(self.fc1(h), approximate="tanh")
        return x + self.fc2(h)

class TinyTransformer(nn.Module):
    def __init__(self, hidden, layers):
        super().__init__()
        self.blocks = nn.ModuleList([Block(hidden) for _ in range(layers)])
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["ddp", "fsdp2"], required=True)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seq", type=int, default=256)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    return p.parse_args()

def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Level 1 需要 NVIDIA GPU；无 GPU 请运行 Level 0。")

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)

    if world < 2:
        raise SystemExit("请用 torchrun 启动至少 2 个进程。")

    model = TinyTransformer(args.hidden, args.layers).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    use_bf16 = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    if args.mode == "ddp":
        model = DDP(model, device_ids=[local_rank])
    else:
        try:
            from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
        except ImportError as exc:
            raise SystemExit("当前 PyTorch 没有 FSDP2，请安装受支持版本。") from exc

        mesh = init_device_mesh("cuda", (world,))
        mp = MixedPrecisionPolicy(
            param_dtype=compute_dtype,
            reduce_dtype=compute_dtype,
        )
        # 关键：Block 先分片，根模型最后分片。
        for block in model.blocks:
            fully_shard(block, mesh=mesh, mp_policy=mp,
                        reshard_after_forward=True)
        fully_shard(model, mesh=mesh, mp_policy=mp)

    # 必须在 FSDP2 处理参数后创建 optimizer。
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def one_step(measure=False):
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        for micro in range(args.accum):
            sync = micro == args.accum - 1
            if args.mode == "fsdp2":
                model.set_requires_gradient_sync(sync)
                sync_ctx = contextlib.nullcontext()
            else:
                sync_ctx = contextlib.nullcontext() if sync else model.no_sync()

            x = torch.randn(args.batch, args.seq, args.hidden,
                            device=device, dtype=compute_dtype)
            target = torch.zeros_like(x)
            with sync_ctx:
                out = model(x)
                loss = (out.float() - target.float()).square().mean()
                loss = loss / args.accum
                loss.backward()
            loss_value += loss.detach().float().item()
        optimizer.step()
        return loss_value

    for _ in range(args.warmup):
        one_step()
    dist.barrier()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    last_loss = 0.0
    for _ in range(args.steps):
        last_loss = one_step(measure=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
    stat = torch.tensor([elapsed, peak_alloc, peak_reserved], device=device)
    dist.all_reduce(stat, op=dist.ReduceOp.MAX)

    if rank == 0:
        global_samples = args.batch * world * args.accum * args.steps
        tokens = global_samples * args.seq
        print({
            "mode": args.mode,
            "world_size": world,
            "params_m": round(total_params / 1e6, 2),
            "dtype": str(compute_dtype),
            "step_ms_max_rank": round(stat[0].item() * 1000 / args.steps, 3),
            "tokens_per_s": round(tokens / stat[0].item(), 1),
            "peak_alloc_gib_max_rank": round(stat[1].item(), 3),
            "peak_reserved_gib_max_rank": round(stat[2].item(), 3),
            "last_loss": round(last_loss, 6),
        })
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

### 8.3 运行命令

先用小模型确认正确性：

```
torchrun --standalone --nproc-per-node=2 fsdp2_benchmark.py \
  --mode ddp --hidden 1024 --layers 8 --batch 2 --seq 256

torchrun --standalone --nproc-per-node=2 fsdp2_benchmark.py \
  --mode fsdp2 --hidden 1024 --layers 8 --batch 2 --seq 256
```

再逐步放大，直到 DDP 接近 OOM；不要第一次就使用最大配置：

```
torchrun --standalone --nproc-per-node=2 fsdp2_benchmark.py \
  --mode ddp --hidden 2048 --layers 12 --batch 2 --seq 512 --steps 20

torchrun --standalone --nproc-per-node=2 fsdp2_benchmark.py \
  --mode fsdp2 --hidden 2048 --layers 12 --batch 2 --seq 512 --steps 20
```

测试正确的梯度累积通信抑制：

```
torchrun --standalone --nproc-per-node=2 fsdp2_benchmark.py \
  --mode fsdp2 --hidden 1024 --layers 8 --batch 1 --seq 256 \
  --accum 4 --steps 20
```

### 8.4 预期现象

- 小模型上 DDP 可能更快，因为 FSDP 的参数通信与调度成本无法被足够计算隐藏。
- 模型放大后，FSDP2 的模型状态分片优势才会明显。
- `peak_reserved` 可能显著高于 `peak_allocated` ，但不能仅凭差值断言“内存泄漏”。
- 双 RTX 3080 通常经 PCIe 通信，FSDP 的 All-Gather/Reduce-Scatter 可能更暴露；实际结果取决于主板拓扑、PCIe 链路和是否跨 CPU 根复合体。
- 所有性能数字只对本次机器、软件版本和命令有效。

## Level 2：架构与集群专项实验

### 8.5 支持矩阵

| 平台 | 可执行内容 | 需要注意 |
| --- | --- | --- |
| Ampere RTX 3080/3090 | FSDP2、BF16/FP16、NCCL、双卡对照 | 3080 无 NVLink；3090 仅特定双卡桥接场景有 NVLink |
| Ampere A100 | 同上，可做 NVLink/NVSwitch 集群实验 | SXM 与 PCIe 型号拓扑不同 |
| Ada RTX 4090 | FSDP2、BF16/FP16、PCIe 对照 | 4090 是 Ada，不是 Blackwell，且无 NVLink |
| Ada L40/L40S | 数据中心 PCIe 多卡实验 | 先检查实际服务器拓扑 |
| Hopper H100/H200 | FSDP2、HSDP、NVLink/NVSwitch、FP8 可选 | FP8 需要配套软件栈与数值验证 |
| Blackwell RTX 5090 | FSDP2、BF16/FP16、PCIe 对照 | 消费卡不能代表 B200/GB200 Fabric |
| Blackwell B100/B200/GB200 | FSDP2、HSDP、NVLink/NVSwitch、低精度专项 | 结果不可外推到消费卡 |

### 8.6 Profile 通信时间线

Nsight Systems 的参数会随版本变化，先执行 `nsys profile --help` 核对本机选项：

```
nsys profile --trace=cuda,nvtx,nccl,osrt \
  --output=fsdp2_trace --force-overwrite=true \
  torchrun --standalone --nproc-per-node=2 fsdp2_benchmark.py \
  --mode fsdp2 --hidden 2048 --layers 12 --batch 2 --seq 512 \
  --warmup 2 --steps 5
```

观察：

1. All-Gather 是否在某层计算结束后才开始。
2. Reduce-Scatter 是否形成长尾。
1. Collective 是否被两个 rank 同步等待。
2. NCCL kernel 与 GEMM 是否有重叠，还是被同一资源争用串行化。
1. 某一 rank 是否持续比其他 rank 晚到 Collective。

### 8.7 HSDP 的不可等价边界

单机双卡可以理解 HSDP 模型，却不能等价模拟多节点网络层次。真正的 HSDP 实验至少需要两个节点，并明确节点内 shard mesh 与节点间 replicate mesh。没有多节点环境时，只完成内存模型与 DeviceMesh 设计题，不伪造带宽或扩展效率。

## 九、结果分析模板

建议用以下表格记录实测结果：

| 模式 | 参数量 | micro-batch | 累积 | step ms | tokens/s | peak alloc | peak reserved | loss | | --------------------------------------------------------------------- | -- | -- | -- | -- | -- | -- | -- | -- | | DDP | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 | | FSDP2 root-only | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 | | FSDP2 per-block | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 |

分析顺序：

1. 先确认 loss 与输出统计一致，性能错误经常来自错误的累积或同步。
2. 比较峰值显存，而不是训练前的静态 `nvidia-smi` 。
1. 比较最慢 rank 的 step time，而不是 rank 0 单独耗时。
2. 计算通过显存节省换来的能力：更大模型、更长序列还是更大 batch。
1. 只有在工作负载相同、结果正确、测量稳定时才报告速度变化。

## 十、优化前后对照

| 问题 | 不理想做法 | 优化做法 | 代价 |
| --- | --- | --- | --- |
| 单卡模型状态 OOM | DDP 复制全部状态 | FSDP2 完全分片 | 参数通信增加 |
| 只包根模块 | 一个超大通信组 | 按 Transformer Block bottom-up 分片 | Collective 数量增加 |
| 激活 OOM | 误以为 FSDP 会解决一切 | 加 Activation Checkpointing | 重计算增加 |
| 累积仍频繁通信 | 每个 micro-batch 都同步 | 非最后 micro-batch 禁止梯度同步 | 错配会导致梯度错误 |
| CPU Offload 很慢 | 把它当免费显存 | 只在容量不足时使用并测 PCIe/NUMA | 吞吐下降 |
| 保存 checkpoint OOM | rank 0 聚合完整状态 | 分布式分片保存 | 运维复杂度增加 |
| 小模型吞吐下降 | 强行使用 FSDP | 能放下时优先 DDP 基线 | 容量不再扩展 |
| 多节点网络慢 | 全局 FULL\_SHARD | 评估 HSDP 分层通信 | 需要二维 mesh |

## 十一、常见错误与排查

### 11.1 Default process group has not been initialized

必须通过 `torchrun` 启动，并在创建 DeviceMesh/FSDP 前执行 `dist.init_process_group()` 。

### 11.2 所有进程都使用 GPU 0

读取 `LOCAL_RANK` ，并在初始化模型前调用：

```
torch.cuda.set_device(local_rank)
```

同时检查 `CUDA_VISIBLE_DEVICES` 是否映射正确。

### 11.3 FSDP2 仍然 OOM

依次检查：

1. 是否先把超大完整模型放到每张 GPU，再开始分片。
2. 是否只对根模型调用 `fully_shard` 。
1. 最大 Block 是否本身就超过可用显存。
2. 激活是否比模型状态更大。
1. 预取是否同时保留多个完整参数组。
2. 梯度累积期间是否保留不必要的完整参数。

### 11.4 创建 optimizer 的顺序错误

FSDP2 会把参数转为 DTensor。应在 `fully_shard` 之后用 `model.parameters()` 创建 optimizer。

### 11.5 调用 model.forward(x) 后报错或参数未 All-Gather

直接调用 `model(x)` ，让 Module hook 生效。自定义非 `forward` 入口要按当前版本文档显式注册。

### 11.6 梯度累积结果不一致

- 只在最后一个 micro-batch 同步。
- loss 除以累积步数。
- 每个 optimizer step 开头清空梯度，而不是每个 micro-batch 清空。
- 所有 rank 的 micro-batch 数必须一致。

### 11.7 NCCL 卡死

- 检查所有 rank 是否以相同顺序进入 Collective。
- 确认没有某个 rank 因数据耗尽、异常或 OOM 提前退出。
- 先用固定 shape 的合成数据复现。
- 开启 `TORCH_DISTRIBUTED_DEBUG=DETAIL` 做正确性排查；诊断环境变量不应永久留在性能基线中。

### 11.8 FSDP 比 DDP 慢很多

可能不是 Bug：

- 模型太小，通信启动开销占比高。
- GPU 间只有 PCIe。
- 分片组过小或过大。
- 激活重计算重复计算过多。
- CPU Offload 让 GPU 等待主机。
- Profile 中通信没有与计算重叠。

### 11.9 Checkpoint 无法跨 world size 恢复

确认保存的是可重分片的分布式状态，模型参数 FQN 没有变化，optimizer state 也经过一致的 state-dict 转换。保存后必须做一次真实恢复演练。

## 十二、面试题与答案

### 题1：FSDP 与 DDP 的根本区别是什么？

DDP 在每个 rank 复制完整参数、梯度和优化器状态，主要同步梯度；FSDP 常驻时分片这些模型状态，计算前 All-Gather 参数，反向后 Reduce-Scatter 梯度，以通信换显存。

### 题2：为什么 FSDP2 要 bottom-up 调用 fully\_shard？

每次调用定义一个通信组。先处理子模块，再处理根模块，能形成逐层参数组，从而控制峰值显存并让下一层 All-Gather 与当前层计算重叠。

### 题3：FULL\\\_SHARD 与 ZeRO-3 有什么关系？

二者核心都分片参数、梯度和优化器状态，生命周期也都围绕按需聚合参数。它们属于不同实现与生态，API、分组、状态保存和调度细节不能直接画等号。

### 题4：FSDP 为什么可能比 DDP 慢？

FSDP 通常多了参数 All-Gather，并有分片、Reshard 和 hook 调度开销。如果模型本来能轻松放进单卡，或网络慢、分片粒度不当、通信无法重叠，容量收益不会自动转化为吞吐收益。

### 题5：FSDP 后仍然 OOM，最可能是什么？

激活过大、分片组过大、预取保留多个完整组、初始化阶段整模先落卡、通信临时缓冲或 allocator 碎片。不能只按 `16P/N` 判断。

### 题6：reshard\_after\_forward=True 的代价是什么？

前向后释放完整参数降低峰值显存，但反向前需要再次 All-Gather。关闭 Reshard 可减少通信，却提高参数驻留显存。

### 题7：FSDP 能降低激活显存吗？

普通 FSDP 的主要对象是模型状态，不能自动按数据并行 rank 分片全部激活。激活问题通常需要 checkpointing、减小 micro-batch、序列/上下文并行或更省内存的算子。

### 题8：如何评价一次 FSDP 优化？

至少同时比较正确性、峰值显存、最慢 rank 的 step time、tokens/s、通信暴露时间和 checkpoint 能力。只看 GPU 利用率或单次 step 不足以证明优化有效。

### 题9：什么时候选择 HSDP？

多节点且节点内通信显著快于节点间通信时，可在节点内分片、节点间复制，减少跨节点高频参数聚合。单机双卡通常没有必要。

### 题10：为什么 optimizer 必须在 FSDP2 后创建？

`fully_shard` 会把参数变成 DTensor 分片表示；optimizer 应绑定处理后的参数，否则可能持有错误参数引用或未分片状态。

## 十三、课后练习

1. 用 Level 0 规划器估算 1B、7B、13B 模型在 2/4/8 卡上的状态显存，并写出假设。
2. 在双卡上比较 DDP 与 FSDP2 的峰值显存和 tokens/s，至少重复三次。
1. 把 Level 1 的分片从 per-block 改成 root-only，解释时间线差异。
2. 把 Block 两两合并为一个通信组，寻找吞吐与峰值显存的折中点。
1. 实现 Activation Checkpointing，对比显存、step time 和可提升的 batch。
2. 使用梯度累积 1/2/4/8，验证非最后 micro-batch 是否仍发生 Reduce-Scatter。
1. 为训练脚本加入 Distributed Checkpoint，完成保存—退出—恢复—继续训练闭环。
2. 设计一个 2 节点 × 8 GPU 的 HSDP DeviceMesh，并标出节点内与节点间 Collective。

## 十四、Checklist

### 正确性

- DDP 与 FSDP2 使用相同模型、输入 shape、随机种子和 loss 定义。
- optimizer 在 fully\\\_shard 后创建。
- 梯度累积只在最后一个 micro-batch 同步。
- loss、梯度范数和 checkpoint 恢复均通过验证。

### 显存

- 同时记录 peak allocated 与 peak reserved。
- 区分模型状态、激活、通信缓冲和碎片。
- 检查最大分片组和预取深度。
- 超大模型避免初始化阶段整模落到每张 GPU。

### 性能

- 有 DDP 或单卡基线。
- 计时前完成 warmup 和同步。
- 使用最慢 rank 的耗时。
- 检查 All-Gather/Reduce-Scatter 与计算重叠。
- 每次只改一个优化变量。

### 可移植性

- 自动检测 GPU、Compute Capability、BF16 支持和显存。
- 不硬编码 RTX 3080、H100 或 B200 的性能结论。
- 4090 标记为 Ada，5090 标记为 Blackwell。
- NVLink/NVSwitch、HSDP、FP8 等专项实验声明硬件边界。

### 交付

- 记录 PyTorch、CUDA、驱动、NCCL 和 GPU 拓扑。
- 保存命令、配置、原始测量和 profiler 文件。
- 性能结论仅限定于本次环境。
- Checkpoint 已做真实恢复演练。