---
title: "第25课：KV Cache 性能优化"
source: "https://everythingai.top/learn/ai-systems-performance-engineering-text/lesson-25"
author:
published:
created: 2026-09-26
description: "面向 AI 工程学习者的专业课程学习空间。"
tags:
  - "clippings"
---
**导航**

本课从显存占用、访问效率和分页管理出发优化 KV Cache，提高长上下文与高并发推理能力。

## 一、课程定位

自回归 LLM 每生成一个 Token，都需要关注之前的 Token。如果每一步都重新计算全部历史 Key 和 Value，生成第 t 个 Token 时会重复执行大量已经完成的工作。KV Cache 保存各层历史 Attention 的 Key/Value，让 Decode 只计算新 Token 的投影，再读取历史缓存完成 Attention。

它用显存换计算，却很快成为服务容量和 Decode 带宽的核心约束：

- 上下文越长，KV 线性增长；
- 并发越高，在途 Token 总数越多；
- Decode 每步都要读取历史 KV；
- 请求长度动态变化，连续预留会产生碎片；
- Prefix 复用、Offload 和解耦推理又把它变成跨请求、跨设备的数据资产。

本课从“为什么缓存”逐步走到容量模型、Paged KV、Prefix Cache、量化、Offload、驱逐、分布式传输和安全边界。

## 二、学习目标

- 能从 Attention 计算推导 KV Cache 的作用和容量公式。
- 区分 MHA、GQA、MQA、MLA 对 KV 容量和带宽的影响。
- 掌握连续预留、Paged KV、Block Table 和内部/外部碎片。
- 理解 Prefix Cache 的命中条件、收益、驱逐和多租户安全。
- 能评价 KV 低精度、滑动窗口、Offload、重算和跨节点传输。
- 建立 KV 容量、带宽、命中率和 Goodput 的统一模型。
- 完成 CPU 容量/碎片实验和通用 NVIDIA GPU 的追加写入实验。
- 能用真实流量诊断 OOM、Preemption、Cache Thrash 和 TPOT 上升。

## 三、前置知识

- Transformer Self-Attention、Causal Mask、MHA/GQA/MQA。
- LLM Prefill、Decode、Continuous Batching 和 Scheduler。
- GPU HBM、PCIe、NVLink、Pinned Memory 与显存分配器。
- TTFT、TPOT、ITL、Tokens/s 和 Goodput。

## 四、核心直觉：KV Cache 是“Attention 的历史索引”

单层 Self-Attention 计算：

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

$$
Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{d}}+Mask\right)V
$$

在 Decode 第 t 步，历史 Token 的 $K_1\ldots K_{t-1}$ 和 $V_1\ldots V_{t-1}$ 不会改变。只需计算新 Token 的 $Q_t$,$K_t$,$V_t$ ，把 $K_t$,$V_t$ 追加进缓存，再执行：

$$
O_t=softmax\left(\frac{Q_t[K_1,\ldots,K_t]^T}{\sqrt{d}}\right)[V_1,\ldots,V_t]
$$

因此 KV Cache 消除了历史 K/V 投影的重复计算，但没有消除对历史 KV 的读取。上下文越长，单步读取量越大；KV 优化既是容量问题，也是 HBM 带宽问题。

## 五、KV Cache 容量模型

### 5.1 每 Token 容量

对常见 Decoder-only Transformer：

$$
M_{token}=2\cdot L\cdot H_{kv}\cdot D_{head}\cdot B_{elem}
$$

- 2：Key 和 Value；

L：Transformer 层数；

$H_{kv}$ ：KV Head 数；

$D_{head}$ ：每个 Head 维度；

B\\\_{elem}：每元素字节数。

一个请求当前序列长度为 S：

$$
M_{request}=S\cdot M_{token}
$$

在线系统所有活动请求：

$$
M_{active}=M_{token}\cdot\sum_{i=1}^{N}S_i
$$

容量取决于所有在途 Token 总数，而不仅是最大并发数。

### 5.2 示例

假设 32 层、8 个 KV Head、Head Dim 128、FP16/BF16 KV：

$$
M_{token}=2\times32\times8\times128\times2=131072\ bytes=128\ KiB
$$

一个 4096 Token 请求约占 512 MiB KV；32 个同长度请求约占 16 GiB。这个例子没有包含权重、Activation、Workspace、CUDA Graph、通信 Buffer 和碎片。

### 5.3 MHA、GQA 与 MQA

| 结构 | KV Head 数 | KV 容量 | Decode 读取 | | ----------------------------- | -------------------------- | -- | -- | | MHA | 通常等于 Query Head | 最大 | 最大 | | GQA | 多个 Query Head 共享一组 KV | 较小 | 较小 | | MQA | 所有 Query Head 共享一个 KV Head | 最小 | 最小 |

GQA/MQA 是模型结构，不是部署时随意切换的 Runtime 参数。转换 Head 结构通常需要训练或专门权重变换及质量验证。

### 5.4 TP 下的 KV

张量并行可把 KV Head 分散到 Rank，每 Rank 容量常近似下降到 1/TP，但具体取决于 Head 数是否可整除、是否复制 KV Head、Attention Backend 和模型结构。不能直接用“总 KV/卡数”替代框架启动日志和实测。

## 六、连续预留为什么浪费显存

朴素 Runtime 可能为每个请求预留 `[max_seq_len]` 的连续 KV：

内部碎片率为：

$$
Waste_{contiguous}=1-\frac{\sum_i S_i}{N\cdot S_{max}}
$$

请求输出长度未知，按最大长度预留会浪费；按当前长度频繁扩容又会复制大 Tensor、产生外部碎片和地址变化。

## 七、Paged KV 与 Block Table

### 7.1 基本结构

Paged KV 把物理缓存拆成固定 Token 数的 Block。请求维护逻辑 Block 到物理 Block 的映射：

逻辑连续不要求物理连续。请求增长时只申请新 Block，完成或取消后归还 Block。

### 7.2 Paged 内部碎片

Block Size 为 P Token，请求长度 $S_i$ ：

$$
Allocated_i=\left\lceil\frac{S_i}{P}\right\rceil P
$$

$$
Waste_{paged}=1-\frac{\sum_i S_i}{\sum_i Allocated_i}
$$

每请求最多浪费 P-1 个尾部 Token。Block 小可减少尾部浪费，却增加 Block Table、哈希、调度和寻址开销；Block 大则相反。

### 7.3 PagedAttention 解决什么

PagedAttention 让 Attention Kernel 根据 Block Table 访问非连续物理 KV。内存管理和 Kernel 必须协同，否则仅有分页分配器而 Kernel 仍要求连续 Tensor，就无法避免拼接/复制。

Paged KV 的收益包括：

- 按需增长；
- 完成/取消后快速回收；
- 支持连续组批；
- 共享只读 Prefix Block；
- 更灵活的抢占、Offload 和跨设备传输。

它并非“零浪费”：仍有尾 Block、元数据、对齐、多个 KV Pool 和保留水位。

## 八、KV Cache 生命周期

### 8.1 分配

Scheduler 在执行前根据本轮新 Token 数预留 Block。预留失败应触发等待、拒绝或抢占，而不是让 Kernel 运行到 OOM。

### 8.2 写入

Prefill 批量写入 Prompt KV；Decode 每步追加一个或多个 Token。热路径应写入预分配槽位，避免每步 `torch.cat` 复制全部历史。

### 8.3 引用计数

共享 Prefix Block 可被多个请求引用。只有引用计数为 0 的 Block 才能成为可驱逐候选；修改共享 Block 需要 Copy-on-Write 或新的尾 Block。

### 8.4 释放

请求完成、失败或取消时必须释放私有 Block；网络断开要把取消传到 Scheduler/Executor/KV Manager。只丢弃 HTTP 响应会造成 KV 泄漏与无效 Decode。

### 8.5 驱逐

缓存空间满时常见策略：

- LRU：驱逐最久未访问且无引用的 Block；
- Priority + LRU：先按业务/重算价值分级，再做 LRU；
- Prefix-depth-aware：保留更常用的共享前缀；
- Cost-aware：比较重算、传输、命中概率和 Deadline。

驱逐的真实成本：

$$
Cost_{evict}=P_{reuse}\cdot(T_{reload}+T_{recompute}+T_{queue\ risk})
$$

## 九、Automatic Prefix Caching

### 9.1 核心思想

多个请求若拥有完全相同的 Token Prefix，可共享已计算 KV：

只需为共享部分计算一次，其余请求直接引用缓存 Block，可降低 TTFT、Prefill FLOPs 和重复显存。

### 9.2 命中必须以 Token 为准

看起来相同的文本可能因 Chat Template、空格、特殊 Token、Tokenizer Revision、图片占位符或 Adapter 不同而产生不同 Token IDs。安全的 Cache Key 至少应绑定：

- 完整前缀 Token 或链式 Block Hash；
- 模型与权重 Revision；
- Tokenizer 与 Chat Template；
- LoRA/Adapter、RoPE/位置相关配置；
- 多模态输入哈希；
- 租户/安全 Salt。

不能只用原始字符串或用户可控短 Hash。

### 9.3 Block 粒度命中

若 Block Size 为 16，两个请求共享 30 个 Token，通常只有前 16 个完整 Token Block 可直接命中；尾部 14 Token 可能需要重算或使用支持 Partial Reuse 的实现。

$$
HitTokens=\left\lfloor\frac{SharedPrefix}{P}\right\rfloor P
$$

### 9.4 Prefix Cache 不是所有请求都受益

适合：

- 长且重复的系统提示词；
- 多轮会话历史；
- Few-shot 模板；
- 同一文档上的多次问答；
- Beam/并行 Sampling 的共享 Prompt。

不适合：

- Prompt 几乎完全不同；
- Cache 工作集远大于容量并持续 Thrash；
- Prefix 很短，Hash/元数据成本接近重算；
- 多租户隔离不允许共享。

### 9.5 安全边界

Prefix 命中时间可能泄露其他租户是否使用过某个 Prompt；错误 Cache Key 还可能造成状态串用。多租户系统应使用安全哈希、模型/Adapter 隔离、租户 Salt 和访问控制，并评估时间侧信道。

## 十、低精度 KV Cache

将 KV 从 FP16/BF16 降到 8-bit，理论容量和读取字节约减半：

$$
M_{KV}^{8bit}\approx\frac{1}{2}M_{KV}^{16bit}
$$

但端到端收益取决于：

- Attention Kernel 是否原生支持该格式；
- Scale 粒度与额外元数据；
- 写入时量化、读取时反量化成本；
- HBM 带宽是否真是瓶颈；
- 长上下文任务的质量误差；
- GPU 架构和框架 Backend。

CLI 接受 `fp8` 不等于当前 GPU 一定走高效原生路径。必须检查启动日志、Kernel、硬件能力，并做模型质量和端到端 TPOT 对照。FP4/NVFP4 等更新格式只能在真实支持的 Blackwell 软件与硬件路径中验证，不能用 FP16 截断模拟。

## 十一、滑动窗口、稀疏保留与模型结构

若模型采用 Sliding Window Attention，只需保留最近 W 个 Token 的相关层 KV：

$$
M_{layer}\propto\min(S,W)
$$

部分模型混合 Full Attention 与 Local Attention，各层缓存形态不同，需要 Hybrid KV Manager。擅自丢弃全注意力层的旧 KV 会改变模型语义。

MQA/GQA、MLA、State Space/Mamba 与跨 Attention 都有不同缓存结构。现代 Runtime 可能创建多个 KV Pool；“统一每层同一 Block”不再适用于全部模型。

## 十二、Offload 与多级缓存

### 12.1 层级

Offload 扩大容量，不会让慢介质变成 HBM。传输时间近似：

$$
T_{move}=T_{fixed}+\frac{Bytes}{Bandwidth}
$$

只有满足以下关系才可能有净收益：

$$
P_{hit}\cdot T_{recompute}>T_{lookup}+T_{move}+T_{contention}
$$

### 12.2 消费级双卡边界

双 RTX 3080 20GB 常通过 PCIe，活动 Decode 每 Token 往返 Host KV 通常代价很高。可以学习 Pinned Memory、异步传输和冷会话恢复，但不能把 PCIe 实验当作 NVLink-C2C、NVSwitch 或 GPUDirect RDMA 的等价模拟。

### 12.3 分布式 KV 传输

Prefill/Decode 解耦需要把 KV 从 Prefill Worker 传到 Decode Worker。除了带宽，还要考虑：

- Block 格式与布局兼容；
- 目标端预留；
- 传输完成事件；
- 请求路由和 Cache Locality；
- 失败重试、重复传输和状态一致性；
- 跨租户加密与访问控制。

NCCL 擅长 Collective；NIXL/KV Connector 类接口更聚焦点到点或外部 KV 状态传输。二者不是互相替代关系。

## 十三、瓶颈分析方法

### 13.1 容量指标

- 总/空闲 KV Block；
- KV 使用率和 Watermark；
- 活动、可复用、Offloaded、Pinned Block 数；
- 每请求 Token、Block 和尾部浪费；
- Preemption、Recompute、Eviction 和 OOM；
- 分配/释放速率与泄漏趋势。

### 13.2 Cache 指标

- Prefix 查询次数；
- Token Hit Rate，而不只是请求 Hit Rate；
- 命中后节省的 Prefill Token 和 TTFT；
- LRU/优先级驱逐；
- Thrash：刚驱逐后很快重新加载；
- 不同租户/模板的工作集大小。

Token Hit Rate：

$$
HitRate_{token}=\frac{ReusedPrefixTokens}{EligiblePrefixTokens}
$$

一个请求只命中 16/4096 Token，按请求统计仍算“命中”，却几乎无收益。

### 13.3 性能症状

| 症状 | 优先检查 | 常见根因 |
| --- | --- | --- |
| 并发上升后 OOM | KV Block、Workspace、Graph Pool | 容量估算漏项、Watermark 太高 |
| TPOT 随上下文增长 | KV 读取、Attention Kernel、HBM | Decode 受内存带宽限制 |
| TTFT 波动大 | Prefix Hit、Eviction、Prefill Queue | Cache Thrash、模板不稳定 |
| GPU 有空间却申请失败 | 最大连续块、Allocator/Pool | 连续预留、外部碎片 |
| 取消后 KV 不释放 | 引用计数、Request 生命周期 | 取消未传播、引用泄漏 |
| Offload 后 P99 恶化 | H2D/D2H、Page Fault、PCIe | 热数据误驱逐、无预取/重叠 |
| Prefix 命中异常低 | Token IDs、Block 边界、Revision | Template/Tokenizer/Adapter 不一致 |
| Prefix 命中高但收益低 | 命中 Token 数、Prompt 长度 | 命中太短、Hash/调度成本占比高 |

### 13.4 诊断顺序

1. 用模型配置计算每 Token 理论 KV。
2. 从启动日志记录实际 Block Size、Block 数、Dtype 和 KV 容量。
1. 用真实长度分布计算有效 Token 与尾部碎片。
2. 区分活动 KV、Prefix Cache、Runtime Reserve 和 Allocator Reserved。
1. 对齐 TTFT/TPOT 尖峰、驱逐、抢占与传输事件。
2. 再用 Nsight 检查 Attention、Memcpy 和 CPU 空洞。

## 十四、实验分层

### 14.1 Level 0：CPU/通用容量实验

比较按最大长度连续预留与 Paged KV，计算 Block 碎片、Prefix 命中和容量上限。

### 14.2 Level 1：常见 NVIDIA GPU

比较逐 Token `torch.cat` 与预分配 KV Buffer 原位追加，观察时间和峰值显存。

### 14.3 Level 2：真实 Runtime

用 vLLM/TensorRT-LLM 的启动参数和指标验证 KV 容量、Prefix Cache、Dtype 与 Offload。专属格式和互联只在真实支持环境测试。

## 十五、Level 0 实战：KV 容量与分页模拟器

将以下代码保存为 `kv_cache_planner.py` 。

### 15.1 运行命令

### 15.2 默认预期现象

- 每 Token KV 为 128 KiB。
- 连续方案为所有请求预留 4096 Token，效率明显低于 Paged 方案。
- Paged 方案每请求只浪费不足一个 Block 的尾部空间。
- Block Size 增大时，尾部浪费通常上升。
- 共享 Prefix 只按完整 Block 计入命中；16 个请求复用 512 Token 时，可避免 15 份重复 Prefix KV。

随机长度固定由 `--seed` 控制。数字只描述该次合成工作集，不代表所有线上流量。

## 十六、Level 1 实战：朴素拼接与预分配追加

将代码保存为 `kv_append_bench.py` 。

### 16.1 安装与运行

双 GPU 分别运行单卡基线：

### 16.2 预期现象与分析

- 两条路径的 Checksum 应一致或仅有可解释的浮点归约误差。
- `torch.cat` 每步分配更大的 Tensor 并复制历史，累计拷贝量随 Step 近似二次增长。
- 预分配方案每步只写新位置，累计写入量近似线性。
- 随 `steps` 增大，朴素路径的时间和峰值临时分配通常恶化更快。
- Python 循环和每步 `copy_` 仍不是生产最优实现；真实 Runtime 会把多层、多请求写入融合进专用 Kernel，并使用 Block Pool。

该实验只验证生命周期和追加方式，不测 Attention 读取、Block Table、Prefix Cache 或真实模型质量。

## 十七、Level 2：真实 Runtime 调优

### 17.1 硬件支持矩阵

| 架构 | 示例 | 通用实验 | 专项可选 |
| --- | --- | --- | --- |
| Ampere | RTX 3080/3090、A100 | FP16/BF16 KV、Paged KV、Prefix Cache | A100 特定 NVLink 形态 |
| Ada Lovelace | RTX 4090、L40/L40S | Paged/Prefix Cache、Runtime 支持的低精度路径 | RTX 4090 无 NVLink |
| Hopper | H100/H200 | FP8 KV、Transformer Engine、NVLink/NVSwitch | 需质量与 Kernel 路径验证 |
| Blackwell | RTX 5090、B100/B200/GB200 | 更新低精度和更大 KV Pool | FP4/NVFP4 仅真实支持路径 |

RTX 4090 属于 Ada，RTX 5090 属于 Blackwell。数据中心互联能力不能从同代消费卡推断。

### 17.2 vLLM 启动检查

启动后保存：模型 Revision、KV Dtype、Block 数、Token Capacity、Prefix Cache 状态、最大模型长度和并行配置。参数名与默认值会变化，以当前版本 `--help` 和官方文档为准。

### 17.3 Prefix Cache A/B 测试

设计三组请求：

1. 相同长系统 Prompt，不同用户问题；
2. 完全不同 Prompt；
1. 文本看似相同，但 Chat Template 或 Adapter 不同。

分别在开启/关闭 Prefix Cache 时运行同一请求序列，记录：

- TTFT P50/P99；
- Reused Prefix Tokens；
- Prefix Token Hit Rate；
- Prefill Tokens/s；
- KV Block 使用与驱逐；
- 输出一致性。

如果只测一个请求，Cache 还没有被填充；必须先预热相同 Prefix，再测命中请求。

### 17.4 KV Dtype A/B 测试

只在框架、模型与硬件明确支持时测试：

比较容量、最大并发、TPOT、Attention Kernel、功耗和任务质量。不能只比较“能否启动”。

### 17.5 Offload 实验边界

Offload 参数和 Backend 变化较快，应先运行：

从冷会话恢复场景开始，记录 H2D/D2H 字节、传输时间、Onboard Hit、重算量和 P99。不要在双 RTX 3080 PCIe 环境中假定活动 Decode KV Offload 会有收益。

## 十八、优化前后对照

| 维度 | 朴素实现 | 性能工程化实现 |
| --- | --- | --- |
| 分配 | 每请求按最大长度连续预留 | 预分配 Block Pool、按需分页 |
| 追加 | 每步 `cat` 并复制历史 | 预留槽位原位写入 |
| 映射 | 逻辑与物理连续绑定 | Block Table 间接映射 |
| 共享 | 每请求复制系统 Prompt | Token/Revision/Salt 安全 Prefix Cache |
| 回收 | 请求结束后依赖 GC | 完成/取消显式释放与引用计数 |
| 驱逐 | 随机或只看空闲时间 | Priority/LRU/重算成本感知 |
| 精度 | KV 固定跟随权重 Dtype | 在支持路径做独立 KV Dtype A/B |
| 长上下文 | 全部 KV 永久驻留 | 模型允许时滑动窗口/分层保留 |
| Offload | OOM 后临时搬运 | 热温冷分层、预取、重叠和背压 |
| 观测 | 只看显存总量 | Block、Token Hit、抢占、传输和 Goodput |

## 十九、常见错误与排查

### 19.1 把权重显存当全部显存

模型能加载不代表可并发。保留 KV、Workspace、Graph、通信和安全水位；用启动日志校准理论公式。

### 19.2 每步 torch.cat

它反复分配和复制全部历史，长上下文下近似二次搬运。使用预分配、分页 Block 或框架 KV Manager。

### 19.3 Prefix Cache 命中却 TTFT 不变

检查命中 Token 数是否太少、最后不完整 Block 是否重算、Prompt 是否短、队列是否主导 TTFT，以及 Cache 查找/传输是否抵消收益。

### 19.4 Prefix Cache 跨租户共享

未使用 Salt、模型/Adapter 隔离和安全 Hash 可能引入信息泄漏。性能优化不能绕过数据边界。

### 19.5 Block Size 越小越好

小 Block 降低尾部浪费，却扩大表、哈希和调度开销，并可能降低 Kernel 效率。以真实分布扫描。

### 19.6 Offload 等于扩显存

容量确实扩大，但命中时需要传输。PCIe、CPU 内存带宽、Pinned Memory 和 NUMA 都可能成为 P99 瓶颈。

### 19.7 开启 FP8 就一定更快

低精度减少容量和字节，但量化/反量化、Scale、Backend 或硬件不匹配可能无收益。检查 Kernel 并验证质量。

### 19.8 nvidia-smi 显存不降就是泄漏

框架和 CUDA Allocator 会保留 Memory Pool。区分 Allocated、Reserved、活动 KV Block 和可复用 Cache；活动引用持续增长才是更强的泄漏证据。

### 19.9 多卡 KV 直接除以卡数

TP 的 KV Head 分片、复制、Padding 和 Backend 各异。读取每 Rank 实际容量并压测。

## 二十、性能优化闭环

1. 固定模型、Tokenizer、Chat Template、Adapter、精度和请求分布。
2. 计算理论每 Token KV，并与 Runtime 报告的 Token Capacity 对照。
1. 扫描 Block Size/容量水位，记录碎片和最大稳定并发。
2. 对 Prefix Cache 使用可重复与不可重复两组流量。
1. 对低精度同时验证质量、容量、TPOT 和 Kernel。
2. 对 Offload 区分冷恢复和活动 Decode，记录真实传输。
1. 在取消、超时、OOM 和 Worker 重启下注入故障，验证 Block 回收。
2. 最终用 TTFT/ITL P99、Goodput、成本和正确性决策。

## 二十一、面试题与答案

### 21.1 KV Cache 为什么能加速 Decode？

它保存历史 Token 在每层的 K/V，后续步骤无需重复计算历史投影，只计算新 Token 并读取缓存完成 Attention。

### 21.2 KV 容量由什么决定？

层数、KV Head 数、Head Dim、KV 元素字节数和所有活动序列 Token 总数；还要加 Block 碎片与 Runtime 预留。

### 21.3 GQA 为什么有利于推理？

多个 Query Head 共享更少的 KV Head，降低每 Token KV 容量和 Decode 读取带宽。

### 21.4 Paged KV 解决什么问题？

把动态增长的 KV 拆成按需分配的固定 Block，减少按最大长度连续预留和外部碎片，并支持共享、回收、抢占和连续组批。

### 21.5 Block 越小越好吗？

不是。小 Block 降低尾部浪费，但增加元数据、哈希、调度和寻址成本；要按真实长度和 Kernel 实测。

### 21.6 Prefix Cache 的 Key 为什么不能只用文本？

KV 由 Token IDs、模型权重、位置、Template、Adapter 和多模态输入共同决定；同样文本也可能产生不同状态。还需租户 Salt 防止跨租户复用。

### 21.7 为什么 Prefix Cache 主要改善 TTFT？

它跳过重复 Prefix 的 Prefill 计算；后续 Decode 仍需逐 Token 读取 KV 和执行模型。

### 21.8 KV 量化有什么风险？

反量化开销、Scale 元数据、Kernel 不支持和长上下文质量下降。必须做任务质量与端到端 A/B。

### 21.9 Offload 什么时候值得？

当被恢复 KV 的重算成本乘以命中概率，大于查找、传输和争用成本；更适合冷/温会话而非每步活动 KV。

### 21.10 如何判断 KV Cache 泄漏？

在请求完成/取消后，活动 Block、引用计数和在途 Token 不回落并随测试轮次持续增长；仅 CUDA Reserved 不下降不足以证明泄漏。

## 二十二、课后练习

1. 用 Level 0 比较 MHA 32 KV Head 与 GQA 8 KV Head 的容量。
2. 对同一长度分布扫描 Block Size 4/8/16/32/64，画碎片曲线。
1. 加入 80% 共享 1024 Token 系统 Prompt，计算 Prefix Cache 节省。
2. 修改 GPU 实验，把每步写入改成一次整段 `copy_` ，区分 Python Launch 与内存复制成本。
1. 为双 RTX 3080 20GB 设计 KV 水位、拒绝阈值和取消回收测试。
2. 设计多租户 Prefix Cache Key 与 Salt 方案，并列出侧信道风险。
1. 比较 `auto` 与受支持低精度 KV 的容量、TPOT 和任务质量。
2. 设计 GPU HBM→CPU DRAM→NVMe 的驱逐与预取策略，给出成本模型。

## 二十三、课程 Checklist

- 能推导每 Token KV 容量公式。
- 能解释 MHA、GQA、MQA 对 KV 的影响。
- 容量按在途 Token 总数而不是只按请求数规划。
- 能区分连续预留、内部碎片和外部碎片。
- 理解 Paged KV、Block Table 和引用计数。
- 请求完成、取消、失败时都能显式回收 Block。
- Prefix Cache Key 包含模型、Token、Template、Adapter 与安全 Salt。
- Prefix Cache 使用 Token Hit Rate 而非只看请求命中。
- 低精度 KV 同时验证 Kernel、容量、延迟和质量。
- Offload 记录真实传输带宽、延迟和命中率。
- 监控 KV 使用率、Watermark、驱逐、抢占和重算。
- 用真实长度分布扫描 Block Size 和并发。
- 自动记录 GPU、Compute Capability、驱动、CUDA 和框架版本。
- RTX 4090 标记为 Ada，RTX 5090 标记为 Blackwell。
- FP8/FP4、Transformer Engine、NVLink/NVSwitch 只在真实支持环境验证。
- 优化前后保持模型、Revision、请求集、Sampling 和到达过程一致。