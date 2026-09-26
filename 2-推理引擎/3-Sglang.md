
# 一、推理核心问题


![0fdca02b2fdbabb776e534268b2bb70f\.png](图片和附件/0fdca02b2fdbabb776e534268b2bb70f.png)

# 二、sglang的整体架构

![[Pasted image 20260926163657.png]]








# 三、一次请求的路径

![image\.png](图片和附件/image.png)



# 四、perfill

## 为什么perfill和decode要分离

瓶颈不同:

perfill的瓶颈在于gpu的计算量太大，需要根据输入，计算模型中每一次的参数；天然具有频繁gpu就计算的诉求

decode是自回归的过程，根据前面k v值，来计算下一个token，需要占据大量的显存资源

两者是矛盾的

## perfill面临三类压力

![image\.png](图片和附件/image%201.png)



## flashAttention







## Chunked perfill
如果prompt太长，会让perfill的时间太长，阻塞短的请求，且使得整体perfill的结果不够丝滑，甚至会由于占据过大显存引发oom问题。因此使用chunked perfill将长的prompt划分开一段一段的
对应的调整参数：**`--chunked-prefill-size`**



 




## Prefix cache--》radix cache







## 如何保证不oom







# 五、Decode

## 性能卡点


## 解码策略




## 投机解码







# 六、Kv cache的实现和优化手段

![[Pasted image 20260926160258.png]]

## 注意力架构（根本上优化）
通过修改 Transformer 注意力结构，直接减少每一个 token 产出的 KV 张量大小，属于模型训练阶段内置优化，推理侧直接生效，无额外运行时开销。

1. **MQA 多查询注意力**  
    
    原理：全部 Query 头共享同一套 K/V 头，KV 头数量压缩至 1 组，KV 显存直接下降 N 倍。 
    
    优点：压缩比极高，推理解码速度快。 
    
    缺点：会带来一定生成质量损失，需要模型重训 / 微调补偿。 
    
    适用：追求吞吐、可接受轻微效果损失的场景。
    
2. **GQA 分组查询注意力（工业界事实标准）**  
    
    原理：把 Query 头划分为若干组，组内共享一套 KV 头。例如 Llama3‑70B：64 个 Q 头，8 个 KV 头，KV Cache 压缩 8 倍。 
    
    优点：MHA 与 MQA 之间最佳折中，质量损失很小，绝大多数开源商用模型（Llama2/3、Qwen、DeepSeek）标配 GQA。 
    
    缺点：仍需要训练阶段支持，不能直接对 MHA 模型直接改推理启用。
    
3. **MLA 多头隐式注意力**  
    
    原理：不维护完整 KV 张量，用低维隐向量替代原始 K/V，推理时再恢复，进一步压缩 KV 体积，专门面向百万级超长上下文。 
    
    优点：同等显存下支持更长上下文，KV 体积远优于 GQA。 
    
    缺点：需要模型原生训练支持，无法直接用于普通 LLM，推理内核需要专门适配。
    

小结：GQA 是线上部署首选；MLA 是新一代长上下文模型方向；MQA 现在已经很少单独使用。

## kv压缩减少存储
将 FP16/BF16 的 KV 张量，用更低比特存储，在显存中减少每个 token 的 KV 字节占用，Prefill 计算不变，仅改变存储格式，解码阶段实时反量化读取 K/V。

1. **FP8 KV Cache（当前生产主流）**  
    
    原理：把 KV 缓存存储为 FP8_e4m3，显存直接减半，vLLM、SGLang 原生支持。 
    
    优点：硬件原生支持，反量化开销小，精度损失很小，工程落地成熟。 缺点：需要支持 FP8 的 GPU（Ada、Hopper、Blackwell）；旧硬件无法加速。
    
2. **INT8 / INT4 KV Cache**  
    
    原理：将 KV 压缩为 8bit、4bit 整数，进一步压缩显存。代表方案 KIVI、TurboQuant。 KIVI 采用**非对称量化**：Key 按通道量化，Value 按 Token 量化，解决 KV 缓存数值分布不一致问题。 
    
    优点：压缩比高，旧 GPU 也可以运行。 
    
    缺点：4bit 会带来长上下文检索能力下降；反量化带来额外算力开销；vLLM/SGLang 官方原生支持有限，更多见于 LMDeploy、TurboMind。
    
3. **KV‑Compress、TurboQuant 编解码方案**  
    
    原理：针对每个注意力头使用可变压缩倍率，块级压缩 KV block，和 PagedAttention 分页架构兼容。 
    
    优点：兼顾压缩率与精度。 
    
    缺点：自定义内核，通用性差，大规模生产落地较少。
    

权衡边界：FP8 优先线上使用；INT4 仅用于对精度容忍度高的业务；高检索、Agent 工具调用场景，不建议直接 4bit KV。

## 动态稀疏与 Token 驱逐 Eviction（动态丢弃不重要 KV）

当显存不足，识别、丢弃注意力权重低的历史 token 对应的 KV Cache，保留关键 token，把 KV 占用从 O (N) 变成 O (窗口大小)，不修改模型权重，推理运行时动态执行，非常适合超长上下文场景。

1. **StreamingLLM（滑动窗口 + 注意力 Sink）**  
    
    原理：永久保留开头少数 Sink token，再保留最近 N 个 token，中间久远历史直接丢弃。 
    
    优点：实现简单，开销低。 
    
    缺点：会丢失中间历史信息，长文档问答任务会出现信息遗忘。
    
2. **H2O、SnapKV**  
    
    原理：基于历史注意力分数，动态统计 token 重要度，驱逐低分 token 的 KV，同时保留最近窗口 token。 
    
    优点：比单纯滑动窗口召回效果更好。 
    
    缺点：需要维护注意力统计，带来少量 GPU 开销；部分长检索任务精度会下降。
    
3. **MorphKV、R‑KV**  
    
    MorphKV：聚合历史注意力信号动态筛选重要旧 token。 
    
    R‑KV：专门针对 CoT 推理长输出，同时考虑 token 重要性与信息冗余，推理场景下可以把 KV 缓存压缩到 10%，精度几乎无损。 
    
    优点：推理 Agent 思维链场景收益明显。 
    
    缺点：算法复杂，对推理引擎内核侵入性强。
    
4. **PagedEviction**  
    
    原理：和 PagedAttention 分页架构深度适配，按 Block 整块驱逐，而不是驱逐零散 token，避免内存碎片，vLLM 生态方案。
    

适用场景：文档解析、超长对话 Agent；风险：驱逐错误关键 token 会造成幻觉、信息丢失；不能无脑开最大驱逐。

## 内存管理工程优化（更高效分配、复用 KV Cache）

不改变 KV 本身大小，优化显存分配、生命周期、共享复用，解决碎片、重复计算浪费，是现代推理引擎的基础底座。

1. **PagedAttention 分页注意力（vLLM 核心）**   
    
    https://arxiv.org/abs/2309.06180
    原理：借鉴操作系统分页机制，KV Cache 切分为固定大小 Block 页，逻辑连续，物理显存可以非连续存放，用 BlockTable 页表记录映射关系。 
    
    收益：消除显存碎片，GPU 显存利用率从 20‑40% 提升至 95% 以上；支持抢占、会话迁移。 
    
    局限：不减少 KV 总字节，只优化分配效率；SGLang、MindIE 都实现同类机制。
    
2. **Prefix Caching 前缀缓存 / RadixAttention（SGLang）**  
    
    原理：不同请求存在公共前缀（相同 System Prompt、知识库上下文、工具描述），把已经计算完成的 KV Block 保存，通过基数树 RadixTree 索引，新请求命中直接复用 KV，跳过重复 Prefill 计算。 
    
    收益：大幅降低 TTFT 首 token 延迟，减少重复 Prefill 算力消耗；System Prompt 越长收益越高。 
    
    变种：vLLM APC 自动前缀缓存，跨会话共享 KV 块。 局限：需要足够显存存放公共前缀；会话多、前缀差异大时命中率下降。
    
3. **KV Cache Offloading 分层卸载**  
    
    原理：GPU 显存放不下全部 KV，把冷 KV 块卸载到 CPU 内存、NVMe SSD，需要时再加载回 GPU 显存计算；代表 FlexGen、InfiniGen、MooncakeStore。 分层 L1 (GPU HBM)、L2 (CPU DRAM)、L3 (SSD / 分布式存储)。 
    
    优点：极大扩展单实例可承载上下文长度。 
    
    缺点：卸载加载会带来延迟抖动；SSD 介质访问延迟高，只适合冷 KV。

## 分布式与系统架构级 KV Cache 优化（集群维度）

单 GPU 显存终究有限，把 KV Cache 从单机 GPU 扩展到整个集群，是大规模 Agent、PD 分离架构下核心方案。

1. **PD 分离（Prefill‑Decode Disaggregation）配套 KV 传输**  
    
    原理：Prefill 集群计算生成 KV Cache，通过 RDMA 高速网络把 KV Cache 块传输给 Decode 集群，不在同一套 GPU 完成两个阶段。 
    
    关键点：传输 KV Cache 中间状态，不传输权重；对网络 RDMA 要求高。 
    
    代表：Mooncake 点对点传输组件，vLLM disaggregated‑prefill 能力。
    
2. **分布式 KV Cache 共享池（Mooncake Store）**  
    
    原理：集群全局统一 KV 存储池，多推理节点共享 KV 资源；KV 块可以存放在集群节点 CPU 内存、SSD；任意推理节点按需读取 KV 块，支持跨实例会话迁移、跨实例 Prefix Cache 命中。 
    
    价值：AI 智能体长任务场景，会话可以调度到任意 Decode 节点，不需要重新 Prefill；是 Agent 推理基础设施关键组件。 
    
    挑战：元数据管理、RDMA 网络、缓存淘汰策略、副本故障处理。
    
3. **多级全局缓存架构 HiCache（SGLang）**  
    
    L1 GPU 显存，L2 本机 CPU 内存，L3 分布式共享存储；统一管理本地 + 分布式 KV 缓存，兼顾访问延迟与总容量。


## 生产应用之radis cache 


## 生产应用之hicache



## 生产应用之mooncake



# 七、serving调度和服务视角
## Continuous batch

主要面临三个问题：

1. 早执行完毕的reqest怎么处理：通过scheduler的loop形式一直检查，有完成的请求就拿出来，并检查后来的request并放入

2. 后加入的request怎么处理

3. 不同长度的prompt如何在一个batch内推理： perfill阶段，将全部reqest进行flatten成一个大的一维向量，decode阶段由于都是单个token维度的计算，因此不涉及这个问题












# 八、分布式通信和并行策略

## 通信协议

### RDMA（远端内存访问）

绕过cpu，直接gpu和gpu之间直接通信





### NVlink





## 通信原语

1. 集合通信\(cc\)：一组节点之间通信

2. 点对点通信\(p2p\)：两个节点的通信

3. 核心区别：集合通信是N个gpu一起抢带宽，而点对点通信仅有2个



⚠️ 注意：通信发生在不同层级的硬件上，带宽差一个数量级——机内 NVLink 可达数百 GB/s，机间 InfiniBand 约 50 GB/s 量级，PCIe 更低。同样一个 AllReduce，放机内还是跨机，耗时可能差十倍。这是后续所有”哪个策略放哪里”决策的物理根源。



![12abe68e57b82901a9c2d8d4a8377ad9\.png](图片和附件/12abe68e57b82901a9c2d8d4a8377ad9.png)





## 并行策略

tp
dp
ep






# 九、attention backend

**Attention Backend 是底层的“计算内核”**：它是真正在 GPU 上执行注意力计算的**算子实现**（如 FlashAttention、FlashInfer、Triton 等）。它负责“如何高效计算一次注意力”，解决的是 **“计算性能”** 问题







