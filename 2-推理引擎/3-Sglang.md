# Sglang

# 推理核心问题

![0fdca02b2fdbabb776e534268b2bb70f\.png](图片和附件/0fdca02b2fdbabb776e534268b2bb70f.png)

# sglang的整体架构









# 一次请求的路径

![image\.png](图片和附件/image.png)







# perfill

## 为什么perfill和decode要分离

瓶颈不同

perfill的瓶颈在于gpu的计算量太大，需要根据输入，计算模型中每一次的参数；天然具有频繁gpu就计算的诉求

decode是自回归的过程，根据前面k v值，来计算下一个token，需要占据大量的显存资源

两者是矛盾的

## perfill面临三类压力

![image\.png](图片和附件/image%201.png)





## flashAttention







## Chunked perfill









## Redix cache





## 如何保证不oom









# Decode









# Kv cache









# Continuous batch

主要面临三个问题：

1. 早执行完毕的reqest怎么处理：通过scheduler的loop形式一直检查，有完成的请求就拿出来，并检查后来的request并放入

2. 后加入的request怎么处理

3. 不同长度的prompt如何在一个batch内推理： perfill阶段，将全部reqest进行flatten成一个大的一维向量，decode阶段由于都是单个token维度的计算，因此不涉及这个问题







# 







# pagedAttention







# 







# 分布式通信和并行策略

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

















