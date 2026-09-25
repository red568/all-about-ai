# GPU

# Why GPU

1. 并行计算的优势，不要求低延迟，但要求大量并行处理的能力（成百上万个核心简单计算）

2. 深度学习的矩阵计算天然适配gpu并行计算的架构

# GPU整体架构



![image\.png](all-about-ai/1-GPU和算子/图片和附件/image.png)



```Plain Text
GPU 芯片
├── GPC (Graphics Processing Cluster)  ← "事业部"
│   ├── TPC (Texture Processing Cluster)  ← "部门"
│   │   ├── SM (Streaming Multiprocessor)  ← "小组"，GPU 的基本调度单元
│   │   │   ├── CUDA Core × N  ← "组员"，执行 FP32/INT32 运算
│   │   │   ├── Tensor Core × M  ← "专家"，执行矩阵运算
│   │   │   ├── SFU (Special Function Unit)  ← 执行 sin/cos/exp 等
│   │   │   ├── Register File  ← 寄存器堆（速度最快的存储）
│   │   │   ├── Shared Memory  ← 共享内存（SM 内所有线程共享）
│   │   │   └── L1 Cache
│   │   └── SM ...
│   └── TPC ...
├── GPC ...
├── L2 Cache（全局共享）
└── Memory Controller → HBM（显存）

```







# SM

SM（Streaming Multiprocessor）是理解 GPU 架构的关键，所有的线程调度和执行都发生在 SM 层面。

📌 关键点：SM 是资源分配的最小粒度。当你编写 CUDA Kernel 时，一个 Thread Block 会被调度到一个 SM 上执行。SM 内的寄存器、共享内存等资源由这个 SM 上的所有 Thread Block 共同分配。



![image\.png](all-about-ai/1-GPU和算子/图片和附件/image%201.png)







# Wrap

GPU 不是一个线程一个线程执行的，而是以 Warp（线程束）为单位，32 个线程锁步执行同一条指令。这就像军队齐步走——32 个士兵听同一声号令，迈出同一只脚。



这种执行模式叫做 SIMT（Single Instruction, Multiple Threads），是 NVIDIA 对 SIMD 的扩展。

⚠️ 注意：如果 Warp 内的线程遇到分支（if\-else），走不同分支的线程会被 掩码（mask），导致部分线程空转。这就是所谓的 Warp Divergence，是 GPU 编程中需要极力避免的性能杀手。























# 如何计算模型需要多大的显存？多少gpu？什么型号的gpu?

