
https://docs.nvidia.com/cuda/cuda-programming-guide
# 相关概念

1. **Host 和 Device**：CPU 叫 host，GPU 叫 device
2. **Kernel**：在 GPU 上并行执行的函数
3. **线程索引**：每个 GPU 线程怎么找到自己该算哪份数据
4. **内存拷贝**：数据要在 CPU 内存和 GPU 显存之间搬来搬去
5. **kernel launch = 从 CPU 端发起一个 GPU kernel，让 GPU 开始执行它**：myKernel<<<grid, block>>>(a, b);


# quick start

```
nvidia-smi          # 看有没有 NVIDIA GPU
nvcc --version      # 看 CUDA 编译器是否安装
```

新建一个hello.cu程序文件
```cpp
#include <stdio.h>

# __global__意味着是gpu 核函数
__global__ void helloGPU() {
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

int main() {
	# 启动入口
    helloGPU<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

编译运行：
```
nvcc hello.cu -o hello   编译代码为可执行文件hello
./hello  执行hello文件
```

# 向量加法
理解 kernel、线程索引、内存拷贝




# 矩阵乘法
 
 理解二维线程组织

# 共享内存
 
 `__shared__`，优化矩阵乘

# Stream 和 Event
 
 异步、重叠计算和拷贝

# cuda graphs

https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html

可以把它理解成：

- **普通 CUDA 执行**：CPU 逐个调用 `kernel<<<>>>`、`cudaMemcpyAsync` 等，每次都有启动/提交开销。
    
- **CUDA Graph**：先把这些操作和依赖“录下来”形成图，节点是 kernel、memcpy、memset 等操作，边表示谁依赖谁。之后每次执行整张图，而不是让 CPU 重新逐个提交。
    

## 核心概念

- **节点 Node**：一个 GPU 操作，比如 kernel launch、内存拷贝、memset、host function、子图等。
    
- **边 Edge**：依赖关系。例如 B 必须在 A 完成后才能执行。
    
- **实例化 Instantiate**：把图编译成可执行图 `cudaGraphExec_t`。
    
- **启动 Launch**：用 `cudaGraphLaunch` 把可执行图提交到 stream。
    
- **重复执行**：同一张图可以反复 launch，适合循环迭代场景。
    

## 三个核心阶段
Graph 的工作提交分为三个明确阶段：**定义（Definition）→ 实例化（Instantiation）→ 执行（Execution）**

### 定义（构建graph）

1. **显式 API**  
    用 `cudaGraphCreate`、`cudaGraphAddKernelNode`、`cudaGraphAddMemcpyNode` 等手动建图。
    
2. **流捕获 Stream Capture**  
    将已有的 stream 代码用 `cudaStreamBeginCapture()` 和 `cudaStreamEndCapture()` 括起来，即可自动捕获为 graph
```
    cudaStreamBeginCapture(stream, ...);
    kernelA<<<..., stream>>>();
    kernelB<<<..., stream>>>();
    cudaMemcpyAsync(..., stream);
    cudaStreamEndCapture(stream, &graph);
```

### 实例化：创建可执行图

Graph 创建后必须实例化才能启动, 实例化时，驱动将 graph 编译为可高效重复执行的可执行图

```
cudaGraphExec_t graphExec;
cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
```


### 执行：启动 Graph

```
cudaGraphLaunch(graphExec, stream);
```

整张图一次性提交到指定 stream。同一个 `cudaGraphExec_t` 可以反复 launch，开销极低

## 生命周期

![[Pasted image 20261006001926.png]]

## 为什么有用

- **降低 CPU 启动开销**：小 kernel 很多时，CPU 逐个 launch 可能成为瓶颈。
    
- **减少 GPU 空闲和抖动**：驱动可以更高效地调度整张图。
    
- **表达复杂依赖**：多流、多 kernel 之间的依赖可以显式描述。
    
- **可更新参数**：图结构不变时，可以只更新节点参数，比如指针、kernel 参数，而不重建整张图。
    

## 适用场景和限制

- 深度学习训练/推理，尤其是 LLM 推理中大量小算子重复执行。
    
- 科学计算中的迭代求解。
    
- 固定流水线、固定拓扑、反复执行的任务。
    
- 多流并行且依赖关系复杂的 GPU 工作负载。
    


**限制**

- 图拓扑最好固定；频繁改变依赖或动态控制流不适合。
    
- 首次实例化有开销，也占额外显存。
    
- 捕获期间很多同步、查询类 API 不能随便用。
    
- 调试和性能分析比普通 stream 复杂一些。
    
- 它描述的是底层 GPU 任务调度图，不是 TensorFlow/PyTorch 那种模型计算图。
    

一句话：**CUDA Graph 就是把一串 CUDA 操作和依赖关系打包成一张可重复高效执行的 GPU 任务图，主要用来减少启动开销、提升重复工作负载的执行效率。**



# Nsight 性能分析

