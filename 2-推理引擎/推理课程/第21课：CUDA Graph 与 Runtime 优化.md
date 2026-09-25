---
title: "第21课：CUDA Graph 与 Runtime 优化"
source: "https://everythingai.top/learn/ai-systems-performance-engineering-text/lesson-21"
author:
published:
created: 2026-09-26
description: "面向 AI 工程学习者的专业课程学习空间。"
tags:
  - "clippings"
---
**导航**

本课通过 CUDA Graph 与 Runtime 优化减少 CPU 提交和 Kernel Launch 开销，提升重复工作负载效率。

## 一、课程定位

前面的课程主要优化“GPU 做一件事有多快”：访存是否合并、Kernel 是否融合、Tensor Core 是否吃满。但真实 AI 工作负载还有另一类瓶颈：GPU 很快，CPU 却来不及把工作提交给它。

一次训练 Step 或一次 LLM Decode Step 可能包含数十到数百个短 Kernel。每个 Kernel 的计算只需几微秒，Python、Dispatcher、CUDA Runtime 和驱动却要反复完成参数准备、依赖检查与 Launch。GPU 时间线中便会出现一串“短 Kernel + 空洞”。

CUDA Graph 的核心价值是：把一段重复的 GPU 工作及其依赖捕获成图，实例化一次，之后用一次 Graph Launch 重放整段工作。它主要降低 Runtime 和 Launch 开销，不会让单个 Kernel 的数学计算自动变快。

## 二、学习目标

- 理解 CUDA Runtime、Stream、Event、Graph Node、Graph Executable 的关系。
- 区分显式建图和 Stream Capture 两种方式。
- 掌握 Capture、Instantiate、Replay、Update 的生命周期。
- 理解静态地址、静态 Shape、禁止 CPU 同步和私有内存池等约束。
- 正确测量 CPU Submit Time、GPU Execution Time、端到端延迟和盈亏平衡点。
- 使用 PyTorch `torch.cuda.CUDAGraph` 完成可运行实验。
- 使用原生 CUDA C++ 对比普通 Kernel Launch 与 Graph Replay。
- 能判断何时应使用 Graph、 `torch.compile` 、Kernel Fusion、Batching 或 Stream 并发。

## 三、前置知识

- CUDA Kernel、Grid、Block、Stream 和 Event 基础。
- PyTorch CUDA 异步执行与 `torch.cuda.synchronize()` 。
- 第 20 课的 `torch.compile` 、Graph Break 和稳态测量方法。
- 第 16～17 课的 NCCL、Collective 与计算通信重叠。

## 四、核心直觉：把逐张工单变成一张固定流程单

普通 Eager 执行像工厂每完成一道工序，都回办公室领取下一张工单：

```
CPU: Launch A ─ Launch B ─ Launch C ─ Launch D
         │          │          │          │
GPU:    Kernel A   Kernel B   Kernel C   Kernel D
```

CUDA Graph 则先记录完整工艺流程，后续只提交一次：

```
首次：Capture(A→B→C→D) → Instantiate
后续：Graph Launch ─────→ GPU 执行 A→B→C→D
```

它省下的是重复提交和调度成本。若 A、B、C、D 本身都是很长的大 GEMM，CPU 早已提前把任务排入队列，Graph 收益可能很小；若它们是大量极短 Kernel，Graph 更容易显著降低空洞。

因此先记住一句判断口诀：

## 五、从 CUDA Runtime 看性能瓶颈

### 5.1 Kernel Launch 是异步的，但不是免费的

典型 CUDA Kernel Launch 对 Host 异步：CPU 提交后可以继续运行，GPU稍后执行。然而每次提交仍经过应用、框架、Runtime、Driver 到设备队列。单次成本不一定大，但 K 个小 Kernel 会累计为：

$$
T_{host}=T_{python}+T_{framework}+\sum_{i=1}^{K}T_{launch,i}+T_{sync}
$$

单 Step 的 GPU 路径近似为：

$$
T_{gpu}=\sum_{i=1}^{K}T_{kernel,i}+T_{memory}+T_{dependency}
$$

端到端时间不是简单相加，而由 CPU 与 GPU 时间线的关键路径决定：

$$
T_{step}\approx \max(T_{host\ submit},T_{gpu})+T_{unhidden\ sync}
$$

当 $T_{host\ submit}>T_{gpu}$ 时，GPU 会等待 CPU，系统处于 Launch-bound。

### 5.2 隐式同步是 Runtime 优化的大敌

下列操作可能让 CPU 等待 GPU：

- `tensor.item()` 、将 CUDA Tensor 打印成具体值；
- 立即把结果复制回 Pageable Host Memory；
- `torch.cuda.synchronize()` 、 `cudaDeviceSynchronize()` ；
- 某些内存分配、错误检查或跨 Stream 依赖；
- 读取尚未完成的 CUDA Event。

同步并非一律错误，错误的是把同步放在高频热路径中，或者不知道它存在。

### 5.3 Runtime 优化不只有 CUDA Graph

从低风险到高约束，常见顺序是：

1. 删除不必要同步与 Python 标量往返；
2. 使用 `inference_mode` 、批处理和预分配；
1. 用 Kernel Fusion 减少 Launch 数和中间显存流量；
2. 用异步 H2D、Pinned Memory、Stream 实现搬运与计算重叠；
1. 用 `torch.compile` 自动捕获和融合；
2. 对稳定热路径使用 CUDA Graph；
1. 必要时用 Persistent Kernel 或自定义 Runtime。

Graph 与 Fusion 解决的不是同一个问题：Fusion 把多个算子变为更少的 Kernel；Graph 让仍然存在的多个 Kernel 以更低 Host 开销被重放。二者可以叠加。

## 六、CUDA Graph 的对象模型

### 6.1 Graph、Node、Edge

CUDA Graph 是有向无环图：

- Node：Kernel、Memcpy、Memset、Event、Host Function、子图等工作单元；
- Edge：节点间依赖；
- `cudaGraph_t` ：图的定义；
- `cudaGraphExec_t` ：完成验证和优化后的可执行实例。

一个常见生命周期是：

```
Warmup
  ↓
Capture / Explicit Construction
  ↓
cudaGraph_t
  ↓ Instantiate
cudaGraphExec_t
  ↓ Launch / Replay many times
Destroy
```

Instantiate 可能进行拓扑验证、资源准备和调度优化，所以不能把它混入稳态延迟。

### 6.2 显式建图与 Stream Capture

显式 API 手工创建 Node 和 Edge，控制精确但代码量大。Stream Capture 更常见：

```
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
kernel_a<<<grid, block, 0, stream>>>(...);
kernel_b<<<grid, block, 0, stream>>>(...);
library_call(..., stream);
cudaStreamEndCapture(stream, &graph);
```

Capture 期间，提交到相关 Stream 的工作被记录到图里。PyTorch `torch.cuda.graph(...)` 和 `make_graphed_callables(...)` 在此基础上提供了更安全的封装。

### 6.3 Replay 为什么更便宜

普通模式每次都重走 K 次提交路径，Graph 模式主要提交一个可执行图：

$$
T_{normal}=K\cdot T_{launch}+T_{gpu}
$$

$$
T_{graph}=T_{graph\ launch}+T_{gpu}+T_{input\ update}
$$

理论节省量近似为：

$$
\Delta T\approx K\cdot T_{launch}-T_{graph\ launch}-T_{input\ update}
$$

这只是直觉模型。Driver 可能批量提交，Kernel 可并发，输入 Copy 也可能成为关键路径，最终必须 Profile。

## 七、最重要的约束

### 7.1 固定虚拟地址

Graph Replay 默认读写与 Capture 时相同的虚拟地址。PyTorch 典型做法是保留长期存活的 `static_input` 和 `static_output` ：

```
static_input.copy_(new_input)
graph.replay()
result = static_output
```

不能每次把一个新 Tensor 对象直接“传给 Graph”。新的业务输入先复制到静态输入 Buffer。

### 7.2 静态 Shape 和布局

捕获后的 Kernel 参数、内存布局和拓扑高度稳定。不同 Batch、序列长度或 dtype 通常需要不同 Graph。生产实践常用有限 Bucket：

```
batch: 1 / 2 / 4 / 8
sequence: 128 / 256 / 512 / 1024
每个高频组合预热并缓存一个 Graph
```

Bucket 太多会增加捕获成本、私有内存池和缓存管理压力。

### 7.3 Capture 期间不能做 CPU-GPU 同步

`.item()` 等操作要求 CPU 读取 GPU 结果，会破坏 Capture。CPU 工作本身也不会在 Replay 时重新执行，因此 Tokenizer、日志、文件 I/O 和 Python 动态分支应放在 Graph 外。

### 7.4 Warmup 必须先完成

第一次调用可能触发：

- CUDA Context 初始化；
- cuBLAS/cuDNN 算法选择与 Handle 创建；
- JIT/编译；
- PyTorch Caching Allocator 扩容；
- Lazy Module 或通信器初始化。

这些动作在 Capture 中可能失败或被错误记录。PyTorch 手工捕获前应在 Side Stream 做足够 Warmup，再等待其完成。

### 7.5 Graph 私有内存池

为保证地址稳定，PyTorch Caching Allocator 会为 Graph 管理私有内存池。Graph 对象和捕获时创建的 Tensor 仍存活时，池可能一直保留。多个 Graph 默认各自拥有池，显存可能明显增加；仅在生命周期不重叠且语义安全时共享池。

### 7.6 随机数与状态更新

PyTorch 支持部分 CUDA RNG 操作，但生成器必须遵循 Graph-safe 状态管理。自定义 RNG、CPU 随机逻辑、动态 Optimizer/GradScaler 状态需要单独验证。Dropout“能运行”不等于随机序列和训练语义必然正确。

## 八、Graph Update 与动态工作负载

原生 CUDA 支持更新已实例化图。若拓扑相同，只是 Kernel 参数、某些内存地址或 Memcpy 参数变化，可以使用整图更新 `cudaGraphExecUpdate()` 或单 Node Update，避免重新 Instantiate。

但它不是“任意动态图”能力：

- 图拓扑或 Node 类型大改通常要重新实例化；
- 内存上下文、Memcpy 类型和 Kernel 能力有更新限制；
- 框架封装不一定暴露所有原生 Update API；
- 维护复杂度可能高于管理有限 Graph Bucket。

更新适合拓扑稳定、参数变化有限的低层 Runtime。高层 PyTorch/LLM 服务通常先用 Bucket；只有 Profile 证明捕获/缓存成本仍是瓶颈，才考虑原生 Update。

## 九、关键性能模型

### 9.1 盈亏平衡次数

设捕获和实例化成本为 $T_c$ ，普通稳态时间为 $T_e$ ，Graph Replay 稳态时间为 $T_g$ ，则：

$$
N^*=\frac{T_c}{T_e-T_g},\quad T_g < T_e
$$

如果每个 Shape Bucket 都需独立捕获， $T_c$ 应按实际 Graph 数累计。

### 9.2 Graph 能优化的比例

若原时间中只有比例 p 属于可捕获、Launch-bound 的热路径，该部分通过 Graph 加速 s 倍，则：

$$
S_{total}=\frac{1}{(1-p)+p/s}
$$

Tokenizer、DataLoader、网络队列、长 GEMM 和未捕获通信所占比例越大，端到端收益越小。

### 9.3 多 Stage Pipeline

一个服务请求可拆为：

$$
T=T_{queue}+T_{H2D}+T_{compute}+T_{D2H}+T_{sync}
$$

Graph 只降低其中可捕获部分。若 H2D 和 D2H 主导，应优先使用 Pinned Memory、批处理和 Stream Pipeline；若 Queue 主导，应优化 Scheduler；若长 GEMM 主导，应优化算子与精度。

## 十、瓶颈分析方法

### 10.1 先判断是否 Launch-bound

用 Nsight Systems 观察：

- GPU Kernel 是否很短且密集；
- Kernel 之间是否有可见空洞；
- CPU CUDA API 调用是否占满一个核心；
- GPU 是否经常等待下一次提交；
- `.item()` 、Memcpy 或 Synchronize 是否切断流水线。

若 GPU 上是连续的大 GEMM，Graph 通常不是第一优先级。

### 10.2 四组数字必须分开

1. Capture + Instantiate 时间；
2. CPU Submit Time；
1. GPU Event 测得的设备执行时间；
2. 包含输入 Copy 和必要同步的端到端时间。

Graph 能大幅降低 CPU Submit Time，却不一定按同样比例降低 GPU Event 时间，这是正常现象。

### 10.3 Nsight Systems 命令

```
nsys profile --trace=cuda,nvtx,osrt --cuda-graph-trace=graph \
  -o cuda_graph_summary python cuda_graph_lab.py

nsys profile --trace=cuda,nvtx,osrt --cuda-graph-trace=node \
  -o cuda_graph_nodes python cuda_graph_lab.py
```

`graph` 粒度扰动更小； `node` 粒度能看内部节点但采集开销更大。最终性能仍用无 Profile 基准确认。

## 十一、Level 0：CPU Runtime 开销模拟实验

该实验不伪装成 GPU Graph，而是帮助没有 NVIDIA GPU 的读者理解“批量提交如何摊薄固定开销”。保存为 `runtime_overhead_model.py` ：

```
#!/usr/bin/env python3
import argparse
import time

def tiny_work(x, rounds):
    for i in range(rounds):
        x = (x * 1664525 + 1013904223 + i) & 0xFFFFFFFF
    return x

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--nodes", type=int, default=32)
    p.add_argument("--work-rounds", type=int, default=2)
    p.add_argument("--launch-us", type=float, default=3.0)
    args = p.parse_args()
    if min(args.steps, args.nodes, args.work_rounds) < 1 or args.launch_us < 0:
        raise SystemExit("参数必须为正数，launch-us 不能为负数")

    # 用算术循环近似每个节点的计算；固定 launch 成本用公式计入，
    # 避免 time.sleep 在微秒区间误差过大。
    x = 1
    t0 = time.perf_counter()
    for _ in range(args.steps):
        for _ in range(args.nodes):
            x = tiny_work(x, args.work_rounds)
    compute_s = time.perf_counter() - t0

    eager_launch_s = args.steps * args.nodes * args.launch_us / 1e6
    graph_launch_s = args.steps * args.launch_us / 1e6
    eager_total = compute_s + eager_launch_s
    graph_total = compute_s + graph_launch_s

    print(f"checksum={x}")
    print(f"compute_s={compute_s:.6f}")
    print(f"eager_model_s={eager_total:.6f}")
    print(f"graph_model_s={graph_total:.6f}")
    print(f"modeled_speedup={eager_total / graph_total:.3f}x")
    print(f"launch_overhead_saved_s={eager_launch_s - graph_launch_s:.6f}")

if __name__ == "__main__":
    main()
```

运行：

```
python3 runtime_overhead_model.py
python3 runtime_overhead_model.py --nodes 4 --work-rounds 100 --launch-us 3
python3 runtime_overhead_model.py --nodes 64 --work-rounds 1 --launch-us 3
```

预期现象：节点越多、单节点计算越短，Launch 开销占比越高，批量提交模型的收益越大；当每个节点计算很重时，Launch 开销被淹没，Speedup 接近 1。

这个实验只验证性能模型，不等价模拟 CUDA Stream、Driver 或 GPU 并发。

## 十二、Level 1：PyTorch CUDA Graph 完整实验

保存为 `cuda_graph_lab.py` 。脚本会自动输出 GPU、Compute Capability、CUDA/PyTorch 版本，并分别测量 Eager、Graph Replay 和“输入 Copy + Replay”。

```
#!/usr/bin/env python3
import argparse
import statistics
import time

import torch
from torch import nn

class LaunchBoundModel(nn.Module):
    def __init__(self, depth):
        super().__init__()
        self.depth = depth

    def forward(self, x):
        # 有意保留多个短 pointwise 算子，突出 Runtime 开销
        for _ in range(self.depth):
            x = torch.tanh(x) * 0.75 + torch.sigmoid(x) * 0.25
        return x

def sync():
    torch.cuda.synchronize()

def gpu_time_ms(fn, iterations):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations

def cpu_submit_us(fn, iterations):
    sync()
    samples = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(iterations):
            fn()
        submit_us = (time.perf_counter() - t0) * 1e6 / iterations
        sync()
        samples.append(submit_us)
    return statistics.median(samples)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--width", type=int, default=4096)
    p.add_argument("--depth", type=int, default=16)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iterations", type=int, default=100)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("需要可用的 NVIDIA CUDA GPU；无 GPU 请运行 Level 0")
    if min(args.batch, args.width, args.depth, args.warmup, args.iterations) < 1:
        raise SystemExit("所有整数参数必须为正数")

    device = torch.device("cuda")
    prop = torch.cuda.get_device_properties(device)
    print(f"torch={torch.__version__} cuda={torch.version.cuda}")
    print(f"gpu={prop.name} capability={prop.major}.{prop.minor} "
          f"vram_gib={prop.total_memory / 2**30:.2f}")

    torch.manual_seed(0)
    model = LaunchBoundModel(args.depth).eval().to(device)
    static_input = torch.randn(args.batch, args.width, device=device)

    # 必须在 Side Stream 预热，完成 Lazy 初始化和内存池准备。
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream), torch.inference_mode():
        for _ in range(args.warmup):
            model(static_input)
    torch.cuda.current_stream().wait_stream(warmup_stream)
    sync()

    graph = torch.cuda.CUDAGraph()
    capture_begin = time.perf_counter()
    with torch.inference_mode(), torch.cuda.graph(graph):
        static_output = model(static_input)
    sync()
    capture_ms = (time.perf_counter() - capture_begin) * 1000.0

    # 新输入必须 copy 到捕获时的静态地址。
    real_input = torch.randn_like(static_input)
    with torch.inference_mode():
        reference = model(real_input)
        static_input.copy_(real_input)
        graph.replay()
    sync()
    torch.testing.assert_close(static_output, reference, rtol=2e-4, atol=2e-4)

    with torch.inference_mode():
        eager = lambda: model(real_input)
        replay_only = graph.replay

        def copy_and_replay():
            static_input.copy_(real_input)
            graph.replay()

        # 再预热计时路径
        for _ in range(5):
            eager()
            replay_only()
            copy_and_replay()
        sync()

        eager_gpu = gpu_time_ms(eager, args.iterations)
        graph_gpu = gpu_time_ms(replay_only, args.iterations)
        graph_copy_gpu = gpu_time_ms(copy_and_replay, args.iterations)
        eager_submit = cpu_submit_us(eager, args.iterations)
        graph_submit = cpu_submit_us(replay_only, args.iterations)
        graph_copy_submit = cpu_submit_us(copy_and_replay, args.iterations)

    print(f"capture_and_instantiate_ms={capture_ms:.3f}")
    print(f"eager_gpu_ms={eager_gpu:.4f}")
    print(f"graph_replay_gpu_ms={graph_gpu:.4f}")
    print(f"graph_copy_replay_gpu_ms={graph_copy_gpu:.4f}")
    print(f"eager_cpu_submit_us={eager_submit:.3f}")
    print(f"graph_replay_cpu_submit_us={graph_submit:.3f}")
    print(f"graph_copy_replay_cpu_submit_us={graph_copy_submit:.3f}")
    print(f"replay_speedup={eager_gpu / graph_gpu:.3f}x")
    print(f"copy_replay_speedup={eager_gpu / graph_copy_gpu:.3f}x")
    print(f"peak_allocated_mib={torch.cuda.max_memory_allocated()/2**20:.1f}")
    print(f"peak_reserved_mib={torch.cuda.max_memory_reserved()/2**20:.1f}")

if __name__ == "__main__":
    main()
```

### 12.1 安装与运行

```
python3 -m venv .venv-cudagraph
source .venv-cudagraph/bin/activate
python -m pip install --upgrade pip
# 按 PyTorch 官方安装页选择与驱动兼容的稳定 CUDA 构建
```

通用运行：

```
nvidia-smi
python cuda_graph_lab.py
python cuda_graph_lab.py --batch 8 --width 1024 --depth 32 --iterations 200
python cuda_graph_lab.py --batch 128 --width 8192 --depth 8 --iterations 50
```

双 RTX 3080 分卡验证：

```
CUDA_VISIBLE_DEVICES=0 python cuda_graph_lab.py
CUDA_VISIBLE_DEVICES=1 python cuda_graph_lab.py
```

### 12.2 预期现象

- `graph_replay_cpu_submit_us` 通常显著低于 Eager，尤其在 `depth` 较大时。
- `graph_replay_gpu_ms` 可能改善，也可能接近 Eager；这取决于 Eager 时间线是否真的被 Host 提交限制。
- `copy_and_replay` 比 Replay-only 更接近真实服务，收益会被输入 Copy 部分抵消。
- 增大每个算子的计算量后，Graph Speedup 往往下降，因为 GPU 计算成为主导。
- Graph 私有内存池可能使 `reserved` 增加；这不是自动等于内存泄漏。

### 12.3 结果分析模板

| 指标 | Eager | Replay-only | Copy + Replay | | ----------------------------------- | -- | --- | --- | | CPU Submit，µs | 实测 | 实测 | 实测 | | GPU 平均时间，ms | 实测 | 实测 | 实测 | | Capture/Instantiate，ms | 0 | 一次性 | 一次性 | | Peak allocated，MiB | 实测 | 实测 | 实测 | | Peak reserved，MiB | 实测 | 实测 | 实测 | | 输出正确性 | 基准 | 对齐 | 对齐 |

性能数字只代表本次环境，不可外推为某种 GPU 的固定结论。

## 十三、Level 2：原生 CUDA C++ Graph 实验

该实验直接调用 CUDA Runtime，用多个极短 Kernel 观察 Launch 压力。保存为 `cuda_graph_native.cu` ：

```
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define CHECK(call) do {                                                   \
  cudaError_t e = (call);                                                  \
  if (e != cudaSuccess) {                                                  \
    std::fprintf(stderr, "%s:%d CUDA error: %s\n",                      \
                 __FILE__, __LINE__, cudaGetErrorString(e));               \
    std::exit(1);                                                          \
  }                                                                        \
} while (0)

__global__ void add_one(float* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] += 1.0f;
}

int main(int argc, char** argv) {
  const int n = argc > 1 ? std::atoi(argv[1]) : 4096;
  const int nodes = argc > 2 ? std::atoi(argv[2]) : 32;
  const int iterations = argc > 3 ? std::atoi(argv[3]) : 1000;
  if (n <= 0 || nodes <= 0 || iterations <= 0) return 2;

  int device = 0;
  cudaDeviceProp prop{};
  CHECK(cudaGetDeviceProperties(&prop, device));
  std::printf("gpu=%s capability=%d.%d\n", prop.name, prop.major, prop.minor);

  float* x = nullptr;
  CHECK(cudaMalloc(&x, n * sizeof(float)));
  cudaStream_t stream;
  CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  const int block = 256;
  const int grid = (n + block - 1) / block;

  for (int i = 0; i < 10; ++i)
    add_one<<<grid, block, 0, stream>>>(x, n);
  CHECK(cudaStreamSynchronize(stream));

  cudaGraph_t graph;
  cudaGraphExec_t graph_exec;
  CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  for (int j = 0; j < nodes; ++j)
    add_one<<<grid, block, 0, stream>>>(x, n);
  CHECK(cudaStreamEndCapture(stream, &graph));
  CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

  cudaEvent_t start, end;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&end));

  CHECK(cudaMemsetAsync(x, 0, n * sizeof(float), stream));
  CHECK(cudaEventRecord(start, stream));
  auto eager_cpu_start = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; ++i)
    for (int j = 0; j < nodes; ++j)
      add_one<<<grid, block, 0, stream>>>(x, n);
  auto eager_cpu_end = std::chrono::steady_clock::now();
  CHECK(cudaEventRecord(end, stream));
  CHECK(cudaEventSynchronize(end));
  float eager_ms = 0.0f;
  CHECK(cudaEventElapsedTime(&eager_ms, start, end));

  CHECK(cudaMemsetAsync(x, 0, n * sizeof(float), stream));
  CHECK(cudaEventRecord(start, stream));
  auto graph_cpu_start = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; ++i)
    CHECK(cudaGraphLaunch(graph_exec, stream));
  auto graph_cpu_end = std::chrono::steady_clock::now();
  CHECK(cudaEventRecord(end, stream));
  CHECK(cudaEventSynchronize(end));
  float graph_ms = 0.0f;
  CHECK(cudaEventElapsedTime(&graph_ms, start, end));

  float first = 0.0f;
  CHECK(cudaMemcpy(&first, x, sizeof(float), cudaMemcpyDeviceToHost));
  const float expected = static_cast<float>(nodes * iterations);
  if (std::fabs(first - expected) > 0.5f) {
    std::fprintf(stderr, "validation failed: got=%f expected=%f\n", first, expected);
    return 3;
  }

  const double eager_submit_ms =
      std::chrono::duration<double, std::milli>(eager_cpu_end-eager_cpu_start).count();
  const double graph_submit_ms =
      std::chrono::duration<double, std::milli>(graph_cpu_end-graph_cpu_start).count();
  std::printf("nodes=%d iterations=%d n=%d\n", nodes, iterations, n);
  std::printf("eager_gpu_total_ms=%.3f graph_gpu_total_ms=%.3f speedup=%.3fx\n",
              eager_ms, graph_ms, eager_ms / graph_ms);
  std::printf("eager_cpu_submit_ms=%.3f graph_cpu_submit_ms=%.3f submit_speedup=%.3fx\n",
              eager_submit_ms, graph_submit_ms, eager_submit_ms / graph_submit_ms);
  std::printf("validation=PASS value=%.0f\n", first);

  CHECK(cudaEventDestroy(start));
  CHECK(cudaEventDestroy(end));
  CHECK(cudaGraphExecDestroy(graph_exec));
  CHECK(cudaGraphDestroy(graph));
  CHECK(cudaStreamDestroy(stream));
  CHECK(cudaFree(x));
  return 0;
}
```

编译与运行：

```
nvcc -O3 -std=c++17 cuda_graph_native.cu -o cuda_graph_native
./cuda_graph_native 4096 32 1000
./cuda_graph_native 1048576 4 100
compute-sanitizer ./cuda_graph_native 4096 32 100
```

第一个配置由大量短 Kernel 构成，Graph 更容易降低提交时间；第二个配置单 Kernel 工作量更大，收益通常缩小。不同驱动、CUDA 和 GPU 的绝对结果不可直接横比。

## 十四、不同硬件的实验边界

| 架构 | 示例 GPU | 可执行内容 | 专项注意 |
| --- | --- | --- | --- |
| Ampere | RTX 3080/3090、A100 | PyTorch/原生 CUDA Graph 基线 | 双 3080 通常走 PCIe，Graph 不会消除通信 |
| Ada Lovelace | RTX 4090、L40/L40S | 同样可执行 | 4090 属于 Ada，不是 Blackwell |
| Hopper | H100/H200 | Graph、NCCL 捕获及数据中心 Profile | 与 FP8/TMA 是独立能力，不应混为一谈 |
| Blackwell | RTX 5090、B100/B200/GB200 | Graph 与更新工具链 | FP4、NVLink 5 等需要真实支持硬件 |

CUDA Graph 不是 Hopper/Blackwell 专属功能。无对应架构时，Ampere/Ada 仍能完整验证 Capture、Replay 和固定地址模型；但不能用 3080 模拟 NVLink/NVSwitch、FP8/FP4 或 GB200 系统级行为。

## 十五、与 torch.compile 的关系

`torch.compile` 和 CUDA Graph 位于不同层：

```
torch.compile：捕获高层程序 → 融合/调度/生成 Kernel
CUDA Graph：捕获已确定的 CUDA 工作序列 → 低开销重放
```

`torch.compile(mode="reduce-overhead")` 可能在合适条件下使用 CUDA Graph 相关优化，但具体策略依赖 PyTorch、TorchInductor、设备和程序可捕获性，不能把模式名理解为“必定使用完整 CUDA Graph”。

选型建议：

- 算子多且能融合：先测 `torch.compile` ；
- Kernel 已经很好但 Launch 碎片多：重点测 CUDA Graph；
- Shape 高度动态：先做 Bucket 或局部 Capture；
- 自定义 Runtime、拓扑稳定：可直接使用原生 Graph API；
- 两者都合适：编译后先 Warmup，再捕获稳定执行序列。

## 十六、多卡与 NCCL Graph Capture

NCCL 2.9 起支持把 Collective、P2P 和 Group 操作捕获到 CUDA Graph，要求至少 CUDA 11.3。工程上还必须满足：

- 所有参与 rank 对同一 Collective 的“是否捕获”保持一致；
- Replay 也是 Collective 行为，相关 rank 必须按相同顺序启动对应 Graph；
- 一进程多 GPU 的 Graph Launch 可能出现死锁风险，官方更推荐一进程一 GPU；
- 多 Communicator、Captured/Uncaptured 混合需要按当前 NCCL 文档验证；
- Capture 前完成 Communicator 和算法路径 Warmup。

双 RTX 3080 的教学实验可以分别做单卡 Graph，再用 DDP 比较端到端收益。不要因为单卡 Launch Time 降低，就推断 All-Reduce 已被优化；需要同时看每个 rank 的时间线。

## 十七、常见错误与排查

### 17.1 Capture 报不支持或非法操作

检查 Capture 内是否有 `.item()` 、同步、CPU I/O、动态分支、首次库初始化或第三方不支持操作。先缩小到最小可捕获区域，再逐段增加。

### 17.2 Replay 输出一直不变

Graph 读取静态地址。若没有把新数据 `copy_` 到 `static_input` ，它会反复处理捕获时的旧数据。也要避免把 `static_output` 的引用丢失。

### 17.3 Replay 后出现非法内存访问

可能是静态输入/输出或捕获期 Tensor 已被释放，或者相关内存被其他 Tensor 重用。确保 Graph 及其静态 Buffer 生命周期一致；新版本 PyTorch 可在适当 API 中启用输入存活检查辅助定位，但接口属于 Beta，应按当前文档使用。

### 17.4 Graph 显存明显增加

检查 Graph 私有池、Graph 数量和 Bucket 数量。不要用 `empty_cache()` 试图破坏地址稳定；应减少并存 Graph、缩小捕获范围，或在生命周期安全时共享 Pool。

### 17.5 Graph 没有加速

可能是长 GEMM/Attention 主导、输入 Copy 主导、Kernel 数太少、CPU 原本能提前提交，或测量中每次都同步。先看 Nsight Systems，再比较 CPU Submit、GPU Event 和端到端三类指标。

### 17.6 动态 Shape 无法捕获

用 Shape Bucket；低频 Shape 走 Eager；高频 Shape 达到阈值后再 Capture。原生 Graph Update 只适合拓扑和资源约束允许的变化，并非任意动态模型的替代品。

### 17.7 随机数或训练结果异常

验证 Dropout、Generator、Optimizer、GradScaler 与梯度 Buffer 的状态语义。训练 Graph 比推理更复杂，优先从 Forward 或重复 Block 的 Partial Capture 开始。

### 17.8 Nsight 中只看到一个 Graph 节点

这是 `--cuda-graph-trace=graph` 的正常表现。需要内部节点时改为 `node` ，但 Profile 扰动更高，采集窗口应更短。

## 十八、优化前后对照

| 优化前 | 优化后 | 主要收益 |
| --- | --- | --- |
| 每个 Kernel 单独提交 | Graph 一次重放多个 Node | 降低 CPU Launch 压力 |
| 每次创建输入 Tensor | 预分配静态 Buffer + `copy_` | 保持地址稳定 |
| Capture 内做 I/O/`.item()` | CPU 逻辑移到 Graph 外 | 避免同步与 Capture 失败 |
| 任意 Shape 各自捕获 | 高频有限 Bucket | 控制缓存和私有池 |
| 首次执行直接 Capture | Side Stream Warmup 后捕获 | 避免 Lazy 初始化 |
| 只看 GPU 利用率 | 看 CPU API、空洞、Submit 与设备时间 | 确认是否 Launch-bound |
| 每 Step `synchronize()` | 仅在必要边界使用 Event/同步 | 保留异步流水线 |
| Graph 代替所有优化 | Graph + Fusion + Batching + Overlap | 优化真实关键路径 |

## 十九、面试题与答案

### 题1：CUDA Graph 主要优化什么？

它降低重复 CUDA 工作序列的 Host 提交和 Kernel Launch 开销，使多个节点能通过一次 Graph Launch 重放；单个 Kernel 的计算复杂度不会因此自动下降。

### 题2：为什么 Graph 需要固定内存地址？

实例化后的节点保存了捕获时的参数和虚拟地址。Replay 按这些地址读写，因此框架通常保留静态输入输出和 Graph 私有内存池。

### 题3：新输入如何送入已捕获的 PyTorch Graph？

将新数据异步或同步复制到长期存活的 `static_input` ，调用 `graph.replay()` ，再读取长期存活的 `static_output` 。

### 题4：Graph 与 Kernel Fusion 有何区别？

Fusion 减少 Kernel 数和中间显存流量；Graph 降低一串现有 Kernel 的提交开销。Fusion 可能改变计算图，Graph 主要改变提交方式。

### 题5：为什么 Capture 前要 Warmup？

首次执行可能初始化 CUDA Context、库 Handle、算法、编译代码和内存池，这些动作不适合出现在 Capture 热路径，甚至可能导致捕获失败。

### 题6：为什么动态 Shape 是难点？

Shape 影响 Kernel 参数、内存大小和拓扑。捕获图通常假设固定 Shape/布局，需要 Bucket、多图缓存或受限 Graph Update。

### 题7：如何证明程序是 Launch-bound？

Nsight Systems 中大量短 Kernel 之间有空洞，CPU CUDA API 提交持续繁忙，GPU 等待 Host；同时 Graph 显著降低 CPU Submit Time，才构成较完整证据。

### 题8：Graph 的 Capture 成本如何纳入决策？

分开测 Capture/Instantiate 与稳态，使用 $N^*=T_c/($  $T_e-T_g$) 估算盈亏平衡，并乘上实际 Bucket/Graph 数量。

### 题9：NCCL Graph Capture 的关键一致性要求是什么？

参与同一通信的 rank 必须一致地捕获或不捕获，并以一致顺序 Replay 相应 Graph；Graph 中的通信启动仍是 Collective 行为。

### 题10：什么时候不应优先使用 CUDA Graph？

Shape/控制流高度动态、I/O 与数据加载主导、只有少量长 Kernel、实例生命周期太短，或显存无法承受多个私有池时。

## 二十、课后练习

1. 修改 Level 0 的 Node 数和计算量，画出 Speedup 热力图。
2. 在 Level 1 中加入一次 `.item()` ，观察 Capture 的报错位置。
1. 为 Batch 8/16/32 各捕获一个 Graph，统计总 Capture 时间和显存。
2. 比较 Replay-only 与 Copy+Replay，计算输入搬运吃掉的收益比例。
1. 用 Nsight Systems 的 `graph` 与 `node` 粒度分别采集，比较采集开销。
2. 将 Level 1 改为一个长 GEMM，解释为什么 Speedup 变化。
1. 在双 RTX 3080 上分别运行单卡实验，记录驱动、PyTorch、时钟和结果差异。
2. 给原生 CUDA 实验增加 `cudaGraphExecUpdate()` ，更新 Kernel 参数并验证输出。

## 二十一、Checklist

### 适用性

- 热路径重复、Kernel 数量多且单 Kernel 较短。
- Shape、dtype、布局和控制流可稳定或可 Bucket。
- 实例生命周期足以摊平 Capture/Instantiate 成本。
- Graph 目标不是掩盖 DataLoader、通信或长 Kernel 瓶颈。

### 捕获

- Capture 前在 Side Stream 完成 Warmup。
- Capture 内没有.item()、CPU I/O 或非法同步。
- 静态输入、输出和 Graph 对象保持长期存活。
- 随机数、Optimizer、GradScaler 和 Mutation 语义已验证。
- 多 Stream 依赖从捕获 Stream 分叉并在结束前汇合。

### 测量

- Capture/Instantiate 与稳态分开计时。
- CPU Submit、GPU Event、Copy+Replay、端到端分别报告。
- 无 Profile、多次重复的结果用于最终结论。
- 输出正确性和显存变化与性能一起验收。
- 性能数字仅归属于本次环境。

### 生产与多卡

- 高频 Shape 有限 Bucket，低频 Shape 有安全回退。
- Graph 数量、私有池和缓存淘汰策略受控。
- NCCL 捕获在所有 rank 上保持一致。
- 一进程一 GPU 为多卡首选基线。
- 4090 标记为 Ada，5090 标记为 Blackwell。
- NVLink/NVSwitch、FP8/FP4 不做无硬件等价模拟。