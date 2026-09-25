---
title: "第10课：CUDA 编程模型"
source: "https://everythingai.top/learn/ai-systems-performance-engineering-text/lesson-10"
author:
published:
created: 2026-09-26
description: "面向 AI 工程学习者的专业课程学习空间。"
tags:
  - "clippings"
---
**导航**

本课建立 CUDA 编程模型的完整认识，掌握线程组织、内存空间、Stream、Event 与异步执行。

## 课程定位

PyTorch、Triton、cuBLAS 和 vLLM 最终都要把工作提交给 GPU。理解 CUDA 编程模型，不是为了把所有算子重写成 CUDA C++，而是为了看懂 Kernel 为什么这样划分线程、为什么一次 Launch 没有立刻完成、为什么 `cudaDeviceSynchronize()` 会让程序突然变慢，以及为什么同一段源码在不同 GPU 上会选择不同指令。

本课从 CPU 与 GPU 的异构协作讲起，建立 Grid、Block、Thread、Warp、Stream、Event、Memory Space、Synchronization、PTX 与 SASS 的完整心智模型。下一阶段再专门深入 Memory Access、Profiling、Fusion 和 Triton。

## 学习目标

完成本课后，你能够：

1. 解释 Host、Device、Kernel 与 CUDA Runtime 的关系。
2. 使用 Grid、Block、Thread 把一维和二维问题映射到 GPU。
1. 理解 Warp/SIMT、分支发散、Predication 与独立线程调度的边界。
2. 区分 Global、Shared、Local、Constant、Register 等 Memory Space。
1. 正确使用 Kernel Launch、Stream、Event 与同步 API。
2. 解释异步 API 为什么不等于一定并发或一定重叠。
1. 编写带错误检查、计时、正确性验证和 Block Size 扫描的 CUDA 程序。
2. 区分源码、PTX、Cubin/Fatbin、SASS 以及 Compute Capability。
1. 为 Ampere、Ada、Hopper、Blackwell 选择正确编译目标与回退路径。

## 前置知识

- 熟悉 C/C++ 数组、指针、函数和编译命令。
- 理解 GPU 的 SM、Warp、Register、Shared Memory 和显存。
- 理解 Latency、Throughput、Roofline 与有效带宽。
- Level 1 需要 NVIDIA GPU、Driver 和 CUDA Toolkit；Level 0 只需 Python。

## 核心直觉：CPU 发任务，GPU 批量执行

CUDA 是异构编程模型：

```
Host / CPU
  ├─ 分配 Host/Device Memory
  ├─ 准备数据
  ├─ 把工作加入 CUDA Stream
  ├─ 发起 Kernel Launch / Memory Copy
  └─ 在需要结果时同步

Device / GPU
  ├─ SM 接收 Thread Block
  ├─ Warp Scheduler 发射 Warp 指令
  ├─ 线程访问寄存器、Shared Memory、Global Memory
  └─ 完成后由 Event/同步点向 Host 暴露状态
```

CPU 不是逐个遥控 GPU Thread。Host 发出一个 Kernel Launch，描述“运行哪个函数、启动多少 Block、每个 Block 多少 Thread、使用哪个 Stream”。GPU Runtime 与硬件负责把 Block 调度到 SM 上。

因此，CUDA 优化有两条主线：

- \*\*并行分解\*\*：有没有足够独立的工作填满 GPU；
- \*\*数据与依赖\*\*：线程需要的数据在哪里，哪些操作必须等待。

## CUDA 程序的生命周期

一个典型 CUDA C++ 程序包含：

```
// 1. Host 分配并初始化数据
cudaMalloc(&device_ptr, bytes);

// 2. Host -> Device
cudaMemcpy(device_ptr, host_ptr, bytes, cudaMemcpyHostToDevice);

// 3. 启动 Kernel
kernel<<<grid_dim, block_dim>>>(device_ptr, n);

// 4. 检查 Launch，并等待需要的结果
cudaGetLastError();
cudaDeviceSynchronize();

// 5. Device -> Host，验证并释放
cudaMemcpy(host_ptr, device_ptr, bytes, cudaMemcpyDeviceToHost);
cudaFree(device_ptr);
```

CUDA Kernel 启动语法不是普通 C++ 函数调用。它配置并提交 GPU Kernel；默认情况下，Host 会继续执行，异步错误可能到后续同步 API 才暴露。

## 函数执行空间

| 限定符 | 调用方 | 执行位置 | 常见用途 |
| --- | --- | --- | --- |
| `__global__` | Host；部分场景可由 Device | Device | Kernel 入口，返回 `void` |
| `__device__` | Device | Device | GPU 辅助函数 |
| `__host__` | Host | Host | 普通 CPU 函数 |
| `__host__ __device__` | Host/Device | 分别编译 | 简单通用数学函数 |

`__host__ __device__` 会生成两个版本，不能因此调用任意 Host-only 标准库或操作系统 API。要检查 Host 与 Device 两条编译路径。

## Thread Hierarchy

### Grid、Block、Thread

```
Kernel Launch
└─ Grid
   ├─ Block 0
   │  ├─ Thread 0
   │  ├─ Thread 1
   │  └─ ...
   ├─ Block 1
   └─ ...
```

CUDA 提供内置坐标：

- `threadIdx.{x,y,z}` ：线程在 Block 内的位置；
- `blockIdx.{x,y,z}` ：Block 在 Grid 内的位置；
- `blockDim.{x,y,z}` ：每个 Block 的形状；
- `gridDim.{x,y,z}` ：Grid 的形状。

一维全局索引：

$$
i=blockIdx.x\times blockDim.x+threadIdx.x
$$

二维矩阵索引：

$$
row=blockIdx.y\times blockDim.y+threadIdx.y
$$

$$
col=blockIdx.x\times blockDim.x+threadIdx.x
$$

### Ceiling Division

若元素数为 N、每 Block 有 T 个线程，需要：

$$
B=\left\lceil\frac{N}{T}\right\rceil=\frac{N+T-1}{T}
$$

由于最后一个 Block 可能越界，Kernel 需要：

```
if (i < n) {
    out[i] = x[i] + y[i];
}
```

### Grid-Stride Loop

通用一维 Kernel 常使用：

```
for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
     i < n;
     i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    out[i] = x[i] + y[i];
}
```

优点：

- Grid 大小不必与问题规模一一对应；
- 同一 Thread 可处理多个元素；
- 便于控制 Block 数、测试长期驻留工作；
- 对超大数组使用 `size_t` 可避免 32-bit Index Overflow。

### Block 是调度和协作单位

同一 Block 的线程：

- 在同一 SM 上执行；
- 可以使用 Shared Memory 交换数据；
- 可以通过 `__syncthreads()` 做 Block 范围 Barrier；
- 共享有限的 Register File、Shared Memory 与 Resident Thread 配额。

普通 CUDA 模型不能假设不同 Block 同时驻留，也不能在一个普通 Kernel 中用 `__syncthreads()` 实现 Grid-wide Barrier。跨 Block 全局同步通常需要拆成多个 Kernel、Cooperative Launch 或其他专用机制。

## Warp 与 SIMT

当前 NVIDIA GPU 的 Warp 通常由 32 个线程组成。Scheduler 以 Warp 为基本发射单位，但每个 Thread 保留自己的 Register State 和逻辑执行路径。

### 分支发散

```
if (threadIdx.x % 2 == 0) {
    path_a();
} else {
    path_b();
}
```

同一 Warp 中线程走不同分支时，硬件需要执行不同路径并屏蔽不参与的 Lane。简化效率模型：

$$
\eta_{branch}\approx\frac{\text{active lane cycles}}{32\times\text{issued warp cycles}}
$$

但“代码里有 if”不等于一定很慢：

- 编译器可能使用 Predication；
- 分支可能在 Warp 间一致；
- 分支体很短；
- Kernel 可能由内存或其他瓶颈主导。

必须用 Profiler 验证，不要只看源码猜。

### 独立线程调度的边界

Volta 及以后支持更细粒度的 Independent Thread Scheduling，但这不意味着 Warp 内线程可以忽略同步。Warp-level Intrinsic 应使用带 Active Mask 的 `_sync` 版本，并保证参与线程集合正确。

```
unsigned mask = __activemask();
int value_from_lane0 = __shfl_sync(mask, value, 0);
```

依赖“同一 Warp 自然同时执行”而没有正确 Primitive 的旧代码，可能产生竞态。

## Memory Space

| 空间 | 典型范围 | 生命周期 | 直觉 |
| --- | --- | --- | --- |
| Register | 单 Thread | Thread | 最快、容量有限 |
| Local Memory | 单 Thread 语义 | Thread | 名为 Local，物理上常落到 Device Memory |
| Shared Memory | 单 Block | Block | 程序员管理的片上共享区 |
| Global Memory | Grid/Host 可管理 | Allocation | 大容量显存，延迟较高 |
| Constant Memory | Grid，只读 | Module/Context | 小型广播常量 |
| Texture/Read-only Path | Grid，只读语义 | Allocation | 特定缓存与寻址行为 |

两个重要误区：

1. C++ 局部变量不保证一定在 Register；Register Pressure 过高可能 Spill 到 Local Memory。
2. `cudaMallocManaged` 不等于“数据永远同时在 CPU 与 GPU”。Unified Memory 仍有迁移、一致性、Page Fault 和 Prefetch 成本。

本课只建立空间模型，下一课专门分析 Coalescing、Shared Memory Bank、Tiling、Vectorized Access 与异步搬运。

## 同步与内存可见性

### Block 内 Barrier

`__syncthreads()` 的语义是 Block 内参与线程等待，并建立对应 Shared/Global Memory 可见性边界。它必须被 Block 内所有活动线程以一致方式到达。

危险代码：

```
if (threadIdx.x < 16) {
    __syncthreads(); // 其他线程不进入，可能死锁或产生未定义行为
}
```

### Barrier 不等于 Atomic

Barrier 只协调到达和可见性，不能把多个线程的 Read-Modify-Write 自动变成原子操作。计数器需要 `atomicAdd` 或正确的层级归约。

### Host 同步粒度

| API | 等待范围 |
| --- | --- |
| `cudaDeviceSynchronize()` | 当前 Device 的先前工作 |
| `cudaStreamSynchronize(s)` | 指定 Stream 的先前工作 |
| `cudaEventSynchronize(e)` | 指定 Event 完成 |
| `cudaStreamWaitEvent(s,e)` | Device-side 依赖，不阻塞 Host Thread |
| `cudaStreamQuery(s)` | 非阻塞查询 Stream 状态 |

同步越粗，越容易正确，也越容易破坏并发。性能工程目标是使用满足依赖的最小同步范围。

## Stream 与异步执行

Stream 是有序工作队列。同一 Stream 中操作按提交顺序执行；不同 Stream 的独立工作\*\*可能\*\*并发。

```
Stream 0: H2D A → Kernel A → D2H A
Stream 1: H2D B → Kernel B → D2H B
```

实际能否重叠取决于：

- GPU 是否有相应 Copy Engine；
- Host Buffer 是否 Page-locked；
- Kernel 是否占满全部 SM/Memory Bandwidth；
- 两项工作是否有真实依赖；
- 默认 Stream 和其他隐式同步；
- Device Allocation、Page-locked Allocation 等同步行为；
- 消息是否大到足以摊薄 Launch 开销。

所以：

### 默认 Stream

Legacy Default Stream 可能与其他 Blocking Stream 发生隐式同步。可创建 `cudaStreamNonBlocking` Stream，或在明确理解语义后使用 Per-thread Default Stream。不要混用后再根据“看起来应该并发”推断结果。

## Event 与计时

CUDA Event 在 Device 时间线上记录位置，适合测量 GPU 工作：

```
cudaEventRecord(start);
kernel<<<grid, block>>>(...);
cudaEventRecord(stop);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&ms, start, stop);
```

Host `std::chrono` 只包住异步 Kernel Launch 时，测到的主要是提交开销。要测 End-to-end，必须在结束前同步；要测单个 Kernel，优先使用正确 Stream 上的 Event。

计时前应 Warmup，避免把 Context 初始化、JIT、Cache Cold Start 和 Frequency Ramp 混入稳定态。

## 错误处理

CUDA 错误分两类：

- \*\*同步 API 错误\*\*：例如 `cudaMalloc` 立即失败；
- \*\*异步执行错误\*\*：例如 Illegal Memory Access，常在同步或后续 API 才报告。

必须同时检查：

```
kernel<<<grid, block>>>(...);
CUDA_CHECK(cudaGetLastError());      // Launch 配置等立即错误
CUDA_CHECK(cudaDeviceSynchronize()); // 执行期错误
```

Benchmark 中每轮都 `cudaDeviceSynchronize()` 会扭曲性能；调试时可严格同步，性能测量时用 Event/Stream 边界并在合理位置检查。

`CUDA_LAUNCH_BLOCKING=1` 适合定位异步报错位置，不适合当作生产性能配置。

## 编译模型：Source、PTX、Cubin 与 SASS

```
CUDA C++ Source
   ├─ Host Compiler → Host Object
   └─ NVCC Device Frontend
        ├─ PTX：虚拟 ISA，可由 Driver JIT
        └─ Cubin：目标架构机器码容器
             └─ SASS：GPU 实际执行指令
```

### Compute Capability

| 架构/产品示例 | 常见目标 |
| --- | --- |
| Ampere A100 | `sm_80` |
| Ampere RTX 3080/3090 | `sm_86` |
| Ada RTX 4090、L40/L40S | `sm_89` |
| Hopper H100/H200 | `sm_90` |
| Blackwell B100/B200/GB200 | `sm_100` ，依 Toolkit/SDK 具体目标 |
| Blackwell RTX 5090 | `sm_120` |

编译双 RTX 3080：

```
nvcc -O3 -std=c++17 -lineinfo -arch=sm_86 cuda_model_lab.cu -o cuda_model_lab
```

通用 Fat Binary 示例：

```
nvcc -O3 -std=c++17 -lineinfo \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_86,code=sm_86 \
  -gencode arch=compute_89,code=sm_89 \
  -gencode arch=compute_90,code=sm_90 \
  -gencode arch=compute_90,code=compute_90 \
  cuda_model_lab.cu -o cuda_model_lab
```

最后一项保留 PTX，允许 Driver 在受支持的更新 GPU 上 JIT。注意：

- 新架构 Cubin 不能反向在旧 GPU 上运行；
- PTX Forward Compatibility 仍受 PTX ISA、Driver JIT 与 Feature 限制；
- `sm_100/sm_120` 需要支持 Blackwell 的 Toolkit；旧 Toolkit 不认识新目标；
- 仅保留 PTX 会产生首次 JIT 开销，生产应按部署矩阵生成合适 Fatbin。

查看产物：

```
cuobjdump --list-elf ./cuda_model_lab
cuobjdump --dump-sass ./cuda_model_lab | less
nvdisasm <extracted-cubin> | less
```

## Launch 配置与 Occupancy

Occupancy 定义为 SM 上 Active Warp 与最大可驻留 Warp 的比值：

$$
Occupancy=\frac{Warps_{active}}{Warps_{max}}
$$

它受以下共同限制：

- Threads/Block 与最大 Resident Threads；
- Register/Thread 与 Register Allocation 粒度；
- Shared Memory/Block；
- 最大 Resident Blocks；
- 架构资源限制。

Occupancy 是隐藏延迟的手段，不是最终目标。提高 Occupancy 可能迫使 Register Spill，或让更多 Warp 争抢 Memory Bandwidth。常用起点是 128/256 Threads，并通过正确性与实测扫描。

理论 Block 数不是越大越好。Grid 至少应有足够 Block 覆盖 SM；对 Grid-stride Kernel，可从“SM 数量的若干倍”起步，再实测。

## 性能模型

### Launch 开销与工作粒度

若一个 Kernel 的有效工作时间为 $T_k$ ，Launch/Runtime 固定成本为 $T_l$ ：

$$
\eta_{launch}=\frac{T_k}{T_l+T_k}
$$

微小 Kernel 中 $T_l$ 占比高，Fusion、Batching 或 CUDA Graph 更有效；大 Kernel 中优化重点转向计算与 Memory。

### 吞吐上界

Vector Add 每个元素读取两个 `float` 、写一个 `float` ，最少 Payload 约 12 Byte，计算只有一次加法：

$$
AI=\frac{1\ FLOP}{12\ Byte}\approx0.083\ FLOP/Byte
$$

它通常是 Memory-bound。有效带宽：

$$
B_{eff}=\frac{3N\times sizeof(float)}{T_{kernel}}
$$

这不是显存物理总线的完整流量计数，因为缓存、Write Policy 和测量边界会影响实际事务；它是便于对比的 Payload Bandwidth。

### Amdahl 与 CPU/GPU 协作

若可加速部分比例为 p，GPU 加速倍数为 s：

$$
Speedup=\frac{1}{(1-p)+p/s}
$$

Kernel 快 10 倍但 Host、Copy、同步占一半，端到端最多约 $1/(0.5+0.5/10)=1.82$ 倍。

## 瓶颈分析方法

### 先判断问题在哪个层级

```
正确性失败？
├─ Launch Error → Grid/Block/Shared Memory/参数
├─ Illegal Access → Index、生命周期、异步依赖
├─ Race → 同步范围、Atomic、Warp Mask
└─ 数值误差 → 精度、归约顺序、Fast Math

正确但慢？
├─ Launch 很多且短 → Runtime/Fusion/Graph
├─ GPU 有空洞 → Host/Data/Sync
├─ Kernel Memory-bound → 访问与数据复用
├─ Kernel Compute-bound → 指令/Tensor Core
└─ Occupancy 低 → Register/Shared/Block 限制
```

### 最小诊断命令

```
compute-sanitizer --tool memcheck ./cuda_model_lab
compute-sanitizer --tool racecheck ./cuda_model_lab
nvcc --resource-usage -O3 -arch=sm_86 cuda_model_lab.cu -o cuda_model_lab
```

Profiler 会在第 12 课系统展开；本课先养成“错误检查、Warmup、Event、验证输出、记录编译目标”的习惯。

## Level 0：CPU 回退——模拟线程映射

没有 NVIDIA GPU 时，仍可验证 Index Mapping 和 Grid-stride Loop 是否覆盖每个元素一次。

### 完整代码：cuda\_index\_sim.py

```
#!/usr/bin/env python3
"""在 CPU 上模拟一维 CUDA Grid-stride 索引，不模拟 GPU 性能。"""

from __future__ import annotations

import argparse
import json

def simulate(n: int, blocks: int, threads: int) -> dict:
    if n <= 0 or blocks <= 0 or threads <= 0:
        raise ValueError("n, blocks, threads must all be positive")
    stride = blocks * threads
    owners = [[] for _ in range(n)]
    work_per_thread = []

    for block_idx in range(blocks):
        for thread_idx in range(threads):
            global_idx = block_idx * threads + thread_idx
            count = 0
            i = global_idx
            while i < n:
                owners[i].append((block_idx, thread_idx))
                count += 1
                i += stride
            work_per_thread.append(count)

    missing = [i for i, x in enumerate(owners) if not x]
    duplicated = [i for i, x in enumerate(owners) if len(x) != 1]
    active_threads = sum(x > 0 for x in work_per_thread)
    total_threads = blocks * threads
    return {
        "n": n,
        "blocks": blocks,
        "threads_per_block": threads,
        "logical_threads": total_threads,
        "grid_stride": stride,
        "active_threads": active_threads,
        "tail_idle_threads": total_threads - active_threads,
        "min_items_per_thread": min(work_per_thread),
        "max_items_per_thread": max(work_per_thread),
        "missing_indices": missing,
        "non_unique_indices": duplicated,
        "correct_exactly_once": not missing and not duplicated,
        "first_assignments": {
            str(i): owners[i] for i in range(min(n, 16))
        },
        "warning": "This validates mapping only; it cannot simulate warp scheduling, memory transactions, or performance.",
    }

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1003)
    p.add_argument("--threads", type=int, default=256)
    p.add_argument("--blocks", type=int, default=4)
    args = p.parse_args()
    print(json.dumps(simulate(args.n, args.blocks, args.threads),
                     ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
```

运行：

```
python3 cuda_index_sim.py --n 1003 --blocks 4 --threads 256
python3 cuda_index_sim.py --n 1000000 --blocks 320 --threads 256
```

预期： `correct_exactly_once=true` 。第一组共 1024 个逻辑线程，最后 21 个线程没有元素，这正是边界检查存在的原因。第二组每个线程处理多个元素，体现 Grid-stride Loop。

## Level 1：通用 CUDA C++ 实验

下面程序自动读取设备属性，完成：

- Vector Add 正确性验证；
- Block Size 扫描；
- CUDA Event Kernel 计时；
- 同步 End-to-end 与多 Stream + Pinned Memory 对照；
- API、Launch 与执行期错误检查。

### 完整代码：cuda\_model\_lab.cu

```
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t error__ = (call);                                          \
        if (error__ != cudaSuccess) {                                          \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__      \
                      << " -> " << cudaGetErrorString(error__) << std::endl;   \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

__global__ void vector_add_grid_stride(const float* x, const float* y,
                                       float* out, size_t n) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = start; i < n; i += stride) {
        out[i] = x[i] + y[i];
    }
}

size_t ceil_div(size_t a, size_t b) {
    return (a + b - 1) / b;
}

void verify(const float* x, const float* y, const float* out, size_t n,
            const std::string& label) {
    double max_error = 0.0;
    size_t bad_index = n;
    for (size_t i = 0; i < n; ++i) {
        const double expected = static_cast<double>(x[i]) + y[i];
        if (!std::isfinite(out[i])) {
            max_error = std::numeric_limits<double>::infinity();
            bad_index = i;
            break;
        }
        const double error = std::abs(static_cast<double>(out[i]) - expected);
        if (error > max_error) {
            max_error = error;
            bad_index = i;
        }
    }
    if (max_error > 1e-6) {
        throw std::runtime_error(label + " verification failed at index " +
                                 std::to_string(bad_index));
    }
    std::cout << label << " verification: PASS, max_error=" << max_error << "\n";
}

float benchmark_kernel(const float* dx, const float* dy, float* dout,
                       size_t n, int block_size, int repeats,
                       int multiprocessors) {
    // Grid-stride Kernel：限制为 SM 数量的若干倍，避免超大 Grid 干扰教学比较。
    const size_t needed = ceil_div(n, static_cast<size_t>(block_size));
    const int grid_size = static_cast<int>(
        std::min<size_t>(needed, static_cast<size_t>(multiprocessors) * 32));

    vector_add_grid_stride<<<grid_size, block_size>>>(dx, dy, dout, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{}, stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < repeats; ++r) {
        vector_add_grid_stride<<<grid_size, block_size>>>(dx, dy, dout, n);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return total_ms / repeats;
}

double synchronous_pipeline(const float* hx, const float* hy, float* hout,
                            float* dx, float* dy, float* dout,
                            size_t n, int block, int grid) {
    const size_t bytes = n * sizeof(float);
    const auto begin = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMemcpy(dx, hx, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dy, hy, bytes, cudaMemcpyHostToDevice));
    vector_add_grid_stride<<<grid, block>>>(dx, dy, dout, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(hout, dout, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

double asynchronous_pipeline(const float* hx, const float* hy, float* hout,
                             float* dx, float* dy, float* dout,
                             size_t n, int block, int multiprocessors,
                             int stream_count) {
    std::vector<cudaStream_t> streams(stream_count);
    for (auto& stream : streams) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    }

    const size_t chunk = ceil_div(n, static_cast<size_t>(stream_count));
    const auto begin = std::chrono::steady_clock::now();
    for (int s = 0; s < stream_count; ++s) {
        const size_t offset = static_cast<size_t>(s) * chunk;
        if (offset >= n) break;
        const size_t count = std::min(chunk, n - offset);
        const size_t bytes = count * sizeof(float);
        const int grid = static_cast<int>(std::min<size_t>(
            ceil_div(count, static_cast<size_t>(block)),
            static_cast<size_t>(multiprocessors) * 16));

        CUDA_CHECK(cudaMemcpyAsync(dx + offset, hx + offset, bytes,
                                   cudaMemcpyHostToDevice, streams[s]));
        CUDA_CHECK(cudaMemcpyAsync(dy + offset, hy + offset, bytes,
                                   cudaMemcpyHostToDevice, streams[s]));
        vector_add_grid_stride<<<grid, block, 0, streams[s]>>>(
            dx + offset, dy + offset, dout + offset, count);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(hout + offset, dout + offset, bytes,
                                   cudaMemcpyDeviceToHost, streams[s]));
    }
    for (auto stream : streams) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    const auto end = std::chrono::steady_clock::now();
    for (auto stream : streams) {
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

int main(int argc, char** argv) {
    try {
        const size_t n = argc > 1 ? std::stoull(argv[1]) : (1ULL << 24);
        const int repeats = argc > 2 ? std::stoi(argv[2]) : 100;
        if (n == 0 || repeats <= 0) {
            throw std::invalid_argument("n and repeats must be positive");
        }

        int device = 0;
        CUDA_CHECK(cudaSetDevice(device));
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        std::cout << "device=" << prop.name
                  << " cc=" << prop.major << "." << prop.minor
                  << " SMs=" << prop.multiProcessorCount
                  << " warp=" << prop.warpSize
                  << " max_threads_per_block=" << prop.maxThreadsPerBlock
                  << " async_engines=" << prop.asyncEngineCount << "\n";

        const size_t bytes = n * sizeof(float);
        float *hx = nullptr, *hy = nullptr, *hout = nullptr;
        float *dx = nullptr, *dy = nullptr, *dout = nullptr;
        CUDA_CHECK(cudaMallocHost(&hx, bytes));
        CUDA_CHECK(cudaMallocHost(&hy, bytes));
        CUDA_CHECK(cudaMallocHost(&hout, bytes));
        CUDA_CHECK(cudaMalloc(&dx, bytes));
        CUDA_CHECK(cudaMalloc(&dy, bytes));
        CUDA_CHECK(cudaMalloc(&dout, bytes));

        for (size_t i = 0; i < n; ++i) {
            hx[i] = static_cast<float>(i % 1024) * 0.25f;
            hy[i] = static_cast<float>(i % 257) * -0.5f;
            hout[i] = std::numeric_limits<float>::quiet_NaN();
        }

        CUDA_CHECK(cudaMemcpy(dx, hx, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dy, hy, bytes, cudaMemcpyHostToDevice));

        std::cout << std::fixed << std::setprecision(3);
        for (int block : {64, 128, 256, 512}) {
            if (block > prop.maxThreadsPerBlock) continue;
            const float ms = benchmark_kernel(dx, dy, dout, n, block, repeats,
                                              prop.multiProcessorCount);
            const double payload_gbs = (3.0 * bytes) / (ms / 1000.0) / 1e9;
            std::cout << "block=" << block << " kernel_ms=" << ms
                      << " payload_GBps=" << payload_gbs << "\n";
        }

        const int block = 256;
        const int grid = static_cast<int>(std::min<size_t>(
            ceil_div(n, static_cast<size_t>(block)),
            static_cast<size_t>(prop.multiProcessorCount) * 32));

        const double sync_ms = synchronous_pipeline(
            hx, hy, hout, dx, dy, dout, n, block, grid);
        verify(hx, hy, hout, n, "synchronous");

        std::fill(hout, hout + n, std::numeric_limits<float>::quiet_NaN());
        const double async_ms = asynchronous_pipeline(
            hx, hy, hout, dx, dy, dout, n, block,
            prop.multiProcessorCount, 4);
        verify(hx, hy, hout, n, "four-stream");

        const double pipeline_bytes = 3.0 * bytes;
        std::cout << "sync_end_to_end_ms=" << sync_ms
                  << " payload_GBps=" << pipeline_bytes / (sync_ms / 1000.0) / 1e9
                  << "\n";
        std::cout << "four_stream_end_to_end_ms=" << async_ms
                  << " payload_GBps=" << pipeline_bytes / (async_ms / 1000.0) / 1e9
                  << " speedup=" << sync_ms / async_ms << "\n";
        std::cout << "note=async overlap is hardware/workload dependent; speedup may be <= 1\n";

        CUDA_CHECK(cudaFree(dout));
        CUDA_CHECK(cudaFree(dy));
        CUDA_CHECK(cudaFree(dx));
        CUDA_CHECK(cudaFreeHost(hout));
        CUDA_CHECK(cudaFreeHost(hy));
        CUDA_CHECK(cudaFreeHost(hx));
        CUDA_CHECK(cudaDeviceReset());
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "fatal: " << e.what() << "\n";
        return 1;
    }
}
```

### 编译与运行

自动读取第一块 GPU 的 Compute Capability：

```
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')
nvcc -O3 -std=c++17 -lineinfo -arch="sm_${CC}" \
  cuda_model_lab.cu -o cuda_model_lab

./cuda_model_lab
./cuda_model_lab 33554432 200
```

如果 `compute_cap` 查询字段不受旧 Driver 支持，手工按设备设置 `-arch` 。双 RTX 3080 使用 `-arch=sm_86` 。

### 预期现象

- 输出 GPU 名称、Compute Capability、SM 数、Warp Size 和 Async Engine 数；
- 64/128/256/512 Threads 的 Kernel 时间可能不同；
- 所有路径都应 `verification: PASS` ；
- Vector Add 的 Payload GB/s 远比 FLOP 数更有解释力；
- 四 Stream 不一定快于同步路径，因为 H2D、Kernel、D2H 可能争用 PCIe、Copy Engine 和显存带宽；
- 首次运行可能包含 Context、Clock 或 JIT 影响，需多轮稳定后记录。

### 结果分析

若 256 比 512 快，不能直接归因于“Warp 更多”。512 Threads/Block 可能减少 Resident Block 数、改变 Register/Shared Resource 分配；64 又可能没有足够 Warp 隐藏延迟。需要后续用 Nsight Compute 查看 Occupancy、Memory Throughput 和 Stall。

若多 Stream Speedup 小于 1：

- 数据量可能太小；
- Vector Add 本身占用大量 Memory Bandwidth；
- Copy 与 Kernel 不能充分重叠；
- Host/PCIe 成为瓶颈；
- 调度开销超过收益。

这不是实验失败，而是“异步 API 不保证加速”的真实结果。

## Level 2：架构专项实验

### 能力检测

```
nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv
nvcc --version
cuobjdump --list-elf ./cuda_model_lab
```

| 架构 | 可选专项能力 | 最低学习回退 |
| --- | --- | --- |
| Ampere `sm_80/sm_86` | `cp.async` 、TF32、Ampere Tensor Core | 通用 Kernel/Stream/Event |
| Ada `sm_89` | Ada Tensor Core、改进 Cache/执行能力 | 通用 Kernel + Ampere-compatible 路径 |
| Hopper `sm_90` | Thread Block Cluster、TMA、Warpgroup 能力 | 普通 Block/Shared Memory |
| Blackwell `sm_100/sm_120` | 新 Tensor Core、Cluster/TMA 演进 | 普通 CUDA 编程模型 |

只有对应架构和 Toolkit 才能编译/验证专属指令。可使用条件编译：

```
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    // Hopper+ 专属 Device Code
#else
    // 通用回退
#endif
```

但“能编译一个回退分支”不等于模拟了 TMA、Cluster 或 Blackwell 特性。双 RTX 3080 可以完整运行 Level 1，并在后续 Memory 课程验证 Ampere `cp.async` ；不能验证 Hopper/Blackwell 专属路径。

## 优化前后对照

| 维度 | 常见初版 | 优化方向 | 验证方式 |
| --- | --- | --- | --- |
| 索引 | 每线程只处理一个元素 | Grid-stride Loop | 覆盖与性能 |
| Block | 固定 1024 Threads | 128/256 起步扫描 | Event + Occupancy |
| 同步 | 每次 Launch 后 Device Sync | Event/Stream 最小依赖 | Timeline、端到端时间 |
| Copy | Pageable + 同步 Copy | Pinned + Async + Chunk | H2D/D2H/总时间 |
| 小 Kernel | 大量碎片 Launch | Batch/Fusion/Graph | Launch 占比 |
| 错误 | 不检查，最后随机失败 | API + Launch + 执行边界 | Sanitizer/错误位置 |
| 编译 | 只编一个本机 Cubin | 按部署矩阵构建 Fatbin/PTX | 目标机启动与 JIT |
| 性能结论 | 只报 Kernel ms | 正确性、Payload、E2E、P99 | 可复现实验记录 |

## 常见错误与排查

### 错误 1：invalid device function / no kernel image is available

Fatbin 不包含当前 GPU 可执行 Cubin，也没有可 JIT 的兼容 PTX。检查 `-gencode` 、 `cuobjdump --list-elf` 、GPU CC 与 Driver。

### 错误 2：invalid configuration argument

Block Threads、Dynamic Shared Memory、Grid Dimension 或 Launch 属性超出设备限制。查看 `cudaGetDeviceProperties` ，Launch 后立即 `cudaGetLastError()` 。

### 错误 3：illegal memory access

检查越界、Index Overflow、已释放指针、异步生命周期和错误 Stream 依赖。使用：

```
compute-sanitizer --tool memcheck ./cuda_model_lab
```

### 错误 4：Kernel 计时几乎为 0

Host Timer 只测到了异步提交。使用 CUDA Event，或在 End-to-end Timer 末尾同步。

### 错误 5：cudaMemcpyAsync 没有重叠

Pageable Host Memory 可能阻碍真正异步传输；还要检查 Copy Engine、Stream、依赖和资源争用。使用 `cudaMallocHost` / `cudaHostAlloc` 并通过 Timeline 验证。

### 错误 6：在条件分支中调用 \_\_syncthreads()

如果 Block 内不是所有活动 Thread 一致到达 Barrier，行为未定义并可能 Hang。重构控制流或使用正确的更小范围 Primitive。

### 错误 7：把 Local Memory 当片上快速内存

CUDA 的 Local Memory 是 Thread-private 地址空间，常存放到 Device Memory。检查编译资源和 Local Load/Store。

### 错误 8：Occupancy 100% 但更慢

可能 Register 被限制而 Spill、Memory Bandwidth 饱和、更多 Warp 争用 Cache。Occupancy 不是 Goodput。

### 错误 9：只构建 sm\_120，想在 RTX 3080 上运行

新架构机器码不能反向运行在旧架构。为每个部署目标生成对应 Cubin，或保留兼容 PTX 用于更新设备。

### 错误 10：使用 CUDA\_LAUNCH\_BLOCKING=1 后性能结论失真

它会把异步 Launch 变得便于调试，但破坏正常重叠。只在定位错误时临时使用。

### 错误 11：异步 Copy 后立即复用 Host Buffer

Stream 尚未完成时修改或释放 Buffer 会产生竞态。用 Event/Stream Sync 管理生命周期。

### 错误 12：假设不同 Block 能直接同步

普通 Kernel 没有 `__syncthreads_grid()` 。拆分 Kernel、使用 Cooperative Groups 的合法 Grid Sync，或重新设计算法。

## 面试题与答案

### 1\. CUDA 中 Grid、Block、Thread 分别是什么？

Grid 是一次 Kernel Launch 的全部线程集合；Grid 由 Block 组成；Block 是被调度到单个 SM、能共享 Shared Memory 并进行 Block 内同步的线程组。

### 2\. 为什么 Block 之间不能依赖执行顺序？

Runtime 可以按任意顺序把 Block 调度到 SM，且不保证全部 Block 同时驻留。普通 Kernel 中跨 Block 自旋等待可能死锁。

### 3\. Warp 与 Block 的区别？

Block 是编程与资源分配/协作单位；Warp 是硬件常用的指令发射和执行分组，当前通常 32 Threads。

### 4\. Grid-stride Loop 有什么好处？

让固定数量线程覆盖任意规模数据，控制 Grid 大小、复用线程、适应超大数组，并保持连续线程访问连续元素的模式。

### 5\. 为什么 Kernel Launch 后要检查两次错误？

`cudaGetLastError()` 捕获 Launch 配置等立即错误；Kernel 执行中的 Illegal Access 等异步错误通常在同步边界暴露。

### 6\. Stream 能保证并发吗？

不能。同一 Stream 保序，不同 Stream只是表达潜在并发；实际并发取决于硬件资源、依赖、Memory/Compute 争用和同步语义。

### 7\. CUDA Event 和 CPU Timer 的区别？

Event 位于 Device 时间线，适合测 GPU 工作；CPU Timer 测 Host 经过时间，若不同步可能只测 Launch。End-to-end 应包含所需同步。

### 8\. \_\_syncthreads() 能做什么？

它是 Block 范围 Barrier，并提供相应内存可见性语义；不能同步其他 Block，也不能替代 Atomic。

### 9\. Occupancy 越高越好吗？

不是。足够 Occupancy 能隐藏延迟，但追求更高可能增加 Register Spill 或资源争用。最终看 Kernel 与端到端性能。

### 10\. PTX 和 SASS 有什么区别？

PTX 是虚拟 ISA，可由 Driver JIT；SASS 是目标 GPU 实际执行的机器指令。Cubin/Fatbin 可包含目标机器码与/或 PTX。

### 11\. cudaMallocManaged 是否消除了数据搬运？

没有。它统一地址和迁移管理，但数据仍可能发生 Page Migration、Fault、Prefetch 与一致性成本。

### 12\. 为什么 Vector Add 通常是 Memory-bound？

每元素只做一次加法，却至少搬运两个输入和一个输出，算术强度约 1/12 FLOP/Byte，远低于现代 GPU 的 Compute/Memory 平衡点。

## 课后练习

1. 把 Level 0 改成二维矩阵索引模拟，验证每个 `(row,col)` 恰好覆盖一次。
2. 给 CUDA 程序增加 32、96、192、320 Threads/Block 扫描，解释非 32 倍数结果。
1. 将 Grid 从 `SM×1` 扫描到 `SM×64` ，找出稳定区间。
2. 故意删除 `i<n` ，用 Compute Sanitizer 找到越界。
1. 把同步 Copy 换成 Pageable/Pinned Host Memory 对照。
2. 用两个 Event 建立 Stream 间依赖，禁止 Host 端全局同步。
1. 生成只含 `sm_86` 和含 `sm_86+compute_86` 的两个 Binary，对比大小和首次运行。
2. 使用 `nvcc --resource-usage` 记录不同 Block 配置的 Register/Shared Memory。
1. 把 Vector Add 改成 SAXPY： `out=a*x+y` ，重新计算算术强度。
2. 设计一个会分支发散的 Kernel，再把条件改成 Warp-uniform，后续用 Profiler 验证。

## Checklist

### 编程模型

- 能解释 Host、Device、Kernel 与 Runtime。
- 能写一维/二维 Index 与 Ceiling Division。
- 能使用 Grid-stride Loop。
- 不假设 Block 执行顺序或同时驻留。
- 能区分 Block 与 Warp。

### 正确性

- 所有 CUDA API 都检查返回值。
- Kernel Launch 后检查立即错误。
- 在合适同步边界检查异步错误。
- 输出与 CPU/Reference 做正确性验证。
- 使用 Sanitizer 排查越界和竞态。

### 异步执行

- 能区分异步、并发与重叠。
- 理解同 Stream 保序和跨 Stream 依赖。
- Pinned Buffer 生命周期覆盖异步 Copy。
- 没有滥用 cudaDeviceSynchronize()。
- 使用 Event 测 Kernel、Host Timer 测端到端。

### 性能

- 已 Warmup 并多轮测量。
- Block/Grid 通过扫描而非猜测选择。
- Occupancy 只作为诊断指标。
- 同时报告 Kernel、Copy、End-to-end 与正确性。
- Vector Add 使用 Payload GB/s 而非虚构 FLOPS 提升。

### 编译与硬件

- 记录 NVCC、Driver、GPU CC 和 -gencode。
- 区分 PTX、Cubin/Fatbin 与 SASS。
- 不把新架构机器码用于旧 GPU。
- RTX 3080/3090 使用 sm\\\_86，RTX 4090/L40S 使用 sm\\\_89。
- H100/H200 使用 sm\\\_90，Blackwell 目标按当前 Toolkit/产品确认。
- 没有用软件回退冒充 TMA、Cluster 或 Blackwell 专属实验。