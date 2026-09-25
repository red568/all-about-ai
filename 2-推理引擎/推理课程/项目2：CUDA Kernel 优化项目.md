---
title: "项目2：CUDA Kernel 优化项目"
source: "https://everythingai.top/learn/ai-systems-performance-engineering-text/project-02"
author:
published:
created: 2026-09-26
description: "面向 AI 工程学习者的专业课程学习空间。"
tags:
  - "clippings"
---
**导航**

本项目通过 CUDA Kernel 基线、剖析、改写和回归验证，完成一次端到端算子优化。

## 项目定位

本项目要求你从一个“结果正确但性能很差”的 CUDA Reduction Kernel 出发，用 Profile-Driven 方法完成两轮优化：

```
Baseline：每个元素执行一次全局 atomicAdd
    ↓ 消除大部分全局原子争用
V1：线程局部累加 + Shared Memory Block Reduction + 每 Block 一次 atomicAdd
    ↓ 减少 Shared Memory 访问与同步
V2：线程局部累加 + Warp Shuffle + 每 Block 一次 atomicAdd
```

最终交付的不只是“更快的代码”，而是一套完整证据：相同输入、相同正确性标准、稳定计时、不同规模扫描、Nsight Systems 时间线、Nsight Compute Kernel 指标，以及清楚的结论边界。

本项目适配常见 NVIDIA GPU。Level 0 不需要 CUDA；Level 1 使用标准 CUDA C++，适合 Ampere RTX 3080/3090、A100，Ada RTX 4090/L40/L40S，Hopper H100/H200，以及工具链支持的 Blackwell RTX 5090、B100/B200/GB200。Level 2 的架构专项实验必须先检测能力，不允许把低代 GPU 的模拟结果写成 FP8、FP4、TMA 或 Cluster 的硬件实测。

## 学习目标

完成项目后，你应该能够：

- 从算法工作量、内存流量和同步次数解释 Kernel 为什么慢；
- 正确使用 Grid-Stride Loop、Shared Memory、 `__syncthreads()` 和 Warp Shuffle；
- 区分全局原子争用、Shared Memory Bank Conflict、Occupancy 与指令依赖；
- 使用 CUDA Event 测量设备执行时间；
- 用 Nsight Systems 找到 Kernel 边界，用 Nsight Compute 验证优化原因；
- 在改变实现后同时验证正确性、P50/P95、跨尺寸稳定性与架构可移植性；
- 避免用峰值带宽、Occupancy 或一次运行结果替代真实证据。

## 前置知识

- 已完成第 10～13 课；
- 理解 Thread、Warp、Block、Grid 与 Grid-Stride Loop；
- 理解 Global Memory、Shared Memory、原子操作和同步；
- Ubuntu 或其他支持 CUDA Toolkit 的系统；
- 可选 NVIDIA GPU；没有 GPU 时先完成 Level 0。

## 项目交付物

```
cuda-kernel-project/
├── reduction_cost_model.py
├── reduction_benchmark.cu
├── reports/
│   ├── environment.txt
│   ├── baseline.csv
│   ├── size_sweep.csv
│   ├── reduction_nsys.nsys-rep
│   └── reduction_ncu.ncu-rep
└── analysis.md
```

`analysis.md` 至少回答：

1. Baseline 最主要的瓶颈是什么？
2. V1 消除了什么成本，又引入了什么成本？
1. V2 为什么可能继续变快，也可能没有明显收益？
2. 哪些结论来自实测，哪些只是模型推断？
1. 结论适用于哪些输入规模、Block Size 与 GPU？

## 核心直觉：先减少昂贵协作，再优化局部细节

Reduction 的数学表达很简单：

$$
y = \sum_{i=0}^{N-1} x_i
$$

但 GPU 实现的关键不是加法本身，而是“数百万个线程如何合并局部结果”。

最朴素的实现让每个元素都竞争同一个全局地址：

$$
N_{atomic}^{baseline}=N
$$

如果每个 Block 先在本地归约，只让 Block Leader 更新全局结果：

$$
N_{atomic}^{optimized}=N_{blocks}
$$

原子操作数量的理论缩减比约为：

$$
R_{atomic}=\frac{N}{N_{blocks}}
$$

这不等于最终加速比。优化版还要付出线程局部累加、Shared Memory、同步和归约指令的成本；Baseline 的原子实现也可能被缓存、硬件原子单元和输入规模影响。真实收益必须测量。

## 性能模型

### 最小数据搬运量

若输入为 FP32，至少要读取：

$$
Bytes_{read}=4N
$$

输出只有一个 FP32，但原子读改写会产生额外流量与序列化，不能只按 4 字节输出估算物理事务。

有效输入带宽可写成：

$$
BW_{effective}=\frac{4N}{t}
$$

它用于比较同一算法不同实现，不代表硬件实际 DRAM 总流量。

### 算术强度

Reduction 每读取一个 FP32 大约做一次加法：

$$
AI \approx \frac{N}{4N}=0.25\ FLOP/Byte
$$

它通常属于低算术强度工作负载，但 Baseline 可能先受全局原子争用限制，而不是先触达 DRAM 带宽屋脊。

### Amdahl 定律

若 Reduction 只占端到端程序的比例为 p，即使 Kernel 加速 $S_k$ 倍，端到端加速上界仍是：

$$
S_{total}=\frac{1}{(1-p)+\frac{p}{S_k}}
$$

因此项目必须同时报告 Kernel 加速和它在真实应用中的时间占比。

## 瓶颈分析顺序

| 层次 | 问题 | 证据 |
| --- | --- | --- |
| 正确性 | 三个 Kernel 是否得到一致结果 | CPU Reference、误差 |
| 墙钟 | 优化是否稳定变快 | CUDA Event、P50/P95 |
| 系统级 | 是否存在启动、同步或空闲问题 | Nsight Systems |
| Kernel 级 | 时间消耗在原子、内存、同步还是执行依赖 | Nsight Compute |
| 资源级 | Block、寄存器、Shared Memory 是否限制并发 | LaunchStats、Occupancy |
| 端到端 | Reduction 是否值得继续优化 | Amdahl 占比 |

## Level 0：CPU 成本模型

下面的脚本不模拟 GPU 的真实并发，只计算三种实现的操作数量，帮助你在上 GPU 前提出可证伪假设。

保存为 `reduction_cost_model.py` ：

```
#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

def estimate(elements: int, blocks: int, block_size: int) -> dict:
    if min(elements, blocks, block_size) <= 0:
        raise ValueError("elements、blocks、block-size 必须大于 0")
    if block_size % 32 != 0 or block_size > 1024:
        raise ValueError("block-size 必须是 32 的倍数且不超过 1024")

    warps_per_block = block_size // 32
    shared_steps = int(math.log2(block_size))
    return {
        "elements": elements,
        "blocks": blocks,
        "block_size": block_size,
        "baseline_global_atomics": elements,
        "optimized_global_atomics": blocks,
        "atomic_reduction_ratio": elements / blocks,
        "v1_shared_reduction_steps_per_block": shared_steps,
        "v1_barriers_per_block_approx": shared_steps,
        "v2_warp_shuffle_rounds": 5,
        "v2_warp_partial_values_per_block": warps_per_block,
        "minimum_input_bytes": elements * 4,
        "warning": "这是操作数量模型，不是 GPU 性能预测。",
    }

def main():
    parser = argparse.ArgumentParser(description="CUDA Reduction 成本模型")
    parser.add_argument("--elements", type=int, default=4_194_304)
    parser.add_argument("--blocks", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--output", default="reports/cost_model.json")
    args = parser.parse_args()

    result = estimate(args.elements, args.blocks, args.block_size)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

if __name__ == "__main__":
    main()
```

### 运行命令

```
mkdir -p reports
python reduction_cost_model.py
python reduction_cost_model.py --elements 16777216 --blocks 512 --block-size 256
```

### 预期现象

默认配置中，全局原子操作从约 419 万次降到 256 次，数量模型会给出很大的缩减比。但这只是提出了“原子争用可能是主要瓶颈”的假设；它没有计算内存层级、原子吞吐、Warp 调度、频率或编译器生成代码。

## Level 1：通用 CUDA C++ Reduction

### 环境检测

```
mkdir -p reports
{
  date -Iseconds
  uname -a
  nvidia-smi -L
  nvidia-smi --query-gpu=name,driver_version,memory.total,pci.bus_id,pstate,temperature.gpu,power.limit --format=csv
  nvcc --version
  nsys --version 2>/dev/null || true
  ncu --version 2>/dev/null || true
} 2>&1 | tee reports/environment.txt
```

### 完整代码

保存为 `reduction_benchmark.cu` ：

```
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t error__ = (call);                                            \
    if (error__ != cudaSuccess) {                                            \
      std::cerr << "CUDA error: " << cudaGetErrorString(error__)            \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;     \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

__global__ void reduce_atomic_baseline(const float* x, float* output, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    atomicAdd(output, x[i]);
  }
}

__global__ void reduce_shared_v1(const float* x, float* output, int n) {
  extern __shared__ float shared[];
  int tid = threadIdx.x;
  int index = blockIdx.x * blockDim.x + tid;
  int stride = blockDim.x * gridDim.x;

  float local = 0.0f;
  for (int i = index; i < n; i += stride) {
    local += x[i];
  }
  shared[tid] = local;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      shared[tid] += shared[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(output, shared[0]);
  }
}

__inline__ __device__ float warp_sum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__global__ void reduce_warp_v2(const float* x, float* output, int n) {
  __shared__ float warp_sums[32];
  int tid = threadIdx.x;
  int lane = tid & 31;
  int warp_id = tid >> 5;
  int warp_count = (blockDim.x + 31) / 32;
  int index = blockIdx.x * blockDim.x + tid;
  int stride = blockDim.x * gridDim.x;

  float local = 0.0f;
  for (int i = index; i < n; i += stride) {
    local += x[i];
  }
  local = warp_sum(local);

  if (lane == 0) {
    warp_sums[warp_id] = local;
  }
  __syncthreads();

  if (warp_id == 0) {
    float block_value = lane < warp_count ? warp_sums[lane] : 0.0f;
    block_value = warp_sum(block_value);
    if (lane == 0) {
      atomicAdd(output, block_value);
    }
  }
}

struct Stats {
  float p50_ms;
  float p95_ms;
  float min_ms;
  float max_ms;
  float result;
};

float percentile(std::vector<float> values, double q) {
  std::sort(values.begin(), values.end());
  double position = (values.size() - 1) * q;
  size_t lower = static_cast<size_t>(std::floor(position));
  size_t upper = static_cast<size_t>(std::ceil(position));
  if (lower == upper) return values[lower];
  double weight = position - lower;
  return static_cast<float>(values[lower] * (1.0 - weight) + values[upper] * weight);
}

template <typename Launcher>
Stats benchmark(Launcher launch, float* d_output, int warmup, int iterations) {
  for (int i = 0; i < warmup; ++i) {
    CUDA_CHECK(cudaMemset(d_output, 0, sizeof(float)));
    launch();
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  std::vector<float> samples;
  samples.reserve(iterations);

  for (int i = 0; i < iterations; ++i) {
    CUDA_CHECK(cudaMemset(d_output, 0, sizeof(float)));
    CUDA_CHECK(cudaEventRecord(start));
    launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());
    float elapsed = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    samples.push_back(elapsed);
  }

  float result = 0.0f;
  CUDA_CHECK(cudaMemcpy(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  std::vector<float> sorted = samples;
  std::sort(sorted.begin(), sorted.end());
  return {percentile(samples, 0.50), percentile(samples, 0.95),
          sorted.front(), sorted.back(), result};
}

void print_stats(const std::string& name, const Stats& stats, size_t elements,
                 float expected) {
  double effective_gbps = (elements * sizeof(float)) / (stats.p50_ms * 1e6);
  double relative_error = std::abs(stats.result - expected) / std::max(1.0f, expected);
  std::cout << std::left << std::setw(18) << name
            << " p50=" << std::setw(9) << stats.p50_ms
            << " p95=" << std::setw(9) << stats.p95_ms
            << " min=" << std::setw(9) << stats.min_ms
            << " max=" << std::setw(9) << stats.max_ms
            << " effective_GB/s=" << std::setw(9) << effective_gbps
            << " result=" << stats.result
            << " rel_error=" << relative_error << '\n';
  if (relative_error > 1e-4) {
    throw std::runtime_error(name + " correctness check failed");
  }
}

int main(int argc, char** argv) {
  size_t elements = 1u << 22;
  int block_size = 256;
  int blocks = 256;
  int warmup = 10;
  int iterations = 50;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--elements" && i + 1 < argc) elements = std::stoull(argv[++i]);
    else if (arg == "--block" && i + 1 < argc) block_size = std::stoi(argv[++i]);
    else if (arg == "--blocks" && i + 1 < argc) blocks = std::stoi(argv[++i]);
    else if (arg == "--warmup" && i + 1 < argc) warmup = std::stoi(argv[++i]);
    else if (arg == "--iters" && i + 1 < argc) iterations = std::stoi(argv[++i]);
    else {
      std::cerr << "Usage: " << argv[0]
                << " [--elements N] [--block N] [--blocks N]"
                << " [--warmup N] [--iters N]\n";
      return EXIT_FAILURE;
    }
  }

  if (elements == 0 || blocks <= 0 || warmup <= 0 || iterations <= 0 ||
      block_size < 32 || block_size > 1024 || block_size % 32 != 0 ||
      (block_size & (block_size - 1)) != 0) {
    std::cerr << "Invalid arguments: block must be a power of two, a multiple of 32, and <= 1024.\n";
    return EXIT_FAILURE;
  }

  int device = 0;
  cudaDeviceProp props{};
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&props, device));
  std::cout << "GPU=" << props.name
            << " compute_capability=" << props.major << "." << props.minor
            << " global_memory_GiB=" << props.totalGlobalMem / double(1ull << 30)
            << " SMs=" << props.multiProcessorCount << '\n';
  std::cout << "elements=" << elements << " blocks=" << blocks
            << " block_size=" << block_size << " warmup=" << warmup
            << " iterations=" << iterations << '\n';

  if (elements > static_cast<size_t>(std::numeric_limits<int>::max())) {
    std::cerr << "This teaching implementation requires elements <= INT_MAX.\n";
    return EXIT_FAILURE;
  }

  std::vector<float> host(elements, 1.0f);
  float expected = static_cast<float>(elements);
  float* d_input = nullptr;
  float* d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, elements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_input, host.data(), elements * sizeof(float), cudaMemcpyHostToDevice));

  auto baseline = benchmark(
      [&] { reduce_atomic_baseline<<<blocks, block_size>>>(d_input, d_output, static_cast<int>(elements)); },
      d_output, warmup, iterations);
  auto shared = benchmark(
      [&] { reduce_shared_v1<<<blocks, block_size, block_size * sizeof(float)>>>(
                d_input, d_output, static_cast<int>(elements)); },
      d_output, warmup, iterations);
  auto warp = benchmark(
      [&] { reduce_warp_v2<<<blocks, block_size>>>(d_input, d_output, static_cast<int>(elements)); },
      d_output, warmup, iterations);

  print_stats("atomic_baseline", baseline, elements, expected);
  print_stats("shared_v1", shared, elements, expected);
  print_stats("warp_v2", warp, elements, expected);
  std::cout << "speedup_shared=" << baseline.p50_ms / shared.p50_ms
            << " speedup_warp=" << baseline.p50_ms / warp.p50_ms
            << " warp_vs_shared=" << shared.p50_ms / warp.p50_ms << '\n';

  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));
  return EXIT_SUCCESS;
}
```

### 编译命令

通用编译：

```
nvcc -O3 -lineinfo -std=c++17 reduction_benchmark.cu -o reduction_benchmark
```

指定常见架构时，选择当前机器对应的一项，不要把多项全部硬编码为唯一方案：

```
# RTX 3080/3090
nvcc -O3 -lineinfo -std=c++17 -arch=sm_86 reduction_benchmark.cu -o reduction_benchmark

# RTX 4090 / L40 / L40S
nvcc -O3 -lineinfo -std=c++17 -arch=sm_89 reduction_benchmark.cu -o reduction_benchmark

# H100/H200
nvcc -O3 -lineinfo -std=c++17 -arch=sm_90 reduction_benchmark.cu -o reduction_benchmark
```

Blackwell 的具体 `sm_` 目标必须以当前 GPU 的 Compute Capability 与已安装 CUDA Toolkit 支持为准。工具链过旧时应升级隔离环境或使用 PTX 兼容路径，不能把编译失败解释为硬件不支持 Reduction。

### 运行命令

```
./reduction_benchmark
./reduction_benchmark --elements 1048576 --block 256 --blocks 128
./reduction_benchmark --elements 4194304 --block 256 --blocks 256
./reduction_benchmark --elements 16777216 --block 256 --blocks 512
```

扫描 Block Size：

```
for block in 64 128 256 512 1024; do
  ./reduction_benchmark --elements 4194304 --block "$block" --blocks 256
done | tee reports/block_sweep.txt
```

扫描 Block 数：

```
for blocks in 32 64 128 256 512 1024; do
  ./reduction_benchmark --elements 4194304 --block 256 --blocks "$blocks"
done | tee reports/grid_sweep.txt
```

### 预期现象

- Baseline 会执行约 `N` 次全局原子操作，通常明显慢于 V1/V2；
- V1/V2 每个 Block 只执行一次全局原子操作；
- V2 减少 Shared Memory 归约步骤和 Block 级同步，常有进一步收益，但收益不保证；
- Block 太小可能无法提供足够并行度，太大可能增加资源压力或降低调度灵活性；
- 输入很小时，三种 Kernel 的差距会缩小，启动开销占比上升；
- 性能数字只能作为本机实测，不得复制为所有 GPU 的结论。

## 正确性分析

### 为什么输入使用全 1

全 1 输入让 Reference 等于 `N` ，便于快速检查。但浮点加法不满足严格结合律，不同归约顺序可能产生不同舍入误差。正式项目还需要加入随机输入，并用 CPU Double Accumulation 作为 Reference。

### 随机输入扩展

将：

```
std::vector<float> host(elements, 1.0f);
float expected = static_cast<float>(elements);
```

替换为固定种子的随机数据，并在 CPU 用 Double 计算：

```
#include <random>

std::mt19937 rng(2026);
std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
std::vector<float> host(elements);
for (float& value : host) value = distribution(rng);
double reference = std::accumulate(host.begin(), host.end(), 0.0);
float expected = static_cast<float>(reference);
```

此时容差应根据输入规模和数值范围制定，并在报告里记录，而不是随意放宽到“总能通过”。

## Nsight Systems：确认 Kernel 边界

```
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  --output=reports/reduction_nsys \
  ./reduction_benchmark --elements 4194304 --block 256 --blocks 256 --warmup 2 --iters 5

nsys stats reports/reduction_nsys.nsys-rep \
  | tee reports/reduction_nsys_stats.txt
```

检查：

- 三个 Kernel 名称和调用次数是否符合预期；
- `cudaMemset` 是否被错误计入 Kernel 计时；
- 是否出现频繁同步或异常长的 CUDA API；
- Kernel 很短时，Launch 开销是否开始主导；
- Profile 运行与纯 Benchmark 运行的时间差异。

## Nsight Compute：验证优化原因

先用 Kernel 名称过滤，避免对所有 Warmup 和全部 Kernel 采集完整指标：

```
ncu \
  --kernel-name regex:reduce_atomic_baseline \
  --launch-count 1 \
  --section SpeedOfLight \
  --section MemoryWorkloadAnalysis \
  --section LaunchStats \
  --section Occupancy \
  --force-overwrite \
  --export reports/ncu_atomic \
  ./reduction_benchmark --warmup 1 --iters 1

ncu \
  --kernel-name regex:reduce_warp_v2 \
  --launch-count 1 \
  --section SpeedOfLight \
  --section MemoryWorkloadAnalysis \
  --section LaunchStats \
  --section Occupancy \
  --force-overwrite \
  --export reports/ncu_warp \
  ./reduction_benchmark --warmup 1 --iters 1
```

重点对比：

| 指标方向 | Baseline 假设 | V2 假设 |
| --- | --- | --- |
| 全局原子相关 Stall | 高 | 显著减少 |
| Kernel Duration | 长 | 下降 |
| DRAM Read | 读取相同输入 | 大体相近 |
| Shared Memory | 很少 | 仅 Warp Partial |
| Barrier | 很少 | 一个 Block 级 Barrier |
| Occupancy | 不一定低 | 不一定更高 |

不要先写结论再挑指标。若报告显示原子不是主要瓶颈，应回到工作负载规模、编译代码、Kernel 过滤和 GPU 原子实现重新分析。

## Level 2：架构专项可选实验

### 能力检测

```
nvidia-smi --query-gpu=name,compute_cap --format=csv
nvcc --list-gpu-arch
./reduction_benchmark --elements 4194304
```

### Ampere：RTX 3080/3090、A100

- 使用 `sm_86` 或 `sm_80` 分别构建，不能混为同一设备；
- 比较 V1 与 V2，观察 Warp Shuffle 减少同步后的收益；
- 双 RTX 3080 不需要参与本单 Kernel 实验，使用 `CUDA_VISIBLE_DEVICES=0` 固定单卡；
- 消费卡没有 A100 的 HBM、NVLink/NVSwitch 与数据中心特性。

### Ada Lovelace：RTX 4090、L40/L40S

- 使用 `sm_89` ；
- 保持相同输入与编译参数才能和 Ampere 做架构对照；
- RTX 4090 属于 Ada，不属于 Blackwell。

### Hopper：H100/H200

- 基础 Reduction 仍使用同一通用代码；
- 可选研究 Cooperative Groups、异步事务和 Cluster，但必须单独实现并验证；
- FP8 与 Transformer Engine 不会自动让 FP32 Reduction 变快。

### Blackwell：RTX 5090、B100/B200/GB200

- 使用当前 Toolkit 支持的准确 Compute Capability；
- 可选研究 Thread Block Cluster、Distributed Shared Memory 或更新的异步流水；
- FP4、TMA、Cluster 是不同优化维度，不能把通用 Warp Shuffle 结果包装成这些特性的验证；
- RTX 5090 与 B200/GB200 的显存、互联和数据中心能力不同。

## 优化前后对照

填写本机真实结果：

| 实现 | P50 | P95 | 有效输入带宽 | 全局原子次数 | 正确性 | 主要瓶颈 | | ---------------------------------- | -- | -- | -- | -------- | ----- | --- | | atomic baseline | 待测 | 待测 | 待测 | 约 N | 通过/失败 | 待分析 | | shared V1 | 待测 | 待测 | 待测 | 约 Blocks | 通过/失败 | 待分析 | | warp V2 | 待测 | 待测 | 待测 | 约 Blocks | 通过/失败 | 待分析 |

一份合格结论应同时包含：

- 墙钟数据：P50/P95 与重复次数；
- 正确性：Reference、误差与容差；
- 结构证据：全局原子次数如何变化；
- Profile 证据：时间和 Stall 是否按假设变化；
- 边界：输入尺寸、Block 配置、GPU、驱动、Toolkit。

## 常见错误与排查

### 1\. 编译报 numeric\_limits 未声明

原因：缺少 C++ 标准库的 `limits` 头文件。

修复：在头文件区加入：

```
#include <limits>
```

### 2\. invalid configuration argument

原因：Block Size 超过硬件限制或 Shared Memory 配置非法。

排查：

```
nvidia-smi -L
./reduction_benchmark --block 256
```

代码已限制 Block 为 32 的倍数、2 的幂且不超过 1024。

### 3\. 结果偶尔不同

浮点原子和并行归约顺序可能变化。先检查误差量级，再决定是否使用更稳定的分层 Reduction、Double Accumulation 或补偿求和。不要为了通过测试无限放大容差。

### 4\. V2 比 V1 更慢

可能原因：输入过小、Block 配置不合适、V1 已足够快、编译器优化差异、测量噪声或 GPU 频率变化。扫描输入、Block 与 Grid，并查看实际 Kernel 指标。

### 5\. 有效带宽超过规格表

本项目计算的是最小输入字节除以时间，不包含全部缓存、原子和物理事务。若公式、单位或计时边界错误，也可能产生虚高。它只能作为同一实验内部的算法有效带宽。

### 6\. ERR\_NVGPUCTRPERM

性能计数器被安全策略限制。请记录环境并由管理员按 NVIDIA 安全建议处理。无法开放时使用墙钟和 Nsight Systems 完成可验证部分，不要声称已经测得 Warp Stall 或 Memory Throughput。

### 7\. ncu 抓到了错误的 Kernel

检查 Kernel 名称：

```
ncu --query-metrics-mode suffix --metrics sm__cycles_elapsed.avg ./reduction_benchmark --warmup 1 --iters 1
```

也可以先用 Nsight Systems 确认名称，再收紧 `--kernel-name` 过滤。

### 8\. 只在默认规模上更快

这不算完整优化。至少测试小、中、大三档输入，并说明转折点。对真实模型，还要确认 Reduction 是否仍是端到端热点。

## 面试题与答案

### 1\. Baseline 为什么慢？

\*\*答：\*\* 每个元素都对同一个全局地址执行 `atomicAdd` ，造成极强的读改写竞争和序列化。输入读取虽然合并，但原子更新路径先成为主要限制。

### 2\. Shared Memory V1 的核心收益是什么？

\*\*答：\*\* 每个线程先累加多个元素，每个 Block 在片上 Shared Memory 内归约，最后只进行一次全局原子更新，把全局原子数量从约 `N` 降为约 `Blocks` 。

### 3\. Warp Shuffle 为什么能继续优化？

\*\*答：\*\* 同一个 Warp 内线程可以通过寄存器交换数据，不必每一步都写 Shared Memory 并执行 Block Barrier。只需把每个 Warp 的部分和写入 Shared Memory，再由一个 Warp 完成最终合并。

### 4\. 为什么 Occupancy 高不代表 Reduction 快？

\*\*答：\*\* Occupancy 只表示可驻留 Warp 比例。全局原子争用、同步、内存流量和依赖链可能仍然主导。更高 Occupancy 也不能消除所有线程对同一地址的竞争。

### 5\. 为什么使用 Grid-Stride Loop？

\*\*答：\*\* 它允许用有限 Block 覆盖任意长度输入，让线程处理多个相隔 `gridDim * blockDim` 的元素，便于控制 Grid 规模并提高线程内局部累加。

### 6\. \_\_syncthreads() 放错位置有什么后果？

\*\*答：\*\* 若只有部分线程执行 Barrier，可能死锁；若在共享数据尚未对所有线程可见前省略 Barrier，会产生竞态和错误结果。V1 的每一步 Shared Reduction 都需要 Block 一致同步。

### 7\. 如何进一步消除最后的全局原子？

\*\*答：\*\* 使用两阶段 Reduction：第一阶段输出每个 Block 的 Partial Sum，第二阶段再归约 Partial。这样可以完全避免大量 Block 竞争同一输出，但会增加一次 Kernel Launch 和 Partial Buffer。

### 8\. 为什么库函数通常比手写 Kernel 更值得优先考虑？

\*\*答：\*\* CUB、Thrust 和框架 Reduction 已针对多架构、边界、数值与调度优化。手写项目用于学习和特殊融合场景；生产中应先基准成熟库，再证明自定义 Kernel 的必要性。

## 课后练习

1. 实现两阶段 Reduction，完全移除第一阶段的全局输出争用。
2. 加入 CUB `DeviceReduce::Sum` 作为成熟库基线。
1. 把输入改成 FP16、累加改为 FP32，比较性能与误差。
2. 扫描 `block={64,128,256,512,1024}` 与 `blocks={32,64,128,256,512,1024}` ，找出本机最优区域。
1. 使用随机输入和 CPU Double Reference，绘制误差随 `N` 的变化。
2. 在 PyTorch C++ Extension 中封装 V2，并与 `torch.sum` 对比。
1. 在 Nsight Compute 中检查是否存在 Shared Memory Bank Conflict。
2. 将 Reduction 融合进 L2Norm 或 Softmax 的第一阶段，比较“独立 Kernel”和“融合 Kernel”的全局流量。

## 项目验收 Checklist

### 正确性

- 三种 Kernel 均通过固定输入测试。
- 已加入随机输入与 CPU Double Reference。
- 容差有依据，没有为通过测试任意放宽。
- 已使用 Compute Sanitizer 或等价工具检查竞态/越界。

### 性能测量

- 使用 CUDA Event 测量 Kernel，不把 Host 提交时间当作执行时间。
- 已完成 Warmup，并报告 P50/P95/Min/Max。
- 已扫描至少三档输入规模。
- 已扫描 Block 与 Grid 配置。
- 最终性能在关闭 Profiler 时重新确认。

### Profile 证据

- 已用 Nsight Systems 确认 Kernel、Memset 与同步边界。
- 已对 Baseline 和 V2 分别运行 Nsight Compute，或记录权限限制。
- 已比较 Duration、Memory、Launch、Occupancy 与 Stall。
- 没有把 Occupancy 当作唯一优化目标。

### 可移植性

- 自动记录 GPU 名称、Compute Capability、显存与 SM 数量。
- 未把 RTX 4090 写成 Blackwell。
- 未把双 RTX 3080 写成 NVLink/NVSwitch 环境。
- Hopper/Blackwell 专属扩展有能力检测与回退路径。
- 性能数字只标记为本次环境实测。

### 交付

- reduction\\\_cost\\\_model.py 可运行并保存 JSON。
- reduction\\\_benchmark.cu 可编译、可运行、可修改。
- reports/ 包含环境、基准与 Profile 证据。
- analysis.md 包含假设、实验、反例、结论和适用边界。