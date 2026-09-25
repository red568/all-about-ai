---
title: "第20课：torch.compile 深度解析"
source: "https://everythingai.top/learn/ai-systems-performance-engineering-text/lesson-20"
author:
published:
created: 2026-09-26
description: "面向 AI 工程学习者的专业课程学习空间。"
tags:
  - "clippings"
---
**导航**

本课拆解 torch.compile 的图捕获、编译与回退机制，掌握适用场景、性能收益和常见 Graph Break。

## 一、课程定位

`torch.compile` 不是一个“打开就必然加速”的开关，而是一套把 Python/PyTorch 程序捕获为计算图、跨算子优化并生成目标代码的编译系统。本课不只讲 API，还要建立完整的性能工程判断链：编译器看到了多大的图、为什么发生 Graph Break、哪些 Guard 导致重编译、冷启动成本何时能摊平，以及如何证明收益来自 Kernel Fusion、调度或更低的 Runtime 开销。

学完后，你应能回答三个工程问题：

1. 这个工作负载适不适合编译？
2. 编译后为什么快、为什么没变快，或者为什么更慢？
1. 如何让动态输入、多卡训练和生产服务既获得收益，又避免编译风暴？

## 二、学习目标

- 理解 TorchDynamo、FX、AOTAutograd、TorchInductor、Triton/C++ 代码生成的分工。
- 区分 Graph Break、Guard Failure、Recompile 和编译失败。
- 正确测量首次编译、预热后稳态、显存和数值误差。
- 掌握 `fullgraph` 、 `dynamic` 、编译模式和局部禁用的使用边界。
- 能用日志定位图断裂、Guard 和动态 Shape 问题。
- 能估算编译投资的盈亏平衡点，并设计线上缓存与回退策略。
- 在 CPU、常见 NVIDIA GPU 和架构专项环境中完成分层实验。

## 三、前置知识

- 熟悉 PyTorch `nn.Module` 、训练或推理循环。
- 理解 Kernel Launch、显存带宽和算子融合。
- 能区分吞吐、P50/P99 延迟、冷启动和稳态延迟。
- 了解 CUDA 异步执行；GPU 计时前后需要同步。

本课建议使用隔离环境。CPU 实验只需要 Python 3.10+；编译实验建议按 PyTorch 官方安装页选择与驱动匹配的稳定版本，不把某个 CUDA Toolkit 版本硬编码为所有读者的前提。

## 四、核心直觉：用一次编译换很多次更便宜的执行

PyTorch eager 模式逐个执行算子，灵活、容易调试，但 Python 调度、算子边界、临时 Tensor 和大量小 Kernel 会产生开销。编译器尝试把一段程序视为整体：

```
Python/PyTorch 程序
        │
        ▼
TorchDynamo 捕获可编译区域
        │ FX Graph
        ▼
AOTAutograd 生成前向/反向图
        │
        ▼
TorchInductor 做融合、调度和代码生成
        │
        ├── GPU：常见为 Triton / CUDA 相关代码
        └── CPU：常见为 C++ / 向量化代码
```

这像把一条每天走很多次的土路修成高速公路。修路有成本，通车后每次更快；如果只走一次，修路反而亏。如果输入形状不断变化，编译器还可能不断“重修不同规格的路”。

因此， `torch.compile` 的适用特征通常是：

- 同一模型或训练 Step 会重复运行很多次；
- 图中有可融合的 Pointwise、归一化、归约等算子；
- Python 控制流和 I/O 没有频繁切断图；
- Shape 集合有限，或者动态维度能被合理泛化；
- 冷启动可以预热、缓存或在部署阶段被吸收。

## 五、编译栈的系统原理

### 5.1 TorchDynamo：从 Python 字节码捕获图

TorchDynamo 在 Python Frame 层观察字节码，把能安全表示的 PyTorch 运算捕获为 FX Graph。它不会盲目假设所有条件永远成立，而是为编译结果生成 Guard，例如：

- 输入 Tensor 的 dtype、device、rank 或部分 Shape；
- 某个 Python 常量或对象身份；
- Module 的训练/推理状态；
- 全局变量、默认设备或 Autocast 状态。

下次调用时 Guard 成立，就复用已编译结果；Guard 失败，可能重新捕获并编译新版本。

### 5.2 FX Graph：可变换的中间表示

FX Graph 把算子和数据依赖表示为节点与边。与逐行 Python 相比，它让后端看到更大的优化范围，例如删除冗余操作、安排内存生命周期、融合多个算子和选择实现。

### 5.3 AOTAutograd：提前生成前向和反向图

训练时，Autograd 原本边执行边记录。AOTAutograd 通过函数化等变换提前得到前向图和反向图，使编译器能跨越 Autograd 边界优化。它也必须处理 Mutation、Alias 和保存用于反向的中间值，所以训练图通常比纯推理更复杂。

### 5.4 TorchInductor：融合、调度与代码生成

TorchInductor 是默认编译后端。其优化方向包括：

- 将多个 Pointwise/Reduction 操作融合为更少的 Kernel；
- 减少中间 Tensor 的显存写回与再次读取；
- 降低 Python 和 Dispatcher 的调用次数；
- 为目标设备选择布局、Tile 和并行策略；
- 在 GPU 上生成或调用 Triton/外部高性能库，在 CPU 上生成向量化代码。

编译器不会把所有 GEMM 都重写。成熟库中的矩阵乘通常仍由 cuBLAS、cuBLASLt 等库执行；收益可能来自 GEMM 前后的 Epilogue 融合、布局和调度，而不是替代 GEMM 本身。

## 六、Graph Break、Guard 与 Recompile

### 6.1 Graph Break 不是同义于报错

默认 `fullgraph=False` 时，无法捕获的代码会结束当前编译区域，回到 Python 执行，再从后续可捕获位置建立新区域。程序可能正确运行，但图被切碎后会失去跨区域融合机会，并增加调度开销。

常见原因包括：

- 数据依赖的 Python 控制流，如 `if tensor.sum() > 0` ；
- `.item()` 把 Tensor 值拉回 Python；
- 日志、文件、网络和第三方 C 扩展；
- 编译器暂不支持的 Python 或算子语义；
- 对对象、容器或全局状态的复杂 Mutation。

`fullgraph=True` 要求一次调用被捕获为完整图，遇到 Graph Break 就报错。它适合测试和定位，不代表生产环境必须全图编译。

### 6.2 Guard 是正确性契约

Guard 不是“多余检查”。编译器可能针对特定 Shape、常量和布局生成专门代码，只有假设仍成立时才能安全复用。优化目标不是删除 Guard，而是避免没有业务价值的输入变化和 Python 状态变化。

### 6.3 Recompile 是缓存未命中

一次 Recompile 不一定有问题：首次遇到新 dtype 或新 Shape，生成新版本很合理。问题是无界增长。若请求长度几乎每次都不同，又没启用合适的动态 Shape，可能产生编译风暴、缓存淘汰和尾延迟尖峰。

PyTorch 对单个 Frame 的重编译缓存有上限；达到上限后的行为与版本和配置有关，不能把内部默认值当作稳定业务接口。应通过日志观察，并在业务层控制 Shape Bucket。

### 6.4 动态 Shape 的选择

`dynamic=None` 是常用默认：先按静态信息编译，检测到 Shape 变化后尝试泛化。 `dynamic=True` 尝试更早生成可处理变化尺寸的代码； `dynamic=False` 更偏向每组 Shape 特化。

动态并非总是更快：

- 优点：少编译几个版本，降低冷启动和缓存压力；
- 代价：部分静态优化、常量折叠或 Autotune 选择空间变弱；
- 边界：不同 rank、dtype、分支或布局仍可能需要不同版本。

生产中更常用的折中是“有限 Bucket + Bucket 内动态”，例如把序列长度归入 128、256、512、1024 四档。

## 七、关键性能模型

### 7.1 编译投资的盈亏平衡点

设：

$T_c$ ：首次捕获、编译和 Autotune 的总成本；

$T_e$ ：eager 模式单次稳态时间；

$T_s$ ：compiled 模式单次稳态时间；

N：部署期复用次数。

总时间为：

$$
T_{eager}=N T_e
$$

$$
T_{compiled}=T_c+N T_s
$$

当 $T_s < T_e$ 时，盈亏平衡调用次数是：

$$
N^*=\frac{T_c}{T_e-T_s}
$$

若编译耗时 20 秒，每次从 10 ms 降到 6 ms，则约需 5000 次调用才能摊平。若模型实例只处理 200 次请求就被回收，即便稳态快 40%，总体仍可能亏损。

### 7.2 融合为什么降低时间

对两个连续的内存受限算子，eager 可能执行：

```
读 x → 算 a → 写临时 t
读 t → 算 b → 写 y
```

融合后可能成为：

```
读 x → 算 a → 算 b → 写 y
```

中间 Tensor 不再往返全局显存。近似字节量从 4S 降到 2S，同时 Kernel Launch 数量由 2 降到 1。真实收益还受 Cache、寄存器压力、占用率和算子库实现影响。

### 7.3 Amdahl 定律约束总体加速

若可编译部分占原时间比例为 p，该部分加速 s 倍，则总体加速上限为：

$$
S=\frac{1}{(1-p)+p/s}
$$

如果数据加载、通信和 Python I/O 占 60%，再优秀的 Kernel Fusion 也只能优化剩下的 40%。所以必须结合第 19 课的方法看端到端 Critical Path。

### 7.4 尾延迟模型

服务端 P99 可能由稳态执行、重编译、Autotune、显存回收和队列等待共同决定：

$$
L_{request}=L_{queue}+L_{compile\ miss}+L_{execute}+L_{sync}
$$

只报告预热后的平均延迟，会掩盖线上最危险的编译缓存未命中。

## 八、编译范围和模式选择

### 8.1 在哪里编译

优先从“能覆盖主要计算、又能保持边界稳定”的最高层开始：

1. 推理：编译完整 `nn.Module` 的计算部分，I/O、Tokenizer 和后处理留在外面。
2. 训练：先编译模型前向；收益不足时再评估训练 Step 或 Compiled Autograd。
1. 重复 Block：整图编译冷启动过大时，编译共享结构的 Transformer Block，形成 Regional Compilation。
2. 分布式：通常在 DDP 包装前编译内层 Module；FSDP、TP 与编译的组合需按当前 PyTorch 官方示例验证。

现代 PyTorch 为 `nn.Module` 提供 `.compile()` ；它会让 Module 的 `__call__` 走编译路径。 `torch.compile(fn_or_module)` 仍适合函数或需要返回独立包装对象的场景。

### 8.2 常用模式

| 模式 | 主要目标 | 常见代价 | 适合场景 |
| --- | --- | --- | --- |
| `default` | 编译成本与稳态性能平衡 | 不保证最低延迟 | 首次基线 |
| `reduce-overhead` | 降低 Python/Launch 开销，可能使用 CUDA Graph 相关优化 | 可能增加显存；输入/控制流有限制 | 小 Batch、Launch-bound 推理 |
| `max-autotune` | 搜索更优 Kernel/配置 | 首次编译显著变慢 | 长生命周期、重复 Shape |

可用模式、内部实现和后端选项会随 PyTorch、设备与构建变化。先以 `default` 建立基线，再逐项测量；不要把某个模式名等同于固定优化集合。

## 九、瓶颈分析方法

### 9.1 四阶段诊断

1. \*\*正确性\*\*：相同权重和输入，对比 eager/compiled 输出或梯度。
2. \*\*可编译性\*\*：查看 Graph Break、Guard 和 Recompile 日志。
1. \*\*收益来源\*\*：比较 Kernel 数量、显存流量、CPU Launch 空洞和稳态时间。
2. \*\*经济性\*\*：把首次编译、缓存复用次数、实例生命周期和 P99 纳入总成本。

### 9.2 推荐日志

```
TORCH_LOGS="graph_breaks,recompiles,guards,dynamic" \
python torch_compile_lab.py --device auto --vary-shapes
```

日志量可能很大，先在最小复现中使用。若完整模型需要跨 Frame 分析，可按当前 PyTorch 文档使用 `TORCH_TRACE` 和 `tlparse` ；这类工具接口变化较快，应锁定与 PyTorch 匹配的版本。

### 9.3 快速决策树

```
结果错误？ ──是→ 缩小模型；检查随机数、Mutation、Alias、精度和自定义算子
   │否
   ▼
首次很慢但稳态快？ ──是→ 计算 N*；预热、缓存、Regional Compilation
   │否
   ▼
频繁 Recompile？ ──是→ 看 Guard；固定状态、Shape Bucket、评估 dynamic
   │否
   ▼
Graph Break 多？ ──是→ 去掉 .item()/I/O；局部重写或 disable 难编译区域
   │否
   ▼
Kernel 少但仍不快？ ──→ 查 GEMM 占比、数据/通信、寄存器压力和真实端到端瓶颈
```

## 十、Level 0：无 GPU 的编译收益规划实验

下面的纯 Python 程序不模拟编译器内部实现，而是训练你做上线前最关键的经济性判断。

保存为 `compile_roi_planner.py` ：

```
#!/usr/bin/env python3
import argparse

def main():
    p = argparse.ArgumentParser(description="torch.compile ROI planner")
    p.add_argument("--compile-seconds", type=float, default=20.0)
    p.add_argument("--eager-ms", type=float, default=10.0)
    p.add_argument("--compiled-ms", type=float, default=6.0)
    p.add_argument("--calls", type=int, nargs="+", default=[100, 1000, 5000, 20000])
    p.add_argument("--variants", type=int, default=1,
                   help="实际会生成的编译版本数量")
    args = p.parse_args()

    if min(args.compile_seconds, args.eager_ms, args.compiled_ms) < 0:
        raise SystemExit("时间不能为负数")
    if args.variants < 1 or any(n < 1 for n in args.calls):
        raise SystemExit("variants 和 calls 必须为正整数")

    compile_ms = args.compile_seconds * 1000.0 * args.variants
    delta = args.eager_ms - args.compiled_ms
    if delta > 0:
        break_even = compile_ms / delta
        print(f"break_even_calls={break_even:.1f}")
    else:
        break_even = float("inf")
        print("break_even_calls=never (compiled steady-state is not faster)")

    print("calls eager_total_s compiled_total_s net_saved_s decision")
    for n in args.calls:
        eager_total = n * args.eager_ms
        compiled_total = compile_ms + n * args.compiled_ms
        saved = (eager_total - compiled_total) / 1000.0
        decision = "compile" if saved > 0 else "eager"
        print(f"{n:5d} {eager_total/1000:13.3f} "
              f"{compiled_total/1000:16.3f} {saved:11.3f} {decision}")

if __name__ == "__main__":
    main()
```

运行：

```
python3 compile_roi_planner.py
python3 compile_roi_planner.py --compile-seconds 20 --eager-ms 10 \
  --compiled-ms 6 --variants 4 --calls 1000 5000 20000 50000
```

预期现象：默认参数约在 5000 次调用达到盈亏平衡；当编译版本从 1 增至 4，编译投入变为四倍，盈亏平衡点也约变为四倍。这解释了为什么无界动态 Shape 会吞掉稳态收益。

## 十一、Level 1：通用 PyTorch CPU/CUDA 实验

该实验自动检测 PyTorch、CUDA、GPU 名称、Compute Capability 和显存，不硬编码具体显卡。它分别报告首次 compiled 调用和预热后的稳态时间。

保存为 `torch_compile_lab.py` ：

```
#!/usr/bin/env python3
import argparse
import copy
import statistics
import time

import torch
from torch import nn

class Block(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width * 2)
        self.fc2 = nn.Linear(width * 2, width)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.fc2(torch.nn.functional.gelu(self.fc1(x)))
        # 保留多个可融合的 pointwise 运算，便于观察编译收益
        x = torch.sigmoid(x) * x + 0.1 * torch.tanh(x)
        return residual + x

class Model(nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.blocks = nn.ModuleList([Block(width) for _ in range(depth)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def timed(model, x, warmup, steps, device):
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        sync(device)
        samples = []
        for _ in range(steps):
            t0 = time.perf_counter()
            model(x)
            sync(device)
            samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples), statistics.quantiles(samples, n=20)[18]

def parse_dynamic(value):
    return {"auto": None, "true": True, "false": False}[value]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--compile-mode", default="default",
                   choices=["default", "reduce-overhead", "max-autotune"])
    p.add_argument("--dynamic", default="auto", choices=["auto", "true", "false"])
    p.add_argument("--fullgraph", action="store_true")
    p.add_argument("--vary-shapes", action="store_true")
    args = p.parse_args()

    if not hasattr(torch, "compile"):
        raise SystemExit("当前 PyTorch 不提供 torch.compile，请升级到受支持的稳定版本")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("请求了 CUDA，但当前 PyTorch/驱动不可用")
    device = torch.device(
        "cuda" if args.device == "cuda" or
        (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )

    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    eager = Model(args.width, args.depth).eval().to(device)
    compiled = copy.deepcopy(eager)

    print(f"torch={torch.__version__} device={device}")
    if device.type == "cuda":
        prop = torch.cuda.get_device_properties(device)
        print(f"gpu={prop.name} capability={prop.major}.{prop.minor} "
              f"vram_gib={prop.total_memory / 2**30:.2f} cuda={torch.version.cuda}")
        torch.cuda.reset_peak_memory_stats(device)

    x = torch.randn(args.batch, args.width, device=device)
    dynamic = parse_dynamic(args.dynamic)
    compiled.compile(mode=args.compile_mode,
                     fullgraph=args.fullgraph,
                     dynamic=dynamic)

    with torch.inference_mode():
        expected = eager(x)
        sync(device)
        t0 = time.perf_counter()
        actual = compiled(x)
        sync(device)
        first_compiled_ms = (time.perf_counter() - t0) * 1000.0
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)

    eager_p50, eager_p95 = timed(eager, x, args.warmup, args.steps, device)
    comp_p50, comp_p95 = timed(compiled, x, args.warmup, args.steps, device)
    print(f"first_compiled_call_ms={first_compiled_ms:.3f}")
    print(f"eager_p50_ms={eager_p50:.3f} eager_p95_ms={eager_p95:.3f}")
    print(f"compiled_p50_ms={comp_p50:.3f} compiled_p95_ms={comp_p95:.3f}")
    print(f"steady_speedup={eager_p50 / comp_p50:.3f}x")

    if args.vary_shapes:
        print("vary_shapes_begin")
        with torch.inference_mode():
            for batch in [8, 16, 24, 32, 48, 64, 16, 32]:
                value = torch.randn(batch, args.width, device=device)
                t0 = time.perf_counter()
                compiled(value)
                sync(device)
                print(f"batch={batch:2d} call_ms={(time.perf_counter()-t0)*1000:.3f}")
        print("vary_shapes_end")

    if device.type == "cuda":
        print(f"peak_allocated_mib={torch.cuda.max_memory_allocated(device)/2**20:.1f}")
        print(f"peak_reserved_mib={torch.cuda.max_memory_reserved(device)/2**20:.1f}")

if __name__ == "__main__":
    main()
```

### 11.1 安装与运行

```
python3 -m venv .venv-compile
source .venv-compile/bin/activate
python -m pip install --upgrade pip
# 按 https://pytorch.org/get-started/locally/ 选择与本机匹配的稳定构建
```

CPU 基线：

```
python torch_compile_lab.py --device cpu --width 512 --depth 3 \
  --warmup 5 --steps 20
```

常见 NVIDIA GPU：

```
nvidia-smi
python torch_compile_lab.py --device cuda --compile-mode default
python torch_compile_lab.py --device cuda --compile-mode reduce-overhead
python torch_compile_lab.py --device cuda --compile-mode max-autotune
```

动态 Shape 与重编译：

```
TORCH_LOGS="recompiles,dynamic" python torch_compile_lab.py \
  --device auto --vary-shapes --dynamic false

TORCH_LOGS="recompiles,dynamic" python torch_compile_lab.py \
  --device auto --vary-shapes --dynamic true
```

Graph Break 严格检查：

```
TORCH_LOGS="graph_breaks" python torch_compile_lab.py \
  --device auto --fullgraph
```

### 11.2 预期现象

- 第一次 compiled 调用通常远慢于稳态，因为包含捕获、代码生成、编译， `max-autotune` 还可能做更多搜索。
- GPU 上的收益通常比小型 CPU 实验更明显，但不保证；矩阵乘占比很高且算子已高度优化时，提升可能有限。
- `reduce-overhead` 对小 Batch、许多短 Kernel 的场景更可能有利，也可能增加显存。
- `dynamic=false` 配合多种 Batch 可能看到更多 Recompile； `dynamic=true` 可能减少版本数，但稳态快慢需实测。
- 首次运行还可能受到 CUDA Context、库加载和磁盘缓存影响；跨进程冷启动要单独重复测试。

### 11.3 结果分析模板

| 项目 | Eager | Compiled | 如何解释 |
| --- | --- | --- | --- |
| 首次调用 | 记录值 | 记录值 | 不混入稳态平均 |
| P50 稳态 | 记录值 | 记录值 | 计算 Speedup |
| P95 稳态 | 记录值 | 记录值 | 观察抖动 |
| Kernel 数量 | Profile | Profile | 判断是否融合 |
| 峰值 allocated | 记录值 | 记录值 | 检查显存副作用 |
| 编译版本数 | 日志 | 日志 | 识别编译风暴 |
| 最大误差 | 0 | 记录值 | 先过正确性门槛 |

不要把本机测量值写成“RTX 3080 一定加速多少”或“所有 CPU 都会变慢”。结果依赖 Shape、dtype、PyTorch/Triton 版本、驱动、功耗、时钟和 Cache 状态。

## 十二、Level 2：架构专项实验与边界

### 12.1 正确的硬件映射

| 架构 | 示例硬件 | 本课可观察重点 |
| --- | --- | --- |
| Ampere | RTX 3080/3090、A100 | TF32/BF16、融合、通用 Inductor 基线 |
| Ada Lovelace | RTX 4090、L40/L40S | 更强 Tensor Core/推理基线；4090 不是 Blackwell |
| Hopper | H100/H200 | FP8/Transformer Engine 生态与数据中心部署 |
| Blackwell | RTX 5090、B100/B200/GB200 | 新低精度与更新编译后端支持 |

`torch.compile` 本身不要求 NVLink、NVSwitch、FP8 或 FP4。没有对应硬件时，CPU 或任意受支持 CUDA GPU 都可学习捕获、Guard、重编译和融合；但无法用软件模拟来证明 FP8/FP4 Tensor Core 的真实吞吐。

### 12.2 双 RTX 3080 可执行路径

每张 RTX 3080 分别运行单卡基准，再对比编译前后；多卡训练时先确保每个 rank 使用相同 Shape Bucket：

```
CUDA_VISIBLE_DEVICES=0 python torch_compile_lab.py --device cuda
CUDA_VISIBLE_DEVICES=1 python torch_compile_lab.py --device cuda
nvidia-smi topo -m
```

消费级 RTX 3080 多卡通常通过 PCIe 通信。 `torch.compile` 优化本地计算图，不会自动消除 DDP All-Reduce；端到端收益需要同时检查计算—通信重叠。

### 12.3 查看生成代码和时间线

仅在最小实验上打开高详细度日志：

```
TORCH_LOGS="output_code,kernel_code" python torch_compile_lab.py \
  --device cuda --width 512 --depth 2 --steps 10

nsys profile --trace=cuda,nvtx,osrt -o compile_lab \
  python torch_compile_lab.py --device cuda --warmup 10 --steps 30
```

观察 compiled 稳态区间中的 Kernel 数量、CPU Launch 间隙和同步，不要把首次编译区间与稳态放在一起比较。

Hopper/Blackwell 的 FP8、FP4 或 Transformer Engine 实验应使用支持的框架、驱动和硬件能力检查。即使低精度可运行，也要单独验证 Scale、溢出、输出误差和端到端 Quality。

## 十三、优化前后对照

| 低效做法 | 优化做法 | 目的 |
| --- | --- | --- |
| 把首次编译计入稳态均值 | 冷启动与稳态分开报告 | 避免错误结论 |
| 任意长度直接输入 | 有限 Shape Bucket，必要时 Bucket 内动态 | 限制版本数 |
| 整个服务函数一起编译 | 只编译纯计算 Module | 隔离 I/O 和副作用 |
| 遇到问题全局关闭编译 | 最小复现，局部 `torch.compiler.disable` | 保留可编译收益 |
| 只看吞吐 | 同看正确性、P99、编译成本、显存 | 生产可用性 |
| 直接使用 `max-autotune` | 从 `default` 建基线再升级 | 控制实验变量 |
| 把 Graph Break 当崩溃 | 用日志判断区域和性能损失 | 有证据地重写 |
| 每次请求创建新模型 | 复用实例并预热稳定 Bucket | 摊薄编译成本 |

## 十四、常见错误与排查

### 14.1 第一次运行卡很久

这是编译、Autotune、CUDA Context 或依赖加载的叠加。先缩小模型、用 `default` 、关闭不必要的 Autotune，分别记录进程级冷启动和实例内稳态。不要直接中断后宣称编译器死锁。

### 14.2 编译后反而更慢

可能是调用次数太少、图太小、Graph Break 多、工作负载由库 GEMM 主导、动态 Shape 导致 Recompile，或编译代码增加寄存器压力。用盈亏平衡模型、日志和 Profile 分开验证。

### 14.3 输出不一致

先固定种子与推理模式，检查 Dropout、随机数、Autocast、Mutation、Alias、自定义算子和容差。低精度与算子重排会造成舍入差异，但不能用“浮点误差”掩盖明显错误。

### 14.4.item() 导致 Graph Break 或同步

尽量让判断留在 Tensor 计算中；确实需要 Python 标量的监控逻辑放到编译区域外。即便能够配置捕获标量，也要考虑它是否引入数据依赖控制流与 GPU 同步。

### 14.5 动态输入不断 Recompile

打开 `recompiles,guards,dynamic` 日志，识别变化的是 Shape、dtype、Module 状态还是 Python 常量。优先稳定业务输入和 Shape Bucket，再评估 `dynamic=True` ，不要盲目提高缓存上限。

### 14.6 显存比 eager 高

编译后端可能保留编译缓存、工作区或 CUDA Graph 私有内存池。分别报告 allocated/reserved 和不同模式；显存预算紧张时降低 Batch、使用 `default` 或缩小编译区域。

### 14.7 后端报错但 eager 正常

用最小输入与 `fullgraph=True` 缩小问题，记录 PyTorch、Triton、驱动、设备和完整错误。 `torch._dynamo.config.suppress_errors=True` 只适合临时回退验证，不应隐藏正确性问题或长期替代修复。

### 14.8 多卡只有部分 rank 变慢

各 rank 的 Shape、控制流或环境不同可能生成不同编译版本，最慢 rank 又会拖住 Collective。保存每个 rank 的日志和 Trace，统一 Bucket、随机分支和预热过程。

## 十五、工程实践：上线前的五道门

1. \*\*Correctness Gate\*\*：输出、梯度和训练收敛满足容差。
2. \*\*Compile Gate\*\*：Graph Break、版本数和首次编译在预算内。
1. \*\*Performance Gate\*\*：无 Profile 的端到端 P50/P99、吞吐有净收益。
2. \*\*Memory Gate\*\*：峰值显存和长时间缓存增长可接受。
1. \*\*Fallback Gate\*\*：新 Shape、后端失败或设备不支持时可安全回到 eager。

推荐将编译缓存 Key 显式关联到模型版本、PyTorch/后端版本、设备架构、dtype 和 Shape Bucket。升级任一层后重新做正确性与性能回归，不能假设旧缓存或旧结论继续成立。

## 十六、面试题与答案

### 题1：torch.compile 的核心流水线是什么？

TorchDynamo 从 Python 字节码捕获 FX Graph，AOTAutograd 为训练生成可编译的前后向图，TorchInductor 做融合、调度和代码生成，GPU 后端常借助 Triton或高性能库执行。

### 题2：Graph Break 与 Recompile 有什么区别？

Graph Break 是一次调用中某段无法继续捕获而切回 Python；Recompile 是下一次调用的 Guard 不满足，需要为新条件生成另一编译版本。二者都会影响性能，但诊断路径不同。

### 题3：为什么 Guard 必不可少？

生成代码依赖 Shape、dtype、设备、常量和状态等假设。Guard 在复用前验证这些假设，保证专门化代码仍然正确。

### 题4：dynamic=True 是否一定更好？

不是。它可能减少 Shape 变化造成的重编译，但也可能限制静态特化或 Autotune。生产上应以版本数量、冷启动、稳态和 P99 联合判断。

### 题5：为什么必须分开测首次调用与稳态？

首次调用包含捕获、代码生成、编译和可能的 Autotune，稳态则主要复用缓存。混在一起既不能预测长生命周期服务，也不能反映短生命周期任务的真实总成本。

### 题6：什么时候 Regional Compilation 更合适？

整图编译延迟太高，而模型由多个重复 Block 组成时。编译一个共享结构区域可以复用结果，以较小冷启动换取主要稳态收益。

### 题7：reduce-overhead 主要解决什么问题？

它面向 Python/Kernel Launch 开销占比较高的场景，可能结合 CUDA Graph 类优化降低 Runtime 开销；它不保证计算密集型大 GEMM 自动更快，也可能增加显存。

### 题8：如何证明收益来自 Fusion？

比较 eager 与 compiled 稳态时间线、Kernel 数量、中间 Tensor/显存流量和 CPU Launch 间隙。只看端到端 Speedup 不能证明因果。

### 题9：DDP 中为何要关注每个 rank 的编译日志？

不同 rank 的 Shape 或分支可能导致不同编译版本和冷启动，Collective 又会等待最慢 rank。只看 rank 0 会漏掉尾部瓶颈。

### 题10：编译收益的盈亏平衡点如何计算？

若 compiled 稳态比 eager 快， $N^*=T_c/($  $T_e-T_s$)；还要把多个 Shape 版本、实例重启和 Autotune 纳入 $T_c$ 。

## 十七、课后练习

1. 用 Level 0 计算你的服务在 1、4、16 个 Shape 版本下的盈亏平衡次数。
2. 在 Level 1 的 `forward` 中加入 `.item()` 和数据依赖分支，观察 Graph Break。
1. 对比 `dynamic=false/true` 的编译次数、首次延迟与稳态时间。
2. 对比三种编译模式，并同时记录峰值显存，不只比较吞吐。
1. 用 Nsight Systems 比较 eager/compiled 的 Kernel 数量和 CPU Launch 空洞。
2. 将完整模型编译改为只编译重复 Block，比较冷启动和稳态。
1. 构造四个序列长度 Bucket，计算真实流量分布下的加权 P99。
2. 在双卡 DDP 中让两个 rank 使用不同 Batch，观察 Recompile 与同步等待。

## 十八、Checklist

### 正确性

- eager 与 compiled 使用相同权重、输入、随机种子和模式。
- 验证输出、梯度或训练收敛，而非只验证程序能跑。
- 容差与 dtype 匹配，异常差异有最小复现。

### 编译行为

- 记录 PyTorch、Triton、驱动、设备和编译模式。
- 检查 Graph Break、Guard Failure 和 Recompile。
- 动态输入采用有限 Bucket 或有依据的动态 Shape。
- 高详细度日志只在最小窗口开启。

### 性能

- 首次编译、预热和稳态分开计时。
- CUDA 计时边界正确同步。
- 计算盈亏平衡次数和实例生命周期总收益。
- 同时报告 P50、P95/P99、吞吐、Kernel 数和峰值显存。
- 最终收益由无 Profile 的端到端基准复验。

### 部署

- 编译边界不包含 Tokenizer、文件、网络或日志 I/O。
- 模型版本、设备架构、dtype 与 Shape Bucket 纳入缓存策略。
- 新 Shape、后端失败和资源不足时有 eager 回退。
- 多卡检查每个 rank，不把单卡收益直接外推到集群。
- 4090 标记为 Ada，5090 标记为 Blackwell。
- FP8/FP4 等结果只在真实支持硬件上验证。