# argparse.ArgumentParser 详解

## 1. argparse 是做什么的

`argparse` 是 Python 标准库，用于解析命令行参数。它的核心职责是：

1. 定义「程序接受哪些参数、参数长什么样」（类型、默认值、是否必填、取值约束）。
2. 把用户从命令行传入的原始字符串（`sys.argv[1:]`）解析成结构化的对象。
3. 自动生成 `--help` 帮助信息。

一句话：**声明式地描述 CLI 接口，然后一行解析得到结果。**

## 2. 三步基本流程

```python
import argparse

# ① 创建解析器
parser = argparse.ArgumentParser(description="MiniSGL Server Arguments")

# ② 逐个声明参数
parser.add_argument("--dtype", type=str, default="auto", help="...")

# ③ 解析命令行参数
args = parser.parse_args(sys.argv[1:])

print(args.dtype)  # 直接通过属性名访问
```

对应到 [args.py:69](python/minisgl/server/args.py#L69)、[args.py:228](python/minisgl/server/args.py#L228)：

```python
parser = argparse.ArgumentParser(description="MiniSGL Server Arguments")  # ①
parser.add_argument(...)  # ② 中间一大堆
kwargs = parser.parse_args(args).__dict__.copy()  # ③
```

## 3. ArgumentParser 构造函数常用参数

```python
parser = argparse.ArgumentParser(
    description="...",   # 一句话说明程序用途，显示在 --help 顶部
    prog="...",          # 程序名，默认取 sys.argv[0]
    add_help=True,       # 是否自动添加 -h/--help，默认 True
)
```

本项目只用到了 `description`，其余保持默认。

## 4. add_argument 的核心参数

`add_argument` 是最重要、也最灵活的方法。下面是 `args.py` 里出现过的所有参数。

### 4.1 名称：短选项、长选项、多个别名

```python
parser.add_argument(
    "--model-path",   # 长选项，生成的属性名是 model_path（- 转 _）
    "--model",        # 别名，同一个参数可以同时接受两个写法
    type=str,
    required=True,
)
```

- `--model-path` 和 `--model` 是**同一个参数的两个名字**，用哪个都行。
- argparse 会自动把 `--model-path` 里的 `-` 转成下划线 `_`，所以解析结果里属性名是 `args.model_path`。
- 多个别名都写在最前面，后面才跟 `type`/`help` 等关键字参数。

### 4.2 `type`：类型转换

```python
parser.add_argument("--tp-size", type=int, default=1)
parser.add_argument("--memory-ratio", type=float, default=...)
parser.add_argument("--dtype", type=str, default="auto")
```

- 命令行传进来的一定是字符串，`type` 指定把它转成什么类型。
- 也可以传**任意可调用对象**，比如本项目用函数做校验（见 [args.py:192](python/minisgl/server/args.py#L192)）：

```python
parser.add_argument(
    "--attention-backend",
    "--attn",
    type=validate_attn_backend,   # 一个函数：既校验合法性，又返回处理后的值
    default=ServerArgs.attention_backend,
)
```

这里的 `validate_attn_backend` 收到字符串，校验是否合法，非法则抛错，合法则返回规范化的值。

### 4.3 `default`：默认值

```python
parser.add_argument("--host", type=str, dest="server_host", default=ServerArgs.server_host)
```

- 用户没传该参数时使用默认值。
- 默认值通常直接引用配置类的字段默认值（`ServerArgs.server_host`），保持「单一数据源」。

### 4.4 `required`：是否必填

```python
parser.add_argument("--model-path", "--model", type=str, required=True)
```

- `required=True` 表示用户必须提供，否则解析时报错退出。
- 默认 `False`，即可选参数。

### 4.5 `choices`：取值约束

```python
parser.add_argument(
    "--dtype",
    type=str,
    default="auto",
    choices=["auto", "float16", "bfloat16", "float32"],
)
```

- 限制参数只能取列表中的值，传了非法值会直接报错，错误信息里会列出所有合法选项。
- 本项目还用它配合动态列表，例如 [args.py:210](python/minisgl/server/args.py#L210)：

```python
choices=SUPPORTED_CACHE_MANAGER.supported_names()
```

### 4.6 `action`：布尔开关

命令行里最常见的布尔参数，不用写 `--xxx true/false`，而是「出现了就生效」。

```python
# store_true：出现该 flag 时，属性值为 True；不出现为 False
parser.add_argument("--shell-mode", action="store_true")

# store_false：出现该 flag 时，属性值为 False；不出现为 True
parser.add_argument("--disable-pynccl", action="store_false", dest="use_pynccl")
```

区别总结：

| action | 出现 flag 时 | 未出现时 |
|--------|-------------|----------|
| `store_true` | `True` | `False` |
| `store_false` | `False` | `True` |

`--disable-pynccl` 用 `store_false` 很有代表性：默认 `use_pynccl=True`，用户加上 `--disable-pynccl` 就把它翻成 `False`，语义上「禁用」正好对应「关掉」。（前置断言见 [args.py:125](python/minisgl/server/args.py#L125) `assert ServerArgs.use_pynccl == True`。）

### 4.7 `dest`：指定目标属性名

```python
parser.add_argument(
    "--max-running-requests",
    type=int,
    dest="max_running_req",      # 解析结果存到这个属性名，而不是默认的 max_running_requests
    default=ServerArgs.max_running_req,
)
```

- 默认情况下，属性名由第一个长选项名去掉 `--` 并把 `-` 转 `_` 得到。
- 当你希望「命令行名字」和「代码里的字段名」不同时，用 `dest` 显式指定目标字段名。这样 `--max-running-requests` 会落到 `args.max_running_req`，正好对应 `ServerArgs` 的字段。

### 4.8 `help`：帮助文本

```python
help="The path of the model weights. This can be a local folder or a Hugging Face repo ID."
```

- 每个参数都应写清楚含义，自动出现在 `--help` 输出里。

## 5. 解析结果如何变成配置对象

[args.py:228](python/minisgl/server/args.py#L228) 之后的整体套路：

```python
# ① 解析成 Namespace，再转成字典，方便后续增删字段
kwargs = parser.parse_args(args).__dict__.copy()

# ② 解析后做二次处理（合并、改写、下载模型等）
run_shell |= kwargs.pop("shell_mode")       # pop：取出并删除该字段
if run_shell:
    kwargs["cuda_graph_max_bs"] = 1
    kwargs["max_running_req"] = 1
    kwargs["silent_output"] = True

# ③ 字段名对齐后，用 **kwargs 直接构造 frozen dataclass
result = ServerArgs(**kwargs)
return result, run_shell
```

要点：

- `parse_args()` 默认返回一个 `Namespace` 对象（属性访问）。调用 `. __dict__` 拿到底层字典，方便后续 `pop`、增删。
- `pop("shell_mode")` 既取出了值又把它从字典删掉，避免 `ServerArgs` 里没有 `shell_mode` 字段导致构造时报错。
- 最终 `ServerArgs(**kwargs)` 把字典展开成关键字参数，构造出强类型的 `@dataclass(frozen=True)` 配置对象。这是「解析出弱类型字符串 → 落到强类型配置对象」的关键一步。

## 6. 完整的最小示例

```python
import argparse

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Demo server")

    parser.add_argument("--model-path", "--model", type=str, required=True)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="auto",
                        choices=["auto", "float16", "bfloat16"])
    parser.add_argument("--shell-mode", action="store_true")

    args = parser.parse_args(argv)
    return args

if __name__ == "__main__":
    cfg = parse_args([
        "--model", "meta-llama/Llama-3.1-8B",
        "--tp-size", "2",
        "--shell-mode",
    ])
    print(cfg.model_path)   # meta-llama/Llama-3.1-8B
    print(cfg.tp_size)      # 2 (int)
    print(cfg.dtype)        # auto
    print(cfg.shell_mode)   # True
```

运行 `python demo.py --help` 会得到 argparse 自动生成的帮助文档。

## 7. 常用参数速查表

| 参数 | 作用 | 本项目示例 |
|------|------|-----------|
| `description` | 程序说明，显示在 `--help` | `"MiniSGL Server Arguments"` |
| `type` | 类型转换 / 校验函数 | `int`、`float`、`validate_attn_backend` |
| `default` | 默认值 | `ServerArgs.memory_ratio` |
| `required` | 是否必填 | `--model-path` |
| `choices` | 取值白名单 | `["auto", "float16", ...]` |
| `action` | 布尔开关 | `store_true`、`store_false` |
| `dest` | 指定目标属性名 | `dest="max_running_req"` |
| `help` | 帮助文本 | 每个参数都有 |

## 8. 容易踩的坑

1. **`-` 会变 `_`**：`--model-path` 解析后属性是 `model_path`，不是 `model-path`。
2. **多个别名写最前面**：`add_argument("--a", "--b", type=..., ...)`，关键字参数必须在所有位置别名之后。
3. **布尔值不要用 `type=bool`**：`type=bool` 会把任何非空字符串转成 `True`（包括 `"false"`），正确做法是用 `action="store_true"` / `store_false`。
4. **`store_false` 语义反转**：出现 flag 时值是 `False`，别和 `store_true` 记混。
5. **`parse_args` 返回 `Namespace` 而非 dict**：要增删字段记得 `. __dict__.copy()`。
6. **`pop` 掉临时字段**：命令行里有但最终配置对象里没有的字段（如 `shell_mode`、`model_source`、`tensor_parallel_size`），构造 dataclass 前要 `pop` 或 `del` 掉，否则 `**kwargs` 会因多余关键字报 `TypeError`。
