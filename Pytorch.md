# Pytorch

[（PyTorch笔试）快速全面掌握PyTorch框架（必背）—— 30 道PyTorch笔试题及\.pdf](https://lcnipys9u0xh.feishu.cn/wiki/E9jhwbmJtiE7Tqk26TPcairwnDh?from=from_copylink)

# tensor操作

## 创建tensor

创建 Tensor 是所有工作的起点。你可以从现有数据创建，也可以生成特定分布的随机数。

- **从列表/数组创建**：`torch.tensor([1, 2, 3])`

- **创建全 0/1 张量**：`torch.zeros((3, 3))`, `torch.ones((2, 3))`

- **随机初始化**：

    - `torch.rand()`: 生成 $\[0, 1\)$ 均匀分布。

    - `torch.randn()`: 生成均值为 0，方差为 1 的正态分布。

- **序列生成**：`torch.arange(0, 10, 2)` \(生成 0, 2, 4, 6, 8\)。

- **连续序列生成**：`torch.arange(12)` \(生成 0\-11\)。



## tensor精度转换

转为浮点数：



## 形状与维度操作 \(Reshaping\)



## 数学运算

PyTorch 的运算非常直观，支持标量运算、向量运算和矩阵运算。

- **基础算术**：`a + b`, `a - b`, `a * b` \(逐元素相乘\)。

- **矩阵乘法**：

    - `torch.mm(a, b)`: 二维矩阵相乘。

    - `torch.matmul(a, b)` 或 **`a @ b`**: 支持高维广播的矩阵乘法（推荐）。

- **聚合操作**：

    - `torch.sum()`, `torch.mean()`, `torch.max()`。

    - 注意 `dim` 参数：`torch.sum(a, dim=0)` 表示压缩行，对列求和。

---

## 索引、切片与拼接

这些操作用于提取或组合数据。

- **索引与切片**：与 Python 列表一致，如 `tensor[:, 1:3]` 提取所有行的第 2 到 3 列。

- **拼接 \(Join\)**：

    - `torch.cat([a, b], dim=0)`：在现有的维度上拼接。

    - `torch.stack([a, b], dim=0)`：在新维度上堆叠（会增加一维）。

- **分割 \(Split\)**：`torch.chunk()` 或 `torch.split()`。

---

## 设备管理 \(GPU 加速\)

这是 PyTorch 强大的原因。将 Tensor 移动到 GPU 只需要一行代码：

Python

```Plain Text
device = "cuda" if torch.cuda.is_available() else "cpu"
x = x.to(device) # 将张量移动到 GPU
```

---

## 自动求导 \(Autograd\)

如果你想让某个 Tensor 参与反向传播，需要设置 `requires_grad=True`。

$y = x^2 \implies \frac{dy}{dx} = 2x$

Python

```Plain Text
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
y.backward()
print(x.grad) # 输出为 4.0
```

# 资源核算





# 数据处理

## 数据加载

Dataset

dataLoader

## 数据预处理



# 神经网络构建

nn\.Module

nn\.sequential

# 模型训练

损失函数

优化器

自动求导

前向传播

反向传播

激活函数，早停

参数更新

模型剪枝

# 模型管理

\.to\(device\)

torch\.ssave

torch\.load保存和加载模型参数



# 并行计算











