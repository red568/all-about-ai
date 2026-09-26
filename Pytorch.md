# Pytorch

[（PyTorch笔试）快速全面掌握PyTorch框架（必背）—— 30 道PyTorch笔试题及\.pdf](https://lcnipys9u0xh.feishu.cn/wiki/E9jhwbmJtiE7Tqk26TPcairwnDh?from=from_copylink)

# 一、tensor操作

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


# 二、数据处理

## 数据加载

Dataset

dataLoader

## 数据预处理



# 三、模型构建
## 模型定义

在 PyTorch 中，模型通常通过继承 `nn.Module` 来定义。核心步骤：

1. **继承 `nn.Module`。**
    
2. **在 `__init__` 中定义网络层（如全连接层、卷积层等）。**
    
3. **在 `forward` 中定义前向传播逻辑。**
    
4. **实例化模型后，可直接调用 `model(input)`，会自动触发 `forward`。**
    
```
import torch
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
model = Net()
print(model)
```


## 训练过程

一个典型的训练流程包括：

1. **准备数据**：使用 `torch.utils.data.Dataset` 和 `DataLoader`。
    
2. **定义模型**：如上。
    
3. **定义损失函数**：如 `nn.CrossEntropyLoss()`。
    
4. **定义优化器**：如 `torch.optim.Adam(model.parameters(), lr=1e-3)`。
    
5. **训练循环**：
    
    - 前向传播：`output = model(input)`
        
    - 计算损失：`loss = criterion(output, target)`
        
    - 反向传播：`loss.backward()`
        
    - 更新参数：`optimizer.step()`
        
    - 梯度清零：`optimizer.zero_grad()`
        
6. **评估**：设置 `model.eval()`，用 `torch.no_grad()` 关闭梯度。
    
7. **保存/加载**：`torch.save(model.state_dict(), path)` 和 `model.load_state_dict(torch.load(path))`。
    
8. **设备管理**：`model.to(device)`，数据也 `.to(device)`。

```
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(10):
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch}, Loss: {loss.item()}')
```


# 四、`torch.nn` 常见函数与模块

`torch.nn` 提供了构建神经网络所需的各种层、损失函数和容器。

### 1. 线性层（线性变换，全连接层）后面通常会跟着激活函数处理

- `nn.Linear(in_features, out_features, bias=True)`：全连接层。
    

`nn.Linear` 内部计算的是一个**仿射变换（线性变换 + 偏置）**，也就是：

$$y = xW^T + b$$

其中：

- 输入 xx 的形状：`(*, in_features)`，最后一维必须等于 `in_features`
    
- 权重 `weight` 的形状：`(out_features, in_features)`
    
- 偏置 `bias` 的形状：`(out_features,)`
    
- 输出 yy 的形状：`(*, out_features)`，除最后一维外其他维度保持不变
### 2. 卷积层

- `nn.Conv1d`, `nn.Conv2d`, `nn.Conv3d`：一维/二维/三维卷积。
    
    - 常用参数：`in_channels`, `out_channels`, `kernel_size`, `stride`, `padding`。
        

### 3. 池化层

- `nn.MaxPool2d(kernel_size, stride=None, padding=0)`：最大池化。
    
- `nn.AvgPool2d`：平均池化。
    
- `nn.AdaptiveAvgPool2d(output_size)`：自适应平均池化。
    

### 4. 激活函数

- `nn.ReLU()`、`nn.LeakyReLU()`、`nn.Sigmoid()`、`nn.Tanh()`、`nn.Softmax(dim=None)`、`nn.GELU()` 等。
    

### 5. 归一化层

- `nn.BatchNorm1d/2d/3d`：批归一化。
    
- `nn.LayerNorm`：层归一化。
    
- `nn.GroupNorm`：组归一化。
    

### 6. 正则化

- `nn.Dropout(p=0.5)`：随机失活。
    
- `nn.Dropout2d`：针对卷积特征图。
    

### 7. 循环神经网络

- `nn.RNN`、`nn.LSTM`、`nn.GRU`：循环层。
    
- `nn.RNNCell`、`nn.LSTMCell`、`nn.GRUCell`：单步版本。
    

### 8. 嵌入层

- `nn.Embedding(num_embeddings, embedding_dim)`：词嵌入。
    

### 9. 损失函数

- `nn.CrossEntropyLoss()`：交叉熵（内含 Softmax）。
    
- `nn.MSELoss()`：均方误差。
    
- `nn.BCELoss()`、`nn.BCEWithLogitsLoss()`：二分类损失。
    
- `nn.NLLLoss()`：负对数似然。
    
- `nn.L1Loss()`：绝对值损失。
    

### 10. 容器

- `nn.Sequential(*args)`：按顺序组合层。
    
- `nn.ModuleList([...])`：像列表一样存储模块，但正确注册参数。
    
- `nn.ModuleDict({...})`：字典形式存储模块。
    

### 11. 其他常用

- `nn.Flatten(start_dim=1)`：展平。
    
- `nn.Identity()`：占位层，不做任何操作。
    
- `nn.Parameter(tensor)`：可学习参数。
    

### 12. 初始化

- `nn.init.xavier_uniform_(tensor)`、`nn.init.kaiming_normal_(tensor)` 等。


# 模型管理

\.to\(device\)

torch\.ssave

torch\.load保存和加载模型参数



# 并行计算











