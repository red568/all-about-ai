# Hugging face

在 Hugging Face 中，处理任务通常遵循 **"AutoClass"** 模式。你不需要记住每个模型具体的类名，代码会自动识别。



# Transformers

**`transformers`**: 核心库。用于加载、训练和推理各类预训练模型（BERT, GPT, Llama, CLIP 等）。



## GenerationConfig

关键模型参数配置：`top_p`, `temperature`, `repetition_penalty`

```Python
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch

model_id = "Qwen/Qwen2-1.5B-Instruct"  # 以千问为例，适合学习
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")

# 1. 定义生成配置
config = GenerationConfig(
    max_new_tokens=512,      # 最大生成长度
    do_sample=True,          # 必须设为 True 才能使用 temperature/top_p
    temperature=0.7,         # 控制随机性
    top_p=0.9,               # 核心采样
    repetition_penalty=1.1,  # 惩罚重复词，1.1-1.2 是常用区间
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id
)

# 2. 准备输入
prompt = "请解释一下什么是 Agent 的规划能力 (Planning)。"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

# 3. 传入 config 进行生成
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        generation_config=config  # 传入配置对象
    )

# 4. 解码输出
response = tokenizer.decode(output_ids[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
print(response)
```





## 参数解析HfArgumentParser

```Python
@dataclass
class ModelArguments:
    """
    通过声明式语法定义了模型加载和配置所需的参数。
    model_name_or_path:模型名字或者路径，必须提供。可以是预训练模型的名字（如"gpt2"）或者本地路径。
    tokenizer_name_or_path:分词器名字或者路径，默认为None。如果不提供，则使用model_name_or_path指定的模型对应的分词器。
    load_in_8bit:是否以8位模式加载模型，默认为False。
    load_in_4bit:是否以4位模式加载模型，默认为False。
    cache_dir:预训练模型下载和缓存的目录，默认为None。
    model_revision:使用的模型版本，可以是分支名、标签名或者提交ID，默认为"main"。
    hf_hub_token:用于登录Hugging Face Hub的认证令牌，默认为None。
    use_fast_tokenizer:是否使用快速分词器（基于tokenizers库），默认为False。
    torch_dtype:覆盖默认的torch.dtype并以该dtype加载模型，如果传入"
    """

    #metadata 元数据字典。它不会改变 Python 运行时的逻辑，但会被 Hugging Face 的 HfArgumentParser 读取。
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The tokenizer for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    load_in_8bit: bool = field(default=False, metadata={"help": "Whether to load the model in 8bit mode or not."})
    load_in_4bit: bool = field(default=False, metadata={"help": "Whether to load the model in 4bit mode or not."})
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    hf_hub_token: Optional[str] = field(default=None, metadata={"help": "Auth token to log in with Hugging Face Hub."})
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    device_map: Optional[str] = field(
        default="auto",
        metadata={"help": "Device to map model to. If `auto` is passed, the device will be selected automatically. "},
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code when loading a model from a remote checkpoint."},
    )

    def __post_init__(self):
        if self.model_name_or_path is None:
            raise ValueError("You must specify a valid model_name_or_path to run training.")

parser = HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments, ScriptArguments))
```

**`HfArgumentParser`**: 这是 `transformers` 对原生 `argparse` 的高级封装。

**元组参数 ****`(ModelArguments, ...)`**: 这里传入的是四个 **Dataclass（数据类）**。每个类代表一类特定的参数：



### 它的工作流程

1. **自动关联**：它会扫描这些 Dataclass 中的变量名和类型注解（Type Hints）。

2. **生成命令行参数**：如果你在 `ModelArguments` 里定义了 `model_name_or_path: str`，解析器会自动生成 `--model_name_or_path` 这个命令行选项。

3. **自动转换**：它能自动处理类型。比如你在命令行输入 `--load_in_4bit True`，它会自动将其转为布尔值 `True` 而不是字符串。

## 加载分词器 \(Tokenizer\)

分词器将文本转换为模型能理解的数字。

```Python
from transformers import AutoTokenizer

model_id = "shibing624/text2vec-base-chinese" # 举例一个中文模型
tokenizer = AutoTokenizer.from_pretrained(model_id)

text = "深度学习改变世界"
inputs = tokenizer(text, return_tensors="pt")
print(inputs)
```

## 加载模型 \(Model\)

根据任务选择不同的 AutoModel 类（如 `AutoModelForCausalLM` 用于对话，`AutoModelForSequenceClassification` 用于分类）。

```Python
from transformers import AutoModelForCausalLM

# device_map="auto" 会自动分配到 GPU (如果安装了 PyTorch + CUDA)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="auto", 
    torch_dtype="auto"
)
```





## 流式输出 \(Streaming\)

在 Agent 开发中，为了用户体验，你必须掌握流式输出，而不是等几秒钟直接弹出整个结果。

```Plain Text
from transformers import TextStreamer

# 创建流式器
streamer = TextStreamer(tokenizer, skip_prompt=True)

# 在 generate 中加入 streamer
_ = model.generate(
    **inputs,
    generation_config=config,
    streamer=streamer  # 实时在控制台打印生成的字符
)
```

## 高级抽象：Pipeline \(流水线\)

如果你只想快速实现功能，不想管张量维度，用 `pipeline`：

```Plain Text
from transformers import pipeline

# 创建一个情感分析流水线
classifier = pipeline("sentiment-analysis", model="uer/roberta-base-finetuned-jd-binary-chinese")
result = classifier("这个模型非常好用！")
print(result)
```



## 对大模型解码环路进行干预

### Stopping Criteria：精准截断生成的“刹车”

在构建 Agent 时，LLM 往往需要作为“思考者”输出内容。如果模型输出 `Action: [search]` 后不停止，而是继续自顾自地伪造 `Observation: [results...]`，就会破坏 ReAct 循环。

**技术实现：**

你需要继承 `StoppingCriteria` 基类，并在 `call` 方法中定义逻辑。如果返回 `True`，模型停止生成。

```Plain Text
from transformers import StoppingCriteria, StoppingCriteriaList

class StopOnWordCriteria(StoppingCriteria):def __init__(self, target_word, tokenizer):
        self.target_word = target_word
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:# 将当前的 token id 序列解码回文本
        generated_text = self.tokenizer.decode(input_ids[0])
        # 检查是否以目标词结尾
        return generated_text.endswith(self.target_word)

# 使用方式
stop_criteria = StoppingCriteriaList([StopOnWordCriteria("Observation:", tokenizer)])
outputs = model.generate(**inputs, stopping_criteria=stop_criteria)
```

> **工程师视角：** 在生产环境中，为了性能，我们通常不直接解码整个序列，而是将 `Observation:` 预先转换为 Token ID 序列进行字节级匹配。
> 
> 

---

### Repetition Penalty 的副作用：过度补偿陷阱

`repetition_penalty` 的数学原理是修改 Logits。如果某个词已经出现过，其得分 $z\_i$ 会被减小：

- 如果 $z\_i \> 0$，则 $z\_i \\leftarrow z\_i / \\text\{penalty\}$

- 如果 $z\_i \< 0$，则 $z\_i \\leftarrow z\_i \\times \\text\{penalty\}$

**为什么 \> 1\.5 会崩坏？**

1. **语法降级：** 像“的”、“is”、“the”这种高频虚词，一旦被惩罚，模型会为了避开它们而被迫选择语法错误的词，导致句子支离破碎。

2. **事实错误：** 如果你在讨论“深度学习”，模型在说了一次“深度学习”后被强行禁止再说第二次，它可能会胡诌出一个不存在的术语。

3. **工程师建议：** 通常设在 **1\.05 \- 1\.15** 之间。如果模型依然复读，应该检查 **Prompt 的质量** 或 **训练数据是否有严重的噪声循环**，而不是无限制提高惩罚值。

---

### Logits Processor：上帝之手，强制格式化

这是最强大的接口。它允许你在计算 Softmax 采样之前，手动修改所有 Token 的原始概率值（Logits）。

**应用场景：**

- **强制 JSON：** 在生成 JSON 时，如果当前位置必须是双引号或括号，你可以把其他所有 Token 的概率设为 $\-\\infty$。

- **关键词增强：** 强制模型必须使用某些词汇。

**技术实现示例（强制下一步输出数字）：**

Python

```Plain Text
from transformers import LogitsProcessor, LogitsProcessorList

class ForceNumericLogitsProcessor(LogitsProcessor):def __init__(self, tokenizer):# 找出所有数字 token 的 ID
        self.allowed_ids = [i for i, t in tokenizer.get_vocab().items() if t.isdigit()]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:# 创建一个全负无穷的遮罩
        mask = torch.full_like(scores, float("-inf"))
        # 只放行数字 token
        mask[:, self.allowed_ids] = 0return scores + mask

# 使用方式
processors = LogitsProcessorList([ForceNumericLogitsProcessor(tokenizer)])
outputs = model.generate(**inputs, logits_processor=processors)
```





# datasets

**`datasets`**: 用于高效访问和处理各种模态的数据集，支持流式加载（大数据集不占用内存）。

Hugging Face 的 `datasets` 库不仅仅是一个下载工具，它是一套基于 **Apache Arrow** 内存格式的系统，核心优势是：**即便数据集有 100GB，你的内存也不会爆**。

---

## 核心哲学：内存映射（Memory Mapping）

传统的 `pandas` 或 `list` 处理数据时，会将所有数据加载进内存。而 `datasets` 使用磁盘映射，只在需要时加载特定行。



## 常用操作\-加载与切片



```Plain Text
from datasets import load_dataset

# 1. 加载远程数据集 (自动缓存)
dataset = load_dataset("shibing624/medical", "pretrain") # 示例：医学预训练数据# 2. 加载本地文件 (JSON, CSV, Parquet)# 常用场景：加载你为 Agent 准备的对话微调数据
dataset = load_dataset("json", data_files="my_trajectories.jsonl", split="train")

# 3. 查看数据结构
print(dataset[0]) # 像操作列表一样操作
print(dataset.column_names)
```

---

## 性能杀手锏：`.map()` 方法

这是 `datasets` 的灵魂。如果你用 `for` 循环处理百万级数据，你会等死；而使用 `.map()`，你可以调用**多进程**并行。

### 场景：将原始文本转换为模型输入（Tokenization）

```Plain Text
def preprocess_function(examples):
# 处理逻辑：将 Prompt 和 Response 拼接并分词
# 这里的 examples 是一个 batch
return tokenizer(
        examples["instruction"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

# 使用多进程加速
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,           # 关键：批量处理
    num_proc=8,             # 关键：开启 8 个进程
    remove_columns=dataset.column_names # 移除原始列，只保留 token_ids
)
```

---

## 进阶技能：数据流（Streaming）

如果你在处理类似 **Common Crawl** 这样数个 TB 的数据集，你根本无法下载完整文件。

```Plain Text
# 开启流式加载：不下载，随用随取
streamed_dataset = load_dataset("oscar", "unshuffled_deduplicated_zh", streaming=True)

# 只能迭代获取for example in streamed_dataset["train"].take(5):
    print(example)
```

---

## 算法工程师的工程实践：RAG 与向量索引

`datasets` 库直接集成了 **FAISS**，这对你做 RAG（检索增强生成）非常有用。

```Plain Text
# 假设你有一列 'embeddings'
dataset.add_faiss_index(column="embeddings")

# 实时检索最近邻
scores, retrieved_examples = dataset.get_nearest_examples(
    "embeddings", query_vector, k=5
)
```









# peft

**`peft`**: \(如果你要做微调\) 专门用于 LoRA、QLoRA 等高效微调技术的库。

## LoRA

### 第一步：配置 LoRA 参数

```Plain Text
from peft import LoraConfig, get_peft_model

peft_config = LoraConfig(
    task_type="CAUSAL_LM", 
    inference_mode=False, 
    r=8,                    # 秩
    lora_alpha=32,          # 缩放系数，通常是 r 的 2 倍
    lora_dropout=0.1,       # 防止过拟合
    target_modules=["q_proj", "v_proj"] # 核心：指定对哪些层注入 LoRA（通常是 Attention 层）
)
```

### 第二步：将 LoRA 注入模型

```Plain Text
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B", device_map="auto")

# 包装模型
model = get_peft_model(model, peft_config)

# 打印可训练参数量
model.print_trainable_parameters()
# 你会发现可训练参数占比通常不到 1%
```

---

## 进阶：QLoRA \(4\-bit 量化微调\)

如果你连 24GB 显存都没有，只有 12GB 甚至更少，那么你需要 **QLoRA**。它通过将预训练模型权重压缩到 4\-bit，同时保持 LoRA 部分为高精度来实现。

```Plain Text
from transformers import BitsAndBytesConfig

# 1. 定义 4-bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 2. 加载量化模型
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
```

---

## 算法工程师的工程实践：合并权重 \(Merge\)

训练完 LoRA 后，你会得到一个几十 MB 的 `adapter_model.bin`。在部署（如使用 vLLM）时，为了不增加推理延迟，通常需要将其合并回原模型。

Python

```Plain Text
# 加载基础模型和适配器
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
model = PeftModel.from_pretrained(base_model, lora_path)

# 合并并保存
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./my_final_model")
```







# 上传模型

进入**第四阶段：生态闭环——Hugging Face Hub、Gradio 演示与 vLLM 高速推理部署**。

作为算法工程师，你的模型如果只躺在服务器的文件夹里是没有价值的。你需要学会如何**托管模型**、**构建可视化 Demo** 以及**实现工业级的高性能推理**。

---

## Hugging Face Hub：模型的“家”

Hub 不仅是下载中心，更是版本控制系统。你需要掌握通过 Git 或 SDK 推送模型。

### 技术实现：推送模型

```Plain Text
from huggingface_hub import login, HfApi

# 1. 登录（需要从 HF 设置页面获取 Write Token）
login("your_hf_token")

# 2. 推送模型（通常在训练脚本末尾）
# 会自动上传权重文件、配置文件和 Tokenizer
model.push_to_hub("your_username/medical-llama-3-8b")
tokenizer.push_to_hub("your_username/medical-llama-3-8b")
```

---

## Gradio：10 分钟搭建 Web 界面

面试时，展示一个可以实时交互的网页比展示代码更有说服力。Hugging Face **Spaces** 免费托管 Gradio 应用。

### 极简实现：对话机器人

```Plain Text
import gradio as gr
from transformers import pipeline

pipe = pipeline("text-generation", model="Qwen/Qwen2-1.5B-Instruct")

def predict(message, history):# 将对话历史转化为模型格式
    response = pipe(message, max_new_tokens=128)[0]['generated_text']
    return response

# 创建聊天界面
gr.ChatInterface(predict).launch()
```



