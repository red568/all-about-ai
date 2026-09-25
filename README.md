# All About AI

个人 AI 学习知识库，系统整理大模型（LLM）相关的基础原理、GPU 与算子、推理引擎、部署与微调等主题的笔记与实战经验。

## 内容概览

| 目录 | 说明 |
| --- | --- |
| [0-llm原理](0-llm原理) | 神经网络与 LLM 基础原理 |
| [1-GPU和算子](1-GPU和算子) | GPU 架构、CUDA 编程与算子 |
| [2-推理引擎](2-推理引擎) | 推理引擎、性能指标、SGLang、vLLM |
| [3-部署](3-部署) | Ray、Kubernetes 与 Docker 部署 |
| [4-压缩量化](4-压缩量化) | 模型压缩与量化（待补充） |
| [5-微调](5-微调) | 后训练与微调技术实战 |
| [Python](Python) | Python 语言笔记 |
| [Pytorch.md](Pytorch.md) | PyTorch 框架笔记 |
| [fastAPI](fastAPI) | FastAPI 服务开发 |
| [linux](linux) | Linux 常用知识与命令 |
| [Hugging face.md](Hugging face.md) | Hugging Face 生态使用 |

## 重点内容

- **推理引擎**：`2-推理引擎/` 下包含《SGLang 性能调优全流程拆解》完整 30 章系列笔记，覆盖调度策略、前缀缓存、连续批处理、显存优化、量化、投机解码、分布式推理、高并发调优等主题。
- **GPU 与算子**：GPU 架构、CUDA 编程、Triton 算子（含 `.canvas` 思维导图）。
- **部署与微调**：Ray 分布式、K8s/Docker 部署，以及后训练微调实战。

## 说明

- 笔记以 Markdown 为主，配图存放在各目录下的 `图片和附件/` 文件夹中。
- 目录中的 `.canvas` 文件为 Obsidian 白板/思维导图，可用 Obsidian 打开查看。
