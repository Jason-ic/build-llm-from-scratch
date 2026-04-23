# Build LLM from Scratch

从零开始构建一个 GPT 风格的大语言模型（LLM），使用 PyTorch 实现，涵盖模型架构、训练、文本生成以及加载 GPT-2 预训练权重。

本项目基于 [*Build a Large Language Model (From Scratch)*](https://www.manning.com/books/build-a-large-language-model-from-scratch) 一书实现。

## 项目结构

```
.
├── llm_arch/                  # 模型核心架构
│   ├── gpt.py                 # GPT 模型（Transformer Block、FeedForward、LayerNorm）
│   ├── multi_headattn.py      # 多头因果自注意力
│   ├── self_attn.py           # 简单自注意力
│   └── layernorm.py           # Layer Normalization
├── tokenization/
│   └── tokenizer.py           # 简易分词器 (SimpleTokenizerV1)
├── datasets/
│   └── dataset_loader.py      # 滑动窗口数据集与 DataLoader
├── process_script/
│   ├── download.py            # 下载训练文本数据
│   └── gpt_download.py        # 下载 GPT-2 预训练权重
├── finetuning_classify/
│   ├── finetune.py            # 垃圾短信分类微调脚本
│   ├── dataset.py             # SpamDataset 数据集类
│   ├── classification.py      # 分类推理
│   ├── train.csv              # 训练集
│   ├── validation.csv         # 验证集
│   └── test.csv               # 测试集
├── finetuning_sft/
│   ├── fintune.py             # 指令微调训练脚本
│   ├── data_process.py        # 数据格式化与划分
│   ├── dataset.py             # InstructionDataset 与 collate 函数
│   └── instruction-data.json  # 指令微调数据集
├── reasoning-model/
│   ├── load_model.py          # 加载 Qwen3-0.6B 权重与分词器
│   ├── load_dataset.py        # 下载/读取 MATH-500 测试集
│   ├── evaluate_model.py      # MATH-500 推理与答案抽取/判分
│   └── inference_time_scaling.py  # 推理时扩展：温度/Top-p 采样 + 自一致性投票
├── classify-datasets/
│   └── sms_spam_collection/   # SMS 垃圾短信原始数据集
├── embedding_text/
│   └── the-verdict.txt        # 预训练用文本
├── pre_train.py               # 预训练脚本
├── load_pretrain.py           # 加载 GPT-2 预训练权重并生成文本
├── every_test.py              # 测试脚本
├── config.yaml                # 模型配置 (GPT-2 124M)
└── requirements.txt
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 训练模型

```bash
python pre_train.py
```

在 `the-verdict.txt` 文本上训练 GPT 模型，训练完成后会保存权重到 `model_and_optimizer.pth`，并绘制训练/验证损失曲线。

### 加载 GPT-2 预训练权重

```bash
python load_pretrain.py
```

自动下载 GPT-2 (124M) 预训练权重并进行文本生成。

### 垃圾短信分类微调

```bash
python finetuning_classify/finetune.py
```

在 SMS Spam Collection 数据集上对 GPT-2 进行分类微调，冻结大部分参数，只训练最后一个 Transformer Block 和分类头。

### 指令微调 (Instruction Fine-Tuning)

```bash
python finetuning_sft/fintune.py
```

在指令数据集上对 GPT-2 进行指令微调，使模型学会根据指令生成回答。支持 Alpaca 风格的 instruction/input/output 数据格式。

### 推理模型评测 (MATH-500)

```bash
python reasoning-model/evaluate_model.py
```

加载 Qwen3-0.6B base 模型在 MATH-500 测试集上逐题流式生成答案，自动从 `\boxed{...}` 抽取最终答案，并通过 sympy 做符号等价判分（支持分数、LaTeX、上下标等归一化）。结果按行写入 `math500-<device>.jsonl`。

### 推理时扩展 (Inference-Time Scaling)

```bash
python reasoning-model/inference_time_scaling.py
```

实现三种推理时解码策略，不改权重就提升推理质量：

- **温度采样** — `logits / T` 后经 softmax，通过 `torch.multinomial` 采样
- **Top-p (nucleus) 采样** — 只在累积概率 ≤ p 的最小词集合内采样
- **自一致性投票 (Self-Consistency)** — 同一 prompt 采样多条答案，按 `\boxed{}` 抽取后做多数投票

## 模型配置

| 参数 | 值 |
|------|-----|
| vocab_size | 50257 |
| context_length | 256 (训练) / 1024 (预训练) |
| emb_dim | 768 |
| n_heads | 12 |
| n_layers | 12 |
| drop_rate | 0.1 |

## 主要特性

- 完整的 Transformer 解码器架构（Pre-Norm、GELU、因果掩码）
- 支持 temperature 和 top-k 采样的文本生成
- 兼容 GPT-2 预训练权重加载
- 训练过程可视化（loss 曲线）

## TODO

- [x] 分类微调 — 垃圾短信分类
- [x] 指令微调 (Instruction Fine-Tuning)
- [x] 推理模型评测 — Qwen3 + MATH-500
- [x] 推理时扩展 — 温度/Top-p 采样、自一致性投票
- [ ] Reinforcement Learning from Human Feedback (RLHF)
- [ ] DPO / PPO 等强化学习对齐方法
- [ ] 指令微调数据集构建
- [ ] LoRA / QLoRA 参数高效微调
- [ ] 更多基准测试 (GSM8K, MMLU 等)
