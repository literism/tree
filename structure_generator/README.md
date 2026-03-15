# Structure Tree Generator

自动生成Wikipedia主题的层次化结构树框架的训练和推理系统。

## 概述

本系统训练一个大语言模型，使其能够根据Wikipedia主题的标题和介绍文本，自动生成该主题的多层次结构树框架。生成的结构树可用于初始化层次化分类系统，特别是对test_hard数据集能够显著提升分类效果。

## 目录结构

```
structure_generator/
├── config.py                          # 配置管理
├── prepare_structure_dataset.py       # 数据集准备
├── train_structure_generator.py       # 模型训练
├── inference_structure.py             # 结构树推理
├── run_pipeline.py                    # 完整流程脚本
├── configs/
│   └── default.json                   # 默认配置
└── README.md                          # 本文档
```

## 数据流程

### 输入数据

1. **wiki_structure.jsonl**: Wikipedia文章的结构树
   - 格式: `{"id": "...", "url": "...", "title": "...", "sections": [...]}`
   - 每个section包含: `title`, `level`, `content`, `children`

2. **wiki_intro.jsonl**: Wikipedia文章的介绍段落
   - 格式: `{"id": "...", "url": "...", "title": "...", "intro": "..."}`

3. **dataset_split.json**: 已有的topic划分
   - 包含: `train`, `test_easy`, `test_hard`
   - 这些topics会被排除，不加入训练数据集

### 数据筛选

从wiki数据中筛选符合以下条件的topics：

1. **不在已有的train/test_easy/test_hard中**
2. **结构树节点数 ≥ min_structure_nodes**（默认10）
   - 删除无用title后统计（如"See also", "References"等）
3. **介绍文本长度 ≥ min_intro_length**（默认500字符）

### 训练数据格式

每条训练数据包含：

**Prompt**:
```
TASK: Generate a hierarchical structure tree for the given topic based on its introduction.

TOPIC: [topic标题]

INTRODUCTION:
[介绍文本]

INSTRUCTIONS:
1. Analyze the introduction and identify the main aspects, themes, or categories...
2. Create a multi-level hierarchical structure...
...

STRUCTURE:
```

**Completion**:
```
- Main Category 1 (level 2)
  - Subcategory 1.1 (level 3)
  - Subcategory 1.2 (level 3)
- Main Category 2 (level 2)
  - Subcategory 2.1 (level 3)
    - Sub-subcategory 2.1.1 (level 4)
```

### 输出数据

1. **excluded_topics_intro.json**: 排除topics的介绍文本
   - 用于后续推理时生成这些topics的结构树

2. **train.jsonl / val.jsonl**: 训练和验证数据集

3. **模型**: LoRA适配器

4. **推理结果**: 为train/test_easy/test_hard生成的结构树

## 使用方法

### 1. 完整流程（推荐）

```bash
cd /home/literism/tree/structure_generator

# 使用默认配置运行完整流程
python3 run_pipeline.py

# 使用自定义配置
python3 run_pipeline.py --config configs/custom.json

# 跳过某些步骤
python3 run_pipeline.py --skip_prepare  # 跳过数据准备
python3 run_pipeline.py --skip_training  # 跳过训练
python3 run_pipeline.py --skip_inference  # 跳过推理

# 只推理特定split
python3 run_pipeline.py --skip_prepare --skip_training --inference_split test_hard
```

### 2. 分步执行

#### Step 1: 准备数据集

```bash
python3 prepare_structure_dataset.py

# 自定义参数
python3 prepare_structure_dataset.py \
    --min_structure_nodes 15 \
    --min_intro_length 800 \
    --train_size 20000
```

输出：
- `{output_base}/data/train.jsonl`
- `{output_base}/data/val.jsonl`
- `{output_base}/data/excluded_topics_intro.json`
- `{output_base}/data/dataset_stats.json`

#### Step 2: 训练模型

```bash
python3 train_structure_generator.py

# 自定义参数
python3 train_structure_generator.py \
    --num_epochs 5 \
    --batch_size 4 \
    --learning_rate 1e-4
```

输出：
- `{output_base}/models/structure_generator/final/` (最终模型)
- `{output_base}/models/structure_generator/checkpoint-*/` (检查点)

#### Step 3: 推理生成结构树

```bash
python3 inference_structure.py

# 指定split
python3 inference_structure.py --split train
python3 inference_structure.py --split test_easy
python3 inference_structure.py --split test_hard
python3 inference_structure.py --split all  # 默认

# 使用特定模型
python3 inference_structure.py --model /path/to/checkpoint-1000
```

输出：
- `{output_base}/inference/train_structures.json`
- `{output_base}/inference/test_easy_structures.json`
- `{output_base}/inference/test_hard_structures.json`

## 配置说明

### 主要配置项

```json
{
  // 数据路径
  "wiki_structure_file": "/mnt/literism/data/wiki_dataset/wiki_structure.jsonl",
  "wiki_intro_file": "/mnt/literism/data/wiki_dataset/wiki_intro.jsonl",
  "dataset_split_file": "/mnt/literism/tree/hierarchical_output/data/dataset_split.json",
  
  // 输出路径
  "output_base": "/mnt/literism/tree/structure_output",
  
  // 模型路径
  "base_model": "/home/literism/model/Qwen3-8B",
  
  // 数据筛选阈值
  "min_structure_nodes": 10,      // 最少节点数
  "min_intro_length": 500,        // 最少intro长度
  
  // 数据集大小
  "train_size": 10000,            // 训练数据量
  "val_ratio": 0.1,               // 验证集比例
  
  // LoRA配置
  "lora_r": 16,
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  
  // 训练配置
  "num_epochs": 3,
  "batch_size": 2,
  "gradient_accumulation_steps": 8,
  "learning_rate": 0.0002,
  "max_length": 8192,             // 最大序列长度
  
  // 推理配置
  "max_new_tokens": 4096,
  "temperature": 0.7,
  "top_p": 0.9
}
```

## 与层次化分类系统集成

生成的结构树可用于初始化层次化分类系统：

```bash
# 1. 生成结构树
cd /home/literism/tree/structure_generator
python3 run_pipeline.py

# 2. 在层次化分类系统中使用
cd /home/literism/tree/hierarchical_classifier
python3 inference.py \
    --use_structure_init \
    --structures_file /mnt/literism/tree/structure_output/inference/test_hard_structures.json \
    --split test_hard
```

## 输出格式

### 结构树JSON格式

```json
{
  "Topic Name": {
    "sections": [
      {
        "title": "Main Category",
        "level": 2,
        "children": [
          {
            "title": "Subcategory",
            "level": 3,
            "children": []
          }
        ]
      }
    ],
    "node_count": 5,
    "generated_text": "- Main Category (level 2)\n  - Subcategory (level 3)\n..."
  }
}
```

## 性能优化建议

1. **GPU内存不足**：
   - 减小 `batch_size`
   - 减小 `max_length`
   - 启用梯度检查点（已默认启用）

2. **训练速度慢**：
   - 增大 `batch_size` 和 `gradient_accumulation_steps`
   - 使用多GPU训练

3. **生成质量不佳**：
   - 增加训练数据量 `train_size`
   - 调整推理参数 `temperature`, `top_p`
   - 增加训练轮数 `num_epochs`

## 常见问题

### Q: 为什么要排除train/test_easy/test_hard的topics？

A: 这些topics已经用于层次化分类系统的训练和测试，为了避免数据泄露，不能加入结构树生成器的训练数据。但我们需要为这些topics生成结构树用于推理，所以会单独保存它们的intro。

### Q: 生成的结构树质量如何评估？

A: 可以通过以下方式评估：
1. 节点数量是否合理
2. 层次结构是否清晰
3. 标题是否与主题相关
4. 在层次化分类系统中的实际效果

### Q: 可以使用其他基础模型吗？

A: 可以，修改配置中的 `base_model` 路径即可。建议使用支持长上下文的模型（8K+）。

## 依赖

- Python 3.10+
- PyTorch 2.0+
- transformers
- peft
- trl
- datasets
- bitsandbytes

## 作者

LLM Hierarchical Classification Project

