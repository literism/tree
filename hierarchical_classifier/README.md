# 层次化文章分类系统

这是一个用于构建Wikipedia文章层次化结构的训练和推理系统。系统会学习如何将文章分类到层次化的标题结构中。

## 系统概述

系统包含以下几个主要组件：

1. **数据集划分** (`data_split.py`): 将数据划分为训练集、验证集和测试集
2. **分类系统** (`classifier.py`): 支持真实模式（使用真实数据）和模型模式（使用训练的模型）
3. **构建系统** (`builder.py`): 递归地构建文章的层次化结构树
4. **数据准备** (`prepare_dataset.py`): 将构建过程的记录转换为SFT训练数据
5. **模型训练** (`train.py`): 使用TRL库的SFTTrainer进行LoRA微调
6. **推理系统** (`inference.py`): 使用训练好的模型构建结构树

## 安装依赖

```bash
pip install torch transformers datasets peft trl vllm bitsandbytes accelerate
```

## 使用方法

### 方法1: 使用完整流程脚本（推荐）

```bash
python3 run_pipeline.py \
    --base_model /path/to/your/base/model \
    --references_file /mnt/literism/tree/data/wikipedia_references_final.json \
    --topic_classified_file /mnt/literism/data/result/topic_classified.json \
    --output_base ./output
```

这个脚本会自动执行所有步骤：
1. 划分数据集
2. 构建训练/验证/测试记录
3. 准备训练数据集
4. 训练模型
5. 推理

如果某些步骤已经完成，可以使用跳过选项：
```bash
python3 run_pipeline.py \
    --base_model /path/to/your/base/model \
    --skip_split \
    --skip_build_records \
    --skip_prepare_dataset
```

只运行推理：
```bash
python3 run_pipeline.py \
    --base_model /path/to/your/base/model \
    --only_inference
```

### 方法2: 逐步执行

#### 步骤 1: 划分数据集

```bash
python3 data_split.py \
    --references_file /mnt/literism/tree/data/wikipedia_references_final.json \
    --topic_classified_file /mnt/literism/data/result/topic_classified.json \
    --output_dir ./data
```

输出：
- `data/dataset_split.json`: 数据集划分信息
- `data/split_stats.json`: 统计信息

#### 步骤 2: 构建记录

使用真实模式构建器生成训练数据：

```bash
# 训练集
python3 builder.py \
    --mode ground_truth \
    --split train \
    --record \
    --references_file /mnt/literism/tree/data/wikipedia_references_final.json \
    --split_file ./data/dataset_split.json \
    --output_dir ./output

# 验证集
python3 builder.py \
    --mode ground_truth \
    --split val \
    --record \
    --references_file /mnt/literism/tree/data/wikipedia_references_final.json \
    --split_file ./data/dataset_split.json \
    --output_dir ./output

# 测试集
python3 builder.py \
    --mode ground_truth \
    --split test \
    --record \
    --references_file /mnt/literism/tree/data/wikipedia_references_final.json \
    --split_file ./data/dataset_split.json \
    --output_dir ./output
```

输出：
- `output/train_records.json`: 训练记录
- `output/val_records.json`: 验证记录
- `output/test_records.json`: 测试记录

#### 步骤 3: 准备训练数据集

```bash
python3 prepare_dataset.py \
    --train_records ./output/train_records.json \
    --val_records ./output/val_records.json \
    --test_records ./output/test_records.json \
    --output_dir ./dataset \
    --ratio 2 1 1
```

输出：
- `dataset/train_dataset.jsonl`: 训练数据集（对话格式）
- `dataset/val_dataset.jsonl`: 验证数据集
- `dataset/test_dataset.jsonl`: 测试数据集

数据按三类比例（2:1:1）采样：
1. 有新子标题的记录
2. 只有现有子标题的记录
3. 都是None的记录

#### 步骤 4: 训练模型

```bash
python3 train.py \
    --base_model /path/to/your/base/model \
    --train_data ./dataset/train_dataset.jsonl \
    --val_data ./dataset/val_dataset.jsonl \
    --output_dir ./models/hierarchical_classifier
```

可选参数：
```bash
python3 train.py \
    --base_model /path/to/your/base/model \
    --train_data ./dataset/train_dataset.jsonl \
    --val_data ./dataset/val_dataset.jsonl \
    --output_dir ./models/hierarchical_classifier \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-4
```

输出：
- `models/hierarchical_classifier/adapter/`: LoRA adapter
- `models/hierarchical_classifier/model/`: 合并后的完整模型

#### 步骤 5: 推理

```bash
python3 inference.py \
    --model_path ./models/hierarchical_classifier/model \
    --references_file /mnt/literism/tree/data/wikipedia_references_final.json \
    --split_file ./data/dataset_split.json \
    --split test \
    --output_dir ./inference_output
```

推理单个topic：
```bash
python3 inference.py \
    --model_path ./models/hierarchical_classifier/model \
    --references_file /mnt/literism/tree/data/wikipedia_references_final.json \
    --topic_key "Person:Albert Einstein" \
    --output_dir ./inference_output
```

快速测试（限制每个topic的references数量）：
```bash
python3 inference.py \
    --model_path ./models/hierarchical_classifier/model \
    --references_file /mnt/literism/tree/data/wikipedia_references_final.json \
    --split test \
    --max_refs 10 \
    --output_dir ./inference_output
```

输出：
- `inference_output/test_inference_trees.json`: 推理生成的结构树

## 数据格式

### 输入数据格式

`wikipedia_references_final.json`:
```json
{
  "Person:Albert Einstein": {
    "topic": "Albert Einstein",
    "category": "Person",
    "pageid": 736,
    "references": {
      "search_1": {
        "url": "https://example.com/article",
        "content": "文章内容...",
        "paths": [
          "Albert Einstein - Early life - Education",
          "Albert Einstein - Career - Patent office"
        ]
      }
    }
  }
}
```

### 输出数据格式

结构树格式：
```json
{
  "title": "Albert Einstein",
  "level": 0,
  "citations": [],
  "children": [
    {
      "title": "Early life",
      "level": 1,
      "citations": ["search_1", "search_2"],
      "children": [
        {
          "title": "Education",
          "level": 2,
          "citations": ["search_1"],
          "children": []
        }
      ]
    }
  ]
}
```

## 测试

测试分类器：
```bash
python3 classifier.py
```

测试构建器（使用一个小的topic）：
```bash
python3 -c "
import json
from classifier import Classifier
from builder import TreeBuilder

with open('/mnt/literism/tree/data/wikipedia_references_final.json', 'r') as f:
    references_data = json.load(f)

classifier = Classifier(mode='ground_truth', references_file='/mnt/literism/tree/data/wikipedia_references_final.json')
builder = TreeBuilder(classifier=classifier, references_data=references_data, max_depth=5)

# 选择一个topic
topic_key = 'Person:Marie Curie'
ref_ids = list(references_data[topic_key]['references'].keys())[:5]

root, records = builder.build_tree(topic_key=topic_key, reference_ids=ref_ids, record_mode=True)
print(json.dumps(root.to_dict(), indent=2, ensure_ascii=False))
"
```

## 配置

### 训练配置

可以通过JSON文件指定训练配置：

```json
{
  "lora": {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "bias": "none"
  },
  "training": {
    "num_epochs": 3,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "warmup_ratio": 0.03,
    "max_length": 4096,
    "bf16": true,
    "gradient_checkpointing": true
  },
  "quantization": {
    "load_in_4bit": true,
    "bnb_4bit_compute_dtype": "bfloat16",
    "bnb_4bit_use_double_quant": true,
    "bnb_4bit_quant_type": "nf4"
  }
}
```

使用配置文件：
```bash
python3 train.py \
    --base_model /path/to/model \
    --train_data ./dataset/train_dataset.jsonl \
    --val_data ./dataset/val_dataset.jsonl \
    --output_dir ./models/hierarchical_classifier \
    --config ./config.json
```

## 项目结构

```
hierarchical_classifier/
├── README.md                   # 本文件
├── data_split.py              # 数据集划分
├── classifier.py              # 分类系统
├── builder.py                 # 构建系统
├── prepare_dataset.py         # 数据准备
├── train.py                   # 模型训练
├── inference.py               # 推理系统
├── run_pipeline.py            # 完整流程脚本
├── data/                      # 数据集划分结果
├── output/                    # 构建记录
├── dataset/                   # 训练数据集
├── models/                    # 训练好的模型
└── inference_output/          # 推理结果
```

## 注意事项

1. **内存需求**: 训练大型语言模型需要大量GPU内存。如果内存不足，可以：
   - 减小 `batch_size`
   - 增加 `gradient_accumulation_steps`
   - 启用4-bit量化（默认已启用）
   - 减小 `max_length`

2. **训练时间**: 完整训练可能需要几个小时到几天，取决于：
   - 数据集大小
   - 模型大小
   - GPU性能
   - batch_size和gradient_accumulation_steps

3. **数据集比例**: 默认按2:1:1比例采样三类数据，可以通过`--ratio`参数调整。

4. **推理性能**: 使用vLLM进行批量推理可以显著提高速度。可以调整：
   - `tensor_parallel_size`: 多GPU并行
   - `max_model_len`: 最大序列长度
   - `gpu_memory_utilization`: GPU内存利用率

## 常见问题

### Q: 训练过程中OOM（内存不足）怎么办？
A: 尝试以下方法：
1. 减小batch_size到2或1
2. 增加gradient_accumulation_steps到8或16
3. 减小max_length到2048或更小
4. 确保启用了4-bit量化

### Q: 如何恢复中断的训练？
A: SFTTrainer会自动保存checkpoint，可以从最新的checkpoint恢复：
```bash
python3 train.py \
    --base_model /path/to/model \
    --train_data ./dataset/train_dataset.jsonl \
    --val_data ./dataset/val_dataset.jsonl \
    --output_dir ./models/hierarchical_classifier
```
trainer会自动检测并加载最新的checkpoint。

### Q: 如何评估模型效果？
A: 可以比较真实模式和模型模式构建的结构树，计算相似度指标。

### Q: 可以使用其他基础模型吗？
A: 可以，只要是HuggingFace格式的causal LM模型即可，如：
- Llama系列
- Mistral系列
- Qwen系列
- 等

## 作者和许可

此项目用于学术研究目的。

