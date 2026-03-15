# Structure Tree Generator - 项目总结

## 项目概述

本项目实现了一个完整的系统，用于训练大语言模型自动生成Wikipedia主题的层次化结构树框架。生成的结构树可用于初始化层次化分类系统，显著提升test_hard数据集的分类效果。

## 核心功能

1. **数据准备**: 从~800万条Wikipedia数据中筛选和准备训练数据
2. **模型训练**: 使用LoRA微调Qwen模型学习结构树生成
3. **结构推理**: 为指定topics生成层次化结构树
4. **系统集成**: 与层次化分类系统无缝集成

## 文件结构

```
structure_generator/
├── config.py                          # 配置管理系统
├── prepare_structure_dataset.py       # 数据集准备脚本
├── train_structure_generator.py       # 模型训练脚本
├── inference_structure.py             # 结构树推理脚本
├── run_pipeline.py                    # 完整流程运行脚本
├── test_data_loading.py              # 数据加载测试脚本
├── test_single_generation.py         # 单个topic生成测试
├── configs/
│   └── default.json                   # 默认配置文件
├── README.md                          # 详细文档
├── QUICKSTART.md                      # 快速开始指南
└── PROJECT_SUMMARY.md                 # 本文档
```

## 详细文件说明

### 1. config.py

**功能**: 统一的配置管理系统

**主要内容**:
- `StructureGeneratorConfig`: 配置数据类
  - 输入数据路径（wiki_structure.jsonl, wiki_intro.jsonl）
  - 输出路径（data_dir, models_dir, inference_dir）
  - 数据筛选阈值（min_structure_nodes, min_intro_length）
  - 训练参数（LoRA配置，训练超参数）
  - 推理参数（max_new_tokens, temperature, top_p）
- `SKIP_TITLES`: 需要过滤的无用Wikipedia标题集合

**关键方法**:
- `from_json()`: 从JSON文件加载配置
- `to_json()`: 保存配置到JSON文件
- `print_config()`: 打印配置信息

### 2. prepare_structure_dataset.py

**功能**: 准备训练数据集

**主要类**: `StructureDatasetPreparator`

**核心流程**:
1. 加载已有的topic划分（train/test_easy/test_hard）
2. 从wiki_structure.jsonl和wiki_intro.jsonl加载数据
3. 筛选符合条件的topics：
   - 不在已有的train/test_easy/test_hard中
   - 清理后的结构树节点数 ≥ min_structure_nodes
   - intro长度 ≥ min_intro_length
4. 清理结构树（删除"See also", "References"等无用节点）
5. 生成训练数据（prompt + completion格式）
6. 划分训练集和验证集
7. 保存excluded topics的intro（用于后续推理）

**关键方法**:
- `clean_structure()`: 递归清理结构树
- `count_nodes()`: 统计节点数
- `structure_to_text()`: 将结构树转换为文本格式
- `create_prompt()`: 创建训练prompt
- `load_and_filter_data()`: 加载并筛选数据
- `prepare_dataset()`: 完整的数据准备流程

**输出文件**:
- `train.jsonl`: 训练数据
- `val.jsonl`: 验证数据
- `excluded_topics_intro.json`: 排除topics的intro
- `dataset_stats.json`: 统计信息

### 3. train_structure_generator.py

**功能**: 训练结构树生成模型

**主要类**: `StructureGeneratorTrainer`

**核心流程**:
1. 加载训练和验证数据集
2. 加载基础模型（Qwen）和tokenizer
3. 配置4-bit量化（节省GPU内存）
4. 配置LoRA参数
5. 使用TRL的SFTTrainer进行训练
6. 保存最终模型和检查点

**关键方法**:
- `load_datasets()`: 加载数据集
- `setup_model_and_tokenizer()`: 设置模型和tokenizer
- `train()`: 完整的训练流程

**训练配置**:
- 4-bit量化 + LoRA微调
- 梯度检查点（节省内存）
- BF16混合精度训练
- Tensorboard日志记录

**输出文件**:
- `models/structure_generator/final/`: 最终模型
- `models/structure_generator/checkpoint-*/`: 训练检查点
- `models/structure_generator/config.json`: 训练配置

### 4. inference_structure.py

**功能**: 使用训练好的模型推理生成结构树

**主要类**: `StructureGenerator`

**核心流程**:
1. 加载基础模型和LoRA适配器
2. 为指定的topics创建prompt
3. 生成结构树文本
4. 解析文本为结构化数据
5. 保存结果

**关键方法**:
- `create_prompt()`: 创建推理prompt（与训练时相同）
- `generate_structure()`: 生成结构树
- `parse_structure_text()`: 解析生成的文本为结构化数据

**推理模式**:
- `--split train`: 为train topics生成
- `--split test_easy`: 为test_easy topics生成
- `--split test_hard`: 为test_hard topics生成
- `--split all`: 为所有topics生成（默认）

**输出文件**:
- `inference/train_structures.json`
- `inference/test_easy_structures.json`
- `inference/test_hard_structures.json`

### 5. run_pipeline.py

**功能**: 完整流程的一键运行脚本

**核心流程**:
1. 准备数据集（可跳过）
2. 训练模型（可跳过）
3. 推理生成结构树（可跳过）

**命令行参数**:
- `--config`: 配置文件路径
- `--skip_prepare`: 跳过数据准备
- `--skip_training`: 跳过训练
- `--skip_inference`: 跳过推理
- `--inference_split`: 推理的split（train/test_easy/test_hard/all）

**使用示例**:
```bash
# 完整流程
python3 run_pipeline.py

# 使用自定义配置
python3 run_pipeline.py --config configs/my_config.json

# 只推理test_hard
python3 run_pipeline.py --skip_prepare --skip_training --inference_split test_hard
```

### 6. test_data_loading.py

**功能**: 测试数据加载和处理逻辑

**测试内容**:
1. wiki_structure.jsonl加载
2. wiki_intro.jsonl加载
3. dataset_split.json加载
4. 结构清理逻辑
5. 结构转文本逻辑
6. Prompt创建
7. 真实数据样本处理

**使用场景**: 在运行完整流程前验证数据和逻辑是否正确

### 7. test_single_generation.py

**功能**: 测试单个topic的结构生成

**测试模式**:
- `--mode test`: 使用预定义的测试数据
- `--mode real`: 使用真实的excluded topic

**使用场景**: 
- 快速验证模型效果
- 调试生成质量
- 测试不同的推理参数

**使用示例**:
```bash
# 使用测试数据
python3 test_single_generation.py --mode test

# 使用真实topic
python3 test_single_generation.py --mode real
```

### 8. configs/default.json

**功能**: 默认配置文件

**主要配置项**:
```json
{
  // 数据路径
  "wiki_structure_file": "...",
  "wiki_intro_file": "...",
  "dataset_split_file": "...",
  
  // 筛选阈值
  "min_structure_nodes": 10,
  "min_intro_length": 500,
  
  // 数据集
  "train_size": 10000,
  "val_ratio": 0.1,
  
  // LoRA
  "lora_r": 16,
  "lora_alpha": 32,
  
  // 训练
  "num_epochs": 3,
  "batch_size": 2,
  "learning_rate": 0.0002,
  
  // 推理
  "temperature": 0.7,
  "top_p": 0.9
}
```

## 数据流程图

```
wiki_structure.jsonl (8M+)  ─┐
                             ├─> prepare_structure_dataset.py
wiki_intro.jsonl (8M+)      ─┤   ├─> 筛选 (min_nodes, min_intro_length)
                             │   ├─> 清理 (删除无用节点)
dataset_split.json          ─┘   ├─> 生成训练数据
                                 └─> train.jsonl + val.jsonl
                                     excluded_topics_intro.json
                                     
train.jsonl + val.jsonl ─────> train_structure_generator.py
                                ├─> LoRA微调
                                └─> models/structure_generator/final/
                                
excluded_topics_intro.json ─┐
                            ├─> inference_structure.py
models/.../final/          ─┤   ├─> 为train/test_easy/test_hard生成
                            │   └─> *_structures.json
dataset_split.json         ─┘
                            
*_structures.json ──────────> hierarchical_classifier/inference.py
                               (--use_structure_init)
```

## 与层次化分类系统的集成

### 1. 生成结构树

```bash
cd /home/literism/tree/structure_generator
python3 run_pipeline.py
```

### 2. 在分类系统中使用

```bash
cd /home/literism/tree/hierarchical_classifier

# 使用结构初始化进行推理
python3 inference.py \
    --use_structure_init \
    --structures_file /mnt/literism/tree/structure_output/inference/test_hard_structures.json \
    --split test_hard

# 评估效果
python3 evaluate.py --split test_hard
```

### 3. 对比效果

**不使用结构初始化**:
```bash
python3 inference.py --split test_hard
python3 evaluate.py --split test_hard
# 记录Omega Index和ONMI
```

**使用结构初始化**:
```bash
python3 inference.py --use_structure_init \
    --structures_file .../test_hard_structures.json \
    --split test_hard
python3 evaluate.py --split test_hard
# 对比Omega Index和ONMI的提升
```

## 关键设计决策

### 1. 为什么排除train/test_easy/test_hard的topics？

- **避免数据泄露**: 这些topics已用于分类系统的训练和测试
- **保证公平性**: 结构生成器不能"记住"测试集的结构
- **单独保存intro**: 虽然不用于训练，但需要为这些topics生成结构树

### 2. 为什么使用LoRA而不是全量微调？

- **效率**: LoRA只训练少量参数（~1-2%）
- **内存**: 可以使用4-bit量化 + LoRA
- **效果**: 对于这类任务，LoRA效果接近全量微调

### 3. 为什么使用文本格式而不是JSON？

- **更自然**: 文本格式对LLM更友好
- **更灵活**: 可以处理不规则的结构
- **更鲁棒**: 即使解析失败，文本仍可读

### 4. 为什么需要清理结构树？

- **减少噪音**: "See also", "References"等不是内容结构
- **提高质量**: 只保留真正的内容组织结构
- **节省token**: 减少训练数据的长度

## 性能指标

### 数据准备

- **输入**: ~800万条Wikipedia数据
- **筛选后**: ~数十万条符合条件的数据
- **采样**: 10000条训练数据
- **时间**: 约10-20分钟

### 训练

- **数据量**: 9000条训练 + 1000条验证
- **Epochs**: 3
- **时间**: 约3-6小时（取决于GPU）
- **GPU内存**: ~20GB（4-bit量化）

### 推理

- **Topics**: 97个（87 train + 0 test_easy + 10 test_hard）
- **时间**: 约30-60分钟
- **每个topic**: 约30秒

## 未来改进方向

1. **多样性增强**: 
   - 为每个topic生成多个候选结构
   - 使用集成方法选择最佳结构

2. **质量评估**:
   - 自动评估生成结构的质量
   - 与真实Wikipedia结构对比

3. **增量学习**:
   - 根据分类系统的反馈持续改进
   - 在线学习新的结构模式

4. **多语言支持**:
   - 扩展到其他语言的Wikipedia
   - 跨语言结构迁移

5. **领域适应**:
   - 针对特定领域微调
   - 支持自定义结构模板

## 依赖项

```
Python 3.10+
torch>=2.0.0
transformers>=4.30.0
peft>=0.4.0
trl>=0.7.0
datasets>=2.14.0
bitsandbytes>=0.41.0
```

## 许可证

本项目是LLM Hierarchical Classification Project的一部分。

## 联系方式

如有问题或建议，请联系项目维护者。

---

**创建日期**: 2025-12-04
**最后更新**: 2025-12-04
**版本**: 1.0.0

