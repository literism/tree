# Oracle策略Pipeline使用说明

## 概述

Oracle策略pipeline包含三个主要步骤：
1. **数据生成**：使用oracle策略生成SFT训练数据（基于BM25的summary）
2. **模型训练**：训练分类生成模型
3. **推理**：使用训练好的分类模型 + BOW Updater进行推理

## 快速开始

### 完整流程（生成数据 + 训练 + 推理）

```bash
cd /home/literism/tree
python -m summary_based_classifier.cli.run_oracle_pipeline \
    --config summary_based_classifier/configs/default.json \
    --inference_gpus 0,1
```

### 分步执行

#### 1. 只生成Oracle数据

```bash
python -m summary_based_classifier.cli.run_oracle_pipeline \
    --config summary_based_classifier/configs/default.json \
    --skip_train \
    --skip_inference \
    --bow_top_k 30 \
    --seed 42 \
    --val_ratio 0.02
```

生成的文件：
- `summary_output/data/classify_generator_oracle_train.jsonl`
- `summary_output/data/classify_generator_oracle_val.jsonl`

#### 2. 只训练模型（数据已生成）

```bash
python -m summary_based_classifier.cli.run_oracle_pipeline \
    --config summary_based_classifier/configs/default.json \
    --skip_generate \
    --skip_inference
```

训练好的模型：
- `summary_output/models/classify_generator_oracle/final_model/`

#### 3. 只推理（数据已生成，模型已训练）

```bash
python -m summary_based_classifier.cli.run_oracle_pipeline \
    --config summary_based_classifier/configs/default.json \
    --skip_generate \
    --skip_train \
    --inference_split test \
    --inference_gpus 0,1
```

推理结果：
- `summary_output/inference/test_trees_oracle.json`

## 主要参数

### 数据生成参数

- `--bow_top_k INT`: BM25 summary保留的top-k词数（默认30）
- `--seed INT`: 随机种子（默认42）
- `--val_ratio FLOAT`: 验证集比例（默认0.02）
- `--max_refs_per_topic INT`: 每个topic最多使用的文章数（用于快速测试）

### 推理参数

- `--inference_split STR`: 推理的数据划分（默认test）
- `--inference_gpus STR`: 使用的GPU，逗号分隔（默认"0,1"）
  - 第一个GPU用于分类模型
  - 第二个GPU用于Updater（BOW模式实际不使用）
- `--classify_generator_model STR`: 指定分类模型路径（可选，默认自动查找）

## 与传统pipeline的区别

### 传统pipeline (run_pipeline.py)
- 需要两个训练好的模型（分类模型 + Updater模型）
- Updater需要单独训练
- 推理时两个模型都需要加载到GPU

### Oracle pipeline (run_oracle_pipeline.py)
- **只需要训练分类模型**
- **Updater使用BOW模式**（不需要训练，不需要GPU）
- 推理时只需要加载分类模型
- Summary使用BM25得分，避免长短文差异

## GPU配置说明

由于只有分类模型需要GPU，你可以：

### 单GPU运行
```bash
--inference_gpus 0
```
分类模型和BOW Updater都在GPU 0（但BOW实际不使用GPU）

### 双GPU运行（推荐）
```bash
--inference_gpus 0,1
```
- GPU 0: 分类模型
- GPU 1: 预留给Updater（虽然BOW不需要，但为代码兼容性保留）

## 示例

### 快速测试（小数据集）

```bash
python -m summary_based_classifier.cli.run_oracle_pipeline \
    --config summary_based_classifier/configs/default.json \
    --max_refs_per_topic 10 \
    --inference_gpus 0,1
```

### 生产环境（完整数据）

```bash
# 步骤1: 生成数据
python -m summary_based_classifier.cli.run_oracle_pipeline \
    --skip_train --skip_inference \
    --bow_top_k 30 --seed 42

# 步骤2: 训练模型
python -m summary_based_classifier.cli.run_oracle_pipeline \
    --skip_generate --skip_inference

# 步骤3: 推理
python -m summary_based_classifier.cli.run_oracle_pipeline \
    --skip_generate --skip_train \
    --inference_split test \
    --inference_gpus 0,1
```

## 输出文件结构

```
summary_output/
├── data/
│   ├── classify_generator_oracle_train.jsonl  # 训练数据
│   └── classify_generator_oracle_val.jsonl    # 验证数据
├── models/
│   └── classify_generator_oracle/
│       └── final_model/                        # 训练好的模型
└── inference/
    └── test_trees_oracle.json                  # 推理结果
```

## 注意事项

1. **磁盘空间**：确保`/mnt/literism`有足够空间（数据和模型缓存都会保存到这里）
2. **GPU内存**：分类模型需要足够的GPU内存，可通过配置文件调整`gpu_memory_utilization`
3. **BM25参数**：`bow_top_k=30`适用于大多数场景，可以根据需要调整
4. **数据一致性**：确保训练和推理使用相同的`bow_top_k`值

## 故障排除

### 问题：磁盘空间不足
**解决**：清理`~/.cache/huggingface/datasets`或使用符号链接到大容量磁盘

### 问题：GPU内存不足
**解决**：降低`gpu_memory_utilization`（配置文件中）或使用更小的`max_model_len`

### 问题：推理结果为空
**检查**：
1. 模型是否训练成功
2. 数据划分是否正确
3. 查看推理日志中的错误信息
