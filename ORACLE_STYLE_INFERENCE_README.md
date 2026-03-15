# Oracle风格推理系统

## 概述

Oracle风格推理系统使用与数据生成相同的推理逻辑，但使用训练好的模型（分类模型 + 总结模型）进行推理，而不是依赖于API或BOW特征。

## 架构特点

### 1. 生产者-消费者模式
- **ClassifierWorker**: 独立进程运行分类模型，处理分类请求
- **UpdaterWorker**: 独立进程运行总结模型，处理总结生成/更新请求
- 每个Worker内部使用批量推理，提高吞吐量
- 多个Topic并行处理，每个Topic在独立线程中运行

### 2. 推理流程（与数据生成一致）
1. **自上而下分类**: 从根节点开始，逐层对文章进行分类
   - 如果没有子节点，创建新节点
   - 如果有子节点，使用分类模型决定：选择现有子节点 或 创建新节点
   - 支持归拢操作（merge）

2. **自下而上更新**: 文章分类后，从叶子节点向上更新summary
   - 使用总结模型生成/更新每个节点的summary
   - 考虑父节点summary、当前summary、兄弟节点summaries

## 文件结构

```
summary_based_classifier/
├── inference/
│   ├── oracle_style_inference_processor.py  # Oracle风格推理处理器（核心）
│   ├── inference_oracle_style.py           # Oracle风格推理入口脚本
│   ├── inference_oracle.py                  # 旧版推理（使用BOW）
│   └── inference_parallel.py                # 通用并行推理入口
├── models/
│   └── model_workers.py                     # ClassifierWorker和UpdaterWorker
└── cli/
    └── run_oracle_pipeline.py              # 完整Pipeline脚本
```

## 使用方法

### 方法1: 通过Pipeline脚本

```bash
python -m summary_based_classifier.cli.run_oracle_pipeline \
  --config ./summary_based_classifier/configs/default.json \
  --skip_generate \
  --skip_train_summary \
  --skip_train_classify \
  --inference_split test \
  --inference_gpus 0,1 \
  --inference_max_workers 4
```

### 方法2: 直接调用推理脚本

```bash
python -m summary_based_classifier.inference.inference_oracle_style \
  --config ./configs/default.json \
  --classify_generator_model ./output/models/classify_generator_oracle/final_model \
  --updater_model ./output/models/summary_generator_oracle/final_model \
  --split test \
  --classify_gpu 0 \
  --updater_gpu 1 \
  --max_workers 4 \
  --max_refs 100
```

## 参数说明

### 推理参数
- `--classify_generator_model`: 分类模型路径
- `--updater_model`: 总结模型路径
- `--split`: 数据集划分（train/val/test）
- `--classify_gpu`: 分类模型使用的GPU ID
- `--updater_gpu`: 总结模型使用的GPU ID
- `--max_workers`: 最大并行topic数
- `--max_refs`: 每个topic最多处理的文章数（用于测试）

### 配置文件参数
- `inference.max_depth`: 最大树深度
- `inference.classifier_batch_size`: 分类器批次大小
- `inference.updater_batch_size`: 更新器批次大小
- `inference.classifier_timeout`: 分类器批次超时（秒）
- `inference.updater_timeout`: 更新器批次超时（秒）
- `inference.max_model_len`: 模型最大长度
- `inference.gpu_memory_utilization`: GPU内存利用率
- `inference.temperature`: 采样温度
- `inference.top_p`: top_p采样
- `summary.max_content_length`: 文章最大token长度

## 输出格式

推理结果保存为JSON文件，格式为：

```json
{
  "topic_key": {
    "topic": "Topic名称",
    "structure": [
      {
        "level": 2,
        "summary": "节点摘要",
        "citations": ["ref_id1", "ref_id2"],
        "children": [...]
      }
    ]
  }
}
```

输出文件名: `{split}_trees_oracle_style.json`

## 与旧版推理的区别

| 特性 | Oracle风格推理 | 旧版推理（BOW） |
|------|----------------|-----------------|
| 总结生成 | 使用训练好的总结模型 | 使用BOW（词袋模型） |
| 推理逻辑 | 与数据生成一致（自上而下+自下而上） | 简化的TreeBuilder逻辑 |
| 归拢操作 | 支持（分类模型输出） | 不支持 |
| 多路径处理 | 支持（但当前实现选择第一个） | 不支持 |
| 并行方式 | 多进程Worker + 多线程Topics | 单进程 或 多线程 |
| 适用场景 | 生产环境，高质量推理 | 快速测试，轻量级推理 |

## 性能优化

1. **批量推理**: Worker内部批量处理请求，减少GPU调用次数
2. **并行处理**: 多个Topic并行处理，充分利用计算资源
3. **队列通信**: 使用multiprocessing.Queue实现进程间通信
4. **超时机制**: 避免长时间等待，保证响应性

## 注意事项

1. **GPU内存**: 两个模型会占用两张GPU，确保GPU内存足够
2. **模型路径**: 确保分类模型和总结模型都已训练完成
3. **数据格式**: 输入数据格式需与训练时一致
4. **进程安全**: vLLM不是线程安全的，因此使用独立进程
5. **资源清理**: 推理完成后会自动清理Worker进程和释放显存

## 故障排查

### 问题1: 找不到模型
```
错误: 分类模型不存在: xxx
错误: 总结模型不存在: xxx
```
**解决**: 确保已完成模型训练，或使用`--classify_generator_model`和`--updater_model`指定正确路径

### 问题2: GPU内存不足
```
torch.cuda.OutOfMemoryError
```
**解决**: 
- 减小`batch_size`
- 减小`max_model_len`
- 降低`gpu_memory_utilization`
- 使用单GPU（两个模型共享）

### 问题3: Worker进程卡住
**解决**: 
- 检查队列是否正常通信
- 确保Worker进程没有死锁
- 增加`timeout`参数
- 检查vLLM日志

### 问题4: 推理结果为空
**解决**: 
- 检查输入数据格式
- 检查模型输出解析
- 查看详细日志
- 尝试单个topic测试

## 后续改进

1. 支持真正的多路径处理（当前只选择第一个）
2. 支持动态批次大小
3. 支持模型预加载和复用
4. 添加更多监控和日志
5. 支持分布式推理（多机多卡）
