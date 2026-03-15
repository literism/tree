# 断点续传使用说明

## 功能说明

程序现在支持断点续传功能：
- **自动保存**：每处理完一个topic就立即保存到输出文件
- **自动恢复**：重新运行时自动识别已处理的topics并跳过
- **增量处理**：只处理未完成的topics

## 使用场景

### 场景1：正常运行
第一次运行：
```bash
python expand_references.py
```

输出：
```
[1] 加载数据...
  ✓ 加载了 100 个topics
  
[2] 开始处理topics...
  总共: 100 个topics
  已处理: 0 个
  剩余: 100 个
  
[1/100] 处理 Topic: Albert Einstein
  ...
```

### 场景2：中断后继续
程序运行到第30个topic时中断（Ctrl+C或崩溃）。

重新运行：
```bash
python expand_references.py
```

输出：
```
[1] 加载数据...
  ✓ 加载了 100 个topics
  ✓ 发现已处理的数据: 30 个topics
  → 将跳过已处理的，继续处理剩余 70 个

[2] 开始处理topics...
  总共: 100 个topics
  已处理: 30 个
  剩余: 70 个
  
[31/100] 处理 Topic: Marie Curie
  ...
```

### 场景3：全部完成
所有topics都处理完后再次运行：

```bash
python expand_references.py
```

输出：
```
[1] 加载数据...
  ✓ 加载了 100 个topics
  ✓ 发现已处理的数据: 100 个topics
  → 将跳过已处理的，继续处理剩余 0 个

[2] 开始处理topics...
  总共: 100 个topics
  已处理: 100 个
  剩余: 0 个

[3] 最终确认...
  ✓ 所有数据已保存

================================================================================
处理完成！
================================================================================
总topics数: 100
  - 之前已处理: 100
  - 本次新处理: 0
  
提示: 所有topics都已处理完成。如需重新处理，请删除输出文件。
================================================================================
```

## 工作原理

### 1. 读取已有输出
```python
# 检查输出文件是否存在
if Path(OUTPUT_STRUCTURE).exists():
    # 读取已处理的结构
    processed_structures = json.load(f)
    processed_topics = set(processed_structures.keys())
```

### 2. 跳过已处理
```python
for topic_key, topic_data in structures.items():
    # 跳过已处理的topic
    if topic_key in processed_topics:
        continue
    
    # 处理新topic
    ...
```

### 3. 立即保存
```python
# 处理完一个topic
topic_data, all_references, search_counter = process_single_topic(...)

# 立即保存到文件（断点续传）
with open(OUTPUT_STRUCTURE, 'w', encoding='utf-8') as f:
    json.dump(structures, f, ensure_ascii=False, indent=2)
```

## 重新开始

如果需要从头重新处理所有topics：

```bash
# 删除输出文件
rm /mnt/literism/tree/data/wikipedia_structures_searched.json
rm /mnt/literism/tree/data/wikipedia_references_searched.json

# 重新运行
python expand_references.py
```

或者重命名输出文件作为备份：

```bash
# 备份旧文件
mv /mnt/literism/tree/data/wikipedia_structures_searched.json \
   /mnt/literism/tree/data/wikipedia_structures_searched_backup.json
   
mv /mnt/literism/tree/data/wikipedia_references_searched.json \
   /mnt/literism/tree/data/wikipedia_references_searched_backup.json

# 重新运行
python expand_references.py
```

## 注意事项

### 1. 不会重复处理
- 已处理的topics会被完整跳过
- 即使修改了输入文件，已处理的也不会更新
- 如需更新某个topic，删除输出文件后重新运行

### 2. 引用编号递增
- 程序会自动找到已有的最大search_编号
- 新引用从下一个编号开始
- 例如：已有search_1到search_100，新增会从search_101开始

### 3. 文件一致性
- 结构文件和引用文件同时更新
- 如果保存失败，会打印警告但继续处理
- 建议定期检查输出文件是否正常

### 4. 处理失败
如果某个topic处理失败：
- 会打印错误信息
- 继续处理下一个topic
- 失败的topic不会标记为"已处理"
- 下次运行会重试失败的topic

## 性能说明

### 保存频率
- 每个topic处理后立即保存（约2-5秒一次）
- 对于大文件（100+ topics），保存一次约1-2秒
- 总体性能影响约5-10%

### 优化建议
如果topics数量很大（1000+），可以考虑：
1. 每N个topics保存一次（需要修改代码）
2. 使用更快的存储（SSD）
3. 分批处理

## 故障排除

**问题1：提示已处理但实际没有**
```
解决：检查输出文件是否损坏
cat /mnt/literism/tree/data/wikipedia_structures_searched.json | head
```

**问题2：想重新处理某个topic**
```
解决：从输出JSON中删除该topic的key，或删除整个输出文件
```

**问题3：断点续传后search_编号重复**
```
解决：程序会自动检测最大编号，如果仍有问题，检查all_references字典
```

**问题4：保存失败**
```
解决：
1. 检查磁盘空间
2. 检查目录权限
3. 查看详细错误信息
```

