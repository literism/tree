# Wikipedia引用扩充工具

通过搜索引擎为Wikipedia结构树的每个叶子节点查找并添加引用。

## 功能说明

1. **读取结构树**：从 `wiki_structure.jsonl` 读取所有topic的结构
2. **识别叶子节点**：找到所有叶子节点（没有子节点的节点）
3. **构建搜索query**：将节点路径拼接为 `topic - title - subtitle` 格式
4. **搜索并提取URL**：使用Bing搜索，提取真实URL（从Bing跳转链接中解析）
5. **管理引用字典**：查找或创建引用ID，更新到结构树
6. **保存结果**：输出扩充后的结构和引用字典

## 使用方法

### 基本用法

```bash
cd /home/literism/tree/init_dataset
python expand_references.py
```

程序会自动：
- 读取 `/mnt/literism/data/wiki_dataset/wiki_structure.jsonl`
- 读取 `/mnt/literism/data/wikipedia_references.json`
- 处理所有topics的所有叶子节点
- 保存到 `wiki_structure_searched.jsonl` 和 `wikipedia_references_searched.json`

### 配置参数

在 `expand_references.py` 中修改：

```python
# 输入输出文件
INPUT_STRUCTURE = "/mnt/literism/tree/data/wikipedia_structures.json"
INPUT_REFERENCES = "/mnt/literism/tree/data/wikipedia_references.json"
OUTPUT_STRUCTURE = "/mnt/literism/tree/data/wikipedia_structures_searched.json"
OUTPUT_REFERENCES = "/mnt/literism/tree/data/wikipedia_references_searched.json"

# 搜索配置
NUM_RESULTS_PER_QUERY = 15  # 每个叶子节点搜索的结果数
DELAY_BETWEEN_QUERIES = 2.0  # query之间的延迟（秒）
```

## 工作流程

### 1. 叶子节点识别

对于结构树：
```
Topic: Albert Einstein
├─ Early life (有子节点，不处理)
│  ├─ overview (叶子节点，处理)
│  └─ Education (叶子节点，处理)
└─ Later life (有子节点，不处理)
   └─ Death (叶子节点，处理)
```

只处理：`overview`, `Education`, `Death`

### 2. Query构建

- `overview` → `Albert Einstein - Early life`
- `Education` → `Albert Einstein - Early life - Education`
- `Death` → `Albert Einstein - Later life - Death`

**注意**：`overview`被过滤掉，使用父节点路径

### 3. URL提取

从Bing跳转URL：
```
https://www.bing.com/ck/a?...&u=a1aHR0cHM6Ly93d3cuZXhhbXBsZS5jb20%3d&...
```

提取并解码得到：
```
https://www.example.com
```

### 4. 引用管理

```python
# 如果URL已存在
url_to_key = {"https://example.com": "existing_key"}
# 使用 "existing_key"

# 如果URL是新的
new_key = "search_1"  # 自动递增
references_dict["search_1"] = "https://example.com"
```

### 5. Citations合并

```python
# 原有citations
existing = ["ref1", "ref2"]

# 新增citations
new = ["search_1", "search_2", "ref1"]  # ref1重复

# 合并去重后
result = ["ref1", "ref2", "search_1", "search_2"]
```

## 输出格式

### 结构文件（JSON）

一个包含所有topics的字典：
```json
{
  "Person:Albert Einstein": {
    "topic": "Albert Einstein",
    "category": "Person",
    "structure": [
      {
        "title": "Early life",
        "citations": ["ref1", "search_1", "search_2"],
        "children": [...]
      }
    ]
  },
  "Company:Apple Inc.": {
    ...
  }
}
```

### 引用文件（JSON）

按topic组织：
```json
{
  "Person:Albert Einstein": {
    "topic": "Albert Einstein",
    "category": "Person",
    "pageid": 736,
    "references": {
      "ref1": "https://...",
      "search_1": "https://...",
      "search_2": "https://..."
    }
  }
}
```

## 进度显示

运行时会显示：
```
[1/100] 
处理 Topic: Albert Einstein
  找到 5 个叶子节点
  开始搜索 5 个queries...
    - Early life > overview: +3 个引用
    - Early life > Education: +5 个引用
    - Later life > Death: +4 个引用

[2/100]
处理 Topic: Marie Curie
  ...
```

## 处理时间估算

假设：
- 100个topics
- 平均每个topic 10个叶子节点
- 每个query延迟2秒

总时间：`100 × 10 × 2 = 2000秒 ≈ 33分钟`

## 注意事项

1. **网络要求**：需要稳定的网络连接和代理（如果访问国际网站）

2. **Cookie配置**：确保 `/mnt/literism/tree/data/cookies.txt` 存在且有效

3. **断点续传**：当前版本不支持断点续传，如果中断需要重新运行
   - 建议：先用少量数据测试
   - 或者修改代码支持跳过已处理的topics

4. **去重机制**：
   - URL级别去重（相同URL使用相同key）
   - Citations级别去重（合并时去除重复）

5. **错误处理**：单个topic失败不会影响其他topics的处理

6. **资源占用**：
   - 每个query会启动浏览器页面
   - 建议在性能较好的机器上运行
   - 可以适当调整延迟时间

## 故障排除

**问题1：搜索失败**
```
✗ 搜索失败: TimeoutError
```
解决：
- 检查网络连接和代理配置
- 增加延迟时间
- 检查cookies文件是否有效

**问题2：URL提取失败**
- Bing的跳转格式可能变化
- 检查 `extract_real_url()` 函数
- 可能需要更新URL解析逻辑

**问题3：内存不足**
- 减少并发数量
- 分批处理topics
- 增加系统内存

**问题4：进度显示不准确**
- 检查JSONL文件格式
- 确保每行是有效的JSON

## 扩展功能

可以添加的功能：
1. 断点续传支持
2. 并行处理多个topics
3. 更智能的query构建
4. 结果质量评分和过滤
5. 失败重试机制
6. 详细日志记录

## 测试建议

首次使用建议：
1. 备份原始数据
2. 用少量数据测试（修改代码只处理前几个topics）
3. 检查输出格式是否正确
4. 验证URL提取是否准确
5. 确认citations合并逻辑正确

然后再处理完整数据集。

