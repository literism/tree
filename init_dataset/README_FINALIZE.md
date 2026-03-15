# 引用和结构最终处理

## 功能概述

`finalize_references.py` 对Wikipedia引用和结构数据进行最终处理，生成可用于模型训练的数据集。

## 处理流程

### Step 0: 加载数据
- 读取 `wikipedia_references_enriched.json`
- 读取 `wikipedia_structures_searched.json`

### Step 1: 过滤引用并限制token长度
**过滤规则**：
1. 删除 `failed` 字段不为 `null` 的引用（爬取失败）
2. 删除 URL 包含 `wikipedia.org` 的引用
3. 限制文章内容的token数量（最大2048 tokens）
   - 使用空格分割估计token数
   - 保留文章开头部分

**输出格式**：
```json
{
  "topic_key": {
    "references": {
      "ref_key": {
        "url": "https://...",
        "content": "文章内容（截断后）"
      }
    }
  }
}
```

### Step 2: 初始化DeepSeek API客户端
- 配置API密钥、并发数等参数
- 并发数默认为8

### Step 3: 使用LLM匹配search引用到叶子节点
这是最核心的步骤，目的是用LLM判断每个search引用与topic叶子节点的相关性。

#### 3.1 收集叶子节点
- 对每个topic，递归遍历结构树
- 找到所有叶子节点（没有children的节点）
- 记录每个叶子节点的完整路径

#### 3.2 构建Prompt
对每个topic的每个 `search_` 开头的引用，构建如下prompt：

```
Article Content:
[文章内容（前4000字符）]

Section Paths (Leaf Nodes):
1. Albert Einstein - Early life - Education
2. Albert Einstein - Early life - Family
3. Albert Einstein - Career - Patent office
4. Albert Einstein - Career - University positions

Question: Which section paths does this article discuss or relate to?
Reply with the numbers only, separated by commas (e.g., "1, 3").
If the article is not relevant to any section, reply "none".
```

**关键点**：
- 只处理 `search_` 开头的引用（来自Bing搜索）
- 使用完整路径（topic - title - subtitle）
- 列出该topic的所有叶子节点
- 每个prompt的标题列表不同，需要正确映射数字到节点

#### 3.3 调用DeepSeek API
- 批量发送所有prompts
- 并发处理（默认8个并发），提高效率

#### 3.4 解析结果
- 从模型输出中提取数字（正则表达式）
- 根据数字找到对应的叶子节点
- 将引用添加到这些叶子节点的 `citations` 列表中

**特殊处理**：
- 如果模型回答 `"none"`，说明该引用与所有叶子节点都不相关
- 直接从 `filtered_refs` 中删除该引用
- 在后续步骤中会自动清理结构树中的无效引用

**示例**：
```
模型输出: "1, 3"
→ 将引用添加到第1和第3个叶子节点的citations中

模型输出: "none"
→ 删除该引用（无效引用）
```

### Step 4: 清理结构中的无效引用
- 遍历所有节点的 `citations` 列表
- 删除不在过滤后引用中的引用ID
- 去重（因为Step 3可能添加了重复的）

### Step 5: 收集叶子节点路径
**叶子节点定义**：没有子节点（`children` 为空或不存在）的节点

对每个叶子节点：
- 记录从根到该节点的完整路径
- 记录该节点包含的所有引用

**路径来源包括**：
1. **原有路径**：引用在爬取Wikipedia和扩充时就已经在结构树中的位置
2. **LLM匹配路径**：Step 3中模型判断相关后新添加到节点的

**输出**：`{ref_key: [[path1], [path2], ...]}`

一个引用可能出现在多个叶子节点中。

### Step 6: 为引用添加路径信息
将收集到的路径信息添加到每个引用中，并进行去重。

**去重处理**：
- 将路径列表（`List[List[str]]`）转换为字符串（用 `" - "` 连接）
- 使用 `set` 去重（保持顺序）
- 因为模型匹配的路径可能与原有路径重复

**最终输出格式**：
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

**paths字段说明**：
- 包含该引用出现的所有叶子节点路径
- 已去重
- 混合了原有路径和LLM匹配的路径

## 配置参数

```python
MAX_TOKENS = 2048              # 最大token数
MAX_CONCURRENT_JOBS = 8        # DeepSeek API并发数
```

## 输入文件

1. **wikipedia_references_enriched.json**
   - 来自 `enrich_references.py` 的输出
   - 包含爬取的文章内容

2. **wikipedia_structures_searched.json**
   - 来自 `expand_references.py` 的输出
   - 包含文章结构树

## 输出文件

1. **wikipedia_references_final.json**
   - 过滤并增强的引用数据
   - 包含URL、内容、路径信息

2. **wikipedia_structures_final.json**
   - 清理后的结构数据
   - 所有引用ID都有效
   - overview节点的引用已分发到兄弟节点

## 运行方法

```bash
cd /home/literism/tree/init_dataset
python finalize_references.py
```

## 处理示例

### 场景1：有效的search引用

**输入（结构片段）**：
```json
{
  "title": "Early life",
  "children": [
    {
      "title": "Education",
      "citations": ["anon_1"]  // 原有引用
    },
    {
      "title": "Family",
      "citations": []
    }
  ]
}
```

**过滤后的引用**：
```json
{
  "search_1": {
    "url": "https://example.com/einstein-education",
    "content": "This article discusses Einstein's education at ETH Zurich..."
  }
}
```

**LLM判断**：
```
Article (search_1): 文章讨论了Einstein在ETH的教育经历...
叶子节点: 1. Albert Einstein - Early life - Education
         2. Albert Einstein - Early life - Family
模型输出: "1"
```

**输出（结构片段）**：
```json
{
  "title": "Early life",
  "children": [
    {
      "title": "Education",
      "citations": ["anon_1", "search_1"]  // ← search_1被添加
    },
    {
      "title": "Family",
      "citations": []
    }
  ]
}
```

**最终引用数据**：
```json
{
  "search_1": {
    "url": "https://example.com/einstein-education",
    "content": "This article discusses Einstein's education...",
    "paths": [
      "Albert Einstein - Early life - Education"  // 由LLM匹配
    ]
  }
}
```

### 场景2：无效的search引用

**过滤后的引用**：
```json
{
  "search_2": {
    "url": "https://example.com/unrelated-topic",
    "content": "This article is about quantum mechanics in general..."
  }
}
```

**LLM判断**：
```
Article (search_2): 文章讨论量子力学的一般原理...
叶子节点: 1. Albert Einstein - Early life - Education
         2. Albert Einstein - Early life - Family
         3. Albert Einstein - Career - Patent office
模型输出: "none"
```

**结果**：
- `search_2` 从 `filtered_refs` 中被删除
- 在后续清理步骤中不会出现在最终数据中

## 性能考虑

- **API调用数**：与 `search_` 引用数量相关（不是所有引用）
- **并发控制**：通过 `MAX_CONCURRENT_JOBS` 调整（默认8）
- **处理时间估算**：
  - 假设有1000个 `search_` 引用需要处理
  - 并发8，每次API调用约1-2秒
  - 总时间：约 1000 / 8 * 1.5 = 190秒 ≈ 3分钟
- **内存使用**：
  - 需要同时加载结构和引用数据
  - 大型数据集可能需要16GB+内存
- **优化建议**：
  - 如果API调用失败率高，降低 `MAX_CONCURRENT_JOBS`
  - 可以考虑分批处理topics以减少内存压力

## 注意事项

1. **DeepSeek API**
   - 需要有效的API密钥
   - 注意API配额和速率限制
   - 建议设置合理的 `max_output_tokens`（默认128足够）

2. **Token估计**
   - 使用空格分割是简单估计
   - 实际token数可能略有不同
   - 对于中文可能不够准确（建议用tiktoken等库）

3. **模型输出解析**
   - 使用正则表达式提取所有数字
   - 特殊处理 "none"、空字符串等情况
   - 对异常输出进行容错（catch异常，跳过该引用）

4. **数据一致性**
   - 确保引用ID在结构和引用文件中一致
   - 所有路径使用统一格式（" - "分隔）
   - paths列表已自动去重

5. **引用删除逻辑**
   - 模型判定为 "none" 的引用会被删除
   - 这是正常的数据清洗过程
   - 被删除的引用不会出现在最终输出中

6. **路径收集**
   - paths包含原有路径和LLM匹配路径
   - 自动去重，无需手动处理
   - 只收集叶子节点的路径

## 故障排查

### 问题：API调用失败
- 检查API密钥是否有效
- 检查网络连接
- 降低 `MAX_CONCURRENT_JOBS`

### 问题：模型输出解析错误
- 查看具体的错误输出
- 检查prompt格式是否正确
- 调整temperature参数

### 问题：内存不足
- 降低 `MAX_TOKENS`
- 分批处理topics

---

最后更新：2025-11-27

