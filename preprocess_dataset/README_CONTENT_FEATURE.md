# Wikipedia结构解析器 - 段落内容保存功能

## 功能说明

`parse_wikipedia_structure.py` 现在支持可选的段落内容保存功能。

## 配置参数

在文件顶部的配置部分添加了新参数：

```python
# 是否保存段落内容（如果为True，会在每个节点中保存content字段）
SAVE_CONTENT = False
```

### 参数说明

- **`SAVE_CONTENT = False`** (默认)
  - 只提取文章结构和引用
  - 不保存段落文字内容
  - 输出文件较小，适合引用分析

- **`SAVE_CONTENT = True`**
  - 提取文章结构、引用和段落内容
  - 在每个节点中添加 `content` 字段
  - 输出文件较大，适合内容分析

## 输出格式对比

### SAVE_CONTENT = False (默认)

```json
{
  "Person:Albert Einstein": {
    "structure": [
      {
        "title": "Early life",
        "level": 2,
        "citations": ["BiographySource", "anon_1"],
        "children": [
          {
            "title": "overview",
            "level": 3,
            "citations": ["BiographySource"],
            "children": []
          }
        ]
      }
    ]
  }
}
```

### SAVE_CONTENT = True

```json
{
  "Person:Albert Einstein": {
    "structure": [
      {
        "title": "Early life",
        "level": 2,
        "citations": ["BiographySource", "anon_1"],
        "content": "Albert Einstein was born in Ulm, Germany...",
        "children": [
          {
            "title": "overview",
            "level": 3,
            "citations": ["BiographySource"],
            "content": "Einstein showed exceptional intellect from a young age...",
            "children": []
          }
        ]
      }
    ]
  }
}
```

## 内容清理

保存的内容会自动进行以下清理：

1. **移除模板标记**: `{{...}}` → 删除（如 `{{Infobox}}`, `{{IPA}}`）
2. **移除HTML标签**: `<...>` → 删除（如 `<ref>`, `<div>`）
3. **移除文件/图片链接**: `[[File:...]]`, `[[Image:...]]` → 删除
4. **处理Wiki链接**: 保留显示文字
   - `[[Albert Einstein]]` → `Albert Einstein`
   - `[[Theory of relativity|relativity]]` → `relativity`
   - `[[Germany|German]]-born` → `German-born`
5. **移除格式标记**: `'''粗体'''` → `粗体`, `''斜体''` → `斜体`
6. **清理空白**: 多余空格、换行 → 单个空格

清理后的文本是纯文本格式，保留了所有实质内容，便于阅读和后续处理。

### 清理示例

**原始文本**:
```
'''Albert Einstein''' ({{IPA|/ˈaɪnstaɪn/}}; [[German language|German]]: 
was a [[Germany|German]]-born [[theoretical physics|theoretical physicist]].
```

**清理后**:
```
Albert Einstein (; German: was a German-born theoretical physicist.
```

**关键特性**:
- ✅ 保留了所有实质内容（人名、国籍、职业）
- ✅ 移除了技术标记（模板、链接语法）
- ✅ 保持了可读性
- ✅ 链接文字完整保留

## 使用方法

### 方法1：修改配置文件

直接修改 `parse_wikipedia_structure.py` 中的配置：

```python
# 改为 True 启用内容保存
SAVE_CONTENT = True
```

然后运行：

```bash
python parse_wikipedia_structure.py
```

### 方法2：程序化调用

```python
from parse_wikipedia_structure import process_all_pages

# 只提取引用（默认）
process_all_pages(
    input_dir="/path/to/wikipedia_pages",
    structure_output="/path/to/output_structures.json",
    references_output="/path/to/output_references.json",
    save_content=False
)

# 保存段落内容
process_all_pages(
    input_dir="/path/to/wikipedia_pages",
    structure_output="/path/to/output_structures_with_content.json",
    references_output="/path/to/output_references.json",
    save_content=True
)
```

## 性能考虑

### 文件大小

- **SAVE_CONTENT = False**: 约 1-2 MB（100个topics）
- **SAVE_CONTENT = True**: 约 10-20 MB（100个topics）

文件大小会随着内容量线性增长。

### 处理速度

保存内容会略微增加处理时间（约10-20%），主要用于文本清理。

### 内存使用

保存内容模式会增加内存使用，但影响不大（通常<500MB）。

## 应用场景

### SAVE_CONTENT = False (推荐)

适用于：
- ✅ 引用链接分析
- ✅ 文章结构分析
- ✅ 引用分发和匹配
- ✅ 需要小文件大小

### SAVE_CONTENT = True

适用于：
- ✅ 内容质量分析
- ✅ 文本摘要生成
- ✅ 段落相关性判断
- ✅ 完整数据集构建

## 注意事项

1. **数据完整性**
   - 内容清理是不可逆的
   - 如果需要原始wikitext，请保留原始文件

2. **中文支持**
   - 文本清理对中文友好
   - 正则表达式支持Unicode

3. **空内容处理**
   - 如果节点没有实质内容，`content` 字段会被省略
   - 避免保存大量空白字段

4. **与其他工具的兼容性**
   - `expand_references.py` - 兼容两种模式
   - `enrich_references.py` - 兼容两种模式
   - `finalize_references.py` - 兼容两种模式

## 示例：启用内容保存

```bash
cd /home/literism/tree/preprocess_dataset

# 1. 修改配置
# 在 parse_wikipedia_structure.py 中设置 SAVE_CONTENT = True

# 2. 运行解析
python parse_wikipedia_structure.py

# 3. 检查输出
# 每个节点会包含 content 字段
```

## 总结

- **默认行为**: 不保存内容，保持向后兼容
- **新功能**: 通过 `SAVE_CONTENT` 参数控制
- **灵活性**: 可以根据需求选择模式
- **性能影响**: 轻微，可接受

---

最后更新：2025-12-07

