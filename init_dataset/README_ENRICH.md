# 引用内容丰富工具

为Wikipedia引用添加URL和文字内容。

## 功能说明

### 第一步：为文本引用搜索URL
- 读取 `wikipedia_references_searched.json`
- 找到所有只有文本没有URL的引用
- 用Bing搜索该文本，取第一个有效URL
- 如果前3个结果都无效，扩大到15个

### 第二步：爬取URL内容
- 对所有引用的URL发起HTTP请求
- 使用trafilatura提取网页文字内容
- 保存文字内容到引用中

### 第三步：保存丰富后的数据
- 每个引用包含：original（原文本/URL）、url、search_query（如果有）、content（网页文字）
- 保存到 `wikipedia_references_enriched.json`

## 安装依赖

```bash
pip install -r requirements.txt
# 或
pip install playwright requests trafilatura

# 安装浏览器
playwright install chromium
```

## 使用方法

```bash
cd /home/literism/tree/init_dataset
python enrich_references.py
```

## 输出格式

### 输入格式（wikipedia_references_searched.json）

```json
{
  "Person:Albert Einstein": {
    "topic": "Albert Einstein",
    "category": "Person",
    "pageid": 736,
    "references": {
      "BiographySource": "http://example.com/bio",
      "anon_1": "Some text without URL",
      "search_1": "http://example.com/article"
    }
  }
}
```

### 输出格式（wikipedia_references_enriched.json）

```json
{
  "Person:Albert Einstein": {
    "topic": "Albert Einstein",
    "category": "Person",
    "pageid": 736,
    "references": {
      "BiographySource": {
        "original": "http://example.com/bio",
        "url": "http://example.com/bio",
        "search_query": null,
        "content": "This is the extracted text content from the webpage..."
      },
      "anon_1": {
        "original": "Some text without URL",
        "url": "http://found-url.com/page",
        "search_query": "Some text without URL",
        "content": "Extracted content from the found URL..."
      },
      "search_1": {
        "original": "http://example.com/article",
        "url": "http://example.com/article",
        "search_query": null,
        "content": "Article text content..."
      }
    }
  }
}
```

## 运行示例

### 首次运行

```
================================================================================
丰富引用内容
================================================================================

[1] 加载数据...
  ✓ 加载了 100 个topics
  ✓ 总引用数: 500
    - 已有URL: 400
    - 只有文本: 100

[2] 处理引用...
  总共: 100 个topics
  已处理: 0 个
  剩余: 100 个

[1/100] Albert Einstein
    [BiographySource] 已有URL
      搜索: http://example.com/bio...
        ✓ 提取 5234 字符
    [anon_1] 需要搜索URL
      搜索: Some reference text...
        ✓ 找到: http://found-url.com/page...
        ✓ 提取 3421 字符
    [search_1] 已有URL
        ✓ 提取 4567 字符

[2/100] Marie Curie
    ...
```

### 中断后继续

按Ctrl+C中断后，重新运行：

```
[1] 加载数据...
  ✓ 加载了 100 个topics
  ✓ 发现已处理的数据: 30 个topics
  → 将跳过已处理的，继续处理剩余 70 个

[2] 处理引用...
  总共: 100 个topics
  已处理: 30 个
  剩余: 70 个

[31/100] Next Topic
    ...
```

## 配置选项

在 `enrich_references.py` 中修改：

```python
# 输入输出文件
INPUT_FILE = "/mnt/literism/tree/data/wikipedia_references_searched.json"
OUTPUT_FILE = "/mnt/literism/tree/data/wikipedia_references_enriched.json"

# 搜索配置
INITIAL_SEARCH_RESULTS = 3   # 初始搜索结果数
MAX_SEARCH_RESULTS = 15       # 最大搜索结果数

# 爬取配置
REQUEST_TIMEOUT = 15          # HTTP请求超时（秒）
DELAY_BETWEEN_REQUESTS = 1.0  # 请求之间延迟（秒）
```

## 时间估算

假设：
- 100个topics
- 每个topic平均20个引用
- 其中30%需要搜索URL
- 每个搜索需要3秒，每次爬取需要2秒

计算：
- URL搜索：`100 × 20 × 0.3 × 3 = 1800秒 = 30分钟`
- 内容爬取：`100 × 20 × 2 = 4000秒 = 67分钟`
- **总计：约100分钟**

## 断点续传

### 优势
- 中断不丢失进度
- 每个topic处理后立即保存
- 重启自动继续

### 重新开始

```bash
# 删除输出文件
rm /mnt/literism/tree/data/wikipedia_references_enriched.json

# 重新运行
python enrich_references.py
```

## 注意事项

1. **网络稳定性**：需要稳定的网络连接和代理

2. **延迟设置**：每个HTTP请求之间延迟1秒，避免被封锁

3. **超时处理**：单个请求超时15秒，失败不影响其他引用

4. **URL搜索**：
   - 先搜索3个结果
   - 如果没找到有效URL，扩大到8个、13个、最多15个
   - 仍未找到则标记为失败

5. **内容提取**：
   - 使用trafilatura自动提取主要内容
   - 过滤掉评论和表格
   - 失败不影响其他引用

6. **错误处理**：
   - 单个引用失败不影响其他引用
   - 单个topic失败不影响其他topics
   - 所有错误都会记录但程序继续

## 检查结果

### 查看文件大小

```bash
ls -lh /mnt/literism/tree/data/wikipedia_references_enriched.json
```

### 快速验证

```python
import json

with open('/mnt/literism/tree/data/wikipedia_references_enriched.json') as f:
    data = json.load(f)

# 统计
total_refs = 0
with_content = 0

for topic_key, topic_data in data.items():
    for ref_key, ref_data in topic_data['references'].items():
        total_refs += 1
        if ref_data.get('content'):
            with_content += 1

print(f"总引用数: {total_refs}")
print(f"有内容: {with_content} ({with_content*100/total_refs:.1f}%)")
```

## 故障排除

**问题1：搜索失败率高**
- 检查网络和代理
- 检查cookies文件
- 增加 MAX_SEARCH_RESULTS

**问题2：内容提取失败率高**
- 某些网站可能阻止爬虫
- 增加 REQUEST_TIMEOUT
- 检查网站是否需要登录

**问题3：处理太慢**
- 减少 DELAY_BETWEEN_REQUESTS（但要小心被封）
- 使用更快的网络
- 考虑并行处理（需修改代码）

**问题4：内存占用过大**
- 当前设计每次保存完整数据
- 如果topics很多（1000+），考虑分批处理

