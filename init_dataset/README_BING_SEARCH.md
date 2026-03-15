# Bing搜索工具（Playwright版本）

基于Playwright的Bing搜索工具，模拟真实浏览器访问，搜索结果更准确。

## 安装依赖

```bash
pip install playwright
playwright install chromium
```

## 快速使用

### 基本用法

```python
from bing_search_playwright import search_bing

# 搜索单个query
results = search_bing("Python tutorial")

# 搜索多个queries
results = search_bing([
    "machine learning",
    "deep learning",
    "neural networks"
])

# 访问结果
for query, items in results.items():
    print(f"{query}:")
    for item in items:
        print(f"  - {item['title']}")
        print(f"    {item['url']}")
```

## 函数参数

```python
search_bing(
    queries,              # 单个字符串或字符串列表
    num_results=10,       # 每个query返回的结果数
    delay=2.0,            # query之间的延迟（秒）
    cookie_file="./cookies.txt",  # cookies文件路径
    proxy="http://10.62.196.96:7891",  # 代理服务器
    headless=True,        # 无头模式
    verbose=False         # 是否打印进度信息
)
```

## 返回格式

```python
{
    "query1": [
        {"title": "标题1", "url": "https://..."},
        {"title": "标题2", "url": "https://..."},
        ...
    ],
    "query2": [
        {"title": "标题1", "url": "https://..."},
        ...
    ]
}
```

## 使用示例

### 示例1：搜索并提取URL

```python
from bing_search_playwright import search_bing

results = search_bing("artificial intelligence", num_results=5)

for query, items in results.items():
    urls = [item['url'] for item in items]
    print(f"找到 {len(urls)} 个结果")
    for url in urls:
        print(url)
```

### 示例2：批量搜索多个关键词

```python
from bing_search_playwright import search_bing

queries = [
    "quantum computing",
    "blockchain technology",
    "5G networks"
]

results = search_bing(
    queries=queries,
    num_results=3,
    delay=2.0  # 每个query之间等待2秒
)

# 统计结果
for query, items in results.items():
    print(f"{query}: {len(items)} 个结果")
```

### 示例3：保存结果到文件

```python
from bing_search_playwright import search_bing
import json

results = search_bing(["data science", "big data"], num_results=10)

# 保存为JSON
with open('search_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
```

### 示例4：显示详细进度

```python
from bing_search_playwright import search_bing

# verbose=True 会打印搜索进度
results = search_bing(
    queries=["topic1", "topic2", "topic3"],
    num_results=5,
    verbose=True  # 显示进度
)
```

### 示例5：自定义配置

```python
from bing_search_playwright import search_bing

results = search_bing(
    queries=["python programming"],
    num_results=20,
    delay=3.0,
    cookie_file="/path/to/cookies.txt",
    proxy="http://your-proxy:port",
    headless=False  # 显示浏览器窗口（调试用）
)
```

## Cookies文件

Cookies文件格式为Netscape格式，可以通过浏览器扩展导出。

示例格式：
```
# Netscape HTTP Cookie File
.bing.com	TRUE	/	TRUE	1234567890	cookie_name	cookie_value
```

## 代理设置

支持HTTP/HTTPS代理：

```python
results = search_bing(
    queries=["query"],
    proxy="http://127.0.0.1:7890"  # 本地代理
)
```

## 注意事项

1. **延迟设置**：默认每个query之间延迟2秒，避免请求过快被封锁
2. **无头模式**：Linux服务器必须使用`headless=True`
3. **代理**：如果需要访问国际版Bing，建议配置代理
4. **Cookies**：可选，某些情况下使用cookies可以提高成功率

## 命令行使用

```bash
# 搜索单个query
python bing_search_playwright.py "machine learning"

# 搜索多个queries
python bing_search_playwright.py "topic1" "topic2" "topic3"
```

## 与原版对比

**优势：**
- 使用真实浏览器，搜索结果更准确
- 支持JavaScript渲染
- 更难被检测为爬虫
- 支持cookies和代理

**劣势：**
- 需要安装Playwright和浏览器
- 速度相对较慢
- 资源占用较大

## 故障排除

**问题1：ImportError: No module named 'playwright'**  
解决：`pip install playwright && playwright install chromium`

**问题2：TimeoutError**  
解决：增加timeout或检查网络连接

**问题3：搜索结果为空**  
解决：
- 检查代理设置
- 尝试使用cookies
- 设置`headless=False`查看浏览器状态

**问题4：Linux服务器无法启动浏览器**  
解决：确保使用`headless=True`，并安装必要的系统依赖

