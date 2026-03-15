# Wikipedia 爬虫和解析使用说明

## 快速开始

### 1. 安装依赖
```bash
pip install requests mwparserfromhell
```

### 2. 爬取Wikipedia页面
```bash
python crawl_wikipedia_topics.py
```

这会从 `/mnt/literism/data/result/topic_classified.json` 读取topic列表，然后爬取页面到 `/mnt/literism/data/result/wikipedia_pages/`

### 3. 解析页面结构和引用
```bash
python parse_wikipedia_structure.py
```

这会解析所有爬取的页面，输出两个文件：
- **结构文件**：`/mnt/literism/data/result/wikipedia_structures.json` - 所有topic的文章结构
- **引用文件**：`/mnt/literism/data/result/wikipedia_references.json` - 所有topic的引用字典

### 4. 测试单个topic（可选）
```bash
python test_wikipedia_pipeline.py "Albert Einstein"
```

## 输出格式

### 结构文件 (wikipedia_structures.json)
```json
{
  "Person:Albert Einstein": {
    "topic": "Albert Einstein",
    "category": "Person",
    "pageid": 736,
    "url": "https://en.wikipedia.org/wiki/Albert_Einstein",
    "structure": [
      {
        "title": "Early life",
        "level": 2,
        "citations": ["BiographySource", "anon_1", "anon_2"],
        "children": [
          {
            "title": "overview",
            "level": 3,
            "citations": ["BiographySource"],
            "children": []
          },
          {
            "title": "Education",
            "level": 3,
            "citations": ["anon_1", "anon_2"],
            "children": []
          }
        ]
      }
    ]
  }
}
```

### 引用文件 (wikipedia_references.json)
```json
{
  "Person:Albert Einstein": {
    "topic": "Albert Einstein",
    "category": "Person",
    "pageid": 736,
    "references": {
      "BiographySource": "http://example.com/biography",
      "EducationHistory": "http://example.com/education",
      "anon_1": "http://example.com/some-url",
      "anon_2": "Reference text without URL"
    }
  }
}
```

## 引用提取逻辑

### 1. 从References部分建立基础字典
- 遍历`== References ==`部分的所有有`name`属性的`<ref>`标签
- 提取第一个URL（如果有），否则提取文本内容
- 格式：`{ref_name: url或文本}`

### 2. 处理段落中的引用
对每个段落中的ref标签：

**情况A：引用中包含URL**
- 例如：`<ref>http://example.com</ref>` 或 `<ref>文字 http://example.com</ref>`
- 在字典中查找是否已有相同URL
- 如果有：使用该URL对应的key
- 如果没有：创建新的`anon_N` key并加入字典

**情况B：引用有name但无URL**
- 例如：`<ref name="smith2020" />` 或 `<ref name="smith2020">内容</ref>`
- 检查name是否在字典中
- 如果在：使用该name作为key
- 如果不在：**跳过该引用**

**情况C：既无URL也无name**
- 跳过该引用

### 3. 父节点包含所有子节点的引用
- 每个标题的`citations`包含该部分及所有子部分的引用

## 文件结构

```
/mnt/literism/data/result/
├── topic_classified.json           # 输入：topic分类
├── wikipedia_pages/                # 爬取的原始页面
│   ├── Person/
│   │   ├── Albert_Einstein.txt
│   │   ├── Albert_Einstein.json
│   │   └── ...
│   └── Company/
│       └── ...
├── wikipedia_structures.json       # 输出：所有topic的结构汇总
└── wikipedia_references.json       # 输出：所有topic的引用汇总
```

## 配置修改

在 `crawl_wikipedia_topics.py` 中：
```python
INPUT_JSON = "/mnt/literism/data/result/topic_classified.json"
OUTPUT_DIR = "/mnt/literism/data/result/wikipedia_pages"
MIN_DELAY = 1.0  # 爬取延迟（秒）
MAX_DELAY = 3.0
```

在 `parse_wikipedia_structure.py` 中：
```python
INPUT_DIR = "/mnt/literism/data/result/wikipedia_pages"
STRUCTURE_OUTPUT = "/mnt/literism/data/result/wikipedia_structures.json"
REFERENCES_OUTPUT = "/mnt/literism/data/result/wikipedia_references.json"
```

## 注意事项

1. 爬取速度：每个topic间隔1-3秒，100个topic约需3-5分钟
2. 断点续传：重新运行爬虫会自动跳过已下载的页面
3. 引用匹配：基于URL匹配和name属性匹配
4. 跳过的章节：See also、Notes、Sources、External links、References等

