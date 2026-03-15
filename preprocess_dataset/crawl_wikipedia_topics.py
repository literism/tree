"""
Wikipedia Topic 爬虫程序
从 topic_classified.json 读取topic列表，然后从Wikipedia爬取对应的页面
"""

import json
import time
import os
from pathlib import Path
import requests
from urllib.parse import quote
import random

# ========== 配置 ==========
INPUT_JSON = "/mnt/literism/data/result/topic_classified.json"
OUTPUT_DIR = "/mnt/literism/tree/data/wikipedia_pages"
# 爬取间隔（秒）
MIN_DELAY = 1.0
MAX_DELAY = 2.0
# Wikipedia API endpoint
WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
# User-Agent
USER_AGENT = "Mozilla/5.0 (compatible; EducationalBot/1.0)"
# ==========================


def load_topics(json_path):
    """加载topic分类数据"""
    print(f"正在加载 {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 统计总数
    total = sum(len(topics) for topics in data.values())
    print(f"加载完成：{len(data)} 个类别，共 {total} 个topics")
    return data


def get_wikipedia_page(topic_name):
    """
    通过Wikipedia API获取页面的wikitext内容
    返回: (wikitext, pageid) 或 (None, None) 如果失败
    """
    params = {
        'action': 'parse',
        'page': topic_name,
        'prop': 'wikitext',
        'format': 'json',
        'redirects': 1  # 自动跟随重定向
    }
    
    headers = {
        'User-Agent': USER_AGENT
    }
    
    try:
        response = requests.get(WIKI_API_URL, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if 'error' in data:
            print(f"  ✗ API错误: {data['error'].get('info', 'Unknown error')}")
            return None, None
        
        if 'parse' in data:
            wikitext = data['parse']['wikitext']['*']
            pageid = data['parse']['pageid']
            return wikitext, pageid
        else:
            print(f"  ✗ 未找到页面内容")
            return None, None
            
    except requests.exceptions.RequestException as e:
        print(f"  ✗ 网络错误: {e}")
        return None, None
    except Exception as e:
        print(f"  ✗ 解析错误: {e}")
        return None, None


def save_page(category, topic_name, wikitext, pageid, output_dir):
    """保存页面内容到文件"""
    # 创建类别文件夹
    category_dir = Path(output_dir) / category
    category_dir.mkdir(parents=True, exist_ok=True)
    
    # 使用安全的文件名
    safe_filename = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in topic_name)
    safe_filename = safe_filename.strip()[:100]  # 限制文件名长度
    
    # 保存wikitext
    output_file = category_dir / f"{safe_filename}.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(wikitext)
    
    # 保存元数据
    metadata = {
        'topic': topic_name,
        'category': category,
        'pageid': pageid,
        'url': f"https://en.wikipedia.org/wiki/{quote(topic_name.replace(' ', '_'))}"
    }
    metadata_file = category_dir / f"{safe_filename}.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    return output_file


def crawl_topics(topic_data, output_dir):
    """爬取所有topics"""
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 统计信息
    total_topics = sum(len(topics) for topics in topic_data.values())
    processed = 0
    success = 0
    failed = 0
    
    print(f"\n开始爬取 {total_topics} 个topics...")
    print("=" * 80)
    
    for category, topics in topic_data.items():
        print(f"\n[{category}] 开始处理 {len(topics)} 个topics")
        
        for i, topic in enumerate(topics, 1):
            processed += 1
            print(f"[{processed}/{total_topics}] {category}/{topic}...", end=" ")
            
            # 检查是否已存在
            safe_filename = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in topic)
            safe_filename = safe_filename.strip()[:100]
            existing_file = Path(output_dir) / category / f"{safe_filename}.txt"
            
            if existing_file.exists():
                print("✓ (已存在，跳过)")
                success += 1
                continue
            
            # 爬取页面
            wikitext, pageid = get_wikipedia_page(topic)
            
            if wikitext:
                try:
                    saved_file = save_page(category, topic, wikitext, pageid, output_dir)
                    print(f"✓ 已保存 ({len(wikitext)} 字符)")
                    success += 1
                except Exception as e:
                    print(f"✗ 保存失败: {e}")
                    failed += 1
            else:
                print("✗ 爬取失败")
                failed += 1
            
            # 延迟，避免被识别为攻击
            if processed < total_topics:  # 最后一个不需要延迟
                delay = random.uniform(MIN_DELAY, MAX_DELAY)
                time.sleep(delay)
    
    # 打印统计信息
    print("\n" + "=" * 80)
    print("爬取完成！")
    print(f"总计: {total_topics} 个topics")
    print(f"成功: {success}")
    print(f"失败: {failed}")
    print(f"输出目录: {output_dir}")
    print("=" * 80)


def main():
    print("=" * 80)
    print("Wikipedia Topic 爬虫程序")
    print("=" * 80)
    
    # 加载topics
    topic_data = load_topics(INPUT_JSON)
    
    # 爬取
    crawl_topics(topic_data, OUTPUT_DIR)


if __name__ == "__main__":
    main()

