"""
扩充Wikipedia结构的引用
通过搜索引擎为每个叶子节点查找并添加引用
"""

import json
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote
from typing import Dict, List, Tuple, Set
from bing_search_playwright import search_bing
import urllib.parse
import re
import base64


# 配置
INPUT_STRUCTURE = "/mnt/literism/tree/data/wikipedia_structures.json"
INPUT_REFERENCES = "/mnt/literism/tree/data/wikipedia_references.json"
OUTPUT_STRUCTURE = "/mnt/literism/tree/data/wikipedia_structures_searched.json"
OUTPUT_REFERENCES = "/mnt/literism/tree/data/wikipedia_references_searched.json"

NUM_RESULTS_PER_QUERY = 15
DELAY_BETWEEN_QUERIES = 2.0


def extract_real_url(bing_url: str) -> str:
    """
    从Bing跳转URL中提取真实URL
    
    参数:
    - bing_url: Bing的跳转URL
    
    返回:
    - 真实的目标URL
    """
    match = re.search(r"[?&]u=([^&]+)", bing_url)
    if not match:
        return bing_url  # 不是跳转链接，直接返回

    encoded = match.group(1)

    # 去掉 a1 前缀
    if encoded.startswith("a1"):
        encoded = encoded[2:]

    # Base64-url -> 标准 Base64
    encoded = encoded.replace("-", "+").replace("_", "/")

    # 补齐 padding
    padding = len(encoded) % 4
    if padding:
        encoded += "=" * (4 - padding)

    try:
        decoded_bytes = base64.b64decode(encoded)
        decoded_url = decoded_bytes.decode("utf-8")
        if decoded_url.startswith("http"):
            return decoded_url
        else:
            return bing_url
    except Exception:
        return bing_url


def build_query_path(node_path: List[str], topic: str) -> str:
    """
    构建搜索query
    
    参数:
    - node_path: 从根到叶子节点的路径（不包括topic）
    - topic: topic名称
    
    返回:
    - 搜索query字符串
    """
    # 过滤掉overview和空字符串
    filtered_path = [p for p in node_path if p.lower() != 'overview' and p.strip()]
    
    # 构建query: topic - title - subtitle - ...
    parts = [topic] + filtered_path
    query = " - ".join(parts)
    
    return query


def find_leaf_nodes(structure: List[Dict], current_path: List[str] = None) -> List[Tuple[List[str], Dict]]:
    """
    递归查找所有叶子节点
    
    参数:
    - structure: 结构树
    - current_path: 当前路径
    
    返回:
    - [(路径, 节点), ...] 列表
    """
    if current_path is None:
        current_path = []
    
    leaf_nodes = []
    
    for node in structure:
        node_title = node['title']
        children = node.get('children', [])
        
        # 如果有子节点，不处理当前节点，递归处理子节点
        if children:
            # 递归处理子节点
            child_path = current_path + [node_title]
            leaf_nodes.extend(find_leaf_nodes(children, child_path))
        else:
            # 叶子节点
            leaf_nodes.append((current_path + [node_title], node))
    
    return leaf_nodes


def process_single_topic(
    topic_key: str,
    topic_data: Dict,
    references_dict: Dict[str, str],
    search_counter: int,
    verbose: bool = True
) -> Tuple[Dict, Dict[str, str], int]:
    """
    处理单个topic
    
    参数:
    - topic_key: topic键（如"Person:Albert Einstein"）
    - topic_data: topic数据
    - references_dict: 引用字典
    - search_counter: 当前search_id计数器
    - verbose: 是否显示详细信息
    
    返回:
    - (更新后的topic_data, 更新后的references_dict, 新的计数器值)
    """
    topic = topic_data['topic']
    structure = topic_data['structure']
    
    if verbose:
        print(f"\n处理 Topic: {topic}")
    
    # 找到所有叶子节点
    leaf_nodes = find_leaf_nodes(structure)
    
    if verbose:
        print(f"  找到 {len(leaf_nodes)} 个叶子节点")
    
    if not leaf_nodes:
        return topic_data, references_dict, search_counter
    
    # 构建所有queries
    queries = []
    for path, node in leaf_nodes:
        query = build_query_path(path, topic)
        queries.append(query)
    
    # 批量搜索
    if verbose:
        print(f"  开始搜索 {len(queries)} 个queries...")
    
    try:
        search_results = search_bing(
            queries=queries,
            num_results=NUM_RESULTS_PER_QUERY,
            delay=DELAY_BETWEEN_QUERIES,
            verbose=False  # 不显示搜索详情
        )
    except Exception as e:
        print(f"  ✗ 搜索失败: {e}")
        return topic_data, references_dict, search_counter
    
    # 处理每个叶子节点的搜索结果
    for (path, node), query in zip(leaf_nodes, queries):
        results = search_results.get(query, [])
        
        if not results:
            if verbose:
                print(f"    - {' > '.join(path)}: 无结果")
            continue
        
        # 提取真实URL
        real_urls = []
        for result in results:
            real_url = extract_real_url(result['url'])
            real_urls.append(real_url)
        
        # 去重
        real_urls = list(dict.fromkeys(real_urls))  # 保持顺序的去重
        
        # 构建URL到key的映射
        url_to_key = {v: k for k, v in references_dict.items() if isinstance(v, str) and v.startswith('http')}
        
        # 为每个URL分配或查找key
        new_citations = []
        for url in real_urls:
            if url in url_to_key:
                # URL已存在，使用已有key
                key = url_to_key[url]
            else:
                # 新URL，创建新key
                key = f"search_{search_counter}"
                search_counter += 1
                references_dict[key] = url
                url_to_key[url] = key
            
            new_citations.append(key)
        
        # 合并到节点的citations中（去重）
        existing_citations = set(node.get('citations', []))
        combined_citations = list(existing_citations | set(new_citations))
        node['citations'] = combined_citations
        
        if verbose:
            print(f"    - {' > '.join(path)}: +{len(new_citations)} 个引用")
    
    return topic_data, references_dict, search_counter


def main():
    """主函数"""
    print("=" * 80)
    print("扩充Wikipedia引用")
    print("=" * 80)
    
    # 1. 读取输入文件
    print("\n[1] 加载数据...")
    
    # 读取结构文件（JSON格式）
    try:
        with open(INPUT_STRUCTURE, 'r', encoding='utf-8') as f:
            structures = json.load(f)
        print(f"  ✓ 加载了 {len(structures)} 个topics")
    except FileNotFoundError:
        print(f"  ✗ 文件不存在: {INPUT_STRUCTURE}")
        return
    except Exception as e:
        print(f"  ✗ 读取结构文件失败: {e}")
        return
    
    # 尝试读取已有的输出文件（断点续传）
    processed_structures = {}
    processed_topics = set()
    if Path(OUTPUT_STRUCTURE).exists():
        try:
            with open(OUTPUT_STRUCTURE, 'r', encoding='utf-8') as f:
                processed_structures = json.load(f)
            processed_topics = set(processed_structures.keys())
            print(f"  ✓ 发现已处理的数据: {len(processed_topics)} 个topics")
            print(f"  → 将跳过已处理的，继续处理剩余 {len(structures) - len(processed_topics)} 个")
        except Exception as e:
            print(f"  ! 读取输出文件失败，将重新开始: {e}")
            processed_structures = {}
            processed_topics = set()
    
    # 读取引用字典
    try:
        with open(INPUT_REFERENCES, 'r', encoding='utf-8') as f:
            references_data = json.load(f)
        
        # 提取所有topic的引用到一个统一的字典
        all_references = {}
        for topic_key, topic_refs in references_data.items():
            if 'references' in topic_refs:
                all_references.update(topic_refs['references'])
        
        print(f"  ✓ 加载了 {len(all_references)} 个现有引用")
    except FileNotFoundError:
        print(f"  ✗ 文件不存在: {INPUT_REFERENCES}")
        all_references = {}
    except Exception as e:
        print(f"  ✗ 读取引用文件失败: {e}")
        all_references = {}
    
    # 2. 处理每个topic
    print("\n[2] 开始处理topics...")
    remaining_count = len(structures) - len(processed_topics)
    print(f"  总共: {len(structures)} 个topics")
    print(f"  已处理: {len(processed_topics)} 个")
    print(f"  剩余: {remaining_count} 个")
    print(f"  每个叶子节点搜索: {NUM_RESULTS_PER_QUERY} 个结果")
    print(f"  延迟: {DELAY_BETWEEN_QUERIES} 秒/query")
    print()
    
    # 找出search_开头的最大编号
    search_counter = 1
    for key in all_references.keys():
        if key.startswith('search_'):
            try:
                num = int(key.split('_')[1])
                search_counter = max(search_counter, num + 1)
            except:
                pass
    
    processed_count = len(processed_topics)
    total_count = len(structures)
    newly_processed = 0
    
    # 合并已处理的结构
    for key in processed_topics:
        if key in processed_structures:
            structures[key] = processed_structures[key]
    
    for topic_key, topic_data in structures.items():
        # 跳过已处理的topic
        if topic_key in processed_topics:
            continue
        
        processed_count += 1
        newly_processed += 1
        print(f"[{processed_count}/{total_count}] ", end='')
        
        try:
            topic_data, all_references, search_counter = process_single_topic(
                topic_key=topic_key,
                topic_data=topic_data,
                references_dict=all_references,
                search_counter=search_counter,
                verbose=True
            )
            
            structures[topic_key] = topic_data
            
            # 立即保存结果（断点续传）
            try:
                # 保存结构
                Path(OUTPUT_STRUCTURE).parent.mkdir(parents=True, exist_ok=True)
                with open(OUTPUT_STRUCTURE, 'w', encoding='utf-8') as f:
                    json.dump(structures, f, ensure_ascii=False, indent=2)
                
                # 保存引用
                output_references = {}
                for tk, td in structures.items():
                    topic_citations = set()
                    
                    def collect_citations(nodes):
                        for node in nodes:
                            topic_citations.update(node.get('citations', []))
                            if node.get('children'):
                                collect_citations(node['children'])
                    
                    collect_citations(td['structure'])
                    
                    output_references[tk] = {
                        'topic': td['topic'],
                        'category': td['category'],
                        'pageid': td.get('pageid'),
                        'references': {k: all_references[k] for k in topic_citations if k in all_references}
                    }
                
                with open(OUTPUT_REFERENCES, 'w', encoding='utf-8') as f:
                    json.dump(output_references, f, ensure_ascii=False, indent=2)
                
            except Exception as save_error:
                print(f"  ! 保存失败: {save_error}")
            
        except Exception as e:
            print(f"  ✗ 处理失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 3. 最终确认保存
    print("\n[3] 最终确认...")
    print(f"  ✓ 所有数据已保存（每个topic处理后立即保存）")
    print(f"  ✓ 输出文件: {OUTPUT_STRUCTURE}")
    print(f"  ✓ 引用文件: {OUTPUT_REFERENCES}")
    
    # 4. 统计信息
    print("\n" + "=" * 80)
    print("处理完成！")
    print("=" * 80)
    print(f"总topics数: {total_count}")
    print(f"  - 之前已处理: {len(processed_topics)}")
    print(f"  - 本次新处理: {newly_processed}")
    print(f"总引用数: {len(all_references)}")
    print(f"  - search_引用: {search_counter - 1}")
    print(f"输出文件:")
    print(f"  - 结构: {OUTPUT_STRUCTURE}")
    print(f"  - 引用: {OUTPUT_REFERENCES}")
    print("=" * 80)
    
    if newly_processed == 0:
        print("\n提示: 所有topics都已处理完成。如需重新处理，请删除输出文件。")


if __name__ == "__main__":
    main()

