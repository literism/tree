"""
最终处理引用和结构数据
1. 过滤 wikipedia_references_enriched.json，删除失败和Wikipedia URL，限制token长度
2. 使用DeepSeek API判断overview节点的文章与兄弟节点的关系
3. 将相关引用分发到兄弟节点
4. 清理 wikipedia_structures_searched.json 中的无效引用
5. 为每个引用添加叶子节点路径信息
6. 保存最终结果
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

# 添加modeling目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "modeling"))
from deepseek_api import DeepSeekAPIClient, DeepSeekConfig


# 配置
INPUT_REFERENCES = "/mnt/literism/tree/data/wikipedia_references_enriched.json"
INPUT_STRUCTURES = "/mnt/literism/tree/data/wikipedia_structures_searched.json"
OUTPUT_REFERENCES = "/mnt/literism/tree/data/wikipedia_references_final.json"
OUTPUT_STRUCTURES = "/mnt/literism/tree/data/wikipedia_structures_final.json"

MAX_TOKENS = 2048  # 最大token数
MAX_CONCURRENT_JOBS = 8  # 并发API调用数
MIN_TOKENS = 300


def is_wikipedia_url(url: str) -> bool:
    """判断是否是Wikipedia URL"""
    if not url:
        return False
    return 'wikipedia.org' in url.lower()


def truncate_to_max_tokens(text: str, max_tokens: int) -> str:
    """
    根据token数截断文本（保留开头部分）
    使用空格分割作为简单的token估计
    """
    if not text:
        return text
    
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text
    
    # 保留前max_tokens个tokens
    return ' '.join(tokens[:max_tokens])


def filter_references(enriched_refs: Dict) -> Dict:
    """
    过滤引用
    - 删除包含 failed 字段的引用
    - 删除 url 是 Wikipedia 的引用
    - 限制内容的token长度
    
    返回: {topic_key: {ref_key: {'url': xxx, 'content': xxx}}}
    """
    print("\n[1] 过滤引用并限制token长度...")
    
    filtered = {}
    total_refs = 0
    kept_refs = 0
    failed_count = 0
    wikipedia_count = 0
    truncated_count = 0
    
    for topic_key, topic_data in enriched_refs.items():
        refs = topic_data.get('references', {})
        filtered_refs = {}
        
        for ref_key, ref_data in refs.items():
            total_refs += 1
            
            # 检查是否失败
            if ref_data.get('failed'):
                failed_count += 1
                continue
            
            # 检查是否是Wikipedia URL
            url = ref_data.get('url', '')
            if is_wikipedia_url(url):
                wikipedia_count += 1
                continue
            
            # 获取内容并截断
            content = ref_data.get('content', '')
            original_tokens = len(content.split()) if content else 0

            if original_tokens < MIN_TOKENS:
                continue
            
            if content and original_tokens > MAX_TOKENS:
                content = truncate_to_max_tokens(content, MAX_TOKENS)
                truncated_count += 1
            
            flag = 0
            for word in topic_key.split():
                if word.lower() in content.lower():
                    flag = 1
            
            if flag == 0:
                continue
            
            # 保留这个引用
            filtered_refs[ref_key] = {
                'url': url if url else ref_data.get('original', ''),
                'content': content
            }
            kept_refs += 1
        
        filtered[topic_key] = {
            'topic': topic_data.get('topic'),
            'category': topic_data.get('category'),
            'pageid': topic_data.get('pageid'),
            'references': filtered_refs
        }
    
    print(f"  总引用数: {total_refs}")
    print(f"  失败引用: {failed_count}")
    print(f"  Wikipedia引用: {wikipedia_count}")
    print(f"  保留引用: {kept_refs}")
    print(f"  截断引用: {truncated_count} (超过{MAX_TOKENS} tokens)")
    
    return filtered


def collect_leaf_nodes_recursive(
    node: Dict,
    current_path: List[str],
    leaf_nodes: List[Tuple[Dict, List[str]]]
) -> None:
    """
    递归收集所有叶子节点及其路径
    
    参数:
    - node: 当前节点
    - current_path: 当前路径（不包括当前节点）
    - leaf_nodes: 结果列表 [(node, path), ...]
    """
    node_title = node.get('title', '')
    new_path = current_path + [node_title]
    
    children = node.get('children', [])
    
    if not children:  # 是叶子节点
        leaf_nodes.append((node, new_path))
    else:
        # 递归处理子节点
        for child in children:
            collect_leaf_nodes_recursive(child, new_path, leaf_nodes)


def process_search_refs_with_llm(
    structures: Dict,
    filtered_refs: Dict,
    api_client: DeepSeekAPIClient
) -> Tuple[Dict, Dict]:
    """
    使用LLM处理所有的引用，匹配到叶子节点
    
    - 对每个topic的每个引用，用LLM判断它与哪些叶子节点相关
    - 如果模型回答"none"，删除该引用
    - 否则将引用添加到对应叶子节点的citations中
    
    返回: (updated_structures, updated_filtered_refs)
    """
    print("\n[3] 使用LLM匹配search引用到叶子节点...")
    
    # 构建prompts
    prompts = []
    prompt_metadata = []
    
    total_search_refs = 0
    
    for topic_key, topic_data in structures.items():
        topic_name = topic_data.get('topic', '')
        structure = topic_data.get('structure', [])
        
        # 收集该topic的所有叶子节点
        all_leaf_nodes = []
        for node in structure:
            collect_leaf_nodes_recursive(node, [topic_name], all_leaf_nodes)
        
        if not all_leaf_nodes:
            continue
        
        topic_refs = filtered_refs.get(topic_key, {}).get('references', {})
        
        for ref_key, ref_data in topic_refs.items():
            
            total_search_refs += 1
            
            content = ref_data.get('content', '')
            if not content or len(content.strip()) < 50:
                continue
            
            # 构建prompt
            prompt = f"""Article Content:
{content[:4000]}

Section Paths (Leaf Nodes):
"""
            for idx, (node, path) in enumerate(all_leaf_nodes, 1):
                path_str = ' - '.join(path)
                prompt += f"{idx}. {path_str}\n"
            
            prompt += """\nQuestion: Which section paths does this article discuss or relate to?
Reply with the numbers only, separated by commas (e.g., "1, 3").
If the article is not relevant to any section, reply "none"."""
            
            prompts.append(prompt)
            prompt_metadata.append({
                'topic_key': topic_key,
                'ref_key': ref_key,
                'leaf_nodes': all_leaf_nodes
            })
    
    print(f"  总search引用数: {total_search_refs}")
    print(f"  构建了 {len(prompts)} 个prompts（内容足够长的）")
    
    if not prompts:
        return structures, filtered_refs
    
    # 调用API
    print(f"  调用DeepSeek API (并发数: {MAX_CONCURRENT_JOBS})...")
    responses = api_client.run_prompts_to_texts(prompts)
    
    # 解析结果
    refs_to_delete = set()  # (topic_key, ref_key)
    added_count = 0
    none_count = 0
    
    import re
    
    for metadata, response in zip(prompt_metadata, responses):
        topic_key = metadata['topic_key']
        ref_key = metadata['ref_key']
        leaf_nodes = metadata['leaf_nodes']
        
        if not response or response.strip().lower() == 'none':
            # 标记删除
            refs_to_delete.add((topic_key, ref_key))
            none_count += 1
            continue
        
        # 解析数字
        try:
            numbers = re.findall(r'\d+', response)
            selected_indices = [int(n) for n in numbers]
            
            if not selected_indices:
                continue
            
            # 添加引用到对应的叶子节点
            for idx in selected_indices:
                if 1 <= idx <= len(leaf_nodes):
                    node, path = leaf_nodes[idx - 1]
                    
                    if 'citations' not in node:
                        node['citations'] = []
                    
                    if ref_key not in node['citations']:
                        node['citations'].append(ref_key)
                        added_count += 1
        
        except Exception as e:
            print(f"  ! 解析响应失败: {response[:50]}... 错误: {e}")
            continue
    
    # 删除none的引用
    for topic_key, ref_key in refs_to_delete:
        if topic_key in filtered_refs and ref_key in filtered_refs[topic_key]['references']:
            del filtered_refs[topic_key]['references'][ref_key]
    
    print(f"  ✓ 添加了 {added_count} 个引用到叶子节点")
    print(f"  ✓ 删除了 {none_count} 个无效引用（模型判定为none）")
    
    return structures, filtered_refs


def clean_citations_recursive(node: Dict, valid_refs: Set[str]) -> None:
    """
    递归清理节点中的无效引用
    """
    if 'citations' in node:
        # 只保留在 valid_refs 中的引用，并去重
        unique_citations = []
        seen = set()
        for ref in node['citations']:
            if ref in valid_refs and ref not in seen:
                unique_citations.append(ref)
                seen.add(ref)
        node['citations'] = unique_citations
    
    # 递归处理子节点
    if 'children' in node and node['children']:
        for child in node['children']:
            clean_citations_recursive(child, valid_refs)


def is_leaf_node(node: Dict) -> bool:
    """判断是否是叶子节点"""
    children = node.get('children', [])
    return not children or len(children) == 0


def collect_leaf_paths_recursive(
    node: Dict,
    current_path: List[str],
    ref_to_paths: Dict[str, List[List[str]]]
) -> None:
    """
    递归收集叶子节点的路径
    
    参数:
    - node: 当前节点
    - current_path: 当前路径（不包括当前节点）
    - ref_to_paths: 引用到路径列表的映射
    """
    # 将当前节点添加到路径
    node_title = node.get('title', '')
    new_path = current_path + [node_title]
    
    # 如果是叶子节点，记录其引用的路径
    if is_leaf_node(node):
        citations = node.get('citations', [])
        for ref_key in citations:
            if ref_key not in ref_to_paths:
                ref_to_paths[ref_key] = []
            ref_to_paths[ref_key].append(new_path)
    else:
        # 不是叶子节点，继续递归
        for child in node.get('children', []):
            collect_leaf_paths_recursive(child, new_path, ref_to_paths)


def process_structures(structures: Dict, filtered_refs: Dict) -> tuple:
    """
    处理结构数据
    1. 清理无效引用
    2. 收集叶子节点路径
    
    返回: (cleaned_structures, ref_to_paths)
    """
    print("\n[4] 清理结构中的无效引用...")
    
    cleaned_structures = {}
    all_ref_to_paths = {}  # {topic_key: {ref_key: [[path1], [path2], ...]}}
    
    total_topics = len(structures)
    
    for idx, (topic_key, topic_data) in enumerate(structures.items(), 1):
        print(f"  [{idx}/{total_topics}] {topic_data.get('topic', topic_key)}", end='\r', flush=True)
        
        # 获取该 topic 的有效引用
        valid_refs = set()
        if topic_key in filtered_refs:
            valid_refs = set(filtered_refs[topic_key]['references'].keys())
        
        # 清理引用（去重并只保留有效的）
        structure = topic_data.get('structure', [])
        for node in structure:
            clean_citations_recursive(node, valid_refs)
        
        # 收集叶子节点路径
        topic_name = topic_data.get('topic', '')
        ref_to_paths = {}
        
        for node in structure:
            collect_leaf_paths_recursive(node, [topic_name], ref_to_paths)
        
        all_ref_to_paths[topic_key] = ref_to_paths
        
        cleaned_structures[topic_key] = topic_data
    
    print(f"\n  ✓ 处理完成 {total_topics} 个topics")
    
    return cleaned_structures, all_ref_to_paths


def add_paths_to_references(filtered_refs: Dict, all_ref_to_paths: Dict) -> Dict:
    """
    为引用添加路径信息
    
    参数:
    - filtered_refs: 过滤后的引用 {topic_key: {'references': {ref_key: {'url': xxx, 'content': xxx}}}}
    - all_ref_to_paths: {topic_key: {ref_key: [[path], ...]}}
    
    返回: 添加了路径信息的引用字典
    """
    print("\n[5] 添加路径信息...")
    
    final_refs = {}
    total_refs_with_paths = 0
    total_paths = 0
    total_paths_after_dedup = 0
    
    for topic_key, topic_data in filtered_refs.items():
        refs = topic_data.get('references', {})
        enhanced_refs = {}
        
        topic_ref_to_paths = all_ref_to_paths.get(topic_key, {})
        
        for ref_key, ref_data in refs.items():
            paths = topic_ref_to_paths.get(ref_key, [])
            
            # 将路径转换为字符串列表并去重（保持顺序）
            path_strings = []
            seen_paths = set()
            for path in paths:
                path_str = ' - '.join(path)
                if path_str not in seen_paths:
                    path_strings.append(path_str)
                    seen_paths.add(path_str)
            
            enhanced_refs[ref_key] = {
                'url': ref_data.get('url', ''),
                'content': ref_data.get('content', ''),
                'paths': path_strings
            }
            
            if paths:
                total_refs_with_paths += 1
                total_paths += len(paths)
                total_paths_after_dedup += len(path_strings)
        
        final_refs[topic_key] = {
            'topic': topic_data.get('topic'),
            'category': topic_data.get('category'),
            'pageid': topic_data.get('pageid'),
            'references': enhanced_refs
        }
    
    print(f"  引用总数: {sum(len(t['references']) for t in final_refs.values())}")
    print(f"  有路径的引用: {total_refs_with_paths}")
    print(f"  总路径数（去重前）: {total_paths}")
    print(f"  总路径数（去重后）: {total_paths_after_dedup}")
    
    return final_refs


def main():
    """主函数"""
    print("=" * 80)
    print("最终处理引用和结构数据")
    print("=" * 80)
    
    # 0. 读取数据
    print("\n[0] 加载数据...")
    
    try:
        with open(INPUT_REFERENCES, 'r', encoding='utf-8') as f:
            enriched_refs = json.load(f)
        print(f"  ✓ 加载 {len(enriched_refs)} 个topics的引用数据")
    except Exception as e:
        print(f"  ✗ 加载引用数据失败: {e}")
        return
    
    try:
        with open(INPUT_STRUCTURES, 'r', encoding='utf-8') as f:
            structures = json.load(f)
        print(f"  ✓ 加载 {len(structures)} 个topics的结构数据")
    except Exception as e:
        print(f"  ✗ 加载结构数据失败: {e}")
        return
    
    # 1. 过滤引用并限制token长度
    filtered_refs = filter_references(enriched_refs)
    
    # 2. 初始化DeepSeek API客户端
    print("\n[2] 初始化DeepSeek API客户端...")
    config = DeepSeekConfig(
        api_key="sk-3f2e7fe4ae6e4d588c619bbff9837dac",
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
        temperature=0.1,
        max_output_tokens=128,
        max_concurrent_jobs=MAX_CONCURRENT_JOBS
    )
    api_client = DeepSeekAPIClient(config)
    print(f"  ✓ 客户端初始化完成 (并发数: {MAX_CONCURRENT_JOBS})")
    
    # 3. 使用LLM匹配引用到叶子节点（新增或删除引用）
    structures, filtered_refs = process_search_refs_with_llm(structures, filtered_refs, api_client)
    
    # 4. 清理结构中的无效引用 + 收集叶子节点路径
    cleaned_structures, all_ref_to_paths = process_structures(structures, filtered_refs)
    
    # 5. 为引用添加路径信息
    final_refs = add_paths_to_references(filtered_refs, all_ref_to_paths)
    
    # 6. 保存结果
    print("\n[6] 保存结果...")
    
    try:
        Path(OUTPUT_REFERENCES).parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_REFERENCES, 'w', encoding='utf-8') as f:
            json.dump(final_refs, f, ensure_ascii=False, indent=2)
        print(f"  ✓ 引用数据已保存: {OUTPUT_REFERENCES}")
    except Exception as e:
        print(f"  ✗ 保存引用数据失败: {e}")
    
    try:
        Path(OUTPUT_STRUCTURES).parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_STRUCTURES, 'w', encoding='utf-8') as f:
            json.dump(cleaned_structures, f, ensure_ascii=False, indent=2)
        print(f"  ✓ 结构数据已保存: {OUTPUT_STRUCTURES}")
    except Exception as e:
        print(f"  ✗ 保存结构数据失败: {e}")
    
    # 7. 统计信息
    print("\n" + "=" * 80)
    print("处理完成！")
    print("=" * 80)
    print(f"总topics数: {len(final_refs)}")
    print(f"总引用数: {sum(len(t['references']) for t in final_refs.values())}")
    print(f"总路径数: {sum(len(r.get('paths', [])) for t in final_refs.values() for r in t['references'].values())}")
    print(f"\n输出文件:")
    print(f"  - {OUTPUT_REFERENCES}")
    print(f"  - {OUTPUT_STRUCTURES}")
    print("=" * 80)


if __name__ == "__main__":
    main()

