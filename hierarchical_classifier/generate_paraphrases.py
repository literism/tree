"""
生成标题的Paraphrase
为每个标题生成5种不同的改写形式，保存为树结构
"""
import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import copy

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))
from modeling.deepseek_api import DeepSeekAPIClient, DeepSeekConfig


def collect_titles_from_tree(node: Dict, current_path: str, tasks: List, is_root: bool = False) -> None:
    """
    从树结构中收集所有需要生成paraphrase的任务
    
    Args:
        node: 当前节点
        current_path: 当前路径
        tasks: 任务列表 [(full_path, title), ...]
        is_root: 是否是根节点（根节点不生成paraphrase）
    """
    if 'title' in node and node['title'] and not is_root:
        title = node['title']
        # 构建完整路径
        full_path = f"{current_path} - {title}" if current_path else title
        tasks.append((full_path, title))
        current_path = full_path
    
    # 递归处理子节点
    if 'children' in node and node['children']:
        for child in node['children']:
            collect_titles_from_tree(child, current_path, tasks, is_root=False)


def collect_all_tasks(structures_file: str) -> Tuple[Dict, List]:
    """
    从wikipedia_structures_final.json中收集所有任务
    
    Args:
        structures_file: 结构文件路径
        
    Returns:
        (原始结构数据, 任务列表 [(topic_key, full_path, title), ...])
    """
    print("=" * 80)
    print("加载结构数据...")
    print("=" * 80)
    
    with open(structures_file, 'r', encoding='utf-8') as f:
        structures = json.load(f)
    
    all_tasks = []
    
    for topic_key, topic_data in structures.items():
        topic_name = topic_data.get('topic', '')
        
        if 'structure' in topic_data and topic_data['structure']:
            if isinstance(topic_data['structure'], list):
                # structure是列表
                for child in topic_data['structure']:
                    tasks = []
                    collect_titles_from_tree(child, topic_name, tasks, is_root=False)
                    for full_path, title in tasks:
                        all_tasks.append((topic_key, full_path, title))
            else:
                # structure是单个节点
                tasks = []
                collect_titles_from_tree(topic_data['structure'], "", tasks, is_root=True)
                for full_path, title in tasks:
                    all_tasks.append((topic_key, full_path, title))
    
    print(f"\n统计:")
    print(f"  - 处理 {len(structures)} 个topics")
    print(f"  - 收集到 {len(all_tasks)} 个标题（排除了topic根节点）")
    
    return structures, all_tasks


def create_paraphrase_prompt(title: str, full_path: str) -> str:
    """
    创建生成Paraphrase的prompt
    
    Args:
        title: 标题
        full_path: 完整路径
        
    Returns:
        prompt文本
    """
    prompt = f"""You are a professional text paraphrasing expert. Your task is to generate 5 different paraphrases for a given title/subtitle while maintaining its original meaning and context.

TITLE: {title}
FULL PATH: {full_path}

Please generate EXACTLY 5 paraphrases with the following characteristics:
1. Synonym Replacement: Use synonyms to replace key words while keeping the same structure
2. Generalization: Express the concept in a more general/abstract way
3. Specification: Express the concept in a more specific/concrete way
4. Concise Version: A shorter, more compact expression of the same idea
5. Restructured: Completely reorganize the sentence structure while keeping the meaning

CRITICAL REQUIREMENTS:
- Each paraphrase MUST maintain the EXACT original meaning within the context of the full path
- If a paraphrase would change the semantic meaning even slightly, output "SKIP" for that field
- Do NOT change the semantic scope or introduce new concepts
- Keep the paraphrases relevant to the hierarchical context
- Output MUST be in valid JSON format only, no additional text

Output format (JSON only):
{{
  "synonym": "..." or "SKIP",
  "generalized": "..." or "SKIP",
  "specific": "..." or "SKIP",
  "concise": "..." or "SKIP",
  "restructured": "..." or "SKIP"
}}"""
    
    return prompt


def parse_paraphrase_result(text: str) -> Dict[str, str]:
    """
    解析Paraphrase生成结果
    
    Args:
        text: API返回的文本
        
    Returns:
        解析后的字典，如果解析失败返回空字典。值为"SKIP"的会被过滤掉
    """
    if not text:
        return {}
    
    try:
        # 尝试直接解析JSON
        result = json.loads(text)
        
        # 验证必需的键
        required_keys = ["synonym", "generalized", "specific", "concise", "restructured"]
        if all(key in result for key in required_keys):
            # 过滤掉值为"SKIP"的
            filtered_result = {k: v for k, v in result.items() if v != "SKIP"}
            return filtered_result
        else:
            return {}
    except json.JSONDecodeError:
        # 尝试提取JSON部分
        try:
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                json_text = text[start:end]
                result = json.loads(json_text)
                
                required_keys = ["synonym", "generalized", "specific", "concise", "restructured"]
                if all(key in result for key in required_keys):
                    # 过滤掉值为"SKIP"的
                    filtered_result = {k: v for k, v in result.items() if v != "SKIP"}
                    return filtered_result
        except:
            pass
        
        return {}


def add_paraphrases_to_tree(node: Dict, paraphrases_map: Dict[str, Dict], current_path: str = "", is_root: bool = False) -> None:
    """
    将paraphrase添加到树节点中
    
    Args:
        node: 当前节点
        paraphrases_map: 路径到paraphrase的映射 {full_path: paraphrases}
        current_path: 当前路径
        is_root: 是否是根节点
    """
    if 'title' in node and node['title'] and not is_root:
        title = node['title']
        # 构建完整路径
        full_path = f"{current_path} - {title}" if current_path else title
        
        # 添加paraphrase
        if full_path in paraphrases_map:
            node['paraphrases'] = paraphrases_map[full_path]
        else:
            node['paraphrases'] = {}
        
        current_path = full_path
    
    # 递归处理子节点
    if 'children' in node and node['children']:
        for child in node['children']:
            add_paraphrases_to_tree(child, paraphrases_map, current_path, is_root=False)


def main():
    parser = argparse.ArgumentParser(description='生成标题的Paraphrase')
    parser.add_argument(
        '--structures_file',
        type=str,
        default='/mnt/literism/tree/data/wikipedia_structures_final.json',
        help='结构文件路径'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./data/paraphrases',
        help='输出目录'
    )
    parser.add_argument(
        '--api_key',
        type=str,
        default='sk-3f2e7fe4ae6e4d588c619bbff9837dac',
        help='DeepSeek API key'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='deepseek-chat',
        help='模型名称'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='生成温度'
    )
    parser.add_argument(
        '--max_output_tokens',
        type=int,
        default=512,
        help='最大输出token数'
    )
    parser.add_argument(
        '--max_concurrent_jobs',
        type=int,
        default=8,
        help='最大并发数'
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 收集所有任务
    structures, all_tasks = collect_all_tasks(args.structures_file)
    
    # 创建API客户端
    config = DeepSeekConfig(
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        max_concurrent_jobs=args.max_concurrent_jobs,
        save_path=None
    )
    api_client = DeepSeekAPIClient(config)
    
    # 生成所有paraphrases
    print("\n" + "=" * 80)
    print("生成Paraphrase...")
    print("=" * 80)
    print(f"\n总计需要处理 {len(all_tasks)} 个标题")
    print("正在批量生成...\n")
    
    # 创建所有prompts
    prompts = [create_paraphrase_prompt(title, full_path) for _, full_path, title in all_tasks]
    
    # 一次性调用API
    responses = api_client.run_prompts_to_texts(prompts)
    
    # 解析所有结果并按topic组织
    print("\n解析结果...")
    
    # 为每个topic创建 {full_path: paraphrases} 映射
    paraphrases_by_topic = defaultdict(dict)
    success_count = 0
    
    for (topic_key, full_path, title), response in zip(all_tasks, responses):
        paraphrases = parse_paraphrase_result(response)
        
        if paraphrases:
            paraphrases_by_topic[topic_key][full_path] = paraphrases
            success_count += 1
        else:
            # 保存空的paraphrase
            paraphrases_by_topic[topic_key][full_path] = {}
    
    print(f"  - 成功生成 {success_count}/{len(all_tasks)} 个标题的paraphrase")
    print(f"  - 成功率: {success_count/len(all_tasks)*100:.1f}%")
    
    # 将paraphrase添加到树结构中并保存
    print("\n" + "=" * 80)
    print("构建树结构并保存...")
    print("=" * 80)
    
    for topic_key, topic_data in structures.items():
        # 深拷贝原始结构
        new_topic_data = copy.deepcopy(topic_data)
        
        # 获取这个topic的paraphrase映射
        paraphrases_map = paraphrases_by_topic.get(topic_key, {})
        topic_name = topic_data.get('topic', '')
        
        # 将paraphrase添加到树中
        if 'structure' in new_topic_data and new_topic_data['structure']:
            if isinstance(new_topic_data['structure'], list):
                for child in new_topic_data['structure']:
                    add_paraphrases_to_tree(child, paraphrases_map, topic_name, is_root=False)
            else:
                add_paraphrases_to_tree(new_topic_data['structure'], paraphrases_map, "", is_root=True)
        
        # 保存结果
        output_file = output_dir / f"{topic_key.replace(':', '_')}_paraphrases.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(new_topic_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n  - 已为 {len(structures)} 个topics保存结果到: {output_dir}")
    
    print("\n" + "=" * 80)
    print("Paraphrase生成完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()
