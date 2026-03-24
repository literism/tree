#!/usr/bin/env python3
"""
浏览结构树，帮助找到要测试的节点路径
"""
import json
import sys
from pathlib import Path


def browse_structure(structures_file: str, topic_key: str):
    """浏览某个topic的结构树"""
    with open(structures_file, 'r', encoding='utf-8') as f:
        structures = json.load(f)
    
    if topic_key not in structures:
        print(f"错误: topic '{topic_key}' 不存在")
        print(f"\n可用的topics (前10个):")
        for i, key in enumerate(list(structures.keys())[:10], 1):
            print(f"  {i}. {key}")
        return
    
    structure = structures[topic_key]
    topic_name = structure.get('topic', topic_key)
    
    print("="*80)
    print(f"Topic: {topic_name}")
    print(f"Key: {topic_key}")
    print("="*80)
    
    def print_node(node, path, indent=0):
        """递归打印节点"""
        prefix = "  " * indent
        title = node['title']
        level = node.get('level', '?')
        has_children = len(node.get('children', []))
        has_content = bool(node.get('content', ''))
        
        print(f"{prefix}├─ {title} (level {level})")
        print(f"{prefix}│  路径: {path} - {title}")
        print(f"{prefix}│  子节点: {has_children}")
        print(f"{prefix}│  有内容: {'是' if has_content else '否'}")
        
        # 递归打印子节点（最多3层）
        if indent < 2 and node.get('children'):
            for child in node['children'][:3]:  # 只显示前3个子节点
                print_node(child, f"{path} - {title}", indent + 1)
            if len(node['children']) > 3:
                print(f"{prefix}  │  ... 还有 {len(node['children']) - 3} 个子节点")
    
    print("\n结构树:")
    for i, root_node in enumerate(structure.get('structure', []), 1):
        print(f"\n第{i}个根节点:")
        print_node(root_node, topic_name)
    
    print("\n" + "="*80)
    print("使用示例:")
    print(f"  python3 generate_summaries.py --test \\")
    print(f"    --test_topic '{topic_key}' \\")
    
    # 给出一些示例路径
    if structure.get('structure'):
        first_root = structure['structure'][0]
        print(f"    --test_path '{topic_name} - {first_root['title']}'")
        
        if first_root.get('children'):
            first_child = first_root['children'][0]
            print(f"  # 或测试第二层:")
            print(f"    --test_path '{topic_name} - {first_root['title']} - {first_child['title']}'")
    
    print("="*80)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python3 browse_structure.py <topic_key>")
        print("\n示例:")
        print("  python3 browse_structure.py 'Book:The Hobbit'")
        print("  python3 browse_structure.py 'Category:Anarchism'")
        sys.exit(1)
    
    structures_file = '/mnt/literism/tree/data/wikipedia_structures_final.json'
    topic_key = sys.argv[1]
    
    browse_structure(structures_file, topic_key)

