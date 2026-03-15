"""
准备结构树生成的训练数据集
从wiki_structure.jsonl和wiki_intro.jsonl中提取数据
"""
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import random

from config import StructureGeneratorConfig, SKIP_TITLES


class StructureDatasetPreparator:
    """结构树数据集准备器"""
    
    def __init__(self, config: StructureGeneratorConfig):
        self.config = config
        
        # 创建输出目录
        Path(config.data_dir).mkdir(parents=True, exist_ok=True)
        
        # 加载已有的topic划分
        print("加载已有的topic划分...")
        with open(config.dataset_split_file, 'r', encoding='utf-8') as f:
            split_data = json.load(f)
        
        self.train_topics = set(split_data.get('train_topics', split_data.get('train', [])))
        self.test_easy_topics = set(split_data.get('test_easy_topics', split_data.get('test_easy', [])))
        self.test_hard_topics = set(split_data.get('test_hard_topics', split_data.get('test_hard', [])))
        self.excluded_topics = self.train_topics | self.test_easy_topics | self.test_hard_topics
        
        print(f"  - 排除的train topics: {len(self.train_topics)}")
        print(f"  - 排除的test_easy topics: {len(self.test_easy_topics)}")
        print(f"  - 排除的test_hard topics: {len(self.test_hard_topics)}")
        print(f"  - 总共排除: {len(self.excluded_topics)} 个topics")
    
    def clean_structure(self, sections: List[Dict]) -> List[Dict]:
        """
        清理结构树，删除无用的title
        
        Args:
            sections: 原始的sections列表
            
        Returns:
            清理后的sections列表
        """
        def clean_section(section: Dict) -> Optional[Dict]:
            """递归清理section"""
            title_lower = section['title'].lower()
            
            # 如果是无用title，跳过
            if title_lower in SKIP_TITLES:
                return None
            
            # 递归清理子节点
            cleaned_children = []
            for child in section.get('children', []):
                cleaned_child = clean_section(child)
                if cleaned_child is not None:
                    cleaned_children.append(cleaned_child)
            
            # 返回清理后的section
            return {
                'title': section['title'],
                'level': section['level'],
                'children': cleaned_children
            }
        
        cleaned = []
        for section in sections:
            cleaned_section = clean_section(section)
            if cleaned_section is not None:
                cleaned.append(cleaned_section)
        
        return cleaned
    
    def count_nodes(self, sections: List[Dict]) -> int:
        """
        统计结构树中的节点总数
        
        Args:
            sections: sections列表
            
        Returns:
            节点总数
        """
        def count_section(section: Dict) -> int:
            count = 1  # 当前节点
            for child in section.get('children', []):
                count += count_section(child)
            return count
        
        total = 0
        for section in sections:
            total += count_section(section)
        return total
    
    def structure_to_text(self, sections: List[Dict]) -> str:
        """
        将结构树转换为文本格式（用于模型输出）
        
        格式：
        - Title1 (level 2)
          - Subtitle1 (level 3)
          - Subtitle2 (level 3)
        - Title2 (level 2)
        
        Args:
            sections: sections列表
            
        Returns:
            文本格式的结构树
        """
        def section_to_text(section: Dict, indent: int = 0) -> str:
            lines = []
            prefix = "  " * indent + "- "
            lines.append(f"{prefix}{section['title']} (level {section['level']})")
            
            for child in section.get('children', []):
                lines.append(section_to_text(child, indent + 1))
            
            return "\n".join(lines)
        
        return "\n".join(section_to_text(section) for section in sections)
    
    def create_prompt(self, topic: str, intro: str) -> str:
        """
        创建训练prompt
        
        Args:
            topic: topic标题
            intro: topic的介绍文本
            
        Returns:
            prompt字符串
        """
        prompt = f"""TASK: Generate a hierarchical structure tree for the given topic based on its introduction.

TOPIC: {topic}

INTRODUCTION:
{intro}

INSTRUCTIONS:
1. Analyze the introduction and identify the main aspects, themes, or categories related to this topic.
2. Create a multi-level hierarchical structure that organizes these aspects logically.
3. Each node should have a clear title.
4. Use appropriate level numbers (level 2 for main titles, level 3 for subtitles, etc.).
5. The structure should be comprehensive but not overly detailed.
6. Format each line as: "- Title (level N)" where N is the level number.
7. Use indentation (2 spaces per level) to show hierarchy.

OUTPUT FORMAT:
- Main Title 1 (level 2)
  - Subtitle 1.1 (level 3)
  - Subtitle 1.2 (level 3)
- Main Title 2 (level 2)
  - Subtitle 2.1 (level 3)
    - Sub-subtitle 2.1.1 (level 4)

STRUCTURE:
"""
        return prompt
    
    def load_and_filter_data(self) -> Tuple[List[Dict], Dict[str, str]]:
        """
        加载并筛选数据
        
        Returns:
            (符合条件的数据列表, 排除topics的intro字典)
        """
        print("\n加载wiki数据...")
        
        # 加载结构数据
        structures = {}
        with open(self.config.wiki_structure_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                structures[data['title']] = data
        print(f"  - 加载了 {len(structures)} 个结构")
        
        # 加载intro数据
        intros = {}
        with open(self.config.wiki_intro_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                intros[data['title']] = data['intro']
        print(f"  - 加载了 {len(intros)} 个intro")
        
        # 筛选数据
        print("\n筛选符合条件的数据...")
        valid_data = []
        excluded_intros = {}
        
        for title, structure_data in structures.items():
            # 检查是否有intro
            if title not in intros:
                continue
            
            intro = intros[title]
            
            # 清理结构树
            cleaned_sections = self.clean_structure(structure_data['sections'])
            node_count = self.count_nodes(cleaned_sections)
            
            # 检查intro长度
            intro_length = len(intro)
            
            # 如果是排除的topic，只保存intro
            if title in self.excluded_topics:
                excluded_intros[title] = intro
                continue
            
            # 检查是否满足阈值
            if node_count < self.config.min_structure_nodes:
                continue
            if intro_length < self.config.min_intro_length:
                continue
            
            # 添加到有效数据
            valid_data.append({
                'title': title,
                'url': structure_data['url'],
                'intro': intro,
                'sections': cleaned_sections,
                'node_count': node_count,
                'intro_length': intro_length
            })
        
        print(f"  - 符合条件的数据: {len(valid_data)}")
        print(f"  - 排除topics的intro: {len(excluded_intros)}")
        
        return valid_data, excluded_intros
    
    def format_for_training(self, data: Dict) -> Dict:
        """
        将数据格式化为训练格式（与hierarchical_classifier保持一致）
        使用 prompt 和 completion 字段
        
        Args:
            data: 原始数据
            
        Returns:
            格式化后的数据，包含 prompt 和 completion 字段
        """
        # 创建prompt
        prompt = self.create_prompt(data['title'], data['intro'])
        
        # 创建completion（结构树文本）
        completion = self.structure_to_text(data['sections'])
        
        return {
            'prompt': prompt,
            'completion': completion
        }
    
    def prepare_dataset(self):
        """准备完整的数据集"""
        # 加载并筛选数据
        valid_data, excluded_intros = self.load_and_filter_data()
        
        # 保存排除topics的intro
        excluded_intro_file = Path(self.config.data_dir) / 'excluded_topics_intro.json'
        with open(excluded_intro_file, 'w', encoding='utf-8') as f:
            json.dump(excluded_intros, f, indent=2, ensure_ascii=False)
        print(f"\n排除topics的intro保存到: {excluded_intro_file}")
        
        # 检查数据量
        if len(valid_data) < self.config.train_size:
            print(f"\n警告: 有效数据量 ({len(valid_data)}) 少于所需训练数据量 ({self.config.train_size})")
            print(f"将使用全部 {len(valid_data)} 条数据")
            sampled_data = valid_data
        else:
            # 随机采样
            print(f"\n从 {len(valid_data)} 条数据中随机采样 {self.config.train_size} 条...")
            random.shuffle(valid_data)
            sampled_data = valid_data[:self.config.train_size]
        
        # 生成训练数据
        print("\n生成训练数据...")
        train_val_data = []
        metadata_list = []  # 保存元数据
        
        for i, data in enumerate(sampled_data):
            if (i + 1) % 1000 == 0:
                print(f"  处理进度: {i+1}/{len(sampled_data)}")
            
            # 格式化为训练格式（prompt + completion）
            formatted = self.format_for_training(data)
            train_val_data.append(formatted)
            
            # 保存元数据
            metadata_list.append({
                'topic': data['title'],
                'node_count': data['node_count'],
                'intro_length': data['intro_length']
            })
        
        # 划分训练集和验证集
        val_size = int(len(train_val_data) * self.config.val_ratio)
        train_size = len(train_val_data) - val_size
        
        # 同时打乱数据和元数据
        combined = list(zip(train_val_data, metadata_list))
        random.shuffle(combined)
        train_val_data, metadata_list = zip(*combined)
        train_val_data = list(train_val_data)
        metadata_list = list(metadata_list)
        
        train_data = train_val_data[:train_size]
        val_data = train_val_data[train_size:]
        train_metadata = metadata_list[:train_size]
        val_metadata = metadata_list[train_size:]
        
        print(f"\n数据集划分:")
        print(f"  - 训练集: {len(train_data)}")
        print(f"  - 验证集: {len(val_data)}")
        
        # 保存数据集（JSON格式，与hierarchical_classifier一致）
        train_file = Path(self.config.data_dir) / 'train_dataset.json'
        val_file = Path(self.config.data_dir) / 'val_dataset.json'
        
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)
        
        with open(val_file, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n训练数据保存到: {train_file}")
        print(f"验证数据保存到: {val_file}")
        
        # 保存统计信息
        stats = {
            'total_valid_data': len(valid_data),
            'sampled_data': len(sampled_data),
            'train_size': len(train_data),
            'val_size': len(val_data),
            'excluded_topics': len(excluded_intros),
            'avg_node_count': sum(d['node_count'] for d in metadata_list) / len(metadata_list),
            'avg_intro_length': sum(d['intro_length'] for d in metadata_list) / len(metadata_list),
        }
        
        stats_file = Path(self.config.data_dir) / 'dataset_stats.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"\n统计信息保存到: {stats_file}")
        print("\n数据集准备完成！")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='准备结构树生成训练数据集')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--min_structure_nodes', type=int, help='最少节点数')
    parser.add_argument('--min_intro_length', type=int, help='最少intro长度')
    parser.add_argument('--train_size', type=int, help='训练数据大小')
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config:
        config = StructureGeneratorConfig.from_json(args.config)
    else:
        config = StructureGeneratorConfig()
    
    # 命令行参数覆盖
    if args.min_structure_nodes:
        config.min_structure_nodes = args.min_structure_nodes
    if args.min_intro_length:
        config.min_intro_length = args.min_intro_length
    if args.train_size:
        config.train_size = args.train_size
    
    # 打印配置
    config.print_config()
    
    # 准备数据集
    preparator = StructureDatasetPreparator(config)
    preparator.prepare_dataset()


if __name__ == '__main__':
    main()

