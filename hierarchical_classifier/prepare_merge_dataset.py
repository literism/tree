"""
准备合并系统的训练数据集
从structures_file读取节点，生成正负样本对
"""
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from tqdm import tqdm


class MergeDatasetPreparator:
    """合并系统数据集准备器"""
    
    def __init__(
        self,
        structures_file: str,
        output_dir: str,
        val_ratio: float = 0.1,
        seed: int = 42
    ):
        """
        Args:
            structures_file: 结构文件路径
            output_dir: 输出目录
            val_ratio: 验证集比例
            seed: 随机种子
        """
        self.structures_file = structures_file
        self.output_dir = Path(output_dir)
        self.val_ratio = val_ratio
        self.seed = seed
        
        random.seed(seed)
        
        self.structures = {}
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_structures(self):
        """加载结构数据"""
        print("="*80)
        print("加载结构数据...")
        print("="*80)
        
        with open(self.structures_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.structures = data
        print(f"  - 加载 {len(self.structures)} 个topics的结构")
    
    def get_all_nodes_with_depth(self, structure: List[Dict], current_depth: int = 1, parent_path: str = None) -> List[Tuple[Dict, int, str]]:
        """
        递归获取所有节点及其深度和父路径
        
        Args:
            structure: 结构列表
            current_depth: 当前深度（从1开始）
            parent_path: 父节点路径（用于区分不同父节点下的节点）
            
        Returns:
            [(节点, 深度, 父路径), ...]
        """
        nodes = []
        
        for node in structure:
            node_path = f"{parent_path}/{node.get('title', 'untitled')}" if parent_path else node.get('title', 'untitled')
            nodes.append((node, current_depth, parent_path))
            
            # 递归处理子节点
            if 'children' in node and node['children']:
                children_nodes = self.get_all_nodes_with_depth(
                    node['children'],
                    current_depth + 1,
                    node_path
                )
                nodes.extend(children_nodes)
        
        return nodes
    
    def generate_pairs_for_topic(self, topic_key: str) -> Tuple[List[Dict], List[Dict]]:
        """
        为一个topic生成正负样本对
        
        规则：
        1. 第一层的两个节点对构成负样本
        2. 不同层之间的节点对构成负样本
        3. 第二层及以后，同一层但不同父节点的节点对，构成负样本
        4. 第二层及以后，同一层且相同父节点的节点对，构成正样本
        
        Args:
            topic_key: topic键
            
        Returns:
            (正样本列表, 负样本列表)
        """
        if topic_key not in self.structures:
            return [], []
        
        topic_data = self.structures[topic_key]
        topic_name = topic_data.get('topic', topic_key)
        structure = topic_data.get('structure', [])
        
        if not structure:
            return [], []
        
        # 获取所有节点及其深度和父路径
        all_nodes = self.get_all_nodes_with_depth(structure)
        
        # 按深度分组
        nodes_by_depth = defaultdict(list)
        for node, depth, parent_path in all_nodes:
            nodes_by_depth[depth].append((node, parent_path))
        
        positive_pairs = []
        negative_pairs = []
        
        # 规则1：第一层的两个节点对构成负样本
        first_layer_nodes = nodes_by_depth.get(1, [])
        for i in range(len(first_layer_nodes)):
            for j in range(i + 1, len(first_layer_nodes)):
                node1, _ = first_layer_nodes[i]
                node2, _ = first_layer_nodes[j]
                
                if 'summary' in node1 and 'summary' in node2:
                    negative_pairs.append({
                        'topic': topic_name,
                        'summary1': node1['summary'],
                        'summary2': node2['summary']
                    })
        
        # 规则2：不同层之间的节点对构成负样本
        depths = sorted(nodes_by_depth.keys())
        for i, depth1 in enumerate(depths):
            for depth2 in depths[i+1:]:
                nodes1 = nodes_by_depth[depth1]
                nodes2 = nodes_by_depth[depth2]
                
                # 随机采样，避免数量过多
                max_pairs = 50  # 每对深度最多采样50对
                sampled_pairs = random.sample(
                    [(n1, n2) for n1 in nodes1 for n2 in nodes2],
                    min(max_pairs, len(nodes1) * len(nodes2))
                )
                
                for (node1, _), (node2, _) in sampled_pairs:
                    if 'summary' in node1 and 'summary' in node2:
                        negative_pairs.append({
                            'topic': topic_name,
                            'summary1': node1['summary'],
                            'summary2': node2['summary']
                        })
        
        # 规则3和4：第二层及以后的节点
        for depth in depths:
            if depth < 2:
                continue
            
            nodes_at_depth = nodes_by_depth[depth]
            
            # 按父路径分组
            nodes_by_parent = defaultdict(list)
            for node, parent_path in nodes_at_depth:
                nodes_by_parent[parent_path].append(node)
            
            # 规则4：相同父节点的节点对 -> 正样本
            for parent_path, nodes in nodes_by_parent.items():
                if len(nodes) < 2:
                    continue
                
                for i in range(len(nodes)):
                    for j in range(i + 1, len(nodes)):
                        if 'summary' in nodes[i] and 'summary' in nodes[j]:
                            positive_pairs.append({
                                'topic': topic_name,
                                'summary1': nodes[i]['summary'],
                                'summary2': nodes[j]['summary']
                            })
            
            # 规则3：不同父节点的节点对 -> 负样本
            parent_paths = list(nodes_by_parent.keys())
            for i in range(len(parent_paths)):
                for j in range(i + 1, len(parent_paths)):
                    nodes1 = nodes_by_parent[parent_paths[i]]
                    nodes2 = nodes_by_parent[parent_paths[j]]
                    
                    # 随机采样
                    max_pairs = 20
                    sampled_pairs = random.sample(
                        [(n1, n2) for n1 in nodes1 for n2 in nodes2],
                        min(max_pairs, len(nodes1) * len(nodes2))
                    )
                    
                    for node1, node2 in sampled_pairs:
                        if 'summary' in node1 and 'summary' in node2:
                            negative_pairs.append({
                                'topic': topic_name,
                                'summary1': node1['summary'],
                                'summary2': node2['summary']
                            })
        
        return positive_pairs, negative_pairs
    
    def generate_all_pairs(self) -> Tuple[List[Dict], List[Dict]]:
        """为所有topics生成正负样本对"""
        print("\n" + "="*80)
        print("生成正负样本对...")
        print("="*80)
        
        all_positive = []
        all_negative = []
        
        for topic_key in tqdm(self.structures.keys(), desc="处理topics"):
            pos, neg = self.generate_pairs_for_topic(topic_key)
            all_positive.extend(pos)
            all_negative.extend(neg)
        
        print(f"\n生成完成:")
        print(f"  - 正样本: {len(all_positive)}")
        print(f"  - 负样本: {len(all_negative)}")
        
        return all_positive, all_negative
    
    def balance_and_split(self, positive_pairs: List[Dict], negative_pairs: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        平衡正负样本并划分训练集和验证集
        
        Args:
            positive_pairs: 正样本列表
            negative_pairs: 负样本列表
            
        Returns:
            (训练集, 验证集)
        """
        print("\n" + "="*80)
        print("平衡样本并划分数据集...")
        print("="*80)
        
        # 平衡：使用较少的一方的数量
        min_count = min(len(positive_pairs), len(negative_pairs))
        
        print(f"\n平衡前:")
        print(f"  - 正样本: {len(positive_pairs)}")
        print(f"  - 负样本: {len(negative_pairs)}")
        print(f"  - 平衡后每类: {min_count}")
        
        # 随机采样
        positive_sampled = random.sample(positive_pairs, min_count)
        negative_sampled = random.sample(negative_pairs, min_count)
        
        # 合并并打乱
        all_data = positive_sampled + negative_sampled
        random.shuffle(all_data)
        
        # 划分训练集和验证集
        val_size = int(len(all_data) * self.val_ratio)
        val_data = all_data[:val_size]
        train_data = all_data[val_size:]
        
        print(f"\n最终数据集:")
        print(f"  - 训练集: {len(train_data)}")
        print(f"  - 验证集: {len(val_data)}")
        
        # 统计训练集和验证集中的正负样本
        train_pos = sum(1 for d in train_data if d in positive_sampled)
        val_pos = sum(1 for d in val_data if d in positive_sampled)
        
        print(f"\n训练集正负样本:")
        print(f"  - 正样本: {train_pos}")
        print(f"  - 负样本: {len(train_data) - train_pos}")
        print(f"\n验证集正负样本:")
        print(f"  - 正样本: {val_pos}")
        print(f"  - 负样本: {len(val_data) - val_pos}")
        
        return train_data, val_data
    
    def format_for_training(self, data_sample: Dict) -> Dict:
        """
        将数据样本格式化为训练格式
        
        Args:
            data_sample: 包含topic, summary1, summary2的字典
            
        Returns:
            包含prompt和completion的字典
        """
        # 判断是正样本还是负样本（通过是否在原始列表中判断，这里简化处理）
        # 实际上我们在generate_all_pairs返回时已经标记了，这里需要修改
        # 暂时通过summary内容的相似度启发式判断（实际应该在生成时就标记）
        
        # 构建prompt
        prompt = f"""You are tasked with determining whether two categories from a Wikipedia topic should be merged into one.

TOPIC: {data_sample['topic']}

CATEGORY 1 SUMMARY:
{data_sample['summary1']}

CATEGORY 2 SUMMARY:
{data_sample['summary2']}

TASK:
Determine if these two categories are similar enough that they should be merged into a single category. Consider:
1. Do they cover similar or overlapping content?
2. Would it make sense to combine them for better organization?
3. Are they at the same conceptual level?

Answer with ONLY "Yes" or "No".

ANSWER:"""
        
        # completion（需要从data_sample中获取label）
        # 注意：我们需要在生成数据时就标记label
        label = data_sample.get('label', 'No')  # 默认No
        
        return {
            'prompt': prompt,
            'completion': label
        }
    
    def save_dataset(self, train_data: List[Dict], val_data: List[Dict]):
        """保存数据集"""
        print("\n" + "="*80)
        print("保存数据集...")
        print("="*80)
        
        # 格式化数据
        train_formatted = [self.format_for_training(d) for d in train_data]
        val_formatted = [self.format_for_training(d) for d in val_data]
        
        # 保存
        train_file = self.output_dir / 'train_dataset.json'
        val_file = self.output_dir / 'val_dataset.json'
        
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_formatted, f, ensure_ascii=False, indent=2)
        
        with open(val_file, 'w', encoding='utf-8') as f:
            json.dump(val_formatted, f, ensure_ascii=False, indent=2)
        
        print(f"  - 训练集保存到: {train_file}")
        print(f"  - 验证集保存到: {val_file}")
    
    def run(self):
        """运行数据准备流程"""
        self.load_structures()
        
        positive_pairs, negative_pairs = self.generate_all_pairs()
        
        # 标记label
        for pair in positive_pairs:
            pair['label'] = 'Yes'
        for pair in negative_pairs:
            pair['label'] = 'No'
        
        train_data, val_data = self.balance_and_split(positive_pairs, negative_pairs)
        
        self.save_dataset(train_data, val_data)
        
        print("\n" + "="*80)
        print("数据准备完成！")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description='准备合并系统训练数据')
    
    parser.add_argument(
        '--structures_file',
        type=str,
        required=True,
        help='结构文件路径'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='输出目录'
    )
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.1,
        help='验证集比例'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子'
    )
    
    args = parser.parse_args()
    
    preparator = MergeDatasetPreparator(
        structures_file=args.structures_file,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed
    )
    
    preparator.run()


if __name__ == '__main__':
    main()

