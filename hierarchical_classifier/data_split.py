"""
数据集划分脚本
根据category和reference数量划分训练集、简单测试集和困难测试集
"""
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


class DatasetSplitter:
    def __init__(
        self, 
        references_file: str,
        topic_classified_file: str,
        output_dir: str,
        test_easy_ratio: float = 0.1,
        seed: int = 42
    ):
        """
        Args:
            references_file: wikipedia_references_final.json文件路径
            topic_classified_file: topic_classified.json文件路径
            output_dir: 输出目录
            test_easy_ratio: 简单测试集比例（从训练topics的文章中划分）
            seed: 随机种子
        """
        self.references_file = references_file
        self.topic_classified_file = topic_classified_file
        self.output_dir = Path(output_dir)
        self.test_easy_ratio = test_easy_ratio
        self.seed = seed
        
        random.seed(seed)
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self) -> Tuple[Dict, Dict]:
        """加载数据文件"""
        print("加载数据文件...")
        
        # 加载references
        with open(self.references_file, 'r', encoding='utf-8') as f:
            references_data = json.load(f)
        print(f"  - 加载 {len(references_data)} 个topics")
        
        # 加载topic分类
        with open(self.topic_classified_file, 'r', encoding='utf-8') as f:
            topic_classified = json.load(f)
        print(f"  - 加载 {len(topic_classified)} 个categories")
        
        return references_data, topic_classified
    
    def select_test_hard_topics(
        self, 
        references_data: Dict, 
        topic_classified: Dict
    ) -> Tuple[List[str], List[str]]:
        """
        从每个category中选择reference最少的topic作为困难测试topic
        
        Returns:
            test_hard_topics: 困难测试topic列表（格式如"Category:TopicName"）
            train_topics: 训练topic列表
        """
        print("\n选择困难测试topics...")
        
        test_hard_topics = []
        train_topics = []
        
        for category, topics in topic_classified.items():
            print(f"\n处理类别: {category}")
            print(f"  - 该类别有 {len(topics)} 个topics")
            
            # 统计每个topic的reference数量
            topic_ref_counts = []
            for topic in topics:
                # 构造完整的topic key
                topic_key = f"{category}:{topic}"
                
                if topic_key in references_data:
                    ref_count = len(references_data[topic_key].get('references', {}))
                    topic_ref_counts.append((topic_key, ref_count))
                else:
                    print(f"  警告: {topic_key} 不在references数据中")
            
            if not topic_ref_counts:
                print(f"  警告: 类别 {category} 没有有效的topics")
                continue
            
            # 找到reference最少的topic作为困难测试topic
            topic_ref_counts.sort(key=lambda x: (x[1] - 250) ** 2)
            # random.shuffle(topic_ref_counts)
            test_hard_topic = topic_ref_counts[0]
            test_hard_topics.append(test_hard_topic[0])
            
            print(f"  - 困难测试topic: {test_hard_topic[0]} (有 {test_hard_topic[1]} 个references)")
            
            # 其他topics是训练topics
            for topic_key, ref_count in topic_ref_counts[1:]:
                train_topics.append(topic_key)
                
        print(f"\n总计:")
        print(f"  - 困难测试topics: {len(test_hard_topics)}")
        print(f"  - 训练topics: {len(train_topics)}")
        
        return test_hard_topics, train_topics
    
    def split_dataset(
        self,
        references_data: Dict,
        test_hard_topics: List[str],
        train_topics: List[str]
    ) -> Dict:
        """
        划分数据集
        
        Returns:
            dataset_split: 包含train/test_easy/test_hard三个部分的字典
        """
        print("\n划分数据集...")
        
        dataset_split = {
            'train': {},  # topic -> list of reference ids
            'test_easy': {},
            'test_hard': {}
        }
        
        # 处理困难测试topics - 所有references都进入困难测试集
        print("\n处理困难测试集...")
        for topic_key in test_hard_topics:
            if topic_key not in references_data:
                continue
            references = references_data[topic_key].get('references', {})
            ref_ids = list(references.keys())
            dataset_split['test_hard'][topic_key] = ref_ids
            print(f"  - {topic_key}: {len(ref_ids)} references")
        
        # 处理训练topics - 一部分进入简单测试集，其余进入训练集
        print("\n处理训练集和简单测试集...")
        for topic_key in train_topics:
            if topic_key not in references_data:
                continue
            
            references = references_data[topic_key].get('references', {})
            ref_ids = list(references.keys())
            
            # 打乱顺序
            random.shuffle(ref_ids)
            
            # 划分简单测试集和训练集
            test_easy_size = max(0, int(len(ref_ids) * self.test_easy_ratio))  # 至少1个
            test_easy_refs = ref_ids[:test_easy_size]
            train_refs = ref_ids[test_easy_size:]
            
            dataset_split['test_easy'][topic_key] = test_easy_refs
            dataset_split['train'][topic_key] = train_refs
            
            print(f"  - {topic_key}: {len(train_refs)} train, {len(test_easy_refs)} test_easy")
        
        # 统计
        total_train = sum(len(refs) for refs in dataset_split['train'].values())
        total_test_easy = sum(len(refs) for refs in dataset_split['test_easy'].values())
        total_test_hard = sum(len(refs) for refs in dataset_split['test_hard'].values())
        
        print(f"\n数据集统计:")
        print(f"  - 训练集: {total_train} references from {len(dataset_split['train'])} topics")
        print(f"  - 简单测试集: {total_test_easy} references from {len(dataset_split['test_easy'])} topics")
        print(f"  - 困难测试集: {total_test_hard} references from {len(dataset_split['test_hard'])} topics")
        
        return dataset_split
    
    def save_split(self, dataset_split: Dict, test_hard_topics: List[str], train_topics: List[str]):
        """保存划分结果"""
        print("\n保存划分结果...")
        
        # 保存详细的划分信息
        split_info = {
            'test_hard_topics': test_hard_topics,
            'train_topics': train_topics,
            'dataset_split': dataset_split,
            'seed': self.seed,
            'test_easy_ratio': self.test_easy_ratio
        }
        
        output_file = self.output_dir / 'dataset_split.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(split_info, f, indent=2, ensure_ascii=False)
        
        print(f"  - 保存到: {output_file}")
        
        # 保存统计信息
        stats = {
            'total_topics': len(test_hard_topics) + len(train_topics),
            'test_hard_topics_count': len(test_hard_topics),
            'train_topics_count': len(train_topics),
            'train_references': sum(len(refs) for refs in dataset_split['train'].values()),
            'test_easy_references': sum(len(refs) for refs in dataset_split['test_easy'].values()),
            'test_hard_references': sum(len(refs) for refs in dataset_split['test_hard'].values()),
        }
        
        stats_file = self.output_dir / 'split_stats.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"  - 统计信息保存到: {stats_file}")
    
    def run(self):
        """执行数据集划分"""
        print("=" * 80)
        print("数据集划分")
        print("=" * 80)
        
        # 1. 加载数据
        references_data, topic_classified = self.load_data()
        
        # 2. 选择困难测试topics
        test_hard_topics, train_topics = self.select_test_hard_topics(
            references_data, topic_classified
        )
        
        # 3. 划分数据集
        dataset_split = self.split_dataset(
            references_data, test_hard_topics, train_topics
        )
        
        # 4. 保存结果
        self.save_split(dataset_split, test_hard_topics, train_topics)
        
        print("\n" + "=" * 80)
        print("数据集划分完成！")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='划分数据集')
    parser.add_argument(
        '--references_file',
        type=str,
        default='/mnt/literism/tree/data/wikipedia_references_final.json',
        help='references文件路径'
    )
    parser.add_argument(
        '--topic_classified_file',
        type=str,
        default='/mnt/literism/data/result/topic_classified.json',
        help='topic分类文件路径'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./data',
        help='输出目录'
    )
    parser.add_argument(
        '--test_easy_ratio',
        type=float,
        default=0.1,
        help='简单测试集比例（从训练topics的文章中划分）'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子'
    )
    
    args = parser.parse_args()
    
    splitter = DatasetSplitter(
        references_file=args.references_file,
        topic_classified_file=args.topic_classified_file,
        output_dir=args.output_dir,
        test_easy_ratio=args.test_easy_ratio,
        seed=args.seed
    )
    
    splitter.run()


if __name__ == '__main__':
    main()

