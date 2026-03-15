"""
数据集划分脚本
根据category和reference数量划分训练集和测试集
"""
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
from summary_based_classifier.config import SummaryBasedConfig


class DatasetSplitter:
    def __init__(
        self, 
        references_file: str,
        topic_classified_file: str,
        output_dir: str,
        target_test_size: int = 250,
        seed: int = 42
    ):
        """
        Args:
            references_file: wikipedia_references_final.json文件路径
            topic_classified_file: topic_classified.json文件路径
            output_dir: 输出目录
            target_test_size: 选择文章数最接近此值的topic作为test
            seed: 随机种子
        """
        self.references_file = references_file
        self.topic_classified_file = topic_classified_file
        self.output_dir = Path(output_dir)
        self.target_test_size = target_test_size
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
    
    def select_test_topics(
        self, 
        references_data: Dict, 
        topic_classified: Dict
    ) -> Tuple[List[str], List[str]]:
        """
        从每个category中选择文章数最接近target_test_size的topic作为测试topic
        
        Returns:
            test_topics: 测试topic列表（格式如"Category:TopicName"）
            train_topics: 训练topic列表
        """
        print(f"\n选择测试topics（目标文章数: {self.target_test_size}）...")
        
        test_topics = []
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
            
            # 找到reference数量最接近target_test_size的topic作为测试topic
            topic_ref_counts.sort(key=lambda x: (x[1] - self.target_test_size) ** 2)
            test_topic = topic_ref_counts[0]
            test_topics.append(test_topic[0])
            
            print(f"  - 测试topic: {test_topic[0]} (有 {test_topic[1]} 个references)")
            
            # 其他topics是训练topics
            for topic_key, ref_count in topic_ref_counts[1:]:
                train_topics.append(topic_key)
                
        print(f"\n总计:")
        print(f"  - 测试topics: {len(test_topics)}")
        print(f"  - 训练topics: {len(train_topics)}")
        
        return test_topics, train_topics
    
    def split_dataset(
        self,
        references_data: Dict,
        test_topics: List[str],
        train_topics: List[str]
    ) -> Dict:
        """
        划分数据集
        
        Returns:
            dataset_split: 包含train/test两个部分的字典
        """
        print("\n划分数据集...")
        
        dataset_split = {
            'train': {},  # topic -> list of reference ids
            'test': {}
        }
        
        # 处理测试topics - 所有references都进入测试集
        print("\n处理测试集...")
        for topic_key in test_topics:
            if topic_key not in references_data:
                continue
            references = references_data[topic_key].get('references', {})
            ref_ids = list(references.keys())
            dataset_split['test'][topic_key] = ref_ids
            print(f"  - {topic_key}: {len(ref_ids)} references")
        
        # 处理训练topics - 所有references都进入训练集
        print("\n处理训练集...")
        for topic_key in train_topics:
            if topic_key not in references_data:
                continue
            
            references = references_data[topic_key].get('references', {})
            ref_ids = list(references.keys())
            dataset_split['train'][topic_key] = ref_ids
            print(f"  - {topic_key}: {len(ref_ids)} references")
        
        # 统计
        train_total = sum(len(refs) for refs in dataset_split['train'].values())
        test_total = sum(len(refs) for refs in dataset_split['test'].values())
        
        print(f"\n数据集统计:")
        print(f"  - 训练集: {len(dataset_split['train'])} topics, {train_total} references")
        print(f"  - 测试集: {len(dataset_split['test'])} topics, {test_total} references")
        
        return dataset_split
    
    def run(self):
        """执行完整的数据划分流程"""
        print("="*80)
        print("数据集划分")
        print("="*80)
        
        # 1. 加载数据
        references_data, topic_classified = self.load_data()
        
        # 2. 选择测试topics
        test_topics, train_topics = self.select_test_topics(
            references_data, topic_classified
        )
        
        # 3. 划分数据集
        dataset_split = self.split_dataset(
            references_data, test_topics, train_topics
        )
        
        # 4. 保存结果
        output_file = self.output_dir / 'dataset_split.json'
        output_data = {
            'dataset_split': dataset_split,
            'test_topics': test_topics,
            'train_topics': train_topics
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n结果已保存到: {output_file}")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description='划分数据集')
    parser.add_argument(
        '--config',
        type=str,
        default='./configs/default.json',
        help='配置文件路径'
    )
    
    args = parser.parse_args()
    
    # 加载配置
    config = SummaryBasedConfig.from_json(args.config)
    
    # 创建splitter并运行
    splitter = DatasetSplitter(
        references_file=config.path.references_file,
        topic_classified_file="/mnt/literism/data/result/topic_classified.json",
        output_dir=config.path.data_dir,
        target_test_size=config.data_split.target_test_size,
        seed=config.data_split.seed
    )
    
    splitter.run()


if __name__ == '__main__':
    main()

