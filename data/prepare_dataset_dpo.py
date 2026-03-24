"""
DPO数据集准备
实现完整的轨迹采样、reward计算、偏好对构建流程
"""
import json
import os
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent))
from summary_based_classifier.config import SummaryBasedConfig
from summary_based_classifier.core.trajectory.trajectory_sampler import TrajectorySampler, TreeNode, Trajectory
from summary_based_classifier.reward.reward_calculator import RewardCalculator
from summary_based_classifier.llm.classify_generator import ClassifyGenerator
from summary_based_classifier.llm.updater import Updater
from trajectory_dpo_trainer import PreferencePair
from modeling.deepseek_api import DeepSeekConfig


class DPODatasetPreparator:
    """DPO数据集准备器"""
    
    def __init__(self, config: SummaryBasedConfig):
        """
        Args:
            config: 配置对象
        """
        self.config = config
        self.output_dir = Path(config.path.data_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载数据
        self.structures_data = None
        self.references_data = None
        self.summaries_data = None
        self.dataset_split = None
    
    def load_data(self):
        """加载所有需要的数据"""
        print("加载数据...")
        
        # 加载structures
        with open(self.config.path.structures_file, 'r', encoding='utf-8') as f:
            self.structures_data = json.load(f)
        print(f"  - Structures: {len(self.structures_data)} topics")
        
        # 加载references
        with open(self.config.path.references_file, 'r', encoding='utf-8') as f:
            self.references_data = json.load(f)
        print(f"  - References: {len(self.references_data)} topics")
        
        # 加载summaries
        summaries_file = Path(self.config.path.summaries_dir) / 'node_summaries.json'
        with open(summaries_file, 'r', encoding='utf-8') as f:
            self.summaries_data = json.load(f)
        print(f"  - Summaries: {len(self.summaries_data)} topics")
        
        # 加载dataset split
        split_file = self.output_dir / 'dataset_split.json'
        with open(split_file, 'r', encoding='utf-8') as f:
            split_data = json.load(f)
            self.dataset_split = split_data['dataset_split']
        print(f"  - Dataset split loaded")
    
    def sample_trajectories_for_articles(
        self,
        iteration: int,
        num_articles: int,
        classifier: ClassifyGenerator,
        updater: Updater
    ) -> List[Dict]:
        """
        为指定数量的文章采样轨迹
        
        Args:
            iteration: 当前迭代次数
            num_articles: 采样的文章数量
            classifier: 分类系统
            updater: 总结系统
            
        Returns:
            轨迹数据列表
        """
        print(f"\n=== 迭代 {iteration + 1}: 采样 {num_articles} 篇文章的轨迹 ===")
        
        # 从训练集中采样文章
        train_topics = self.dataset_split.get('train', {})
        all_articles = []
        
        for topic_key, ref_ids in train_topics.items():
            if topic_key not in self.references_data:
                continue
            
            topic_data = self.references_data[topic_key]
            topic_name = topic_data.get('topic', topic_key)
            
            for ref_id in ref_ids[:num_articles]:  # 简化：每个topic取前N篇
                if ref_id in topic_data.get('references', {}):
                    ref = topic_data['references'][ref_id]
                    content = ref.get('content', '')
                    paths = ref.get('paths', [])
                    
                    if content and paths:
                        all_articles.append({
                            'topic_key': topic_key,
                            'topic_name': topic_name,
                            'ref_id': ref_id,
                            'content': content,
                            'paths': paths
                        })
            
            if len(all_articles) >= num_articles:
                break
        
        all_articles = all_articles[:num_articles]
        
        # 为每篇文章采样轨迹
        trajectory_data = []
        
        for article in tqdm(all_articles, desc="采样轨迹"):
            # 创建初始树
            initial_tree = TreeNode(
                summary="",
                citations=[],
                children=[],
                depth=0
            )
            
            # 创建轨迹采样器
            sampler = TrajectorySampler(
                classifier=classifier,
                updater=updater,
                topic_name=article['topic_name'],
                initial_tree=initial_tree,
                max_depth=self.config.inference.max_depth,
                num_samples=self.config.dpo_training.num_trajectories_per_article,
                top_k=self.config.dpo_training.trajectory_top_k,
                temperature=0.7
            )
            
            # 采样轨迹
            trajectories = sampler.sample_trajectories(
                article_id=article['ref_id'],
                article_content=article['content']
            )
            
            trajectory_data.append({
                'article': article,
                'trajectories': trajectories
            })
        
        return trajectory_data
    
    def calculate_rewards(
        self,
        trajectory_data: List[Dict],
        classifier: ClassifyGenerator
    ) -> List[Dict]:
        """
        计算所有轨迹的reward
        
        Args:
            trajectory_data: 轨迹数据
            classifier: 分类系统
            
        Returns:
            带reward的轨迹数据
        """
        print("\n计算轨迹reward...")
        
        # 创建标注系统API配置
        labeling_api_config = DeepSeekConfig(
            api_key=self.config.summary.api_key,
            base_url=self.config.summary.api_url.replace('/chat/completions', ''),
            model=self.config.summary.model_name,
            temperature=0.0,
            max_output_tokens=512,
            max_concurrent_jobs=self.config.summary.max_workers
        )
        
        for data in tqdm(trajectory_data, desc="计算reward"):
            article = data['article']
            trajectories = data['trajectories']
            
            # 创建reward计算器
            reward_calc = RewardCalculator(
                classifier=classifier,
                labeling_api_config=labeling_api_config,
                topic_name=article['topic_name'],
                beta=self.config.dpo_training.reward_beta,
                gamma=self.config.dpo_training.reward_gamma,
                lambda_=self.config.dpo_training.reward_lambda,
                margin_penalty=self.config.dpo_training.reward_margin_penalty
            )
            
            # 计算每条轨迹的reward
            rewards = reward_calc.calculate_rewards_batch(
                trajectories=trajectories,
                article_content=article['content'],
                ground_truth_paths=article['paths']
            )
            
            data['rewards'] = rewards
        
        return trajectory_data
    
    def construct_preference_pairs(
        self,
        trajectory_data: List[Dict]
    ) -> Tuple[List[PreferencePair], List[PreferencePair]]:
        """
        构建偏好对（Top-1 vs Rest）
        
        Args:
            trajectory_data: 带reward的轨迹数据
            
        Returns:
            (classifier_pairs, updater_pairs)
        """
        print("\n构建偏好对...")
        
        classifier_pairs = []
        updater_pairs = []
        
        for data in trajectory_data:
            trajectories = data['trajectories']
            rewards = data['rewards']
            
            if len(trajectories) < 2:
                continue
            
            # 按reward排序
            sorted_indices = sorted(
                range(len(rewards)),
                key=lambda i: rewards[i]['R_global'],
                reverse=True
            )
            
            # 取Top-1
            best_idx = sorted_indices[0]
            best_trajectory = trajectories[best_idx]
            best_reward = rewards[best_idx]['R_global']
            
            # 与其余的构建偏好对
            for i in sorted_indices[1:]:
                worse_trajectory = trajectories[i]
                worse_reward = rewards[i]['R_global']
                margin = best_reward - worse_reward
                
                # 提取分类actions和总结actions
                classifier_actions_pos = self._extract_classifier_actions(best_trajectory)
                classifier_actions_neg = self._extract_classifier_actions(worse_trajectory)
                
                updater_actions_pos = self._extract_updater_actions(best_trajectory)
                updater_actions_neg = self._extract_updater_actions(worse_trajectory)
                
                # 创建偏好对
                if classifier_actions_pos and classifier_actions_neg:
                    classifier_pairs.append(PreferencePair(
                        positive_trajectory=classifier_actions_pos,
                        negative_trajectory=classifier_actions_neg,
                        margin=margin
                    ))
                
                if updater_actions_pos and updater_actions_neg:
                    updater_pairs.append(PreferencePair(
                        positive_trajectory=updater_actions_pos,
                        negative_trajectory=updater_actions_neg,
                        margin=margin
                    ))
        
        print(f"  - 分类器偏好对: {len(classifier_pairs)}")
        print(f"  - 总结器偏好对: {len(updater_pairs)}")
        
        return classifier_pairs, updater_pairs
    
    def _extract_classifier_actions(self, trajectory: Trajectory) -> List[Tuple[str, str]]:
        """提取分类actions"""
        # TODO: 实现提取逻辑
        # 这里需要将trajectory中的分类action转换为(prompt, completion)对
        return []
    
    def _extract_updater_actions(self, trajectory: Trajectory) -> List[Tuple[str, str]]:
        """提取总结actions"""
        # TODO: 实现提取逻辑
        return []
    
    def save_preference_pairs(
        self,
        classifier_pairs: List[PreferencePair],
        updater_pairs: List[PreferencePair],
        iteration: int
    ):
        """保存偏好对"""
        classifier_file = self.output_dir / f'classifier_pairs_iter{iteration}.json'
        updater_file = self.output_dir / f'updater_pairs_iter{iteration}.json'
        
        # 简化：保存为JSON（实际训练时需要转换）
        with open(classifier_file, 'w', encoding='utf-8') as f:
            json.dump([{
                'pos': pair.positive_trajectory,
                'neg': pair.negative_trajectory,
                'margin': pair.margin
            } for pair in classifier_pairs], f, indent=2, ensure_ascii=False)
        
        with open(updater_file, 'w', encoding='utf-8') as f:
            json.dump([{
                'pos': pair.positive_trajectory,
                'neg': pair.negative_trajectory,
                'margin': pair.margin
            } for pair in updater_pairs], f, indent=2, ensure_ascii=False)
        
        print(f"  - 偏好对已保存")


def main():
    parser = argparse.ArgumentParser(description='准备DPO训练数据')
    parser.add_argument(
        '--config',
        type=str,
        default='./configs/default.json',
        help='配置文件路径'
    )
    
    args = parser.parse_args()
    
    # 加载配置
    config = SummaryBasedConfig.from_json(args.config)
    
    # 创建准备器并运行
    preparator = DPODatasetPreparator(config)
    preparator.load_data()
    
    print("\nDPO数据集准备完成（需要实际实现完整流程）")


if __name__ == '__main__':
    main()

