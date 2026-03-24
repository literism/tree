"""
Reward计算器
实现R_global = R_margin + λ * R_len
"""
from typing import List, Dict
import math
from summary_based_classifier.core.trajectory.trajectory_sampler import Trajectory
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from modeling.deepseek_api import DeepSeekConfig, DeepSeekAPIClient
from summary_based_classifier.llm.prompts import PromptTemplates


class RewardCalculator:
    """Reward计算器"""
    
    def __init__(
        self,
        classifier,
        labeling_api_config: DeepSeekConfig,
        topic_name: str,
        beta: float = 2.0,
        gamma: float = 2.0,
        lambda_: float = 0.1,
        margin_penalty: float = -10.0
    ):
        """
        Args:
            classifier: 分类系统
            labeling_api_config: 标注系统API配置
            topic_name: topic名称
            beta: NEW action的惩罚系数
            gamma: summary update的惩罚系数
            lambda_: 长度惩罚权重
            margin_penalty: margin为负时的惩罚
        """
        self.classifier = classifier
        self.labeling_api_config = labeling_api_config
        self.topic_name = topic_name
        self.beta = beta
        self.gamma = gamma
        self.lambda_ = lambda_
        self.margin_penalty = margin_penalty
    
    def calculate_rewards_batch(
        self,
        trajectories: List[Trajectory],
        article_content: str,
        ground_truth_paths: List[str]
    ) -> List[Dict]:
        """
        批量计算轨迹的reward
        
        Args:
            trajectories: 轨迹列表
            article_content: 文章内容
            ground_truth_paths: 真实路径
            
        Returns:
            reward列表，每个元素为 {'R_margin': ..., 'R_len': ..., 'R_global': ...}
        """
        rewards = []
        
        for trajectory in trajectories:
            reward = self.calculate_reward(
                trajectory=trajectory,
                article_content=article_content,
                ground_truth_paths=ground_truth_paths
            )
            rewards.append(reward)
        
        return rewards
    
    def calculate_reward(
        self,
        trajectory: Trajectory,
        article_content: str,
        ground_truth_paths: List[str]
    ) -> Dict:
        """
        计算单条轨迹的reward
        
        Args:
            trajectory: 轨迹
            article_content: 文章内容
            ground_truth_paths: 真实路径
            
        Returns:
            {'R_margin': ..., 'R_len': ..., 'R_global': ...}
        """
        # 计算R_margin
        r_margin = self._calculate_margin_reward(
            trajectory=trajectory,
            article_content=article_content,
            ground_truth_paths=ground_truth_paths
        )
        
        # 计算R_len
        r_len = self._calculate_length_penalty(trajectory)
        
        # 计算R_global
        r_global = r_margin + self.lambda_ * r_len
        
        return {
            'R_margin': r_margin,
            'R_len': r_len,
            'R_global': r_global
        }
    
    def _calculate_margin_reward(
        self,
        trajectory: Trajectory,
        article_content: str,
        ground_truth_paths: List[str]
    ) -> float:
        """
        计算margin reward
        
        R_margin = max_{c ∈ C+} s(c) - max_{c ∉ C+} s(c)
        s(c) = logP(Yes | article, summary(c))
        """
        # 在final tree上重新分类文章
        final_tree = trajectory.final_tree
        root = final_tree
        
        # 收集所有叶子节点
        all_leaves = self._collect_leaves(root)
        
        if not all_leaves:
            return self.margin_penalty
        
        # 使用标注系统标注正确类别
        correct_categories = self._label_correct_categories(
            article_content=article_content,
            leaves=all_leaves,
            ground_truth_paths=ground_truth_paths
        )
        
        # 计算每个类别的score s(c)
        scores = []
        for leaf in all_leaves:
            # 使用classifier计算logP(Yes)
            score = self._calculate_leaf_score(leaf, article_content)
            scores.append(score)
        
        # 分离正负样本
        pos_scores = [scores[i] for i in range(len(scores)) if i in correct_categories]
        neg_scores = [scores[i] for i in range(len(scores)) if i not in correct_categories]
        
        if not pos_scores:
            return self.margin_penalty
        
        max_pos = max(pos_scores)
        max_neg = max(neg_scores) if neg_scores else -float('inf')
        
        r_margin = max_pos - max_neg
        
        return r_margin
    
    def _calculate_length_penalty(self, trajectory: Trajectory) -> float:
        """
        计算长度惩罚
        
        R_len = -∑_{t=1}^{|τ|} α(a_t)
        α(classify(existing)) = 1
        α(classify(NEW)) = β > 1
        α(summary_update at depth d) = γ/d
        """
        total_penalty = 0.0
        
        for action in trajectory.actions:
            if action.action_type == 'classify':
                if action.need_new:
                    total_penalty += self.beta
                else:
                    total_penalty += 1.0
            elif action.action_type == 'update_summary':
                depth = action.node.depth if action.node.depth > 0 else 1
                total_penalty += self.gamma / depth
        
        return -total_penalty
    
    def _collect_leaves(self, node) -> List:
        """收集所有叶子节点"""
        if not node.children:
            return [node]
        
        leaves = []
        for child in node.children:
            leaves.extend(self._collect_leaves(child))
        
        return leaves
    
    def _label_correct_categories(
        self,
        article_content: str,
        leaves: List,
        ground_truth_paths: List[str]
    ) -> List[int]:
        """
        使用标注系统标注正确类别
        
        Returns:
            正确类别的索引列表
        """
        # 构建标注prompt
        child_summaries = [leaf.summary for leaf in leaves]
        
        prompt = PromptTemplates.format_labeling_prompt(
            topic_name=self.topic_name,
            current_summary="",
            article_content=article_content,
            child_summaries=child_summaries,
            ground_truth_paths=ground_truth_paths
        )
        
        # 调用DeepSeek API
        try:
            client = DeepSeekAPIClient(self.labeling_api_config)
            responses = client.run_prompts_to_texts(
                prompts=[prompt],
                show_progress=False
            )
            
            if responses:
                response = responses[0]
                parsed = PromptTemplates.parse_labeling_output(response)
                
                if parsed:
                    return parsed['selected_indices']
        except Exception as e:
            print(f"标注失败: {e}")
        
        return []
    
    def _calculate_leaf_score(self, leaf, article_content: str) -> float:
        """
        计算叶子节点的score
        s(c) = logP(Yes | article, summary(c))
        """
        # 简化版：使用classifier的logprobs
        # TODO: 实际应该提取logP(Yes)
        return 0.0
