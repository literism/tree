"""
Reward计算器
正确实现R_global = R_margin + λ * R_len
"""
from typing import List, Dict, Tuple
import math
from summary_based_classifier.core.trajectory.trajectory_sampler import Trajectory, TreeNode, Action
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
        margin_penalty: float = -4.0
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
        self.api_client = DeepSeekAPIClient(labeling_api_config)
    
    def calculate_reward(
        self,
        trajectory: Trajectory,
        article_id: str,
        article_content: str,
        ground_truth_paths: List[str]
    ) -> Dict:
        """
        计算单条轨迹的reward
        
        正确逻辑：
        1. 提取trajectory的final_tree
        2. 在final_tree上用分类系统重新递归分类文章（不调用总结系统，不更新summary，不真的加入文章）
        3. 对每次分类，用标注系统让线上模型识别正确答案
        4. 计算R_margin
        5. 计算R_len
        
        Args:
            trajectory: 轨迹
            article_id: 文章ID
            article_content: 文章内容
            ground_truth_paths: 真实路径列表
            
        Returns:
            {'R_margin': ..., 'R_len': ..., 'R_global': ...}
        """
        # 第1步：在final_tree上重新分类
        classification_records = self._re_classify_on_tree(
            tree_root=trajectory.final_tree,
            article_content=article_content
        )
        
        # 第2步：对每次分类，用标注系统获取正确答案
        labeling_results = self._label_classifications(
            classification_records=classification_records,
            article_content=article_content,
            ground_truth_paths=ground_truth_paths
        )
        
        # 第3步：计算R_margin
        r_margin = self._calculate_margin_reward(labeling_results)
        
        # 第4步：计算R_len（基于trajectory的actions）
        r_len = self._calculate_length_penalty(trajectory.actions)
        
        # 第5步：计算R_global
        r_global = r_margin + self.lambda_ * r_len
        
        return {
            'R_margin': r_margin,
            'R_len': r_len,
            'R_global': r_global
        }
    
    def _re_classify_on_tree(
        self,
        tree_root: TreeNode,
        article_content: str
    ) -> List[Dict]:
        """
        在树上重新递归分类文章（不调用总结系统，不更新，不真的加入文章）
        
        返回分类记录列表，每个记录包含：
        - current_node: 当前节点
        - child_summaries: 子节点summaries
        - classification_output: 分类系统的输出
        - scores: 每个类别的logP(Yes)
        """
        records = []
        
        def classify_recursive(current_node: TreeNode, depth: int = 0):
            # 达到最大深度，停止
            if depth >= 10:  # max_depth
                return
            
            children = current_node.children
            if not children:
                # 叶子节点，停止
                return
            
            child_summaries = [c.summary for c in children]
            
            # 调用分类系统
            from summary_based_classifier.llm.classify_generator import ClassificationInput
            classification_input = ClassificationInput(
                article_content=article_content,
                current_node_summary=current_node.summary if current_node.summary else self.topic_name,
                child_summaries=child_summaries,
                topic_name=self.topic_name
            )
            
            # 获取分类输出，并要求返回logprobs
            classification_output, scores = self._classify_with_logprobs(
                classification_input, len(child_summaries)
            )
            
            if classification_output is None:
                return
            
            # 记录这次分类
            records.append({
                'current_node': current_node,
                'child_summaries': child_summaries,
                'classification_output': classification_output,
                'scores': scores  # {0: logP(Yes), 1: logP(Yes), ..., 'NEW': logP(Yes)}
            })
            
            # 递归到选中的子节点（只用于继续收集分类记录，不真的修改树）
            for idx in classification_output.selected_indices:
                if 0 <= idx < len(children):
                    classify_recursive(children[idx], depth + 1)
            
            # 如果需要NEW，不递归（因为树上没有这个新节点）
        
        classify_recursive(tree_root)
        return records
    
    def _classify_with_logprobs(
        self,
        classification_input,
        num_children: int
    ):
        """
        调用分类系统并获取每个类别的logP(Yes)
        
        返回: (classification_output, scores)
        - classification_output: 分类输出
        - scores: {0: logP(Yes), 1: logP(Yes), ..., 'NEW': logP(Yes)}
        """
        # 如果classifier有classify_with_logprobs方法，直接调用
        if hasattr(self.classifier, 'classify_with_logprobs'):
            return self.classifier.classify_with_logprobs(classification_input)
        
        # 否则fallback到原来的逻辑（兼容性）
        if hasattr(self.classifier, 'llm'):
            # 直接使用classify_generator中的方法
            from summary_based_classifier.llm.classify_generator import ClassifyGenerator
            if isinstance(self.classifier, ClassifyGenerator):
                return self.classifier.classify_with_logprobs(classification_input)
        
        # 最后的fallback：不返回logprobs
        outputs = self.classifier.classify_with_sampling(classification_input, n=1)
        if not outputs:
            return None, {}
        
        classification_output = outputs[0]
        
        # 无法获取logprobs，使用0作为占位
        scores = {}
        for i in range(num_children):
            scores[i] = 0.0
        scores['NEW'] = 0.0
        
        return classification_output, scores
    
    def _label_classifications(
        self,
        classification_records: List[Dict],
        article_content: str,
        ground_truth_paths: List[str]
    ) -> List[Dict]:
        """
        对每次分类，用标注系统获取正确答案和质量检查
        
        标注系统输入：
        - topic_name
        - current_node_summary
        - child_summaries
        - ground_truth_paths（真实的分类路径）
        
        标注系统输出：
        - exceed_parent: 超出父类范围的子类索引列表
        - overlapping_pairs: 有重复的子类索引对列表
        - correct_indices: 正确的子类别索引列表
        - need_new: 是否需要NEW
        """
        # 批量构建prompts
        prompts = []
        for record in classification_records:
            prompt = self._create_labeling_prompt(
                topic_name=self.topic_name,
                current_node_summary=record['current_node'].summary if record['current_node'].summary else self.topic_name,
                child_summaries=record['child_summaries'],
                ground_truth_paths=ground_truth_paths
            )
            prompts.append(prompt)
        
        # 批量调用API
        responses = self.api_client.run_prompts_to_texts(prompts, show_progress=False)
        
        # 解析响应
        labeling_results = []
        for i, (record, response) in enumerate(zip(classification_records, responses)):
            if response is None:
                # API失败，使用默认值
                labeling_results.append({
                    'record': record,
                    'exceed_parent': None,
                    'overlapping_pairs': None,
                    'correct_indices': [],
                    'need_new': False
                })
                continue
            
            # 解析标注结果
            parsed = self._parse_labeling_output(response, len(record['child_summaries']))
            labeling_results.append({
                'record': record,
                'exceed_parent': parsed['exceed_parent'],
                'overlapping_pairs': parsed['overlapping_pairs'],
                'correct_indices': parsed['correct_indices'],
                'need_new': parsed['need_new']
            })
        
        return labeling_results
    
    def _create_labeling_prompt(
        self,
        topic_name: str,
        current_node_summary: str,
        child_summaries: List[str],
        ground_truth_paths: List[str]
    ) -> str:
        """创建标注系统的prompt（不再需要文章内容）"""
        return PromptTemplates.format_labeling_prompt(
            topic_name=topic_name,
            current_summary=current_node_summary,
            child_summaries=child_summaries,
            ground_truth_paths=ground_truth_paths
        )
    
    def _parse_labeling_output(self, response: str, num_children: int) -> Dict:
        """解析标注系统的输出（使用prompts.py中的解析逻辑）"""
        parsed = PromptTemplates.parse_labeling_output(response, num_children)
        if parsed is None:
            # 解析失败，返回默认值
            return {
                'exceed_parent': None,
                'overlapping_pairs': None,
                'correct_indices': [],
                'need_new': False
            }
        return parsed
    
    def _calculate_margin_reward(self, labeling_results: List[Dict]) -> float:
        """
        计算margin reward
        
        R_margin = 1/len(C) Σ (mean_{c ∈ C+} s(c) - mean_{c ∉ C+} s(c)) + quality_penalty
        
        quality_penalty：如果标注系统发现质量问题，给予惩罚
        - exceed_parent：子类超出父类范围
        - overlapping_pairs：子类之间有重复
        """
        total_margin = 0.0
        quality_penalty_per_issue = self.margin_penalty / 4  # 每个质量问题的惩罚（减半以避免过度惩罚）
        
        for result in labeling_results:
            record = result['record']
            scores = record['scores']
            correct_indices = set(result['correct_indices'])
            need_new_correct = result['need_new']
            
            # 1. 检查子类是否超出父类范围
            exceed_parent = result.get('exceed_parent')
            
            # 2. 检查子类之间是否有重复
            overlapping_pairs = result.get('overlapping_pairs')
            
            # 计算正确类别的最大score
            correct_scores = []
            for idx in correct_indices:
                if idx in scores:
                    correct_scores.append(scores[idx])
            if need_new_correct and 'NEW' in scores:
                correct_scores.append(scores['NEW'])
            
            # 计算错误类别的最大score
            incorrect_scores = []
            for idx, score in scores.items():
                if idx == 'NEW':
                    if not need_new_correct:
                        incorrect_scores.append(score)
                else:
                    if idx not in correct_indices:
                        incorrect_scores.append(score)
            
            # 计算margin
            if exceed_parent or overlapping_pairs:
                total_margin += self.margin_penalty
            elif need_new_correct:
                total_margin += quality_penalty_per_issue
            elif correct_scores and incorrect_scores:
                margin = sum(correct_scores) / len(correct_scores) - sum(incorrect_scores) / len(incorrect_scores)
                total_margin += margin
            elif correct_scores and not incorrect_scores:
                # 所有类别都是正确的，不存在margin，跳过
                total_margin += 0
            elif not correct_scores:
                # 没有正确类别，惩罚
                total_margin += self.margin_penalty

        if len(labeling_results) > 0:
            total_margin = total_margin / len(labeling_results)

        return total_margin
    
    def _calculate_length_penalty(self, actions: List[Action]) -> float:
        """
        计算长度惩罚
        
        R_len = -Σ α(a_t)
        α(classify(existing)) = 1
        α(classify(NEW)) = β
        α(summary_update at depth d) = γ/d
        """
        total_penalty = 0.0
        
        for action in actions:
            if action.action_type == 'classify':
                if action.need_new:
                    total_penalty -= self.beta  # NEW的惩罚更大
                else:
                    total_penalty -= 1.0  # 已有类别的惩罚
            
            elif action.action_type == 'generate_new_summary':
                total_penalty -= self.beta  # 生成新summary的惩罚
            
            elif action.action_type == 'update_summary':
                # 获取节点深度
                depth = action.node.depth if hasattr(action.node, 'depth') else 1
                if depth > 0:
                    total_penalty -= (self.gamma / depth)
        
        return total_penalty

