"""
IW SFT迭代训练完整流程
实现轨迹采样 -> Reward计算 -> Advantage计算 -> 加权SFT训练的完整循环
"""
import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["RAY_memory_monitor_refresh_ms"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import sys
import subprocess

sys.path.append(str(Path(__file__).parent.parent))
from summary_based_classifier.config import SummaryBasedConfig
from summary_based_classifier.core.trajectory.trajectory_sampler import TrajectorySampler, TreeNode, Trajectory, Action
from summary_based_classifier.reward.reward_calculator import RewardCalculator
from summary_based_classifier.llm.classify_generator import ClassifyGenerator, ClassificationInput
from summary_based_classifier.llm.updater import Updater, SummaryInput
from summary_based_classifier.core.topic_state import TopicState, TopicStateManager
from summary_based_classifier.core.trajectory.parallel_trajectory_processor import ParallelTrajectoryProcessor
from sequential_article_processor import StepSample  # 使用sequential_article_processor的定义
# 不再使用DPO训练，改为加权SFT
# from trajectory_dpo_trainer import TrajectoryDPOTrainer, TrajectoryDPODataset
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from modeling.deepseek_api import DeepSeekConfig
from summary_based_classifier.llm.prompts import PromptTemplates
import numpy as np
import torch
import pickle
import gzip

# 导入新的模块
from summary_based_classifier.core.trajectory.trajectory_storage import TrajectoryStorage, StoredTrajectory, StoredDecisionPoint
from summary_based_classifier.data.batch_labeler import BatchLabeler, LabelingRequest, LabelingResult


def compute_advantages_and_weights(
    samples: List[StepSample],
    advantage_temperature: float = 1.0,
    advantage_min: float = -10.0,
    advantage_max: float = 10.0,
    advantage_epsilon: float = 0.1,
    reward_margin_penalty: float = -4
) -> List[StepSample]:
    """
    计算advantage和weight
    
    根据说明文档：
    1. 对每个state，计算baseline: b(s) = mean(R_global)
    2. 计算advantage: A(s,a) = R_global - b(s)
    3. 过滤: A(s,a) < epsilon的样本被丢弃, R_global < -4的样本被丢弃
    4. 计算weight: w(s,a) = exp(τ * clip(A(s,a), A_min, A_max))
    
    Args:
        samples: StepSample列表
        advantage_temperature: 控制重加权的锐度（τ）
        advantage_min: advantage裁剪下界
        advantage_max: advantage裁剪上界
        advantage_epsilon: 过滤阈值
        
    Returns:
        过滤后的样本列表，每个样本的advantage和weight已计算
    """
    # 1. 按state分组
    state_groups = {}
    for sample in samples:
        state = sample.state
        if state not in state_groups:
            state_groups[state] = []
        state_groups[state].append(sample)
    
    # 2. 对每组计算baseline和advantage
    filtered_samples = []
    
    for state, group_samples in state_groups.items():
        # 计算baseline: b(s) = mean(R_global)
        rewards = [s.global_reward for s in group_samples]
        baseline = np.mean(rewards)
        
        # 计算advantage并过滤
        for sample in group_samples:
            advantage = sample.global_reward - baseline
            
            # 过滤: A(s,a) < epsilon
            if advantage < advantage_epsilon or sample.global_reward < reward_margin_penalty:
                continue
            
            # 裁剪advantage
            clipped_advantage = np.clip(advantage, advantage_min, advantage_max)
            
            # 计算weight: exp(τ * clipped_advantage)
            weight = np.exp(advantage_temperature * clipped_advantage)
            
            # 更新sample
            sample.advantage = advantage
            sample.weight = weight
            
            filtered_samples.append(sample)
    
    return filtered_samples


class IWSFTTrainingPipeline:
    """IW SFT训练完整pipeline（Advantage-weighted SFT）"""
    
    def __init__(self, config: SummaryBasedConfig):
        self.config = config
        self.output_dir = Path(config.path.data_dir)
        self.data_dir = Path(config.path.data_dir)  # 数据目录（包含各topic子目录）
        self.models_dir = Path(config.path.models_dir)
        
        # 加载数据
        self.structures_data = None
        self.references_data = None
        self.summaries_data = None
        self.dataset_split = None
        
        # 初始化轨迹存储管理器
        self.trajectory_storage = TrajectoryStorage(str(self.output_dir))
    
    def load_data(self):
        """加载所有需要的数据"""
        print("加载数据...")
        
        with open(self.config.path.structures_file, 'r', encoding='utf-8') as f:
            self.structures_data = json.load(f)
        
        with open(self.config.path.references_file, 'r', encoding='utf-8') as f:
            self.references_data = json.load(f)
        
        summaries_file = Path(self.config.path.summaries_dir) / 'node_summaries.json'
        with open(summaries_file, 'r', encoding='utf-8') as f:
            self.summaries_data = json.load(f)
        
        split_file = self.output_dir / 'dataset_split.json'
        with open(split_file, 'r', encoding='utf-8') as f:
            self.dataset_split = json.load(f)['dataset_split']
        
        # 初始化topic states
        self._initialize_topic_states()
        
        print(f"  ✓ 数据加载完成")
    
    def _initialize_topic_states(self):
        """初始化topic状态（使用TopicStateManager管理持久化）"""
        # 创建state manager（所有迭代共享同一个状态目录）
        state_dir = self.output_dir / 'topic_states'
        self.state_manager = TopicStateManager(save_dir=state_dir)
        
        train_topics = self.dataset_split.get('train', {})
        
        print(f"  初始化topic状态（从 {state_dir} 加载或创建）...")
        for topic_key in train_topics.keys():
            # 获取topic名称和文章列表
            topic_data = self.references_data.get(topic_key, {})
            topic_name = topic_data.get('topic', topic_key)
            articles = list(topic_data.get('references', {}).keys())
            
            if not articles:
                continue
            
            # 总是尝试加载已有状态；如果不存在，会创建新的
            self.state_manager.initialize_topic(
                topic_key=topic_key,
                topic_name=topic_name,
                article_ids=articles,
                load_if_exists=True
            )
        
        # 提供self.topic_states作为快捷访问（指向state_manager.states）
        self.topic_states = self.state_manager.states
        
        print(f"  ✓ 初始化了 {len(self.topic_states)} 个topic的状态")
        
        # 显示状态摘要
        loaded_count = sum(1 for s in self.topic_states.values() if s.next_article_idx > 0)
        if loaded_count > 0:
            print(f"  ℹ️  其中 {loaded_count} 个topic从已保存状态恢复")
    
    def _reset_all_topics_for_iteration(self):
        """
        为新的iteration重置所有topic的状态（取消状态继承）
        
        每个iteration应该是独立的：
        1. 重置树结构（新建初始树）
        2. 重新打乱文章顺序
        """
        for topic_key, state in self.topic_states.items():
            # 重置树结构
            state.reset_tree()
            
            # 重新打乱文章
            topic_references = self.references_data.get(topic_key, {}).get('references', {})
            article_ids = list(topic_references.keys())
            if article_ids:
                state.shuffle_articles(article_ids)
    
    def _cleanup_models(self):
        """清理当前加载的模型并释放显存"""
        print("\n清理模型资源...")
        
        # 主要是清理torch缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("  ✓ 显存清理完成")
    
    def run_sampling_phase(
        self,
        iteration: int,
        num_articles: int,
        classify_generator_model: str,
        updater_model: str
    ) -> str:
        """
        运行采样阶段：采样轨迹和重新分类
        
        Returns:
            保存的轨迹文件路径
        """
        print("\n" + "="*80)
        print(f"阶段1: 采样和重新分类")
        print(f"Iteration: iter_{iteration}")
        print("="*80)
        
        # 0. 重置所有topic状态
        print(f"\n步骤0: 重置topic状态...")
        self._reset_all_topics_for_iteration()
        print(f"  ✓ 已重置所有topic的树结构并重新打乱文章顺序")
        
        # 1. 准备模型路径
        print(f"\n步骤1: 准备模型路径...")
        print(f"  - 分类生成模型: {classify_generator_model}")
        print(f"  - 总结更新模型: {updater_model}")
        
        # 2. 采样轨迹
        print(f"\n步骤2: 顺序处理文章（采样轨迹→使用主轨迹更新树）...")
        
        from sequential_article_processor import SequentialArticleProcessor
        
        processor = SequentialArticleProcessor(
            classifier_model_path=classify_generator_model,
            updater_model_path=updater_model,
            topic_states=self.topic_states,
            sampling_num=self.config.dpo_training.sampling_top_k,
            top_k=self.config.dpo_training.trajectory_top_k,
            max_depth=self.config.inference.max_depth,
            config=self.config
        )
        
        # 处理文章（获取轨迹数据）
        trajectory_data_list = processor.process_articles(
            target_article_count=num_articles,
            references_data=self.references_data,
            summaries_data=self.summaries_data
        )
        
        print(f"\n  ✓ 文章处理完成")
        print(f"    - 文章数: {len(trajectory_data_list)}")
        total_trajectories = sum(len(td.trajectories) for td in trajectory_data_list)
        print(f"    - 总轨迹数: {total_trajectories}")
        
        # 保存topic状态
        print(f"\n  保存topic状态...")
        self.state_manager.save_all()
        print(f"  ✓ 状态已保存")
        
        # 清理processor资源
        print(f"\n  清理processor资源...")
        del processor
        import gc
        gc.collect()
        print(f"  ✓ Processor资源已清理")
        
        # 保存轨迹数据
        trajectory_file = self.output_dir / 'train_trajectories' / f'iteration_{iteration}_trajectories.pkl.gz'
        trajectory_file.parent.mkdir(parents=True, exist_ok=True)
        
        with gzip.open(trajectory_file, 'wb') as f:
            pickle.dump({
                'iteration': iteration,
                'trajectory_data_list': trajectory_data_list
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"\n✓ 轨迹数据已保存到: {trajectory_file}")
        
        # 清理模型
        self._cleanup_models()
        
        return str(trajectory_file)
    
    def run_labeling_phase(
        self,
        iteration: int,
        trajectory_file: str
    ) -> Tuple[List[StepSample], List[StepSample]]:
        """
        运行标注阶段：批量标注和计算reward
        
        Args:
            iteration: iteration编号
            trajectory_file: 轨迹文件路径
        
        Returns:
            (classifier_samples, updater_samples)
        """
        print("\n" + "="*80)
        print(f"阶段2: 批量标注和reward计算")
        print(f"Iteration: iter_{iteration}")
        print("="*80)
        
        # 1. 加载轨迹数据
        print(f"\n步骤1: 加载轨迹数据...")
        with gzip.open(trajectory_file, 'rb') as f:
            data = pickle.load(f)
        
        trajectory_data_list = data['trajectory_data_list']
        
        print(f"  ✓ 数据已加载")
        print(f"    - 文章数: {len(trajectory_data_list)}")
        total_trajectories = sum(len(td.trajectories) for td in trajectory_data_list)
        print(f"    - 总轨迹数: {total_trajectories}")
        
        # 检查是否跳过标注阶段
        labeling_file = self.output_dir / 'train_trajectories' / f'iteration_{iteration}_labeled.pkl.gz'
        
        if self.config.dpo_training.skip_labeling and labeling_file.exists():
            print(f"\n  跳过标注阶段，加载已有标注结果...")
            print(f"  标注文件: {labeling_file}")
            
            with gzip.open(labeling_file, 'rb') as f:
                labeled_data = pickle.load(f)
            
            trajectory_data_list = labeled_data['trajectory_data_list']
            print(f"  ✓ 标注结果已加载")
            
            # 如果需要训练标注模型，直接加载训练数据集
            if iteration == 0 and self.config.labeling.train_labeling_model:
                print(f"\n  训练本地标注模型（使用已有数据集）...")
                dataset_file = self.output_dir / 'labeling_model_training' / f'iteration_{iteration}_labeling_train.jsonl'
                
                if dataset_file.exists():
                    print(f"    - 找到训练数据集: {dataset_file}")
                    self._run_sft_training(
                        dataset_file=dataset_file,
                        output_dir=self.output_dir / 'labeling_model_training' / 'checkpoints',
                        iteration=iteration
                    )
                else:
                    print(f"    ⚠ 未找到训练数据集: {dataset_file}")
                    print(f"    跳过训练")
        else:
            # 2. 批量标注
            print(f"\n步骤2: 批量标注...")
            print(f"  标注模式: {self.config.labeling.mode}")
            
            # 2.1 收集所有需要标注的classification records
            print(f"  收集标注请求...")
            all_labeling_requests = []
            request_to_trajectory_mapping = []  # 记录每个请求对应哪个文章和轨迹
            
            for article_idx, article_data in enumerate(trajectory_data_list):
                if len(article_data.trajectories) <= 1:
                    continue
                for traj_idx, records in enumerate(article_data.re_classification_records):
                    for record_idx, record in enumerate(records):
                        # 构建标注请求
                        request = LabelingRequest(
                            topic_name=article_data.topic_name,
                            current_summary=record['current_node'].summary if record['current_node'].summary else article_data.topic_name,
                            child_summaries=record['child_summaries'],
                            ground_truth_paths=article_data.ground_truth_paths,
                            metadata={
                                'article_idx': article_idx,
                                'traj_idx': traj_idx,
                                'record_idx': record_idx
                            }
                        )
                        all_labeling_requests.append(request)
                        request_to_trajectory_mapping.append((article_idx, traj_idx, record_idx))
            
            print(f"  ✓ 收集了 {len(all_labeling_requests)} 个标注请求")
            
            # 2.2 批量调用标注系统
            # 如果是 iteration 0 且需要训练标注模型，强制使用 API 模式
            labeling_mode = self.config.labeling.mode
            force_api_for_training = (iteration == 0 and self.config.labeling.train_labeling_model)
            
            if force_api_for_training and labeling_mode == 'local':
                print(f"  ⚠ 检测到需要训练标注模型，强制使用API模式获取标注数据")
                labeling_mode = 'api'
            
            print(f"  批量标注... (模式: {labeling_mode})")
            labeler = BatchLabeler(self.config, mode=labeling_mode)
            labeling_results = labeler.label_batch(all_labeling_requests)
            labeler.cleanup()
            
            print(f"  ✓ 批量标注完成")
            
            # 2.3 将标注结果分配回对应的轨迹
            print(f"  分配标注结果...")
            for result, (article_idx, traj_idx, record_idx) in zip(labeling_results, request_to_trajectory_mapping):
                article_data = trajectory_data_list[article_idx]
                record = article_data.re_classification_records[traj_idx][record_idx]
                
                # 将标注结果添加到record中
                record['labeling_result'] = {
                    'exceed_parent': result.exceed_parent,
                    'overlapping_pairs': result.overlapping_pairs,
                    'correct_indices': result.correct_indices,
                    'need_new': result.need_new
                }
            
            print(f"  ✓ 标注结果已分配")
            
            # 2.4 保存标注结果
            print(f"\n  保存标注结果...")
            labeling_file.parent.mkdir(parents=True, exist_ok=True)
            
            with gzip.open(labeling_file, 'wb') as f:
                pickle.dump({
                    'iteration': iteration,
                    'trajectory_data_list': trajectory_data_list
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            print(f"  ✓ 标注结果已保存到: {labeling_file}")
            
            # 2.5 训练标注模型（仅在iteration 0且配置要求时）
            if iteration == 0 and self.config.labeling.train_labeling_model:
                print(f"\n  训练本地标注模型...")
                self._train_labeling_model(
                    all_labeling_requests=all_labeling_requests,
                    api_results=labeling_results,
                    iteration=iteration
                )
            
            # 2.6 评估本地标注模型效果（如果是local模式）
            if self.config.labeling.mode == 'local':
                print(f"\n  评估本地标注模型效果...")
                self._evaluate_local_labeling(
                    all_labeling_requests=all_labeling_requests,
                    local_results=labeling_results,
                    iteration=iteration
                )
        
        # 3. 计算reward
        print(f"\n步骤3: 计算reward...")
        trajectory_data_list = self._calculate_rewards(trajectory_data_list)
        print(f"  ✓ Reward计算完成")
        
        # 4. 展平轨迹为step-level samples
        print(f"\n步骤4: 展平轨迹为step-level samples...")
        classifier_samples, updater_samples = self._flatten_trajectories(trajectory_data_list)
        
        print(f"  ✓ 展平完成")
        print(f"    - 分类系统样本数: {len(classifier_samples)}")
        print(f"    - 更新系统样本数: {len(updater_samples)}")
        
        # 5. 计算advantage和weight
        print(f"\n步骤5: 计算advantage和weight...")
        classifier_samples = compute_advantages_and_weights(
            classifier_samples,
            advantage_temperature=self.config.dpo_training.advantage_temperature,
            reward_margin_penalty=self.config.dpo_training.reward_margin_penalty
        )
        updater_samples = compute_advantages_and_weights(
            updater_samples,
            advantage_temperature=self.config.dpo_training.advantage_temperature,
            reward_margin_penalty=self.config.dpo_training.reward_margin_penalty
        )
        print(f"  ✓ Advantage计算完成")
        print(f"    - 分类系统过滤后: {len(classifier_samples)} 样本")
        print(f"    - 更新系统过滤后: {len(updater_samples)} 样本")
        
        # 6. 保存数据集
        print(f"\n步骤6: 保存训练数据...")
        self._save_step_dataset(
            classifier_samples=classifier_samples,
            updater_samples=updater_samples,
            iteration=iteration
        )
        
        return classifier_samples, updater_samples
    
    def _calculate_rewards(
        self,
        trajectory_data_list: List
    ) -> List:
        """
        为每个轨迹计算reward
        
        Args:
            trajectory_data_list: 轨迹数据列表（已包含标注结果）
        
        Returns:
            更新了reward的trajectory_data_list
        """
        # 创建 RewardCalculator（使用API配置）
        from modeling.deepseek_api import DeepSeekConfig
        labeling_api_config = DeepSeekConfig(
            api_key=self.config.summary.api_key,
            base_url=self.config.summary.api_url,
            model=self.config.summary.model_name,
            temperature=0.0,
            max_output_tokens=512,
            max_concurrent_jobs=self.config.summary.max_workers
        )
        
        # 创建 RewardCalculator（只用它的计算方法）
        reward_calc = RewardCalculator(
            classifier=None,  # 不需要，因为我们已经有了分类结果
            labeling_api_config=labeling_api_config,
            topic_name="",  # 不需要
            beta=self.config.dpo_training.reward_beta,
            gamma=self.config.dpo_training.reward_gamma,
            lambda_=self.config.dpo_training.reward_lambda,
            margin_penalty=self.config.dpo_training.reward_margin_penalty
        )
        
        for article_data in tqdm(trajectory_data_list, desc="计算reward"):
            for traj_idx, traj in enumerate(article_data.trajectories):
                # 获取该轨迹的classification records（已包含标注结果）
                records_with_labels = article_data.re_classification_records[traj_idx]
                
                # 构建 labeling_results 格式（适配 RewardCalculator）
                labeling_results = []
                for record in records_with_labels:
                    if 'labeling_result' not in record:
                        continue
                    
                    labeling_result = record['labeling_result']
                    
                    # 构建与 RewardCalculator 期望的格式一致的数据
                    labeling_results.append({
                        'record': {
                            'current_node': record['current_node'],
                            'child_summaries': record['child_summaries'],
                            'classification_output': record['classification_output'],
                            'scores': record.get('scores', {})  # 如果有 scores 就用，没有就用空字典
                        },
                        'exceed_parent': labeling_result['exceed_parent'],
                        'overlapping_pairs': labeling_result['overlapping_pairs'],
                        'correct_indices': labeling_result['correct_indices'],
                        'need_new': labeling_result['need_new']
                    })
                
                # 使用 RewardCalculator 的方法计算 reward
                r_margin = reward_calc._calculate_margin_reward(labeling_results)
                r_len = reward_calc._calculate_length_penalty(traj.actions)
                
                # 计算R_global
                r_global = r_margin + self.config.dpo_training.reward_lambda * r_len
                
                # 保存reward
                traj.reward = r_global
                traj.reward_details = {
                    'R_margin': r_margin,
                    'R_len': r_len,
                    'R_global': r_global
                }
        
        return trajectory_data_list
    
    def _flatten_trajectories(
        self,
        trajectory_data_list: List
    ) -> Tuple[List[StepSample], List[StepSample]]:
        """
        展平轨迹为step-level samples
        
        Args:
            trajectory_data_list: 轨迹数据列表
        
        Returns:
            (classifier_samples, updater_samples)
        """
        from sequential_article_processor import StepSample
        
        classifier_samples = []
        updater_samples = []
        
        for article_data in trajectory_data_list:
            for traj_idx, traj in enumerate(article_data.trajectories):
                trajectory_id = f"{article_data.topic_key}_{article_data.article_id}_traj{traj_idx}"
                global_reward = traj.reward
                
                # 收集该轨迹中的所有actions
                for action in traj.actions:
                    sample = StepSample(
                        system=action.system,
                        state=action.prompt,
                        action=action.completion,
                        global_reward=global_reward,
                        trajectory_id=trajectory_id
                    )
                    
                    if action.system == 'classify_generator':
                        classifier_samples.append(sample)
                    elif action.system == 'updater':
                        updater_samples.append(sample)
        
        return classifier_samples, updater_samples
    
    def _train_labeling_model(
        self,
        all_labeling_requests: List[LabelingRequest],
        api_results: List[LabelingResult],
        iteration: int
    ):
        """
        训练本地标注模型（仅在iteration 0时）
        
        Args:
            all_labeling_requests: 所有标注请求
            api_results: API的标注结果
            iteration: 当前iteration编号
        """
        import json
        import random
        from summary_based_classifier.llm.prompts import PromptTemplates
        
        print(f"  开始训练本地标注模型...")
        
        # 1. 构建训练数据集
        print(f"    - 构建训练数据集...")
        training_samples = []
        
        for request, result in zip(all_labeling_requests, api_results):
            # 只使用成功解析的样本
            if not result.success:
                continue
            
            # 格式化prompt（使用标注系统的prompt模板）
            prompt = PromptTemplates.format_labeling_prompt(
                topic_name=request.topic_name,
                current_summary=request.current_summary,
                child_summaries=request.child_summaries,
                ground_truth_paths=request.ground_truth_paths
            )
            
            # 使用API的原始响应作为输出
            completion = result.raw_response
            
            training_samples.append({
                'prompt': prompt,
                'completion': completion
            })
        
        print(f"    - 收集了 {len(training_samples)} 个有效样本")
        
        # 2. 限制样本数量
        max_samples = self.config.labeling.train_labeling_max_samples
        if len(training_samples) > max_samples:
            print(f"    - 随机抽取 {max_samples} 个样本用于训练")
            random.seed(self.config.data_prepare.seed)
            training_samples = random.sample(training_samples, max_samples)
        
        # 3. 保存训练数据集
        dataset_dir = self.output_dir / 'labeling_model_training'
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_file = dataset_dir / f'iteration_{iteration}_labeling_train.jsonl'
        
        print(f"    - 保存训练数据到: {dataset_file}")
        with open(dataset_file, 'w', encoding='utf-8') as f:
            for sample in training_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"  ✓ 训练数据集已保存")
        print(f"    - 样本数: {len(training_samples)}")
        print(f"    - 文件: {dataset_file}")
        
        # 4. 训练模型（使用SFT）
        print(f"\n    - 开始SFT训练...")
        self._run_sft_training(
            dataset_file=dataset_file,
            output_dir=dataset_dir / 'checkpoints',
            iteration=iteration
        )
        
        print(f"  ✓ 标注模型训练完成")
    
    def _run_sft_training(
        self,
        dataset_file: Path,
        output_dir: Path,
        iteration: int
    ):
        """
        运行SFT训练
        
        Args:
            dataset_file: 训练数据文件路径
            output_dir: 模型输出目录
            iteration: 当前iteration编号
        """
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from trl import SFTTrainer, SFTConfig
        from peft import LoraConfig, TaskType
        from datasets import load_dataset
        
        print(f"      加载基础模型: {self.config.path.base_model}")
        
        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.path.base_model,
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Model
        model = AutoModelForCausalLM.from_pretrained(
            self.config.path.base_model,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )
        
        print(f"      加载训练数据集...")
        dataset = load_dataset('json', data_files=str(dataset_file), split='train')
        print(f"      - 训练集: {len(dataset)} 条")
        
        # LoRA配置
        peft_config = LoraConfig(
            r=self.config.training.lora_r,
            lora_alpha=self.config.training.lora_alpha,
            lora_dropout=self.config.training.lora_dropout,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 
                           'gate_proj', 'up_proj', 'down_proj'],
            bias='none',
            task_type=TaskType.CAUSAL_LM
        )
        
        # 训练参数
        output_dir.mkdir(parents=True, exist_ok=True)
        training_args = SFTConfig(
            output_dir=str(output_dir),
            num_train_epochs=self.config.training.num_epochs,
            per_device_train_batch_size=self.config.training.batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            learning_rate=self.config.training.learning_rate,
            warmup_ratio=self.config.training.warmup_ratio,
            logging_steps=self.config.training.logging_steps,
            save_steps=self.config.training.save_steps,
            save_total_limit=self.config.training.save_total_limit,
            bf16=self.config.training.bf16,
            gradient_checkpointing=self.config.training.gradient_checkpointing,
            dataloader_num_workers=0,  # 避免多进程资源泄漏
            report_to='none',
            max_length=self.config.training.max_length,
        )
        
        # SFTTrainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            peft_config=peft_config,
            processing_class=tokenizer,
        )
        
        # 训练
        print(f"      开始训练...")
        trainer.train()
        
        # 保存模型
        final_model_dir = output_dir / 'final_model'
        print(f"      保存模型到: {final_model_dir}")
        trainer.model.save_pretrained(str(final_model_dir))
        tokenizer.save_pretrained(str(final_model_dir))
        
        print(f"      ✓ 训练完成")
        
        # 清理GPU内存
        del model
        del trainer
        torch.cuda.empty_cache()
    
    def _evaluate_local_labeling(
        self,
        all_labeling_requests: List[LabelingRequest],
        local_results: List[LabelingResult],
        iteration: int,
        num_samples: int = 100
    ):
        """
        评估本地标注模型的效果，随机抽取样本用API重新标注并对比
        
        Args:
            all_labeling_requests: 所有标注请求
            local_results: 本地模型的标注结果
            iteration: 当前iteration编号
            num_samples: 抽取的样本数量
        """
        import random
        import json
        
        print(f"  评估本地标注模型（抽取{num_samples}个样本与API对比）...")
        
        # 只抽取成功解析的样本进行对比
        success_indices = [i for i, res in enumerate(local_results) if res.success]
        
        if len(success_indices) == 0:
            print(f"  ⚠ 没有成功的本地标注样本，跳过评估")
            return
        
        # 随机抽取
        num_to_sample = min(num_samples, len(success_indices))
        sampled_indices = random.sample(success_indices, num_to_sample)
        
        print(f"    - 本地成功样本数: {len(success_indices)}")
        print(f"    - 抽取样本数: {num_to_sample}")
        
        # 获取对应的请求和本地结果
        sampled_requests = [all_labeling_requests[i] for i in sampled_indices]
        sampled_local_results = [local_results[i] for i in sampled_indices]
        
        # 使用API重新标注
        print(f"    - 使用API重新标注...")
        api_labeler = BatchLabeler(self.config, mode='api')
        api_results = api_labeler.label_batch(sampled_requests)
        api_labeler.cleanup()
        
        # 对比结果
        agree_exceed = 0
        agree_overlap = 0
        agree_correct = 0
        agree_need_new = 0
        fully_agree = 0
        
        disagreements = []
        
        for i, (local_res, api_res) in enumerate(zip(sampled_local_results, api_results)):
            if not api_res.success:
                continue  # API也失败了，跳过
            
            # 对比各个字段
            local_exceed = set(local_res.exceed_parent or [])
            api_exceed = set(api_res.exceed_parent or [])
            exceed_match = local_exceed == api_exceed
            
            local_overlap = set(tuple(sorted(pair)) for pair in (local_res.overlapping_pairs or []))
            api_overlap = set(tuple(sorted(pair)) for pair in (api_res.overlapping_pairs or []))
            overlap_match = local_overlap == api_overlap
            
            local_correct = set(local_res.correct_indices or [])
            api_correct = set(api_res.correct_indices or [])
            correct_match = local_correct == api_correct
            
            need_new_match = local_res.need_new == api_res.need_new
            
            if exceed_match:
                agree_exceed += 1
            if overlap_match:
                agree_overlap += 1
            if correct_match:
                agree_correct += 1
            if need_new_match:
                agree_need_new += 1
            
            if exceed_match and overlap_match and correct_match and need_new_match:
                fully_agree += 1
            else:
                # 记录不一致的案例
                disagreements.append({
                    'sample_idx': sampled_indices[i],
                    'num_children': len(sampled_requests[i].child_summaries),
                    'local': local_res.parsed_output,
                    'api': api_res.parsed_output,
                    'exceed_match': exceed_match,
                    'overlap_match': overlap_match,
                    'correct_match': correct_match,
                    'need_new_match': need_new_match
                })
        
        # 统计
        total = len([res for res in api_results if res.success])
        
        print(f"\n  评估结果:")
        print(f"    - 对比样本数: {total}")
        print(f"    - 完全一致率: {fully_agree}/{total} = {fully_agree/total*100:.2f}%")
        print(f"    - EXCEED_PARENT 一致率: {agree_exceed}/{total} = {agree_exceed/total*100:.2f}%")
        print(f"    - OVERLAPPING_PAIRS 一致率: {agree_overlap}/{total} = {agree_overlap/total*100:.2f}%")
        print(f"    - CORRECT_INDICES 一致率: {agree_correct}/{total} = {agree_correct/total*100:.2f}%")
        print(f"    - NEED_NEW 一致率: {agree_need_new}/{total} = {agree_need_new/total*100:.2f}%")
        
        # 保存评估结果
        eval_output = self.output_dir / 'labeling_evaluation'
        eval_output.mkdir(parents=True, exist_ok=True)
        
        eval_file = eval_output / f'iteration_{iteration}_evaluation.json'
        eval_data = {
            'iteration': iteration,
            'total_samples': total,
            'fully_agree': fully_agree,
            'fully_agree_rate': fully_agree / total if total > 0 else 0,
            'agree_exceed': agree_exceed,
            'agree_exceed_rate': agree_exceed / total if total > 0 else 0,
            'agree_overlap': agree_overlap,
            'agree_overlap_rate': agree_overlap / total if total > 0 else 0,
            'agree_correct': agree_correct,
            'agree_correct_rate': agree_correct / total if total > 0 else 0,
            'agree_need_new': agree_need_new,
            'agree_need_new_rate': agree_need_new / total if total > 0 else 0,
            'disagreements': disagreements[:10]  # 只保存前10个不一致案例
        }
        
        with open(eval_file, 'w', encoding='utf-8') as f:
            json.dump(eval_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n  ✓ 评估结果已保存到: {eval_file}")
        
        # 如果不一致率过高，给出警告
        if fully_agree / total < 0.8:
            print(f"\n  ⚠️ 警告：完全一致率低于80%，建议:")
            print(f"     1. 检查本地模型是否正确加载")
            print(f"     2. 检查prompt格式是否适合本地模型")
            print(f"     3. 考虑使用API模式或重新训练本地模型")
    
    def sample_articles(self, num_articles: int) -> List[Dict]:
        """从训练集中采样文章"""
        train_topics = self.dataset_split.get('train', {})
        all_articles = []
        
        for topic_key, ref_ids in train_topics.items():
            if topic_key not in self.references_data:
                continue
            
            topic_data = self.references_data[topic_key]
            topic_name = topic_data.get('topic', topic_key)
            
            for ref_id in ref_ids:
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
        
        # 随机采样
        random.shuffle(all_articles)
        return all_articles[:num_articles]
    
    def sample_trajectories(
        self,
        num_articles: int,
        classifier: ClassifyGenerator,
        updater: Updater,
        iteration: int
    ) -> Dict[str, List[Tuple[str, List[Trajectory]]]]:
        """
        为文章采样轨迹（使用并行处理器）
        
        Returns:
            {topic_key: [(article_id, [trajectories])]}
        """
        # 创建或加载topic state manager
        state_dir = self.output_dir / 'topic_states' / f'iter{iteration}'
        state_manager = TopicStateManager(save_dir=state_dir)
        
        # 加载参考数据（从单个references_file）
        if self.references_data is None:
            print(f"\n  加载references数据...")
            with open(self.config.path.references_file, 'r', encoding='utf-8') as f:
                self.references_data = json.load(f)
        
        # 加载结构数据（获取topic名称）
        if self.structures_data is None:
            print(f"  加载structures数据...")
            with open(self.config.path.structures_file, 'r', encoding='utf-8') as f:
                self.structures_data = json.load(f)
        
        # 加载dataset split（获取train topics）
        if self.dataset_split is None:
            split_file = self.output_dir / 'dataset_split.json'
            with open(split_file, 'r', encoding='utf-8') as f:
                self.dataset_split = json.load(f)['dataset_split']
        
        train_topics = self.dataset_split.get('train', {})
        
        # 初始化所有topics
        print(f"\n  初始化topics (仅train集)...")
        for topic_key in train_topics.keys():
            # 检查topic是否在references中
            if topic_key not in self.references_data:
                continue
            
            topic_ref_data = self.references_data[topic_key]
            
            # 从structures获取topic名称
            topic_name = topic_key
            if topic_key in self.structures_data:
                topic_name = self.structures_data[topic_key].get('topic', topic_key)
            
            # 获取train集的文章列表
            train_article_ids = train_topics[topic_key]
            
            if not train_article_ids:
                continue
            
            # load_if_exists=True 表示继承上次采样的状态（树结构和文章位置）
            # 如果是iter0且第一次运行，会创建新状态；否则加载已有状态继续
            state_manager.initialize_topic(
                topic_key=topic_key,
                topic_name=topic_name,
                article_ids=train_article_ids,
                load_if_exists=True  # 总是尝试加载，如果没有会创建新的
            )
        
        print(f"  ✓ {len(state_manager.states)} topics初始化完成")
        
        # 创建并行处理器
        processor = ParallelTrajectoryProcessor(
            state_manager=state_manager,
            classifier=classifier,
            updater=updater,
            max_depth=self.config.inference.max_depth,
            num_samples_per_prompt=self.config.dpo_training.sampling_top_k,
            batch_size=self.config.dpo_training.sampling_batch_size,
            timeout_seconds=self.config.dpo_training.sampling_timeout_seconds
        )
        
        # 并行采样
        all_results = processor.process_articles(
            references_data=self.references_data,
            num_articles=num_articles
        )
        
        # 保存states（跨迭代继承）
        print(f"\n  保存topic states...")
        state_manager.save_all()
        print(f"  ✓ 状态已保存到: {state_dir}")
        
        return all_results
    
    def calculate_rewards(
        self,
        trajectory_data: List[Dict],
        classifier: ClassifyGenerator
    ) -> List[Dict]:
        """计算轨迹rewards"""
        print("\n计算轨迹rewards...")
        
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
            topic_key = data['topic_key']
            article_id = data['article_id']
            trajectories = data['trajectories']
            
            # 从references_data加载文章信息
            if topic_key in self.references_data:
                topic_ref_data = self.references_data[topic_key]
                references = topic_ref_data.get('references', {})
                
                if article_id in references:
                    article_info = references[article_id]
                    article_content = article_info.get('content', '')
                    ground_truth_paths = article_info.get('paths', [])
                else:
                    article_content = ""
                    ground_truth_paths = []
            else:
                article_content = ""
                ground_truth_paths = []
            
            # 从structures获取topic名称
            topic_name = topic_key
            if topic_key in self.structures_data:
                topic_name = self.structures_data[topic_key].get('topic', topic_key)
            
            reward_calc = RewardCalculator(
                classifier=classifier,
                labeling_api_config=labeling_api_config,
                topic_name=topic_name,
                beta=self.config.dpo_training.reward_beta,
                gamma=self.config.dpo_training.reward_gamma,
                lambda_=self.config.dpo_training.reward_lambda,
                margin_penalty=self.config.dpo_training.reward_margin_penalty
            )
            
            if len(trajectories) > 1:
                rewards = reward_calc.calculate_rewards_batch(
                    trajectories=trajectories,
                    article_content=article_content,
                    ground_truth_paths=ground_truth_paths
                )
            else:
                rewards = {'R_margin': 0, 'R_len': 0, 'R_global': 0}
            
            data['rewards'] = rewards
            data['article'] = {
                'topic_name': topic_name,
                'content': article_content,
                'paths': ground_truth_paths,
                'ref_id': article_id
            }
        
        return trajectory_data
    
    def _save_step_dataset(
        self,
        classifier_samples: List[StepSample],
        updater_samples: List[StepSample],
        iteration: int
    ):
        """保存step-level训练数据集（包含advantage和weight）"""
        import json
        
        # 创建数据集目录
        dataset_dir = self.output_dir / 'iwsft_datasets' / f'iter{iteration}'
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存分类系统数据集
        if classifier_samples:
            classifier_data = []
            for sample in classifier_samples:
                classifier_data.append({
                    'system': sample.system,
                    'state': sample.state,
                    'action': sample.action,
                    'global_reward': sample.global_reward,
                    'trajectory_id': sample.trajectory_id,
                    'advantage': sample.advantage,
                    'weight': sample.weight
                })
            
            classifier_file = dataset_dir / 'classifier_samples.json'
            with open(classifier_file, 'w', encoding='utf-8') as f:
                json.dump(classifier_data, f, ensure_ascii=False, indent=2)
            print(f"  ✓ 分类系统数据集已保存: {classifier_file}")
            print(f"    - 样本数量: {len(classifier_data)}")
            print(f"    - 平均weight: {np.mean([s.weight for s in classifier_samples]):.3f}")
        
        # 保存更新系统数据集
        if updater_samples:
            updater_data = []
            for sample in updater_samples:
                updater_data.append({
                    'system': sample.system,
                    'state': sample.state,
                    'action': sample.action,
                    'global_reward': sample.global_reward,
                    'trajectory_id': sample.trajectory_id,
                    'advantage': sample.advantage,
                    'weight': sample.weight
                })
            
            updater_file = dataset_dir / 'updater_samples.json'
            with open(updater_file, 'w', encoding='utf-8') as f:
                json.dump(updater_data, f, ensure_ascii=False, indent=2)
            print(f"  ✓ 更新系统数据集已保存: {updater_file}")
            print(f"    - 样本数量: {len(updater_data)}")
            print(f"    - 平均weight: {np.mean([s.weight for s in updater_samples]):.3f}")
        
        # 保存统计信息
        stats = {
            'iteration': iteration,
            'classifier_samples_count': len(classifier_samples),
            'updater_samples_count': len(updater_samples),
            'classifier_avg_weight': float(np.mean([s.weight for s in classifier_samples])) if classifier_samples else 0,
            'updater_avg_weight': float(np.mean([s.weight for s in updater_samples])) if updater_samples else 0,
            'timestamp': str(Path(dataset_dir).stat().st_mtime)
        }
        
        stats_file = dataset_dir / 'stats.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"  ✓ 统计信息已保存: {stats_file}")
    
    def train_weighted_sft_model(
        self,
        model_type: str,
        samples: List[StepSample],
        base_model_path: str,
        output_dir: str,
        iteration: int
    ):
        """训练加权SFT模型（Advantage-weighted SFT，使用TRL）"""
        print(f"\n训练{model_type}模型（加权SFT with TRL）...")
        print(f"  - 样本数量: {len(samples)}")
        print(f"  - 基础模型: {base_model_path}")
        print(f"  - 输出目录: {output_dir}")
        
        if len(samples) == 0:
            print(f"  ⚠️ 没有样本，跳过训练")
            return
        
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
        from peft import LoraConfig, TaskType, get_peft_model
        from datasets import Dataset
        import torch
        
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 准备数据集 - 正确处理prompt和completion的分离
        print(f"  - 预处理数据集...")
        dataset_dict = []
        for sample in samples:
            # 分别tokenize prompt和completion
            prompt_tokens = tokenizer(
                sample.state,  # prompt
                add_special_tokens=True,
                truncation=False,
                return_tensors=None
            )
            completion_tokens = tokenizer(
                sample.action,  # completion
                add_special_tokens=False,  # completion不需要特殊token
                truncation=False,
                return_tensors=None
            )
            
            # 拼接
            input_ids = prompt_tokens['input_ids'] + completion_tokens['input_ids']
            attention_mask = prompt_tokens['attention_mask'] + completion_tokens['attention_mask']
            
            # 截断（如果太长）
            if len(input_ids) > self.config.dpo_training.max_length:
                input_ids = input_ids[:self.config.dpo_training.max_length]
                attention_mask = attention_mask[:self.config.dpo_training.max_length]
            
            # 创建labels：只对completion部分计算loss
            prompt_len = len(prompt_tokens['input_ids'])
            labels = [-100] * prompt_len + input_ids[prompt_len:]  # prompt部分mask掉
            
            dataset_dict.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,  # 提前创建labels
                'weight': float(sample.weight),
                'advantage': float(sample.advantage)  # 用于调试
            })
        
        print(f"  - 数据集样本数: {len(dataset_dict)}")
        print(f"  - Weight统计: min={min(s['weight'] for s in dataset_dict):.3f}, "
              f"max={max(s['weight'] for s in dataset_dict):.3f}, "
              f"mean={np.mean([s['weight'] for s in dataset_dict]):.3f}")
        
        # 创建dataset（已经tokenized）
        train_dataset = Dataset.from_list(dataset_dict)
        
        # 自定义DataCollator - 处理padding，保留weight字段
        class WeightedDataCollator:
            """处理padding，保留weight字段，labels已在数据集中"""
            def __init__(self, tokenizer, pad_to_multiple_of=None):
                self.tokenizer = tokenizer
                self.pad_to_multiple_of = pad_to_multiple_of
            
            def __call__(self, features):
                # 提取字段
                input_ids = [f["input_ids"] for f in features]
                attention_mask = [f["attention_mask"] for f in features]
                labels = [f["labels"] for f in features]
                weights = [f["weight"] for f in features]
                
                # 计算batch中最大长度
                max_length = max(len(ids) for ids in input_ids)
                if self.pad_to_multiple_of is not None:
                    max_length = ((max_length + self.pad_to_multiple_of - 1) 
                                  // self.pad_to_multiple_of * self.pad_to_multiple_of)
                
                # Padding
                batch_input_ids = []
                batch_attention_mask = []
                batch_labels = []
                
                for ids, mask, label in zip(input_ids, attention_mask, labels):
                    padding_length = max_length - len(ids)
                    
                    # Pad input_ids
                    padded_ids = ids + [self.tokenizer.pad_token_id] * padding_length
                    batch_input_ids.append(padded_ids)
                    
                    # Pad attention_mask
                    padded_mask = mask + [0] * padding_length
                    batch_attention_mask.append(padded_mask)
                    
                    # Pad labels（padding位置为-100）
                    padded_labels = label + [-100] * padding_length
                    batch_labels.append(padded_labels)
                
                # 转为tensor
                batch = {
                    "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
                    "labels": torch.tensor(batch_labels, dtype=torch.long),
                    "weight": torch.tensor(weights, dtype=torch.float32)
                }
                
                return batch
        
        data_collator = WeightedDataCollator(tokenizer)
        
        # 在加载新模型前，彻底清理GPU状态
        print(f"  - 清理GPU状态...")
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # 打印当前显存使用
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                print(f"    GPU {i}: 已分配 {allocated:.2f}GB, 已保留 {reserved:.2f}GB")
        
        # 加载模型（使用device_map="auto"让模型自动分配到多个GPU）
        print(f"  - 加载模型: {base_model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if self.config.dpo_training.bf16 else torch.float32,
            device_map="auto"
        )
        
        # 关键：使用gradient checkpointing + PEFT时必须先启用input require grads，再get_peft_model
        if self.config.dpo_training.gradient_checkpointing:
            model.enable_input_require_grads()
        
        # 配置LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.training.lora_r,
            lora_alpha=self.config.training.lora_alpha,
            lora_dropout=self.config.training.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        model = get_peft_model(model, peft_config)
        
        model.print_trainable_parameters()
        
        # 训练配置
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.dpo_training.num_epochs_per_iteration,
            per_device_train_batch_size=self.config.dpo_training.batch_size,
            gradient_accumulation_steps=self.config.dpo_training.gradient_accumulation_steps,
            learning_rate=self.config.dpo_training.learning_rate,
            warmup_ratio=self.config.dpo_training.warmup_ratio,
            logging_steps=self.config.dpo_training.logging_steps,
            save_steps=self.config.dpo_training.save_steps,
            bf16=self.config.dpo_training.bf16,
            gradient_checkpointing=self.config.dpo_training.gradient_checkpointing,
            save_total_limit=2,
            remove_unused_columns=False,
            report_to=[]
        )
        
        # 自定义Trainer，重写损失函数以加入weight
        class WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                """重写损失函数，加入advantage weight"""
                # 取出weight
                weights = inputs.pop("weight", None)
                
                # 标准的forward（这会自动计算loss）
                outputs = model(**inputs)
                
                # 如果有weight，需要重新计算weighted loss
                if weights is not None:
                    try:
                        logits = outputs.logits
                        labels = inputs["labels"]
                        
                        # shift for causal LM
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = labels[..., 1:].contiguous()
                        
                        # 确保labels是long类型
                        shift_labels = shift_labels.long()
                        
                        # token-level CE (no reduction)
                        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
                        
                        # 展平并计算loss
                        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
                        flat_labels = shift_labels.view(-1)
                        
                        per_token_loss = loss_fct(flat_logits, flat_labels)
                        per_token_loss = per_token_loss.view(shift_labels.size(0), shift_labels.size(1))
                        
                        # sequence-level loss
                        mask = (shift_labels != -100).float()
                        per_sample_loss = (per_token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                        
                        # IW-SFT weighting
                        weights = weights.to(per_sample_loss.device)
                        loss = (per_sample_loss * weights).mean()
                    except RuntimeError as e:
                        print(f"\n[ERROR] Loss计算失败: {e}")
                        print(f"  logits shape: {logits.shape}")
                        print(f"  labels shape: {labels.shape}")
                        print(f"  weights shape: {weights.shape}")
                        if torch.cuda.is_available():
                            print(f"  GPU显存: {torch.cuda.memory_allocated() / 1024**3:.2f}GB / {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")
                        raise
                else:
                    # 没有weight就用标准loss
                    loss = outputs.loss
                
                return (loss, outputs) if return_outputs else loss
        
        # 创建Trainer
        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator
        )
        
        # 训练前再次同步GPU
        print(f"\n  准备开始训练...")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        # 训练
        print(f"  开始训练...")
        trainer.train()
        
        # 保存模型
        print(f"\n  保存{model_type}模型...")
        
        # 保存LoRA adapter
        adapter_path = Path(output_dir) / 'adapter'
        print(f"    - 保存LoRA adapter到: {adapter_path}")
        trainer.model.save_pretrained(str(adapter_path))
        tokenizer.save_pretrained(str(adapter_path))
        
        # 合并并保存完整模型（vLLM需要完整模型）
        final_path = Path(output_dir) / 'final'
        print(f"    - 合并并保存完整模型...")
        print(f"      Base模型路径: {base_model_path}")
        
        # Step 1: 加载干净的base模型（使用函数参数传入的base_model_path）
        print(f"      Step 1/5: 加载干净的base模型...")
        from transformers import AutoModelForCausalLM
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype="auto",
            trust_remote_code=True,
        )
        
        # Step 3: 加载LoRA adapter
        print(f"      Step 2/5: 加载LoRA adapter...")
        from peft import PeftModel
        adapter_path = Path(output_dir) / 'adapter'
        peft_model = PeftModel.from_pretrained(base_model, str(adapter_path))
        
        # Step 4: 合并LoRA（此后是纯transformers模型）
        print(f"      Step 3/5: 合并LoRA...")
        merged_model = peft_model.merge_and_unload()
        
        # Step 5: 对齐embedding vocab（关键步骤！在合并后进行）
        print(f"      Step 4/5: 对齐embedding层...")
        num_added = len(tokenizer.get_added_vocab())
        true_vocab_size = tokenizer.vocab_size + num_added
        print(f"        - 原始embedding size: {merged_model.get_input_embeddings().weight.shape[0]}")
        print(f"        - Tokenizer vocab size: {true_vocab_size}")
        if merged_model.get_input_embeddings().weight.shape[0] != true_vocab_size:
            merged_model.resize_token_embeddings(true_vocab_size)
            print(f"        - 调整后embedding size: {merged_model.get_input_embeddings().weight.shape[0]}")
            print("        ✓ Embedding层对齐完成")
        else:
            print("        ✓ Embedding层已对齐")
        merged_model.config.vocab_size = true_vocab_size
        
        # Step 6: 保存最终模型
        print(f"      Step 5/5: 保存最终模型...")
        merged_model.save_pretrained(str(final_path), safe_serialization=True)
        tokenizer.save_pretrained(str(final_path))
        
        print(f"  ✓ {model_type}模型训练和保存完成")
        
        # 显式清理显存
        print(f"  清理显存...")
        import gc
        
        # 先同步所有CUDA操作
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # 删除所有模型对象
        del trainer
        del model
        del base_model
        del peft_model
        del merged_model
        
        # 垃圾回收
        gc.collect()
        
        # 清理CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            # 打印清理后的显存
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                print(f"    GPU {i}: 已分配 {allocated:.2f}GB, 已保留 {reserved:.2f}GB")
        
        print(f"  ✓ 显存清理完成")
    
    def run_iteration(
        self,
        iteration: int,
        num_articles: int,
        classify_generator_model: str,
        updater_model: str
    ):
        """运行一次IW SFT迭代"""
        print("\n" + "="*80)
        print(f"IW SFT迭代: iter_{iteration} ({iteration + 1}/{self.config.dpo_training.num_iterations})")
        print(f"采样文章数: {num_articles}")
        print("="*80)
        
        # 检查是否跳过采样阶段（直接检查轨迹文件是否已存在）
        skip_sampling = self.config.dpo_training.skip_trajectory_sampling
        trajectory_file = self.output_dir / 'train_trajectories' / f'iteration_{iteration}_trajectories.pkl.gz'
        
        if skip_sampling and trajectory_file.exists():
            # 跳过采样，直接使用已有的轨迹文件
            print(f"\n⚠️ 跳过采样阶段，从已有文件加载: {trajectory_file}")
            trajectory_file = str(trajectory_file)
        else:
            # 运行采样阶段
            trajectory_file = self.run_sampling_phase(
                iteration=iteration,
                num_articles=num_articles,
                classify_generator_model=classify_generator_model,
                updater_model=updater_model
            )
        
        # 运行标注阶段
        classifier_samples, updater_samples = self.run_labeling_phase(
            iteration=iteration,
            trajectory_file=trajectory_file
        )
        
        # 3. 训练分类生成模型（加权SFT）
        print(f"\n步骤3: 训练分类生成模型（加权SFT）...")
        classifier_output_dir = self.models_dir / f'classify_generator_iwsft_iter{iteration}'
        self.train_weighted_sft_model(
            model_type='分类生成',
            samples=classifier_samples,
            base_model_path=classify_generator_model,
            output_dir=str(classifier_output_dir),
            iteration=iteration
        )
        
        # 4. 训练总结更新模型（加权SFT）
        print(f"\n步骤4: 训练总结更新模型（加权SFT）...")
        updater_output_dir = self.models_dir / f'updater_iwsft_iter{iteration}'
        self.train_weighted_sft_model(
            model_type='总结更新',
            samples=updater_samples,
            base_model_path=updater_model,
            output_dir=str(updater_output_dir),
            iteration=iteration
        )
        
        # 5. 返回新模型路径
        new_classifier_path = str(classifier_output_dir / 'final')
        new_updater_path = str(updater_output_dir / 'final')
        
        return new_classifier_path, new_updater_path
    
    def run_full_pipeline(self, start_iteration: int = None):
        """运行完整的IW SFT训练pipeline"""
        # 使用config中的start_iteration，如果参数没传的话
        if start_iteration is None:
            start_iteration = self.config.dpo_training.start_iteration
        
        print("\n" + "="*80)
        print("IW SFT训练完整Pipeline（Advantage-weighted SFT）")
        print("="*80)
        print(f"迭代次数: {self.config.dpo_training.num_iterations}")
        print(f"采样批次大小: {self.config.dpo_training.sampling_batch_sizes}")
        print(f"开始迭代: start_iteration={start_iteration}")
        print(f"  → 将从 iter_{start_iteration} 开始训练")
        if start_iteration > 0:
            print(f"  → 将加载 iter_{start_iteration-1} 的模型作为起点")
        print(f"使用已有数据集: {self.config.dpo_training.use_existing_dataset}")
        
        # 初始模型路径
        if start_iteration == 0:
            # 使用SFT训练的模型作为起点
            classify_generator_model = str(self.models_dir / 'classify_generator_sft' / 'final_model')
            updater_model = str(self.models_dir / 'updater_sft' / 'final_model')
            
            # 检查模型是否存在
            if not Path(classify_generator_model).exists():
                print(f"\n错误: 找不到SFT训练的分类生成模型: {classify_generator_model}")
                print("请先运行SFT训练")
                return
            
            if not Path(updater_model).exists():
                print(f"\n错误: 找不到SFT训练的总结更新模型: {updater_model}")
                print("请先运行SFT训练")
                return
        else:
            # 从checkpoint恢复
            prev_iteration = start_iteration - 1
            classify_generator_model = str(self.models_dir / f'classify_generator_iwsft_iter{prev_iteration}' / 'final')
            updater_model = str(self.models_dir / f'updater_iwsft_iter{prev_iteration}' / 'final')
            
            # 检查模型是否存在
            if not Path(classify_generator_model).exists():
                print(f"\n错误: 找不到iter_{prev_iteration}的分类生成模型: {classify_generator_model}")
                print(f"请确保iter_{prev_iteration}训练已完成")
                return
            
            if not Path(updater_model).exists():
                print(f"\n错误: 找不到iter_{prev_iteration}的总结更新模型: {updater_model}")
                print(f"请确保iter_{prev_iteration}训练已完成")
                return
            
            print(f"\n从迭代{prev_iteration}恢复:")
            print(f"  - 分类生成模型: {classify_generator_model}")
            print(f"  - 总结更新模型: {updater_model}")
        
        # 运行迭代
        for iteration in range(start_iteration, self.config.dpo_training.num_iterations):
            # 动态计算文章数量：早期小，中期中等，后期大
            total_iterations = self.config.dpo_training.num_iterations
            early_size = self.config.dpo_training.sampling_batch_sizes[0]
            mid_size = self.config.dpo_training.sampling_batch_sizes[1]
            late_size = self.config.dpo_training.sampling_batch_sizes[2]
            
            if iteration < total_iterations / 3:
                num_articles = early_size
            elif iteration < total_iterations * 2 / 3:
                num_articles = mid_size
            else:
                num_articles = late_size
            
            classify_generator_model, updater_model = self.run_iteration(
                iteration=iteration,
                num_articles=num_articles,
                classify_generator_model=classify_generator_model,
                updater_model=updater_model
            )
            
            print(f"\n✓ 迭代{iteration + 1}完成")
            print(f"  - 新分类生成模型: {classify_generator_model}")
            print(f"  - 新总结更新模型: {updater_model}")
            
            # 迭代结束后进行全面清理，避免资源累积
            print(f"\n  迭代间资源清理...")
            import gc
            import torch
            import time
            
            # 同步所有CUDA操作
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # 垃圾回收
            gc.collect()
            
            # 清理CUDA缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            # 短暂等待，确保资源完全释放
            time.sleep(2)
            
            print(f"  ✓ 迭代间资源清理完成")
        
        # 保存最终模型
        print("\n" + "="*80)
        print("保存最终模型...")
        print("="*80)
        
        final_classifier_dir = self.models_dir / 'classify_generator_iwsft_final'
        final_updater_dir = self.models_dir / 'updater_iwsft_final'
        
        # 复制最后一次迭代的模型
        import shutil
        if Path(classify_generator_model).exists():
            shutil.copytree(classify_generator_model, final_classifier_dir, dirs_exist_ok=True)
            print(f"  ✓ 分类生成最终模型: {final_classifier_dir}")
        
        if Path(updater_model).exists():
            shutil.copytree(updater_model, final_updater_dir, dirs_exist_ok=True)
            print(f"  ✓ 总结更新最终模型: {final_updater_dir}")
        
        print("\n" + "="*80)
        print("DPO训练完整Pipeline完成！")
        print("="*80)


def main():

    parser = argparse.ArgumentParser(description='IW SFT迭代训练（Advantage-weighted SFT）')
    parser.add_argument('--config', type=str, default='./configs/default.json')
    parser.add_argument('--num_iterations', type=int, help='迭代次数（覆盖配置）')
    parser.add_argument('--start_iteration', type=int, help='开始迭代（覆盖配置，断点续训）')
    parser.add_argument('--use_existing_dataset', action='store_true', help='使用已有数据集（覆盖配置）')
    
    args = parser.parse_args()
    
    # 加载配置
    config = SummaryBasedConfig.from_json(args.config)
    
    # 覆盖配置（如果提供了命令行参数）
    if args.num_iterations:
        config.dpo_training.num_iterations = args.num_iterations
    if args.start_iteration is not None:
        config.dpo_training.start_iteration = args.start_iteration
    if args.use_existing_dataset:
        config.dpo_training.use_existing_dataset = True
    
    # 创建pipeline
    pipeline = IWSFTTrainingPipeline(config)
    pipeline.load_data()
    
    # 运行（不需要再传start_iteration，从config读取）
    pipeline.run_full_pipeline()


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    main()

