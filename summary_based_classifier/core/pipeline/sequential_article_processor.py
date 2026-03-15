"""
顺序文章处理器
每个topic线程顺序处理文章：采样轨迹 → 使用主轨迹更新树 → 保存轨迹数据
使用multiprocessing.Queue+批量推理机制

注意：reward计算已从此模块移除，将在标注阶段统一进行
"""
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json
import threading
import multiprocessing as mp
from queue import Queue
from tqdm import tqdm
from summary_based_classifier.core.trajectory.trajectory_sampler import TreeNode, Trajectory
from summary_based_classifier.core.pipeline.builder import TreeBuilder
from summary_based_classifier.core.topic_state import TopicState
from summary_based_classifier.models.model_workers import ClassifierWorker, UpdaterWorker
import time


@dataclass
class ArticleTrajectoryData:
    """单篇文章的轨迹数据"""
    topic_key: str
    topic_name: str
    article_id: str
    article_content: str
    ground_truth_paths: List[str]
    trajectories: List[Trajectory]  # 所有采样的轨迹
    re_classification_records: List[List[Dict]]  # 每个轨迹对应的重新分类records


@dataclass
class StepSample:
    """单步样本 (state, action, reward)"""
    system: str  # 'classify_generator' or 'updater'
    state: str  # prompt (完整的输入)
    action: str  # completion (模型输出)
    global_reward: float  # 该轨迹的全局reward
    trajectory_id: str  # 轨迹ID（用于调试）
    
    # 后续计算的字段
    advantage: float = 0.0  # A(s,a) = R_global - b(s)
    weight: float = 1.0  # w(s,a) = exp(β * clip(A(s,a)))


class GlobalCounter:
    """全局文章计数器（线程安全）"""
    def __init__(self, target_count: int):
        self.target_count = target_count
        self.current_count = 0
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.pbar = None
    
    def set_pbar(self, pbar: tqdm):
        """设置进度条"""
        with self.lock:
            self.pbar = pbar
    
    def increment(self) -> int:
        """增加计数，返回新的计数"""
        with self.lock:
            self.current_count += 1
            if self.pbar:
                self.pbar.update(1)
            if self.current_count >= self.target_count:
                self.stop_event.set()
            return self.current_count
    
    def should_stop(self) -> bool:
        """是否应该停止"""
        return self.stop_event.is_set()
    
    def get_count(self) -> int:
        """获取当前计数"""
        with self.lock:
            return self.current_count


class SequentialArticleProcessor:
    """顺序文章处理器（使用Prompt池+批量推理）"""
    
    def __init__(
        self,
        classifier_model_path: str,
        updater_model_path: str,
        topic_states: Dict[str, TopicState],
        sampling_num: int,
        top_k: int,
        max_depth: int,
        config
    ):
        """
        Args:
            classifier_model_path: 分类模型路径
            updater_model_path: 更新模型路径
            topic_states: topic状态字典
            sampling_num: 每次采样的结果数
            top_k: 从采样结果中选择前k个
            max_depth: 最大深度
            config: 配置对象
        """
        self.classifier_model_path = classifier_model_path
        self.updater_model_path = updater_model_path
        self.topic_states = topic_states
        self.sampling_num = sampling_num
        self.top_k = top_k
        self.max_depth = max_depth
        self.config = config
        
        # Prompt池和结果池（使用multiprocessing.Manager创建，以便进程间共享）
        batch_size = config.dpo_training.inference_batch_size
        timeout = config.dpo_training.inference_wait_timeout
        
        # 使用multiprocessing.Queue代替线程池（进程安全）
        self.classifier_prompt_queue = mp.Queue()
        self.classifier_result_queue = mp.Queue()
        self.updater_prompt_queue = mp.Queue()
        self.updater_result_queue = mp.Queue()
        
        # Worker进程（传入模型路径和Queue，将在进程内加载）
        self.classifier_worker = ClassifierWorker(
            model_path=classifier_model_path,
            prompt_queue=self.classifier_prompt_queue,
            result_queue=self.classifier_result_queue,
            batch_size=batch_size,
            timeout=timeout,
            gpu_id=0,
            max_model_len=config.inference.max_model_len,
            gpu_memory_utilization=config.inference.gpu_memory_utilization,
            temperature=0.7
        )
        
        self.updater_worker = UpdaterWorker(
            model_path=updater_model_path,
            prompt_queue=self.updater_prompt_queue,
            result_queue=self.updater_result_queue,
            batch_size=batch_size,
            timeout=timeout,
            gpu_id=1,
            max_model_len=config.inference.max_model_len,
            gpu_memory_utilization=config.inference.gpu_memory_utilization,
            temperature=0.5
        )
    
    def process_articles(
        self,
        target_article_count: int,
        references_data: Dict,
        summaries_data: Dict
    ) -> List[ArticleTrajectoryData]:
        """
        并行处理多个topic的文章，每个topic内顺序处理
        使用Prompt池+批量推理机制
        
        Args:
            target_article_count: 目标文章数量
            references_data: 参考数据
            summaries_data: Summary数据
            
        Returns:
            ArticleTrajectoryData列表（包含完整的轨迹信息）
        """
        print(f"\n{'='*60}")
        print(f"开始采样轨迹并收集step-level样本")
        print(f"{'='*60}")
        print(f"  目标文章数: {target_article_count}")
        print(f"  Topic数量: {len(self.topic_states)}")
        print(f"  批量推理配置: batch_size={self.config.dpo_training.inference_batch_size}, timeout={self.config.dpo_training.inference_wait_timeout}s\n")
        
        # 启动Worker线程
        self.classifier_worker.start()
        self.updater_worker.start()
        
        # 全局计数器
        global_counter = GlobalCounter(target_article_count)
        
        # 创建进度条
        pbar = tqdm(total=target_article_count, desc="处理文章", unit="篇")
        global_counter.set_pbar(pbar)
        
        # 结果队列
        result_queue = Queue()
        
        # 创建topic线程
        threads = []
        for topic_key, topic_state in self.topic_states.items():
            thread = threading.Thread(
                target=self._process_topic,
                args=(topic_state, global_counter, result_queue, references_data, summaries_data),
                name=f"Topic-{topic_key}"
            )
            threads.append(thread)
            thread.start()

            if len(threads) >= 100:
                break
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 关闭进度条
        pbar.close()
        
        # 停止Worker进程并等待它们完全退出
        print(f"\n停止Worker进程...")
        self.classifier_worker.stop()
        self.updater_worker.stop()
        
        # 收集所有文章的轨迹数据
        all_trajectory_data = []
        
        while not result_queue.empty():
            trajectory_data = result_queue.get()
            all_trajectory_data.append(trajectory_data)
        
        # 清理multiprocessing Queue，避免资源泄漏
        print(f"清理资源...")
        for queue in [self.classifier_prompt_queue, self.classifier_result_queue,
                      self.updater_prompt_queue, self.updater_result_queue]:
            # 清空队列（此时应该已经空了，topic线程都已join）
            while not queue.empty():
                try:
                    queue.get_nowait()
                except:
                    break
            # 关闭队列（使用cancel_join_thread避免死锁）
            queue.close()
            queue.cancel_join_thread()  # 不等待后台线程，直接关闭
        
        # 强制释放显存
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # 等待一下确保进程完全退出和显存释放
        import time
        time.sleep(2)
        print(f"✓ Worker进程已停止，资源已清理，显存已释放")
        
        print(f"\n{'='*60}")
        print(f"✓ 轨迹采样完成")
        print(f"{'='*60}")
        print(f"  处理文章数: {global_counter.get_count()}")
        print(f"  轨迹数据条数: {len(all_trajectory_data)}\n")
        
        return all_trajectory_data
    
    def _process_topic(
        self,
        topic_state: TopicState,
        global_counter: GlobalCounter,
        result_queue: Queue,
        references_data: Dict,
        summaries_data: Dict
    ):
        """处理一个topic（顺序处理文章）"""
        topic_key = topic_state.topic_key
        
        # 获取该topic的references
        # references_data的结构是: {topic_key: {'topic': ..., 'references': {ref_id: {'content': ..., 'paths': ...}}}}
        topic_references = references_data.get(topic_key, {}).get('references', {})
        
        # 按article_order顺序处理
        article_processed_count = 0
        
        while True:
            # 检查全局停止
            if global_counter.should_stop():
                break
            
            # 检查是否还有文章待处理
            if not topic_state.has_next_article():
                # 所有文章处理完毕，重置树和文章顺序
                topic_state.reset_tree()
                
                # 重新获取文章列表并打乱
                article_ids = list(topic_references.keys())
                if article_ids:
                    topic_state.shuffle_articles(article_ids)
                else:
                    # 没有文章，退出
                    break
            
            # 获取下一篇文章
            article_id = topic_state.article_order[topic_state.next_article_idx]
            article_data = topic_references.get(article_id, {})
            article_content = article_data.get('content', '')
            
            if not article_content:
                topic_state.next_article_idx += 1
                continue
            
            # 处理这篇文章
            try:
                trajectory_data = self._process_single_article(
                    topic_state, article_id, article_content,
                    references_data, summaries_data
                )
                
                # 如果成功处理，放入结果队列
                if trajectory_data is not None:
                    result_queue.put(trajectory_data)
                
                article_processed_count += 1
                topic_state.next_article_idx += 1
                
                # 更新全局计数
                total_count = global_counter.increment()
                
                # 更新监控
                
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                topic_state.next_article_idx += 1
                continue
        
    
    def _process_single_article(
        self,
        topic_state: TopicState,
        article_id: str,
        article_content: str,
        references_data: Dict,
        summaries_data: Dict
    ) -> ArticleTrajectoryData:
        """
        处理单篇文章：采样轨迹 → 使用主轨迹更新树 → 返回轨迹数据
        
        注意：不再在采样阶段计算reward，reward计算移到标注阶段统一进行
        
        Returns:
            ArticleTrajectoryData
        """
        # 1. 创建使用Prompt队列的客户端
        from summary_based_classifier.models.pooled_model_client import PooledClassifierClient, PooledUpdaterClient
        
        pooled_classifier = PooledClassifierClient(
            prompt_queue=self.classifier_prompt_queue,
            result_queue=self.classifier_result_queue,
            topic_key=topic_state.topic_key,
            article_id=article_id
        )
        
        pooled_updater = PooledUpdaterClient(
            prompt_queue=self.updater_prompt_queue,
            result_queue=self.updater_result_queue,
            topic_key=topic_state.topic_key,
            article_id=article_id
        )
        
        # 2. 采样轨迹（使用pooled客户端）
        builder = TreeBuilder(
            classifier=pooled_classifier,
            updater=pooled_updater,
            topic_name=topic_state.topic_name,
            max_depth=self.max_depth
        )
        
        trajectories = builder.classify_and_update_with_sampling(
            article_id=article_id,
            article_content=article_content,
            root=topic_state.current_tree,
            sampling_num=self.sampling_num,
            top_k=self.top_k
        )
        
        if not trajectories:
            return None
        
        # 3. 直接使用主轨迹（第一条轨迹）的结构树来更新状态
        # 不再在采样阶段计算reward，而是在标注阶段统一计算
        main_trajectory = trajectories[0]
        topic_state.current_tree = main_trajectory.final_tree
        
        # 4. 对每个轨迹，在其final_tree上重新分类文章
        # 这样可以获取classification_records用于后续的reward计算
        re_classification_records = []
        
        for traj in trajectories:
            records = self._re_classify_on_tree(
                tree_root=traj.final_tree,
                article_content=article_content,
                pooled_classifier=pooled_classifier
            )
            re_classification_records.append(records)
        
        # 5. 获取ground truth paths
        topic_references = references_data.get(topic_state.topic_key, {}).get('references', {})
        article_data = topic_references.get(article_id, {})
        ground_truth_paths = article_data.get('paths', [])
        
        # 6. 返回完整的轨迹数据（包含重新分类的records）
        trajectory_data = ArticleTrajectoryData(
            topic_key=topic_state.topic_key,
            topic_name=topic_state.topic_name,
            article_id=article_id,
            article_content=article_content,
            ground_truth_paths=ground_truth_paths,
            trajectories=trajectories,
            re_classification_records=re_classification_records
        )
        
        return trajectory_data
    
    def _re_classify_on_tree(
        self,
        tree_root: TreeNode,
        article_content: str,
        pooled_classifier
    ) -> List[Dict]:
        """
        在树上重新递归分类文章（不调用总结系统，不更新，不真的加入文章）
        
        返回分类记录列表，每个记录包含：
        - current_node: 当前节点
        - child_summaries: 子节点summaries
        - classification_output: 分类系统的输出
        - scores: 每个类别的logP(Yes)（这里暂时为空，因为pooled_classifier没有logprobs功能）
        """
        records = []
        
        def classify_recursive(current_node: TreeNode, depth: int = 0):
            # 达到最大深度，停止
            if depth >= self.max_depth:
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
                current_node_summary=current_node.summary if current_node.summary else pooled_classifier.topic_key,
                child_summaries=child_summaries,
                topic_name=pooled_classifier.topic_key
            )
            
            # 使用pooled_classifier获取分类输出
            outputs = pooled_classifier.classify_with_sampling(classification_input, n=1)
            
            if not outputs:
                return
            
            classification_output = outputs[0]
            
            # 记录这次分类（暂时不包含scores，因为pooled client没有logprobs）
            records.append({
                'current_node': current_node,
                'child_summaries': child_summaries,
                'classification_output': classification_output,
                'scores': {}  # 空dict，后续可以通过标注系统获取
            })
            
            # 递归到选中的子节点（只用于继续收集分类记录，不真的修改树）
            for idx in classification_output.selected_indices:
                if 0 <= idx < len(children):
                    classify_recursive(children[idx], depth + 1)
            
            # 如果需要NEW，不递归（因为树上没有这个新节点）
        
        classify_recursive(tree_root)
        return records
    
    def _collect_step_samples(
        self,
        trajectories: List[Trajectory]
    ) -> Tuple[List[StepSample], List[StepSample]]:
        """收集所有轨迹的step-level样本"""
        classifier_samples = []
        updater_samples = []
        
        # 遍历所有轨迹，收集(state, action, global_reward)样本
        for traj_idx, traj in enumerate(trajectories):
            trajectory_id = f"traj_{traj_idx}"
            global_reward = traj.reward
            
            # 收集该轨迹中的所有actions
            for action in traj.actions:
                sample = StepSample(
                    system=action.system,
                    state=action.prompt,  # state就是输入prompt
                    action=action.completion,  # action就是模型输出
                    global_reward=global_reward,
                    trajectory_id=trajectory_id
                )
                
                if action.system == 'classify_generator':
                    classifier_samples.append(sample)
                elif action.system == 'updater':
                    updater_samples.append(sample)
        
        return classifier_samples, updater_samples

