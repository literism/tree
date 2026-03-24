"""
并行推理处理器
使用多进程Worker和多线程Topic处理实现并行推理

设计：
- 每个模型在独立进程中运行（避免CUDA fork问题）
- 每个topic在独立线程中处理
- 使用生产者-消费者模式进行批量推理
"""
import threading
import time
from queue import Queue
from tqdm import tqdm
from typing import Dict, List, Any
import multiprocessing as mp

from summary_based_classifier.models.model_workers import ClassifierWorker, UpdaterWorker
from summary_based_classifier.core.trajectory.trajectory_sampler import TreeNode
from summary_based_classifier.core.pipeline.builder import TreeBuilder


class ParallelInferenceProcessor:
    """并行推理处理器"""
    
    def __init__(
        self,
        classify_generator_model: str,
        updater_model: str,
        max_depth: int,
        classify_generator_gpu_id: int = 0,
        updater_gpu_id: int = 1,
        classifier_batch_size: int = 8,
        updater_batch_size: int = 4,
        classifier_timeout: float = 1.0,
        updater_timeout: float = 2.0,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.85,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_workers: int = 4
    ):
        """
        Args:
            classify_generator_model: 分类生成模型路径
            updater_model: 总结更新模型路径
            max_depth: 最大树深度
            classify_generator_gpu_id: 分类生成模型GPU ID
            updater_gpu_id: 总结更新模型GPU ID
            classifier_batch_size: 分类器批次大小
            updater_batch_size: 更新器批次大小
            classifier_timeout: 分类器批次超时（秒）
            updater_timeout: 更新器批次超时（秒）
            max_model_len: 模型最大长度
            gpu_memory_utilization: GPU内存利用率
            temperature: 采样温度
            top_p: top_p采样
            max_workers: 最大并行topic数
        """
        self.max_depth = max_depth
        self.max_workers = max_workers
        
        # 创建multiprocessing队列
        mp_ctx = mp.get_context('spawn')
        self.classifier_prompt_queue = mp_ctx.Queue()
        self.classifier_result_queue = mp_ctx.Queue()
        self.updater_prompt_queue = mp_ctx.Queue()
        self.updater_result_queue = mp_ctx.Queue()
        
        # 创建Worker进程
        print(f"\n启动Worker进程...")
        self.classifier_worker = ClassifierWorker(
            model_path=classify_generator_model,
            prompt_queue=self.classifier_prompt_queue,
            result_queue=self.classifier_result_queue,
            gpu_id=classify_generator_gpu_id,
            batch_size=classifier_batch_size,
            timeout=classifier_timeout,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            temperature=temperature,
            top_p=top_p,
            max_tokens=256
        )
        
        self.updater_worker = UpdaterWorker(
            model_path=updater_model,
            prompt_queue=self.updater_prompt_queue,
            result_queue=self.updater_result_queue,
            gpu_id=updater_gpu_id,
            batch_size=updater_batch_size,
            timeout=updater_timeout,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            temperature=temperature,
            top_p=top_p,
            max_tokens=512
        )
        
        print(f"  ✓ Worker进程已启动")
    
    def process_topics(
        self,
        topics_data: Dict[str, Dict[str, Any]],
        max_refs: int = None
    ) -> Dict[str, Dict]:
        """
        并行处理多个topic的推理
        
        Args:
            topics_data: {topic_key: {'topic': topic_name, 'articles': [...]}}
            max_refs: 每个topic最多处理的文章数
            
        Returns:
            {topic_key: tree_dict}
        """
        print(f"\n{'='*60}")
        print(f"开始并行推理")
        print(f"  - Topic数量: {len(topics_data)}")
        print(f"  - 最大并行线程数: {self.max_workers}")
        print(f"{'='*60}")
        
        # 结果存储（线程安全）
        results = {}
        results_lock = threading.Lock()
        
        # 进度条
        pbar = tqdm(total=len(topics_data), desc="推理进度")
        
        def process_topic(topic_key: str, topic_info: Dict):
            """单个topic的推理线程"""
            try:
                topic_name = topic_info['topic']
                articles = topic_info['articles']
                
                if max_refs:
                    articles = articles[:max_refs]
                
                # 创建根节点
                root = TreeNode(summary="", citations=[], children=[], depth=0)
                
                # 创建构建器（使用Worker队列）
                builder = TreeBuilder(
                    classifier=None,  # 将使用队列
                    updater=None,     # 将使用队列
                    topic_name=topic_name,
                    max_depth=self.max_depth
                )
                
                # 注入队列到builder
                builder.classifier_prompt_queue = self.classifier_prompt_queue
                builder.classifier_result_queue = self.classifier_result_queue
                builder.updater_prompt_queue = self.updater_prompt_queue
                builder.updater_result_queue = self.updater_result_queue
                builder.use_workers = True
                
                # 构建树
                root = builder.build_tree_for_articles(articles, root)
                
                # 转换为字典
                tree_dict = {
                    'topic': topic_name,
                    'structure': [builder.tree_to_dict(child, level=2) for child in root.children]
                }
                
                # 保存结果（线程安全）
                with results_lock:
                    results[topic_key] = tree_dict
                
                pbar.update(1)
                
            except Exception as e:
                print(f"\n[错误] Topic {topic_key} 推理失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 使用线程池处理topics
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(process_topic, topic_key, topic_info): topic_key
                for topic_key, topic_info in topics_data.items()
            }
            
            # 等待所有任务完成
            for future in as_completed(futures):
                topic_key = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"\n[错误] Topic {topic_key} 处理异常: {e}")
        
        # 关闭进度条
        pbar.close()
        
        # 停止Worker进程
        print(f"\n停止Worker进程...")
        self.classifier_worker.stop()
        self.updater_worker.stop()
        
        # 清理资源
        print(f"清理资源...")
        for queue in [self.classifier_prompt_queue, self.classifier_result_queue,
                      self.updater_prompt_queue, self.updater_result_queue]:
            while not queue.empty():
                try:
                    queue.get_nowait()
                except:
                    break
            queue.close()
            queue.cancel_join_thread()
        
        # 强制释放显存
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        time.sleep(2)
        print(f"✓ Worker进程已停止，资源已清理")
        
        print(f"\n{'='*60}")
        print(f"✓ 推理完成")
        print(f"{'='*60}")
        
        return results

