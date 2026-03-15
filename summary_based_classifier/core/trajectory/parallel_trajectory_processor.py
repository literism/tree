"""
并行轨迹处理器（多线程版）
实现多topic并行采样，批量prompt池+timeout管理
"""
import time
import threading
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
from queue import Queue
from summary_based_classifier.core.trajectory.trajectory_sampler import TreeNode, Trajectory, Action
from summary_based_classifier.core.topic_state import TopicState, TopicStateManager
from summary_based_classifier.llm.classify_generator import ClassifyGenerator, ClassificationInput, ClassificationOutput
from summary_based_classifier.llm.updater import Updater, SummaryInput, SummaryOutput


class GlobalCounter:
    """全局文章计数器（线程安全）"""
    
    def __init__(self, target: int):
        self.target = target
        self.count = 0
        self.lock = threading.Lock()
    
    def increment(self) -> int:
        """增加计数，返回当前值"""
        with self.lock:
            self.count += 1
            return self.count
    
    def get_count(self) -> int:
        """获取当前计数"""
        with self.lock:
            return self.count
    
    def should_stop(self) -> bool:
        """是否应该停止"""
        with self.lock:
            return self.count >= self.target


@dataclass
class PromptRequest:
    """Prompt请求"""
    request_id: str  # 唯一ID
    topic_key: str
    article_id: str
    branch_idx: int
    prompt_input: Any  # ClassificationInput 或 SummaryInput
    num_samples: int = 1
    created_time: float = field(default_factory=time.time)


@dataclass
class PromptResponse:
    """Prompt响应"""
    request_id: str
    results: List[Any]  # List[ClassificationOutput] 或 List[SummaryOutput]


class ThreadSafePromptPool:
    """线程安全的Prompt池"""
    
    def __init__(self, batch_size: int = 32, timeout_seconds: float = 1.0):
        self.batch_size = batch_size
        self.timeout = timeout_seconds
        
        self.requests: Dict[str, PromptRequest] = {}
        self.responses: Dict[str, PromptResponse] = {}
        
        self.lock = threading.Lock()
        self.first_request_time: Optional[float] = None
        
        # 用于等待响应
        self.response_events: Dict[str, threading.Event] = {}
    
    def submit_request(self, request: PromptRequest) -> str:
        """提交请求，返回request_id"""
        with self.lock:
            if not self.requests:
                self.first_request_time = time.time()
            
            self.requests[request.request_id] = request
            self.response_events[request.request_id] = threading.Event()
        
        return request.request_id
    
    def wait_for_response(self, request_id: str, timeout: float = 60.0) -> Optional[PromptResponse]:
        """等待响应"""
        event = self.response_events.get(request_id)
        if not event:
            return None
        
        # 等待响应
        if event.wait(timeout):
            with self.lock:
                response = self.responses.pop(request_id, None)
                del self.response_events[request_id]
                return response
        
        return None
    
    def should_execute(self) -> bool:
        """判断是否应该执行（由系统线程调用）"""
        with self.lock:
            if not self.requests:
                return False
            
            # 达到batch_size
            if len(self.requests) >= self.batch_size:
                return True
            
            # 超时
            if self.first_request_time and (time.time() - self.first_request_time) >= self.timeout:
                return True
            
            return False
    
    def get_batch(self) -> List[PromptRequest]:
        """获取一批请求（由系统线程调用）"""
        with self.lock:
            if not self.requests:
                return []
            
            requests = list(self.requests.values())
            self.requests.clear()
            self.first_request_time = None
            
            return requests
    
    def put_responses(self, responses: List[PromptResponse]):
        """放入响应（由系统线程调用）"""
        with self.lock:
            for response in responses:
                self.responses[response.request_id] = response
                
                # 通知等待的线程
                event = self.response_events.get(response.request_id)
                if event:
                    event.set()


class ClassifySystemWorker(threading.Thread):
    """分类系统工作线程"""
    
    def __init__(self, pool: ThreadSafePromptPool, classifier: ClassifyGenerator):
        super().__init__(daemon=True)
        self.pool = pool
        self.classifier = classifier
        self.running = True
    
    def run(self):
        """主循环"""
        while self.running:
            # 检查是否应该执行
            if self.pool.should_execute():
                batch = self.pool.get_batch()
                
                if batch:
                    print(f"    [分类系统] 执行批量推理: {len(batch)} 请求")
                    
                    # 批量推理
                    inputs = [req.prompt_input for req in batch]
                    num_samples = batch[0].num_samples if batch else 1
                    
                    try:
                        results_batch = self.classifier.classify_with_multiple_samples(inputs, n=num_samples)
                        
                        # 构建响应
                        responses = [
                            PromptResponse(request_id=req.request_id, results=results)
                            for req, results in zip(batch, results_batch)
                        ]
                        
                        # 放入响应
                        self.pool.put_responses(responses)
                        
                    except Exception as e:
                        print(f"    [分类系统] 推理失败: {e}")
                        
                        # 返回空结果
                        responses = [
                            PromptResponse(request_id=req.request_id, results=[])
                            for req in batch
                        ]
                        self.pool.put_responses(responses)
            
            # 短暂休眠，避免CPU占用过高
            time.sleep(0.01)
    
    def stop(self):
        """停止线程"""
        self.running = False


class UpdateSystemWorker(threading.Thread):
    """更新系统工作线程"""
    
    def __init__(self, pool: ThreadSafePromptPool, updater: Updater):
        super().__init__(daemon=True)
        self.pool = pool
        self.updater = updater
        self.running = True
    
    def run(self):
        """主循环"""
        while self.running:
            # 检查是否应该执行
            if self.pool.should_execute():
                batch = self.pool.get_batch()
                
                if batch:
                    print(f"    [更新系统] 执行批量推理: {len(batch)} 请求")
                    
                    # 批量推理
                    inputs = [req.prompt_input for req in batch]
                    num_samples = batch[0].num_samples if batch else 1
                    
                    try:
                        results_batch = self.updater.update_with_multiple_samples(inputs, n=num_samples)
                        
                        # 构建响应
                        responses = [
                            PromptResponse(request_id=req.request_id, results=results)
                            for req, results in zip(batch, results_batch)
                        ]
                        
                        # 放入响应
                        self.pool.put_responses(responses)
                        
                    except Exception as e:
                        print(f"    [更新系统] 推理失败: {e}")
                        
                        # 返回空结果
                        responses = [
                            PromptResponse(request_id=req.request_id, results=[])
                            for req in batch
                        ]
                        self.pool.put_responses(responses)
            
            # 短暂休眠
            time.sleep(0.01)
    
    def stop(self):
        """停止线程"""
        self.running = False


class TopicWorker(threading.Thread):
    """Topic工作线程"""
    
    def __init__(
        self,
        topic_key: str,
        state: TopicState,
        references_data: Dict,
        classify_pool: ThreadSafePromptPool,
        update_pool: ThreadSafePromptPool,
        classifier: ClassifyGenerator,
        updater: Updater,
        max_depth: int,
        num_samples: int,
        result_queue: Queue,
        stop_event: threading.Event,
        global_counter: 'GlobalCounter'
    ):
        super().__init__(daemon=True)
        self.topic_key = topic_key
        self.state = state
        self.references_data = references_data
        self.classify_pool = classify_pool
        self.update_pool = update_pool
        self.classifier = classifier
        self.updater = updater
        self.max_depth = max_depth
        self.num_samples = num_samples
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.global_counter = global_counter
        
        self.request_counter = 0
    
    def _generate_request_id(self, article_id: str) -> str:
        """生成唯一请求ID"""
        self.request_counter += 1
        return f"{self.topic_key}_{article_id}_{self.request_counter}"
    
    def run(self):
        """处理topic的所有文章"""
        print(f"  [Topic {self.topic_key}] 开始处理")
        
        processed = 0
        
        while self.state.has_next_article():
            # 检查是否应该停止
            if self.stop_event.is_set():
                print(f"  [Topic {self.topic_key}] 收到停止信号，已处理 {processed} 篇")
                break
            
            # 检查全局计数是否已达标
            if self.global_counter.should_stop():
                print(f"  [Topic {self.topic_key}] 全局已达标，停止处理（已处理 {processed} 篇）")
                break
            
            article_id = self.state.get_next_article_id()
            if not article_id:
                break
            
            # 获取文章数据
            topic_refs = self.references_data.get(self.topic_key, {}).get('references', {})
            if article_id not in topic_refs:
                continue
            
            article_data = topic_refs[article_id]
            article_content = article_data.get('content', '')
            ground_truth_paths = article_data.get('paths', [])
            
            # 处理这篇文章（使用Builder）
            trajectories = self._process_article(article_id, article_content, ground_truth_paths)
            
            # 放入结果队列
            if trajectories:
                self.result_queue.put((self.topic_key, article_id, trajectories))
                processed += 1
                
                # 增加全局计数
                count = self.global_counter.increment()
                print(f"  [Topic {self.topic_key}] 完成文章 {article_id} (全局: {count})")
                
                # 再次检查是否达到目标
                if self.global_counter.should_stop():
                    print(f"  [Topic {self.topic_key}] 全局达到目标，停止")
                    break
        
        print(f"  [Topic {self.topic_key}] 退出: {processed} 篇文章")
    
    def _process_article(
        self,
        article_id: str,
        article_content: str,
        ground_truth_paths: List[str]
    ) -> List[Trajectory]:
        """处理一篇文章（直接使用Builder）"""
        from summary_based_classifier.core.pipeline.builder import TreeBuilder
        
        # 创建builder
        builder = TreeBuilder(
            classifier=self.classifier,
            updater=self.updater,
            topic_name=self.state.topic_name,
            max_depth=self.max_depth
        )
        
        # 调用builder的多采样方法
        trajectories = builder.classify_and_update_with_sampling(
            article_id=article_id,
            article_content=article_content,
            root=self.state.current_tree,
            num_samples=self.num_samples
        )
        
        return trajectories


class ParallelTrajectoryProcessor:
    """并行轨迹处理器（多线程版）"""
    
    def __init__(
        self,
        state_manager: TopicStateManager,
        classifier: ClassifyGenerator,
        updater: Updater,
        max_depth: int = 3,
        num_samples_per_prompt: int = 4,
        batch_size: int = 32,
        timeout_seconds: float = 1.0
    ):
        self.state_manager = state_manager
        self.classifier = classifier
        self.updater = updater
        self.max_depth = max_depth
        self.num_samples = num_samples_per_prompt
        
        # Prompt池
        self.classify_pool = ThreadSafePromptPool(batch_size, timeout_seconds)
        self.update_pool = ThreadSafePromptPool(batch_size, timeout_seconds)
        
        # 系统工作线程
        self.classify_worker = ClassifySystemWorker(self.classify_pool, classifier)
        self.update_worker = UpdateSystemWorker(self.update_pool, updater)
    
    def process_articles(
        self,
        references_data: Dict,
        num_articles: int
    ) -> Dict[str, List[Tuple[str, List[Trajectory]]]]:
        """
        处理指定数量的文章（多线程并行）
        
        Returns:
            {topic_key: [(article_id, [trajectories])]}
        """
        print(f"\n开始多线程并行采样（目标: {num_articles}篇文章）...")
        
        # 启动系统工作线程
        print("  启动系统工作线程...")
        self.classify_worker.start()
        self.update_worker.start()
        
        # 全局计数器和停止事件
        global_counter = GlobalCounter(target=num_articles)
        stop_event = threading.Event()
        
        # 结果队列
        result_queue = Queue()
        
        # 创建topic工作线程
        topic_workers = []
        active_topics = self.state_manager.get_topics_with_articles()
        
        print(f"  启动 {len(active_topics)} 个topic工作线程...")
        for topic_key in active_topics:
            state = self.state_manager.get_state(topic_key)
            if not state:
                continue
            
            worker = TopicWorker(
                topic_key=topic_key,
                state=state,
                references_data=references_data,
                classify_pool=self.classify_pool,
                update_pool=self.update_pool,
                classifier=self.classifier,
                updater=self.updater,
                max_depth=self.max_depth,
                num_samples=self.num_samples,
                result_queue=result_queue,
                stop_event=stop_event,
                global_counter=global_counter
            )
            
            worker.start()
            topic_workers.append(worker)

            break
        
        # 监控全局计数，达到目标时设置停止事件
        while global_counter.get_count() < num_articles:
            time.sleep(0.1)
            
            # 检查是否所有线程都已结束（可能文章不够）
            all_done = all(not w.is_alive() for w in topic_workers)
            if all_done:
                print(f"\n  所有topic线程已完成，实际处理: {global_counter.get_count()} 篇")
                break
        
        # 设置停止事件（通知所有topic线程停止）
        print(f"\n  已达到目标 ({global_counter.get_count()}/{num_articles})，通知所有线程停止...")
        stop_event.set()
        
        # 等待所有topic线程结束（最多等待10秒）
        for worker in topic_workers:
            worker.join(timeout=10.0)
        
        # 停止系统工作线程
        self.classify_worker.stop()
        self.update_worker.stop()
        
        # 等待系统工作线程结束
        self.classify_worker.join(timeout=5.0)
        self.update_worker.join(timeout=5.0)
        
        # 收集结果
        all_results = defaultdict(list)
        while not result_queue.empty():
            topic_key, article_id, trajectories = result_queue.get()
            all_results[topic_key].append((article_id, trajectories))
        
        total_articles = sum(len(v) for v in all_results.values())
        print(f"\n  ✓ 完成: {total_articles}篇文章")
        
        return dict(all_results)
