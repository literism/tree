"""
线程安全的Prompt池和结果池
用于批量推理
"""
import threading
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from queue import Queue


@dataclass
class PromptRequest:
    """Prompt请求"""
    prompt_id: str  # 唯一ID，格式：{topic_key}_{article_id}_{step_id}
    prompt: str
    context: Any  # 额外上下文信息


@dataclass
class PromptResult:
    """Prompt结果"""
    prompt_id: str
    result: Any  # 模型输出


class ThreadSafePromptPool:
    """线程安全的Prompt池"""
    
    def __init__(self, batch_size: int = 32, timeout_seconds: float = 1.0):
        """
        Args:
            batch_size: 批量大小
            timeout_seconds: 超时时间（秒）
        """
        self.batch_size = batch_size
        self.timeout_seconds = timeout_seconds
        
        self.prompts: List[PromptRequest] = []
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        
        self.last_batch_time = time.time()
        self.stopped = False
    
    def submit(self, prompt_request: PromptRequest):
        """提交一个prompt"""
        with self.condition:
            self.prompts.append(prompt_request)
            # print(f"[PromptPool] 提交prompt {prompt_request.prompt_id}, 当前队列长度: {len(self.prompts)}")
            # 如果达到批量大小，通知消费者
            if len(self.prompts) >= self.batch_size:
                self.condition.notify()
                # print(f"[PromptPool] 达到批量大小 {self.batch_size}, 通知Worker")
    
    def get_batch(self, max_wait: float = None) -> List[PromptRequest]:
        """
        获取一批prompts（阻塞，直到有足够的prompts或超时）
        
        Args:
            max_wait: 最大等待时间（秒），None表示使用默认timeout
        
        Returns:
            prompt列表
        """
        wait_time = max_wait if max_wait is not None else self.timeout_seconds
        
        with self.condition:
            start_wait_time = time.time()
            
            # 等待直到有足够的prompts或超时
            while len(self.prompts) < self.batch_size and not self.stopped:
                elapsed = time.time() - start_wait_time
                remaining = wait_time - elapsed
                
                if remaining <= 0:
                    # 超时，返回当前所有prompts
                    break
                
                # 等待通知或超时
                self.condition.wait(timeout=remaining)
            
            # 取出prompts
            if self.prompts:
                batch = self.prompts[:self.batch_size]
                self.prompts = self.prompts[self.batch_size:]
                self.last_batch_time = time.time()
                return batch
            else:
                # 没有prompts，稍微休息一下避免CPU空转
                self.last_batch_time = time.time()
                return []
    
    def stop(self):
        """停止池子"""
        with self.condition:
            self.stopped = True
            self.condition.notify_all()
    
    def is_empty(self) -> bool:
        """检查池子是否为空"""
        with self.lock:
            return len(self.prompts) == 0


class ThreadSafeResultPool:
    """线程安全的结果池（简化版）"""
    
    def __init__(self):
        self.results: Dict[str, PromptResult] = {}
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)  # 只用一个全局condition
    
    def put(self, result: PromptResult):
        """放入一个结果"""
        with self.lock:
            self.results[result.prompt_id] = result
            # 通知所有等待的线程
            self.condition.notify_all()
    
    def put_batch(self, results: List[PromptResult]):
        """批量放入结果"""
        with self.lock:
            for result in results:
                self.results[result.prompt_id] = result
            # 通知所有等待的线程
            self.condition.notify_all()
    
    def get(self, prompt_id: str, timeout: float = 60.0) -> Optional[PromptResult]:
        """
        获取指定ID的结果（阻塞）
        
        Args:
            prompt_id: prompt ID
            timeout: 超时时间（秒）
        
        Returns:
            结果，或None（超时）
        """
        
        start_time = time.time()
        with self.condition:
            while prompt_id not in self.results:
                elapsed = time.time() - start_time
                remaining = timeout - elapsed
                
                if remaining <= 0:
                    return None
                
                # 等待通知或超时
                self.condition.wait(timeout=remaining)
            
            # 获取结果并删除
            result = self.results.pop(prompt_id)
            
            return result

