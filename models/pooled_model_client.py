"""
使用Prompt队列的模型客户端
为Builder提供透明的接口，内部使用multiprocessing.Queue+批量推理
"""
from typing import List, Tuple
import uuid
import multiprocessing as mp
import time
from dataclasses import dataclass, field
from summary_based_classifier.llm.classify_generator import ClassificationInput, ClassificationOutput
from summary_based_classifier.llm.updater import SummaryInput, SummaryOutput
from summary_based_classifier.llm.prompts import PromptTemplates


@dataclass
class PromptRequest:
    """Prompt请求"""
    prompt_id: str
    prompt: str
    context: dict
    need_logprobs: bool = False  # 是否需要返回logprobs


@dataclass
class PromptResult:
    """Prompt结果"""
    prompt_id: str
    result: any
    logprobs: dict = None  # logprobs数据（如果请求了的话）


class PooledClassifierClient:
    """使用Prompt队列的分类系统客户端"""
    
    def __init__(
        self,
        prompt_queue: mp.Queue,
        result_queue: mp.Queue,
        topic_key: str,
        article_id: str
    ):
        self.prompt_queue = prompt_queue
        self.result_queue = result_queue
        self.topic_key = topic_key
        self.article_id = article_id
        self.step_counter = 0
    
    def create_prompt(self, input_data: ClassificationInput) -> str:
        """创建分类prompt"""
        return PromptTemplates.format_classification_prompt(
            topic_name=input_data.topic_name,
            current_summary=input_data.current_node_summary,
            child_summaries=input_data.child_summaries,
            article_content=input_data.article_content
        )
    
    def classify_with_sampling(
        self,
        classification_input: ClassificationInput,
        n: int = 1
    ) -> List[ClassificationOutput]:
        """
        分类（使用prompt池）
        
        Args:
            classification_input: 分类输入
            n: 采样数量
        
        Returns:
            分类输出列表
        """
        # 生成唯一ID
        self.step_counter += 1
        prompt_id = f"{self.topic_key}_{self.article_id}_{self.step_counter}_classify"
        
        # 创建prompt
        prompt = self.create_prompt(classification_input)
        
        # 提交到prompt队列
        self.prompt_queue.put(PromptRequest(
            prompt_id=prompt_id,
            prompt=prompt,
            context={
                'input': classification_input,
                'num_children': len(classification_input.child_summaries),
                'n': n,
                'temperature': 0.7
            }
        ))
        
        # 等待结果（从result队列轮询）
        start_time = time.time()
        timeout = 3600.0
        while (time.time() - start_time) < timeout:
            try:
                result_obj = self.result_queue.get(timeout=1.0)
                if result_obj.prompt_id == prompt_id:
                    if result_obj.result:
                        return result_obj.result
                    else:
                        return []
                else:
                    # 不是我们的结果，放回队列让其他线程取
                    self.result_queue.put(result_obj)
                    time.sleep(0.01)  # 短暂休眠避免忙等
            except:
                # 队列空，继续等待
                time.sleep(0.01)
        
        # 超时，返回空结果
        return []
    
    def classify_with_logprobs(
        self,
        classification_input: ClassificationInput
    ) -> Tuple:
        """
        分类并返回logprobs（用于reward计算）
        
        Args:
            classification_input: 分类输入
        
        Returns:
            (classification_output, scores)
            - classification_output: ClassificationOutput对象或None
            - scores: Dict[Union[int, str], float]，每个类别的logP(Yes)
        """
        # 生成唯一ID
        self.step_counter += 1
        prompt_id = f"{self.topic_key}_{self.article_id}_{self.step_counter}_classify_logprobs"
        
        # 创建prompt
        prompt = self.create_prompt(classification_input)
        
        # 提交到prompt队列，标记需要logprobs
        self.prompt_queue.put(PromptRequest(
            prompt_id=prompt_id,
            prompt=prompt,
            context={
                'input': classification_input,
                'num_children': len(classification_input.child_summaries),
                'n': 1,
                'temperature': 0.0
            },
            need_logprobs=True  # 关键：请求logprobs
        ))
        
        # 等待结果
        start_time = time.time()
        timeout = 120.0
        while (time.time() - start_time) < timeout:
            try:
                result_obj = self.result_queue.get(timeout=1.0)
                if result_obj.prompt_id == prompt_id:
                    if result_obj.result and len(result_obj.result) > 0:
                        classification_output = result_obj.result[0]
                        scores = result_obj.logprobs if result_obj.logprobs else {}
                        return classification_output, scores
                    else:
                        return None, {}
                else:
                    # 不是我们的结果，放回队列让其他线程取
                    self.result_queue.put(result_obj)
                    time.sleep(0.01)  # 短暂休眠避免忙等
            except:
                # 队列空，继续等待
                time.sleep(0.01)
        
        # 超时
        return None, {}


class PooledUpdaterClient:
    """使用Prompt队列的总结系统客户端"""
    
    def __init__(
        self,
        prompt_queue: mp.Queue,
        result_queue: mp.Queue,
        topic_key: str,
        article_id: str
    ):
        self.prompt_queue = prompt_queue
        self.result_queue = result_queue
        self.topic_key = topic_key
        self.article_id = article_id
        self.step_counter = 0
        self.pending_results = {}  # {prompt_id: result}
    
    def create_prompt(self, input_data: SummaryInput) -> str:
        """创建总结prompt"""
        return PromptTemplates.format_summary_prompt(
            topic_name=input_data.topic_name,
            node_summary=input_data.node_summary,
            parent_summary=input_data.parent_summary,
            sibling_summaries=input_data.sibling_summaries,
            new_content=input_data.new_content
        )
    
    def update_summary(
        self,
        summary_input: SummaryInput,
        n_samples: int = 1
    ) -> List[SummaryOutput]:
        """
        更新总结（使用prompt池）
        
        Args:
            summary_input: 总结输入
            n_samples: 采样数量
        
        Returns:
            总结输出列表
        """
        # 生成唯一ID
        self.step_counter += 1
        prompt_id = f"{self.topic_key}_{self.article_id}_{self.step_counter}_update"
        
        # 创建prompt
        prompt = self.create_prompt(summary_input)
        
        # 提交到prompt队列
        self.prompt_queue.put(PromptRequest(
            prompt_id=prompt_id,
            prompt=prompt,
            context={
                'input': summary_input,
                'n': n_samples,
                'temperature': 0.7
            }
        ))
        
        # 等待结果（从result队列轮询）
        start_time = time.time()
        timeout = 120.0
        while (time.time() - start_time) < timeout:
            try:
                result_obj = self.result_queue.get(timeout=1.0)
                if result_obj.prompt_id == prompt_id:
                    if result_obj.result:
                        return result_obj.result
                    else:
                        return []
                else:
                    # 不是我们的结果，放回队列让其他线程取
                    self.result_queue.put(result_obj)
                    time.sleep(0.01)  # 短暂休眠避免忙等
            except:
                # 队列空，继续等待
                time.sleep(0.01)
        
        # 超时，返回空结果
        return []

