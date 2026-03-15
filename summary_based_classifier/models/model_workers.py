"""
模型推理Worker进程
分类系统和总结系统各一个独立进程，负责批量推理
"""
import multiprocessing as mp
import time
from typing import List
from dataclasses import dataclass
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


class ClassifierWorker:
    """分类系统Worker（独立进程）"""
    
    def __init__(
        self,
        model_path: str,
        prompt_queue: mp.Queue,
        result_queue: mp.Queue,
        batch_size: int = 32,
        timeout: float = 1.0,
        gpu_id: int = 0,
        max_model_len: int = 16384,
        gpu_memory_utilization: float = 0.9,
        temperature: float = 0.7
    ):
        """
        Args:
            model_path: 模型路径
            prompt_queue: prompt队列(multiprocessing.Queue)
            result_queue: 结果队列(multiprocessing.Queue)
            batch_size: 批量大小
            timeout: 超时时间
            gpu_id: GPU ID
            max_model_len: 最大模型长度
            gpu_memory_utilization: GPU内存利用率
            temperature: 温度参数
        """
        self.model_path = model_path
        self.prompt_queue = prompt_queue
        self.result_queue = result_queue
        self.batch_size = batch_size
        self.timeout = timeout
        self.gpu_id = gpu_id
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.temperature = temperature
        self.process = None
        self.stop_event = mp.Event()  # 停止信号
    
    def start(self):
        """启动Worker进程"""
        self.stop_event.clear()  # 清除停止信号
        self.process = mp.Process(
            target=self._run, 
            name="ClassifierWorker",
            daemon=False  # vLLM会创建子进程，不能用daemon
        )
        self.process.start()
    
    def stop(self):
        """停止Worker进程（优雅关闭）"""
        if self.process and self.process.is_alive():
            # 1. 清空prompt队列，避免worker继续处理
            while not self.prompt_queue.empty():
                try:
                    self.prompt_queue.get_nowait()
                except:
                    break
            
            # 2. 发送停止信号，让进程内部清理模型
            self.stop_event.set()
            
            # 3. 等待进程优雅退出
            self.process.join(timeout=10)
            
            # 4. 如果还没退出，强制终止
            if self.process.is_alive():
                print("[ClassifierWorker] 优雅关闭超时，强制终止进程")
                self.process.terminate()
                self.process.join(timeout=5)
                
            # 5. 如果还没退出，强制kill
            if self.process.is_alive():
                print("[ClassifierWorker] 强制终止失败，使用kill")
                self.process.kill()
                self.process.join(timeout=2)
    
    def _run(self):
        """Worker主循环（在独立进程中运行）"""
        # 在进程内加载模型
        print(f"[ClassifierWorker] 进程启动，加载模型: {self.model_path} (GPU {self.gpu_id})")
        from summary_based_classifier.llm.classify_generator import ClassifyGenerator
        classifier = ClassifyGenerator(
            mode='model',
            model_path=self.model_path,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            temperature=self.temperature,
            gpu_id=self.gpu_id
        )
        print(f"[ClassifierWorker] 模型加载完成")
        
        # 批量收集prompts
        batch = []
        last_batch_time = time.time()
        
        try:
            while not self.stop_event.is_set():
                try:
                    # 检查停止信号
                    if self.stop_event.is_set():
                        break
                    
                    # 尝试从队列获取prompt（非阻塞，带超时）
                    timeout_remaining = max(0.1, self.timeout - (time.time() - last_batch_time))
                    prompt_req = self.prompt_queue.get(timeout=timeout_remaining)
                    batch.append(prompt_req)
                    
                    # 检查是否达到batch_size或超时
                    if len(batch) >= self.batch_size or (time.time() - last_batch_time) >= self.timeout:
                        if batch:
                            # 批量推理
                            results = ClassifierWorker._batch_inference(classifier, batch)
                            
                            # 将结果放回队列（results现在是(parsed_outputs, logprobs_dict)元组）
                            for prompt_req, (parsed_outputs, logprobs_dict) in zip(batch, results):
                                self.result_queue.put(PromptResult(
                                    prompt_id=prompt_req.prompt_id,
                                    result=parsed_outputs,
                                    logprobs=logprobs_dict
                                ))
                            
                            batch = []
                            last_batch_time = time.time()
                
                except Exception as e:
                    # 检查停止信号
                    if self.stop_event.is_set():
                        break
                    
                    # 队列空或其他错误，如果有积累的batch则处理
                    if batch and (time.time() - last_batch_time) >= self.timeout:
                        try:
                            results = ClassifierWorker._batch_inference(classifier, batch)
                            for prompt_req, (parsed_outputs, logprobs_dict) in zip(batch, results):
                                self.result_queue.put(PromptResult(
                                    prompt_id=prompt_req.prompt_id,
                                    result=parsed_outputs,
                                    logprobs=logprobs_dict
                                ))
                        except Exception as batch_error:
                            print(f"[ClassifierWorker] 批量推理错误: {batch_error}")
                            for prompt_req in batch:
                                self.result_queue.put(PromptResult(
                                    prompt_id=prompt_req.prompt_id,
                                    result=None,
                                    logprobs=None
                                ))
                        batch = []
                        last_batch_time = time.time()
                    else:
                        time.sleep(0.01)  # 短暂休眠
        
        finally:
            # 清理模型和释放显存
            print(f"[ClassifierWorker] 开始清理模型...")
            del classifier
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            print(f"[ClassifierWorker] 模型已清理，显存已释放")
        
    
    @staticmethod
    def _batch_inference(classifier, batch: List) -> List:
        """
        批量推理
        
        Args:
            classifier: 分类器实例
            batch: prompt请求列表
        
        Returns:
            结果列表，每个元素是 (ClassificationOutput列表, logprobs_dict或None)
        """
        # 提取prompts、contexts和need_logprobs标志
        prompts = [req.prompt for req in batch]
        contexts = [req.context for req in batch]
        need_logprobs_list = [getattr(req, 'need_logprobs', False) for req in batch]
        
        # 批量调用分类系统
        results = []
        
        if hasattr(classifier, 'llm'):
            # vLLM模式，批量生成
            from vllm import SamplingParams
            
            sampling_params = SamplingParams(
                temperature=contexts[0].get('temperature', 0.7) if contexts and isinstance(contexts[0], dict) else 0.7,
                max_tokens=512,  # 增加到512以容纳更长的说明性输出
                n=contexts[0].get('n', 1) if contexts and isinstance(contexts[0], dict) else 1,
                logprobs=5  # 统一都启用logprobs，开销很小
            )
            
            outputs = classifier.llm.generate(prompts, sampling_params, use_tqdm=False)
            
            # 解析每个输出
            for i, output in enumerate(outputs):
                context = contexts[i]
                num_children = context.get('num_children', 0) if isinstance(context, dict) else 0
                need_logprobs = need_logprobs_list[i]
                
                # 解析输出
                parsed_outputs = []
                logprobs_dict = None
                
                for out in output.outputs:
                    response_text = out.text
                    parsed_dict = PromptTemplates.parse_classification_output(response_text, num_children)
                    
                    # 转换dict为ClassificationOutput对象
                    if parsed_dict:
                        from summary_based_classifier.llm.classify_generator import ClassificationOutput
                        parsed_output = ClassificationOutput(
                            selected_indices=parsed_dict['selected_indices'],
                            need_new=parsed_dict['need_new'],
                            merge_with=parsed_dict.get('merge_with'),
                            raw_response=response_text
                        )
                        parsed_outputs.append(parsed_output)
                    
                    # 总是提取logprobs（如果有的话），然后根据need_logprobs决定是否返回
                    if out.logprobs:
                        extracted_logprobs = classifier._extract_yes_probs_from_logprobs(
                            out.logprobs, response_text, num_children
                        )
                        # 只在需要时才返回
                        if need_logprobs:
                            logprobs_dict = extracted_logprobs
                
                results.append((parsed_outputs, logprobs_dict))
        
        else:
            # API模式，逐个调用（API通常不支持批量，也不支持logprobs）
            for i, prompt in enumerate(prompts):
                context = contexts[i]
                classification_input = context.get('input') if isinstance(context, dict) else None
                n = context.get('n', 1) if isinstance(context, dict) else 1
                
                if classification_input:
                    outputs = classifier.classify_with_sampling(classification_input, n=n)
                    results.append((outputs, None))  # API模式不返回logprobs
                else:
                    results.append(([], None))
        
        return results


class UpdaterWorker:
    """总结系统Worker（独立进程）"""
    
    def __init__(
        self,
        model_path: str,
        prompt_queue: mp.Queue,
        result_queue: mp.Queue,
        batch_size: int = 32,
        timeout: float = 1.0,
        gpu_id=1,
        tensor_parallel_size: int = 1,
        max_model_len: int = 16384,
        gpu_memory_utilization: float = 0.9,
        temperature: float = 0.5
    ):
        """
        Args:
            model_path: 模型路径
            prompt_queue: prompt队列(multiprocessing.Queue)
            result_queue: 结果队列(multiprocessing.Queue)
            batch_size: 批量大小
            timeout: 超时时间
            gpu_id: GPU ID或逗号分隔GPU列表（如 "0,1"）
            tensor_parallel_size: 张量并行卡数
            max_model_len: 最大模型长度
            gpu_memory_utilization: GPU内存利用率
            temperature: 温度参数
        """
        self.model_path = model_path
        self.prompt_queue = prompt_queue
        self.result_queue = result_queue
        self.batch_size = batch_size
        self.timeout = timeout
        self.gpu_id = gpu_id
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.temperature = temperature
        self.process = None
        self.stop_event = mp.Event()  # 停止信号
    
    def start(self):
        """启动Worker进程"""
        self.stop_event.clear()  # 清除停止信号
        self.process = mp.Process(
            target=self._run,
            name="UpdaterWorker",
            daemon=False  # vLLM会创建子进程，不能用daemon
        )
        self.process.start()
    
    def stop(self):
        """停止Worker进程（优雅关闭）"""
        if self.process and self.process.is_alive():
            # 1. 清空prompt队列，避免worker继续处理
            while not self.prompt_queue.empty():
                try:
                    self.prompt_queue.get_nowait()
                except:
                    break
            
            # 2. 发送停止信号，让进程内部清理模型
            self.stop_event.set()
            
            # 3. 等待进程优雅退出
            self.process.join(timeout=10)
            
            # 4. 如果还没退出，强制终止
            if self.process.is_alive():
                print("[UpdaterWorker] 优雅关闭超时，强制终止进程")
                self.process.terminate()
                self.process.join(timeout=5)
                
            # 5. 如果还没退出，强制kill
            if self.process.is_alive():
                print("[UpdaterWorker] 强制终止失败，使用kill")
                self.process.kill()
                self.process.join(timeout=2)
    
    def _run(self):
        """Worker主循环（在独立进程中运行）"""
        # 在进程内加载模型
        print(f"[UpdaterWorker] 进程启动，加载模型: {self.model_path} (GPU {self.gpu_id})")
        from summary_based_classifier.llm.updater import Updater
        updater = Updater(
            mode='model',
            model_path=self.model_path,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            temperature=self.temperature,
            gpu_id=self.gpu_id,
            tensor_parallel_size=self.tensor_parallel_size,
        )
        print(f"[UpdaterWorker] 模型加载完成")
        
        # 批量收集prompts
        batch = []
        last_batch_time = time.time()
        
        try:
            while not self.stop_event.is_set():
                try:
                    # 检查停止信号
                    if self.stop_event.is_set():
                        break
                    
                    # 尝试从队列获取prompt（非阻塞，带超时）
                    timeout_remaining = max(0.1, self.timeout - (time.time() - last_batch_time))
                    prompt_req = self.prompt_queue.get(timeout=timeout_remaining)
                    batch.append(prompt_req)
                    
                    # 检查是否达到batch_size或超时
                    if len(batch) >= self.batch_size or (time.time() - last_batch_time) >= self.timeout:
                        if batch:
                            # 批量推理
                            results = UpdaterWorker._batch_inference(updater, batch)
                            
                            # 将结果放回队列
                            for prompt_req, result in zip(batch, results):
                                self.result_queue.put(PromptResult(
                                    prompt_id=prompt_req.prompt_id,
                                    result=result
                                ))
                            
                            batch = []
                            last_batch_time = time.time()
                
                except Exception as e:
                    # 检查停止信号
                    if self.stop_event.is_set():
                        break
                    
                    # 队列空或其他错误，如果有积累的batch则处理
                    if batch and (time.time() - last_batch_time) >= self.timeout:
                        try:
                            results = UpdaterWorker._batch_inference(updater, batch)
                            for prompt_req, result in zip(batch, results):
                                self.result_queue.put(PromptResult(
                                    prompt_id=prompt_req.prompt_id,
                                    result=result
                                ))
                        except Exception as batch_error:
                            print(f"[UpdaterWorker] 批量推理错误: {batch_error}")
                            for prompt_req in batch:
                                self.result_queue.put(PromptResult(
                                    prompt_id=prompt_req.prompt_id,
                                    result=None
                                ))
                        batch = []
                        last_batch_time = time.time()
                    else:
                        time.sleep(0.01)  # 短暂休眠
        
        finally:
            # 清理模型和释放显存
            print(f"[UpdaterWorker] 开始清理模型...")
            del updater
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            print(f"[UpdaterWorker] 模型已清理，显存已释放")
    
    @staticmethod
    def _batch_inference(updater, batch: List[PromptRequest]) -> List:
        """
        批量推理
        
        Args:
            updater: 更新器实例
            batch: prompt请求列表
        
        Returns:
            结果列表（SummaryOutput）
        """
        # 提取prompts和contexts
        prompts = [req.prompt for req in batch]
        contexts = [req.context for req in batch]
        
        # 批量调用总结系统
        results = []
        
        if hasattr(updater, 'llm'):
            # vLLM模式，批量生成
            from vllm import SamplingParams
            
            sampling_params = SamplingParams(
                temperature=contexts[0].get('temperature', 0.7) if contexts and isinstance(contexts[0], dict) else 0.7,
                max_tokens=2048,
                n=contexts[0].get('n', 1) if contexts and isinstance(contexts[0], dict) else 1
            )
            
            outputs = updater.llm.generate(prompts, sampling_params, use_tqdm=False)
            
            # 解析每个输出
            for output in outputs:
                parsed_outputs = []
                for out in output.outputs:
                    response_text = out.text
                    parsed_dict = PromptTemplates.parse_summary_output(response_text)
                    
                    # 转换dict为SummaryOutput对象
                    if parsed_dict:
                        from summary_based_classifier.llm.updater import SummaryOutput
                        parsed_output = SummaryOutput(
                            needs_update=parsed_dict.get('needs_update', False),
                            explanation=parsed_dict.get('explanation', ''),
                            scope=parsed_dict.get('scope', ''),
                            raw_response=response_text
                        )
                        parsed_outputs.append(parsed_output)
                
                results.append(parsed_outputs)
        
        else:
            # API模式，逐个调用
            for i, prompt in enumerate(prompts):
                context = contexts[i]
                summary_input = context.get('input') if isinstance(context, dict) else None
                n = context.get('n', 1) if isinstance(context, dict) else 1
                
                if summary_input:
                    outputs = updater.update_summary(summary_input, n_samples=n)
                    results.append(outputs)
                else:
                    results.append([])
        
        return results

