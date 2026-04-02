"""
分类生成系统
输出多行Yes/No格式，判断文章属于哪些类别以及是否需要新类
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from summary_based_classifier.llm.prompts import PromptTemplates
from pathlib import Path
import sys
import os


def _pick_safe_vllm_dtype(requested_dtype):
    """Use float16 on legacy GPUs (e.g., V100) to avoid bf16 init errors."""
    if requested_dtype is not None:
        return requested_dtype
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        for idx in range(torch.cuda.device_count()):
            major, _minor = torch.cuda.get_device_capability(idx)
            if major < 8:
                return "half"
    except Exception:
        return None
    return None


@dataclass
class ClassificationInput:
    """分类输入"""
    article_content: str
    current_node_summary: str  # 当前节点的summary
    child_summaries: List[str]  # 子节点的summary列表
    topic_name: str  # topic名称
    # 新增：结构特征
    child_num_children: List[int] = field(default_factory=list)  # 每个子类的子节点数
    child_max_depth: List[int] = field(default_factory=list)     # 每个子类的子树最大深度
    current_depth: int = 0  # 当前节点深度
    num_children: int = 0   # 当前节点子节点数


@dataclass
class ClassificationOutput:
    """分类输出"""
    selected_indices: List[int]  # 选中的类别索引（从0开始）
    need_new: bool  # 是否需要新类
    raw_response: str  # 原始响应（用于计算log概率）
    merge_with: Optional[int] = None  # 当need_new=True时，是否与某个现有类别归拢（InsertParentPath）；None表示不归拢
    new_node_direction: Dict = field(default_factory=dict)  # 新类方向（用于指导新节点summary生成）
    merge_candidate_probs: Dict[str, float] = field(default_factory=dict)  # MERGE_WITH候选概率（含null）


class ClassifyGenerator:
    """分类生成系统"""
    
    def __init__(self, mode: str = 'model', model_path: Optional[str] = None, **kwargs):
        """
        Args:
            mode: 模式 ('model' 或 'api')
            model_path: 模型路径（mode='model'时需要）
            **kwargs: 其他参数
        """
        self.mode = mode
        self.model_path = model_path
        
        if mode == 'model':
            from vllm import LLM, SamplingParams
            
            # 获取GPU设备ID
            gpu_id = kwargs.get('gpu_id', 0)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            effective_tp = int(kwargs.get('tensor_parallel_size', 1))

            llm_kwargs = dict(
                model=model_path,
                tensor_parallel_size=effective_tp,
                max_model_len=kwargs.get('max_model_len', 16384),
                gpu_memory_utilization=kwargs.get('gpu_memory_utilization', 0.9),
                trust_remote_code=True,
                disable_log_stats=True,  # 禁用日志统计，减少进程间通信
                enforce_eager=True,
            )
            safe_dtype = _pick_safe_vllm_dtype(kwargs.get('dtype'))
            if safe_dtype is not None:
                llm_kwargs['dtype'] = safe_dtype

            self.llm = LLM(**llm_kwargs)
            
            self.sampling_params = SamplingParams(
                temperature=kwargs.get('temperature', 0.1),
                top_p=kwargs.get('top_p', 0.9),
                max_tokens=kwargs.get('max_tokens', 256),
                stop=["\n\n", "###"],
                logprobs=5  # 返回top-5 logprobs用于reward计算
            )
        elif mode == 'api':
            # API模式配置 - 保存参数，延迟创建配置对象
            self.api_kwargs = {
                'api_key': kwargs.get('api_key', ''),
                'api_url': kwargs.get('api_url', 'https://api.deepseek.com'),
                'model_name': kwargs.get('model_name', 'deepseek-chat'),
                'temperature': kwargs.get('temperature', 0.1),
                'max_tokens': kwargs.get('max_tokens', 256),
                'max_workers': kwargs.get('max_workers', 5)
            }
    
    def create_prompt(self, input_data: ClassificationInput) -> str:
        """创建分类prompt（带结构特征）"""
        return PromptTemplates.format_classification_prompt(
            topic_name=input_data.topic_name,
            current_summary=input_data.current_node_summary,
            article_content=input_data.article_content,
            child_summaries=input_data.child_summaries,
            child_num_children=input_data.child_num_children,
            child_max_depth=input_data.child_max_depth,
            current_depth=input_data.current_depth,
            num_children=input_data.num_children
        )
    
    def parse_output(self, response: str, num_categories: int) -> Optional[ClassificationOutput]:
        """
        解析分类输出
        
        Args:
            response: 模型返回的文本
            num_categories: 类别数量
            
        Returns:
            ClassificationOutput对象，解析失败返回None
        """
        parsed = PromptTemplates.parse_classification_output(response, num_categories)
        if parsed is None:
            return None
        
        return ClassificationOutput(
            selected_indices=parsed['selected_indices'],
            need_new=parsed['need_new'],
            merge_with=parsed.get('merge_with'),
            new_node_direction=parsed.get('new_node_direction', {}) or {},
            merge_candidate_probs=parsed.get('merge_candidate_probs', {}) or {},
            raw_response=response
        )
    
    def classify(self, input_data: ClassificationInput) -> ClassificationOutput:
        """
        执行分类
        
        Args:
            input_data: 分类输入
            
        Returns:
            ClassificationOutput对象
        """
        if self.mode == 'model':
            # 创建prompt
            prompt = self.create_prompt(input_data)
            
            # 调用模型
            outputs = self.llm.generate([prompt], self.sampling_params, use_tqdm=False)
            response = outputs[0].outputs[0].text
            
            # 解析输出
            num_categories = len(input_data.child_summaries)
            result = self.parse_output(response, num_categories)
            
            if result is None:
                # 解析失败，返回默认值
                return ClassificationOutput(
                    selected_indices=[],
                    need_new=True,
                    merge_with=None,
                    raw_response=response
                )
            
            return result
        else:
            raise ValueError(f"不支持的模式: {self.mode}")
    
    def classify_with_multiple_samples(
        self, 
        inputs: List[ClassificationInput],
        n: int = 1
    ) -> List[List[ClassificationOutput]]:
        """
        批量分类，每个输入采样n个结果
        
        Args:
            inputs: 分类输入列表
            n: 每个输入采样的结果数
            
        Returns:
            List[List[ClassificationOutput]]，外层list对应inputs，内层list是n个采样结果
        """
        if self.mode == 'model':
            # 创建prompts
            prompts = [self.create_prompt(inp) for inp in inputs]
            
            # 设置采样参数（采样n个结果）
            from vllm import SamplingParams
            sampling_params = SamplingParams(
                temperature=self.sampling_params.temperature,
                top_p=self.sampling_params.top_p,
                max_tokens=self.sampling_params.max_tokens,
                stop=self.sampling_params.stop,
                logprobs=self.sampling_params.logprobs,
                n=n  # 关键：采样n个结果
            )
            
            # 批量调用模型
            outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)
            
            # 解析输出
            results = []
            for i, (inp, output) in enumerate(zip(inputs, outputs)):
                num_categories = len(inp.child_summaries)
                sample_results = []
                
                # 处理n个采样结果
                for sample_output in output.outputs:
                    response = sample_output.text
                    result = self.parse_output(response, num_categories)
                    
                    if result is None:
                        # 解析失败，使用默认值
                        result = ClassificationOutput(
                            selected_indices=[],
                            need_new=True,
                            merge_with=None,
                            raw_response=response
                        )
                    
                    sample_results.append(result)
                
                results.append(sample_results)
            
            return results
        else:
            raise ValueError(f"不支持的模式: {self.mode}")
    
    def classify_batch(self, inputs: List[ClassificationInput]) -> List[ClassificationOutput]:
        """
        批量分类
        
        Args:
            inputs: 分类输入列表
            
        Returns:
            分类输出列表
        """
        if self.mode == 'model':
            # 创建prompts
            prompts = [self.create_prompt(inp) for inp in inputs]
            
            # 批量调用模型
            outputs = self.llm.generate(prompts, self.sampling_params, use_tqdm=False)
            
            # 解析输出
            results = []
            for i, output in enumerate(outputs):
                response = output.outputs[0].text
                num_categories = len(inputs[i].child_summaries)
                result = self.parse_output(response, num_categories)
                
                if result is None:
                    # 解析失败，使用默认值
                    result = ClassificationOutput(
                        selected_indices=[],
                        need_new=True,
                        merge_with=None,
                        raw_response=response
                    )
                
                results.append(result)
            
            return results
        else:
            raise ValueError(f"不支持的模式: {self.mode}")
    
    def classify_with_sampling(
        self,
        input_data: ClassificationInput,
        n: int = 1,
        temperature: float = 0.7
    ) -> List[ClassificationOutput]:
        """
        执行分类并采样多个结果（用于轨迹采样）
        
        Args:
            input_data: 分类输入
            n: 采样数量
            temperature: 采样温度
            
        Returns:
            ClassificationOutput列表
        """
        if self.mode == 'model':
            from vllm import SamplingParams
            
            # 创建prompt
            prompt = self.create_prompt(input_data)
            
            # 修改采样参数以支持采样
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=0.9,
                max_tokens=256,
                stop=["\n\n", "###"],
                n=n,  # 采样n个结果
                logprobs=5
            )
            
            # 调用模型
            outputs = self.llm.generate([prompt], sampling_params, use_tqdm=False)
            
            # 解析所有采样结果
            results = []
            num_categories = len(input_data.child_summaries)
            for output_choice in outputs[0].outputs:
                response = output_choice.text
                result = self.parse_output(response, num_categories)
                
                if result is not None:
                    results.append(result)
            
            # 如果没有成功解析任何结果，返回默认值
            if not results:
                results = [ClassificationOutput(
                    selected_indices=[],
                    need_new=True,
                    raw_response=""
                )]
            
            return results
        elif self.mode == 'api':
            # API模式 - 采样多次
            sys.path.append(str(Path(__file__).parent.parent))
            from modeling.deepseek_api import DeepSeekAPIClient, DeepSeekConfig
            
            # 创建API配置
            api_config = DeepSeekConfig(
                api_key=self.api_kwargs['api_key'],
                base_url=self.api_kwargs['api_url'].replace('/chat/completions', ''),
                model=self.api_kwargs['model_name'],
                temperature=self.api_kwargs['temperature'],
                max_output_tokens=self.api_kwargs['max_tokens'],
                max_concurrent_jobs=self.api_kwargs['max_workers']
            )
            
            # 创建API客户端
            client = DeepSeekAPIClient(api_config)
            
            # 创建prompt
            prompt = self.create_prompt(input_data)
            
            # 调用API多次以获得多个采样结果
            prompts = [prompt] * n
            responses = client.run_prompts_to_texts(prompts, show_progress=False)
            
            # 解析所有结果
            results = []
            num_categories = len(input_data.child_summaries)
            for response in responses:
                result = self.parse_output(response, num_categories)
                if result is not None:
                    results.append(result)
            
            # 如果没有成功解析任何结果，返回默认值
            if not results:
                results = [ClassificationOutput(
                    selected_indices=[],
                    need_new=True,
                    raw_response=""
                )]
            
            return results
        else:
            raise ValueError(f"不支持的模式: {self.mode}")
    
    def get_logprobs(self, input_data: ClassificationInput) -> Optional[Dict]:
        """
        获取分类输出的log概率（用于reward计算）
        
        Args:
            input_data: 分类输入
            
        Returns:
            包含每个类别"Yes"概率的字典
        """
        if self.mode != 'model':
            return None
        
        # 创建prompt
        prompt = self.create_prompt(input_data)
        
        # 调用模型并获取logprobs
        outputs = self.llm.generate([prompt], self.sampling_params, use_tqdm=False)
        
        # 提取logprobs
        output = outputs[0].outputs[0]
        logprobs_data = output.logprobs
        
        if logprobs_data is None:
            return None
        
        # 解析每一行的Yes概率
        # 格式: "Category 0: Yes\nCategory 1: No\n..."
        # 我们需要找到每个"Yes"或"No" token的位置并提取概率
        num_categories = len(input_data.child_summaries)
        category_yes_logprobs = {}
        
        # 这里简化处理：假设每行只有一个Yes/No token
        # 实际实现中可能需要更复杂的解析逻辑
        try:
            text = output.text
            lines = text.strip().split('\n')
            
            for i, line in enumerate(lines):
                if line.startswith('Category'):
                    # 提取Yes/No
                    if 'Yes' in line:
                        # 这里需要从logprobs中提取对应位置的概率
                        # 简化：使用固定值（实际需要从vLLM的logprobs中提取）
                        category_yes_logprobs[i] = 0.0  # placeholder
                elif line.startswith('NEW'):
                    if 'Yes' in line:
                        category_yes_logprobs['NEW'] = 0.0  # placeholder
            
            return category_yes_logprobs
        except Exception as e:
            return None

    def classify_with_logprobs(
        self,
        input_data: ClassificationInput
    ) -> Tuple:
        """
        执行分类并返回logprobs
        
        Args:
            input_data: 分类输入
            
        Returns:
            (classification_output, scores)
            - classification_output: ClassificationOutput对象
            - scores: Dict[Union[int, str], float]，每个类别的logP(Yes)
        """
        if self.mode == 'model':
            from vllm import SamplingParams
            
            # 创建prompt
            prompt = self.create_prompt(input_data)
            
            # 设置sampling参数，要求返回logprobs
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=200,
                logprobs=5,  # 返回top-5 token的logprobs
                n=1
            )
            
            # 生成
            outputs = self.llm.generate([prompt], sampling_params, use_tqdm=False)
            
            if not outputs or not outputs[0].outputs:
                return None, {}
            
            output = outputs[0].outputs[0]
            response_text = output.text
            
            # 解析分类输出
            num_categories = len(input_data.child_summaries)
            classification_output = self.parse_output(response_text, num_categories)
            
            if classification_output is None:
                return None, {}
            
            # 从logprobs中提取每个类别的Yes概率
            scores = self._extract_yes_probs_from_logprobs(
                output.logprobs, response_text, num_categories
            )
            
            return classification_output, scores
        
        else:
            # API模式，无法获取logprobs
            outputs = self.classify_with_sampling(input_data, n=1)
            if not outputs:
                return None, {}
            
            classification_output = outputs[0]
            
            # API模式无法获取logprobs，使用0作为占位
            num_categories = len(input_data.child_summaries)
            scores = {}
            for i in range(num_categories):
                scores[i] = 0.0
            scores['NEW'] = 0.0
            
            return classification_output, scores
    
    def _extract_yes_probs_from_logprobs(
        self,
        logprobs_list: List,
        response_text: str,
        num_children: int
    ) -> Dict:
        """
        从logprobs中提取每个类别的Yes概率
        
        分类系统输出格式：
        Category 0: Yes
        Category 1: No
        Category 2: Yes
        NEW: No
        
        需要找到每一行的"Yes"或"No"，提取其log概率
        """
        scores = {}
        
        # 解析response_text，找到每个类别的Yes/No位置
        lines = response_text.strip().split('\n')
        
        # 当前token位置
        token_pos = 0
        
        for line in lines:
            if ':' not in line:
                # 跳过空行
                token_pos += max(1, len(line.split()))
                continue
            
            parts = line.split(':', 1)
            if len(parts) != 2:
                token_pos += max(1, len(line.split()))
                continue
            
            category_part = parts[0].strip()
            answer_part = parts[1].strip()
            
            # 确定是哪个类别
            category_idx = None
            if category_part.startswith('Category'):
                try:
                    idx = int(category_part.split()[1])
                    if 0 <= idx < num_children:
                        category_idx = idx
                except:
                    pass
            elif category_part == 'NEW':
                category_idx = 'NEW'
            
            if category_idx is None:
                token_pos += max(1, len(line.split()))
                continue
            
            # 在logprobs中找到对应的Yes/No token
            yes_logprob = -float('inf')
            found_yes_no = False
            
            # 估算这一行的token数量
            line_token_count = len(line.split())
            search_range = min(line_token_count + 5, len(logprobs_list) - token_pos)
            
            for offset in range(search_range):
                if token_pos + offset >= len(logprobs_list):
                    break
                
                token_logprobs = logprobs_list[token_pos + offset]
                
                # 检查这个位置的top tokens
                for token_id, logprob in token_logprobs.items():
                    token_str = self._decode_token(token_id).strip().lower()
                    
                    # 检查是否是Yes或No token
                    if 'yes' in token_str or 'no' in token_str:
                        if 'yes' in token_str:
                            yes_logprob = logprob.logprob
                        else:
                            # 找到No，在备选tokens中查找Yes的logprob
                            for alt_token_id, alt_logprob in token_logprobs.items():
                                alt_token_str = self._decode_token(alt_token_id).strip().lower()
                                if 'yes' in alt_token_str:
                                    yes_logprob = alt_logprob.logprob
                                    break
                        
                        found_yes_no = True
                        token_pos += offset + 1
                        break
                
                if found_yes_no:
                    break
            
            if not found_yes_no:
                token_pos += line_token_count
            
            scores[category_idx] = yes_logprob if yes_logprob > -float('inf') else -4.0
        
        # 确保所有类别都有score
        for i in range(num_children):
            if i not in scores:
                scores[i] = -10.0
        if 'NEW' not in scores:
            scores['NEW'] = -10.0
        
        return scores
    
    def _decode_token(self, token_id: int) -> str:
        """解码token ID为字符串"""
        try:
            if hasattr(self.llm, 'get_tokenizer'):
                tokenizer = self.llm.get_tokenizer()
                return tokenizer.decode([token_id])
        except:
            pass
        return ""
