"""
分类系统
支持两种模式：真实模式（使用真实数据）和模型模式（使用训练的模型）
"""
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ClassificationInput:
    """分类输入"""
    topic_key: str  # 例如 "Person:Albert Einstein"
    reference_id: str  # 例如 "search_1"
    article_content: str  # 文章内容
    current_path: str  # 当前路径，例如 "Albert Einstein" 或 "Albert Einstein - Early life"
    existing_subtitles: List[str]  # 现有子标题列表
    constraint_paths: List[str] = None  # 约束路径列表（推理时使用）
    
    def __post_init__(self):
        if self.constraint_paths is None:
            self.constraint_paths = []


@dataclass
class ClassificationOutput:
    """分类输出"""
    selected_existing: List[str]  # 从现有子标题中选择的
    new_subtitles: List[str]  # 新增的子标题
    

class GroundTruthClassifier:
    """真实模式分类器：使用真实数据进行分类"""
    
    def __init__(self, references_file: str):
        """
        Args:
            references_file: wikipedia_references_final.json文件路径
        """
        print("初始化真实模式分类器...")
        with open(references_file, 'r', encoding='utf-8') as f:
            self.references_data = json.load(f)
        print(f"  - 加载 {len(self.references_data)} 个topics")
        
    def _parse_path(self, path: str) -> List[str]:
        """解析路径为层级列表"""
        return [p.strip() for p in path.split(' - ')]
    
    def _match_path_level(
        self, 
        full_path: List[str], 
        current_path: List[str]
    ) -> Optional[str]:
        """
        检查full_path是否匹配current_path，并返回下一层的标题
        
        Args:
            full_path: 完整路径，例如 ["Albert Einstein", "Early life", "Education"]
            current_path: 当前路径，例如 ["Albert Einstein"]
            
        Returns:
            下一层的标题，如果匹配的话。例如 "Early life"
        """
        # 检查current_path是否是full_path的前缀
        if len(current_path) >= len(full_path):
            return None
        
        for i, title in enumerate(current_path):
            if i >= len(full_path) or full_path[i] != title:
                return None
        
        # 返回下一层的标题
        return full_path[len(current_path)]
    
    def classify_single(self, input_data: ClassificationInput, track_errors: bool = False) -> tuple[ClassificationOutput, Optional[bool]]:
        """
        对单篇文章进行分类
        
        Args:
            input_data: 分类输入
            track_errors: 是否跟踪错误（真实模式始终成功）
            
        Returns:
            (分类输出, 是否成功) 真实模式始终返回True
        """
        topic_key = input_data.topic_key
        reference_id = input_data.reference_id
        current_path_str = input_data.current_path
        existing_subtitles = set(input_data.existing_subtitles)
        
        # 获取该reference的所有paths
        if topic_key not in self.references_data:
            if track_errors:
                return ClassificationOutput(selected_existing=[], new_subtitles=[]), True
            return ClassificationOutput(selected_existing=[], new_subtitles=[]), None
        
        topic_data = self.references_data[topic_key]
        references = topic_data.get('references', {})
        
        if reference_id not in references:
            if track_errors:
                return ClassificationOutput(selected_existing=[], new_subtitles=[]), True
            return ClassificationOutput(selected_existing=[], new_subtitles=[]), None
        
        reference = references[reference_id]
        paths = reference.get('paths', [])
        
        # 解析当前路径
        current_path = self._parse_path(current_path_str)
        
        # 收集下一层的所有标题
        next_level_titles = set()
        for path_str in paths:
            full_path = self._parse_path(path_str)
            next_title = self._match_path_level(full_path, current_path)
            if next_title:
                next_level_titles.add(next_title)
        
        # 分类为现有和新增
        selected_existing = []
        new_subtitles = []
        
        for title in next_level_titles:
            if title in existing_subtitles:
                selected_existing.append(title)
            else:
                new_subtitles.append(title)
        
        # 排序以保证一致性
        selected_existing.sort()
        new_subtitles.sort()
        
        output = ClassificationOutput(
            selected_existing=selected_existing,
            new_subtitles=new_subtitles
        )
        
        if track_errors:
            return output, True
        return output, None
    
    def classify_batch(self, inputs: List[ClassificationInput], track_errors: bool = False) -> tuple[List[ClassificationOutput], Optional[Dict]]:
        """
        批量分类
        
        Args:
            inputs: 输入列表
            track_errors: 是否跟踪错误（真实模式始终成功）
            
        Returns:
            (输出列表, 错误统计) 真实模式的错误统计全部为成功
        """
        outputs = []
        for input_data in inputs:
            output, _ = self.classify_single(input_data, track_errors=False)
            outputs.append(output)
        
        error_stats = None
        if track_errors:
            error_stats = {
                'success': len(inputs),
                'failed': 0
            }
        
        return outputs, error_stats


class ModelClassifier:
    """模型模式分类器：使用训练的模型进行分类"""
    
    def __init__(
        self, 
        model_path: str,
        tensor_parallel_size: int = 1,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.9
    ):
        """
        Args:
            model_path: 模型路径
            tensor_parallel_size: 张量并行大小
            max_model_len: 最大序列长度
            gpu_memory_utilization: GPU内存利用率
        """
        print("初始化模型分类器...")
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError("需要安装vllm")
        
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True
        )
        
        self.sampling_params = SamplingParams(
            temperature=0.0,  # 确定性输出
            max_tokens=256,  # 减少最大token数，只需要输出JSON
            stop=["</s>", "<|endoftext|>", "<|im_end|>", "\n\nTOPIC", "\n\nTASK", "\n\nCRITICAL"]  # 添加更多stop tokens
        )
        
        print(f"  - 模型加载完成: {model_path}")
    
    def _create_prompt(self, input_data: ClassificationInput) -> str:
        """
        创建模型输入prompt（与训练数据格式完全一致，不使用chat template）
        
        Args:
            input_data: 分类输入
            
        Returns:
            格式化的prompt
        """
        existing_str = ", ".join(input_data.existing_subtitles) if input_data.existing_subtitles else "None"
        constraint_str = ", ".join(input_data.constraint_paths) if input_data.constraint_paths else "None"
        
        prompt = f"""You are a hierarchical content classifier for Wikipedia articles.

TOPIC PATH: {input_data.current_path}
This is the current hierarchical path in the Wikipedia structure. Any subtitles you identify MUST be direct children of this topic path.

EXISTING SUBTITLES: {existing_str}
These are subtitles that already exist under the current topic path.

CONSTRAINT SUBTITLES: {constraint_str}
These are subtitles from other branches in the hierarchy. You should NOT output or create subtitles that are the same as or similar to these constraint subtitles, as they belong to different parts of the structure.

ARTICLE CONTENT:
{input_data.article_content[:3000]}

TASK:
1. If the article relates to any EXISTING subtitles, list them in "selected_existing"
2. If the article introduces NEW content that needs new subtitles under "{input_data.current_path}", list them in "new_subtitles"
3. All subtitles must be direct children of "{input_data.current_path}" in the Wikipedia hierarchy
4. Use concise, Wikipedia-style subtitle names (1-10 words)
5. If no subtitles apply, return empty arrays
6. IMPORTANT: Do NOT create subtitles similar to the CONSTRAINT SUBTITLES listed above

CRITICAL: Output ONLY a valid JSON object. No explanation, no additional text.

JSON OUTPUT:
{{
  "selected_existing": [],
  "new_subtitles": []
}}"""
        
        return prompt
    
    def _parse_output(self, output_text: str, track_errors: bool = False) -> tuple[ClassificationOutput, bool]:
        """
        解析模型输出（改进版，更健壮）
        
        Args:
            output_text: 模型生成的文本
            track_errors: 是否跟踪错误（返回是否成功）
            
        Returns:
            (分类输出, 是否成功解析)
        """
        try:
            # 清理输出
            output_text = output_text.strip()
            
            # 移除<think>标签（Qwen模型可能输出）
            if '<think>' in output_text:
                # 找到</think>后的内容
                think_end = output_text.find('</think>')
                if think_end != -1:
                    output_text = output_text[think_end + len('</think>'):].strip()
            
            # 特殊情况：直接输出空数组
            if output_text == '[]' or output_text == '{}':
                return ClassificationOutput(selected_existing=[], new_subtitles=[]), True
            
            # 方法1: 尝试直接解析整个输出（如果模型输出就是JSON）
            try:
                result = json.loads(output_text)
                if isinstance(result, dict):
                    selected_existing = result.get('selected_existing', [])
                    new_subtitles = result.get('new_subtitles', [])
                    
                    # 处理None和非列表情况
                    if selected_existing == "None" or selected_existing is None:
                        selected_existing = []
                    if new_subtitles == "None" or new_subtitles is None:
                        new_subtitles = []
                    
                    # 确保是列表
                    if not isinstance(selected_existing, list):
                        selected_existing = []
                    if not isinstance(new_subtitles, list):
                        new_subtitles = []
                    
                    # 过滤空字符串和None
                    selected_existing = [s for s in selected_existing if s and s != "None"]
                    new_subtitles = [s for s in new_subtitles if s and s != "None"]
                    
                    return ClassificationOutput(
                        selected_existing=selected_existing,
                        new_subtitles=new_subtitles
                    ), True
            except:
                pass
            
            # 方法2: 查找JSON块（处理有额外文本的情况）
            # 使用更健壮的方法查找匹配的括号对
            start_idx = output_text.find('{')
            if start_idx != -1:
                # 从start_idx开始，找到匹配的右括号
                brace_count = 0
                end_idx = -1
                for i in range(start_idx, len(output_text)):
                    if output_text[i] == '{':
                        brace_count += 1
                    elif output_text[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i
                            break
                
                if end_idx != -1:
                    json_str = output_text[start_idx:end_idx+1]
                    result = json.loads(json_str)
                    
                    if isinstance(result, dict):
                        selected_existing = result.get('selected_existing', [])
                        new_subtitles = result.get('new_subtitles', [])
                        
                        # 处理None和非列表情况
                        if selected_existing == "None" or selected_existing is None:
                            selected_existing = []
                        if new_subtitles == "None" or new_subtitles is None:
                            new_subtitles = []
                        
                        # 确保是列表
                        if not isinstance(selected_existing, list):
                            selected_existing = []
                        if not isinstance(new_subtitles, list):
                            new_subtitles = []
                        
                        # 过滤空字符串和None
                        selected_existing = [s for s in selected_existing if s and s != "None"]
                        new_subtitles = [s for s in new_subtitles if s and s != "None"]
                        
                        return ClassificationOutput(
                            selected_existing=selected_existing,
                            new_subtitles=new_subtitles
                        ), True
            
        except Exception as e:
            if not track_errors:  # 只在非跟踪模式下打印详细错误
                # 只打印前200个字符，避免输出过长
                preview = output_text[:200] + "..." if len(output_text) > 200 else output_text
                print(f"警告: 解析模型输出失败: {e}")
                print(f"  原始输出预览: {preview}")
        
        # 如果所有方法都失败，返回空结果
        return ClassificationOutput(selected_existing=[], new_subtitles=[]), False
    
    def classify_batch(self, inputs: List[ClassificationInput], track_errors: bool = False) -> tuple[List[ClassificationOutput], Optional[Dict]]:
        """
        批量分类
        
        Args:
            inputs: 输入列表
            track_errors: 是否跟踪错误统计
            
        Returns:
            (输出列表, 错误统计字典)
        """
        # 创建prompts
        prompts = [self._create_prompt(input_data) for input_data in inputs]
        
        # 批量推理
        outputs = self.llm.generate(prompts, self.sampling_params, use_tqdm=False)
        
        # 解析输出
        results = []
        error_stats = {'success': 0, 'failed': 0} if track_errors else None
        
        for input_data, output in zip(inputs, outputs):
            output_text = output.outputs[0].text
            result, success = self._parse_output(output_text, track_errors)
            results.append(result)
            
            if track_errors:
                # 检查是否成功解析
                if not success:
                    error_stats['failed'] += 1
                else:
                    # 额外检查：根节点不应该被判断为叶子节点
                    is_root = '-' not in input_data.current_path
                    is_leaf = (not result.selected_existing) and (not result.new_subtitles)
                    
                    if is_root and is_leaf:
                        # 根节点被错误判断为叶子节点
                        error_stats['failed'] += 1
                    else:
                        error_stats['success'] += 1
        
        return results, error_stats
    
    def classify_single(self, input_data: ClassificationInput, track_errors: bool = False) -> tuple[ClassificationOutput, Optional[bool]]:
        """
        对单篇文章进行分类
        
        Args:
            input_data: 分类输入
            track_errors: 是否跟踪错误
            
        Returns:
            (分类输出, 是否成功) 如果track_errors=False，第二个值为None
        """
        results, error_stats = self.classify_batch([input_data], track_errors)
        if track_errors:
            return results[0], error_stats['success'] > 0
        return results[0], None


class Classifier:
    """
    统一的分类器接口
    支持真实模式和模型模式
    """
    
    def __init__(
        self,
        mode: str = "ground_truth",
        references_file: Optional[str] = None,
        model_path: Optional[str] = None,
        **model_kwargs
    ):
        """
        Args:
            mode: "ground_truth" 或 "model"
            references_file: references文件路径（真实模式需要）
            model_path: 模型路径（模型模式需要）
            **model_kwargs: 模型的其他参数
        """
        self.mode = mode
        
        if mode == "ground_truth":
            if references_file is None:
                raise ValueError("真实模式需要提供references_file")
            self.classifier = GroundTruthClassifier(references_file)
        elif mode == "model":
            if model_path is None:
                raise ValueError("模型模式需要提供model_path")
            self.classifier = ModelClassifier(model_path, **model_kwargs)
        else:
            raise ValueError(f"未知模式: {mode}")
        
        print(f"分类器初始化完成，模式: {mode}")
    
    def classify_single(self, input_data: ClassificationInput, track_errors: bool = False):
        """单个分类"""
        return self.classifier.classify_single(input_data, track_errors)
    
    def classify_batch(self, inputs: List[ClassificationInput], track_errors: bool = False):
        """批量分类"""
        return self.classifier.classify_batch(inputs, track_errors)


# 测试代码
if __name__ == '__main__':
    # 测试真实模式
    print("=" * 80)
    print("测试真实模式分类器")
    print("=" * 80)
    
    classifier = Classifier(
        mode="ground_truth",
        references_file="/mnt/literism/tree/data/wikipedia_references_final.json"
    )
    
    # 创建测试输入
    test_input = ClassificationInput(
        topic_key="Book:The Hobbit",
        reference_id="search_84",
        article_content="Some content about The Hobbit",
        current_path="The Hobbit",
        existing_subtitles=["Concept and creation", "Influences", "Legacy"]
    )
    
    # 单个分类
    output = classifier.classify_single(test_input)
    print(f"\n输入:")
    print(f"  Topic: {test_input.topic_key}")
    print(f"  Reference: {test_input.reference_id}")
    print(f"  Path: {test_input.current_path}")
    print(f"  Existing: {test_input.existing_subtitles}")
    print(f"\n输出:")
    print(f"  Selected existing: {output.selected_existing}")
    print(f"  New subtitles: {output.new_subtitles}")
    
    # 批量分类测试
    batch_inputs = [test_input] * 3
    batch_outputs = classifier.classify_batch(batch_inputs)
    print(f"\n批量分类测试: {len(batch_outputs)} 个结果")

