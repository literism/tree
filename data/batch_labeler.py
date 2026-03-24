"""
批量标注系统
支持两种模式：
1. API模式：使用DeepSeek API
2. Local模式：使用vllm加载本地模型
"""

import os
import sys
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import torch
from tqdm import tqdm

# 添加modeling目录到sys.path
modeling_path = str(Path(__file__).parent.parent / 'modeling')
if modeling_path not in sys.path:
    sys.path.append(modeling_path)
sys.path.append(str(Path(__file__).parent.parent))
from modeling.deepseek_api import DeepSeekAPIClient
from summary_based_classifier.llm.prompts import PromptTemplates


@dataclass
class LabelingRequest:
    """单个标注请求"""
    topic_name: str
    current_summary: str
    child_summaries: List[str]
    ground_truth_paths: List[str]
    metadata: Dict[str, Any] = None  # 用于存储额外信息（如trajectory_id, decision_point_id等）


@dataclass
class LabelingResult:
    """标注结果"""
    exceed_parent: Optional[List[int]]
    overlapping_pairs: Optional[List[List[int]]]
    correct_indices: List[int]
    need_new: bool
    raw_response: str
    success: bool = True  # 标注是否成功
    parsed_output: Optional[Dict] = None  # 保存解析后的完整字典
    metadata: Dict[str, Any] = None


class BatchLabeler:
    """批量标注系统"""
    
    def __init__(self, config, mode: str = 'api'):
        """
        Args:
            config: 配置对象
            mode: 'api' 或 'local'
        """
        self.config = config
        self.mode = mode
        self.model = None
        
        if mode == 'api':
            self._init_api_mode()
        elif mode == 'local':
            self._init_local_mode()
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'api' or 'local'")
    
    def _init_api_mode(self):
        """初始化API模式"""
        print("  初始化标注系统（API模式）...")
        
        # 创建DeepSeekConfig对象
        from modeling.deepseek_api import DeepSeekConfig
        deepseek_config = DeepSeekConfig(
            api_key=self.config.summary.api_key,
            base_url=self.config.summary.api_url,
            model=self.config.summary.model_name,
            temperature=self.config.summary.temperature,
            max_output_tokens=self.config.summary.max_tokens,
            max_concurrent_jobs=self.config.summary.max_workers
        )
        
        self.api_client = DeepSeekAPIClient(config=deepseek_config)
        print("  ✓ API模式初始化完成")
    
    def _init_local_mode(self):
        """初始化本地模型模式（使用vllm）"""
        print("  初始化标注系统（本地模型模式）...")
        
        if self.config.labeling.local_model_path is None:
            raise ValueError("local_model_path must be set in config for local mode")
        
        # 清理环境变量（避免干扰vllm的GPU分配）
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            original_cuda_devices = os.environ['CUDA_VISIBLE_DEVICES']
            del os.environ['CUDA_VISIBLE_DEVICES']
        else:
            original_cuda_devices = None
        
        try:
            from vllm import LLM, SamplingParams
            
            print(f"  加载模型: {self.config.labeling.local_model_path}")
            print(f"  使用 {self.config.labeling.tensor_parallel_size} 张显卡")
            
            self.model = LLM(
                model=self.config.labeling.local_model_path,
                tensor_parallel_size=self.config.labeling.tensor_parallel_size,
                gpu_memory_utilization=self.config.labeling.gpu_memory_utilization,
                max_model_len=self.config.labeling.max_model_len,
                trust_remote_code=True
            )
            
            # 创建采样参数（用于标注任务）
            # 对于Qwen模型，使用temperature=0.01而不是0（避免某些问题）
            # 不设置stop tokens，让模型自然结束输出
            self.sampling_params = SamplingParams(
                temperature=0.01,  # 接近0但不是0
                top_p=1.0,
                max_tokens=128, 
                stop=None,
                skip_special_tokens=True
            )
            
            print("  ✓ 本地模型加载完成")
        
        finally:
            # 恢复环境变量
            if original_cuda_devices is not None:
                os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_devices
    
    def label_batch(self, requests: List[LabelingRequest]) -> List[LabelingResult]:
        """
        批量标注
        
        Args:
            requests: 标注请求列表
        
        Returns:
            标注结果列表
        """
        if not requests:
            return []
        
        print(f"\n开始批量标注 {len(requests)} 个请求...")
        
        # 构建prompts
        prompts = []
        for req in requests:
            prompt = PromptTemplates.format_labeling_prompt(
                topic_name=req.topic_name,
                current_summary=req.current_summary,
                child_summaries=req.child_summaries,
                ground_truth_paths=req.ground_truth_paths
            )
            prompts.append(prompt)
        
        # 根据模式调用不同的推理方法
        if self.mode == 'api':
            responses = self._label_with_api(prompts)
        else:  # local
            responses = self._label_with_local_model(prompts)
        
        # 解析结果
        results = []
        failed_indices = []  # 记录解析失败的索引
        failed_prompts = []  # 记录解析失败的prompts
        
        for i, (req, response) in enumerate(zip(requests, responses)):
            parsed = PromptTemplates.parse_labeling_output(
                response,
                num_children=len(req.child_summaries)
            )
            
            if parsed is None:
                # 解析失败，记录索引和prompt
                failed_indices.append(i)
                failed_prompts.append(prompts[i])
                # 先用默认值占位
                result = LabelingResult(
                    exceed_parent=None,
                    overlapping_pairs=None,
                    correct_indices=[],
                    need_new=False,
                    raw_response=response,
                    success=False,
                    parsed_output=None,
                    metadata=req.metadata
                )
            else:
                result = LabelingResult(
                    exceed_parent=parsed['exceed_parent'],
                    overlapping_pairs=parsed['overlapping_pairs'],
                    correct_indices=parsed['correct_indices'],
                    need_new=parsed['need_new'],
                    raw_response=response,
                    success=True,
                    parsed_output=parsed,
                    metadata=req.metadata
                )
            
            results.append(result)
        
        # 如果是local模式且有解析失败的，用API模式补充
        if self.mode == 'local' and failed_indices:
            print(f"\n⚠ 检测到 {len(failed_indices)} 个解析失败的结果")
            print(f"  使用API模式重新标注这些样本...")
            
            # 初始化临时API客户端
            from modeling.deepseek_api import DeepSeekConfig
            deepseek_config = DeepSeekConfig(
                api_key=self.config.summary.api_key,
                base_url=self.config.summary.api_url,
                model=self.config.summary.model_name,
                temperature=self.config.summary.temperature,
                max_output_tokens=self.config.summary.max_tokens,
                max_concurrent_jobs=self.config.summary.max_workers
            )
            
            api_client = DeepSeekAPIClient(config=deepseek_config)
            
            # 用API重新标注失败的样本
            api_responses = api_client.run_prompts_to_texts(
                failed_prompts,
                show_progress=True
            )
            
            # 调试：检查API响应
            print(f"  API fallback返回 {len(api_responses)} 个响应")
            empty_api = sum(1 for r in api_responses if not r or len(r.strip()) == 0)
            print(f"  其中空响应: {empty_api}/{len(api_responses)}")
            
            # 更新失败的结果
            success_count = 0
            for idx, api_response in zip(failed_indices, api_responses):
                req = requests[idx]
                parsed = PromptTemplates.parse_labeling_output(
                    api_response,
                    num_children=len(req.child_summaries)
                )
                
                if parsed is not None:
                    # API标注成功，更新结果
                    results[idx] = LabelingResult(
                        exceed_parent=parsed['exceed_parent'],
                        overlapping_pairs=parsed['overlapping_pairs'],
                        correct_indices=parsed['correct_indices'],
                        need_new=parsed['need_new'],
                        raw_response=api_response,
                        success=True,
                        parsed_output=parsed,
                        metadata=req.metadata
                    )
                    success_count += 1
                else:
                    # API也失败了，保持默认值，但更新raw_response
                    results[idx].raw_response = f"[LOCAL FAILED] {results[idx].raw_response}\n[API FAILED] {api_response}"
            
            print(f"  ✓ API补充完成，成功修复 {success_count}/{len(failed_indices)} 个样本")
        
        print(f"✓ 批量标注完成")
        return results
    
    def _label_with_api(self, prompts: List[str]) -> List[str]:
        """使用API进行标注"""
        print("  使用API模式标注...")
        
        # 使用DeepSeekAPIClient批量处理
        responses = self.api_client.run_prompts_to_texts(
            prompts,
            show_progress=True  # 显示进度条
        )
        
        print(f"  收到responses数量: {len(responses)}")
        # 统计空响应
        empty_count = sum(1 for r in responses if not r or len(r.strip()) == 0)
        print(f"  空响应数量: {empty_count}/{len(responses)}")
        if empty_count > 0 and len(responses) > 0:
            # 打印第一个非空响应的前100字符用于调试
            non_empty = [r for r in responses if r and len(r.strip()) > 0]
            if non_empty:
                print(f"  示例响应（前100字符）: {non_empty[0][:100]}")
        
        return responses
    
    def _label_with_local_model(self, prompts: List[str]) -> List[str]:
        """使用本地模型进行标注"""
        print("  使用本地模型标注...")
        
        # vllm会自动批处理
        outputs = self.model.generate(prompts, self.sampling_params)
        
        # 提取文本结果
        responses = [output.outputs[0].text for output in outputs]
        
        return responses
    
    def cleanup(self):
        """清理资源"""
        if self.mode == 'local' and self.model is not None:
            print("\n  清理本地模型资源...")
            del self.model
            self.model = None
            
            # 清理显卡缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print("  ✓ 资源清理完成")
    
    def __del__(self):
        """析构函数"""
        self.cleanup()


def test_batch_labeler():
    """测试批量标注系统"""
    import json
    from types import SimpleNamespace
    
    # 创建测试配置
    config_dict = {
        'summary': {
            'api_url': 'https://api.deepseek.com/chat/completions',
            'api_key': 'your-api-key',
            'model_name': 'deepseek-chat',
            'max_workers': 10,
            'temperature': 0.7,
            'max_tokens': 2048
        },
        'labeling': {
            'mode': 'api',
            'local_model_path': None,
            'tensor_parallel_size': 2,
            'gpu_memory_utilization': 0.9,
            'max_model_len': 16384
        }
    }
    
    config = json.loads(json.dumps(config_dict), object_hook=lambda d: SimpleNamespace(**d))
    
    # 创建测试请求
    requests = [
        LabelingRequest(
            topic_name="Biology",
            current_summary="Study of living organisms",
            child_summaries=["Plants and photosynthesis", "Animals and behavior"],
            ground_truth_paths=["Biology - Plants - Trees"],
            metadata={'test_id': 1}
        ),
        LabelingRequest(
            topic_name="Physics",
            current_summary="Study of matter and energy",
            child_summaries=["Mechanics", "Thermodynamics", "Electromagnetism"],
            ground_truth_paths=["Physics - Mechanics - Newton's Laws"],
            metadata={'test_id': 2}
        )
    ]
    
    # 测试API模式
    print("=== 测试API模式 ===")
    labeler = BatchLabeler(config, mode='api')
    results = labeler.label_batch(requests)
    
    for i, result in enumerate(results):
        print(f"\n请求 {i+1}:")
        print(f"  EXCEED_PARENT: {result.exceed_parent}")
        print(f"  OVERLAPPING_PAIRS: {result.overlapping_pairs}")
        print(f"  CORRECT_INDICES: {result.correct_indices}")
        print(f"  NEED_NEW: {result.need_new}")
        print(f"  Metadata: {result.metadata}")


if __name__ == '__main__':
    test_batch_labeler()
