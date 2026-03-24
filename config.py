"""
配置管理模块
"""
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import json


@dataclass
class PathConfig:
    """路径配置"""
    # 输入文件
    structures_file: str = '/mnt/literism/tree/data/wikipedia_structures_final.json'
    references_file: str = '/mnt/literism/tree/data/wikipedia_references_final.json'
    
    # 输出目录
    output_base: str = '/mnt/literism/tree/summary_output'
    data_dir: str = None  # 自动设置为 output_base/data
    summaries_dir: str = None  # 自动设置为 output_base/summaries
    models_dir: str = None  # 自动设置为 output_base/models
    inference_dir: str = None  # 自动设置为 output_base/inference
    
    # 模型路径
    base_model: str = '/home/literism/model/Qwen2.5-7B-Instruct'
    
    def __post_init__(self):
        base = Path(self.output_base)
        if self.data_dir is None:
            self.data_dir = str(base / 'data')
        if self.summaries_dir is None:
            self.summaries_dir = str(base / 'summaries')
        if self.models_dir is None:
            self.models_dir = str(base / 'models')
        if self.inference_dir is None:
            self.inference_dir = str(base / 'inference')


@dataclass
class DataSplitConfig:
    """数据划分配置"""
    target_test_size: int = 250  # 每个类别选择文章数最接近此值的topic作为test
    seed: int = 42


@dataclass
class SummaryConfig:
    """总结生成配置"""
    max_content_length: int = 2048  # 节点内容最大长度（拼接子节点时）
    api_url: str = "https://api.deepseek.com/chat/completions"
    api_key: str = ""  # 需要设置
    model_name: str = "deepseek-chat"
    max_workers: int = 10  # 并发worker数
    temperature: float = 0.7
    max_tokens: int = 500


@dataclass
class DataPrepareConfig:
    """训练数据准备配置"""
    # 分类生成系统数据配置
    # 数据比例 [type1, type2, type3, type4]
    # Type1: 删除部分正确类别
    # Type2: 保留所有类别
    # Type3: 叶子节点（空候选）
    # Type4: 需要新类（空候选+生成新类）
    classify_generator_ratios: list = field(default_factory=lambda: [2, 4, 1, 1])
    classify_generator_total_samples: int = 8000
    classify_generator_delete_multiple_ratio: float = 0.1  # Type1中删除多个类别的比例
    max_new_categories: int = 2  # 单次最多生成的新类数量
    
    # 总结更新系统数据配置
    updater_total_samples: int = 10000
    updater_update_ratio: float = 0.5  # 需要更新的样本比例
    # 1 - updater_update_ratio 为不需要更新的样本比例
    
    # 快速生成模式配置（用于总结更新系统数据集）
    updater_fast_type1_ratio: float = 0.5  # Type1（叶子+文章）的比例
    updater_fast_type2_ratio: float = 0.5  # Type2（中间节点+子节点）的比例，应该 type1 + type2 = 1
    updater_fast_perturb_prob: float = 0.5  # summary扰动概率
    updater_fast_max_sentence_tokens: int = 5  # 短句最大token数
    
    # 层级采样配置
    # 第一层：从topic根节点直接分出的子节点
    # 第二层及以后：更深层级的节点
    layer1_ratio: float = 0.3  # 第一层数据的比例
    # layer2_ratio自动为 1 - layer1_ratio
    
    # 并行配置（用于总结更新数据集构建）
    max_parallel_topics: int = 4  # 并行处理的topic数量
    
    seed: int = 42


@dataclass
class TrainingConfig:
    """训练配置"""
    # LoRA配置
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # 训练参数
    num_epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    
    # 其他
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    max_length: int = 4096
    bf16: bool = True
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4
    
    # 量化配置
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"


@dataclass
class InferenceConfig:
    """推理配置"""
    split: str = 'test'  # 要推理的数据集划分
    max_refs: Optional[int] = None  # 每个topic最多处理的references数量（测试用）
    max_depth: int = 10  # 最大树深度
    
    # vLLM配置 - 两个模型分别占用不同GPU
    # 分类生成系统使用GPU 0，总结更新系统使用GPU 1
    classify_generator_gpu_id: int = 0
    updater_gpu_id: int = 1
    max_model_len: int = 16384
    gpu_memory_utilization: float = 0.9  # 每个模型占其GPU的90%
    
    # 生成配置
    temperature: float = 0.1
    top_p: float = 0.9
    max_tokens: int = 512
    
    # Topic并行配置
    max_parallel_topics: int = 10  # 同时处理的topic数量
    
    # Summary去重配置
    similarity_threshold: float = 0.85  # 相似度阈值，超过则认为是重复


@dataclass
class DPOTrainingConfig:
    """DPO训练配置"""
    # 流程控制
    start_iteration: int = 0  # 从哪个迭代开始（0表示从头开始，>0表示断点续训）
    use_existing_dataset: bool = False  # 是否使用已有数据集（True=跳过采样直接训练）
    
    # 轨迹采样控制
    skip_trajectory_sampling: bool = False  # 是否跳过采样阶段（自动检测轨迹文件是否存在）
    skip_labeling: bool = False  # 是否跳过标注阶段（需要已有标注结果文件）
    
    # 迭代配置
    num_iterations: int = 3  # 训练迭代次数
    sampling_batch_sizes: list = field(default_factory=lambda: [300, 1000, 3000])  # 每次迭代采样的文章数
    
    # Prompt池配置
    sampling_batch_size: int = 32  # 批量推理的batch size
    sampling_timeout_seconds: float = 1.0  # 最长等待时间（秒）
    sampling_top_k: int = 4  # 每个prompt采样k个结果（Top-1 vs Rest）
    
    # 轨迹采样配置
    num_trajectories_per_article: int = 4  # 每篇文章采样的轨迹数
    trajectory_top_k: int = 4  # 从采样结果中取top-k条轨迹
    num_preference_pairs_per_article: int = 3  # 每篇文章构建的偏好对数量（Top-1 vs Rest）
    
    # 推理配置
    inference_batch_size: int = 32  # vLLM推理批次大小
    inference_wait_timeout: float = 1.0  # vLLM推理等待超时（秒）
    
    # Reward计算参数
    reward_beta: float = 2.0  # α(classify(NEW)) = β
    reward_gamma: float = 2.0  # α(summary_update at depth d) = γ/d
    reward_lambda: float = 0.1  # R_global = R_margin + λ * R_len
    reward_margin_penalty: float = 10.0  # R_margin = -penalty if C_pos is empty
    
    # Advantage计算参数
    advantage_temperature: float = 1.0  # 控制advantage weighting的锐度（τ）
    
    # DPO训练参数
    learning_rate: float = 1e-5
    num_epochs_per_iteration: int = 1  # 每次迭代训练的epoch数
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    warmup_ratio: float = 0.1
    logging_steps: int = 10
    save_steps: int = 500
    max_length: int = 16384
    bf16: bool = True
    gradient_checkpointing: bool = True
    load_in_4bit: bool = False  # 是否使用4bit量化


@dataclass
class LabelingConfig:
    """标注系统配置"""
    mode: str = 'api'  # 标注模式：'api' 或 'local'
    local_model_path: Optional[str] = None  # 本地模型路径（local模式）
    tensor_parallel_size: int = 2  # vllm使用的显卡数量
    gpu_memory_utilization: float = 0.9  # GPU内存利用率
    max_model_len: int = 16384  # 最大序列长度
    
    # 评估配置
    eval_samples: int = 100  # 评估时抽取的样本数
    max_disagreement_samples: int = 20  # 保存的不一致案例数量上限
    
    # 训练配置
    train_labeling_model: bool = False  # 是否训练本地标注模型（仅在iteration 0时）
    train_labeling_max_samples: int = 10000  # 训练标注模型的最大样本数


@dataclass
class SummaryBasedConfig:
    """总配置类"""
    path: PathConfig = field(default_factory=PathConfig)
    data_split: DataSplitConfig = field(default_factory=DataSplitConfig)
    summary: SummaryConfig = field(default_factory=SummaryConfig)
    data_prepare: DataPrepareConfig = field(default_factory=DataPrepareConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    dpo_training: DPOTrainingConfig = field(default_factory=DPOTrainingConfig)
    labeling: LabelingConfig = field(default_factory=LabelingConfig)
    
    @classmethod
    def from_json(cls, json_path: str) -> 'SummaryBasedConfig':
        """从JSON文件加载配置"""
        with open(json_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return cls(
            path=PathConfig(**config_dict.get('path', {})),
            data_split=DataSplitConfig(**config_dict.get('data_split', {})),
            summary=SummaryConfig(**config_dict.get('summary', {})),
            data_prepare=DataPrepareConfig(**config_dict.get('data_prepare', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            inference=InferenceConfig(**config_dict.get('inference', {})),
            dpo_training=DPOTrainingConfig(**config_dict.get('dpo_training', {})),
            labeling=LabelingConfig(**config_dict.get('labeling', {}))
        )
    
    def to_json(self, json_path: str):
        """保存配置到JSON文件"""
        config_dict = {
            'path': self.path.__dict__,
            'data_split': self.data_split.__dict__,
            'summary': self.summary.__dict__,
            'data_prepare': self.data_prepare.__dict__,
            'training': self.training.__dict__,
            'inference': self.inference.__dict__,
            'dpo_training': self.dpo_training.__dict__,
            'labeling': self.labeling.__dict__
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

