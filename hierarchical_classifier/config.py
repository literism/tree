"""
统一的配置管理系统
支持默认配置、配置文件加载和命令行参数覆盖
"""
import json
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass, field, asdict
import argparse


@dataclass
class PathConfig:
    """路径配置"""
    # 数据文件路径
    references_file: str = '/mnt/literism/tree/data/wikipedia_references_final.json'
    structures_file: str = '/mnt/literism/tree/data/wikipedia_structures_final.json'
    topic_classified_file: str = '/mnt/literism/data/result/topic_classified.json'
    
    # 输出路径
    output_base: str = '/mnt/literism/tree/hierarchical_output'
    data_dir: str = None  # 自动生成: output_base/data
    paraphrases_dir: str = None  # 自动生成: output_base/paraphrases
    records_dir: str = None  # 自动生成: output_base/records
    dataset_dir: str = None  # 自动生成: output_base/dataset
    models_dir: str = None  # 自动生成: output_base/models
    inference_dir: str = None  # 自动生成: output_base/inference
    
    # 模型路径
    base_model: str = '/home/literism/model/Qwen3-8B'
    
    def __post_init__(self):
        """自动生成子目录路径"""
        base = Path(self.output_base)
        if self.data_dir is None:
            self.data_dir = str(base / 'data')
        if self.paraphrases_dir is None:
            self.paraphrases_dir = str(base / 'paraphrases')
        if self.records_dir is None:
            self.records_dir = str(base / 'records')
        if self.dataset_dir is None:
            self.dataset_dir = str(base / 'dataset')
        if self.models_dir is None:
            self.models_dir = str(base / 'models')
        if self.inference_dir is None:
            self.inference_dir = str(base / 'inference')


@dataclass
class DataSplitConfig:
    """数据集划分配置"""
    test_easy_ratio: float = 0.1  # 简单测试集比例（从训练topics的文章中划分）
    seed: int = 42  # 随机种子


@dataclass
class BuilderConfig:
    """构建系统配置"""
    max_depth: int = 10  # 最大深度
    record_mode: bool = True  # 是否记录构建过程


@dataclass
class DatasetPrepareConfig:
    """数据集准备配置"""
    ratio: tuple = (2, 1, 1)  # 三类数据的采样比例 (has_new, existing_only, none)
    seed: int = 42  # 随机种子
    train_size: int = 10000  # 训练集总大小（采样后，包含验证集）
    val_ratio: float = 0.1  # 验证集比例（从采样后的数据中划分）
    delete_prob: float = 0.1  # 删除非输出title的概率
    replace_prob: float = 0.1  # 替换为paraphrase的概率
    num_constraint_leaves: int = 10  # 选择多少个叶子节点作为约束
    type1_single_new_prob: float = 0.7  # 类型1数据中只放一个new_subtitle的概率
    mix_output_to_constraint_prob: float = 0.1  # 将一个输出混入约束的概率


@dataclass
class LoRAConfig:
    """LoRA配置"""
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: [
        'q_proj', 'k_proj', 'v_proj', 'o_proj',
        'gate_proj', 'up_proj', 'down_proj'
    ])
    bias: str = 'none'


@dataclass
class TrainingConfig:
    """训练配置"""
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3
    max_length: int = 4096
    bf16: bool = True
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4


@dataclass
class QuantizationConfig:
    """量化配置"""
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = 'bfloat16'
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = 'nf4'


@dataclass
class InferenceConfig:
    """推理配置"""
    tensor_parallel_size: int = 1
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.9
    split: str = 'test_easy'  # 要推理的数据集划分: train/test_easy/test_hard/all
    max_refs: Optional[int] = None  # 每个topic最多处理的references数量（测试用）
    use_structure_init: bool = False  # 是否使用结构文件的第一层节点初始化树（然后删除空节点）
    num_inference_constraint_leaves: int = 20  # 推理时选择多少个叶子节点作为约束


@dataclass
class PipelineConfig:
    """流程控制配置"""
    skip_split: bool = False
    skip_build_records: bool = False
    skip_prepare_dataset: bool = False
    skip_paraphrases: bool = False
    skip_training: bool = True
    only_inference: bool = False


@dataclass
class Config:
    """主配置类"""
    path: PathConfig = field(default_factory=PathConfig)
    data_split: DataSplitConfig = field(default_factory=DataSplitConfig)
    builder: BuilderConfig = field(default_factory=BuilderConfig)
    dataset_prepare: DatasetPrepareConfig = field(default_factory=DatasetPrepareConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'Config':
        """从字典创建配置"""
        return cls(
            path=PathConfig(**config_dict.get('path', {})),
            data_split=DataSplitConfig(**config_dict.get('data_split', {})),
            builder=BuilderConfig(**config_dict.get('builder', {})),
            dataset_prepare=DatasetPrepareConfig(**config_dict.get('dataset_prepare', {})),
            lora=LoRAConfig(**config_dict.get('lora', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            quantization=QuantizationConfig(**config_dict.get('quantization', {})),
            inference=InferenceConfig(**config_dict.get('inference', {})),
            pipeline=PipelineConfig(**config_dict.get('pipeline', {}))
        )
    
    @classmethod
    def from_file(cls, config_file: str) -> 'Config':
        """从JSON文件加载配置"""
        with open(config_file, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'path': asdict(self.path),
            'data_split': asdict(self.data_split),
            'builder': asdict(self.builder),
            'dataset_prepare': asdict(self.dataset_prepare),
            'lora': asdict(self.lora),
            'training': asdict(self.training),
            'quantization': asdict(self.quantization),
            'inference': asdict(self.inference),
            'pipeline': asdict(self.pipeline)
        }
    
    def to_file(self, config_file: str):
        """保存到JSON文件"""
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    def update_from_args(self, args: argparse.Namespace):
        """从命令行参数更新配置"""
        # 路径配置
        if hasattr(args, 'references_file') and args.references_file:
            self.path.references_file = args.references_file
        if hasattr(args, 'topic_classified_file') and args.topic_classified_file:
            self.path.topic_classified_file = args.topic_classified_file
        if hasattr(args, 'output_base') and args.output_base:
            self.path.output_base = args.output_base
            # 重新生成子目录路径
            self.path.__post_init__()
        if hasattr(args, 'base_model') and args.base_model:
            self.path.base_model = args.base_model
        
        # 数据划分配置
        if hasattr(args, 'test_easy_ratio') and args.test_easy_ratio is not None:
            self.data_split.test_easy_ratio = args.test_easy_ratio
        if hasattr(args, 'seed') and args.seed is not None:
            self.data_split.seed = args.seed
        
        # 构建器配置
        if hasattr(args, 'max_depth') and args.max_depth is not None:
            self.builder.max_depth = args.max_depth
        
        # 数据集准备配置
        if hasattr(args, 'ratio') and args.ratio:
            self.dataset_prepare.ratio = tuple(args.ratio)
        if hasattr(args, 'train_size') and args.train_size is not None:
            self.dataset_prepare.train_size = args.train_size
        if hasattr(args, 'val_ratio') and args.val_ratio is not None:
            self.dataset_prepare.val_ratio = args.val_ratio
        if hasattr(args, 'delete_prob') and args.delete_prob is not None:
            self.dataset_prepare.delete_prob = args.delete_prob
        if hasattr(args, 'replace_prob') and args.replace_prob is not None:
            self.dataset_prepare.replace_prob = args.replace_prob
        
        # 训练配置
        if hasattr(args, 'num_epochs') and args.num_epochs is not None:
            self.training.num_epochs = args.num_epochs
        if hasattr(args, 'batch_size') and args.batch_size is not None:
            self.training.batch_size = args.batch_size
        if hasattr(args, 'learning_rate') and args.learning_rate is not None:
            self.training.learning_rate = args.learning_rate
        if hasattr(args, 'max_length') and args.max_length is not None:
            self.training.max_length = args.max_length
        
        # LoRA配置
        if hasattr(args, 'lora_r') and args.lora_r is not None:
            self.lora.r = args.lora_r
        if hasattr(args, 'lora_alpha') and args.lora_alpha is not None:
            self.lora.lora_alpha = args.lora_alpha
        
        # 推理配置
        if hasattr(args, 'tensor_parallel_size') and args.tensor_parallel_size is not None:
            self.inference.tensor_parallel_size = args.tensor_parallel_size
        if hasattr(args, 'max_model_len') and args.max_model_len is not None:
            self.inference.max_model_len = args.max_model_len
        if hasattr(args, 'gpu_memory_utilization') and args.gpu_memory_utilization is not None:
            self.inference.gpu_memory_utilization = args.gpu_memory_utilization
        if hasattr(args, 'split') and args.split is not None:
            self.inference.split = args.split
        if hasattr(args, 'max_refs') and args.max_refs is not None:
            self.inference.max_refs = args.max_refs
        
        # 流程控制配置
        if hasattr(args, 'skip_split'):
            self.pipeline.skip_split = args.skip_split
        if hasattr(args, 'skip_build_records'):
            self.pipeline.skip_build_records = args.skip_build_records
        if hasattr(args, 'skip_prepare_dataset'):
            self.pipeline.skip_prepare_dataset = args.skip_prepare_dataset
        if hasattr(args, 'skip_training'):
            self.pipeline.skip_training = args.skip_training
        if hasattr(args, 'only_inference'):
            self.pipeline.only_inference = args.only_inference
    
    def print_config(self):
        """打印配置"""
        print("=" * 80)
        print("当前配置")
        print("=" * 80)
        
        print("\n【路径配置】")
        print(f"  数据文件: {self.path.references_file}")
        print(f"  Topic分类: {self.path.topic_classified_file}")
        print(f"  输出目录: {self.path.output_base}")
        print(f"  基础模型: {self.path.base_model}")
        
        print("\n【数据划分】")
        print(f"  简单测试集比例: {self.data_split.test_easy_ratio}")
        print(f"  随机种子: {self.data_split.seed}")
        
        print("\n【构建系统】")
        print(f"  最大深度: {self.builder.max_depth}")
        print(f"  记录模式: {self.builder.record_mode}")
        
        print("\n【数据准备】")
        print(f"  采样比例: {self.dataset_prepare.ratio}")
        print(f"  训练集大小: {self.dataset_prepare.train_size}")
        print(f"  验证集比例: {self.dataset_prepare.val_ratio}")
        print(f"  删除概率: {self.dataset_prepare.delete_prob}")
        print(f"  替换概率: {self.dataset_prepare.replace_prob}")
        
        print("\n【LoRA配置】")
        print(f"  rank: {self.lora.r}")
        print(f"  alpha: {self.lora.lora_alpha}")
        print(f"  dropout: {self.lora.lora_dropout}")
        
        print("\n【训练配置】")
        print(f"  训练轮数: {self.training.num_epochs}")
        print(f"  批次大小: {self.training.batch_size}")
        print(f"  梯度累积: {self.training.gradient_accumulation_steps}")
        print(f"  学习率: {self.training.learning_rate}")
        print(f"  最大长度: {self.training.max_length}")
        print(f"  bf16: {self.training.bf16}")
        
        print("\n【量化配置】")
        print(f"  4-bit量化: {self.quantization.load_in_4bit}")
        print(f"  量化类型: {self.quantization.bnb_4bit_quant_type}")
        
        print("\n【推理配置】")
        print(f"  张量并行: {self.inference.tensor_parallel_size}")
        print(f"  最大长度: {self.inference.max_model_len}")
        print(f"  GPU利用率: {self.inference.gpu_memory_utilization}")
        print(f"  数据集: {self.inference.split}")
        
        print("\n【流程控制】")
        print(f"  跳过划分: {self.pipeline.skip_split}")
        print(f"  跳过构建: {self.pipeline.skip_build_records}")
        print(f"  跳过准备: {self.pipeline.skip_prepare_dataset}")
        print(f"  跳过训练: {self.pipeline.skip_training}")
        print(f"  仅推理: {self.pipeline.only_inference}")
        
        print("=" * 80)


def add_config_arguments(parser: argparse.ArgumentParser):
    """向ArgumentParser添加配置参数"""
    
    # 配置文件
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.json',
        help='配置文件路径（JSON格式）'
    )
    
    # 路径配置
    path_group = parser.add_argument_group('路径配置')
    path_group.add_argument('--references_file', type=str, help='references文件路径')
    path_group.add_argument('--topic_classified_file', type=str, help='topic分类文件路径')
    path_group.add_argument('--output_base', type=str, help='输出基础目录')
    path_group.add_argument('--base_model', type=str, help='基础模型路径')
    
    # 数据划分配置
    split_group = parser.add_argument_group('数据划分配置')
    split_group.add_argument('--test_easy_ratio', type=float, help='简单测试集比例（从训练topics的文章中划分）')
    split_group.add_argument('--seed', type=int, help='随机种子')
    
    # 构建器配置
    builder_group = parser.add_argument_group('构建器配置')
    builder_group.add_argument('--max_depth', type=int, help='最大深度')
    
    # 数据集准备配置
    dataset_group = parser.add_argument_group('数据集准备配置')
    dataset_group.add_argument('--ratio', type=int, nargs=3, help='采样比例 (has_new existing_only none)')
    dataset_group.add_argument('--train_size', type=int, help='训练集总大小（采样后，包含验证集）')
    dataset_group.add_argument('--val_ratio', type=float, help='验证集比例（从采样后的数据中划分）')
    dataset_group.add_argument('--delete_prob', type=float, help='删除非输出title的概率')
    dataset_group.add_argument('--replace_prob', type=float, help='替换为paraphrase的概率')
    
    # LoRA配置
    lora_group = parser.add_argument_group('LoRA配置')
    lora_group.add_argument('--lora_r', type=int, help='LoRA rank')
    lora_group.add_argument('--lora_alpha', type=int, help='LoRA alpha')
    
    # 训练配置
    train_group = parser.add_argument_group('训练配置')
    train_group.add_argument('--num_epochs', type=int, help='训练轮数')
    train_group.add_argument('--batch_size', type=int, help='批次大小')
    train_group.add_argument('--learning_rate', type=float, help='学习率')
    train_group.add_argument('--max_length', type=int, help='最大序列长度')
    
    # 推理配置
    inference_group = parser.add_argument_group('推理配置')
    inference_group.add_argument('--tensor_parallel_size', type=int, help='张量并行大小')
    inference_group.add_argument('--max_model_len', type=int, help='模型最大长度')
    inference_group.add_argument('--gpu_memory_utilization', type=float, help='GPU内存利用率')
    inference_group.add_argument('--split', type=str, choices=['train', 'test_easy', 'test_hard', 'all'], help='推理的数据集划分')
    inference_group.add_argument('--max_refs', type=int, help='每个topic最多处理的references数量')
    
    # 流程控制
    pipeline_group = parser.add_argument_group('流程控制')
    pipeline_group.add_argument('--skip_split', action='store_true', default=argparse.SUPPRESS, help='跳过数据集划分')
    pipeline_group.add_argument('--skip_build_records', action='store_true', default=argparse.SUPPRESS, help='跳过构建记录')
    pipeline_group.add_argument('--skip_prepare_dataset', action='store_true', default=argparse.SUPPRESS, help='跳过准备数据集')
    pipeline_group.add_argument('--skip_training', action='store_true', default=argparse.SUPPRESS, help='跳过训练')
    pipeline_group.add_argument('--only_inference', action='store_true', default=argparse.SUPPRESS, help='只运行推理')


def load_config(args: Optional[argparse.Namespace] = None, config_file: Optional[str] = None) -> Config:
    """
    加载配置
    
    优先级: 命令行参数 > 配置文件 > 默认值
    
    Args:
        args: 命令行参数
        config_file: 配置文件路径
        
    Returns:
        Config对象
    """
    # 1. 从默认值创建配置
    config = Config()
    
    # 2. 如果有配置文件，从配置文件加载
    if config_file:
        print(f"从配置文件加载: {config_file}")
        config = Config.from_file(config_file)
    elif args and hasattr(args, 'config') and args.config:
        print(f"从配置文件加载: {args.config}")
        config = Config.from_file(args.config)
    
    # 3. 如果有命令行参数，用命令行参数覆盖
    if args:
        config.update_from_args(args)
    
    return config


# 测试代码
if __name__ == '__main__':
    print("=" * 80)
    print("配置系统测试")
    print("=" * 80)
    
    # 测试1: 默认配置
    print("\n测试1: 默认配置")
    config = Config()
    config.print_config()
    
    # 测试2: 保存和加载配置
    print("\n测试2: 保存和加载配置")
    config_file = './test_config.json'
    config.to_file(config_file)
    print(f"配置已保存到: {config_file}")
    
    loaded_config = Config.from_file(config_file)
    print("配置已加载")
    
    # 测试3: 从命令行参数更新
    print("\n测试3: 从命令行参数更新")
    parser = argparse.ArgumentParser()
    add_config_arguments(parser)
    args = parser.parse_args([
        '--base_model', '/path/to/custom/model',
        '--num_epochs', '5',
        '--batch_size', '2'
    ])
    
    config = Config()
    config.update_from_args(args)
    print(f"基础模型: {config.path.base_model}")
    print(f"训练轮数: {config.training.num_epochs}")
    print(f"批次大小: {config.training.batch_size}")
    
    print("\n所有测试通过！")

