"""
结构树生成器的配置管理
"""
import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict


@dataclass
class StructureGeneratorConfig:
    """结构树生成器配置"""
    
    # 输入数据路径
    wiki_structure_file: str = '/mnt/literism/data/wiki_dataset/wiki_structure.jsonl'
    wiki_intro_file: str = '/mnt/literism/data/wiki_dataset/wiki_intro.jsonl'
    
    # 已有的topic划分
    dataset_split_file: str = '/mnt/literism/tree/hierarchical_output/data/dataset_split.json'
    
    # 输出路径
    output_base: str = '/mnt/literism/tree/structure_output'
    data_dir: str = None  # 自动生成
    models_dir: str = None  # 自动生成
    inference_dir: str = None  # 自动生成
    
    # 模型路径
    base_model: str = '/home/literism/model/Qwen3-8B'
    
    # 数据筛选阈值
    min_structure_nodes: int = 10  # 删除无用title后，最少节点数
    min_intro_length: int = 500  # 最少intro字符数
    
    # 数据集大小
    train_size: int = 10000
    val_ratio: float = 0.1
    
    # 训练配置
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    num_epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    max_length: int = 8192  # 结构树可能比较长
    
    # 推理配置
    max_new_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    
    def __post_init__(self):
        """自动生成子目录路径"""
        base = Path(self.output_base)
        if self.data_dir is None:
            self.data_dir = str(base / 'data')
        if self.models_dir is None:
            self.models_dir = str(base / 'models')
        if self.inference_dir is None:
            self.inference_dir = str(base / 'inference')
    
    @classmethod
    def from_json(cls, json_file: str) -> 'StructureGeneratorConfig':
        """从JSON文件加载配置"""
        with open(json_file, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_json(self, json_file: str):
        """保存配置到JSON文件"""
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)
    
    def print_config(self):
        """打印配置"""
        print("\n" + "="*60)
        print("结构树生成器配置")
        print("="*60)
        for key, value in asdict(self).items():
            print(f"  {key}: {value}")
        print("="*60 + "\n")


# 无用的Wikipedia标题（从parse_wikipedia_structure.py复制）
SKIP_TITLES = {
    'see also', 'see', 'notes', 'sources', 'external links',
    'further reading', 'bibliography', 'explanatory notes',
    'references'  # 添加references
}

