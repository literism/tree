"""
训练模型
使用TRL库的SFTTrainer进行LoRA微调
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
from pathlib import Path
from typing import Dict, Optional
import argparse

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig


class HierarchicalClassifierTrainer:
    """层次化分类器训练器"""
    
    def __init__(
        self,
        base_model: str,
        train_data_file: str,
        val_data_file: str,
        output_dir: str,
        config: Optional[Dict] = None
    ):
        """
        Args:
            base_model: 基础模型路径
            train_data_file: 训练数据文件（jsonl格式）
            val_data_file: 验证数据文件（jsonl格式）
            output_dir: 输出目录
            config: 训练配置
        """
        self.base_model = base_model
        self.train_data_file = train_data_file
        self.val_data_file = val_data_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 默认配置
        default_config = {
            # LoRA配置
            'lora': {
                'r': 16,
                'lora_alpha': 32,
                'lora_dropout': 0.05,
                'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 
                                   'gate_proj', 'up_proj', 'down_proj'],
                'bias': 'none',
            },
            # 训练配置
            'training': {
                'num_epochs': 3,
                'batch_size': 4,
                'gradient_accumulation_steps': 4,
                'learning_rate': 2e-4,
                'warmup_ratio': 0.03,
                'logging_steps': 10,
                'save_steps': 100,
                'eval_steps': 100,
                'save_total_limit': 3,
                'max_length': 4096,
                'bf16': True,
                'gradient_checkpointing': True,
                'dataloader_num_workers': 4,
            },
            # 量化配置（可选）
            'quantization': {
                'load_in_4bit': True,
                'bnb_4bit_compute_dtype': torch.bfloat16,
                'bnb_4bit_use_double_quant': True,
                'bnb_4bit_quant_type': 'nf4',
            }
        }
        
        # 合并配置
        if config:
            default_config.update(config)
        self.config = default_config
        
        # 保存配置
        config_file = self.output_dir / 'train_config.json'
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        print(f"配置保存到: {config_file}")
    
    def load_datasets(self):
        """加载数据集"""
        print("\n加载数据集...")
        
        train_dataset = load_dataset(
            'json',
            data_files=self.train_data_file,
            split='train'
        )
        print(f"  - 训练集: {len(train_dataset)} 条")
        
        val_dataset = load_dataset(
            'json',
            data_files=self.val_data_file,
            split='train'
        )
        print(f"  - 验证集: {len(val_dataset)} 条")
        
        return train_dataset, val_dataset
    
    def load_model_and_tokenizer(self):
        """加载模型和tokenizer"""
        print(f"\n加载模型和tokenizer: {self.base_model}")
        
        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            trust_remote_code=True
        )
        
        # 设置padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"  - Tokenizer加载完成")
        
        # 量化配置
        quantization_config = None
        if self.config['quantization']['load_in_4bit']:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.config['quantization']['bnb_4bit_compute_dtype'],
                bnb_4bit_use_double_quant=self.config['quantization']['bnb_4bit_use_double_quant'],
                bnb_4bit_quant_type=self.config['quantization']['bnb_4bit_quant_type'],
            )
            print(f"  - 使用4-bit量化")
        
        # 模型
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map='auto',
        )
        
        print(f"  - 模型加载完成")
        
        return model, tokenizer
    
    def train(self):
        """执行训练"""
        print("="*80)
        print("开始训练层次化分类器")
        print("="*80)
        
        # 1. 加载数据集
        train_dataset, val_dataset = self.load_datasets()
        
        # 2. 加载模型和tokenizer
        model, tokenizer = self.load_model_and_tokenizer()
        
        # 3. LoRA配置
        print("\n配置LoRA...")
        lora_config = self.config['lora']
        peft_config = LoraConfig(
            r=lora_config['r'],
            lora_alpha=lora_config['lora_alpha'],
            lora_dropout=lora_config['lora_dropout'],
            target_modules=lora_config['target_modules'],
            bias=lora_config['bias'],
            task_type=TaskType.CAUSAL_LM
        )
        print(f"  - LoRA rank: {lora_config['r']}")
        print(f"  - LoRA alpha: {lora_config['lora_alpha']}")
        
        # 4. 训练参数
        print("\n配置训练参数...")
        train_config = self.config['training']
        training_args = SFTConfig(
            output_dir=str(self.output_dir / 'adapter'),
            num_train_epochs=train_config['num_epochs'],
            per_device_train_batch_size=train_config['batch_size'],
            per_device_eval_batch_size=train_config['batch_size'],
            gradient_accumulation_steps=train_config['gradient_accumulation_steps'],
            learning_rate=train_config['learning_rate'],
            warmup_ratio=train_config['warmup_ratio'],
            logging_steps=train_config['logging_steps'],
            save_steps=train_config['save_steps'],
            eval_steps=train_config['eval_steps'],
            eval_strategy='steps',
            save_strategy='steps',
            save_total_limit=train_config['save_total_limit'],
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            bf16=train_config['bf16'],
            gradient_checkpointing=train_config['gradient_checkpointing'],
            dataloader_num_workers=train_config['dataloader_num_workers'],
            report_to='none',
            max_length=train_config['max_length'],
        )
        
        # 5. SFTTrainer
        print("\n创建SFTTrainer...")
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            peft_config=peft_config,
            processing_class=tokenizer,
        )
        
        # 6. 训练
        print("\n" + "="*80)
        print("开始训练...")
        print("="*80)
        trainer.train()
        
        # 7. 保存模型
        print("\n" + "="*80)
        print("保存模型...")
        print("="*80)
        
        # 保存LoRA adapter
        adapter_path = self.output_dir / 'adapter'
        print(f"  保存LoRA adapter到: {adapter_path}")
        trainer.model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)
        
        # 合并并保存完整模型
        merged_model_path = self.output_dir / 'model'
        print(f"  合并并保存完整模型到: {merged_model_path}")
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(merged_model_path)
        tokenizer.save_pretrained(merged_model_path)
        
        print("\n" + "="*80)
        print("训练完成！")
        print("="*80)
        print(f"LoRA adapter: {adapter_path}")
        print(f"完整模型: {merged_model_path}")


def main():
    parser = argparse.ArgumentParser(description='训练层次化分类器')
    parser.add_argument(
        '--base_model',
        type=str,
        required=True,
        help='基础模型路径'
    )
    parser.add_argument(
        '--train_data',
        type=str,
        default='./dataset/train_dataset.jsonl',
        help='训练数据文件'
    )
    parser.add_argument(
        '--val_data',
        type=str,
        default='./dataset/val_dataset.jsonl',
        help='验证数据文件'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./models/hierarchical_classifier',
        help='输出目录'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='配置文件路径（JSON格式）'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        help='训练轮数'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        help='批次大小'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        help='学习率'
    )
    
    args = parser.parse_args()
    
    # 加载配置
    config = None
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        config = {}
    
    # 命令行参数覆盖配置文件
    if args.num_epochs:
        config.setdefault('training', {})['num_epochs'] = args.num_epochs
    if args.batch_size:
        config.setdefault('training', {})['batch_size'] = args.batch_size
    if args.learning_rate:
        config.setdefault('training', {})['learning_rate'] = args.learning_rate
    
    # 创建训练器
    trainer = HierarchicalClassifierTrainer(
        base_model=args.base_model,
        train_data_file=args.train_data,
        val_data_file=args.val_data,
        output_dir=args.output_dir,
        config=config
    )
    
    # 训练
    trainer.train()


if __name__ == '__main__':
    main()

