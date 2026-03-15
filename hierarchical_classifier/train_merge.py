"""
训练合并系统模型
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


class MergeSystemTrainer:
    """合并系统训练器"""
    
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
            train_data_file: 训练数据文件（json格式）
            val_data_file: 验证数据文件（json格式）
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
            # 量化配置
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
        """加载模型和分词器"""
        print("\n加载模型和分词器...")
        
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            trust_remote_code=True
        )
        
        # 设置padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 量化配置
        quant_config = BitsAndBytesConfig(
            load_in_4bit=self.config['quantization']['load_in_4bit'],
            bnb_4bit_compute_dtype=self.config['quantization']['bnb_4bit_compute_dtype'],
            bnb_4bit_use_double_quant=self.config['quantization']['bnb_4bit_use_double_quant'],
            bnb_4bit_quant_type=self.config['quantization']['bnb_4bit_quant_type']
        )
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=quant_config,
            device_map='auto',
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        
        # 启用gradient checkpointing
        if self.config['training']['gradient_checkpointing']:
            model.gradient_checkpointing_enable()
        
        return model, tokenizer
    
    def setup_trainer(self, model, tokenizer, train_dataset, val_dataset):
        """设置训练器"""
        print("\n设置训练器...")
        
        # LoRA配置
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config['lora']['r'],
            lora_alpha=self.config['lora']['lora_alpha'],
            lora_dropout=self.config['lora']['lora_dropout'],
            target_modules=self.config['lora']['target_modules'],
            bias=self.config['lora']['bias'],
        )
        
        # 训练配置
        training_args = SFTConfig(
            output_dir=str(self.output_dir),
            num_train_epochs=self.config['training']['num_epochs'],
            per_device_train_batch_size=self.config['training']['batch_size'],
            per_device_eval_batch_size=self.config['training']['batch_size'],
            gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
            learning_rate=self.config['training']['learning_rate'],
            warmup_ratio=self.config['training']['warmup_ratio'],
            logging_steps=self.config['training']['logging_steps'],
            save_steps=self.config['training']['save_steps'],
            eval_steps=self.config['training']['eval_steps'],
            save_total_limit=self.config['training']['save_total_limit'],
            max_seq_length=self.config['training']['max_length'],
            bf16=self.config['training']['bf16'],
            gradient_checkpointing=self.config['training']['gradient_checkpointing'],
            dataloader_num_workers=self.config['training']['dataloader_num_workers'],
            # SFT特定参数
            dataset_text_field=None,  # 我们会使用formatting_func
            packing=False,
            # 评估设置
            eval_strategy='steps',
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            # 其他
            remove_unused_columns=False,
            report_to=['tensorboard'],
        )
        
        # 格式化函数
        def formatting_func(example):
            """格式化样本为训练文本"""
            return f"{example['prompt']}{example['completion']}{tokenizer.eos_token}"
        
        # 创建训练器
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            peft_config=peft_config,
            tokenizer=tokenizer,
            formatting_func=formatting_func,
        )
        
        return trainer
    
    def train(self):
        """执行训练"""
        print("="*80)
        print("开始训练合并系统...")
        print("="*80)
        
        # 加载数据
        train_dataset, val_dataset = self.load_datasets()
        
        # 加载模型
        model, tokenizer = self.load_model_and_tokenizer()
        
        # 设置训练器
        trainer = self.setup_trainer(model, tokenizer, train_dataset, val_dataset)
        
        # 训练
        print("\n开始训练...")
        trainer.train()
        
        # 保存最终模型
        final_model_dir = self.output_dir / 'final_model'
        print(f"\n保存最终模型到: {final_model_dir}")
        trainer.save_model(str(final_model_dir))
        tokenizer.save_pretrained(str(final_model_dir))
        
        print("\n="*80)
        print("训练完成！")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description='训练合并系统')
    
    parser.add_argument(
        '--base_model',
        type=str,
        required=True,
        help='基础模型路径'
    )
    parser.add_argument(
        '--train_data',
        type=str,
        required=True,
        help='训练数据文件'
    )
    parser.add_argument(
        '--val_data',
        type=str,
        required=True,
        help='验证数据文件'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='输出目录'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='配置文件（JSON格式）'
    )
    
    args = parser.parse_args()
    
    # 加载配置
    config = None
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    
    # 创建训练器并训练
    trainer = MergeSystemTrainer(
        base_model=args.base_model,
        train_data_file=args.train_data,
        val_data_file=args.val_data,
        output_dir=args.output_dir,
        config=config
    )
    
    trainer.train()


if __name__ == '__main__':
    main()

