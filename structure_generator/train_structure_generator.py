"""
训练结构树生成模型
使用TRL库的SFTTrainer进行LoRA微调
"""
import os
import json
from pathlib import Path
from typing import Dict, Optional
import argparse

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig

from config import StructureGeneratorConfig


class StructureGeneratorTrainer:
    """结构树生成器训练器"""
    
    def __init__(self, config: StructureGeneratorConfig):
        self.config = config
        
        # 创建输出目录
        self.output_dir = Path(config.models_dir) / 'structure_generator'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"模型输出目录: {self.output_dir}")
    
    def load_datasets(self):
        """加载数据集（JSON格式，与hierarchical_classifier一致）"""
        print("\n加载数据集...")
        
        # 使用JSON格式文件（与hierarchical_classifier一致）
        train_file = Path(self.config.data_dir) / 'train_dataset.json'
        val_file = Path(self.config.data_dir) / 'val_dataset.json'
        
        train_dataset = load_dataset(
            'json',
            data_files=str(train_file),
            split='train'
        )
        print(f"  - 训练集: {len(train_dataset)} 条")
        
        val_dataset = load_dataset(
            'json',
            data_files=str(val_file),
            split='train'
        )
        print(f"  - 验证集: {len(val_dataset)} 条")
        
        return train_dataset, val_dataset
    
    def load_model_and_tokenizer(self):
        """加载模型和tokenizer"""
        print(f"\n加载模型和tokenizer: {self.config.base_model}")
        
        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True
        )
        
        # 设置padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"  - Tokenizer加载完成")
        
        # 量化配置
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
        )
        print(f"  - 使用4-bit量化")
        
        # 模型
        model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map='auto',
            use_cache=False,  # 训练时关闭KV cache，节省显存
        )
        
        # 启用梯度检查点
        model.gradient_checkpointing_enable()
        
        print(f"  - 模型加载完成")
        
        return model, tokenizer
    
    def train(self):
        """执行训练"""
        print("="*80)
        print("开始训练结构树生成器")
        print("="*80)
        
        # 1. 加载数据集
        train_dataset, val_dataset = self.load_datasets()
        
        # 2. 加载模型和tokenizer
        model, tokenizer = self.load_model_and_tokenizer()
        
        # 3. LoRA配置
        print("\n配置LoRA...")
        peft_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                          'gate_proj', 'up_proj', 'down_proj'],
            bias='none',
            task_type=TaskType.CAUSAL_LM
        )
        print(f"  - LoRA rank: {self.config.lora_r}")
        print(f"  - LoRA alpha: {self.config.lora_alpha}")
        
        # 4. 训练参数
        print("\n配置训练参数...")
        training_args = SFTConfig(
            output_dir=str(self.output_dir / 'adapter'),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            logging_steps=50,
            save_steps=300,
            eval_steps=300,
            eval_strategy='steps',
            save_strategy='steps',
            save_total_limit=3,
            load_best_model_at_end=False,  # 改为False，减少内存占用
            metric_for_best_model='eval_loss',
            bf16=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant': False},  # 更好的内存管理
            dataloader_num_workers=0,  # 改为0，避免多进程内存问题
            dataloader_pin_memory=False,  # 关闭pin memory
            report_to='none',
            max_length=self.config.max_length,
            # 关键：防止内存泄漏
            eval_accumulation_steps=1,  # 评估时不累积，立即释放
            max_grad_norm=1.0,  # 梯度裁剪
        )
        
        print(f"  - Epochs: {self.config.num_epochs}")
        print(f"  - Batch size: {self.config.batch_size}")
        print(f"  - Gradient accumulation: {self.config.gradient_accumulation_steps}")
        print(f"  - Learning rate: {self.config.learning_rate}")
        print(f"  - Max length: {self.config.max_length}")
        
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
        
        # 清理显存
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        trainer.train()
        
        # 训练后清理显存
        gc.collect()
        torch.cuda.empty_cache()
        
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
        
        # 保存配置
        config_file = self.output_dir / 'config.json'
        self.config.to_json(str(config_file))
        print(f"配置文件: {config_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练结构树生成模型')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--num_epochs', type=int, help='训练轮数')
    parser.add_argument('--batch_size', type=int, help='批次大小')
    parser.add_argument('--learning_rate', type=float, help='学习率')
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config:
        config = StructureGeneratorConfig.from_json(args.config)
    else:
        config = StructureGeneratorConfig()
    
    # 命令行参数覆盖
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    
    # 打印配置
    config.print_config()
    
    # 训练
    trainer = StructureGeneratorTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()

