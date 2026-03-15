"""
训练总结生成系统
使用LoRA进行SFT训练
"""
import argparse
import json
import os
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import torch
from summary_based_classifier.config import SummaryBasedConfig


def format_prompt_completion(example):
    """格式化prompt和completion为训练文本"""
    return {
        "text": example["prompt"] + example["completion"]
    }


def main():
    parser = argparse.ArgumentParser(description='训练总结生成系统')
    parser.add_argument('--base_model', type=str, required=True, help='基础模型路径')
    parser.add_argument('--train_data', type=str, required=True, help='训练数据文件')
    parser.add_argument('--val_data', type=str, required=True, help='验证数据文件')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--max_samples', type=int, default=None, help='最大训练样本数（None表示使用全部数据）')
    
    args = parser.parse_args()
    
    print("="*80)
    print("训练总结生成系统")
    print("="*80)
    print(f"基础模型: {args.base_model}")
    print(f"训练数据: {args.train_data}")
    print(f"验证数据: {args.val_data}")
    print(f"输出目录: {args.output_dir}")
    
    # 加载配置
    if args.config:
        config = SummaryBasedConfig.from_json(args.config)
        training_config = config.training
    else:
        # 使用默认配置
        from summary_based_classifier.config import TrainingConfig
        training_config = TrainingConfig()
    
    print(f"\n训练配置:")
    print(f"  - LoRA rank: {training_config.lora_r}")
    print(f"  - LoRA alpha: {training_config.lora_alpha}")
    print(f"  - Learning rate: {training_config.learning_rate}")
    print(f"  - Epochs: {training_config.num_epochs}")
    print(f"  - Batch size: {training_config.batch_size}")
    print(f"  - Gradient accumulation: {training_config.gradient_accumulation_steps}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置datasets缓存目录到移动硬盘，避免系统盘满
    cache_dir = "/mnt/literism/.cache/huggingface/datasets"
    os.makedirs(cache_dir, exist_ok=True)
    os.environ['HF_DATASETS_CACHE'] = cache_dir
    
    # 加载数据集
    print("\n加载数据集...")
    print(f"  - 缓存目录: {cache_dir}")
    dataset = load_dataset(
        'json',
        data_files={
            'train': args.train_data,
            'validation': args.val_data
        },
        cache_dir=cache_dir
    )
    
    print(f"  - 训练集大小: {len(dataset['train'])}")
    print(f"  - 验证集大小: {len(dataset['validation'])}")
    
    # 限制数据量（如果指定）
    if args.max_samples is not None and args.max_samples > 0:
        print(f"\n限制训练数据量到: {args.max_samples}")
        if len(dataset['train']) > args.max_samples:
            dataset['train'] = dataset['train'].select(range(args.max_samples))
            print(f"  - 训练集已限制到: {len(dataset['train'])}")
        
        # 验证集也按比例缩减
        max_val_samples = max(int(args.max_samples * 0.1), 10)
        if len(dataset['validation']) > max_val_samples:
            dataset['validation'] = dataset['validation'].select(range(max_val_samples))
            print(f"  - 验证集已限制到: {len(dataset['validation'])}")
    
    # 加载tokenizer
    print("\n加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 格式化数据：拼接prompt和completion为text
    print("\n格式化数据集...")
    dataset = dataset.map(
        format_prompt_completion, 
        remove_columns=['prompt', 'completion']
    )
    print(f"  - 格式化完成")
    
    # 配置量化（可选）
    quantization_config = None
    if training_config.load_in_4bit:
        print("使用4-bit量化...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(torch, training_config.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=training_config.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=training_config.bnb_4bit_quant_type
        )
    
    # 加载模型
    print("\n加载基础模型...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if training_config.bf16 else torch.float16
    )
    
    # 准备模型用于训练（如果使用量化）
    if training_config.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    
    # 配置LoRA
    print("\n配置LoRA...")
    peft_config = LoraConfig(
        r=training_config.lora_r,
        lora_alpha=training_config.lora_alpha,
        lora_dropout=training_config.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=training_config.num_epochs,
        per_device_train_batch_size=training_config.batch_size,
        per_device_eval_batch_size=training_config.batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        learning_rate=training_config.learning_rate,
        warmup_ratio=training_config.warmup_ratio,
        logging_steps=training_config.logging_steps,
        save_steps=training_config.save_steps,
        eval_steps=training_config.eval_steps,
        save_total_limit=training_config.save_total_limit,
        bf16=training_config.bf16,
        gradient_checkpointing=training_config.gradient_checkpointing,
        dataloader_num_workers=training_config.dataloader_num_workers,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",  # 不使用wandb等
        remove_unused_columns=False
    )
    
    # 创建训练器
    print("\n创建训练器...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        processing_class=tokenizer,
    )
    
    # 开始训练
    print("\n" + "="*80)
    print("开始训练...")
    print("="*80)
    
    trainer.train()
    
    # 保存最终模型
    print("\n" + "="*80)
    print("保存模型...")
    print("="*80)
    
    # 保存LoRA adapter
    adapter_path = os.path.join(args.output_dir, "adapter")
    print(f"  保存LoRA adapter到: {adapter_path}")
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    
    # 合并并保存完整模型
    final_model_path = os.path.join(args.output_dir, "final_model")
    print(f"  合并并保存完整模型...")
    
    # Step 1: 加载干净的base模型
    print("    Step 1/5: 加载干净的base模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    
    # Step 2: 加载LoRA adapter
    print("    Step 2/5: 加载LoRA adapter...")
    from peft import PeftModel
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)
    
    # Step 3: 合并LoRA（此后是纯transformers模型）
    print("    Step 3/5: 合并LoRA...")
    merged_model = peft_model.merge_and_unload()
    
    # Step 4: 对齐embedding vocab（关键步骤！在合并后进行）
    print(f"    Step 4/5: 对齐embedding层...")
    num_added = len(tokenizer.get_added_vocab())
    true_vocab_size = tokenizer.vocab_size + num_added
    print(f"      - 原始embedding size: {merged_model.get_input_embeddings().weight.shape[0]}")
    print(f"      - Tokenizer vocab size: {true_vocab_size}")
    if merged_model.get_input_embeddings().weight.shape[0] != true_vocab_size:
        merged_model.resize_token_embeddings(true_vocab_size)
        print(f"      - 调整后embedding size: {merged_model.get_input_embeddings().weight.shape[0]}")
        print("      ✓ Embedding层对齐完成")
    else:
        print("      ✓ Embedding层已对齐")
    merged_model.config.vocab_size = true_vocab_size
    
    # Step 5: 保存最终模型
    print("    Step 5/5: 保存最终模型...")
    merged_model.save_pretrained(final_model_path, safe_serialization=True)
    tokenizer.save_pretrained(final_model_path)
    
    print(f"  ✓ 模型已保存到: {final_model_path}")
    
    print("\n" + "="*80)
    print(f"训练完成！模型已保存到: {final_model_path}")
    print("="*80)


if __name__ == '__main__':
    main()
