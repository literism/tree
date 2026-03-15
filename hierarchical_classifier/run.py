"""
使用统一配置系统的完整流程脚本
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path
from config import Config, add_config_arguments, load_config


def run_command(cmd, description, check=True):
    """运行命令"""
    print("\n" + "="*80)
    print(description)
    print("="*80)
    print(f"命令: {' '.join(str(c) for c in cmd)}")
    print()
    
    result = subprocess.run(cmd, check=check)
    return result


def main():
    parser = argparse.ArgumentParser(
        description='使用统一配置的完整训练和推理流程',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置
  python3 run.py
  
  # 使用自定义配置文件
  python3 run.py --config configs/fast_test.json
  
  # 使用配置文件并覆盖部分参数
  python3 run.py --config configs/default.json --num_epochs 5 --batch_size 8
  
  # 只运行推理
  python3 run.py --only_inference
  
  # 快速测试（限制数据量）
  python3 run.py --config configs/fast_test.json --max_refs 5
        """
    )
    
    # 添加配置参数
    add_config_arguments(parser)
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args)
    
    # 打印配置
    config.print_config()
    
    # 创建输出目录
    output_base = Path(config.path.output_base)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # 保存当前配置
    config_save_path = output_base / 'run_config.json'
    config.to_file(str(config_save_path))
    print(f"\n当前配置已保存到: {config_save_path}")
    
    # 如果只运行推理
    if config.pipeline.only_inference:
        model_path = Path(config.path.models_dir) / 'hierarchical_classifier' / 'model'
        if not model_path.exists():
            print(f"\n错误: 模型不存在: {model_path}")
            print("请先训练模型或指定正确的模型路径")
            return 1
        
        run_command(
            [
                'python3', 'inference.py',
                '--model_path', str(model_path),
                '--references_file', config.path.references_file,
                '--split_file', str(Path(config.path.data_dir) / 'dataset_split.json'),
                '--split', config.inference.split,
                '--output_dir', config.path.inference_dir,
                '--tensor_parallel_size', str(config.inference.tensor_parallel_size),
                '--max_model_len', str(config.inference.max_model_len),
                '--gpu_memory_utilization', str(config.inference.gpu_memory_utilization),
                '--max_depth', str(config.builder.max_depth),
                '--structures_file', config.path.structures_file
            ] + (['--max_refs', str(config.inference.max_refs)] if config.inference.max_refs else [])
              + (['--use_structure_init'] if config.inference.use_structure_init else []),
            "步骤: 推理"
        )
        
        # 评估
        pred_file = Path(config.path.inference_dir) / f'{config.inference.split}_trees.json'
        if pred_file.exists():
            eval_output = Path(config.path.inference_dir) / f'{config.inference.split}_evaluation.json'
            run_command(
                [
                    'python3', 'evaluate.py',
                    '--pred_file', str(pred_file),
                    '--true_file', config.path.structures_file,
                    '--output_file', str(eval_output)
                ],
                "步骤: 评估"
            )
            print(f"\n评估结果已保存到: {eval_output}")
        
        print("\n" + "="*80)
        print("推理完成！")
        print("="*80)
        return 0
    
    # 步骤 1: 划分数据集
    if not config.pipeline.skip_split:
        run_command(
            [
                'python3', 'data_split.py',
                '--references_file', config.path.references_file,
                '--topic_classified_file', config.path.topic_classified_file,
                '--output_dir', config.path.data_dir,
                '--test_easy_ratio', str(config.data_split.test_easy_ratio),
                '--seed', str(config.data_split.seed)
            ],
            "步骤 1: 划分数据集"
        )
    else:
        print("\n跳过步骤 1: 划分数据集")
    
    # 步骤 2: 生成Paraphrase
    skip_paraphrases = getattr(config.pipeline, 'skip_paraphrases', False)
    if not skip_paraphrases:
        run_command(
            [
                'python3', 'generate_paraphrases.py',
                '--structures_file', config.path.structures_file,
                '--output_dir', config.path.paraphrases_dir,
            ],
            "步骤 2: 生成标题Paraphrase"
        )
    else:
        print("\n跳过步骤 2: 生成Paraphrase")
    
    # 步骤 3: 准备训练数据集
    if not config.pipeline.skip_prepare_dataset:
        cmd = [
            'python3', 'prepare_dataset.py',
            '--references_file', config.path.references_file,
            '--structures_file', config.path.structures_file,
            '--paraphrases_dir', config.path.paraphrases_dir,
            '--dataset_split_file', str(Path(config.path.data_dir) / 'dataset_split.json'),
            '--output_dir', config.path.dataset_dir,
            '--class_ratio', ':'.join(map(str, config.dataset_prepare.ratio)),
            '--train_size', str(config.dataset_prepare.train_size),
            '--val_ratio', str(config.dataset_prepare.val_ratio),
            '--delete_prob', str(config.dataset_prepare.delete_prob),
            '--replace_prob', str(config.dataset_prepare.replace_prob),
            '--seed', str(config.dataset_prepare.seed),
            '--num_constraint_leaves', str(config.dataset_prepare.num_constraint_leaves),
            '--type1_single_new_prob', str(config.dataset_prepare.type1_single_new_prob),
            '--mix_output_to_constraint_prob', str(config.dataset_prepare.mix_output_to_constraint_prob)
        ]
        
        run_command(cmd, "步骤 3: 准备训练数据集")
    else:
        print("\n跳过步骤 3: 准备数据集")
    
    # 步骤 4: 训练模型
    if not config.pipeline.skip_training:
        # 创建训练配置文件
        train_config = {
            'lora': {
                'r': config.lora.r,
                'lora_alpha': config.lora.lora_alpha,
                'lora_dropout': config.lora.lora_dropout,
                'target_modules': config.lora.target_modules,
                'bias': config.lora.bias
            },
            'training': {
                'num_epochs': config.training.num_epochs,
                'batch_size': config.training.batch_size,
                'gradient_accumulation_steps': config.training.gradient_accumulation_steps,
                'learning_rate': config.training.learning_rate,
                'warmup_ratio': config.training.warmup_ratio,
                'logging_steps': config.training.logging_steps,
                'save_steps': config.training.save_steps,
                'eval_steps': config.training.eval_steps,
                'save_total_limit': config.training.save_total_limit,
                'max_length': config.training.max_length,
                'bf16': config.training.bf16,
                'gradient_checkpointing': config.training.gradient_checkpointing,
                'dataloader_num_workers': config.training.dataloader_num_workers
            },
            'quantization': {
                'load_in_4bit': config.quantization.load_in_4bit,
                'bnb_4bit_compute_dtype': config.quantization.bnb_4bit_compute_dtype,
                'bnb_4bit_use_double_quant': config.quantization.bnb_4bit_use_double_quant,
                'bnb_4bit_quant_type': config.quantization.bnb_4bit_quant_type
            }
        }
        
        import json
        train_config_file = output_base / 'train_config.json'
        with open(train_config_file, 'w') as f:
            json.dump(train_config, f, indent=2)
        
        run_command(
            [
                'python3', 'train.py',
                '--base_model', config.path.base_model,
                '--train_data', str(Path(config.path.dataset_dir) / 'train_dataset.json'),
                '--val_data', str(Path(config.path.dataset_dir) / 'val_dataset.json'),
                '--output_dir', str(Path(config.path.models_dir) / 'hierarchical_classifier'),
                '--config', str(train_config_file)
            ],
            "步骤 4: 训练模型"
        )
    else:
        print("\n跳过步骤 4: 训练")
    
    # 步骤 5: 推理
    model_path = Path(config.path.models_dir) / 'hierarchical_classifier' / 'model'
    for split in ['test_hard']:
        run_command(
            [
                'python3', 'inference.py',
                '--model_path', str(model_path),
                '--references_file', config.path.references_file,
                '--split_file', str(Path(config.path.data_dir) / 'dataset_split.json'),
                '--split', split,
                '--output_dir', config.path.inference_dir,
                '--tensor_parallel_size', str(config.inference.tensor_parallel_size),
                '--max_model_len', str(config.inference.max_model_len),
                '--gpu_memory_utilization', str(config.inference.gpu_memory_utilization),
                '--max_depth', str(config.builder.max_depth),
                '--structures_file', config.path.structures_file,
                '--num_inference_constraint_leaves', str(config.inference.num_inference_constraint_leaves)
            ] + (['--max_refs', str(config.inference.max_refs)] if config.inference.max_refs else [])
              + (['--use_structure_init'] if config.inference.use_structure_init else []),
            f"步骤 5: 推理 ({split})"
        )
        
        # 评估
        pred_file = Path(config.path.inference_dir) / f'{split}_trees.json'
        if pred_file.exists():
            eval_output = Path(config.path.inference_dir) / f'{split}_evaluation.json'
            run_command(
                [
                    'python3', 'evaluate.py',
                    '--pred_file', str(pred_file),
                    '--true_file', config.path.structures_file,
                    '--output_file', str(eval_output)
                ],
                f"步骤 6: 评估 ({split})"
            )
            print(f"\n{split} 评估结果已保存到: {eval_output}")
    
    print("\n" + "="*80)
    print("完整流程执行完成！")
    print("="*80)
    print(f"配置文件: {config_save_path}")
    print(f"数据集划分: {config.path.data_dir}")
    print(f"Paraphrases: {config.path.paraphrases_dir}")
    print(f"训练数据集: {config.path.dataset_dir}")
    print(f"训练模型: {config.path.models_dir}")
    print(f"推理结果: {config.path.inference_dir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
