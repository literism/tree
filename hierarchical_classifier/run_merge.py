"""
合并系统完整流程运行脚本
包括数据准备、训练、推理
"""
import argparse
import subprocess
from pathlib import Path


def run_command(cmd: list, description: str):
    """运行命令"""
    print("\n" + "="*80)
    print(f"{description}")
    print("="*80)
    print(f"命令: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, check=True)
    
    if result.returncode != 0:
        print(f"错误: {description} 失败")
        exit(1)
    
    print(f"\n{description} 完成")


def main():
    parser = argparse.ArgumentParser(description='合并系统完整流程')
    
    parser.add_argument(
        '--structures_file',
        type=str,
        default='/mnt/literism/tree/data/wikipedia_structures_final.json',
        help='结构文件路径'
    )
    parser.add_argument(
        '--base_model',
        type=str,
        default='/data_turbo/literism/models/Qwen2.5-7B-Instruct',
        help='基础模型路径'
    )
    parser.add_argument(
        '--output_base_dir',
        type=str,
        default='/mnt/literism/tree/merge_output',
        help='输出基础目录'
    )
    parser.add_argument(
        '--test_trees',
        type=str,
        default='/mnt/literism/tree/summary_output/inference/test_trees.json',
        help='测试树文件路径'
    )
    parser.add_argument(
        '--skip_data_prep',
        action='store_true',
        help='跳过数据准备'
    )
    parser.add_argument(
        '--skip_training',
        action='store_true',
        help='跳过训练'
    )
    parser.add_argument(
        '--skip_inference',
        action='store_true',
        help='跳过推理'
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data_dir = output_dir / 'data'
    train_dir = output_dir / 'training'
    inference_dir = output_dir / 'inference'
    
    # 步骤1：准备数据
    if not args.skip_data_prep:
        run_command(
            [
                'python', 'prepare_merge_dataset.py',
                '--structures_file', args.structures_file,
                '--output_dir', str(data_dir),
                '--val_ratio', '0.1',
                '--seed', '42'
            ],
            "步骤1: 准备训练数据"
        )
    else:
        print("\n跳过数据准备")
    
    # 步骤2：训练
    if not args.skip_training:
        run_command(
            [
                'python', 'train_merge.py',
                '--base_model', args.base_model,
                '--train_data', str(data_dir / 'train_dataset.json'),
                '--val_data', str(data_dir / 'val_dataset.json'),
                '--output_dir', str(train_dir)
            ],
            "步骤2: 训练合并模型"
        )
    else:
        print("\n跳过训练")
    
    # 步骤3：推理合并
    if not args.skip_inference:
        run_command(
            [
                'python', 'merge_inference.py',
                '--base_model', args.base_model,
                '--lora_model', str(train_dir / 'final_model'),
                '--test_trees', args.test_trees,
                '--output', str(inference_dir / 'merged_trees.json'),
                '--resolution', '1.0'
            ],
            "步骤3: 推理并合并树"
        )
    else:
        print("\n跳过推理")
    
    print("\n" + "="*80)
    print("完整流程执行完成！")
    print("="*80)
    print(f"\n输出目录: {output_dir}")
    print(f"  - 数据: {data_dir}")
    print(f"  - 训练: {train_dir}")
    print(f"  - 推理: {inference_dir}")


if __name__ == '__main__':
    main()

