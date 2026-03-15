"""
结构树生成器完整流程
"""
import subprocess
import sys
from pathlib import Path
import argparse

from config import StructureGeneratorConfig
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run_command(cmd: list, description: str):
    """运行命令并处理错误"""
    print("\n" + "="*60)
    print(f"执行: {description}")
    print("="*60)
    print(f"命令: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"\n错误: {description} 失败")
        sys.exit(1)
    
    print(f"\n✓ {description} 完成")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='结构树生成器完整流程')
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/default.json',
        help='配置文件路径'
    )
    parser.add_argument(
        '--skip_prepare', 
        default=True, 
        help='跳过数据准备'
    )
    parser.add_argument(
        '--skip_training', 
        default=True, 
        help='跳过训练'
    )
    parser.add_argument(
        '--skip_inference', 
        default=False, 
        help='跳过推理'
    )
    parser.add_argument(
        '--inference_split', type=str, 
        choices=['train', 'test_easy', 'test_hard', 'all'],
        default='test_hard', 
        help='推理的split'
    )
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config:
        config = StructureGeneratorConfig.from_json(args.config)
    else:
        config = StructureGeneratorConfig()
    
    config.print_config()
    
    # 获取当前脚本目录
    script_dir = Path(__file__).parent
    
    # 配置参数
    config_arg = ['--config', args.config] if args.config else []
    
    # Step 1: 准备数据集
    if not args.skip_prepare:
        run_command(
            ['python3', str(script_dir / 'prepare_structure_dataset.py')] + config_arg,
            '准备数据集'
        )
    else:
        print("\n⊘ 跳过数据准备")
    
    # Step 2: 训练模型
    if not args.skip_training:
        run_command(
            ['python3', str(script_dir / 'train_structure_generator.py')] + config_arg,
            '训练模型'
        )
    else:
        print("\n⊘ 跳过训练")
    
    # Step 3: 推理生成结构树
    if not args.skip_inference:
        run_command(
            ['python3', str(script_dir / 'inference_structure.py')] + config_arg + 
            ['--split', args.inference_split],
            f'推理生成结构树 ({args.inference_split})'
        )
    else:
        print("\n⊘ 跳过推理")
    
    print("\n" + "="*60)
    print("完整流程执行完成！")
    print("="*60)
    
    # 打印输出位置
    print("\n输出文件位置:")
    print(f"  - 数据集: {config.data_dir}")
    print(f"  - 模型: {config.models_dir}")
    print(f"  - 推理结果: {config.inference_dir}")


if __name__ == '__main__':
    main()

