"""
Oracle策略完整Pipeline
协调Oracle数据生成、模型训练和推理步骤

完整流程：
1. 生成Oracle SFT数据
2. 训练总结模型
3. 训练分类生成模型
4. 推理（使用训练好的模型）
"""
import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["RAY_memory_monitor_refresh_ms"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import argparse
import json
import subprocess
import sys
from pathlib import Path
from summary_based_classifier.config import SummaryBasedConfig


def run_command(cmd: list, description: str, cwd: str = None, allow_failure: bool = False, env: dict = None):
    """运行命令并打印输出"""
    print("\n" + "="*80)
    print(f"{description}")
    print("="*80)
    print(f"命令: {' '.join(cmd)}")
    if cwd:
        print(f"工作目录: {cwd}")
    
    result = subprocess.run(cmd, capture_output=False, cwd=cwd, env=env)
    
    if result.returncode != 0:
        print(f"\n子进程退出码: {result.returncode}")
        if result.returncode < 0:
            print(f"子进程被信号终止: {-result.returncode} (常见: 9=SIGKILL, 15=SIGTERM)")
        if allow_failure:
            print(f"\n警告: {description} 失败（已忽略）")
        else:
            print(f"\n错误: {description} 失败")
            sys.exit(1)
    else:
        print(f"\n{description} 完成")
    
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description='运行Oracle策略完整pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # 基本参数
    parser.add_argument(
        '--config',
        type=str,
        default='./summary_based_classifier/configs/default.json',
        help='配置文件路径'
    )
    
    # 步骤控制
    parser.add_argument(
        '--skip_generate', 
        default=False,
        help='跳过Oracle数据生成'
    )
    parser.add_argument(
        '--skip_balance', 
        default=False,
        help='跳过数据平衡'
    )
    parser.add_argument(
        '--skip_train_summary', 
        default=True,
        help='跳过总结模型训练'
    )
    parser.add_argument(
        '--skip_train_classify', 
        default=False,
        help='跳过分类模型训练'
    )
    parser.add_argument(
        '--skip_inference', 
        default=False,
        help='跳过推理'
    )
    parser.add_argument(
        '--train_classify_gpus',
        type=str,
        default='0,1',
        help='训练分类模型使用的GPU列表（逗号分隔），如0或0,1'
    )
    
    # Oracle数据生成参数
    parser.add_argument(
        '--data_mode',
        type=str,
        default='model',
        choices=['api', 'model'],
        help='数据生成模式：api或model（默认model）'
    )
    parser.add_argument(
        '--bow_top_k',
        type=int,
        default=10,
        help='BM25 summary保留的top-k词数（默认30）'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（默认42）'
    )
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.02,
        help='验证集比例（默认0.02）'
    )
    parser.add_argument(
        '--max_refs_per_topic',
        type=int,
        default=None,
        help='每个topic最多使用的文章数（用于快速测试）'
    )
    
    # 数据平衡参数
    parser.add_argument(
        '--use_balanced_data',
        default=True,
        help='训练时使用平衡后的数据集'
    )
    parser.add_argument(
        '--summary_no_update_ratio',
        type=float,
        default=0.4,
        help='总结模型数据平衡：不需要更新的样本目标比例（默认0.3）'
    )
    parser.add_argument(
        '--classify_new_ratio',
        type=float,
        default=0.15,
        help='分类模型数据平衡：创建新类的样本目标比例（默认0.2）'
    )
    parser.add_argument(
        '--classify_merge_ratio',
        type=float,
        default=0.05,
        help='分类模型数据平衡：归拢的样本目标比例（默认0.1）'
    )
    
    # 推理参数
    parser.add_argument(
        '--inference_split',
        type=str,
        default='test',
        help='推理的数据集划分（默认test）'
    )
    parser.add_argument(
        '--inference_gpus',
        type=str,
        default='0,1',
        help='推理使用的GPU（默认0,1，分类模型用第一张，updater用第二张）'
    )
    parser.add_argument(
        '--classify_generator_model',
        type=str,
        default=None,
        help='推理时使用的分类模型路径（默认自动查找训练好的模型）'
    )
    parser.add_argument(
        '--updater_model',
        type=str,
        default=None,
        help='推理时使用的总结模型路径（默认自动查找训练好的模型）'
    )
    parser.add_argument(
        '--inference_max_workers',
        type=int,
        default=1,
        help='推理时最大并行topic数（默认4）'
    )
    
    args = parser.parse_args()
    
    # 获取项目根目录
    project_root = Path(__file__).parent.parent.parent
    
    # 加载配置（将相对路径转为绝对路径）
    config_path = args.config if Path(args.config).is_absolute() else str(project_root / args.config)
    config = SummaryBasedConfig.from_json(config_path)
    
    print("="*80)
    print("Oracle策略完整Pipeline")
    print("="*80)
    print(f"项目根目录: {project_root}")
    print(f"配置文件: {config_path}")
    print(f"输出目录: {config.path.output_base}")
    print("="*80)
    print(f"步骤控制:")
    print(f"  - 生成Oracle数据: {'跳过' if args.skip_generate else '执行'} (模式: {args.data_mode})")
    print(f"  - 数据平衡: {'跳过' if args.skip_balance else '执行'}")
    print(f"  - 训练总结模型: {'跳过' if args.skip_train_summary else '执行'}")
    print(f"  - 训练分类模型: {'跳过' if args.skip_train_classify else '执行'} (使用{'平衡后' if args.use_balanced_data else '原始'}数据)")
    print(f"  - 推理: {'跳过' if args.skip_inference else '执行'}")
    print("="*80)
    
    # ========== 步骤1: 生成Oracle SFT数据 ==========
    
    if not args.skip_generate:
        cmd = [
            sys.executable, '-m', 'summary_based_classifier.data.prepare_dataset_oracle',
            '--config', config_path,
            '--mode', args.data_mode
        ]
        
        if args.max_refs_per_topic:
            cmd.extend(['--max_refs_per_topic', str(args.max_refs_per_topic)])
        
        description = f"步骤1: 生成Oracle SFT数据（{args.data_mode.upper()}模式）"
        run_command(cmd, description, cwd=str(project_root))
        
        # 检查生成的文件
        data_dir = Path(config.path.data_dir) / f'oracle_data_{args.data_mode}'
        classification_file = data_dir / 'classification_train.jsonl'
        summary_file = data_dir / 'summary_train.jsonl'
        
        if classification_file.exists():
            print(f"\n✓ 分类数据生成成功: {classification_file}")
        else:
            print(f"\n✗ 分类数据生成失败")
            sys.exit(1)
        
        if args.data_mode == 'api' and summary_file.exists():
            print(f"✓ 总结数据生成成功: {summary_file}")
        elif args.data_mode == 'model':
            print(f"(Model模式：仅生成分类数据)")
    else:
        print("\n跳过步骤1: 生成Oracle SFT数据")
        
        # 检查数据文件是否存在
        data_dir = Path(config.path.data_dir) / f'oracle_data_{args.data_mode}'
        classification_file = data_dir / 'classification_train.jsonl'
        
        if not classification_file.exists():
            print(f"\n警告: 分类训练数据文件不存在: {classification_file}")
            print("请先运行数据生成步骤")
            if not args.skip_train_classify:
                sys.exit(1)
    
    # ========== 步骤2: 训练总结模型 ==========

    if not args.skip_train_summary:
        # 总结模型只需要在API模式下训练（因为只有API模式生成summary数据）
        if args.data_mode != 'api':
            print("\n跳过步骤2: 训练总结模型（非API模式不需要训练）")
        else:
            output_dir = Path(config.path.models_dir) / 'summary_generator_oracle'
            data_dir = Path(config.path.data_dir) / 'oracle_data_api'
            
            # 根据参数选择使用原始数据还是平衡后的数据
            if args.use_balanced_data:
                train_file = data_dir / 'summary_train_balanced.jsonl'
            else:
                train_file = data_dir / 'summary_train.jsonl'
            
            if not train_file.exists():
                print(f"\n错误: 总结训练数据不存在: {train_file}")
                if args.use_balanced_data:
                    print("请先运行数据平衡步骤")
                else:
                    print("请先运行数据生成步骤（API模式）")
                sys.exit(1)
            
            # 统计样本数并创建验证集（10%）
            print(f"\n准备总结模型训练数据...")
            import random
            with open(train_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                random.shuffle(lines)
                split_idx = int(len(lines) * 0.9)
                train_lines = lines[:split_idx]
                val_lines = lines[split_idx:]
            
            # 保存分割后的数据
            train_file_split = data_dir / 'summary_train_split.jsonl'
            val_file = data_dir / 'summary_val.jsonl'
            
            with open(train_file_split, 'w', encoding='utf-8') as f:
                f.writelines(train_lines)
            with open(val_file, 'w', encoding='utf-8') as f:
                f.writelines(val_lines)
            
            print(f"  - 训练集: {len(train_lines)} 样本 -> {train_file_split}")
            print(f"  - 验证集: {len(val_lines)} 样本 -> {val_file}")
            
            cmd = [
                sys.executable, '-m', 'summary_based_classifier.training.train_summary_generator',
                '--base_model', config.path.base_model,
                '--train_data', str(train_file_split),
                '--val_data', str(val_file),
                '--output_dir', str(output_dir),
                '--config', config_path
            ]
            
            description = f"步骤2: 训练总结模型（Oracle SFT，使用{'平衡后' if args.use_balanced_data else '原始'}数据）"
            run_command(cmd, description, cwd=str(project_root))
            
            # 检查训练结果
            model_file = output_dir / 'final_model'
            if model_file.exists():
                print(f"\n✓ 总结模型训练成功: {model_file}")
            else:
                print(f"\n✗ 总结模型训练失败")
                sys.exit(1)
    else:
        print("\n跳过步骤2: 训练总结模型")
        
        # 检查模型是否存在（只有在需要推理时才检查）
        if not args.skip_inference:
            model_dir = Path(config.path.models_dir) / 'summary_generator_oracle' / 'final_model'
            if not model_dir.exists():
                print(f"\n警告: 训练好的总结模型不存在: {model_dir}")
                print("请先运行总结模型训练步骤")
    
    # ========== 步骤2.5: 数据平衡 ==========
    
    if not args.skip_balance:
        data_dir = Path(config.path.data_dir) / f'oracle_data_{args.data_mode}'
        
        cmd = [
            sys.executable, '-m', 'summary_based_classifier.data.balance_dataset',
            '--input_dir', str(data_dir),
            '--output_dir', str(data_dir),
            '--summary_no_update_ratio', str(args.summary_no_update_ratio),
            '--classify_new_ratio', str(args.classify_new_ratio),
            '--classify_merge_ratio', str(args.classify_merge_ratio),
            '--seed', str(args.seed)
        ]
        
        run_command(cmd, "步骤2.5: 数据平衡", cwd=str(project_root))
        
        # 检查平衡后的文件
        balanced_classification_file = data_dir / 'classification_train_balanced.jsonl'
        balanced_summary_file = data_dir / 'summary_train_balanced.jsonl'
        
        if balanced_classification_file.exists():
            print(f"\n✓ 分类数据平衡成功: {balanced_classification_file}")
        if balanced_summary_file.exists():
            print(f"\n✓ 总结数据平衡成功: {balanced_summary_file}")
    else:
        print("\n跳过步骤2.5: 数据平衡")
    
    # ========== 步骤3: 训练分类生成模型 ==========
    
    if not args.skip_train_classify:
        output_dir = Path(config.path.models_dir) / 'classify_generator_oracle'
        data_dir = Path(config.path.data_dir) / f'oracle_data_{args.data_mode}'
        
        # 根据参数选择使用原始数据还是平衡后的数据
        if args.use_balanced_data:
            train_file = data_dir / 'classification_train_balanced.jsonl'
        else:
            train_file = data_dir / 'classification_train.jsonl'
        
        # 检查训练文件是否存在
        if not train_file.exists():
            print(f"\n错误: 分类训练数据不存在: {train_file}")
            if args.use_balanced_data:
                print("请先运行数据平衡步骤")
            else:
                print("请先运行数据生成步骤")
            sys.exit(1)
        
        # 准备验证集（从训练集中分割10%）
        print(f"\n准备分类模型训练数据...")
        import random
        with open(train_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            random.shuffle(lines)
            split_idx = int(len(lines) * 0.9)
            train_lines = lines[:split_idx]
            val_lines = lines[split_idx:]
        
        # 保存分割后的数据
        train_file_split = data_dir / 'classification_train_split.jsonl'
        val_file = data_dir / 'classification_val.jsonl'
        
        with open(train_file_split, 'w', encoding='utf-8') as f:
            f.writelines(train_lines)
        with open(val_file, 'w', encoding='utf-8') as f:
            f.writelines(val_lines)
        
        print(f"  - 训练集: {len(train_lines)} 样本 -> {train_file_split}")
        print(f"  - 验证集: {len(val_lines)} 样本 -> {val_file}")
        
        train_gpu_list = [g.strip() for g in str(args.train_classify_gpus).split(',') if g.strip()]
        if not train_gpu_list:
            train_gpu_list = ['0']
        train_cuda_visible = ",".join(train_gpu_list)
        cmd = [
            sys.executable, '-m', 'summary_based_classifier.training.train_classify_generator',
            '--base_model', config.path.base_model,
            '--train_data', str(train_file_split),
            '--val_data', str(val_file),
            '--output_dir', str(output_dir),
            '--config', config_path
        ]
        
        description = (
            f"步骤3: 训练分类生成模型（Oracle SFT，使用{'平衡后' if args.use_balanced_data else '原始'}数据，"
            f"GPU={train_cuda_visible}，单进程模型并行）"
        )
        train_env = os.environ.copy()
        train_env["CUDA_VISIBLE_DEVICES"] = train_cuda_visible
        run_command(cmd, description, cwd=str(project_root), env=train_env)
        
        # 检查训练结果
        model_file = output_dir / 'final_model'
        if model_file.exists():
            print(f"\n✓ 分类模型训练成功: {model_file}")
        else:
            print(f"\n✗ 分类模型训练失败")
            sys.exit(1)
    else:
        print("\n跳过步骤3: 训练分类生成模型")
        
        # 检查模型是否存在
        model_dir = Path(config.path.models_dir) / 'classify_generator_oracle' / 'final_model'
        if not model_dir.exists():
            print(f"\n警告: 训练好的模型不存在: {model_dir}")
            print("请先运行训练步骤")
            if not args.skip_inference:
                sys.exit(1)
    
    # ========== 步骤4: 推理 ==========
    
    if not args.skip_inference:
        # 确定使用哪个模型
        if args.classify_generator_model:
            classify_model = args.classify_generator_model
        else:
            # 默认使用训练好的oracle模型
            classify_model = str(Path(config.path.models_dir) / 'classify_generator_oracle' / 'final_model')
        
        if not Path(classify_model).exists():
            print(f"\n错误: 分类模型不存在: {classify_model}")
            sys.exit(1)
        
        # 确定总结模型路径
        if args.updater_model:
            updater_model = args.updater_model
        else:
            updater_model = str(config.path.base_model)
        
        if not Path(updater_model).exists():
            print(f"\n错误: 总结模型不存在: {updater_model}")
            sys.exit(1)
        
        # 解析GPU配置
        gpu_list = args.inference_gpus.split(',')
        if len(gpu_list) < 2:
            print(f"\n警告: 只有{len(gpu_list)}个GPU，分类模型和总结模型将共享GPU")
            classify_gpu = int(gpu_list[0].strip())
            updater_gpu = int(gpu_list[0].strip())
        else:
            classify_gpu = int(gpu_list[0].strip())
            updater_gpu = int(gpu_list[1].strip())
        
        print(f"\nGPU分配:")
        print(f"  - 分类模型GPU: {classify_gpu}")
        print(f"  - 总结模型GPU: {updater_gpu}")
        
        cmd = [
            sys.executable, '-m', 'summary_based_classifier.inference.inference_oracle_style',
            '--config', config_path,
            '--classify_generator_model', classify_model,
            '--updater_model', updater_model,
            '--split', args.inference_split,
            '--classify_gpu', str(classify_gpu),
            '--updater_gpu', str(updater_gpu),
            '--max_workers', str(args.inference_max_workers)
        ]
        
        if args.max_refs_per_topic:
            cmd.extend(['--max_refs', str(args.max_refs_per_topic)])
        
        run_command(cmd, "步骤4: Oracle风格推理（分类模型 + 总结模型）", cwd=str(project_root))
        
        # 检查推理结果
        output_file = Path(config.path.inference_dir) / f'{args.inference_split}_trees_oracle_style.json'
        if output_file.exists():
            print(f"\n✓ 推理成功: {output_file}")
        else:
            print(f"\n✗ 推理失败")
            sys.exit(1)
    else:
        print("\n跳过步骤4: 推理")
        # 如果要评估，检查推理结果是否存在
        output_file = Path(config.path.inference_dir) / f'{args.inference_split}_trees_oracle_style.json'
    
    # ========== 步骤5: 评估 ==========
    
    if not args.skip_inference:  # 只有进行了推理才执行评估
        print("\n" + "="*80)
        print("步骤5: 评估")
        print("="*80)
        
        # 确保推理结果存在
        if not output_file.exists():
            print(f"\n错误: 推理结果不存在: {output_file}")
            print("请先运行推理步骤")
            sys.exit(1)
        
        # 准备评估参数
        eval_output_dir = Path(config.path.inference_dir) / 'evaluation'
        eval_output_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            sys.executable, '-m', 'summary_based_classifier.evaluation.evaluate',
            '--pred_file', str(output_file),
            '--true_file', config.path.gold_structures_file,
            '--output_file', str(eval_output_dir / f'{args.inference_split}_metrics.json')
        ]
        
        run_command(cmd, "步骤5: 评估推理结果", cwd=str(project_root))
        
        # 检查评估结果
        eval_output_dir = Path(config.path.inference_dir) / 'evaluation'
        metrics_file = eval_output_dir / f'{args.inference_split}_metrics.json'
        if metrics_file.exists():
            print(f"\n✓ 评估完成: {metrics_file}")
            
            # 打印评估结果
            try:
                with open(metrics_file, 'r', encoding='utf-8') as f:
                    metrics = json.load(f)
                    print("\n评估指标:")
                    if 'overall' in metrics:
                        overall = metrics['overall']
                        print(f"  总体:")
                        for key, value in overall.items():
                            if isinstance(value, float):
                                print(f"    - {key}: {value:.4f}")
                            else:
                                print(f"    - {key}: {value}")
            except Exception as e:
                print(f"  (无法读取评估结果: {e})")
        else:
            print(f"\n警告: 评估结果文件不存在")
    else:
        print("\n跳过步骤5: 评估（因为跳过了推理）")
    
    # ========== 完成 ==========
    
    print("\n" + "="*80)
    print("Pipeline完成！")
    print("="*80)
    
    print("\n输出文件位置:")
    print(f"  - 数据目录: {config.path.data_dir}")
    if not args.skip_generate:
        data_subdir = f"oracle_data_{args.data_mode}"
        print(f"    - {data_subdir}/classification_train.jsonl")
        if args.data_mode == 'api':
            print(f"    - {data_subdir}/summary_train.jsonl")
    if not args.skip_balance:
        data_subdir = f"oracle_data_{args.data_mode}"
        print(f"    - {data_subdir}/classification_train_balanced.jsonl")
        if args.data_mode == 'api':
            print(f"    - {data_subdir}/summary_train_balanced.jsonl")
    
    print(f"  - 模型目录: {config.path.models_dir}")
    if not args.skip_train_summary and args.data_mode == 'api':
        print(f"    - summary_generator_oracle/final_model/")
    if not args.skip_train_classify:
        print(f"    - classify_generator_oracle/final_model/")
    
    if not args.skip_inference:
        print(f"  - 推理目录: {config.path.inference_dir}")
        print(f"    - {args.inference_split}_trees_oracle_style.json")
        print(f"    - evaluation/{args.inference_split}_metrics.json")
        print(f"    - evaluation/{args.inference_split}_detailed.json")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    main()
