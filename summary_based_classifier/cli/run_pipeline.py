"""
主运行脚本
协调整个新系统pipeline的执行

完整流程：
1. 数据划分
2. 生成summaries
3. 准备初始SFT训练数据
4. 训练分类生成系统（SFT）
5. 训练总结更新系统（SFT）
6. DPO迭代训练（可选）
7. 推理
8. 评估
"""
import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["RAY_memory_monitor_refresh_ms"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
import argparse
import subprocess
import sys
from pathlib import Path
from summary_based_classifier.config import SummaryBasedConfig


def run_command(cmd: list, description: str, allow_failure: bool = False):
    """运行命令并打印输出"""
    print("\n" + "="*80)
    print(f"{description}")
    print("="*80)
    print(f"命令: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
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
        description='运行Summary-Based分类器完整pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # 基本参数
    parser.add_argument(
        '--config',
        type=str,
        default='./configs/default.json',
        help='配置文件路径'
    )
    
    # 步骤控制
    parser.add_argument(
        '--skip_split', 
        default=True, 
        help='跳过数据划分'
    )
    parser.add_argument(
        '--skip_summaries', 
        default=True, 
        help='跳过生成summaries'
    )
    parser.add_argument(
        '--skip_prepare', 
        default=True, 
        help='跳过准备SFT训练数据'
    )
    parser.add_argument(
        '--skip_train', 
        default=False, 
        help='跳过所有训练（SFT和DPO）'
    )
    parser.add_argument(
        '--skip_train_classify_generator', 
        default=True,
        help='跳过训练分类生成系统'
    )
    parser.add_argument(
        '--skip_train_updater', 
        default=True,
        help='跳过训练总结更新系统'
    )
    parser.add_argument(
        '--skip_dpo', 
        default=False, 
        help='跳过DPO训练'
    )
    parser.add_argument(
        '--skip_inference', 
        default=True,
        help='跳过推理'
    )
    parser.add_argument(
        '--skip_eval', 
        default=True,
        help='运行评估'
    )
    
    # DPO训练参数
    parser.add_argument(
        '--dpo_iterations', 
        type=int, 
        help='DPO训练迭代次数（覆盖配置文件）'
    )
    parser.add_argument(
        '--dpo_start_iteration', 
        type=int, 
        default=None,  # 改为None，从config读取
        help='DPO开始迭代（用于断点续训，不指定则从config读取）'
    )
    
    # 推理参数
    parser.add_argument(
        '--inference_split', 
        type=str, 
        help='推理的数据集划分（覆盖配置文件）'
    )
    parser.add_argument(
        '--max_workers',
        type=int,
        default=4,
        help='并行推理的最大topic数（默认4）'
    )
    parser.add_argument(
        '--classify_generator_model', 
        type=str, 
        help='分类生成模型路径（覆盖默认）'
    )
    parser.add_argument(
        '--updater_model', 
        type=str, 
        help='总结更新模型路径（覆盖默认）'
    )
    
    args = parser.parse_args()
    
    # 加载配置
    config = SummaryBasedConfig.from_json(args.config)
    
    print("="*80)
    print("Summary-Based Classifier Pipeline (新系统)")
    print("="*80)
    print(f"配置文件: {args.config}")
    print(f"基础模型: {config.path.base_model}")
    print(f"输出目录: {config.path.output_base}")
    print("="*80)
    
    # ========== 阶段1: 数据准备 ==========
    
    # 1. 数据划分
    if not args.skip_split:
        run_command(
            [sys.executable, '-m', 'summary_based_classifier.data.data_split', '--config', args.config],
            "步骤1: 数据划分"
        )
    else:
        print("\n跳过步骤1: 数据划分")
    
    # 2. 生成summaries
    if not args.skip_summaries:
        run_command(
            [sys.executable, '-m', 'summary_based_classifier.llm.generate_summaries', '--config', args.config],
            "步骤2: 生成节点summaries"
        )
    else:
        print("\n跳过步骤2: 生成summaries")
    
    # 3. 准备初始SFT训练数据
    if not args.skip_prepare:
        run_command(
            [sys.executable, '-m', 'summary_based_classifier.data.prepare_dataset', '--config', args.config],
            "步骤3: 准备SFT训练数据"
        )
    else:
        print("\n跳过步骤3: 准备训练数据")
    
    # ========== 阶段2: 初始SFT训练 ==========
    
    if not args.skip_train:
        # 4. 训练分类生成系统
        if not args.skip_train_classify_generator:
            classify_generator_output = str(Path(config.path.models_dir) / 'classify_generator_sft')
            train_data = str(Path(config.path.data_dir) / 'classify_generator_train.jsonl')
            val_data = str(Path(config.path.data_dir) / 'classify_generator_val.jsonl')
            
            # 检查数据文件是否存在
            if not Path(train_data).exists():
                print(f"\n警告: 训练数据不存在: {train_data}")
                print("请先运行 prepare_dataset.py")
            else:
                run_command(
                    [
                        sys.executable, '-m', 'summary_based_classifier.training.train_classify_generator',
                        '--base_model', config.path.base_model,
                        '--train_data', train_data,
                        '--val_data', val_data,
                        '--output_dir', classify_generator_output,
                        '--config', args.config
                    ],
                    "步骤4: 训练分类生成系统（SFT）"
                )
        else:
            print("\n跳过步骤4: 训练分类生成系统")
        
        # 5. 训练总结更新系统
        if not args.skip_train_updater:
            updater_output = str(Path(config.path.models_dir) / 'updater_sft')
            train_data = str(Path(config.path.data_dir) / 'updater_train.jsonl')
            val_data = str(Path(config.path.data_dir) / 'updater_val.jsonl')
            
            # 检查数据文件是否存在
            if not Path(train_data).exists():
                print(f"\n警告: 训练数据不存在: {train_data}")
                print("请先运行 prepare_dataset.py")
            else:
                run_command(
                    [
                        sys.executable, '-m', 'summary_based_classifier.training.train_updater',
                        '--base_model', config.path.base_model,
                        '--train_data', train_data,
                        '--val_data', val_data,
                        '--output_dir', updater_output,
                        '--config', args.config
                    ],
                    "步骤5: 训练总结更新系统（SFT）"
                )
        else:
            print("\n跳过步骤5: 训练总结更新系统")
    else:
        print("\n跳过阶段2: 初始SFT训练")
    
    # ========== 阶段3: DPO迭代训练（可选）==========
    
    if not args.skip_dpo and not args.skip_train:
        print("\n" + "="*80)
        print("阶段3: DPO迭代训练")
        print("="*80)
        
        dpo_iterations = args.dpo_iterations if args.dpo_iterations else config.dpo_training.num_iterations
        dpo_start_iteration = args.dpo_start_iteration if args.dpo_start_iteration is not None else config.dpo_training.start_iteration
        
        print(f"DPO迭代次数: {dpo_iterations}")
        print(f"采样批次: {config.dpo_training.sampling_batch_sizes}")
        print(f"开始迭代: {dpo_start_iteration}")
        if dpo_start_iteration > 0:
            print(f"  → 将加载 iter_{dpo_start_iteration-1} 的模型作为起点")
        
        # 检查SFT模型是否存在
        sft_classifier = Path(config.path.models_dir) / 'classify_generator_sft' / 'final_model'
        sft_updater = Path(config.path.models_dir) / 'updater_sft' / 'final_model'
        
        if not sft_classifier.exists() or not sft_updater.exists():
            print("\n警告: 找不到SFT训练的模型")
            print(f"  分类生成模型: {sft_classifier} {'✓' if sft_classifier.exists() else '✗'}")
            print(f"  总结更新模型: {sft_updater} {'✓' if sft_updater.exists() else '✗'}")
            print("请先完成SFT训练")
        else:
            # DPO训练 - 使用完整的train_dpo.py
            cmd = [
                sys.executable, '-m', 'summary_based_classifier.training.train_dpo',
                '--config', args.config,
                '--start_iteration', str(dpo_start_iteration)
            ]
            
            if args.dpo_iterations:
                cmd.extend(['--num_iterations', str(dpo_iterations)])
            
            run_command(cmd, "步骤6: DPO迭代训练")
    else:
        if args.skip_dpo:
            print("\n跳过阶段3: DPO训练")
        else:
            print("\n跳过阶段3: DPO训练（因为跳过了SFT训练）")
    
    # ========== 阶段4: 推理 ==========
    
    if not args.skip_inference:
        print("\n" + "="*80)
        print("阶段4: 推理")
        print("="*80)
        
        # 确定使用哪个模型
        if args.classify_generator_model:
            classify_generator_model = args.classify_generator_model
        else:
            # 优先级：IW-SFT最新迭代 > DPO训练模型 > SFT模型
            models_dir = Path(config.path.models_dir)
            
            # 查找所有IW-SFT迭代模型
            iwsft_models = sorted(models_dir.glob('classify_generator_iwsft_iter*'))
            if iwsft_models:
                # 找到最新的迭代
                latest_iwsft = iwsft_models[-1] / 'final'
                if latest_iwsft.exists():
                    classify_generator_model = str(latest_iwsft)
                    print(f"使用IW-SFT迭代训练模型: {classify_generator_model}")
                else:
                    latest_iwsft = None
            else:
                latest_iwsft = None
            
            if not latest_iwsft:
                dpo_model = models_dir / 'classify_generator_dpo_final'
                sft_model = models_dir / 'classify_generator_sft' / 'final_model'
                
                if dpo_model.exists():
                    classify_generator_model = str(dpo_model)
                    print(f"使用DPO训练模型: {classify_generator_model}")
                elif sft_model.exists():
                    classify_generator_model = str(sft_model)
                    print(f"使用SFT训练模型: {classify_generator_model}")
                else:
                    print(f"错误: 找不到训练好的分类生成模型")
                    print(f"  尝试路径: IW-SFT迭代模型")
                    print(f"  尝试路径: {dpo_model}")
                    print(f"  尝试路径: {sft_model}")
                    sys.exit(1)
        
        if args.updater_model:
            updater_model = args.updater_model
        else:
            # 优先级：IW-SFT最新迭代 > DPO训练模型 > SFT模型
            models_dir = Path(config.path.models_dir)
            
            # 查找所有IW-SFT迭代模型
            iwsft_models = sorted(models_dir.glob('updater_iwsft_iter*'))
            if iwsft_models:
                # 找到最新的迭代
                latest_iwsft = iwsft_models[-1] / 'final'
                if latest_iwsft.exists():
                    updater_model = str(latest_iwsft)
                    print(f"使用IW-SFT迭代训练模型: {updater_model}")
                else:
                    latest_iwsft = None
            else:
                latest_iwsft = None
            
            if not latest_iwsft:
                dpo_model = models_dir / 'updater_dpo_final'
                sft_model = models_dir / 'updater_sft' / 'final_model'
                
                if dpo_model.exists():
                    updater_model = str(dpo_model)
                    print(f"使用DPO训练模型: {updater_model}")
                elif sft_model.exists():
                    updater_model = str(sft_model)
                    print(f"使用SFT训练模型: {updater_model}")
                else:
                    print(f"错误: 找不到训练好的总结更新模型")
                    print(f"  尝试路径: IW-SFT迭代模型")
                    print(f"  尝试路径: {dpo_model}")
                    print(f"  尝试路径: {sft_model}")
                    sys.exit(1)
        
        inference_split = args.inference_split if args.inference_split else config.inference.split
        max_workers = args.max_workers if args.max_workers else 4
        
        run_command(
            [
                sys.executable, '-m', 'summary_based_classifier.inference.inference_parallel',
                '--config', args.config,
                '--classify_generator_model', classify_generator_model,
                '--updater_model', updater_model,
                '--split', inference_split,
                '--max_workers', str(max_workers)
            ],
            f"步骤7: 并行推理 ({inference_split} split)"
        )
    else:
        print("\n跳过阶段4: 推理")
    
    # ========== 阶段5: 评估 ==========
    
    if not args.skip_eval:
        print("\n" + "="*80)
        print("阶段5: 评估")
        print("="*80)
        
        inference_split = args.inference_split if args.inference_split else config.inference.split
        pred_file = str(Path(config.path.inference_dir) / f'{inference_split}_trees.json')
        
        if not Path(pred_file).exists():
            print(f"错误: 推理结果文件不存在: {pred_file}")
            print("请先运行推理步骤")
            sys.exit(1)
        
        true_file = config.path.structures_file
        output_file = str(Path(config.path.inference_dir) / f'evaluation_results_{inference_split}.json')
        
        # 检查evaluate.py是否存在
        if Path('evaluate.py').exists():
            run_command(
                [
                    sys.executable, '-m', 'summary_based_classifier.evaluation.evaluate',
                    '--pred_file', pred_file,
                    '--true_file', true_file,
                    '--output_file', output_file
                ],
                "步骤8: 评估"
            )
        else:
            print("警告: evaluate.py 不存在，跳过评估")
    else:
        print("\n跳过阶段5: 评估")
    
    # ========== 完成 ==========
    
    print("\n" + "="*80)
    print("Pipeline完成！")
    print("="*80)
    
    # 打印输出路径
    print("\n输出文件位置:")
    print(f"  - 数据目录: {config.path.data_dir}")
    print(f"  - 模型目录: {config.path.models_dir}")
    print(f"  - 推理目录: {config.path.inference_dir}")
    
    if not args.skip_inference:
        inference_split = args.inference_split if args.inference_split else config.inference.split
        pred_file = Path(config.path.inference_dir) / f'{inference_split}_trees.json'
        if pred_file.exists():
            print(f"  - 推理结果: {pred_file}")
    
    if args.eval:
        output_file = Path(config.path.inference_dir) / f'evaluation_results_{inference_split}.json'
        if output_file.exists():
            print(f"  - 评估结果: {output_file}")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    main()
