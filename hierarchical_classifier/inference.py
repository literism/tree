"""
推理脚本
使用训练好的模型进行推理，构建结构树
"""
import json
import argparse
from pathlib import Path
from classifier import Classifier
from builder import TreeBuilder


def main():
    parser = argparse.ArgumentParser(description='使用模型进行推理')
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='模型路径'
    )
    parser.add_argument(
        '--references_file',
        type=str,
        default='/mnt/literism/tree/data/wikipedia_references_final.json',
        help='references文件路径'
    )
    parser.add_argument(
        '--split_file',
        type=str,
        default='./data/dataset_split.json',
        help='数据集划分文件'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test_easy',
        choices=['train', 'test_easy', 'test_hard', 'all'],
        help='要推理的数据集划分'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./inference_output',
        help='输出目录'
    )
    parser.add_argument(
        '--topic_key',
        type=str,
        help='指定单个topic进行推理'
    )
    parser.add_argument(
        '--max_refs',
        type=int,
        help='每个topic最多处理的references数量（用于快速测试）'
    )
    parser.add_argument(
        '--tensor_parallel_size',
        type=int,
        default=1,
        help='张量并行大小'
    )
    parser.add_argument(
        '--max_model_len',
        type=int,
        default=8192,
        help='最大序列长度'
    )
    parser.add_argument(
        '--gpu_memory_utilization',
        type=float,
        default=0.9,
        help='GPU内存利用率'
    )
    parser.add_argument(
        '--max_depth',
        type=int,
        default=10,
        help='最大树深度'
    )
    parser.add_argument(
        '--use_structure_init',
        action='store_true',
        help='使用结构文件的第一层节点初始化树（然后删除空节点）'
    )
    parser.add_argument(
        '--structures_file',
        type=str,
        default='/mnt/literism/tree/data/wikipedia_structures_final.json',
        help='结构文件路径（用于初始化模式）'
    )
    parser.add_argument(
        '--num_inference_constraint_leaves',
        type=int,
        default=20,
        help='推理时选择多少个叶子节点作为约束'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("模型推理")
    print("="*80)
    
    # 加载数据
    print("\n加载数据...")
    with open(args.references_file, 'r', encoding='utf-8') as f:
        references_data = json.load(f)
    print(f"  - 加载 {len(references_data)} 个topics")
    
    # 创建分类器
    print("\n创建模型分类器...")
    classifier = Classifier(
        mode='model',
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization
    )
    
    # 创建构建器
    builder = TreeBuilder(
        classifier=classifier,
        references_data=references_data,
        max_depth=args.max_depth,
        structures_file=args.structures_file if args.use_structure_init else None,
        num_inference_constraint_leaves=args.num_inference_constraint_leaves
    )
    
    # 确定要处理的topics
    if args.topic_key:
        # 单个topic
        topics_to_process = {args.topic_key: list(references_data[args.topic_key]['references'].keys())}
        if args.max_refs:
            topics_to_process[args.topic_key] = topics_to_process[args.topic_key][:args.max_refs]
    else:
        # 从划分文件中获取
        with open(args.split_file, 'r', encoding='utf-8') as f:
            split_info = json.load(f)
        dataset_split = split_info['dataset_split']
        
        if args.split == 'all':
            topics_to_process = {}
            for split_name in ['train', 'test_easy', 'test_hard']:
                topics_to_process.update(dataset_split[split_name])
        else:
            topics_to_process = dataset_split[args.split]
        
        if args.max_refs:
            for topic_key in topics_to_process:
                topics_to_process[topic_key] = topics_to_process[topic_key][:args.max_refs]
    
    print(f"\n要处理的topics: {len(topics_to_process)}")
    
    if args.use_structure_init:
        print(f"\n使用结构初始化模式:")
        print(f"  - 结构文件: {args.structures_file}")
        print(f"  - 将使用结构树的第一层节点初始化，然后删除空节点")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 推理（使用批处理）
    print("\n开始推理（topic并行，topic内文章串行）...")
    
    # 构建数据结构用于批处理
    dataset_for_inference = {args.split: topics_to_process}
    
    # 调用批处理构建，启用错误跟踪
    all_trees_dict, error_stats = builder.build_trees_for_split_with_tracking(
        dataset_split=dataset_for_inference,
        split_name=args.split,
        output_dir=str(output_dir),
        record_mode=False,
        use_structure_init=args.use_structure_init
    )
    
    # 输出统计
    print(f"\n{'='*80}")
    print("推理完成！")
    print(f"{'='*80}")
    print(f"处理了 {len(all_trees_dict)} 个topics")
    print(f"\n错误统计:")
    print(f"  - 成功解析: {error_stats['total_success']}")
    print(f"  - 解析失败: {error_stats['total_failed']}")
    if error_stats['total_failed'] > 0:
        print(f"  - 失败率: {error_stats['total_failed'] / (error_stats['total_success'] + error_stats['total_failed']) * 100:.2f}%")
    
    # 输出每个topic的统计
    if error_stats['by_topic']:
        print(f"\n各topic错误统计:")
        for topic_key, stats in error_stats['by_topic'].items():
            if stats['failed'] > 0:
                print(f"  {topic_key}: 成功{stats['success']}, 失败{stats['failed']}")
    
    trees_file = output_dir / f'{args.split}_trees.json'
    print(f"\n结果保存到: {trees_file}")


if __name__ == '__main__':
    main()

