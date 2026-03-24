"""
并行推理脚本
使用并行推理处理器进行高效推理
"""
import os
import json
import argparse
from pathlib import Path
from summary_based_classifier.config import SummaryBasedConfig
from summary_based_classifier.inference.parallel_inference_processor import ParallelInferenceProcessor


def main():
    parser = argparse.ArgumentParser(description='使用并行推理系统进行推理')
    parser.add_argument('--config', type=str, default='./configs/default.json')
    parser.add_argument('--classify_generator_model', type=str, required=True)
    parser.add_argument('--updater_model', type=str, required=True)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--max_refs', type=int, default=None, help='每个topic最多处理的文章数')
    parser.add_argument('--max_workers', type=int, default=4, help='最大并行topic数')
    
    args = parser.parse_args()
    
    # 加载配置
    config = SummaryBasedConfig.from_json(args.config)
    
    print("="*80)
    print("并行推理系统")
    print("="*80)
    
    # 加载数据
    print("\n加载数据...")
    with open(config.path.references_file, 'r', encoding='utf-8') as f:
        references_data = json.load(f)
    
    with open(Path(config.path.data_dir) / 'dataset_split.json', 'r', encoding='utf-8') as f:
        split_data = json.load(f)
        dataset_split = split_data['dataset_split']
    
    # 准备topics数据
    print(f"\n准备 {args.split} split 数据...")
    topics_data = {}
    max_refs = args.max_refs if args.max_refs else config.inference.max_refs
    
    for topic_key, ref_ids in dataset_split.get(args.split, {}).items():
        if topic_key not in references_data:
            continue
        
        topic_data = references_data[topic_key]
        topic_name = topic_data.get('topic', topic_key)
        
        # 准备文章列表
        articles = []
        for ref_id in ref_ids:
            if ref_id in topic_data.get('references', {}):
                ref = topic_data['references'][ref_id]
                articles.append({
                    'id': ref_id,
                    'content': ref.get('content', '')
                })
        
        topics_data[topic_key] = {
            'topic': topic_name,
            'articles': articles
        }
    
    print(f"  - Topic数量: {len(topics_data)}")
    total_articles = sum(len(t['articles']) for t in topics_data.values())
    print(f"  - 总文章数: {total_articles}")
    
    # 创建并行推理处理器
    print(f"\n初始化并行推理处理器...")
    processor = ParallelInferenceProcessor(
        classify_generator_model=args.classify_generator_model,
        updater_model=args.updater_model,
        max_depth=config.inference.max_depth,
        classify_generator_gpu_id=config.inference.classify_generator_gpu_id,
        updater_gpu_id=config.inference.updater_gpu_id,
        classifier_batch_size=config.inference.classifier_batch_size,
        updater_batch_size=config.inference.updater_batch_size,
        classifier_timeout=config.inference.classifier_timeout,
        updater_timeout=config.inference.updater_timeout,
        max_model_len=config.inference.max_model_len,
        gpu_memory_utilization=config.inference.gpu_memory_utilization,
        temperature=config.inference.temperature,
        top_p=config.inference.top_p,
        max_workers=args.max_workers
    )
    
    # 执行并行推理
    all_trees = processor.process_topics(topics_data, max_refs=max_refs)
    
    # 保存结果
    output_dir = Path(config.path.inference_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'{args.split}_trees.json'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_trees, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"推理完成！")
    print(f"  - 结果已保存到: {output_file}")
    print(f"  - 成功推理的topic数: {len(all_trees)}/{len(topics_data)}")
    print("="*80)


if __name__ == '__main__':
    main()

