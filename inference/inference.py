"""
新的推理脚本
使用新的构建系统进行推理
"""
import os
import json
import argparse
from pathlib import Path
from summary_based_classifier.config import SummaryBasedConfig
from summary_based_classifier.core.pipeline.builder import TreeBuilder
from summary_based_classifier.core.trajectory.trajectory_sampler import TreeNode
from summary_based_classifier.llm.classify_generator import ClassifyGenerator
from summary_based_classifier.llm.updater import Updater


def main():
    parser = argparse.ArgumentParser(description='使用新模型进行推理')
    parser.add_argument('--config', type=str, default='./configs/default.json')
    parser.add_argument('--classify_generator_model', type=str, required=True)
    parser.add_argument('--updater_model', type=str, required=True)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--mode', type=str, default='direct', choices=['direct', 'service'])
    
    args = parser.parse_args()
    
    # 加载配置
    config = SummaryBasedConfig.from_json(args.config)
    
    print("="*80)
    print("新推理系统")
    print("="*80)
    
    # 加载数据
    print("\n加载数据...")
    with open(config.path.references_file, 'r', encoding='utf-8') as f:
        references_data = json.load(f)
    
    with open(Path(config.path.data_dir) / 'dataset_split.json', 'r', encoding='utf-8') as f:
        split_data = json.load(f)
        dataset_split = split_data['dataset_split']
    
    # 加载模型
    print("\n加载模型...")
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.inference.classify_generator_gpu_id)
    
    classifier = ClassifyGenerator(
        mode='model',
        model_path=args.classify_generator_model,
        max_model_len=config.inference.max_model_len,
        gpu_memory_utilization=config.inference.gpu_memory_utilization,
        temperature=config.inference.temperature,
        top_p=config.inference.top_p,
        max_tokens=256
    )
    
    updater = Updater(
        mode='model',
        model_path=args.updater_model,
        max_model_len=config.inference.max_model_len,
        gpu_memory_utilization=config.inference.gpu_memory_utilization,
        temperature=config.inference.temperature,
        top_p=config.inference.top_p,
        max_tokens=512
    )
    
    # 推理
    print(f"\n开始推理 ({args.split} split)...")
    all_trees = {}
    
    for topic_key, ref_ids in dataset_split.get(args.split, {}).items():
        if topic_key not in references_data:
            continue
        
        topic_data = references_data[topic_key]
        topic_name = topic_data.get('topic', topic_key)
        
        print(f"\n处理topic: {topic_name}")
        
        # 创建根节点
        root = TreeNode(summary="", citations=[], children=[], depth=0)
        
        # 创建构建器
        builder = TreeBuilder(
            classifier=classifier,
            updater=updater,
            topic_name=topic_name,
            max_depth=config.inference.max_depth
        )
        
        # 准备文章列表
        articles = []
        for ref_id in ref_ids[:config.inference.max_refs] if config.inference.max_refs else ref_ids:
            if ref_id in topic_data.get('references', {}):
                ref = topic_data['references'][ref_id]
                articles.append({
                    'id': ref_id,
                    'content': ref.get('content', '')
                })
        
        # 构建树
        root = builder.build_tree_for_articles(articles, root)
        
        # 转换为字典
        tree_dict = {
            'topic': topic_name,
            'structure': [builder.tree_to_dict(child, level=2) for child in root.children]
        }
        
        all_trees[topic_key] = tree_dict
    
    # 保存结果
    output_dir = Path(config.path.inference_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'{args.split}_trees.json'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_trees, f, indent=2, ensure_ascii=False)
    
    print(f"\n推理完成！结果已保存到: {output_file}")
    print("="*80)


if __name__ == '__main__':
    main()
