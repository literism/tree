"""
Oracle策略推理脚本
使用训练好的分类模型 + BOW模式的Updater进行推理
支持双GPU配置（分类模型和Updater各占一张卡）
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
    parser = argparse.ArgumentParser(description='Oracle策略推理')
    parser.add_argument('--config', type=str, default='./configs/default.json')
    parser.add_argument('--classify_generator_model', type=str, required=True, help='分类生成模型路径')
    parser.add_argument('--split', type=str, default='test', help='推理的数据划分')
    parser.add_argument('--classify_gpu', type=str, default='0', help='分类模型使用的GPU')
    parser.add_argument('--updater_gpu', type=str, default='1', help='Updater使用的GPU（BOW模式不实际使用）')
    parser.add_argument('--bow_top_k', type=int, default=30, help='BOW summary的top-k词数')
    parser.add_argument('--max_refs', type=int, default=None, help='每个topic最多处理的文章数')
    
    args = parser.parse_args()
    
    # 加载配置
    config = SummaryBasedConfig.from_json(args.config)
    
    print("="*80)
    print("Oracle策略推理系统")
    print("="*80)
    print(f"配置文件: {args.config}")
    print(f"分类模型: {args.classify_generator_model}")
    print(f"Updater模式: BOW (top_k={args.bow_top_k})")
    print(f"数据划分: {args.split}")
    print(f"GPU配置: 分类模型GPU={args.classify_gpu}, Updater GPU={args.updater_gpu}")
    print("="*80)
    
    # 检查模型文件
    if not Path(args.classify_generator_model).exists():
        print(f"\n错误: 分类模型不存在: {args.classify_generator_model}")
        return
    
    # 加载数据
    print("\n加载数据...")
    with open(config.path.references_file, 'r', encoding='utf-8') as f:
        references_data = json.load(f)
    
    with open(Path(config.path.data_dir) / 'dataset_split.json', 'r', encoding='utf-8') as f:
        split_data = json.load(f)
        dataset_split = split_data['dataset_split']
    
    # 加载分类模型（使用第一张GPU）
    print(f"\n加载分类模型到GPU {args.classify_gpu}...")
    os.environ['CUDA_VISIBLE_DEVICES'] = args.classify_gpu
    
    classifier = ClassifyGenerator(
        mode='model',
        model_path=args.classify_generator_model,
        max_model_len=config.inference.max_model_len,
        gpu_memory_utilization=config.inference.gpu_memory_utilization,
        temperature=config.inference.temperature,
        top_p=config.inference.top_p,
        max_tokens=256
    )
    
    # 创建BOW模式的Updater（不需要GPU）
    print(f"\n创建BOW Updater (top_k={args.bow_top_k})...")
    updater = Updater(
        mode='bow',
        bow_top_k=args.bow_top_k
    )
    
    # 推理
    print(f"\n开始推理 ({args.split} split)...")
    all_trees = {}
    
    topics = list(dataset_split.get(args.split, {}).items())
    total_topics = len(topics)
    
    for idx, (topic_key, ref_ids) in enumerate(topics, 1):
        if topic_key not in references_data:
            continue
        
        topic_data = references_data[topic_key]
        topic_name = topic_data.get('topic', topic_key)
        
        print(f"\n[{idx}/{total_topics}] 处理topic: {topic_name}")
        print(f"  文章数: {len(ref_ids)}")
        
        # 准备文章列表
        articles = []
        max_refs = args.max_refs if args.max_refs else config.inference.max_refs
        ref_list = ref_ids[:max_refs] if max_refs else ref_ids
        
        for ref_id in ref_list:
            if ref_id in topic_data.get('references', {}):
                ref = topic_data['references'][ref_id]
                articles.append({
                    'id': ref_id,
                    'content': ref.get('content', '')
                })
        
        print(f"  实际处理: {len(articles)}篇文章")
        
        # 收集BM25统计信息
        from collections import defaultdict
        df = defaultdict(int)  # 文档频率
        doc_lengths = []
        
        for article in articles:
            content = article['content']
            if not content:
                continue
            # tokenize并统计
            bow = updater._bow_from_text(content)
            doc_lengths.append(sum(bow.values()))
            
            # 更新df
            for term in bow.keys():
                df[term] += 1
        
        avg_len = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 1.0
        bm25_stats = {
            'df': dict(df),
            'total_docs': len(doc_lengths),
            'avg_doc_length': avg_len
        }
        
        # 创建根节点
        root = TreeNode(summary="", citations=[], children=[], depth=0)
        
        # 创建构建器（传递BM25统计信息）
        builder = TreeBuilder(
            classifier=classifier,
            updater=updater,
            topic_name=topic_name,
            max_depth=config.inference.max_depth,
            bm25_stats=bm25_stats
        )
        
        # 构建树
        try:
            # 使用build_tree_for_articles方法
            builder.build_tree_for_articles(articles, root)
            
            # 转换为可序列化格式
            tree_dict = builder.tree_to_dict(root)
            all_trees[topic_key] = tree_dict
            
            print(f"  ✓ 构建完成，树深度: {builder.get_tree_depth(root)}")
        except Exception as e:
            print(f"  ✗ 构建失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存结果
    output_dir = Path(config.path.inference_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'{args.split}_trees_oracle.json'
    
    print(f"\n保存推理结果到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_trees, f, ensure_ascii=False, indent=2)
    
    print(f"\n推理完成！")
    print(f"  - 处理的topics: {len(all_trees)}/{total_topics}")
    print(f"  - 输出文件: {output_file}")
    print("="*80)


if __name__ == '__main__':
    main()
