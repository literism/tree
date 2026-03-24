"""
Oracle风格推理脚本
使用与数据生成相同的推理逻辑，但使用训练好的模型
"""
import os
import json
import argparse
from pathlib import Path
from summary_based_classifier.config import SummaryBasedConfig
from summary_based_classifier.inference.oracle_style_inference_processor import OracleStyleInferenceProcessor
from summary_based_classifier.llm.deepseek_api import DeepSeekConfig


def main():
    parser = argparse.ArgumentParser(description='Oracle风格推理（使用训练好的模型）')
    parser.add_argument('--config', type=str, default='./configs/default.json')
    parser.add_argument('--classify_generator_model', type=str, required=True, help='分类生成模型路径')
    parser.add_argument('--updater_model', type=str, default=None, help='总结更新模型路径（updater_mode=model时需要）')
    parser.add_argument('--updater_mode', type=str, default='model', choices=['model', 'api'], help='总结后端：model或api')
    parser.add_argument('--updater_api_key', type=str, default="", help='API Key（updater_mode=api时需要）')
    parser.add_argument('--updater_api_url', type=str, default='https://api.deepseek.com', help='API Base URL（updater_mode=api）')
    parser.add_argument('--updater_api_model', type=str, default='deepseek-chat', help='API模型名（updater_mode=api）')
    parser.add_argument('--updater_api_max_output_tokens', type=int, default=2048, help='API最大输出tokens')
    parser.add_argument('--updater_api_max_concurrent_jobs', type=int, default=8, help='API并发数')
    parser.add_argument('--split', type=str, default='test', help='推理的数据划分')
    parser.add_argument('--max_refs', type=int, default=None, help='每个topic最多处理的文章数')
    parser.add_argument('--max_workers', type=int, default=4, help='最大并行topic数')
    parser.add_argument('--classify_gpu', type=str, default='0', help='分类模型使用的GPU，可为单卡ID')
    parser.add_argument('--updater_gpu', type=str, default='1', help='总结模型使用的GPU，可为逗号分隔多卡ID')
    
    args = parser.parse_args()
    
    # 加载配置
    config = SummaryBasedConfig.from_json(args.config)
    
    print("="*80)
    print("Oracle风格推理系统")
    print("="*80)
    print(f"配置文件: {args.config}")
    print(f"分类模型: {args.classify_generator_model}")
    print(f"总结后端: {args.updater_mode}")
    print(f"总结模型: {args.updater_model if args.updater_mode == 'model' else args.updater_api_model}")
    print(f"数据划分: {args.split}")
    print(f"GPU配置: 分类模型GPU={args.classify_gpu}, 总结模型GPU={args.updater_gpu}")
    print(f"最大并行数: {args.max_workers}")
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
    
    # 创建Oracle风格推理处理器
    print(f"\n初始化Oracle风格推理处理器...")
    
    # 获取tokenizer名称
    try:
        tokenizer_name = config.updater.tokenizer_name if hasattr(config, 'updater') else None
    except:
        tokenizer_name = None
    
    if not tokenizer_name or tokenizer_name == "base_model":
        # 使用base_model作为tokenizer
        tokenizer_name = config.path.base_model
    
    updater_api_config = None
    if args.updater_mode == "api":
        if not args.updater_api_key:
            raise ValueError("updater_mode=api 时必须提供 --updater_api_key")
        updater_api_config = DeepSeekConfig(
            api_key=args.updater_api_key,
            base_url=args.updater_api_url,
            model=args.updater_api_model,
            temperature=config.inference.temperature,
            max_output_tokens=args.updater_api_max_output_tokens,
            max_concurrent_jobs=args.updater_api_max_concurrent_jobs,
        )
    elif not args.updater_model:
        raise ValueError("updater_mode=model 时必须提供 --updater_model")

    processor = OracleStyleInferenceProcessor(
        classify_generator_model=args.classify_generator_model,
        updater_model=args.updater_model,
        max_depth=config.inference.max_depth,
        classify_generator_gpu_id=args.classify_gpu,
        updater_gpu_id=args.updater_gpu,
        updater_mode=args.updater_mode,
        updater_api_config=updater_api_config,
        classifier_batch_size=getattr(config.inference, 'classifier_batch_size', 32),
        updater_batch_size=getattr(config.inference, 'updater_batch_size', 32),
        classifier_timeout=getattr(config.inference, 'classifier_timeout', 0.1),
        updater_timeout=getattr(config.inference, 'updater_timeout', 0.1),
        max_model_len=config.inference.max_model_len,
        gpu_memory_utilization=config.inference.gpu_memory_utilization,
        temperature=config.inference.temperature,
        top_p=config.inference.top_p,
        max_workers=args.max_workers,
        max_content_length=config.summary.max_content_length,
        tokenizer_name=tokenizer_name
    )
    
    # 执行并行推理
    all_trees = processor.process_topics(topics_data, max_refs=max_refs)
    
    # 保存结果
    output_dir = Path(config.path.inference_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'{args.split}_trees_oracle_style.json'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_trees, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"推理完成！")
    print(f"  - 结果已保存到: {output_file}")
    print(f"  - 成功推理的topic数: {len(all_trees)}/{len(topics_data)}")
    print("="*80)


if __name__ == '__main__':
    main()
