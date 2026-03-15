#!/usr/bin/env python3
"""
生成器服务
使用vLLM在指定GPU上部署生成器模型，提供HTTP API
"""
import os
import argparse
from flask import Flask, request, jsonify
from generator import SummaryGenerator, GenerationInput

app = Flask(__name__)
generator = None


@app.route('/generate', methods=['POST'])
def generate():
    """生成接口"""
    try:
        data = request.json
        
        # 创建输入
        input_data = GenerationInput(
            article_content=data['article_content'],
            current_node_summary=data['current_node_summary'],
            existing_summaries=data['existing_summaries'],
            topic_name=data['topic_name']
        )
        
        # 调用生成器
        max_new_classes = data.get('max_new_classes', 2)
        output = generator.generate(input_data, max_new_classes=max_new_classes)
        
        # 返回结果
        return jsonify({
            'new_summaries': output.new_summaries
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({'status': 'ok'})


def main():
    parser = argparse.ArgumentParser(description='生成器服务')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--gpu_id', type=int, default=1, help='GPU ID')
    parser.add_argument('--port', type=int, default=5001, help='服务端口')
    parser.add_argument('--max_model_len', type=int, default=8192, help='最大模型长度')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9, help='GPU显存利用率')
    parser.add_argument('--temperature', type=float, default=0.3, help='温度')
    parser.add_argument('--top_p', type=float, default=0.9, help='top_p')
    parser.add_argument('--max_tokens', type=int, default=512, help='最大生成token数')
    
    args = parser.parse_args()
    
    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    
    print("="*80)
    print("生成器服务")
    print("="*80)
    print(f"模型路径: {args.model_path}")
    print(f"GPU ID: {args.gpu_id}")
    print(f"服务端口: {args.port}")
    print(f"显存利用率: {args.gpu_memory_utilization}")
    
    # 加载模型
    print("\n加载生成器模型...")
    global generator
    generator = SummaryGenerator(
        mode='model',
        model_path=args.model_path,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens
    )
    print("✓ 模型加载完成")
    
    # 启动服务
    print(f"\n启动服务于 http://0.0.0.0:{args.port}")
    print("="*80)
    # 使用单线程模式，避免与vLLM冲突
    app.run(host='0.0.0.0', port=args.port, threaded=False, processes=1)


if __name__ == '__main__':
    main()

