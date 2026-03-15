#!/usr/bin/env python3
"""
合并LoRA adapter到基础模型
不使用量化，生成可用于vLLM推理的完整模型
"""
import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_adapter(
    base_model_path: str,
    adapter_path: str,
    output_path: str
):
    """
    合并LoRA adapter到基础模型
    
    Args:
        base_model_path: 基础模型路径
        adapter_path: LoRA adapter路径
        output_path: 输出路径
    """
    print("="*80)
    print("合并LoRA Adapter到基础模型")
    print("="*80)
    
    # 0. 加载tokenizer（必须是训练时最终版本）
    print(f"\n0. 加载tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,  # 从adapter路径加载，确保是训练时的版本
        trust_remote_code=True
    )
    print(f"   ✓ Tokenizer加载完成 (vocab size: {len(tokenizer)})")
    
    # 1. 加载干净的base模型
    print(f"\n1. 加载干净的base模型: {base_model_path}")
    print("   - 不使用量化")
    print("   - 使用auto精度")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    print("   ✓ 基础模型加载完成")
    
    # 3. 加载LoRA adapter
    print(f"\n3. 加载LoRA adapter: {adapter_path}")
    model_with_adapter = PeftModel.from_pretrained(
        base_model,
        adapter_path,
    )
    print("   ✓ Adapter加载完成")
    
    # 4. 合并LoRA（此后是纯transformers模型）
    print(f"\n4. 合并adapter到基础模型")
    merged_model = model_with_adapter.merge_and_unload()
    print("   ✓ 合并完成")

        # 2. 对齐embedding vocab（关键步骤！）
    print(f"\n2. 对齐embedding层")
    num_added = len(tokenizer.get_added_vocab())
    true_vocab_size = tokenizer.vocab_size + num_added
    print(f"   - 原始embedding size: {merged_model.get_input_embeddings().weight.shape[0]}")
    print(f"   - Tokenizer vocab size: {true_vocab_size}")
    if merged_model.get_input_embeddings().weight.shape[0] != true_vocab_size:
        merged_model.resize_token_embeddings(true_vocab_size)
        print(f"   - 调整后embedding size: {merged_model.get_input_embeddings().weight.shape[0]}")
        print("   ✓ Embedding层对齐完成")
    else:
        print("   ✓ Embedding层已对齐")
    merged_model.config.vocab_size = true_vocab_size

    output_dir = Path(output_path)
    
    # 7. 保存最终模型
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n7. 保存最终模型到: {output_path}")
    merged_model.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))
    print("   ✓ 最终保存完成")

    print(f"   - 模型大小: {sum(p.numel() for p in merged_model.parameters()) / 1e9:.2f}B 参数")
    
    print("\n" + "="*80)
    print("合并完成！")
    print("="*80)
    print(f"输出路径: {output_path}")
    print("此模型可用于vLLM推理")


def main():
    parser = argparse.ArgumentParser(description='合并LoRA adapter到基础模型')
    parser.add_argument(
        '--base_model',
        type=str,
        default='/home/literism/model/Qwen3-32B',
        help='基础模型路径'
    )
    parser.add_argument(
        '--adapter',
        type=str,
        default='/mnt/literism/tree/summary_output/data/labeling_model_training/checkpoints/checkpoint-2700',
        help='LoRA adapter路径'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='/mnt/literism/tree/summary_output/data/labeling_model_training/final_model',
        help='输出路径'
    )
    
    args = parser.parse_args()
    
    merge_adapter(
        base_model_path=args.base_model,
        adapter_path=args.adapter,
        output_path=args.output
    )


if __name__ == '__main__':
    main()

