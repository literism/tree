"""
使用训练好的模型推理生成结构树
"""
import json
from pathlib import Path
from typing import Dict, List, Optional
import argparse
import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from config import StructureGeneratorConfig


class StructureGenerator:
    """结构树生成器"""
    
    def __init__(
        self,
        base_model: str,
        adapter_model: str,
        device: str = 'cuda'
    ):
        """
        Args:
            base_model: 基础模型路径
            adapter_model: LoRA适配器路径
            device: 设备
        """
        self.device = device
        
        print("加载模型...")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"  - Tokenizer加载完成")
        
        # 加载基础模型
        base_model_obj = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            trust_remote_code=True
        )
        
        # 加载LoRA适配器
        self.model = PeftModel.from_pretrained(
            base_model_obj,
            adapter_model,
            torch_dtype=torch.bfloat16
        )
        
        self.model.eval()
        
        print(f"  - 模型加载完成")
    
    def create_prompt(self, topic: str, intro: str) -> str:
        """
        创建推理prompt（与训练时相同）
        
        Args:
            topic: topic标题
            intro: topic的介绍文本
            
        Returns:
            prompt字符串
        """
        prompt = f"""TASK: Generate a hierarchical structure tree for the given topic based on its introduction.

TOPIC: {topic}

INTRODUCTION:
{intro}

INSTRUCTIONS:
1. Analyze the introduction and identify the main aspects, themes, or categories related to this topic.
2. Create a multi-level hierarchical structure that organizes these aspects logically.
3. Each node should have a clear title.
4. Use appropriate level numbers (level 2 for main titles, level 3 for subtitles, etc.).
5. The structure should be comprehensive but not overly detailed.
6. Format each line as: "- Title (level N)" where N is the level number.
7. Use indentation (2 spaces per level) to show hierarchy.

OUTPUT FORMAT:
- Main Title 1 (level 2)
  - Subtitle 1.1 (level 3)
  - Subtitle 1.2 (level 3)
- Main Title 2 (level 2)
  - Subtitle 2.1 (level 3)
    - Sub-subtitle 2.1.1 (level 4)

STRUCTURE:
"""
        return prompt
    
    def parse_structure_text(self, text: str) -> List[Dict]:
        """
        解析模型生成的结构树文本
        
        Args:
            text: 模型生成的文本
            
        Returns:
            结构树（sections格式）
        """
        lines = text.strip().split('\n')
        
        # 解析每一行
        parsed_lines = []
        for line in lines:
            # 匹配格式: "  - Title (level N)"
            match = re.match(r'^(\s*)- (.+?) \(level (\d+)\)$', line)
            if match:
                indent = len(match.group(1))
                title = match.group(2).strip()
                level = int(match.group(3))
                parsed_lines.append({
                    'indent': indent,
                    'title': title,
                    'level': level
                })
        
        if not parsed_lines:
            return []
        
        # 构建树结构
        root_sections = []
        stack = []  # (indent, section_dict)
        
        for item in parsed_lines:
            section = {
                'title': item['title'],
                'level': item['level'],
                'children': []
            }
            
            # 找到父节点
            while stack and stack[-1][0] >= item['indent']:
                stack.pop()
            
            if not stack:
                # 顶层节点
                root_sections.append(section)
            else:
                # 添加到父节点
                stack[-1][1]['children'].append(section)
            
            stack.append((item['indent'], section))
        
        return root_sections
    
    def generate_structure(
        self,
        topic: str,
        intro: str,
        max_new_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> Dict:
        """
        生成结构树
        
        Args:
            topic: topic标题
            intro: topic的介绍文本
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            top_p: top_p参数
            
        Returns:
            包含原始文本和解析后结构的字典
        """
        # 创建prompt
        prompt = self.create_prompt(topic, intro)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=8192 - max_new_tokens
        ).to(self.model.device)
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # 解码
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # 解析结构
        sections = self.parse_structure_text(generated_text)
        
        return {
            'topic': topic,
            'generated_text': generated_text,
            'sections': sections,
            'node_count': self._count_nodes(sections)
        }
    
    def _count_nodes(self, sections: List[Dict]) -> int:
        """统计节点数"""
        count = 0
        for section in sections:
            count += 1
            count += self._count_nodes(section.get('children', []))
        return count


def generate_for_split(
    generator: StructureGenerator,
    intro_file: str,
    output_file: str,
    config: StructureGeneratorConfig
):
    """
    为指定的topics生成结构树
    
    Args:
        generator: 结构树生成器
        intro_file: intro文件路径
        output_file: 输出文件路径
        config: 配置
    """
    # 加载intro数据
    print(f"\n加载intro数据: {intro_file}")
    with open(intro_file, 'r', encoding='utf-8') as f:
        intros = json.load(f)
    
    print(f"  - 共 {len(intros)} 个topics")
    
    # 生成结构树
    results = {}
    
    for i, (topic, intro) in enumerate(intros.items()):
        print(f"\n处理 [{i+1}/{len(intros)}]: {topic}")
        
        try:
            result = generator.generate_structure(
                topic=topic,
                intro=intro,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p
            )
            
            results[topic] = {
                'sections': result['sections'],
                'node_count': result['node_count'],
                'generated_text': result['generated_text']
            }
            
            print(f"  ✓ 生成成功，节点数: {result['node_count']}")
            
        except Exception as e:
            print(f"  ✗ 生成失败: {e}")
            results[topic] = {
                'sections': [],
                'node_count': 0,
                'error': str(e)
            }
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果保存到: {output_file}")
    
    # 统计
    success_count = sum(1 for r in results.values() if r['node_count'] > 0)
    print(f"\n统计:")
    print(f"  - 成功: {success_count}/{len(results)}")
    print(f"  - 失败: {len(results) - success_count}/{len(results)}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='生成结构树')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--model', type=str, help='模型路径（LoRA适配器）')
    parser.add_argument('--split', type=str, choices=['train', 'test_easy', 'test_hard', 'all'],
                       default='all', help='生成哪个split的结构树')
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config:
        config = StructureGeneratorConfig.from_json(args.config)
    else:
        config = StructureGeneratorConfig()
    
    # 模型路径
    if args.model:
        model_path = args.model
    else:
        model_path = str(Path(config.models_dir) / 'structure_generator' / 'adapter' / 'checkpoint-7200')
    
    print(f"使用模型: {model_path}")
    
    # 创建生成器
    generator = StructureGenerator(
        base_model=config.base_model,
        adapter_model=model_path
    )
    
    # 创建输出目录
    output_dir = Path(config.inference_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载excluded topics intro
    excluded_intro_file = Path(config.data_dir) / 'excluded_topics_intro.json'
    with open(excluded_intro_file, 'r', encoding='utf-8') as f:
        all_intros = json.load(f)
    
    # 加载split信息
    with open(config.dataset_split_file, 'r', encoding='utf-8') as f:
        split_data = json.load(f)
    
    # 根据split生成
    splits_to_process = []
    if args.split == 'all':
        splits_to_process = ['train', 'test_easy', 'test_hard']
    else:
        splits_to_process = [args.split]
    
    for split in splits_to_process:
        print("\n" + "="*60)
        print(f"生成 {split} 的结构树")
        print("="*60)
        
        # 筛选该split的topics（兼容两种格式）
        split_key = f'{split}_topics' if f'{split}_topics' in split_data else split
        split_topics = split_data[split_key]
        split_intros = {topic: all_intros[topic] for topic in split_topics if topic in all_intros}
        
        # 保存该split的intro（临时文件）
        temp_intro_file = output_dir / f'{split}_intros.json'
        with open(temp_intro_file, 'w', encoding='utf-8') as f:
            json.dump(split_intros, f, indent=2, ensure_ascii=False)
        
        # 生成结构树
        output_file = output_dir / f'{split}_structures.json'
        generate_for_split(generator, str(temp_intro_file), str(output_file), config)
        
        # 删除临时文件
        temp_intro_file.unlink()
    
    print("\n" + "="*60)
    print("全部完成！")
    print("="*60)


if __name__ == '__main__':
    main()

