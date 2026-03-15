"""
临时脚本：将旧格式的分类数据集转换为新格式（包含说明性文字）

同时转换prompt和completion：
- Prompt: 移除结构信息，简化输出格式说明
- Completion: 添加说明性文字和 "Final answer:"
"""

import json
import argparse
import re
from pathlib import Path
from typing import Dict, Optional, List


def extract_prompt_parts(prompt: str) -> Dict:
    """从旧prompt中提取关键部分"""
    parts = {}
    
    # 提取Topic
    topic_match = re.search(r'\*\*Topic\*\*:\s*(.+?)(?:\n\n|\*\*)', prompt, re.DOTALL)
    if topic_match:
        parts['topic_name'] = topic_match.group(1).strip()
    else:
        parts['topic_name'] = "Unknown Topic"
    
    # 提取Current Node Summary
    summary_match = re.search(r'\*\*Current Node Summary\*\*:\s*(.+?)(?:\n\n|\*\*)', prompt, re.DOTALL)
    if summary_match:
        parts['current_summary'] = summary_match.group(1).strip()
    else:
        parts['current_summary'] = ""
    
    # 提取Article Content
    article_match = re.search(r'\*\*Article Content\*\*:\s*(.+?)(?:\n\n|\*\*)', prompt, re.DOTALL)
    if article_match:
        parts['article_content'] = article_match.group(1).strip()
    else:
        parts['article_content'] = ""
    
    # 提取Existing Child Categories
    categories_match = re.search(r'\*\*Existing Child Categories\*\*:\s*(.+?)(?:\n\n\*\*(?:Current Structure|Your Task|Output Format)|$)', prompt, re.DOTALL)
    if categories_match:
        parts['children_text'] = categories_match.group(1).strip()
    else:
        parts['children_text'] = "No existing child categories."
    
    return parts


def format_new_prompt(parts: Dict) -> str:
    """使用新模板生成prompt"""
    
    # 新的prompt模板（简化版）
    new_template = """You are a hierarchical article classifier. Your task is to classify an article into existing categories and create a new category if needed.

**Topic**: {topic_name}

**Current Node Summary**:
{current_summary}

**Article Content**:
{article_content}

**Existing Child Categories**:
{children_text}

**Your Task**:
Analyze the article and determine:
1) Which existing categories (if any) does the article belong to? You can select multiple categories.
2) Does the article require a NEW category that doesn't exist yet?
3) Should any two existing categories be MERGED because they are semantically similar?

**Output Format** (you MUST follow this exact format with three lines):

Line 1 - SELECTED: [Explain which categories match the article and why, then conclude with the indices]
Output format: "Based on analysis, the article matches category/categories [X, Y] because [reason]. Final answer: X,Y" or "None of the existing categories match. Final answer: NONE"

Line 2 - NEW: [Explain whether a new category is needed and why, then give Yes/No]
Output format: "The article [does/does not] require a new category because [reason]. Final answer: Yes/No"

Line 3 - MERGE_WITH: [Explain which categories should be merged and why, then give the index]
Output format: "Categories [X] and [Y] should be merged because [reason]. Final answer: X" or "No merge needed. Final answer: NONE"

**Example 1** (No existing categories match):
SELECTED: None of the existing categories match the article's focus on quantum computing. Final answer: NONE
NEW: The article requires a new category for quantum computing topics. Final answer: Yes
MERGE_WITH: The new category should be merged with category 2 (Physics) as they are closely related. Final answer: 2

**Example 2** (Classify to existing, no merge):
SELECTED: The article discusses World War II battles, matching category 1 (Military History). Final answer: 1
NEW: The existing category adequately covers this topic. Final answer: No
MERGE_WITH: No merge is needed at this time. Final answer: NONE

Now classify the article:
"""
    
    return new_template.format(
        topic_name=parts['topic_name'],
        current_summary=parts['current_summary'],
        article_content=parts['article_content'],
        children_text=parts['children_text']
    )


def parse_old_completion(completion: str) -> Dict:
    """解析旧格式的completion"""
    selected = None
    need_new = False
    merge_with = None
    
    lines = completion.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if line.startswith('SELECTED'):
            parts = line.split(':', 1)
            if len(parts) >= 2:
                value = parts[1].strip()
                if value.upper() not in ['NONE', 'N/A', '']:
                    selected = value
                else:
                    selected = 'NONE'
        
        elif line.startswith('NEW'):
            parts = line.split(':', 1)
            if len(parts) >= 2:
                value = parts[1].strip().upper()
                need_new = (value in ['YES', 'Y', 'TRUE'])
        
        elif line.startswith('MERGE_WITH'):
            parts = line.split(':', 1)
            if len(parts) >= 2:
                value = parts[1].strip()
                if value.upper() not in ['NONE', 'N/A', '']:
                    merge_with = value
                else:
                    merge_with = 'NONE'
    
    return {
        'selected': selected,
        'need_new': need_new,
        'merge_with': merge_with
    }


def format_new_completion(parsed: Dict) -> str:
    """生成新格式的completion（包含说明性文字）"""
    lines = []
    
    # Line 1 - SELECTED
    selected = parsed['selected']
    if selected == 'NONE':
        lines.append("SELECTED: None of the existing categories match this article. Final answer: NONE")
    elif ',' in selected:
        # 多分类
        lines.append(f"SELECTED: The article matches multiple categories. Final answer: {selected}")
    else:
        # 单分类
        lines.append(f"SELECTED: The article belongs to category {selected} as it aligns with that category's scope. Final answer: {selected}")
    
    # Line 2 - NEW
    if parsed['need_new']:
        lines.append("NEW: A new category is required for content not covered by existing categories. Final answer: Yes")
    else:
        lines.append("NEW: Existing categories adequately cover this article's content. Final answer: No")
    
    # Line 3 - MERGE_WITH
    merge_with = parsed['merge_with']
    if merge_with == 'NONE':
        lines.append("MERGE_WITH: No merge operation is needed at this time. Final answer: NONE")
    else:
        lines.append(f"MERGE_WITH: Categories should be merged as they are semantically similar. Final answer: {merge_with}")
    
    return '\n'.join(lines)


def convert_file(input_file: Path, output_file: Path):
    """转换单个文件（同时转换prompt和completion）"""
    print(f"\n处理文件: {input_file}")
    
    if not input_file.exists():
        print(f"  警告: 文件不存在，跳过")
        return
    
    samples = []
    converted_count = 0
    error_count = 0
    
    # 读取原始数据
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                sample = json.loads(line)
                
                # 1. 转换prompt：提取关键部分并重新生成
                old_prompt = sample['prompt']
                prompt_parts = extract_prompt_parts(old_prompt)
                new_prompt = format_new_prompt(prompt_parts)
                
                # 2. 转换completion：解析旧格式并生成新格式
                old_completion = sample['completion']
                parsed_completion = parse_old_completion(old_completion)
                new_completion = format_new_completion(parsed_completion)
                
                # 创建新样本
                new_sample = {
                    'prompt': new_prompt,
                    'completion': new_completion,
                }
                
                # 保留metadata（如果有）
                if 'metadata' in sample:
                    new_sample['metadata'] = sample['metadata']
                
                samples.append(new_sample)
                converted_count += 1
                
            except Exception as e:
                print(f"  错误: 第{line_num}行转换失败: {e}")
                error_count += 1
                import traceback
                traceback.print_exc()
    
    # 保存转换后的数据
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"  ✓ 转换完成: {converted_count} 条")
    if error_count > 0:
        print(f"  ✗ 错误: {error_count} 条")
    print(f"  保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='转换分类数据集格式')
    parser.add_argument(
        '--input_dir',
        type=str,
        default='/mnt/literism/tree/summary_output/data/oracle_data_model',
        help='输入目录'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='输出目录（默认为输入目录下的 converted/ 子目录）'
    )
    parser.add_argument(
        '--files',
        nargs='+',
        default=['classification_train.jsonl', 'classification_train_balanced.jsonl'],
        help='要转换的文件名列表'
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_dir / 'converted'
    
    print("="*80)
    print("转换分类数据集格式")
    print("="*80)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"文件列表: {args.files}")
    print("="*80)
    
    # 转换每个文件
    for filename in args.files:
        input_file = input_dir / filename
        output_file = output_dir / filename
        convert_file(input_file, output_file)
    
    print("\n" + "="*80)
    print("转换完成！")
    print("="*80)
    print(f"\n转换后的文件保存在: {output_dir}")
    print("\n转换内容：")
    print("  ✓ Prompt: 移除结构信息，简化输出格式说明")
    print("  ✓ Completion: 添加说明性文字和 'Final answer:'")
    print("\n使用方法：")
    print(f"1. 检查转换后的文件: {output_dir}")
    print(f"   head -n 1 {output_dir / args.files[0]} | jq .")
    print(f"\n2. 如果确认无误，可以替换原文件：")
    for filename in args.files:
        print(f"   cp {output_dir / filename} {input_dir / filename}")
    print("\n3. 或者直接在训练时使用转换后的文件路径")
    print(f"   注意：需要修改训练脚本中的数据路径为 {output_dir}")


if __name__ == '__main__':
    main()
