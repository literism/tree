"""
训练前的数据集平衡

对总结模型和分类模型的训练数据进行平衡：
- 总结模型：平衡 "需要更新" 和 "不需要更新" 的样本
- 分类模型：平衡 "创建新类"、"归拢"、"只分类" 的样本
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import Counter


def _parse_classification_completion(completion: str) -> Optional[Tuple[bool, bool]]:
    """
    解析分类completion，返回 (has_new, has_merge)
    兼容：
    1) 新格式JSON字符串
    2) 旧三段式文本（SELECTED/NEW/MERGE_WITH）
    3) 字面量 "\\n" 换行
    """
    if not isinstance(completion, str):
        return None

    clean = completion.strip().replace("```json", "").replace("```", "").strip()
    clean = clean.replace("\\r\\n", "\n").replace("\\n", "\n")

    # 0) 当前格式：MATCHED_CATEGORIES / NEED_NEW / MERGE_WITH
    lines = [ln.strip() for ln in clean.split("\n") if ln.strip()]
    need_new_line = next((ln for ln in lines if ln.upper().startswith("NEED_NEW:")), None)
    merge_line = next((ln for ln in lines if ln.upper().startswith("MERGE_WITH:")), None)
    if need_new_line is not None:
        need_val = need_new_line.split(":", 1)[1].strip().lower()
        has_new = need_val in {"true", "yes", "1"}
        has_merge = False
        if merge_line is not None:
            merge_val = merge_line.split(":", 1)[1].strip().lower()
            has_merge = merge_val not in {"none", "null", "", "n/a", "no"}
        return has_new, has_merge

    # 新格式可能是：推理文本 + 分隔符 + JSON
    marker = "<<<JSON>>>"
    json_candidates = []
    if marker in clean:
        tail = clean.split(marker)[-1].strip()
        if tail:
            json_candidates.append(tail)
    json_candidates.append(clean)

    # 1) 优先JSON解析
    for candidate in json_candidates:
        parsed = None
        try:
            parsed = json.loads(candidate)
        except Exception:
            # 从文本中提取最后一个JSON对象
            obj_matches = re.findall(r"\{[\s\S]*?\}", candidate)
            for obj_text in reversed(obj_matches):
                try:
                    parsed = json.loads(obj_text)
                    break
                except Exception:
                    continue

        if isinstance(parsed, dict):
            need_new_raw = parsed.get("need_new", False)
            merge_raw = parsed.get("merge_with", None)

            if isinstance(need_new_raw, bool):
                has_new = need_new_raw
            elif isinstance(need_new_raw, str):
                has_new = need_new_raw.strip().lower() in {"yes", "y", "true", "1"}
            else:
                has_new = bool(need_new_raw)

            if isinstance(merge_raw, str) and merge_raw.strip().lower() in {"none", "null", "", "n/a", "no"}:
                merge_raw = None
            has_merge = merge_raw is not None
            return has_new, has_merge

    # 2) 回退三段文本解析
    has_new = False
    has_merge = False
    lines = [ln.strip() for ln in clean.split("\n") if ln.strip()]
    for line in lines:
        up = line.upper()
        if up.startswith("NEW:") or " NEW:" in up:
            value = line.split(":", 1)[1].strip() if ":" in line else line
            # 支持 Final answer: Yes/No
            m = re.search(r"final\s*answer\s*[:：]\s*([^\n\r]+)", value, flags=re.IGNORECASE)
            if m:
                value = m.group(1).strip()
            value_norm = re.sub(r"[。.!;；\s]+$", "", value.strip().lower())
            has_new = value_norm in {"yes", "y", "true", "1", "是", "需要", "要"}
        elif up.startswith("MERGE_WITH:") or up.startswith("MERGE:") or " MERGE_WITH:" in up:
            value = line.split(":", 1)[1].strip() if ":" in line else line
            m = re.search(r"final\s*answer\s*[:：]\s*([^\n\r]+)", value, flags=re.IGNORECASE)
            if m:
                value = m.group(1).strip()
            value_up = value.strip().upper()
            has_merge = value_up not in {"NONE", "N/A", "NULL", "[]", "NO", ""}

    return has_new, has_merge


def balance_summary_dataset(
    input_file: Path,
    output_file: Path,
    target_no_update_ratio: float = 0.3,
    seed: int = 42
):
    """
    平衡总结模型数据集
    
    Args:
        input_file: 输入的 summary_train.jsonl 文件
        output_file: 输出的平衡后的文件
        target_no_update_ratio: 不需要更新的样本的目标比例
        seed: 随机种子
    """
    random.seed(seed)
    
    print(f"\n{'='*60}")
    print("平衡总结模型数据集")
    print(f"{'='*60}")
    
    # 读取数据
    samples = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
    
    print(f"原始样本数: {len(samples)}")
    
    # 分类样本：需要更新 vs 不需要更新
    need_update_samples = []
    no_update_samples = []
    
    for sample in samples:
        completion = sample['completion']
        # 解析 NEEDS_UPDATE 字段
        needs_update = False
        for line in completion.split('\n'):
            if line.strip().startswith('NEEDS_UPDATE:'):
                answer = line.split(':', 1)[1].strip().upper()
                needs_update = (answer == 'YES')
                break
        
        if needs_update:
            need_update_samples.append(sample)
        else:
            no_update_samples.append(sample)
    
    print(f"需要更新的样本: {len(need_update_samples)}")
    print(f"不需要更新的样本: {len(no_update_samples)}")
    
    # 计算目标数量
    total_need_update = len(need_update_samples)
    target_no_update = int(total_need_update * target_no_update_ratio / (1 - target_no_update_ratio))
    
    print(f"\n目标比例: {target_no_update_ratio:.2%} (不需要更新)")
    print(f"目标不需要更新的样本数: {target_no_update}")
    
    # 如果不需要更新的样本不足，进行上采样
    if len(no_update_samples) < target_no_update:
        print(f"不需要更新的样本不足，进行上采样...")
        upsampled_no_update = []
        while len(upsampled_no_update) < target_no_update:
            upsampled_no_update.extend(random.choices(no_update_samples, k=min(len(no_update_samples), target_no_update - len(upsampled_no_update))))
        no_update_samples = upsampled_no_update
        print(f"上采样后: {len(no_update_samples)}")
    else:
        # 如果过多，随机采样
        no_update_samples = random.sample(no_update_samples, target_no_update)
        print(f"下采样后: {len(no_update_samples)}")
    
    # 合并并打乱
    balanced_samples = need_update_samples + no_update_samples
    random.shuffle(balanced_samples)
    
    print(f"\n平衡后总样本数: {len(balanced_samples)}")
    print(f"  - 需要更新: {len(need_update_samples)} ({len(need_update_samples)/len(balanced_samples):.2%})")
    print(f"  - 不需要更新: {len(no_update_samples)} ({len(no_update_samples)/len(balanced_samples):.2%})")
    
    # 保存
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in balanced_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"\n✓ 平衡后的数据已保存: {output_file}")


def balance_classification_dataset(
    input_file: Path,
    output_file: Path,
    target_new_ratio: float = 0.3,
    target_merge_ratio: float = 0.2,
    seed: int = 42
):
    """
    平衡分类模型数据集
    
    Args:
        input_file: 输入的 classification_train.jsonl 文件
        output_file: 输出的平衡后的文件
        target_new_ratio: 创建新类的目标比例
        target_merge_ratio: 归拢的目标比例
        seed: 随机种子
    """
    random.seed(seed)
    
    print(f"\n{'='*60}")
    print("平衡分类模型数据集")
    print(f"{'='*60}")
    
    # 读取数据
    samples = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
    
    print(f"原始样本数: {len(samples)}")
    
    # 分类样本：创建新类、归拢、只分类
    new_samples = []
    merge_samples = []
    select_only_samples = []
    
    parse_failed = 0
    for sample in samples:
        completion = sample.get('completion', '')
        parsed = _parse_classification_completion(completion)
        if parsed is None:
            parse_failed += 1
            has_new = False
            has_merge = False
        else:
            has_new, has_merge = parsed
        
        if has_new and has_merge:
            # 既创建新类又归拢
            merge_samples.append(sample)
        elif has_new:
            # 只创建新类
            new_samples.append(sample)
        elif has_merge:
            # 只归拢（理论上不应该出现，但保险起见）
            merge_samples.append(sample)
        else:
            # 只分类
            select_only_samples.append(sample)
    
    print(f"创建新类的样本: {len(new_samples)}")
    print(f"归拢的样本: {len(merge_samples)}")
    print(f"只分类的样本: {len(select_only_samples)}")
    if parse_failed > 0:
        print(f"completion解析失败样本: {parse_failed}（按只分类处理）")
    
    # 计算目标数量
    total_select = len(select_only_samples)
    target_new = int(total_select * target_new_ratio / (1 - target_new_ratio - target_merge_ratio))
    target_merge = int(total_select * target_merge_ratio / (1 - target_new_ratio - target_merge_ratio))
    
    print(f"\n目标比例:")
    print(f"  - 创建新类: {target_new_ratio:.2%}")
    print(f"  - 归拢: {target_merge_ratio:.2%}")
    print(f"  - 只分类: {1 - target_new_ratio - target_merge_ratio:.2%}")
    print(f"目标样本数:")
    print(f"  - 创建新类: {target_new}")
    print(f"  - 归拢: {target_merge}")
    
    # 上采样或下采样
    if len(new_samples) < target_new:
        print(f"创建新类的样本不足，进行上采样...")
        upsampled_new = []
        while len(upsampled_new) < target_new:
            upsampled_new.extend(random.choices(new_samples, k=min(len(new_samples), target_new - len(upsampled_new))))
        new_samples = upsampled_new
        print(f"  上采样后: {len(new_samples)}")
    else:
        new_samples = random.sample(new_samples, target_new)
        print(f"  下采样后: {len(new_samples)}")
    
    if len(merge_samples) < target_merge:
        print(f"归拢的样本不足，进行上采样...")
        upsampled_merge = []
        while len(upsampled_merge) < target_merge:
            upsampled_merge.extend(random.choices(merge_samples, k=min(len(merge_samples), target_merge - len(upsampled_merge))))
        merge_samples = upsampled_merge
        print(f"  上采样后: {len(merge_samples)}")
    else:
        merge_samples = random.sample(merge_samples, target_merge)
        print(f"  下采样后: {len(merge_samples)}")
    
    # 合并并打乱
    balanced_samples = new_samples + merge_samples + select_only_samples
    random.shuffle(balanced_samples)
    
    print(f"\n平衡后总样本数: {len(balanced_samples)}")
    print(f"  - 创建新类: {len(new_samples)} ({len(new_samples)/len(balanced_samples):.2%})")
    print(f"  - 归拢: {len(merge_samples)} ({len(merge_samples)/len(balanced_samples):.2%})")
    print(f"  - 只分类: {len(select_only_samples)} ({len(select_only_samples)/len(balanced_samples):.2%})")
    
    # 保存
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in balanced_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"\n✓ 平衡后的数据已保存: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='平衡训练数据集')
    parser.add_argument('--input_dir', type=str, default="/mnt/literism/tree/summary_output/data/oracle_data_api/", help='输入数据目录（包含 classification_train.jsonl 和 summary_train.jsonl）')
    parser.add_argument('--output_dir', type=str, default=None, help='输出数据目录（默认为输入目录）')
    parser.add_argument('--summary_no_update_ratio', type=float, default=0.3, help='总结模型：不需要更新的样本的目标比例')
    parser.add_argument('--classify_new_ratio', type=float, default=0.2, help='分类模型：创建新类的目标比例')
    parser.add_argument('--classify_merge_ratio', type=float, default=0.1, help='分类模型：归拢的目标比例')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--skip_summary', action='store_true', help='跳过总结模型数据平衡')
    parser.add_argument('--skip_classification', action='store_true', help='跳过分类模型数据平衡')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    
    # 平衡总结模型数据
    if not args.skip_summary:
        summary_input = input_dir / 'summary_train.jsonl'
        summary_output = output_dir / 'summary_train_balanced.jsonl'
        if summary_input.exists():
            balance_summary_dataset(
                summary_input,
                summary_output,
                target_no_update_ratio=args.summary_no_update_ratio,
                seed=args.seed
            )
        else:
            print(f"警告: 找不到 {summary_input}")
    
    # 平衡分类模型数据
    if not args.skip_classification:
        classification_input = input_dir / 'classification_train.jsonl'
        classification_output = output_dir / 'classification_train_balanced.jsonl'
        if classification_input.exists():
            balance_classification_dataset(
                classification_input,
                classification_output,
                target_new_ratio=args.classify_new_ratio,
                target_merge_ratio=args.classify_merge_ratio,
                seed=args.seed
            )
        else:
            print(f"警告: 找不到 {classification_input}")
    
    print(f"\n{'='*60}")
    print("数据平衡完成！")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
