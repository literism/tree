"""
临时脚本：将分类数据集的 completion 从“三段文本格式”转换为 JSON 字符串格式。

目标completion格式：
{"selected_indices":[...],"need_new":true/false,"merge_with":int/null}
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional


def _extract_answer_part(line_text: str) -> str:
    markers = [
        "Final answer:",
        "Final Answer:",
        "FINAL ANSWER:",
        "Answer:",
        "ANSWER:",
        "Conclusion:",
        "CONCLUSION:",
        "最终答案：",
        "最终答案:",
        "结论：",
        "结论:",
    ]
    for marker in markers:
        if marker in line_text:
            return line_text.split(marker, 1)[1].strip()
    if ":" in line_text:
        return line_text.split(":", 1)[1].strip()
    if "：" in line_text:
        return line_text.split("：", 1)[1].strip()
    return line_text.strip()


def _line_type(line_text: str) -> Optional[str]:
    norm = re.sub(r"^\s*(line\s*\d+\s*[-:]\s*|\d+\s*[\)\.\-]\s*|[-*]\s*)", "", line_text, flags=re.IGNORECASE).strip()
    upper = norm.upper()
    if upper.startswith("SELECTED"):
        return "selected"
    if upper.startswith("NEW") or upper.startswith("NEED_NEW") or upper.startswith("NEW_CATEGORY"):
        return "new"
    if upper.startswith("MERGE_WITH") or upper.startswith("MERGE"):
        return "merge"
    return None


def parse_completion_to_struct(completion: str) -> Optional[Dict]:
    clean = completion.strip().replace("```json", "").replace("```", "").strip()
    # 兼容 jsonl 中把换行写成字面量 "\\n" 的情况
    clean = clean.replace("\\r\\n", "\n").replace("\\n", "\n")

    # 已是JSON则直接解析
    try:
        parsed = json.loads(clean)
        if isinstance(parsed, dict) and {"selected_indices", "need_new", "merge_with"}.issubset(parsed.keys()):
            selected_indices = [int(x) for x in parsed.get("selected_indices", []) if str(x).isdigit()]
            need_new = bool(parsed.get("need_new"))
            merge_with = parsed.get("merge_with")
            if merge_with is not None:
                merge_with = int(merge_with)
            return {
                "selected_indices": sorted(set(selected_indices)),
                "need_new": need_new,
                "merge_with": merge_with,
            }
    except Exception:
        pass

    lines = [ln.strip() for ln in clean.splitlines() if ln.strip()]
    sections: Dict[str, List[str]] = {"selected": [], "new": [], "merge": []}
    current = None

    for line in lines:
        kind = _line_type(line)
        if kind is not None:
            current = kind
            sections[kind].append(_extract_answer_part(line))
        elif current is not None:
            sections[current].append(_extract_answer_part(line))

    if not any(sections.values()) and len(lines) >= 3:
        sections["selected"] = [_extract_answer_part(lines[0])]
        sections["new"] = [_extract_answer_part(lines[1])]
        sections["merge"] = [_extract_answer_part(lines[2])]

    if not sections["selected"] or not sections["new"]:
        return None

    selected_text = " ".join(sections["selected"]).strip()
    selected_upper = selected_text.upper()
    if selected_upper in {"NONE", "N/A", "NULL", "[]", "NO", ""}:
        selected_indices: List[int] = []
    else:
        selected_indices = sorted(set(int(x) for x in re.findall(r"\d+", selected_text)))

    new_text = " ".join(sections["new"]).strip().lower()
    new_text = re.sub(r"[。.!;；\s]+$", "", new_text)
    if new_text in {"yes", "y", "true", "1", "是", "需要", "要"}:
        need_new = True
    elif new_text in {"no", "n", "false", "0", "否", "不需要", "不用"}:
        need_new = False
    else:
        return None

    merge_with = None
    if sections["merge"]:
        merge_text = " ".join(sections["merge"]).strip()
        merge_upper = merge_text.upper()
        if merge_upper not in {"NONE", "N/A", "NULL", "[]", "NO", ""}:
            nums = re.findall(r"\d+", merge_text)
            if nums:
                merge_with = int(nums[0])

    return {
        "selected_indices": selected_indices,
        "need_new": need_new,
        "merge_with": merge_with,
    }


def convert_file(input_file: Path, output_file: Path):
    if not input_file.exists():
        print(f"跳过（不存在）: {input_file}")
        return

    ok = 0
    err = 0
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with input_file.open("r", encoding="utf-8") as fin, output_file.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, 1):
            try:
                sample = json.loads(line)
                parsed = parse_completion_to_struct(sample.get("completion", ""))
                if parsed is None:
                    raise ValueError("completion 解析失败")
                sample["completion"] = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                ok += 1
            except Exception as e:
                err += 1
                print(f"[错误] {input_file.name}:{line_no} -> {e}")

    print(f"{input_file.name}: 成功 {ok}，失败 {err} -> {output_file}")


def main():
    parser = argparse.ArgumentParser(description="将分类数据completion转换为JSON格式")
    parser.add_argument("--input_dir", type=str, default="/mnt/literism/tree/summary_output/data/oracle_data_model")
    parser.add_argument("--output_dir", type=str, default="/mnt/literism/tree/summary_output/data/oracle_data_model/converted", help="默认输出到 input_dir/json_completion")
    parser.add_argument(
        "--files",
        nargs="+",
        default=[
            "classification_train.jsonl",
            "classification_val.jsonl",
            "classification_train_split.jsonl",
            "classification_train_balanced.jsonl",
        ],
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "json_completion"

    for name in args.files:
        convert_file(input_dir / name, output_dir / name)

    print(f"完成，输出目录: {output_dir}")


if __name__ == "__main__":
    main()
