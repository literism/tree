"""
Oracle π* + BOW summary 的 SFT 训练入口（新流程，不替换旧采样/IWSFT/DPO）

流程：
1) 用 `summary_based_classifier.data.prepare_dataset_oracle` 生成分类系统 SFT 数据
2) 用 `summary_based_classifier.training.train_classify_generator` 做 SFT 训练
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from summary_based_classifier.config import SummaryBasedConfig


def main():
    p = argparse.ArgumentParser(description="Oracle π* 生成数据并 SFT 训练（分类系统）")
    p.add_argument("--config", type=str, default="./configs/default.json")
    p.add_argument("--base_model", type=str, default="/home/literism/model/Qwen3-32B")
    p.add_argument("--output_dir", type=str, default="/mnt/literism/tree/summary_output/")
    p.add_argument("--split", type=str, default="train", choices=["train"])
    p.add_argument("--bow_top_k", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_refs_per_topic", type=int, default=None)
    p.add_argument("--val_ratio", type=float, default=0.02)
    args = p.parse_args()

    # 确保从项目根目录运行子进程（不管当前 cwd 是什么）
    script_dir = Path(__file__).resolve().parent  # cli/
    project_root = script_dir.parent.parent  # tree/
    
    # 转换 config 路径为绝对路径（相对于当前工作目录）
    config_path = Path(args.config).resolve()
    
    config = SummaryBasedConfig.from_json(str(config_path))
    data_dir = Path(config.path.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    train_file = data_dir / "classify_generator_oracle_train.jsonl"
    val_file = data_dir / "classify_generator_oracle_val.jsonl"

    # 1) 生成 oracle 数据
    print("\n" + "=" * 80)
    print("Step 1: 生成 oracle SFT 数据（分类系统）")
    print("=" * 80)
    print(f"Working dir: {project_root}")
    
    cmd1 = [
        sys.executable,
        "-m",
        "summary_based_classifier.data.prepare_dataset_oracle",
        "--config",
        str(config_path),
        "--split",
        args.split,
        "--bow_top_k",
        str(args.bow_top_k),
        "--seed",
        str(args.seed),
        "--val_ratio",
        str(args.val_ratio),
    ]
    if args.max_refs_per_topic is not None:
        cmd1.extend(["--max_refs_per_topic", str(args.max_refs_per_topic)])
    
    print("CMD:", " ".join(cmd1))
    result = subprocess.run(cmd1, cwd=str(project_root))
    if result.returncode != 0:
        raise SystemExit(result.returncode)

    if not train_file.exists() or not val_file.exists():
        raise SystemExit(f"oracle 数据文件不存在：{train_file} / {val_file}")

    # 2) SFT 训练分类生成系统
    print("\n" + "=" * 80)
    print("Step 2: SFT 训练 classify_generator（oracle 数据）")
    print("=" * 80)
    
    cmd2 = [
        sys.executable,
        "-m",
        "summary_based_classifier.training.train_classify_generator",
        "--base_model",
        args.base_model,
        "--train_data",
        str(train_file),
        "--val_data",
        str(val_file),
        "--output_dir",
        args.output_dir,
        "--config",
        str(config_path),
    ]
    
    print("CMD:", " ".join(cmd2))
    result = subprocess.run(cmd2, cwd=str(project_root))
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

