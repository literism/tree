"""
评估本地标注模型和API标注模型的效果对比

从训练好的轨迹数据中抽取100个标注样本，分别用本地模型和API模型进行标注，对比结果。
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
from tqdm import tqdm

from summary_based_classifier.core.trajectory.trajectory_storage import TrajectoryStorage, ArticleTrajectoryData
from summary_based_classifier.data.batch_labeler import LabelingRequest, BatchLabeler
from summary_based_classifier.llm.prompts import PromptTemplates
from summary_based_classifier.config import SummaryBasedConfig


@dataclass
class ComparisonResult:
    """对比结果"""
    sample_id: int
    prompt: str
    num_children: int
    local_result: Dict
    api_result: Dict
    agree_exceed: bool
    agree_overlap: bool
    agree_correct: bool
    agree_need_new: bool
    fully_agree: bool


def extract_labeling_samples(trajectory_file: str, num_samples: int = 100) -> List[Tuple[int, LabelingRequest]]:
    """
    从轨迹文件中提取标注样本
    
    Args:
        trajectory_file: 轨迹数据文件路径
        num_samples: 要抽取的样本数量
    
    Returns:
        List of (sample_id, LabelingRequest)
    """
    print(f"从 {trajectory_file} 加载轨迹数据...")
    storage = TrajectoryStorage()
    all_trajectories = storage.load(trajectory_file)
    
    # 收集所有标注请求
    all_requests = []
    sample_id = 0
    
    for traj_data in all_trajectories:
        for traj in traj_data.trajectories:
            for record in traj.re_classification_records:
                all_requests.append((sample_id, record))
                sample_id += 1
    
    print(f"总共找到 {len(all_requests)} 个标注样本")
    
    # 随机抽取
    if len(all_requests) > num_samples:
        sampled = random.sample(all_requests, num_samples)
        print(f"随机抽取 {num_samples} 个样本")
    else:
        sampled = all_requests
        print(f"样本总数不足 {num_samples}，使用全部 {len(sampled)} 个样本")
    
    return sampled


def compare_results(local_result: Dict, api_result: Dict) -> Dict[str, bool]:
    """
    对比两个标注结果
    
    Returns:
        {
            'agree_exceed': bool,
            'agree_overlap': bool,
            'agree_correct': bool,
            'agree_need_new': bool,
            'fully_agree': bool
        }
    """
    # 对比EXCEED_PARENT
    local_exceed = set(local_result.get('exceed_parent') or [])
    api_exceed = set(api_result.get('exceed_parent') or [])
    agree_exceed = local_exceed == api_exceed
    
    # 对比OVERLAPPING_PAIRS（需要转换为set of tuples）
    local_overlap = set(tuple(sorted(pair)) for pair in (local_result.get('overlapping_pairs') or []))
    api_overlap = set(tuple(sorted(pair)) for pair in (api_result.get('overlapping_pairs') or []))
    agree_overlap = local_overlap == api_overlap
    
    # 对比CORRECT_INDICES
    local_correct = set(local_result.get('correct_indices') or [])
    api_correct = set(api_result.get('correct_indices') or [])
    agree_correct = local_correct == api_correct
    
    # 对比NEED_NEW
    agree_need_new = local_result.get('need_new') == api_result.get('need_new')
    
    fully_agree = agree_exceed and agree_overlap and agree_correct and agree_need_new
    
    return {
        'agree_exceed': agree_exceed,
        'agree_overlap': agree_overlap,
        'agree_correct': agree_correct,
        'agree_need_new': agree_need_new,
        'fully_agree': fully_agree
    }


def evaluate_labeling_models(
    trajectory_file: str,
    config: SummaryBasedConfig,
    num_samples: int = 100,
    output_file: str = None
):
    """
    评估本地模型和API模型的标注效果
    
    Args:
        trajectory_file: 轨迹数据文件路径
        config: 配置对象
        num_samples: 抽取的样本数量
        output_file: 结果保存路径（JSON格式）
    """
    # 1. 提取样本
    samples = extract_labeling_samples(trajectory_file, num_samples)
    
    # 2. 使用本地模型标注
    print("\n" + "="*80)
    print("使用本地模型进行标注...")
    print("="*80)
    
    local_labeler = BatchLabeler(
        mode='local',
        local_model_path=config.labeling.local_model_path,
        tensor_parallel_size=config.labeling.tensor_parallel_size,
        gpu_memory_utilization=config.labeling.gpu_memory_utilization,
        max_model_len=config.labeling.max_model_len
    )
    
    local_results = local_labeler.label_batch([req for _, req in samples])
    local_labeler.cleanup()
    
    # 3. 使用API模型标注
    print("\n" + "="*80)
    print("使用API模型进行标注...")
    print("="*80)
    
    api_labeler = BatchLabeler(mode='api')
    api_results = api_labeler.label_batch([req for _, req in samples])
    api_labeler.cleanup()
    
    # 4. 对比结果
    print("\n" + "="*80)
    print("对比结果...")
    print("="*80)
    
    comparisons = []
    for (sample_id, req), local_res, api_res in zip(samples, local_results, api_results):
        if local_res.success and api_res.success:
            agreement = compare_results(local_res.parsed_output, api_res.parsed_output)
            
            comparison = ComparisonResult(
                sample_id=sample_id,
                prompt=req.prompt,
                num_children=len(req.child_summaries),
                local_result=local_res.parsed_output,
                api_result=api_res.parsed_output,
                agree_exceed=agreement['agree_exceed'],
                agree_overlap=agreement['agree_overlap'],
                agree_correct=agreement['agree_correct'],
                agree_need_new=agreement['agree_need_new'],
                fully_agree=agreement['fully_agree']
            )
            comparisons.append(comparison)
    
    # 5. 统计
    total = len(comparisons)
    if total == 0:
        print("没有成功对比的样本！")
        return
    
    fully_agree = sum(c.fully_agree for c in comparisons)
    agree_exceed = sum(c.agree_exceed for c in comparisons)
    agree_overlap = sum(c.agree_overlap for c in comparisons)
    agree_correct = sum(c.agree_correct for c in comparisons)
    agree_need_new = sum(c.agree_need_new for c in comparisons)
    
    print(f"\n成功对比样本数: {total}/{len(samples)}")
    print(f"完全一致率: {fully_agree}/{total} = {fully_agree/total*100:.2f}%")
    print(f"EXCEED_PARENT 一致率: {agree_exceed}/{total} = {agree_exceed/total*100:.2f}%")
    print(f"OVERLAPPING_PAIRS 一致率: {agree_overlap}/{total} = {agree_overlap/total*100:.2f}%")
    print(f"CORRECT_INDICES 一致率: {agree_correct}/{total} = {agree_correct/total*100:.2f}%")
    print(f"NEED_NEW 一致率: {agree_need_new}/{total} = {agree_need_new/total*100:.2f}%")
    
    # 6. 保存详细结果
    if output_file:
        output_data = {
            'summary': {
                'total_samples': total,
                'fully_agree': fully_agree,
                'fully_agree_rate': fully_agree / total,
                'agree_exceed': agree_exceed,
                'agree_exceed_rate': agree_exceed / total,
                'agree_overlap': agree_overlap,
                'agree_overlap_rate': agree_overlap / total,
                'agree_correct': agree_correct,
                'agree_correct_rate': agree_correct / total,
                'agree_need_new': agree_need_new,
                'agree_need_new_rate': agree_need_new / total
            },
            'comparisons': [
                {
                    'sample_id': c.sample_id,
                    'num_children': c.num_children,
                    'local_result': c.local_result,
                    'api_result': c.api_result,
                    'agree_exceed': c.agree_exceed,
                    'agree_overlap': c.agree_overlap,
                    'agree_correct': c.agree_correct,
                    'agree_need_new': c.agree_need_new,
                    'fully_agree': c.fully_agree,
                    'prompt': c.prompt[:200] + '...' if len(c.prompt) > 200 else c.prompt
                }
                for c in comparisons
            ]
        }
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n详细结果已保存到: {output_file}")
    
    # 7. 显示不一致的案例
    print("\n" + "="*80)
    print("不一致案例示例（前5个）:")
    print("="*80)
    
    disagreements = [c for c in comparisons if not c.fully_agree]
    for i, comp in enumerate(disagreements[:5], 1):
        print(f"\n案例 {i}:")
        print(f"  子类数量: {comp.num_children}")
        print(f"  本地结果: {comp.local_result}")
        print(f"  API结果:  {comp.api_result}")
        print(f"  EXCEED一致: {comp.agree_exceed}, OVERLAP一致: {comp.agree_overlap}, "
              f"CORRECT一致: {comp.agree_correct}, NEED_NEW一致: {comp.agree_need_new}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="评估本地标注模型和API标注模型的效果对比")
    parser.add_argument("--trajectory_file", type=str, required=True,
                        help="轨迹数据文件路径")
    parser.add_argument("--config", type=str, default="summary_based_classifier/configs/default.json",
                        help="配置文件路径")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="抽取的样本数量")
    parser.add_argument("--output", type=str, default="labeling_evaluation.json",
                        help="结果保存路径")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 加载配置
    config = SummaryBasedConfig.from_json(args.config)
    
    # 运行评估
    evaluate_labeling_models(
        trajectory_file=args.trajectory_file,
        config=config,
        num_samples=args.num_samples,
        output_file=args.output
    )


if __name__ == "__main__":
    main()
