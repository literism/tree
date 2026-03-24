"""
准备训练数据（新版本）
构造分类生成系统和总结更新系统的训练数据
"""
import json
import random
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from summary_based_classifier.config import SummaryBasedConfig
from summary_based_classifier.llm.prompts import PromptTemplates
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# 添加modeling目录到sys.path以导入DeepSeekAPIClient
sys.path.append(str(Path(__file__).parent.parent))
from modeling.deepseek_api import DeepSeekAPIClient, DeepSeekConfig


class TrainingDataPreparator:
    def __init__(self, config: SummaryBasedConfig):
        """
        Args:
            config: 配置对象
        """
        self.config = config
        self.output_dir = Path(config.path.data_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        random.seed(config.data_prepare.seed)
        
        self.structures_data = None
        self.references_data = None
        self.summaries_data = None
        self.dataset_split = None
    
    def load_data(self):
        """加载所有需要的数据"""
        print("加载数据...")
        
        # 加载structures
        print(f"  - 加载structures: {self.config.path.structures_file}")
        with open(self.config.path.structures_file, 'r', encoding='utf-8') as f:
            self.structures_data = json.load(f)
        
        # 加载references
        print(f"  - 加载references: {self.config.path.references_file}")
        with open(self.config.path.references_file, 'r', encoding='utf-8') as f:
            self.references_data = json.load(f)
        
        # 加载summaries
        summaries_file = Path(self.config.path.summaries_dir) / 'node_summaries.json'
        print(f"  - 加载summaries: {summaries_file}")
        with open(summaries_file, 'r', encoding='utf-8') as f:
            self.summaries_data = json.load(f)
        
        # 加载dataset split
        split_file = self.output_dir / 'dataset_split.json'
        print(f"  - 加载dataset split: {split_file}")
        with open(split_file, 'r', encoding='utf-8') as f:
            split_data = json.load(f)
            self.dataset_split = split_data['dataset_split']
        
        print("数据加载完成")
    
    def _get_all_children_from_structure(self, structure_dict: Dict, current_path: str) -> List[str]:
        """
        从structure中获取当前路径的所有子节点titles
        
        Args:
            structure_dict: topic的structure字典
            current_path: 当前路径，如 "Topic - A - a1"
            
        Returns:
            所有子节点的titles列表
        """
        structure = structure_dict.get('structure', [])
        topic_name = structure_dict.get('topic', '')
        
        # 根节点
        if current_path == topic_name:
            return [node['title'] for node in structure]
        
        # 解析路径层次
        path_parts = [p.strip() for p in current_path.split(' - ')]
        
        # 从根开始查找
        def find_node(nodes, parts, start_idx=1):
            if start_idx >= len(parts):
                return []
            
            target_title = parts[start_idx]
            for node in nodes:
                if node['title'] == target_title:
                    if start_idx == len(parts) - 1:
                        # 找到目标节点，返回其子节点
                        return [child['title'] for child in node.get('children', [])]
                    else:
                        # 继续向下查找
                        return find_node(node.get('children', []), parts, start_idx + 1)
            
            return []
        
        return find_node(structure, path_parts)
    
    def parse_paths_to_classifications(self, paths: List[str]) -> List[Tuple[str, List[str]]]:
        """
        将路径列表解析为分类步骤列表
        
        Args:
            paths: 路径列表，如 ["T - A - a1", "T - A - a2", "T - B - b1"]
            
        Returns:
            分类步骤列表 [(current_path, output_titles), ...]
            例如: [("T", ["A", "B"]), ("T - A", ["a1", "a2"]), ("T - B", ["b1"])]
        """
        classifications = defaultdict(set)
        
        for path in paths:
            parts = [p.strip() for p in path.split(' - ')]
            
            # 对每一层都记录分类
            for i in range(1, len(parts)):
                # 当前路径
                current_path = ' - '.join(parts[:i])
                # 输出标题
                output_title = parts[i]
                classifications[current_path].add(output_title)
        
        # 转换为列表
        result = [(path, sorted(list(titles))) for path, titles in classifications.items()]
        # 按照路径深度排序
        result.sort(key=lambda x: x[0].count(' - '))
        
        return result
    
    def collect_classify_generate_samples(self) -> List[Dict]:
        """
        从训练集中收集分类生成样本
        
        Returns:
            classify_generate_samples
        """
        print("\n收集分类生成训练样本...")
        
        samples = []
        train_topics = self.dataset_split.get('train', {})
        
        # 统计数据不一致情况
        missing_summaries_count = 0
        invalid_root_samples = 0  # 无效的根节点样本（无children无correct_indices）
        
        for topic_key in tqdm(train_topics.keys(), desc="处理topics"):
            if topic_key not in self.structures_data:
                continue
            if topic_key not in self.references_data:
                continue
            if topic_key not in self.summaries_data:
                continue
            
            structure = self.structures_data[topic_key]
            topic_name = structure.get('topic', topic_key)
            references = self.references_data[topic_key].get('references', {})
            summaries = self.summaries_data[topic_key]
            
            # 为每篇文章构建样本
            for ref_id, ref_data in references.items():
                content = ref_data.get('content', '')
                paths = ref_data.get('paths', [])
                
                if not content or not paths:
                    continue
                
                # 解析paths为分类步骤
                classifications = self.parse_paths_to_classifications(paths)
                
                # 添加叶子节点（用于Type3数据）
                leaf_paths = set()
                for path in paths:
                    is_leaf = True
                    for other_path in paths:
                        if other_path != path and other_path.startswith(path + ' - '):
                            is_leaf = False
                            break
                    if is_leaf:
                        leaf_paths.add(path)
                
                # 将叶子节点添加到classifications中（output_titles为空）
                for leaf_path in leaf_paths:
                    classifications.append((leaf_path, []))
                
                # 为每个分类步骤创建样本
                for current_path, output_titles in classifications:
                    # 计算depth
                    depth = current_path.count(' - ')
                    
                    # 获取当前节点的summary
                    current_summary = ""
                    if depth == 0:
                        # 根节点使用topic_name
                        current_summary = topic_name
                    else:
                        # 非根节点必须从summaries中获取
                        if current_path not in summaries:
                            # 当前节点不在summaries中（可能是没有content的节点），跳过
                            continue
                        
                        current_summary_data = summaries[current_path]
                        if isinstance(current_summary_data, dict):
                            # 构建格式化的summary（必须有explanation和scope）
                            explanation = current_summary_data.get('explanation', '')
                            scope = current_summary_data.get('scope', '')
                            if explanation and scope:
                                current_summary = f"EXPLANATION: {explanation}\nSCOPE: {scope}"
                            else:
                                # 非根节点但没有完整的summary数据，跳过
                                continue
                        else:
                            continue
                    
                    # 获取所有子节点
                    all_children_titles = self._get_all_children_from_structure(structure, current_path)
                    
                    # 诊断：检查根节点没有children的情况
                    if depth == 0 and len(all_children_titles) == 0 and len(output_titles) > 0:
                        print(f"      ⚠️ 诊断: 根节点无children但有output_titles")
                        print(f"         topic={topic_key}, current_path={current_path}")
                        print(f"         output_titles={output_titles}")
                        print(f"         这可能表示structure和references数据不一致")
                    
                    # 获取所有子节点的summaries（字符串格式，用于输入）
                    all_child_summaries = []
                    # 同时保存原始的summary数据（字典格式，用于提取explanation和scope）
                    all_child_summary_dicts = []
                    # 保存实际有summary的child titles（用于计算correct_indices）
                    valid_child_titles = []
                    has_empty_child = False
                    
                    for child_title in all_children_titles:
                        child_path = current_path + ' - ' + child_title
                        
                        # 检查child_path是否在summaries中
                        if child_path not in summaries:
                            # 数据不一致：structure中有这个节点，但summaries中没有
                            # 跳过这个子节点
                            missing_summaries_count += 1
                            continue
                        
                        child_summary_data = summaries[child_path]
                        
                        if isinstance(child_summary_data, dict):
                            # 构建格式化的child summary（必须有explanation和scope）
                            explanation = child_summary_data.get('explanation', '')
                            scope = child_summary_data.get('scope', '')
                            if explanation and scope:
                                child_summary = f"EXPLANATION: {explanation}\nSCOPE: {scope}"
                                all_child_summary_dicts.append(child_summary_data)
                            else:
                                # 子节点缺少explanation或scope，跳过
                                missing_summaries_count += 1
                                continue
                        else:
                            # 数据格式错误，跳过
                            missing_summaries_count += 1
                            continue
                        
                        # 检查summary是否为空（理论上不应该为空了）
                        if not child_summary or not child_summary.strip():
                            print(f"      ⚠️ 警告: 子节点summary为空 - topic={topic_key}, path={child_path}")
                            has_empty_child = True
                            break
                        
                        all_child_summaries.append(child_summary)
                        valid_child_titles.append(child_title)
                        
                    # 如果有空的child summary，跳过这个样本
                    if has_empty_child:
                        continue
                    
                    # 找到正确的子节点索引（基于valid_child_titles而不是all_children_titles）
                    correct_indices = []
                    for title in output_titles:
                        if title in valid_child_titles:
                            idx = valid_child_titles.index(title)
                            correct_indices.append(idx)
                        
                    # 获取正确子节点的summary数据（字典格式，包含explanation和scope）
                    correct_summary_dicts = []
                    for idx in correct_indices:
                        if 0 <= idx < len(all_child_summary_dicts):
                            correct_summary_dicts.append(all_child_summary_dicts[idx])
                    
                    # 过滤无效样本：根节点但既没有children也没有correct_indices
                    if depth == 0 and len(all_child_summaries) == 0 and len(correct_indices) == 0:
                        # 这种样本既不能分类也不能生成新类，跳过
                        invalid_root_samples += 1
                        continue
                    
                    sample = {
                        'topic_key': topic_key,
                        'topic_name': topic_name,
                        'ref_id': ref_id,
                        'content': content,
                        'current_path': current_path,
                        'current_summary': current_summary,
                        'all_children_titles': valid_child_titles,  # 使用实际有summary的children
                        'all_child_summaries': all_child_summaries,
                        'correct_indices': correct_indices,
                        'correct_summary_dicts': correct_summary_dicts,  # 字典格式，包含explanation和scope
                        'depth': depth
                    }
                    samples.append(sample)
        
        print(f"  - 收集到 {len(samples)} 个原始样本")
        
        if missing_summaries_count > 0:
            print(f"  - ⚠️ 发现 {missing_summaries_count} 个子节点在structure中但不在summaries中（已跳过）")
        
        if invalid_root_samples > 0:
            print(f"  - ⚠️ 过滤掉 {invalid_root_samples} 个无效根节点样本（无children且无correct_indices）")
        
        # 详细统计
        depth_dist = defaultdict(int)
        child_count_dist = defaultdict(int)
        correct_count_dist = defaultdict(int)
        
        for sample in samples:
            depth_dist[sample['depth']] += 1
            child_count = len(sample['all_child_summaries'])
            child_count_dist[child_count] += 1
            correct_count = len(sample['correct_indices'])
            correct_count_dist[correct_count] += 1
        
        print(f"\n  - 原始样本depth分布:")
        for depth in sorted(depth_dist.keys()):
            count = depth_dist[depth]
            percentage = count / len(samples) * 100 if len(samples) > 0 else 0
            print(f"      depth={depth}: {count} ({percentage:.1f}%)")
        
        print(f"\n  - 原始样本child_summaries数量分布:")
        for child_count in sorted(child_count_dist.keys())[:10]:  # 只显示前10个
            count = child_count_dist[child_count]
            percentage = count / len(samples) * 100 if len(samples) > 0 else 0
            print(f"      {child_count}个子节点: {count} ({percentage:.1f}%)")
        if len(child_count_dist) > 10:
            print(f"      ... 还有 {len(child_count_dist) - 10} 个其他数量")
        
        print(f"\n  - 原始样本correct_indices数量分布:")
        for correct_count in sorted(correct_count_dist.keys())[:10]:
            count = correct_count_dist[correct_count]
            percentage = count / len(samples) * 100 if len(samples) > 0 else 0
            print(f"      {correct_count}个正确分类: {count} ({percentage:.1f}%)")
        if len(correct_count_dist) > 10:
            print(f"      ... 还有 {len(correct_count_dist) - 10} 个其他数量")
        
        return samples
    
    def _sample_list(self, items: List, target_count: int) -> List:
        """从列表中采样指定数量"""
        if len(items) <= target_count:
            return items.copy()
        return random.sample(items, target_count)
    
    def _sample_by_layer(
        self,
        samples: List[Dict],
        target_count: int,
        sample_type: str = None
    ) -> List[Dict]:
        """
        按层级采样数据
        
        Args:
            samples: 样本列表（每个样本必须有depth字段）
            target_count: 目标样本数量
            sample_type: 样本类型名称（用于日志）
            
        Returns:
            采样后的列表
        """
        if not samples:
            if sample_type:
                print(f"      警告: {sample_type}没有候选样本！")
            return []
        
        # 分层
        layer1_samples = [s for s in samples if s.get('depth', 0) == 0]
        layer2_samples = [s for s in samples if s.get('depth', 0) > 0]
        
        # 计算各层应采样的数量
        layer1_ratio = self.config.data_prepare.layer1_ratio
        layer1_count = int(target_count * layer1_ratio)
        layer2_count = target_count - layer1_count
        
        # 采样
        result = []
        sampled_layer1 = self._sample_list(layer1_samples, layer1_count)
        sampled_layer2 = self._sample_list(layer2_samples, layer2_count)
        result.extend(sampled_layer1)
        result.extend(sampled_layer2)
        
        # 检查是否满足目标
        if len(result) < target_count and sample_type:
            shortage = target_count - len(result)
            print(f"      警告: {sample_type}数据不足，目标{target_count}，实际{len(result)}，短缺{shortage}")
        
        # 打乱
        random.shuffle(result)
        
        return result
    
    def construct_classify_generator_dataset(
        self,
        raw_samples: List[Dict],
        debug_mode: bool = False,
        debug_output_file: str = None
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        构造分类生成系统的训练数据
        
        使用API补全输出：已知brother_node和new_categories，让API补充spans等其他字段
        
        Args:
            raw_samples: 原始样本列表
            debug_mode: 调试模式，只生成prompt不调用API
            debug_output_file: 调试模式下保存prompt和基础信息的文件路径
            
        Returns:
            (train_samples, val_samples) 或 调试模式下返回空
        """
        print("\n构造分类生成系统训练数据...")
        print(f"  - 层级采样比例: Layer1={self.config.data_prepare.layer1_ratio:.0%}, Layer2+={1-self.config.data_prepare.layer1_ratio:.0%}")
        
        ratios = self.config.data_prepare.classify_generator_ratios
        total = self.config.data_prepare.classify_generator_total_samples
        delete_multiple_ratio = self.config.data_prepare.classify_generator_delete_multiple_ratio
        
        # 计算每类应有的样本数
        ratio_sum = sum(ratios)
        counts = [int(total * r / ratio_sum) for r in ratios]
        
        print(f"  - Type1: {counts[0]} 样本 (删除多个类别比例: {delete_multiple_ratio:.0%})")
        print(f"  - Type2: {counts[1]} 样本")
        print(f"  - Type3: {counts[2]} 样本")
        print(f"  - Type4: {counts[3]} 样本")
        
        # 收集四种类型的候选样本（输入部分）
        type1_candidates = []
        type2_candidates = []
        type3_candidates = []
        type4_candidates = []
        
        # 统计跳过的样本
        skipped_samples = {
            'no_correct_no_children': 0,  # 既没有correct也没有children
            'has_correct_no_children_depth0': 0,  # 有correct但没有children且是根节点（应该进Type4）
        }
        
        for sample in raw_samples:
            depth = sample.get('depth', 0)
            all_child_summaries = sample['all_child_summaries']
            correct_indices = sample['correct_indices']
            correct_summary_dicts = sample['correct_summary_dicts']  # 使用字典格式的summary数据
            
            # Type2: 保留所有类别，分类到正确类别，不生成新类
            if len(correct_indices) > 0 and len(all_child_summaries) > 0:
                type2_candidates.append({
                    'sample': sample,
                    'brother_node': correct_indices,
                    'new_categories': [],
                    'depth': depth
                })
            
            # Type1: 删除部分正确类别
            if len(correct_indices) > 0 and len(all_child_summaries) > 1:
                # 决定删除几个（1个或多个）
                max_delete = min(len(correct_indices), len(all_child_summaries) - 1)
                if max_delete > 0:
                    # 大部分情况删除1个，小部分情况删除多个
                    if random.random() < delete_multiple_ratio and max_delete > 1:
                        num_to_delete = random.randint(2, max_delete)
                    else:
                        num_to_delete = 1
                    
                    # 随机选择要删除的索引
                    to_delete = set(random.sample(correct_indices, num_to_delete))
                    remaining_indices = [idx for idx in correct_indices if idx not in to_delete]
                    
                    # 提取被删除类别的summaries作为new_categories
                    # 从correct_summary_dicts中获取对应的字典数据
                    new_categories = []
                    for i in sorted(to_delete):
                        idx_in_correct = correct_indices.index(i)
                        if idx_in_correct < len(correct_summary_dicts):
                            summary_dict = correct_summary_dicts[idx_in_correct]
                            if isinstance(summary_dict, dict) and 'explanation' in summary_dict and 'scope' in summary_dict:
                                new_categories.append({
                                    'explanation': summary_dict['explanation'],
                                    'scope': summary_dict['scope']
                                })
                
                # 删除这些类别后的summaries
                    filtered_summaries = [s for i, s in enumerate(all_child_summaries) if i not in to_delete]
                
                    # 重新映射remaining_indices
                remaining_mapped = []
                for old_idx in remaining_indices:
                    new_idx = old_idx - sum(1 for d in to_delete if d < old_idx)
                    remaining_mapped.append(new_idx)
                
                    type1_candidates.append({
                        'sample': sample,
                        'filtered_summaries': filtered_summaries,
                        'brother_node': remaining_mapped,
                        'new_categories': new_categories,
                        'depth': depth
                    })
            
            # Type3: 叶子节点，无子节点（但不是根节点）
            if len(all_child_summaries) == 0 and depth > 0:
                type3_candidates.append({
                    'sample': sample,
                    'brother_node': [],
                    'new_categories': [],
                    'depth': depth,
                    'is_type3': True  # 标记Type3，确保使用空候选列表
                })
            
            # Type4: 空类别，需要新建（模拟首次创建子类的情况）
            # 与Type2使用相同的样本池，但在构建prompt时使用空的child_summaries
            if len(all_child_summaries) > 0 and len(correct_indices) > 0:
                # 模拟首次创建子类的情况
                new_categories = []
                for summary_dict in correct_summary_dicts:
                    if isinstance(summary_dict, dict) and 'explanation' in summary_dict and 'scope' in summary_dict:
                        new_categories.append({
                            'explanation': summary_dict['explanation'],
                            'scope': summary_dict['scope']
                        })
                
                type4_candidates.append({
                    'sample': sample,
                    'brother_node': [],
                    'new_categories': new_categories[:self.config.data_prepare.max_new_categories],
                    'depth': depth,
                    'is_type4': True  # 标记Type4，用于在构建prompt时使用空候选列表
                })
            
            # 统计不满足任何类型的样本
            if (len(correct_indices) == 0 and len(all_child_summaries) == 0):
                skipped_samples['no_correct_no_children'] += 1
            elif (len(correct_indices) > 0 and len(all_child_summaries) == 0 and depth == 0):
                skipped_samples['has_correct_no_children_depth0'] += 1
        
        print(f"\n  - 收集到的候选样本统计:")
        print(f"    Type1候选: {len(type1_candidates)}")
        print(f"    Type2候选: {len(type2_candidates)}")
        print(f"    Type3候选: {len(type3_candidates)}")
        print(f"    Type4候选: {len(type4_candidates)}")
        print(f"    总候选数: {len(type1_candidates) + len(type2_candidates) + len(type3_candidates) + len(type4_candidates)}")
        
        print(f"\n  - 跳过的样本统计:")
        print(f"    无correct无children: {skipped_samples['no_correct_no_children']}")
        print(f"    有correct无children且depth=0: {skipped_samples['has_correct_no_children_depth0']}")
        print(f"    总跳过: {sum(skipped_samples.values())}")
        
        # 统计各类型的depth分布
        print(f"\n  - 各类型的depth分布:")
        for type_name, candidates in [('Type1', type1_candidates), ('Type2', type2_candidates), 
                                      ('Type3', type3_candidates), ('Type4', type4_candidates)]:
            if candidates:
                type_depth_dist = defaultdict(int)
                for cand in candidates:
                    type_depth_dist[cand['depth']] += 1
                print(f"    {type_name}:")
                for depth in sorted(type_depth_dist.keys())[:5]:
                    count = type_depth_dist[depth]
                    percentage = count / len(candidates) * 100
                    print(f"      depth={depth}: {count} ({percentage:.1f}%)")
                if len(type_depth_dist) > 5:
                    print(f"      ... 还有 {len(type_depth_dist) - 5} 个其他深度")
        
        # 按层级采样
        print(f"\n  - 开始按层级采样:")
        type1_sampled = self._sample_by_layer(type1_candidates, counts[0], "Type1")
        type2_sampled = self._sample_by_layer(type2_candidates, counts[1], "Type2")
        type3_sampled = self._sample_list(type3_candidates, counts[2])  # Type3不按层级
        type4_sampled = self._sample_by_layer(type4_candidates, counts[3], "Type4")
        
        # 合并所有采样的候选
        all_sampled = type1_sampled + type2_sampled + type3_sampled + type4_sampled
        random.shuffle(all_sampled)
        
        # 调试模式：只生成prompt，不调用API
        if debug_mode:
            print(f"\n  - 调试模式：生成prompt但不调用API")
            print(f"    总共生成: {len(all_sampled)} 个样本")
            
            debug_samples = []
            for cand in all_sampled:
                sample = cand['sample']
                brother_node = cand['brother_node']
                new_categories = cand['new_categories']
                
                # 确定使用哪个child_summaries
                if 'filtered_summaries' in cand:
                    child_summaries = cand['filtered_summaries']
                elif cand.get('is_type3', False) or cand.get('is_type4', False):
                    child_summaries = []
                else:
                    child_summaries = sample['all_child_summaries']
                
                # 构建prompt
                prompt = PromptTemplates.format_classify_generator_prompt(
                    topic_name=sample['topic_name'],
                    current_summary=sample['current_summary'],
                    article_content=sample['content'],
                    child_summaries=child_summaries
                )
                
                # 确定类型
                sample_type = None
                if 'filtered_summaries' in cand:
                    sample_type = 'Type1'
                elif cand.get('is_type3', False):
                    sample_type = 'Type3'
                elif cand.get('is_type4', False):
                    sample_type = 'Type4'
                else:
                    sample_type = 'Type2'
                
                debug_samples.append({
                    'type': sample_type,
                    'depth': cand['depth'],
                    'topic_key': sample['topic_key'],
                    'topic_name': sample['topic_name'],
                    'ref_id': sample['ref_id'],
                    'current_path': sample['current_path'],
                    'current_summary': sample['current_summary'],
                    'num_child_summaries': len(child_summaries),
                    'child_summaries': child_summaries,
                    'brother_node': brother_node,
                    'new_categories': new_categories,
                    'prompt': prompt,
                    'prompt_length': len(prompt)
                })
            
            # 保存到文件
            if debug_output_file:
                with open(debug_output_file, 'w', encoding='utf-8') as f:
                    for debug_sample in debug_samples:
                        f.write(json.dumps(debug_sample, ensure_ascii=False) + '\n')
                print(f"    调试信息已保存到: {debug_output_file}")
            
            return [], []  # 调试模式不返回训练数据
        
        print(f"\n  - 构建训练样本（不再需要API调用）...")
        print(f"    总共构建: {len(all_sampled)} 个样本")
        
        # 直接构建训练样本
        completed_samples = []
        for cand in all_sampled:
            sample = cand['sample']
            brother_node = cand['brother_node']
            new_categories = cand['new_categories']
            
            # 确定使用哪个child_summaries
            # Type1: 使用filtered版本（删除了部分类别）
            # Type3: 使用空列表（叶子节点，无子节点）
            # Type4: 使用空列表（模拟首次创建子类）
            # Type2: 使用完整列表
            if 'filtered_summaries' in cand:
                child_summaries = cand['filtered_summaries']
            elif cand.get('is_type3', False) or cand.get('is_type4', False):
                child_summaries = []  # Type3和Type4都使用空候选列表
            else:
                child_summaries = sample['all_child_summaries']
            
            # 构建prompt
            prompt = PromptTemplates.format_classification_prompt(
                        topic_name=sample['topic_name'],
                        current_summary=sample['current_summary'],
                        article_content=sample['content'],
                child_summaries=child_summaries
            )
            
            # 直接构建completion（多行Yes/No格式）
            # 不再需要API调用，直接根据known answer构建
            completion_lines = []
            
            # 对每个候选类别，判断是否被选中
            for idx in range(len(child_summaries)):
                if idx in brother_node:
                    completion_lines.append(f"Category {idx}: Yes")
                else:
                    completion_lines.append(f"Category {idx}: No")
            
            # NEW类别（只输出Yes/No，不输出新类的summary）
            if new_categories:
                completion_lines.append("NEW: Yes")
            else:
                completion_lines.append("NEW: No")
            
            completion = "\n".join(completion_lines)
            
            # 直接添加到completed_samples，不需要API调用
            completed_samples.append({
                'prompt': prompt,
                'completion': completion
            })
        
        print(f"\n  - 构建完成:")
        print(f"    成功: {len(completed_samples)} 样本")
        
        # 划分训练集和验证集
        random.shuffle(completed_samples)
        val_size = int(len(completed_samples) * 0.1)
        val_samples = completed_samples[:val_size]
        train_samples = completed_samples[val_size:]
        
        print(f"\n  - 最终生成:")
        print(f"    训练集: {len(train_samples)} 样本")
        print(f"    验证集: {len(val_samples)} 样本")
        
        # 最终数据集质量检查
        print(f"\n  - 最终数据集质量检查:")
        for dataset_name, dataset in [('训练集', train_samples), ('验证集', val_samples)]:
            empty_child_count = 0
            for sample in dataset:
                prompt = sample['prompt']
                # 检查prompt中是否有空的child summary
                if '[0] \n[1]' in prompt or '[0]\n[1]' in prompt:
                    empty_child_count += 1
            if empty_child_count > 0:
                print(f"    ⚠️ {dataset_name}中有 {empty_child_count} 个样本包含空的child summary")
            else:
                print(f"    ✅ {dataset_name}通过检查")
        
        return train_samples, val_samples
    
    def _perturb_summary(self, summary_text: str, max_sentence_tokens: int = 5) -> Optional[str]:
        """
        扰动summary：从explanation或scope中随机删除一句短句
        
        Args:
            summary_text: summary文本（格式：EXPLANATION: ... SCOPE: ...）
            max_sentence_tokens: 短句最大token数
            
        Returns:
            扰动后的summary，如果无法扰动则返回None
        """
        import re
        
        # 解析summary
        if 'EXPLANATION:' not in summary_text or 'SCOPE:' not in summary_text:
            return None
        
        parts = summary_text.split('SCOPE:')
        explanation_part = parts[0].replace('EXPLANATION:', '').strip()
        scope_part = parts[1].strip() if len(parts) > 1 else ""
        
        # 随机选择一个字段进行扰动
        field_name = random.choice(['explanation', 'scope'])
        field_text = explanation_part if field_name == 'explanation' else scope_part
        
        if not field_text:
            return None
        
        # 获取完整的第一句（第一个句号之前）
        first_period_idx = field_text.find('.')
        if first_period_idx == -1:
            first_sentence = field_text
        else:
            first_sentence = field_text[:first_period_idx+1]
        
        # 将字段分割成短句（用逗号和句号分隔）
        sentences = re.split(r'[,.]', field_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # 过滤：找到符合条件的短句
        # 1. 不是第一句的一部分
        # 2. token数不超过max_sentence_tokens
        eligible_sentences = []
        for sent in sentences:
            # 检查是否是第一句的一部分
            if sent in first_sentence:
                continue
            # 检查长度（简单估算：中文1字≈1token，英文按空格分）
            token_count = len(sent) if any('\u4e00' <= c <= '\u9fff' for c in sent) else len(sent.split())
            if token_count <= max_sentence_tokens:
                eligible_sentences.append(sent)
        
        if not eligible_sentences:
            return None
        
        # 随机选择一句删除
        to_remove = random.choice(eligible_sentences)
        
        # 从原文中删除（包括其后的标点）
        new_field_text = field_text
        for punct in ['，', '。', ',', '.']:
            pattern = to_remove + punct
            if pattern in new_field_text:
                new_field_text = new_field_text.replace(pattern, '', 1)
                break
        else:
            # 如果没有标点，直接删除
            new_field_text = new_field_text.replace(to_remove, '', 1)
        
        # 清理多余空格
        new_field_text = re.sub(r'\s+', ' ', new_field_text).strip()
        
        # 重构summary
        if field_name == 'explanation':
            return f"EXPLANATION: {new_field_text}\nSCOPE: {scope_part}"
        else:
            return f"EXPLANATION: {explanation_part}\nSCOPE: {new_field_text}"
    
    def collect_updater_samples_fast(
        self
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        快速构造总结更新系统的训练数据
        
        新的三种数据类型：
        1. 生成新节点（无子节点）：node_summary为空，parent_summary为当前节点，sibling_summaries为空
        2. 生成新节点（有子节点）：node_summary为空，parent_summary为当前节点，sibling_summaries为当前节点的子节点
        3. 更新现有节点：node_summary为当前节点，parent_summary为父节点，sibling_summaries为兄弟节点
        
        比例：1:3:4
        
        Returns:
            (train_samples, val_samples)
        """
        print("\n快速构造总结更新系统训练数据...")
        print("  数据类型：")
        print("    Type1: 在叶子节点下首次创建子节点")
        print("           - node_summary: 空")
        print("           - parent_summary: 当前叶子节点")
        print("           - sibling_summaries: 空")
        print("    Type2: 在非叶子节点下创建新子节点")
        print("           - node_summary: 空")
        print("           - parent_summary: 当前非叶子节点")
        print("           - sibling_summaries: 当前节点的已有子节点")
        print("    Type3: 更新现有节点")
        print("           - node_summary: 当前节点")
        print("           - parent_summary: 父节点")
        print("           - sibling_summaries: 兄弟节点")
        print("  比例: 1:3:4")
        
        # 配置参数
        total_samples = self.config.data_prepare.updater_total_samples
        perturb_prob = self.config.data_prepare.updater_fast_perturb_prob
        max_sentence_tokens = self.config.data_prepare.updater_fast_max_sentence_tokens
        
        # 计算每种类型的数量（比例1:3:4，总和为8）
        type1_ratio = 1.0 / 8.0  # 无子节点生成新节点
        type2_ratio = 3.0 / 8.0  # 有子节点生成新节点
        type3_ratio = 4.0 / 8.0  # 更新现有节点
        
        # 计算需要生成的数量（×2留余地）
        type1_needed = int(total_samples * type1_ratio * 2)
        type2_needed = int(total_samples * type2_ratio * 2)
        type3_needed = int(total_samples * type3_ratio * 2)
        
        print(f"  - 目标样本总数: {total_samples}")
        print(f"  - Type1（生成新节点-无子节点）需生成: {type1_needed}")
        print(f"  - Type2（生成新节点-有子节点）需生成: {type2_needed}")
        print(f"  - Type3（更新现有节点）需生成: {type3_needed}")
        print(f"  - 扰动概率: {perturb_prob}")
        
        # 获取训练集topics
        train_topics = list(self.dataset_split.get('train', {}).keys())
        print(f"  - 训练集topics数量: {len(train_topics)}")
        
        # 收集候选数据
        type1_candidates = []  # 生成新节点（无子节点）
        type2_candidates = []  # 生成新节点（有子节点）
        type3_candidates = []  # 更新现有节点
        
        print("\n  收集候选数据...")
        for topic_key in tqdm(train_topics, desc="处理topics"):
            if topic_key not in self.structures_data or topic_key not in self.summaries_data:
                continue
            if topic_key not in self.references_data:
                continue
            
            structure = self.structures_data[topic_key]
            summaries = self.summaries_data[topic_key]
            references = self.references_data[topic_key].get('references', {})
            topic_name = self.references_data[topic_key]['topic']
            
            # 遍历结构树收集数据
            def collect_from_node(node_dict, parent_summary, siblings_summaries, current_path):
                """递归收集节点数据"""
                node_path = current_path
                
                # 获取当前节点summary
                if node_path not in summaries:
                    return
                
                node_summary_data = summaries[node_path]
                if isinstance(node_summary_data, dict):
                    if 'explanation' in node_summary_data and 'scope' in node_summary_data:
                        node_summary = f"EXPLANATION: {node_summary_data['explanation']}\nSCOPE: {node_summary_data['scope']}"
                        explanation = node_summary_data['explanation']
                        scope = node_summary_data['scope']
                    elif 'full' in node_summary_data:
                        node_summary = node_summary_data['full']
                        # 尝试解析
                        try:
                            parts = node_summary.split('\nSCOPE: ')
                            explanation = parts[0].replace('EXPLANATION: ', '')
                            scope = parts[1] if len(parts) > 1 else ""
                        except:
                            return
                    else:
                        return
                else:
                    return
                
                # 获取子节点
                children = node_dict.get('children', [])
                
                # 构建子节点summary列表（用于Type2和Type3）
                child_summaries_list = []
                if children:
                    for child in children:
                        child_title = child.get('title', '')
                        child_path = f"{node_path} - {child_title}" if node_path else child_title
                        if child_path in summaries:
                            child_summary_data = summaries[child_path]
                            if isinstance(child_summary_data, dict):
                                if 'explanation' in child_summary_data and 'scope' in child_summary_data:
                                    child_summary = f"EXPLANATION: {child_summary_data['explanation']}\nSCOPE: {child_summary_data['scope']}"
                                elif 'full' in child_summary_data:
                                    child_summary = child_summary_data['full']
                                else:
                                    continue
                            else:
                                child_summary = str(child_summary_data)
                            child_summaries_list.append(child_summary)
                
                # Type1: 生成新节点（无子节点）- 模拟在叶子节点下首次创建子节点
                # parent_summary是当前节点，sibling_summaries为空
                if not children:
                    node_citations = node_dict.get('citations', [])
                    for ref_id in node_citations:
                        if ref_id in references:
                            article_content = references[ref_id].get('content', '')
                            if article_content:
                                type1_candidates.append({
                                    'topic_name': topic_name,
                                    'node_summary': "",  # 空的，表示生成新节点
                                    'parent_summary': node_summary,  # 当前节点作为parent
                                    'sibling_summaries': [],  # 空的，表示首次创建
                                    'new_content': article_content,
                                    'target_explanation': explanation,
                                    'target_scope': scope
                                })
                
                # Type2: 生成新节点（有子节点）- 模拟在非叶子节点下创建新子节点
                # parent_summary是当前节点，sibling_summaries是当前节点的子节点
                if len(child_summaries_list) > 0:
                    # 随机选一个子节点作为"新创建"的节点
                    for child_summary in child_summaries_list:
                        # 其他子节点作为siblings（对于新节点来说，它们是兄弟节点）
                        other_siblings = [s for s in child_summaries_list if s != child_summary]
                        
                        type2_candidates.append({
                            'topic_name': topic_name,
                            'node_summary': "",  # 空的，表示生成新节点
                            'parent_summary': node_summary,  # 当前节点作为parent
                            'sibling_summaries': other_siblings,  # 当前节点的其他子节点
                            'new_content': child_summary,  # 使用子节点作为内容
                            'target_explanation': explanation,
                            'target_scope': scope
                        })
                
                # Type3: 更新现有节点 - 使用子节点或文章作为new_content
                # 如果是叶子节点，使用文章
                if not children:
                    node_citations = node_dict.get('citations', [])
                    for ref_id in node_citations:
                        if ref_id in references:
                            article_content = references[ref_id].get('content', '')
                            if article_content:
                                type3_candidates.append({
                                    'topic_name': topic_name,
                                    'node_summary': node_summary,  # 当前节点的summary
                                    'parent_summary': parent_summary,
                                    'sibling_summaries': siblings_summaries,
                                    'new_content': article_content
                                })
                
                # Type3: 如果有子节点，使用子节点作为new_content
                if children and node_summary != topic_name:  # 不是根节点
                    for child_summary in child_summaries_list:
                        type3_candidates.append({
                            'topic_name': topic_name,
                            'node_summary': node_summary,  # 当前节点的summary
                            'parent_summary': parent_summary,
                            'sibling_summaries': siblings_summaries,
                            'new_content': child_summary
                        })
                
                # 递归处理子节点
                for i, child in enumerate(children):
                    child_title = child.get('title', '')
                    child_path = f"{node_path} - {child_title}" if node_path else child_title
                    
                    # 获取子节点的兄弟节点summaries
                    child_siblings = []
                    for j, sibling in enumerate(children):
                        if i != j:
                            sibling_title = sibling.get('title', '')
                            sibling_path = f"{node_path} - {sibling_title}" if node_path else sibling_title
                            if sibling_path in summaries:
                                sibling_summary_data = summaries[sibling_path]
                                if isinstance(sibling_summary_data, dict):
                                    if 'explanation' in sibling_summary_data and 'scope' in sibling_summary_data:
                                        sibling_summary = f"EXPLANATION: {sibling_summary_data['explanation']}\nSCOPE: {sibling_summary_data['scope']}"
                                    elif 'full' in sibling_summary_data:
                                        sibling_summary = sibling_summary_data['full']
                                    else:
                                        continue
                                else:
                                    sibling_summary = str(sibling_summary_data)
                                child_siblings.append(sibling_summary)
                    
                    collect_from_node(child, node_summary, child_siblings, child_path)
            
            # 从根节点开始收集
            root_structure = structure.get('structure', [])
            for node in root_structure:
                node_title = node.get('title', '')
                node_path = f"{topic_name} - {node_title}"
                
                # 获取根节点下的兄弟节点summaries
                root_siblings = []
                for sibling in root_structure:
                    if sibling != node:
                        sibling_title = sibling.get('title', '')
                        sibling_path = f"{topic_name} - {sibling_title}"
                        if sibling_path in summaries:
                            sibling_summary_data = summaries[sibling_path]
                            if isinstance(sibling_summary_data, dict):
                                if 'explanation' in sibling_summary_data and 'scope' in sibling_summary_data:
                                    sibling_summary = f"EXPLANATION: {sibling_summary_data['explanation']}\nSCOPE: {sibling_summary_data['scope']}"
                                elif 'full' in sibling_summary_data:
                                    sibling_summary = sibling_summary_data['full']
                                else:
                                    continue
                            else:
                                sibling_summary = str(sibling_summary_data)
                            root_siblings.append(sibling_summary)
                
                collect_from_node(node, topic_name, root_siblings, node_path)
        
        print(f"\n  - Type1候选数据: {len(type1_candidates)}")
        print(f"  - Type2候选数据: {len(type2_candidates)}")
        print(f"  - Type3候选数据: {len(type3_candidates)}")
        
        # 随机采样到目标数量
        type1_samples = random.sample(type1_candidates, min(len(type1_candidates), type1_needed))
        type2_samples = random.sample(type2_candidates, min(len(type2_candidates), type2_needed))
        type3_samples = random.sample(type3_candidates, min(len(type3_candidates), type3_needed))
        
        print(f"  - Type1采样后: {len(type1_samples)}")
        print(f"  - Type2采样后: {len(type2_samples)}")
        print(f"  - Type3采样后: {len(type3_samples)}")
        
        # 对Type3应用扰动（Type1和Type2不需要扰动，因为node_summary是空的）
        print(f"\n  应用summary扰动（仅Type3，概率={perturb_prob}）...")
        for sample in type3_samples:
            if random.random() < perturb_prob:
                perturbed = self._perturb_summary(sample['node_summary'], max_sentence_tokens)
                if perturbed:
                    sample['node_summary'] = perturbed
        
        # 生成所有prompts
        print(f"\n  生成prompts...")
        all_prompts = []
        all_metadata = []  # 保存元数据：类型、原始数据
        
        for sample in type1_samples + type2_samples + type3_samples:
            prompt = PromptTemplates.format_summary_prompt(
                topic_name=sample['topic_name'],
                node_summary=sample['node_summary'],
                parent_summary=sample['parent_summary'],
                sibling_summaries=sample['sibling_summaries'],
                new_content=sample['new_content'][:3000]
            )
            all_prompts.append(prompt)
            # 判断类型
            if sample in type1_samples:
                sample_type = 'type1'
            elif sample in type2_samples:
                sample_type = 'type2'
            else:
                sample_type = 'type3'
            all_metadata.append({'type': sample_type, 'data': sample})
        
        print(f"  - 总prompt数: {len(all_prompts)}")
        
        # 批量调用API
        print(f"\n  批量调用API...")
        api_config = DeepSeekConfig(
            api_key=self.config.summary.api_key,
            base_url=self.config.summary.api_url.replace('/chat/completions', ''),
            model=self.config.summary.model_name,
            temperature=self.config.summary.temperature,
            max_output_tokens=self.config.summary.max_tokens,
            max_concurrent_jobs=self.config.summary.max_workers
        )
        api_client = DeepSeekAPIClient(api_config)
        
        responses = api_client.run_prompts_to_texts(all_prompts, show_progress=True)
        
        # 解析响应并分类
        print(f"\n  解析响应...")
        type1_valid = []  # Type1有效样本
        type2_valid = []  # Type2有效样本
        type3_valid = []  # Type3有效样本
        
        for i, (response_text, metadata) in enumerate(zip(responses, all_metadata)):
            if response_text is None:
                continue
            
            try:
                parsed = PromptTemplates.parse_summary_output(response_text)
                if parsed and parsed.get('explanation') and parsed.get('scope'):
                    prompt = all_prompts[i]
                    sample_data = {
                        'prompt': prompt,
                        'completion': response_text
                    }
                    
                    sample_type = metadata['type']
                    
                    if sample_type == 'type1':
                        type1_valid.append(sample_data)
                    elif sample_type == 'type2':
                        type2_valid.append(sample_data)
                    else:  # type3
                        type3_valid.append(sample_data)
            except Exception as e:
                continue
        
        print(f"  - Type1有效样本: {len(type1_valid)}")
        print(f"  - Type2有效样本: {len(type2_valid)}")
        print(f"  - Type3有效样本: {len(type3_valid)}")
        
        # 最终采样（按比例1:3:4）
        print(f"\n  最终采样...")
        type1_final_needed = int(total_samples * type1_ratio)
        type2_final_needed = int(total_samples * type2_ratio)
        type3_final_needed = total_samples - type1_final_needed - type2_final_needed
        
        final_samples = []
        final_samples.extend(self._sample_list(type1_valid, type1_final_needed))
        final_samples.extend(self._sample_list(type2_valid, type2_final_needed))
        final_samples.extend(self._sample_list(type3_valid, type3_final_needed))
        
        random.shuffle(final_samples)
        
        # 划分训练集和验证集
        val_size = int(len(final_samples) * 0.1)
        val_samples = final_samples[:val_size]
        train_samples = final_samples[val_size:]
        
        print(f"\n  - 最终生成:")
        print(f"    训练集: {len(train_samples)} 样本")
        print(f"    验证集: {len(val_samples)} 样本")
        print(f"    Type1（生成新节点-无子节点）: ~{len([s for s in final_samples if '节点summary为空' in str(s) or len(type1_valid) > 0])} 样本")
        print(f"    Type2（生成新节点-有子节点）: ~{len([s for s in final_samples if len(type2_valid) > 0])} 样本")
        print(f"    Type3（更新现有节点）: ~{len([s for s in final_samples if len(type3_valid) > 0])} 样本")
        
        return train_samples, val_samples
    
    def collect_updater_samples(
        self,
        classify_generator_model_path: str = None,
        use_service: bool = False,
        service_url: str = None
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        构造总结更新系统的训练数据（模拟构建模式）
        
        使用训练好的分类生成系统构建结构树，在需要更新时调用API生成数据
        
        Args:
            classify_generator_model_path: 分类生成系统模型路径（direct模式）
            use_service: 是否使用HTTP服务模式
            service_url: 服务URL（service模式）
            
        Returns:
            (train_samples, val_samples)
        """
        print("\n构造总结更新系统训练数据...")
        
        # 加载分类生成系统
        if use_service:
            # HTTP服务模式
            print(f"  - 使用分类生成系统HTTP服务: {service_url}")
            from summary_based_classifier.models.model_clients import ClassifyGeneratorClient
            classify_generator = ClassifyGeneratorClient(service_url=service_url)
        else:
            # 直接加载模式（使用两个GPU加速推理）
            print(f"  - 直接加载分类生成系统到两个GPU: {classify_generator_model_path}")
            from summary_based_classifier.llm.classify_generator import ClassifyGenerator
            classify_generator = ClassifyGenerator(
                mode='model',
                model_path=classify_generator_model_path,
                tensor_parallel_size=2,  # 使用两个GPU进行tensor并行
                max_model_len=self.config.inference.max_model_len,
                gpu_memory_utilization=self.config.inference.gpu_memory_utilization,
                temperature=self.config.inference.temperature,
                top_p=self.config.inference.top_p,
                max_tokens=self.config.inference.max_tokens
            )
        
        # 初始化API客户端（用于生成更新系统的输出）
        print(f"  - 初始化API客户端")
        api_config = DeepSeekConfig(
            api_key=self.config.summary.api_key,
            base_url=self.config.summary.api_url.replace('/chat/completions', ''),
            model=self.config.summary.model_name,
            temperature=self.config.summary.temperature,
            max_output_tokens=self.config.summary.max_tokens,
            max_concurrent_jobs=self.config.summary.max_workers
        )
        api_client = DeepSeekAPIClient(api_config)
        
        # 动态采样参数
        total_samples = self.config.data_prepare.updater_total_samples
        update_ratio = self.config.data_prepare.updater_update_ratio
        positive_target = int(total_samples * update_ratio)
        negative_target = total_samples - positive_target
        
        print(f"  - 目标样本: 需要更新={positive_target}, 不需要更新={negative_target}, 总计={total_samples}")
        
        # 使用锁保护共享变量
        positive_samples = []
        negative_samples = []
        samples_lock = Lock()
        
        # 获取训练集topics
        train_topics = list(self.dataset_split.get('train', {}).keys())
        print(f"  - 训练集topics数量: {len(train_topics)}")
        
        def process_topic(topic_key: str, topic_positive_target: int, topic_negative_target: int):
            """处理单个topic"""
            if topic_key not in self.structures_data:
                return 0, 0, [], []
            if topic_key not in self.references_data:
                return 0, 0, [], []
            
            topic_data = self.references_data[topic_key]
            topic_name = topic_data['topic']
            references = topic_data.get('references', {})
            
            # 创建一个特殊的TreeBuilder，用API记录更新数据
            from summary_based_classifier.core.pipeline.builder import TreeBuilder
            from summary_based_classifier.llm.updater import SummaryUpdater
            
            # 创建一个虚拟的updater（不实际使用）
            class APIRecordingUpdater(SummaryUpdater):
                """用于记录API调用的Updater"""
                def __init__(self, api_client, prompt_templates):
                    self.api_client = api_client
                    self.prompt_templates = prompt_templates
                    self.recorded_samples = {'positive': [], 'negative': []}
                
                def update(self, input_data):
                    """调用API并记录结果"""
                    # 构建prompt
                    prompt = self.prompt_templates.format_updater_prompt(
                        topic_name=input_data.topic_name,
                        node_summary=input_data.node_summary,
                        parent_summary=input_data.parent_summary,
                        sibling_summaries=input_data.sibling_summaries,
                        new_content=input_data.new_content
                    )
                    
                    # 调用API
                    try:
                        jobs = self.api_client.run_prompts([prompt], show_progress=False)
                        if jobs and jobs[0].status == 'completed':
                            response_text = self.api_client.extract_text(jobs[0].result)
                            
                            # 解析输出
                            parsed = self.prompt_templates.parse_updater_json_output(response_text)
                            if parsed:
                                # 记录样本
                                sample = {
                                    'prompt': prompt,
                                    'completion': response_text
                                }
                                
                                if parsed.get('needs_update', False):
                                    self.recorded_samples['positive'].append(sample)
                                else:
                                    self.recorded_samples['negative'].append(sample)
                                
                                # 返回解析后的结果
                                from summary_based_classifier.llm.updater import UpdateOutput
                                return UpdateOutput(
                                    needs_update=parsed['needs_update'],
                                    updated_summary=parsed.get('updated_summary'),
                                    reasoning=parsed
                                )
                    except Exception as e:
                        print(f"      API调用失败: {e}")
                    
                    return None
                
                def update_batch(self, inputs):
                    """批量更新（这里简化为逐个调用）"""
                    return [self.update(inp) for inp in inputs]
            
            # 创建API recording updater
            recording_updater = APIRecordingUpdater(api_client, PromptTemplates)
            
            # 创建TreeBuilder
            builder = TreeBuilder(
                classify_generator=classify_generator,
                updater=recording_updater,
                references_data=self.references_data,
                summaries_data=self.summaries_data,
                max_depth=self.config.inference.max_depth,
                similarity_threshold=self.config.inference.similarity_threshold
            )
            
            # 构建结构树（处理所有文章）
            reference_ids = list(references.keys())
            try:
                root, _ = builder.build_tree(topic_key, reference_ids, record_mode=False)
            except Exception as e:
                print(f"      构建树失败: {e}")
                return 0, 0, [], []
            
            # 获取记录的样本
            topic_positive = recording_updater.recorded_samples['positive']
            topic_negative = recording_updater.recorded_samples['negative']
            
            # 根据目标数量采样
            sampled_positive = self._sample_list(topic_positive, topic_positive_target)
            sampled_negative = self._sample_list(topic_negative, topic_negative_target)
            
            return len(sampled_positive), len(sampled_negative), sampled_positive, sampled_negative
        
        # 多线程处理topics
        from concurrent.futures import ThreadPoolExecutor, as_completed
        max_workers = self.config.data_prepare.max_parallel_topics
        print(f"  - 使用 {max_workers} 个线程并行处理")
        
        # 平均分配目标到每个线程
        avg_positive_per_topic = positive_target // len(train_topics)
        avg_negative_per_topic = negative_target // len(train_topics)
        
        # 为了应对某些topic可能没有足够样本的情况，我们给每个topic稍高的目标
        # 并在收集过程中动态调整
        topic_targets = []
        for i in range(len(train_topics)):
            topic_targets.append((
                max(1, avg_positive_per_topic),
                max(1, avg_negative_per_topic)
            ))
        
        # 使用线程池处理
        completed_topics = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_topic = {}
            for i, topic_key in enumerate(train_topics):
                future = executor.submit(
                    process_topic,
                    topic_key,
                    topic_targets[i][0],
                    topic_targets[i][1]
                )
                future_to_topic[future] = (i, topic_key)
            
            # 处理完成的任务
            for future in tqdm(as_completed(future_to_topic), total=len(train_topics), desc="处理topics"):
                i, topic_key = future_to_topic[future]
                
                # 检查是否已经达标
                with samples_lock:
                    current_positive = len(positive_samples)
                    current_negative = len(negative_samples)
                    
                    if current_positive >= positive_target and current_negative >= negative_target:
                        # 已达标，取消未完成的任务
                        for f in future_to_topic:
                            if not f.done():
                                f.cancel()
                        print(f"\n  - 已达标，停止处理")
                        break
                
                try:
                    actual_positive, actual_negative, pos_samples, neg_samples = future.result()
                    
                    # 更新样本列表
                    with samples_lock:
                        positive_samples.extend(pos_samples)
                        negative_samples.extend(neg_samples)
                        completed_topics += 1
                    
                    print(f"\n  完成topic {completed_topics}/{len(train_topics)}: {topic_key}")
                    print(f"    收集: 需要更新={actual_positive}, 不需要更新={actual_negative}")
                    print(f"    总进度: 需要更新={len(positive_samples)}/{positive_target}, 不需要更新={len(negative_samples)}/{negative_target}")
                    
                except Exception as e:
                    print(f"\n  处理topic {topic_key} 失败: {e}")
        
        # 合并所有样本
        all_samples = positive_samples + negative_samples
        random.shuffle(all_samples)
        
        # 划分训练集和验证集
        val_size = int(len(all_samples) * 0.1)
        val_samples = all_samples[:val_size]
        train_samples = all_samples[val_size:]
        
        print(f"\n  - 最终生成:")
        print(f"    需要更新样本: {len(positive_samples)}")
        print(f"    不需要更新样本: {len(negative_samples)}")
        print(f"    训练集: {len(train_samples)} 样本")
        print(f"    验证集: {len(val_samples)} 样本")
        
        return train_samples, val_samples


def main():
    parser = argparse.ArgumentParser(description='准备训练数据')
    parser.add_argument(
        '--config',
        type=str,
        default='./configs/default.json',
        help='配置文件路径'
    )
    parser.add_argument(
        '--skip_classify_generator',
        default=False,
        help='跳过分类生成系统数据集构建'
    )
    parser.add_argument(
        '--skip_updater',
        default=False,
        help='跳过总结更新系统数据集构建'
    )
    parser.add_argument(
        '--classify_generator_model',
        type=str,
        default='/mnt/literism/tree/summary_output/models/classify_generator/model',
        help='分类生成系统模型路径（direct模式，用于构建更新系统数据集）'
    )
    parser.add_argument(
        '--use_service',
        default=False,
        help='使用HTTP服务模式调用分类生成系统（推荐，支持并发）'
    )
    parser.add_argument(
        '--service_url',
        type=str,
        default='http://localhost:8000',
        help='分类生成系统服务URL（service模式）'
    )
    parser.add_argument(
        '--fast_mode',
        default=True,
        help='快速模式：直接从目标结构树生成更新系统数据集（不模拟构建）'
    )
    parser.add_argument(
        '--debug_mode',
        action='store_true',
        help='调试模式：只生成prompt不调用API'
    )
    parser.add_argument(
        '--debug_output',
        type=str,
        default='./debug_samples.jsonl',
        help='调试模式下输出文件路径'
    )
    args = parser.parse_args()
    
    # 加载配置
    config = SummaryBasedConfig.from_json(args.config)
    
    print("="*80)
    print("训练数据准备")
    print("="*80)
    
    # 创建preparator
    preparator = TrainingDataPreparator(config)
    preparator.load_data()
    
    # 1. 构建分类生成系统数据集
    if not args.skip_classify_generator:
        print("\n" + "="*80)
        print("步骤1: 构建分类生成系统数据集")
        print("="*80)
        
        raw_samples = preparator.collect_classify_generate_samples()
        
        if args.debug_mode:
            # 调试模式
            preparator.construct_classify_generator_dataset(
                raw_samples, 
                debug_mode=True, 
                debug_output_file=args.debug_output
            )
            print(f"\n调试模式完成！请检查: {args.debug_output}")
        else:
            # 正常模式
            train_samples, val_samples = preparator.construct_classify_generator_dataset(raw_samples)
            
            # 保存
            train_file = preparator.output_dir / 'classify_generator_train.jsonl'
            val_file = preparator.output_dir / 'classify_generator_val.jsonl'
            
            with open(train_file, 'w', encoding='utf-8') as f:
                for sample in train_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            with open(val_file, 'w', encoding='utf-8') as f:
                for sample in val_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            print(f"\n分类生成系统数据集已保存:")
            print(f"  - 训练集: {train_file}")
            print(f"  - 验证集: {val_file}")
    
    # 2. 构建总结更新系统数据集
    if not args.skip_updater:
        print("\n" + "="*80)
        print("步骤2: 构建总结更新系统数据集")
        print("="*80)
        
        if args.fast_mode:
            # 快速模式：直接从目标结构树生成
            print("使用快速模式（直接从目标结构树生成）")
            train_samples, val_samples = preparator.collect_updater_samples_fast()
        else:
            # 模拟构建模式
            if not args.use_service and not args.classify_generator_model:
                print("错误: 需要指定 --classify_generator_model 参数或使用 --use_service 模式")
                print("请先训练分类生成系统，然后使用训练好的模型来构建更新系统数据集")
                return
            
            print("使用模拟构建模式")
            train_samples, val_samples = preparator.collect_updater_samples(
                classify_generator_model_path=args.classify_generator_model,
                use_service=args.use_service,
                service_url=args.service_url
            )
        
        # 保存
        train_file = preparator.output_dir / 'updater_train.jsonl'
        val_file = preparator.output_dir / 'updater_val.jsonl'
        
        with open(train_file, 'w', encoding='utf-8') as f:
            for sample in train_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        with open(val_file, 'w', encoding='utf-8') as f:
            for sample in val_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"\n总结更新系统数据集已保存:")
        print(f"  - 训练集: {train_file}")
        print(f"  - 验证集: {val_file}")
    
    print("\n" + "="*80)
    print("数据准备完成！")
    print("="*80)


if __name__ == '__main__':
    main()

