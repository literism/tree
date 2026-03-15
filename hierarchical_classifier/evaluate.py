"""
评估脚本
将推理结果与真实结果进行聚类比较，计算Omega Index和ONMI
"""
import json
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score


def extract_leaf_clusters(tree_dict: Dict) -> List[Set[str]]:
    """
    从树结构中提取叶子节点作为聚类簇
    
    Args:
        tree_dict: 树的字典表示
        
    Returns:
        簇列表，每个簇是一个引用ID的集合
    """
    clusters = []
    
    def traverse(node):
        """递归遍历树"""
        if not node.get('children'):
            # 叶子节点
            citations = node.get('citations', [])
            if citations:  # 只保留非空的簇
                clusters.append(set(citations))
        else:
            # 非叶子节点，递归处理子节点
            for child in node['children']:
                traverse(child)
    
    # 检查是否有'structure'字段（真实数据格式）
    if 'structure' in tree_dict:
        # 真实数据格式：有structure字段，包含多个顶级节点
        for node in tree_dict['structure']:
            traverse(node)
    else:
        # 推理结果格式：整个dict就是一个树节点
        traverse(tree_dict)
    
    return clusters


def get_all_citations(clusters: List[Set[str]]) -> Set[str]:
    """获取所有簇中的引用文章"""
    all_citations = set()
    for cluster in clusters:
        all_citations.update(cluster)
    return all_citations


def filter_clusters(clusters: List[Set[str]], valid_citations: Set[str]) -> List[Set[str]]:
    """
    过滤簇，只保留valid_citations中的文章
    
    Args:
        clusters: 簇列表
        valid_citations: 有效的引用ID集合
        
    Returns:
        过滤后的簇列表（去除了空簇）
    """
    filtered_clusters = []
    for cluster in clusters:
        filtered_cluster = cluster & valid_citations
        if filtered_cluster:  # 只保留非空的簇
            filtered_clusters.append(filtered_cluster)
    return filtered_clusters


def prune_tree(tree_dict: Dict, valid_citations: Set[str]) -> Dict:
    """
    裁剪树结构，只保留包含valid_citations中文章的节点
    
    递归处理：
    1. 叶子节点：只保留在valid_citations中的文章，如果文章为空则删除节点
    2. 非叶子节点：递归处理所有子节点，如果所有子节点都被删除则删除该节点
    
    Args:
        tree_dict: 树的字典表示
        valid_citations: 有效的引用ID集合
        
    Returns:
        裁剪后的树字典，如果整棵树都应被删除则返回None
    """
    def prune_node(node: Dict) -> Dict:
        """
        递归裁剪节点
        返回裁剪后的节点，如果节点应被删除则返回None
        """
        # 过滤引用
        filtered_citations = [c for c in node.get('citations', []) if c in valid_citations]
        
        # 处理子节点
        if node.get('children'):
            # 非叶子节点：递归处理所有子节点
            filtered_children = []
            for child in node['children']:
                pruned_child = prune_node(child)
                if pruned_child is not None:
                    filtered_children.append(pruned_child)
            
            # 如果所有子节点都被删除了，检查当前节点是否有引用
            if not filtered_children and not filtered_citations:
                return None  # 删除这个节点
            
            # 保留这个节点
            return {
                'title': node['title'],
                'level': node['level'],
                'citations': filtered_citations,
                'children': filtered_children
            }
        else:
            # 叶子节点：如果没有有效引用，删除该节点
            if not filtered_citations:
                return None
            
            return {
                'title': node['title'],
                'level': node['level'],
                'citations': filtered_citations,
                'children': []
            }
    
    # 处理真实数据格式（有structure字段）
    if 'structure' in tree_dict:
        pruned_structure = []
        for node in tree_dict['structure']:
            pruned_node = prune_node(node)
            if pruned_node is not None:
                pruned_structure.append(pruned_node)
        
        # 创建新的树字典
        result = tree_dict.copy()
        result['structure'] = pruned_structure
        return result
    else:
        # 推理结果格式（整个dict就是一个树节点）
        pruned_node = prune_node(tree_dict)
        return pruned_node if pruned_node is not None else {
            'title': tree_dict.get('title', ''),
            'level': tree_dict.get('level', 0),
            'citations': [],
            'children': []
        }


def clusters_to_labels(clusters: List[Set[str]], all_citations: List[str]) -> np.ndarray:
    """
    将簇表示转换为标签数组
    
    Args:
        clusters: 簇列表
        all_citations: 所有引用ID的列表（定义顺序）
        
    Returns:
        标签数组
    """
    citation_to_idx = {citation: idx for idx, citation in enumerate(all_citations)}
    labels = np.full(len(all_citations), -1, dtype=int)
    
    for cluster_id, cluster in enumerate(clusters):
        for citation in cluster:
            if citation in citation_to_idx:
                labels[citation_to_idx[citation]] = cluster_id
    
    return labels


def compute_omega_index(pred_clusters: List[Set[str]], true_clusters: List[Set[str]], 
                       all_citations: List[str]) -> float:
    """
    计算Omega Index
    
    使用Adjusted Rand Index作为Omega Index的实现
    
    Args:
        pred_clusters: 预测的簇
        true_clusters: 真实的簇
        all_citations: 所有引用ID
        
    Returns:
        Omega Index值
    """
    pred_labels = clusters_to_labels(pred_clusters, all_citations)
    true_labels = clusters_to_labels(true_clusters, all_citations)
    
    # 只考虑被分配到簇的样本
    valid_mask = (pred_labels >= 0) & (true_labels >= 0)
    
    if not valid_mask.any():
        return 0.0
    
    pred_labels_filtered = pred_labels[valid_mask]
    true_labels_filtered = true_labels[valid_mask]
    
    # 使用Adjusted Rand Index
    return adjusted_rand_score(true_labels_filtered, pred_labels_filtered)


def compute_onmi(pred_clusters: List[Set[str]], true_clusters: List[Set[str]], 
                 all_citations: List[str]) -> float:
    """
    计算ONMI (Overlapping Normalized Mutual Information)
    
    这里使用标准的NMI作为近似
    
    Args:
        pred_clusters: 预测的簇
        true_clusters: 真实的簇
        all_citations: 所有引用ID
        
    Returns:
        ONMI值
    """
    pred_labels = clusters_to_labels(pred_clusters, all_citations)
    true_labels = clusters_to_labels(true_clusters, all_citations)
    
    # 只考虑被分配到簇的样本
    valid_mask = (pred_labels >= 0) & (true_labels >= 0)
    
    if not valid_mask.any():
        return 0.0
    
    pred_labels_filtered = pred_labels[valid_mask]
    true_labels_filtered = true_labels[valid_mask]
    
    # 使用NMI（average方法）
    return normalized_mutual_info_score(true_labels_filtered, pred_labels_filtered, 
                                       average_method='arithmetic')


def compute_avg_clusters_per_citation(clusters: List[Set[str]]) -> float:
    """
    计算每篇文章平均属于多少个簇
    
    Args:
        clusters: 簇列表
        
    Returns:
        平均每篇文章属于的簇数
    """
    if not clusters:
        return 0.0
    
    # 统计每篇文章出现在多少个簇中
    citation_count = {}
    for cluster in clusters:
        for citation in cluster:
            citation_count[citation] = citation_count.get(citation, 0) + 1
    
    if not citation_count:
        return 0.0
    
    # 计算平均值
    total_count = sum(citation_count.values())
    num_citations = len(citation_count)
    
    return total_count / num_citations


def evaluate_topic(pred_tree: Dict, true_tree: Dict, topic_key: str, min_cluster_size: int = 1) -> Dict:
    """
    评估单个topic的聚类结果
    
    Args:
        pred_tree: 预测的树结构
        true_tree: 真实的树结构
        topic_key: topic键
        min_cluster_size: 最小簇大小阈值，文章数少于此值的预测簇将被删除
        
    Returns:
        评估结果字典
    """
    # 1. 提取预测结果的叶子簇
    pred_clusters = extract_leaf_clusters(pred_tree)
    
    # 2. 过滤掉文章数小于阈值的预测簇
    if min_cluster_size > 1:
        pred_clusters_before = len(pred_clusters)
        pred_clusters = [cluster for cluster in pred_clusters if len(cluster) >= min_cluster_size]
        pred_clusters_after = len(pred_clusters)
        if pred_clusters_before != pred_clusters_after:
            print(f"    - 过滤簇: {pred_clusters_before} -> {pred_clusters_after} (删除了 {pred_clusters_before - pred_clusters_after} 个小簇)")
    
    # 3. 获取预测结果中的所有引用
    pred_citations = get_all_citations(pred_clusters)
    
    # 3. 裁剪真实结构树，只保留预测结果中存在的文章
    true_tree_pruned = prune_tree(true_tree, pred_citations)
    
    # 4. 从裁剪后的真实树中提取叶子簇
    true_clusters_filtered = extract_leaf_clusters(true_tree_pruned)
    
    # 5. 计算真实结果中每篇文章平均属于多少个簇
    avg_clusters_per_citation = compute_avg_clusters_per_citation(true_clusters_filtered)
    
    # 6. 获取所有引用的列表（用于标签转换）
    all_citations = sorted(list(pred_citations))
    
    if not all_citations:
        return {
            'topic_key': topic_key,
            'num_citations': 0,
            'num_pred_clusters': 0,
            'num_true_clusters': 0,
            'avg_clusters_per_citation': 0.0,
            'omega_index': 0.0,
            'onmi': 0.0,
            'error': 'No citations found'
        }
    
    # 7. 计算指标
    try:
        omega_index = compute_omega_index(pred_clusters, true_clusters_filtered, all_citations)
        onmi = compute_onmi(pred_clusters, true_clusters_filtered, all_citations)
    except Exception as e:
        return {
            'topic_key': topic_key,
            'num_citations': len(all_citations),
            'num_pred_clusters': len(pred_clusters),
            'num_true_clusters': len(true_clusters_filtered),
            'avg_clusters_per_citation': avg_clusters_per_citation,
            'omega_index': 0.0,
            'onmi': 0.0,
            'error': str(e)
        }
    
    return {
        'topic_key': topic_key,
        'num_citations': len(all_citations),
        'num_pred_clusters': len(pred_clusters),
        'num_true_clusters': len(true_clusters_filtered),
        'avg_clusters_per_citation': avg_clusters_per_citation,
        'omega_index': omega_index,
        'onmi': onmi
    }


def main():
    parser = argparse.ArgumentParser(description='评估推理结果')
    parser.add_argument(
        '--pred_file',
        type=str,
        default="/mnt/literism/tree/hierarchical_output/inference/test_hard_trees.json",
        help='推理结果文件路径'
    )
    parser.add_argument(
        '--true_file',
        type=str,
        default='/mnt/literism/tree/data/wikipedia_structures_final.json',
        help='真实结果文件路径'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        help='评估结果输出文件路径（可选）'
    )
    parser.add_argument(
        '--min_cluster_size',
        type=int,
        default=1,
        help='最小簇大小阈值，文章数少于此值的预测簇将被删除（默认为1，即不过滤）'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("评估聚类结果")
    print("="*80)
    
    if args.min_cluster_size > 1:
        print(f"\n最小簇大小阈值: {args.min_cluster_size} (将删除文章数少于此值的预测簇)")
    
    # 加载预测结果
    print(f"\n加载预测结果: {args.pred_file}")
    with open(args.pred_file, 'r', encoding='utf-8') as f:
        pred_trees = json.load(f)
    print(f"  - 加载 {len(pred_trees)} 个topics")
    
    # 加载真实结果
    print(f"\n加载真实结果: {args.true_file}")
    with open(args.true_file, 'r', encoding='utf-8') as f:
        true_trees = json.load(f)
    print(f"  - 加载 {len(true_trees)} 个topics")
    
    # 评估每个topic
    print("\n开始评估...")
    results = []
    
    for topic_key in pred_trees:
        if topic_key not in true_trees:
            print(f"警告: topic {topic_key} 不在真实结果中，跳过")
            continue
        
        result = evaluate_topic(pred_trees[topic_key], true_trees[topic_key], topic_key, args.min_cluster_size)
        results.append(result)
        
        if 'error' not in result:
            print(f"  {topic_key}: "
                  f"Omega={result['omega_index']:.4f}, "
                  f"ONMI={result['onmi']:.4f}, "
                  f"Citations={result['num_citations']}, "
                  f"Pred={result['num_pred_clusters']}, "
                  f"True={result['num_true_clusters']}, "
                  f"Avg_per_cite={result['avg_clusters_per_citation']:.2f}")
        else:
            print(f"  {topic_key}: 错误 - {result['error']}")
    
    # 计算平均指标
    print("\n"+"="*80)
    print("总体评估结果")
    print("="*80)
    
    valid_results = [r for r in results if 'error' not in r and r['num_citations'] > 0]
    
    if valid_results:
        avg_omega = np.mean([r['omega_index'] for r in valid_results])
        avg_onmi = np.mean([r['onmi'] for r in valid_results])
        avg_clusters_per_citation = np.mean([r['avg_clusters_per_citation'] for r in valid_results])
        total_citations = sum([r['num_citations'] for r in valid_results])
        total_pred_clusters = sum([r['num_pred_clusters'] for r in valid_results])
        total_true_clusters = sum([r['num_true_clusters'] for r in valid_results])
        
        print(f"\n评估的topics数量: {len(valid_results)}")
        print(f"总引用文章数: {total_citations}")
        print(f"总预测簇数: {total_pred_clusters}")
        print(f"总真实簇数: {total_true_clusters}")
        print(f"\n真实结果中，平均每篇文章属于 {avg_clusters_per_citation:.4f} 个簇")
        print(f"\n平均Omega Index: {avg_omega:.4f}")
        print(f"平均ONMI: {avg_onmi:.4f}")
        
        # 计算加权平均（按引用数量加权）
        weighted_omega = sum([r['omega_index'] * r['num_citations'] for r in valid_results]) / total_citations
        weighted_onmi = sum([r['onmi'] * r['num_citations'] for r in valid_results]) / total_citations
        weighted_avg_clusters = sum([r['avg_clusters_per_citation'] * r['num_citations'] for r in valid_results]) / total_citations
        
        print(f"\n加权平均Omega Index: {weighted_omega:.4f}")
        print(f"加权平均ONMI: {weighted_onmi:.4f}")
        print(f"加权平均每篇文章属于簇数: {weighted_avg_clusters:.4f}")
        
        # 保存详细结果
        if args.output_file:
            output_data = {
                'summary': {
                    'min_cluster_size': args.min_cluster_size,
                    'num_topics': len(valid_results),
                    'total_citations': total_citations,
                    'total_pred_clusters': total_pred_clusters,
                    'total_true_clusters': total_true_clusters,
                    'avg_clusters_per_citation': avg_clusters_per_citation,
                    'weighted_avg_clusters_per_citation': weighted_avg_clusters,
                    'avg_omega_index': avg_omega,
                    'avg_onmi': avg_onmi,
                    'weighted_omega_index': weighted_omega,
                    'weighted_onmi': weighted_onmi
                },
                'per_topic': results
            }
            
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            print(f"\n详细结果已保存到: {args.output_file}")
    else:
        print("\n没有有效的评估结果！")
    
    print("\n"+"="*80)


if __name__ == '__main__':
    main()

