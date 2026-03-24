"""
Oracle π* 数据生成（SFT）

目标（对应 command.txt:1104-1110）：
- 不再使用轨迹采样 / reward / iwsft 生成分类数据
- 对 train topics：打乱文章顺序，真实模拟结构树重构
- 由 oracle 策略 π* 决定：
  - Top-down 下钻 / CreateLeaf
  - CreateLeaf 后最多一次 InsertParentPath（由 π* 决定是否归拢以及归拢对象）
- 全流程走一遍 bottom-up summary 更新，但 updater 使用 BOW 模式（词频）而非模型
- 记录"分类模型应有的输入 prompt + 正确输出 completion"，用于 SFT 训练
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from summary_based_classifier.config import SummaryBasedConfig
from summary_based_classifier.core.policy.oracle_pi import (
    anc,
    decide_merge_with_after_create_leaf,
    decide_top_down_child_by_target_label,
    parse_gold_path,
)
from summary_based_classifier.core.trajectory.trajectory_sampler import TreeNode
from summary_based_classifier.llm.classify_generator import ClassificationInput
from summary_based_classifier.llm.prompts import PromptTemplates
from summary_based_classifier.llm.updater import SummaryInput, Updater


def collect_docs_in_subtree(node: TreeNode) -> List[str]:
    docs: List[str] = []
    stack = [node]
    while stack:
        cur = stack.pop()
        if getattr(cur, "citations", None):
            docs.extend(cur.citations)
        for ch in getattr(cur, "children", []) or []:
            stack.append(ch)
    return docs


def recompute_depths(node: TreeNode, depth: int = 0):
    node.depth = depth
    for c in node.children:
        recompute_depths(c, depth + 1)


def get_root(node: TreeNode) -> TreeNode:
    cur = node
    while cur.parent is not None:
        cur = cur.parent
    return cur


def insert_parent_path(parent: TreeNode, new_leaf: TreeNode, sibling: TreeNode) -> TreeNode:
    """
    InsertParentPath（受限版本）：只允许把 new_leaf + sibling 归拢到新父节点下。
    """
    if new_leaf.parent != parent:
        raise ValueError("InsertParentPath: new_leaf 必须是 parent 的直接子节点")
    if sibling.parent != parent:
        raise ValueError("InsertParentPath: sibling 必须是 parent 的直接子节点")
    if new_leaf == sibling:
        raise ValueError("InsertParentPath: sibling 不能等于 new_leaf")

    removed = {new_leaf, sibling}
    parent.children = [c for c in parent.children if c not in removed]

    new_parent = TreeNode(summary="", citations=[], children=[])
    parent.add_child(new_parent)
    new_parent.add_child(sibling)
    new_parent.add_child(new_leaf)

    recompute_depths(get_root(parent), 0)
    return new_parent


def set_bow_summary_from_children(
    updater: Updater, 
    node: TreeNode, 
    children: Sequence[TreeNode],
    bm25_stats: Optional[Dict] = None
):
    """
    合并子节点summary。
    - 如果bm25_stats提供：把子节点得分当作"虚拟文档的tf"，重新计算BM25
    - 否则：直接累加（原BOW逻辑）
    """
    from collections import Counter, defaultdict
    
    if bm25_stats:
        # Q2-b: 重新计算BM25
        # 把所有子节点得分累加作为"虚拟tf"
        virtual_tf = Counter()
        for ch in children:
            parsed = updater._extract_bow_json(ch.summary) or {}
            for word, score in parsed.items():
                virtual_tf[word] += float(score)  # 累加得分作为tf
        
        # 虚拟文档长度 = 累加的总得分
        virtual_doc_len = sum(virtual_tf.values())
        
        # 重新应用BM25
        scores = updater._compute_bm25_scores(
            virtual_tf, 
            int(virtual_doc_len),
            bm25_stats['df'],
            bm25_stats['total_docs'],
            bm25_stats['avg_doc_length']
        )
        node.summary = updater._format_bow_summary(scores)
    else:
        # 原BOW逻辑：直接累加
        bow = Counter()
        for ch in children:
            parsed = updater._extract_bow_json(ch.summary) or {}
            bow += Counter(parsed)
        node.summary = updater._format_bow_summary(bow)


@dataclass
class OracleSample:
    prompt: str
    completion: str
    topic_key: str
    article_id: str
    depth: int


@dataclass
class PendingMerge:
    """待归拢状态 - 表示一组应该归拢在一起的节点"""
    nodes: List[TreeNode]  # 待归拢的节点集合
    target_label: str  # 归拢目标的target_label（它们的LCA）
    parent: TreeNode  # 这些节点的共同父节点
    created_at_article_idx: int  # 第一个节点创建时的索引
    triggering_samples: List[Tuple['OracleSample', TreeNode]] = None  # (样本, 对应的节点) 对
    
    def __post_init__(self):
        if self.triggering_samples is None:
            self.triggering_samples = []


class OracleSFTGenerator:
    def __init__(self, config: SummaryBasedConfig, bow_top_k: int = 30, seed: int = 42, updater_mode: str = "bow", **updater_kwargs):
        self.config = config
        self.seed = seed
        self.rng = random.Random(seed)
        self.updater_mode = updater_mode
        
        # 初始化updater（支持bow/hybrid模式）
        if updater_mode == "hybrid":
            # 创建vLLM信号量
            import threading
            self.vllm_semaphore = threading.Semaphore(1)
            
            # hybrid模式需要传递model_path用于证据提取
            self.updater = Updater(
                mode="hybrid", 
                model_path=config.path.base_model,
                bow_top_k=bow_top_k,
                vllm_semaphore=self.vllm_semaphore,  # 传递信号量
                **updater_kwargs
            )
            # 为hybrid模式加载transformers模型用于embedding计算
            from transformers import AutoModel, AutoTokenizer
            import torch
            
            self.embedding_tokenizer = AutoTokenizer.from_pretrained(
                config.path.base_model, 
                trust_remote_code=True
            )
            self.embedding_model = AutoModel.from_pretrained(
                "/home/literism/model/Qwen3-Embedding-0.6B",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.embedding_model.eval()
            
            # Embedding缓存（避免重复计算）
            self.embedding_cache = {}
        else:
            self.updater = Updater(mode="bow", bow_top_k=bow_top_k)

        self.structures_data: Dict = {}
        self.references_data: Dict = {}
        self.dataset_split: Dict = {}

        # topic_key -> {article_id -> parts}（多路径文章直接用 multi_paths_by_topic 保存）
        self.article_parts_by_topic: Dict[str, Dict[str, Sequence[str]]] = {}
        # topic_key -> {article_id -> list[parts]}（所有路径）
        self.multi_paths_by_topic: Dict[str, Dict[str, List[Sequence[str]]]] = {}
        # 映射缓存：topic_key -> {target_label(str) -> current TreeNode}
        self.target_to_current: Dict[str, Dict[str, TreeNode]] = {}
        # BM25统计：topic_key -> {'df': {}, 'total_docs': int, 'avg_doc_length': float}
        self.bm25_stats_by_topic: Dict[str, Dict] = {}
        # 待归拢管理：topic_key -> List[PendingMerge]
        self.pending_merges: Dict[str, List[PendingMerge]] = {}
        
        # 多线程相关
        import threading
        self.lock = threading.Lock()
        
        # vLLM调用信号量：确保同一时间只有一个线程调用vLLM
        if updater_mode == "hybrid":
            self.vllm_semaphore = threading.Semaphore(1)

    def load_data(self):
        with open(self.config.path.structures_file, "r", encoding="utf-8") as f:
            self.structures_data = json.load(f)
        with open(self.config.path.references_file, "r", encoding="utf-8") as f:
            self.references_data = json.load(f)
        with open(Path(self.config.path.data_dir) / "dataset_split.json", "r", encoding="utf-8") as f:
            self.dataset_split = json.load(f)["dataset_split"]

    def _prepare_topic_article_parts(self, topic_key: str, ref_ids: List[str]):
        topic = self.references_data[topic_key]
        refs = topic.get("references", {})
        mm: Dict[str, List[Sequence[str]]] = {}
        for rid in ref_ids:
            ref = refs.get(rid, {})
            paths = ref.get("paths") or []
            if not paths:
                continue
            parts_list = [parse_gold_path(p) for p in paths if isinstance(p, str) and p.strip()]
            if not parts_list:
                continue
            mm[rid] = parts_list
        self.article_parts_by_topic[topic_key] = {}  # 在递归时动态填充（用于查找历史文章路径）
        self.multi_paths_by_topic[topic_key] = mm
        self.target_to_current[topic_key] = {}

    def _collect_bm25_stats(self, topic_key: str, ref_ids: List[str]):
        """第一遍扫描：收集该topic的BM25统计信息"""
        topic = self.references_data[topic_key]
        refs = topic.get("references", {})
        
        from collections import defaultdict
        df = defaultdict(int)  # 文档频率
        doc_lengths = []
        
        for rid in ref_ids:
            ref = refs.get(rid, {})
            content = ref.get("content", "") or ""
            if not content:
                continue
            
            # tokenize并统计
            bow = self.updater._bow_from_text(content)
            doc_lengths.append(sum(bow.values()))
            
            # 更新df
            for term in bow.keys():
                df[term] += 1
        
        avg_len = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 1.0
        
        self.bm25_stats_by_topic[topic_key] = {
            'df': dict(df),
            'total_docs': len(doc_lengths),
            'avg_doc_length': avg_len
        }
    
    def _get_node_bm25_features(self, topic_key: str, node: TreeNode, top_k: int = None, min_score: float = 0.1) -> Dict[str, float]:
        """
        获取节点的BM25特征（top-k关键词及得分）
        
        改进：
        1. 使用与prompt相同的top_k（bow_top_k，默认30）
        2. 添加min_score阈值，过滤低分词
        """
        # 使用与updater相同的top_k配置
        if top_k is None:
            top_k = self.updater.bow_top_k
        # 收集节点子树的所有文档内容
        docs = collect_docs_in_subtree(node)
        if not docs:
            return {}
        
        # 获取文档内容
        topic = self.references_data.get(topic_key, {})
        refs = topic.get("references", {})
        
        from collections import Counter
        all_terms = Counter()
        
        for doc_id in docs:
            ref = refs.get(doc_id, {})
            content = ref.get("content", "") or ""
            if content:
                bow = self.updater._bow_from_text(content)
                all_terms.update(bow)
        
        # 使用BM25得分对关键词排序
        bm25_stats = self.bm25_stats_by_topic.get(topic_key, {})
        if not bm25_stats:
            # 如果没有BM25统计，直接返回词频
            most_common = all_terms.most_common(top_k)
            return {term: float(count) for term, count in most_common if count >= 2}  # 至少出现2次
        
        # 计算BM25得分
        k1 = 1.5
        b = 0.75
        df = bm25_stats.get('df', {})
        N = bm25_stats.get('total_docs', 1)
        avgdl = bm25_stats.get('avg_doc_length', 1.0)
        
        term_scores = {}
        doc_length = sum(all_terms.values())
        
        for term, tf in all_terms.items():
            # BM25得分
            idf = max(0.01, (N - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5))
            score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_length / avgdl))
            term_scores[term] = score
        
        # 排序并过滤：取top-k且得分 >= min_score
        sorted_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)
        filtered_terms = [(term, score) for term, score in sorted_terms[:top_k] if score >= min_score]
        
        return dict(filtered_terms)
    
    def _bm25_query_score(self, topic_key: str, query_features: Dict[str, float], doc_ids: List[str]) -> float:
        """使用BM25特征查询文档集合，返回平均得分"""
        if not query_features or not doc_ids:
            return 0.0
        
        topic = self.references_data.get(topic_key, {})
        refs = topic.get("references", {})
        bm25_stats = self.bm25_stats_by_topic.get(topic_key, {})
        
        if not bm25_stats:
            return 0.0
        
        k1 = 1.5
        b = 0.75
        df = bm25_stats.get('df', {})
        N = bm25_stats.get('total_docs', 1)
        avgdl = bm25_stats.get('avg_doc_length', 1.0)
        
        total_score = 0.0
        
        for doc_id in doc_ids:
            ref = refs.get(doc_id, {})
            content = ref.get("content", "") or ""
            if not content:
                continue
            
            doc_bow = self.updater._bow_from_text(content)
            doc_length = sum(doc_bow.values())
            
            doc_score = 0.0
            for term, query_weight in query_features.items():
                if term not in doc_bow:
                    continue
                
                tf = doc_bow[term]
                idf = max(0.01, (N - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5))
                bm25_score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_length / avgdl))
                doc_score += query_weight * bm25_score
            
            total_score += doc_score
        
        return total_score / len(doc_ids) if doc_ids else 0.0
    
    def _calculate_bm25_similarity(self, topic_key: str, node1: TreeNode, node2: TreeNode) -> float:
        """
        计算两个节点的特征相似度（BOW模式）
        
        改进：
        1. 只使用top-k高得分词（已在_get_node_bm25_features中实现）
        2. 使用加权余弦相似度（考虑词的重要性）
        3. 结合词重叠度
        """
        # 收集两个节点子树的所有文档
        docs1 = collect_docs_in_subtree(node1)
        docs2 = collect_docs_in_subtree(node2)
        
        if not docs1 or not docs2:
            return 0.0
        
        # 获取两个类的BM25特征（使用与prompt相同的top_k）
        bm25_1 = self._get_node_bm25_features(topic_key, node1, min_score=0.1)
        bm25_2 = self._get_node_bm25_features(topic_key, node2, min_score=0.1)
        
        if not bm25_1 or not bm25_2:
            return 0.0
        
        # 方法1：加权余弦相似度
        common_terms = set(bm25_1.keys()) & set(bm25_2.keys())
        
        if not common_terms:
            return 0.0
        
        # 计算余弦相似度（基于BM25得分）
        dot_product = sum(bm25_1[term] * bm25_2[term] for term in common_terms)
        norm1 = sum(score ** 2 for score in bm25_1.values()) ** 0.5
        norm2 = sum(score ** 2 for score in bm25_2.values()) ** 0.5
        
        if norm1 > 0 and norm2 > 0:
            cosine_sim = dot_product / (norm1 * norm2)
        else:
            cosine_sim = 0.0
        
        # 方法2：Jaccard相似度（词集合重叠度）
        all_terms = set(bm25_1.keys()) | set(bm25_2.keys())
        jaccard_sim = len(common_terms) / len(all_terms) if all_terms else 0.0
        
        # 综合：余弦相似度权重0.7，Jaccard相似度权重0.3
        combined_sim = 0.7 * cosine_sim + 0.3 * jaccard_sim
        
        return combined_sim
    
    def _calculate_hybrid_similarity(self, topic_key: str, node1: TreeNode, node2: TreeNode) -> float:
        """
        计算两个节点的特征相似度（Hybrid模式）
        
        Hybrid特征 = Keywords (BM25) + Evidence (文本片段)
        相似度 = 0.4 * keyword_sim + 0.6 * semantic_sim
        """
        # 1. 关键词相似度（复用BM25逻辑）
        keyword_sim = self._calculate_bm25_similarity(topic_key, node1, node2)
        
        # 2. 证据片段语义相似度（使用transformers模型）
        try:
            evidence1 = self._extract_evidence_from_node(node1)
            evidence2 = self._extract_evidence_from_node(node2)
            
            if not evidence1 or not evidence2:
                # 如果没有证据片段，只使用关键词相似度
                return keyword_sim
            
            # 批量获取embeddings（更高效）
            embeddings = self._get_text_embeddings_batch([evidence1, evidence2])
            
            if embeddings[0] is None or embeddings[1] is None:
                # Embedding计算失败，只使用关键词相似度
                return keyword_sim
            
            # 计算cosine相似度
            import numpy as np
            dot_product = np.dot(embeddings[0], embeddings[1])
            norm1 = np.linalg.norm(embeddings[0])
            norm2 = np.linalg.norm(embeddings[1])
            
            if norm1 > 0 and norm2 > 0:
                semantic_sim = dot_product / (norm1 * norm2)
                semantic_sim = float(semantic_sim)
            else:
                semantic_sim = 0.0
            
        except Exception as e:
            print(f"警告: 计算语义相似度失败: {e}")
            semantic_sim = 0.0
        
        # 3. 综合相似度：关键词40%，语义60%
        combined_sim = 0.4 * keyword_sim + 0.6 * semantic_sim
        
        return combined_sim
    
    def _extract_evidence_from_node(self, node: TreeNode) -> str:
        """从节点的hybrid summary中提取evidence字段"""
        if not node.summary:
            return ""
        try:
            parsed = json.loads(node.summary.strip())
            if isinstance(parsed, dict):
                return parsed.get('evidence', '')
        except:
            pass
        return ""
    
    def _get_text_embedding(self, text: str):
        """使用transformers模型获取文本的embedding（带缓存）"""
        import torch
        
        if not text:
            return None
        
        # 检查缓存
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        try:
            # Tokenize
            inputs = self.embedding_tokenizer(
                text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.embedding_model.device)
            
            # 获取embedding（使用mean pooling）
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
                # 使用最后一层hidden state的mean pooling
                embeddings = outputs.last_hidden_state.mean(dim=1)
                embedding = embeddings[0].cpu().numpy()
            
            # 缓存结果
            self.embedding_cache[text] = embedding
            return embedding
            
        except Exception as e:
            print(f"警告: 获取embedding失败: {e}")
            return None
    
    def _get_text_embeddings_batch(self, texts: list):
        """批量获取文本embeddings（更高效）"""
        import torch
        
        if not texts:
            return []
        
        # 检查哪些需要计算
        results = [None] * len(texts)
        texts_to_compute = []
        indices_to_compute = []
        
        for i, text in enumerate(texts):
            if text in self.embedding_cache:
                results[i] = self.embedding_cache[text]
            else:
                texts_to_compute.append(text)
                indices_to_compute.append(i)
        
        if not texts_to_compute:
            return results
        
        try:
            # 批量tokenize
            inputs = self.embedding_tokenizer(
                texts_to_compute,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.embedding_model.device)
            
            # 批量获取embeddings
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                embeddings_np = embeddings.cpu().numpy()
            
            # 更新结果和缓存
            for i, idx in enumerate(indices_to_compute):
                embedding = embeddings_np[i]
                results[idx] = embedding
                self.embedding_cache[texts_to_compute[i]] = embedding
            
            return results
            
        except Exception as e:
            print(f"警告: 批量获取embeddings失败: {e}")
            return [None] * len(texts)
    
    def _calculate_similarity(self, topic_key: str, node1: TreeNode, node2: TreeNode) -> float:
        """根据updater模式自动选择相似度计算方法"""
        if self.updater_mode == "hybrid":
            return self._calculate_hybrid_similarity(topic_key, node1, node2)
        else:
            return self._calculate_bm25_similarity(topic_key, node1, node2)
    
    def _get_subtree_depth(self, node: TreeNode) -> int:
        """计算节点子树的最大深度（相对于该节点）"""
        if not node.children:
            return 0
        max_depth = 0
        for child in node.children:
            child_depth = self._get_subtree_depth(child) + 1
            max_depth = max(max_depth, child_depth)
        return max_depth
    
    def _article_passes_node(self, current_path: List[TreeNode], target_node: TreeNode) -> bool:
        """检查文章分类路径是否经过目标节点"""
        # 检查target_node是否在current_path中，或者是current_path中某个节点的祖先
        for node in current_path:
            cur = node
            while cur:
                if cur == target_node:
                    return True
                cur = cur.parent
        return False
    
    def _check_and_trigger_merge(
        self, 
        topic_key: str, 
        current_path: List[TreeNode],
        article_idx: int, 
        is_last_article: bool,
        all_samples: List[OracleSample]
    ) -> None:
        """检查并触发待归拢操作"""
        if topic_key not in self.pending_merges:
            return
        
        merges_to_remove = []
        
        for pending in self.pending_merges[topic_key]:
            # 条件1: 文章经过待归拢集合中的任一节点
            passes_any_node = any(self._article_passes_node(current_path, node) for node in pending.nodes)
            if not passes_any_node:
                continue
            
            # 条件2: 相似度 OR 结构约束 OR 最后一篇
            parent = pending.parent
            if not parent:
                continue
            
            num_siblings = len(parent.children)
            
            # 计算集合中节点之间的平均相似度
            similarities = []
            for i, node1 in enumerate(pending.nodes):
                for node2 in pending.nodes[i+1:]:
                    sim = self._calculate_similarity(topic_key, node1, node2)
                    similarities.append(sim)
            
            avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
            
            # 计算待归拢类别中的文章数量
            total_articles = sum(len(collect_docs_in_subtree(n)) for n in pending.nodes)
            
            # 根据不同条件决定是否归拢，并记录原因
            merge_reason = None
            if avg_similarity > 0.3:  # 相似度高
                should_merge = True
                merge_reason = "SEMANTIC_HIGH"
            elif num_siblings > 20:  # 子类过多
                should_merge = True
                merge_reason = "OVERCROWDING"
            elif total_articles > 15 or is_last_article:  # 文章数量达标或最后一篇
                should_merge = True
                merge_reason = "PROGRESS"
            else:
                should_merge = False
            
            if should_merge:
                # 执行归拢，传入merge_reason
                self._execute_merge(topic_key, pending, all_samples, merge_reason)
                merges_to_remove.append(pending)
        
        # 移除已执行的归拢
        for pending in merges_to_remove:
            self.pending_merges[topic_key].remove(pending)
    
    def _execute_merge(self, topic_key: str, pending: PendingMerge, all_samples: List[OracleSample], merge_reason: str = "PROGRESS") -> None:
        """执行多节点归拢并更新相关训练样本"""
        parent = pending.parent
        if not parent:
            return
        
        # 确保所有节点还在父节点下
        valid_nodes = [n for n in pending.nodes if n in parent.children]
        if len(valid_nodes) < 2:
            return  # 至少需要2个节点才能归拢
        
        # 记录归拢前的children顺序（用于更新MERGE_WITH索引）
        children_before_merge = list(parent.children)
        
        # 创建新的父节点
        new_parent = TreeNode(summary="", citations=[], children=[])
        new_parent.depth = parent.depth + 1
        
        # 将所有待归拢的节点移到新父节点下
        for node in valid_nodes:
            parent.children.remove(node)
            new_parent.add_child(node)
        
        # 将新父节点加入原父节点
        parent.add_child(new_parent)
        
        # 更新新父节点的summary（根据updater模式）
        if self.updater_mode == "hybrid":
            # Hybrid模式：使用merge_hybrid_summaries
            child_summaries = [child.summary for child in new_parent.children if child.summary]
            parent_summary = parent.summary if parent.summary else ""
            new_parent.summary = self.updater.merge_hybrid_summaries(child_summaries, parent_summary)
        else:
            # BOW模式：使用set_bow_summary_from_children
            set_bow_summary_from_children(
                self.updater, 
                new_parent, 
                new_parent.children,
                bm25_stats=self.bm25_stats_by_topic.get(topic_key)
            )
        
        # 设置新父节点的target_label
        setattr(new_parent, "target_label", pending.target_label)
        
        # 更新所有相关样本的MERGE_WITH和MERGE_REASON字段（新格式：列表）
        for sample, node in pending.triggering_samples:
            old_completion = sample.completion
            
            # 检查是否需要更新（MERGE_WITH: NONE）
            if "MERGE_WITH: NONE" in old_completion or "MERGE_WITH: None" in old_completion:
                # 构建merge_with列表：["NEW", idx1, idx2, ...]
                if node in valid_nodes and node in children_before_merge:
                    # 找到所有需要归拢的节点的索引
                    merge_indices = []
                    for other_node in valid_nodes:
                        if other_node != node and other_node in children_before_merge:
                            idx = children_before_merge.index(other_node)
                            merge_indices.append(idx)
                    
                    if merge_indices:
                        # 构建新的merge_with列表：[NEW, idx1, idx2, ...]
                        merge_with_list = ["NEW"] + sorted(merge_indices)
                        merge_str = "[" + ", ".join(str(x) for x in merge_with_list) + "]"
                        
                        # 更新completion：添加MERGE_WITH和MERGE_REASON
                        new_completion = old_completion.replace("MERGE_WITH: NONE", f"MERGE_WITH: {merge_str}")
                        new_completion = new_completion.replace("MERGE_WITH: None", f"MERGE_WITH: {merge_str}")
                        
                        # 添加MERGE_REASON（如果不存在）
                        if "MERGE_REASON:" not in new_completion:
                            new_completion += f"\nMERGE_REASON: {merge_reason}"
                        
                        sample.completion = new_completion


    def _ensure_node_target_label(self, topic_key: str, node: TreeNode) -> Optional[str]:
        """
        为当前节点补齐其 target_label（current -> gold 的映射）。
        
        关键设计：
        target_label = 包含该节点子树所有文章的最小gold节点（LCA）
        
        具体规则：
        - 根节点（depth=0）：target_label = anc(parts, 0)，即gold树的根
        - 叶子节点：target_label = 完整gold路径（最深节点）
        - 内部节点：target_label = 子树所有文章的LCA
        """
        label = getattr(node, "target_label", None)
        if label:
            return label
        docs = collect_docs_in_subtree(node)
        if not docs:
            return None
        parts_by_id = self.article_parts_by_topic[topic_key]
        
        # 辅助函数：获取第一个path（处理parts_list格式）
        def get_first_parts(doc_id):
            """从parts_by_id获取第一个path"""
            parts_or_list = parts_by_id.get(doc_id)
            if not parts_or_list:
                return None
            # 检查是否是列表的列表
            if parts_or_list and isinstance(parts_or_list[0], (list, tuple)):
                return parts_or_list[0]  # 返回第一个path
            else:
                return parts_or_list  # 已经是单个path
        
        # 根节点特殊处理：指向gold树的根
        if node.depth == 0:
            parts = get_first_parts(docs[0])
            if not parts:
                return None
            label = anc(parts, 0)
        elif len(docs) == 1:
            # 只有一篇文章：使用完整路径（叶子节点）
            parts = get_first_parts(docs[0])
            if not parts:
                return None
            label = " - ".join(parts)
        else:
            # 多篇文章：计算LCA
            all_parts = []
            for doc_id in docs:
                parts = get_first_parts(doc_id)
                if parts:
                    all_parts.append([p.strip() for p in parts if p.strip()])
            
            if not all_parts:
                return None
            
            # 找所有文章的LCA（最深的共同前缀）
            lca_parts = []
            min_len = min(len(p) for p in all_parts)
            for i in range(min_len):
                # 检查所有文章在第i层是否相同
                first_part = all_parts[0][i]
                if all(p[i] == first_part for p in all_parts):
                    lca_parts.append(first_part)
                else:
                    break
            
            label = " - ".join(lca_parts) if lca_parts else " - ".join(all_parts[0])
        
        setattr(node, "target_label", label)
        # 维护 target -> current 的映射
        self.target_to_current[topic_key][label] = node
        return label

    def _oracle_route_multi(
        self,
        topic_key: str,
        topic_name: str,
        root: TreeNode,
        article_id: str,
        article_content: str,
        parts_list: List[Sequence[str]],
        samples: List[OracleSample],
        article_idx: int,
        is_last_article: bool,
    ):
        """
        处理一篇文章，使用所有paths。
        
        新逻辑：
        1. 使用所有paths让Oracle策略做决策
        2. Oracle可能返回多个分类结果（对应不同的paths）
        3. 每个分类结果分别递归处理
        4. 如果多个新类被创建，它们应该归拢在一起（使用第一个path对齐）
        """
        if not parts_list:
            return
        
        # 记录所有路径到article_parts_by_topic（用于其他文章的oracle判断）
        if article_id not in self.article_parts_by_topic[topic_key]:
            self.article_parts_by_topic[topic_key][article_id] = parts_list
        
        # 确保根节点有target_label
        self._ensure_node_target_label(topic_key, root)
        
        # 使用所有路径处理文章
        self._oracle_route_single(
            topic_key, topic_name, root, article_id, article_content, parts_list, 
            samples, article_idx, is_last_article
        )

    def _oracle_route_single(
        self,
        topic_key: str,
        topic_name: str,
        root: TreeNode,
        article_id: str,
        article_content: str,
        parts_list: List[Sequence[str]],  # 改为接受多个paths
        samples: List[OracleSample],
        article_idx: int,
        is_last_article: bool,
    ):
        """
        单篇文章的处理流程（延迟归拢版本，支持多paths）：
        1. 从根开始，对每个节点进行分类判断
        2. Oracle策略根据所有paths返回分类决策（可能有多个）
        3. 每个分类决策分别递归处理
        4. 如果多个新类被创建，它们应该归拢在一起（使用第一个path对齐）
        5. 创建新类时，Oracle指定归拢目标但不立即执行，记录到PendingMerge
        6. 每次分类后检查并触发归拢
        7. Bottom-up更新summary
        """
        # 创建一个转换后的parts_by_id，将列表的列表转换为只包含第一个path
        parts_by_id_raw = self.article_parts_by_topic[topic_key]
        parts_by_id = {}
        for doc_id, parts_or_list in parts_by_id_raw.items():
            if parts_or_list and isinstance(parts_or_list[0], (list, tuple)):
                parts_by_id[doc_id] = parts_or_list[0]  # 取第一个path
            else:
                parts_by_id[doc_id] = parts_or_list
        
        # 使用第一个path作为主path（用于对齐新类）
        primary_article_parts = parts_list[0] if parts_list else []
        
        # 第一阶段：Top-down分类
        # 使用队列处理多个分类路径
        from collections import deque
        queue = deque([(root, [])])  # (当前节点, 到达该节点的路径)
        
        all_route_paths = []  # 所有路径
        merged_parents = set()  # 记录归拢产生的新父节点（不应该用文章内容更新）
        
        while queue:
            cur, route_path = queue.popleft()
            self._ensure_node_target_label(topic_key, cur)
            children = cur.children
            child_summaries = [c.summary for c in children]
            
            # 计算结构特征
            child_num_children = []
            child_max_depth = []
            for child in children:
                child_num_children.append(len(child.children))
                # 计算子树深度
                max_depth = self._get_subtree_depth(child)
                child_max_depth.append(max_depth)
            
            # 生成分类prompt（带结构特征）
            classification_input = ClassificationInput(
                article_content=article_content,
                current_node_summary=cur.summary if cur.summary else topic_name,
                child_summaries=child_summaries,
                topic_name=topic_name,
                child_num_children=child_num_children,
                child_max_depth=child_max_depth,
                current_depth=cur.depth,
                num_children=len(children),
            )
            prompt = PromptTemplates.format_classification_prompt(
                topic_name=classification_input.topic_name,
                current_summary=classification_input.current_node_summary,
                article_content=classification_input.article_content,
                child_summaries=classification_input.child_summaries,
                child_num_children=classification_input.child_num_children,
                child_max_depth=classification_input.child_max_depth,
                current_depth=classification_input.current_depth,
                num_children=classification_input.num_children,
            )
            
            # 使用策略π*决定（支持多个paths）
            if len(children) > 0:
                # 有子节点：使用oracle策略判断
                # 收集所有child的target_label
                child_target_labels = []
                for child in children:
                    self._ensure_node_target_label(topic_key, child)
                    child_target_labels.append(getattr(child, "target_label", ""))
                
                # 对所有paths调用oracle策略，收集所有匹配的子节点
                all_selected_indices = set()
                for article_parts in parts_list:
                    indices = decide_top_down_child_by_target_label(
                        child_target_labels, article_parts
                    )
                    all_selected_indices.update(indices)
                
                selected_indices = sorted(list(all_selected_indices))
                need_new = (len(selected_indices) == 0)
            else:
                # 没有子节点：需要创建新类
                selected_indices = []
                need_new = True
            
            # 如果需要创建新类，判断是否归拢
            merge_with_list = []  # 改为列表，支持多个归拢目标
            if need_new:
                # 使用第一个path（primary）创建新类
                # 生成叶子summary
                leaf_sum_inp = SummaryInput(
                    node_summary="",
                    parent_summary=cur.summary if cur.summary else topic_name,
                    sibling_summaries=child_summaries,
                    new_content=article_content,
                    topic_name=topic_name,
                )
                leaf_out = self.updater.update_summary(
                    leaf_sum_inp, 
                    n_samples=1,
                    bm25_stats=self.bm25_stats_by_topic.get(topic_key)
                )[0]
                leaf_summary = leaf_out.explanation
                
                # 创建新叶子
                new_leaf = TreeNode(summary=leaf_summary, citations=[], children=[])
                cur.add_child(new_leaf)
                new_leaf.add_citation(article_id)
                
                # 设置target_label（使用第一个path对齐）
                target_label_for_leaf = " - ".join(primary_article_parts)
                setattr(new_leaf, "target_label", target_label_for_leaf)
                self.target_to_current[topic_key][target_label_for_leaf] = new_leaf
                
                # Oracle判断是否应该归拢（但不立即执行）
                # 检查所有paths，看是否有任何一个path需要归拢
                old_siblings = [c for c in cur.children if c != new_leaf]
                if len(old_siblings) > 0:
                    sibling_docs = [collect_docs_in_subtree(sib) for sib in old_siblings]
                    
                    # 对每个path检查归拢
                    merge_targets_set = set()
                    for article_parts in parts_list:
                        merge_with_idx = decide_merge_with_after_create_leaf(
                            sibling_docs_by_index=sibling_docs,
                            parent_depth_k=cur.depth,
                            article_parts=article_parts,
                            article_parts_by_id=parts_by_id,
                        )
                        if merge_with_idx is not None:
                            merge_targets_set.add(merge_with_idx)
                    
                    # 如果有归拢目标，记录到PendingMerge（不立即更新completion）
                    if merge_targets_set:
                        # 对每个归拢目标，记录到PendingMerge
                        for merge_with_idx in merge_targets_set:
                            if 0 <= merge_with_idx < len(old_siblings):
                                merge_target = old_siblings[merge_with_idx]
                                
                                # 计算归拢后的target_label（LCA）
                                new_leaf_target = getattr(new_leaf, "target_label", None)
                                merge_target_target = getattr(merge_target, "target_label", None)
                                
                                if new_leaf_target and merge_target_target:
                                    new_parts = [p.strip() for p in new_leaf_target.split(" - ") if p.strip()]
                                    merge_parts = [p.strip() for p in merge_target_target.split(" - ") if p.strip()]
                                    
                                    # 计算LCA
                                    lca_parts = []
                                    for i in range(min(len(new_parts), len(merge_parts))):
                                        if new_parts[i] == merge_parts[i]:
                                            lca_parts.append(new_parts[i])
                                        else:
                                            break
                                    
                                    if lca_parts:
                                        target_label = " - ".join(lca_parts)
                                    else:
                                        target_label = new_leaf_target
                                else:
                                    target_label = anc(primary_article_parts, cur.depth + 1)
                                
                                # 查找是否已有相同target_label的PendingMerge
                                existing_pending = None
                                for p in self.pending_merges[topic_key]:
                                    if p.parent == cur and p.target_label == target_label:
                                        existing_pending = p
                                        break
                                
                                if existing_pending:
                                    # 加入现有的待归拢集合
                                    if new_leaf not in existing_pending.nodes:
                                        existing_pending.nodes.append(new_leaf)
                                else:
                                    # 创建新的待归拢集合（初始包含新叶子和归拢目标）
                                    pending = PendingMerge(
                                        nodes=[merge_target, new_leaf],
                                        target_label=target_label,
                                        parent=cur,
                                        created_at_article_idx=article_idx,
                                        triggering_samples=[]
                                    )
                                    self.pending_merges[topic_key].append(pending)
                
                # 新叶子是终点，记录这条路径
                new_route_path = route_path + [new_leaf]
                all_route_paths.append(new_route_path)
            
            # 生成completion并记录样本（初始MERGE_WITH为NONE，延迟归拢触发时更新）
            completion = PromptTemplates.format_classification_completion(
                selected_indices=selected_indices,
                need_new=need_new,
                num_categories=len(children) if not need_new else len(children) - 1,  # 创建新类时不包括新叶子
                merge_with=None,  # 初始为NONE，延迟归拢触发时在_execute_merge中更新
            )
            current_sample = OracleSample(
                prompt=prompt,
                completion=completion,
                topic_key=topic_key,
                article_id=article_id,
                depth=cur.depth,
            )
            samples.append(current_sample)
            
            # 如果创建了新类，将这个样本加入到对应的PendingMerge的triggering_samples中
            if need_new:
                # 找到包含new_leaf的PendingMerge
                for pending in self.pending_merges.get(topic_key, []):
                    if new_leaf in pending.nodes:
                        pending.triggering_samples.append((current_sample, new_leaf))
                        break
            
            # 检查并触发归拢（在每次分类后）
            # 使用当前路径进行检查
            current_route_path = route_path + ([new_leaf] if need_new else [])
            self._check_and_trigger_merge(
                topic_key, 
                current_route_path, 
                article_idx, 
                is_last_article,
                samples
            )
            
            # 决定下一步：将所有选中的子节点加入队列
            if not need_new and len(selected_indices) > 0:
                for idx in selected_indices:
                    next_node = children[idx]
                    new_route_path = route_path + [next_node]
                    
                    # 如果是叶子节点，加入文章并记录路径
                    if len(next_node.children) == 0:
                        next_node.add_citation(article_id)
                        all_route_paths.append(new_route_path)
                    else:
                        # 非叶子节点：加入队列继续处理
                        queue.append((next_node, new_route_path))
        
        # 第二阶段：Bottom-up更新summary
        # 遍历所有路径，对每条路径从叶子到根更新
        updated_nodes = set()  # 记录已更新的节点，避免重复更新
        
        for route_path in all_route_paths:
            for i in range(len(route_path) - 1, -1, -1):
                node = route_path[i]
                
                # 如果已经更新过，跳过
                if id(node) in updated_nodes:
                    continue
                
                # 跳过归拢产生的新父节点（它们的summary已经正确设置）
                if id(node) in merged_parents:
                    continue
                
                if node.parent is not None:
                    # 如果父节点是归拢产生的，需要从子节点汇总而不是加文章内容
                    if id(node.parent) in merged_parents:
                        # 父节点需要重新汇总所有子节点
                        set_bow_summary_from_children(
                            self.updater, 
                            node.parent, 
                            node.parent.children,
                            bm25_stats=self.bm25_stats_by_topic.get(topic_key)
                        )
                    else:
                        # 正常更新：加入文章内容的词频/BM25得分
                        upd_inp = SummaryInput(
                            topic_name=topic_name,
                            node_summary=node.summary,
                            parent_summary=node.parent.summary if node.parent.summary else topic_name,
                            sibling_summaries=[sib.summary for sib in node.get_siblings()],
                            new_content=article_content,
                        )
                        out = self.updater.update_summary(
                            upd_inp, 
                            n_samples=1,
                            bm25_stats=self.bm25_stats_by_topic.get(topic_key)
                        )[0]
                        node.summary = out.explanation
                
                updated_nodes.add(id(node))

    def _validate_tree_structure(self, topic_key: str, root: TreeNode):
        """验证树结构的target_label一致性"""
        def validate_node(node: TreeNode, parent_target: Optional[str] = None):
            node_target = getattr(node, "target_label", None)
            if node_target and parent_target:
                # 子节点的target_label应该是父节点的延伸或LCA
                parent_parts = [p.strip() for p in parent_target.split(" - ") if p.strip()]
                node_parts = [p.strip() for p in node_target.split(" - ") if p.strip()]
                
                # 检查：node的前缀应该与parent一致（或parent是node的前缀）
                min_len = min(len(parent_parts), len(node_parts))
                common_prefix = []
                for i in range(min_len):
                    if parent_parts[i] == node_parts[i]:
                        common_prefix.append(parent_parts[i])
                    else:
                        break
                
                if len(common_prefix) < min(len(parent_parts), len(node_parts)):
                    print(f"\n❌ 错误：父子节点target_label不一致")
                    print(f"  Topic: {topic_key}")
                    print(f"  父节点: target_label={parent_target}, depth={node.parent.depth if node.parent else '?'}")
                    print(f"  子节点: target_label={node_target}, depth={node.depth}")
                    print(f"  共同前缀: {' - '.join(common_prefix)}")
                    return False
            
            for child in node.children:
                if not validate_node(child, node_target):
                    return False
            return True
        
        return validate_node(root)
    
    def _classify_sample_type(self, sample: OracleSample) -> str:
        """分类样本类型：merge, new, select"""
        completion = sample.completion
        # 检查是否有merge操作（且不是None）
        if 'MERGE_WITH:' in completion:
            merge_part = completion.split('MERGE_WITH:')[1].strip().split()[0]
            if merge_part.upper() != 'NONE':
                return 'merge'
        # 检查是否创建新类（格式是"NEW: Yes"）
        if 'NEW: Yes' in completion:
            return 'new'
        return 'select'
    
    def _process_single_topic(
        self,
        topic_key: str,
        ref_ids: List[str],
        max_refs_per_topic: Optional[int],
        topic_seed: int
    ) -> List[OracleSample]:
        """处理单个topic并返回生成的样本（线程安全）"""
        samples: List[OracleSample] = []
        
        try:
            if topic_key not in self.references_data:
                return samples
                
            topic_data = self.references_data[topic_key]
            topic_name = topic_data.get("topic", topic_key)

            # 使用topic专属随机数生成器
            topic_rng = random.Random(topic_seed)
            ref_ids = list(ref_ids)
            topic_rng.shuffle(ref_ids)
            if max_refs_per_topic is not None:
                ref_ids = ref_ids[:max_refs_per_topic]

            # 收集BM25统计（线程安全）
            with self.lock:
                self._collect_bm25_stats(topic_key, ref_ids)
                self._prepare_topic_article_parts(topic_key, ref_ids)
                # 初始化该topic的pending_merges列表
                self.pending_merges[topic_key] = []

            root = TreeNode(summary="", citations=[], children=[], depth=0)

            for article_idx, rid in enumerate(ref_ids):
                ref = topic_data.get("references", {}).get(rid, {})
                content = ref.get("content", "") or ""
                
                with self.lock:
                    multi = self.multi_paths_by_topic.get(topic_key, {}).get(rid) or []
                    
                if not multi:
                    continue

                is_last_article = (article_idx == len(ref_ids) - 1)
                self._oracle_route_multi(
                    topic_key, topic_name, root, rid, content, multi, 
                    samples, article_idx, is_last_article
                )
            
            with self.lock:
                self._validate_tree_structure(topic_key, root)
                
        except Exception as e:
            print(f"处理topic {topic_key} 时出错: {e}")
            import traceback
            traceback.print_exc()

        return samples
    
    def _generate_one_round(
        self, 
        split: str, 
        max_refs_per_topic: Optional[int],
        seed_offset: int = 0,
        num_workers: int = 4
    ) -> List[OracleSample]:
        """生成一轮Oracle SFT数据（多线程并行）"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # 使用新的随机种子
        round_rng = random.Random(self.seed + seed_offset)
        
        topics = list(self.dataset_split.get(split, {}).items())
        round_rng.shuffle(topics)
        
        print(f"  使用 {num_workers} 个线程并行处理 {len(topics)} 个topics")
        
        all_samples: List[OracleSample] = []
        
        # 为每个topic生成独立的seed
        topic_seeds = {topic_key: round_rng.randint(0, 1000000) for topic_key, _ in topics}
        
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # 提交所有任务
            future_to_topic = {
                executor.submit(
                    self._process_single_topic,
                    topic_key,
                    ref_ids,
                    max_refs_per_topic,
                    topic_seeds[topic_key]
                ): topic_key
                for topic_key, ref_ids in topics
            }
            
            # 收集结果
            for future in as_completed(future_to_topic):
                topic_key = future_to_topic[future]
                try:
                    samples = future.result()
                    all_samples.extend(samples)
                    print(f"    ✓ {topic_key}: {len(samples)} 样本")
                except Exception as e:
                    print(f"    ✗ {topic_key}: {e}")

        return all_samples
    
    def generate(
        self, 
        split: str = "train", 
        max_refs_per_topic: Optional[int] = None,
        target_total: int = 50000,
        target_merge_ratio: float = 0.10,
        target_new_ratio: float = 0.20,
        max_rounds: int = 20,
        num_workers: int = 10
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        生成Oracle SFT数据（多轮生成以平衡类别）
        
        Args:
            split: 数据划分
            max_refs_per_topic: 每个topic最多处理的文章数
            target_total: 目标总样本数
            target_merge_ratio: 目标归拢操作比例
            target_new_ratio: 目标创建新类比例
            max_rounds: 最多生成轮数
            
        返回：train_samples, val_samples（jsonl dict，含 prompt/completion）
        """
        print(f"\n生成Oracle SFT数据（split={split}，多轮平衡策略）")
        print(f"目标总数: {target_total}")
        print(f"目标比例 - 归拢:{target_merge_ratio:.0%}, 创建新类:{target_new_ratio:.0%}, 选择已有:{1-target_merge_ratio-target_new_ratio:.0%}")
        
        # 计算各类别目标数量
        target_merge_count = int(target_total * target_merge_ratio)
        target_new_count = int(target_total * target_new_ratio)
        target_select_count = target_total - target_merge_count - target_new_count
        
        print(f"\n目标数量设定:")
        print(f"  - 归拢目标: {target_merge_count} ({target_merge_ratio:.0%})")
        print(f"  - 创建新类目标: {target_new_count} ({target_new_ratio:.0%})")
        print(f"  - 选择已有目标: {target_select_count} ({(1-target_merge_ratio-target_new_ratio):.0%})")
        
        # 存储各类样本
        merge_samples = []
        new_samples = []
        select_samples = []
        
        # 多轮生成：只补充未达标的类别
        for round_idx in range(max_rounds):
            # 检查哪些类别还需要补充
            need_merge = len(merge_samples) < target_merge_count
            need_new = len(new_samples) < target_new_count
            need_select = len(select_samples) < target_select_count
            
            if not need_merge and not need_new and not need_select:
                print(f"\n✓ 所有类别都已达标！停止生成。")
                break
            
            print(f"\n[第 {round_idx + 1}/{max_rounds} 轮]")
            needed = []
            if need_merge:
                needed.append(f"归拢(还需{target_merge_count - len(merge_samples)})")
            if need_new:
                needed.append(f"创建新类(还需{target_new_count - len(new_samples)})")
            if need_select:
                needed.append(f"选择已有(还需{target_select_count - len(select_samples)})")
            print(f"  需要补充: {', '.join(needed)}")
            
            # 生成一轮数据（多线程）
            round_samples = self._generate_one_round(split, max_refs_per_topic, seed_offset=round_idx * 10000, num_workers=num_workers)
            
            # 分类并只保留需要的样本
            round_merge_count = 0
            round_new_count = 0
            round_select_count = 0
            
            for sample in round_samples:
                sample_type = self._classify_sample_type(sample)
                if sample_type == 'merge' and need_merge and len(merge_samples) < target_merge_count:
                    merge_samples.append(sample)
                    round_merge_count += 1
                elif sample_type == 'new' and need_new and len(new_samples) < target_new_count:
                    new_samples.append(sample)
                    round_new_count += 1
                elif sample_type == 'select' and need_select and len(select_samples) < target_select_count:
                    select_samples.append(sample)
                    round_select_count += 1
            
            print(f"  本轮贡献: 归拢+{round_merge_count}, 创建新类+{round_new_count}, 选择已有+{round_select_count}")
            print(f"  当前进度: 归拢={len(merge_samples)}/{target_merge_count} ({len(merge_samples)/target_merge_count*100:.1f}%), "
                  f"创建新类={len(new_samples)}/{target_new_count} ({len(new_samples)/target_new_count*100:.1f}%), "
                  f"选择已有={len(select_samples)}/{target_select_count} ({len(select_samples)/target_select_count*100:.1f}%)")
            
            if round_idx == max_rounds - 1:
                print(f"\n已达到最大轮数 {max_rounds}。")
        
        # 合并所有样本
        all_samples = merge_samples + new_samples + select_samples
        
        print(f"\n最终数据集统计:")
        print(f"  - 归拢操作: {len(merge_samples)} ({len(merge_samples)/len(all_samples)*100:.1f}%)")
        print(f"  - 创建新类: {len(new_samples)} ({len(new_samples)/len(all_samples)*100:.1f}%)")
        print(f"  - 选择已有: {len(select_samples)} ({len(select_samples)/len(all_samples)*100:.1f}%)")
        print(f"  - 总计: {len(all_samples)}")

        # train/val split：按样本随机划分
        self.rng.shuffle(all_samples)
        val_ratio = 0.02
        val_n = max(1, int(len(all_samples) * val_ratio))
        val = all_samples[:val_n]
        train = all_samples[val_n:]

        # 训练默认仅输出 prompt/completion
        train_out = [{"prompt": s.prompt, "completion": s.completion} for s in train]
        val_out = [{"prompt": s.prompt, "completion": s.completion} for s in val]
        return train_out, val_out


def main():
    parser = argparse.ArgumentParser(description="Oracle π* 生成分类 SFT 数据（带 BOW summary + 多轮平衡）")
    parser.add_argument("--config", type=str, default="./configs/default.json")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--output_dir", type=str, default=None, help="默认写入 config.path.data_dir")
    parser.add_argument("--bow_top_k", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_refs_per_topic", type=int, default=None)
    parser.add_argument("--val_ratio", type=float, default=0.02)
    
    # 类别平衡参数
    parser.add_argument("--target_total", type=int, default=5000, help="目标总样本数")
    parser.add_argument("--target_merge_ratio", type=float, default=0.25, help="目标归拢操作比例")
    parser.add_argument("--target_new_ratio", type=float, default=0.15, help="目标创建新类比例")
    parser.add_argument("--max_rounds", type=int, default=20, help="最多生成轮数")
    parser.add_argument("--num_workers", type=int, default=10, help="并行处理的线程数")
    
    args = parser.parse_args()

    config = SummaryBasedConfig.from_json(args.config)
    out_dir = Path(args.output_dir) if args.output_dir else Path(config.path.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 从配置文件读取updater参数
    with open(args.config, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    updater_config = config_dict.get('updater', {})
    updater_mode = updater_config.get('mode', 'bow')
    
    # 处理tokenizer_name：如果是"base_model"，使用config.path.base_model
    tokenizer_name = updater_config.get('tokenizer_name', 'Qwen/Qwen2.5-7B-Instruct')
    if tokenizer_name == 'base_model':
        tokenizer_name = config.path.base_model
    
    updater_kwargs = {
        'hybrid_keywords_top_k': updater_config.get('hybrid_keywords_top_k', 10),
        'hybrid_evidence_max_tokens': updater_config.get('hybrid_evidence_max_tokens', 200),
        'hybrid_llm_model': updater_config.get('hybrid_llm_model', 'deepseek-chat'),
        'hybrid_api_key': updater_config.get('hybrid_api_key', ''),
        'hybrid_api_url': updater_config.get('hybrid_api_url', 'https://api.deepseek.com'),
        'tokenizer_name': tokenizer_name,
    }
    
    gen = OracleSFTGenerator(
        config, 
        bow_top_k=updater_config.get('bow_top_k', args.bow_top_k), 
        seed=args.seed,
        updater_mode=updater_mode,
        **updater_kwargs
    )
    gen.load_data()

    # 生成数据（带类别平衡）
    train_samples, val_samples = gen.generate(
        split=args.split, 
        max_refs_per_topic=args.max_refs_per_topic,
        target_total=args.target_total,
        target_merge_ratio=args.target_merge_ratio,
        target_new_ratio=args.target_new_ratio,
        max_rounds=args.max_rounds,
        num_workers=args.num_workers
    )
    if args.val_ratio != 0.02:
        # 重新按用户 ratio 切分（不重新模拟）
        rng = random.Random(args.seed)
        rng.shuffle(train_samples)
        all_samples = train_samples + val_samples
        rng.shuffle(all_samples)
        val_n = max(1, int(len(all_samples) * args.val_ratio))
        val_samples = all_samples[:val_n]
        train_samples = all_samples[val_n:]

    train_file = out_dir / "classify_generator_oracle_train.jsonl"
    val_file = out_dir / "classify_generator_oracle_val.jsonl"

    with train_file.open("w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with val_file.open("w", encoding="utf-8") as f:
        for s in val_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print("Oracle SFT 数据生成完成：")
    print(f"  - train: {train_file}")
    print(f"  - val:   {val_file}")


if __name__ == "__main__":
    main()
