"""
Oracle 策略 π*（来自 `符号化说明与greedy有效性说明.md` 第 2 章与 5.2.2）

核心思想：
- Top-down：在当前节点 v 处，若存在某个子树 y 可容纳 d_t（不会违反目标簇一致性），则进入 y；否则 CreateLeaf。
- 若发生 CreateLeaf@v：允许最多一次 InsertParentPath，把新叶 x 与一个满足目标第 (k+1) 层同簇的兄弟子树 y 收拢。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence


@dataclass(frozen=True)
class OracleDecision:
    selected_indices: List[int]
    need_new: bool
    merge_with: Optional[int] = None  # 当 need_new=True 时，选择与哪个兄弟归拢；None 表示不归拢


def parse_gold_path(path_str: str) -> List[str]:
    """
    gold path 形如: "Topic - A - a1"
    返回按层级分割后的 parts: ["Topic","A","a1"]
    """
    return [p.strip() for p in path_str.split(" - ") if p.strip()]


def anc(parts: Sequence[str], k: int) -> str:
    """
    anc_k^*(d)：文章 d 在目标树 T* 上深度为 k 的祖先（用 path 前缀字符串表示）。
    约定：根(Topic)深度为 0。
    若 k 超过叶子深度，则 clamp 到叶子（返回完整 path）。
    """
    if not parts:
        return ""
    kk = max(0, int(k))
    kk = min(kk, len(parts) - 1)
    return " - ".join(parts[: kk + 1])


def decide_top_down_child(
    child_docs_by_index: List[List[str]],
    child_depths: List[int],
    article_parts_list: List[Sequence[str]],
    article_parts_by_id: Dict[str, Sequence[str]],
) -> List[int]:
    """
    在当前节点 v 的子节点中选择要进入的子节点 index（支持多路径）。

    判断规则：对每个 child y，若 y 子树中所有历史文章 d 都满足：
    它们的gold path与新文章的某条gold path有共同前缀（至少到 depth(y)+1 层）
    且该共同前缀包含 depth(y)+1 层（即它们在child这一层确实应该在一起）

    关键：文章A-B-C不应该被分到对应A-B-D的节点，因为虽然它们共享A-B，
    但在B的下一层（C vs D）就分开了。

    返回：所有满足的 index 列表；若为空则表示需要 CreateLeaf@v。
    """
    target_indices_set = set()
    
    for article_parts in article_parts_list:
        for idx, docs in enumerate(child_docs_by_index):
            if not docs:
                continue
            depth_y = child_depths[idx]
            
            # 新文章在 depth_y+1 层的祖先
            article_prefix = anc(article_parts, depth_y + 1)
            
            # 检查：child子树中所有历史文章是否都与新文章在 depth_y+1 层一致
            # 并且它们的完整path应该是"兼容"的（共享到至少 depth_y+1）
            ok = True
            for did in docs:
                parts_d = article_parts_by_id.get(did)
                if parts_d is None:
                    ok = False
                    break
                
                # 历史文章在 depth_y+1 层的祖先
                doc_prefix = anc(parts_d, depth_y + 1)
                
                # 必须在 depth_y+1 层完全一致
                if doc_prefix != article_prefix:
                    ok = False
                    break
                
                # 额外检查：如果文章和历史文档的完整路径长度都大于 depth_y+1
                # 它们应该继续兼容（即在下一层也应该相同，或者至少一个在 depth_y+1 就结束了）
                # 实际上，如果它们在 depth_y+1 相同，且都有更深的层级，
                # 那么它们应该继续递归下去，所以这里只判断 depth_y+1 即可
            
            if ok:
                target_indices_set.add(idx)
    
    return sorted(target_indices_set)


def decide_top_down_child_by_target_label(
    child_target_labels: List[str],
    article_parts: Sequence[str],
) -> List[int]:
    """
    基于target_label直接判断文章应该进入哪些子节点。
    
    判断规则：child的target_label必须是article_parts的前缀。
    
    例如：
    - 文章path: Root-A-B-C
    - child1 target: Root-A → 匹配（是前缀）
    - child2 target: Root-A-B → 匹配（是前缀）
    - child3 target: Root-A-B-C → 匹配（完全相同）
    - child4 target: Root-A-B-D → 不匹配（D != C）
    - child5 target: Root-A-D → 不匹配（第三层D != B）
    
    返回：所有匹配的child索引列表
    """
    selected_indices = []
    article_parts_list = [p.strip() for p in article_parts if p.strip()]
    
    for idx, target_label in enumerate(child_target_labels):
        if not target_label:
            continue
        
        target_parts = [p.strip() for p in target_label.split(" - ") if p.strip()]
        
        # 判断：target_parts必须是article_parts的前缀
        if len(target_parts) <= len(article_parts_list):
            is_prefix = all(
                target_parts[i] == article_parts_list[i]
                for i in range(len(target_parts))
            )
            if is_prefix:
                selected_indices.append(idx)
    
    return selected_indices


def decide_merge_with_after_create_leaf(
    sibling_docs_by_index: List[List[str]],
    parent_depth_k: int,
    article_parts: Sequence[str],
    article_parts_by_id: Dict[str, Sequence[str]],
) -> Optional[int]:
    """
    π* 的 InsertParentPath 选择规则（5.2.2）：
    若本步发生 CreateLeaf@v，k=d_T(v)，在兄弟集合中寻找唯一 y，使得
        ∀d∈Docs_T(y), anc_{k+1}^*(d) = anc_{k+1}^*(d_t)
    
    注意：用 k+1（父节点的下一层），因为InsertParentPath是在父节点层面的操作，
    判断的是"在父节点下一层，两个叶子是否应该属于同一gold子树"。
    
    例如：
    - parent depth=0 (root)
    - 叶子1: Root-A-B-C，target_label=Root-A-B (depth+1=2)
    - 叶子2: Root-A-D，target_label=Root-A-D (depth+1=2)
    - 比较 k+1=1 层：都是 Root-A，应该归拢
    
    若存在则返回 y 的 index，否则返回 None。
    """
    tgt = anc(article_parts, parent_depth_k + 1)
    candidates: List[int] = []

    for idx, docs in enumerate(sibling_docs_by_index):
        if not docs:
            continue
        ok = True
        for did in docs:
            parts_d = article_parts_by_id.get(did)
            if parts_d is None:
                ok = False
                break
            if anc(parts_d, parent_depth_k + 1) != tgt:
                ok = False
                break
        if ok:
            candidates.append(idx)

    if not candidates:
        return None
    # 如果有多个满足条件的sibling，选择index最大的（最后创建的）
    # 这样能保证稳定性，且倾向于与最新的相关类归拢
    return candidates[-1]  # 取最后一个（index最大）

