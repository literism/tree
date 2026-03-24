"""
轨迹采样器
实现Top-1 vs Rest的轨迹采样策略
"""
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field
from copy import deepcopy
import random


@dataclass(eq=False)  # 禁用自动生成的__eq__
class TreeNode:
    """树节点"""
    summary: str
    citations: List[str] = field(default_factory=list)
    children: List['TreeNode'] = field(default_factory=list)
    parent: Optional['TreeNode'] = None
    depth: int = 0
    node_id: Optional[str] = None  # 唯一ID，延迟初始化
    
    def __post_init__(self):
        """初始化后设置node_id为对象自身的id"""
        if self.node_id is None:
            self.node_id = str(id(self))
    
    def __eq__(self, other):
        """自定义相等比较，只比较node_id，避免递归"""
        if not isinstance(other, TreeNode):
            return False
        return self.node_id == other.node_id
    
    def __hash__(self):
        """使节点可以作为dict的key"""
        return hash(self.node_id)
    
    def add_child(self, child: 'TreeNode'):
        """添加子节点"""
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)
    
    def add_citation(self, article_id: str):
        """添加引用"""
        if article_id not in self.citations:
            self.citations.append(article_id)
    
    def get_siblings(self) -> List['TreeNode']:
        """获取兄弟节点"""
        if self.parent is None:
            return []
        return [child for child in self.parent.children if child != self]
    
    def clone(self) -> 'TreeNode':
        """深拷贝节点"""
        return deepcopy(self)


@dataclass
class Action:
    """动作"""
    action_type: str  # 'classify' or 'update_summary' or 'generate_new_summary'
    system: str  # 'classify_generator' or 'updater'
    node: TreeNode
    prompt: str  # 系统输入prompt
    completion: str  # 系统原始输出
    
    # 分类系统的输出
    selected_indices: List[int] = field(default_factory=list)
    need_new: bool = False
    merge_with: Optional[int] = None  # 当need_new=True时的归拢对象（InsertParentPath）；None表示不归拢
    
    # 总结系统的输出
    needs_update: bool = False
    updated_summary: Optional[str] = None
    explanation: Optional[str] = None
    scope: Optional[str] = None


@dataclass
class State:
    """状态"""
    tree: TreeNode  # 当前结构树
    current_node: TreeNode  # 当前节点
    article_id: str
    article_content: str


@dataclass
class Trajectory:
    """轨迹"""
    states: List[State] = field(default_factory=list)
    actions: List[Action] = field(default_factory=list)
    final_tree: TreeNode = None
    
    def add_step(self, state: State, action: Action):
        """添加一步"""
        self.states.append(state)
        self.actions.append(action)


class TrajectorySampler:
    """轨迹采样器"""
    
    def __init__(
        self,
        classifier,
        updater,
        topic_name: str,
        initial_tree: TreeNode,
        max_depth: int = 10,
        num_samples: int = 4,
        top_k: int = 1,
        temperature: float = 0.7
    ):
        """
        Args:
            classifier: 分类系统
            updater: 总结系统
            topic_name: topic名称
            initial_tree: 初始树
            max_depth: 最大深度
            num_samples: 每次采样的数量（用于Top-1 vs Rest）
            top_k: Top-k（通常为1）
            temperature: 采样温度
        """
        self.classifier = classifier
        self.updater = updater
        self.topic_name = topic_name
        self.initial_tree = initial_tree
        self.max_depth = max_depth
        self.num_samples = num_samples
        self.top_k = top_k
        self.temperature = temperature
    
    def sample_trajectories(
        self,
        article_id: str,
        article_content: str
    ) -> List[Trajectory]:
        """
        采样轨迹（Top-1 vs Rest）
        
        Args:
            article_id: 文章ID
            article_content: 文章内容
            
        Returns:
            轨迹列表（4条：1条best + 3条others）
        """
        # 初始化状态
        initial_state = State(
            tree=self.initial_tree.clone(),
            current_node=self.initial_tree.clone(),
            article_id=article_id,
            article_content=article_content
        )
        
        # 采样多条轨迹
        trajectories = []
        
        for _ in range(self.num_samples):
            trajectory = self._sample_single_trajectory(initial_state)
            trajectories.append(trajectory)
        
        return trajectories
    
    def _sample_single_trajectory(self, initial_state: State) -> Trajectory:
        """采样单条轨迹"""
        trajectory = Trajectory()
        current_state = initial_state
        
        # Top-down classification
        while current_state.current_node.depth < self.max_depth:
            # 分类决策
            children = current_state.current_node.children
            child_summaries = [child.summary for child in children]
            
            # 调用分类系统（采样）
            from summary_based_classifier.llm.classify_generator import ClassificationInput
            classification_input = ClassificationInput(
                article_content=current_state.article_content,
                current_node_summary=current_state.current_node.summary or self.topic_name,
                child_summaries=child_summaries,
                topic_name=self.topic_name
            )
            
            outputs = self.classifier.classify_with_sampling(
                classification_input,
                n=1
            )
            
            if not outputs:
                break
            
            output = outputs[0]
            
            # 创建action
            action = Action(
                action_type='classify',
                node=current_state.current_node,
                selected_indices=output.selected_indices,
                need_new=output.need_new,
                prompt="",  # TODO: 保存prompt
                completion=output.raw_response
            )
            
            trajectory.add_step(current_state, action)
            
            # 判断是否需要创建新类
            if output.need_new:
                # 生成新summary
                from summary_based_classifier.llm.updater import SummaryInput
                updater_input = SummaryInput(
                    topic_name=self.topic_name,
                    node_summary="",
                    parent_summary=current_state.current_node.summary or self.topic_name,
                    sibling_summaries=child_summaries,
                    new_content=current_state.article_content
                )
                
                updater_outputs = self.updater.update_summary(updater_input, n_samples=1)
                
                if updater_outputs and (updater_outputs[0].explanation and updater_outputs[0].scope):
                    output_summary = updater_outputs[0]
                    new_summary = f"EXPLANATION: {output_summary.explanation}\nSCOPE: {output_summary.scope}"
                    
                    # 创建新节点
                    new_node = TreeNode(
                        summary=new_summary,
                        citations=[],
                        children=[]
                    )
                    current_state.current_node.add_child(new_node)
                    
                    # 转移到新节点
                    next_node = new_node
                else:
                    # 生成失败，停留在当前节点
                    current_state.current_node.add_citation(current_state.article_id)
                    break
            else:
                # 选择已有节点
                if output.selected_indices and len(children) > 0:
                    idx = output.selected_indices[0]
                    if 0 <= idx < len(children):
                        next_node = children[idx]
                    else:
                        current_state.current_node.add_citation(current_state.article_id)
                        break
                else:
                    # 叶子节点
                    current_state.current_node.add_citation(current_state.article_id)
                    break
            
            # 更新状态
            current_state = State(
                tree=current_state.tree,
                current_node=next_node,
                article_id=current_state.article_id,
                article_content=current_state.article_content
            )
            
            # 检查是否到达叶子
            if len(next_node.children) == 0 and not output.need_new:
                next_node.add_citation(current_state.article_id)
                break
        
        # Bottom-up summary update（简化版，不采样）
        # TODO: 如果需要，可以在这里添加summary update的采样
        
        trajectory.final_tree = current_state.tree
        return trajectory


def tree_to_dict(node: TreeNode, level: int = 1) -> Dict:
    """将树转换为字典"""
    return {
        'level': level,
        'summary': node.summary,
        'citations': node.citations,
        'children': [tree_to_dict(child, level + 1) for child in node.children]
    }
