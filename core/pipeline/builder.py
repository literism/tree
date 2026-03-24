"""
新的构建系统（简化版分叉逻辑）
主线：完整执行，记录所有决策点和采样结果
支线：对每个决策点的其他结果，从该点继续完成剩余过程（不再分叉）
"""
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from summary_based_classifier.core.trajectory.trajectory_sampler import TreeNode, Trajectory, Action
from summary_based_classifier.llm.classify_generator import ClassifyGenerator, ClassificationInput, ClassificationOutput
from summary_based_classifier.llm.updater import Updater, SummaryInput, SummaryOutput


@dataclass
class DecisionPoint:
    """决策点：记录一个模型调用的所有采样结果"""
    decision_type: str  # 'classify', 'generate_new', 'update'
    # 状态信息
    current_node: TreeNode  # 决策发生的节点
    path: List[TreeNode]  # 当前路径
    article_id: str
    article_content: str
    # 模型信息
    prompt: str
    all_outputs: List  # 所有采样结果（ClassificationOutput或SummaryOutput）
    chosen_index: int = 0  # 主线选择的索引
    # 恢复状态用
    tree_snapshot: TreeNode = None  # 决策前的树状态快照
    actions_so_far: List[Action] = field(default_factory=list)  # 此决策前的所有actions
    # 额外信息
    child_summaries: List[str] = field(default_factory=list)  # 分类时的child_summaries
    parent_summary: str = ""  # 更新时的parent_summary
    sibling_summaries: List[str] = field(default_factory=list)  # 更新时的sibling_summaries


class TreeBuilder:
    """结构树构建器"""
    
    def __init__(
        self,
        classifier: ClassifyGenerator,
        updater: Updater,
        topic_name: str,
        max_depth: int = 10,
        bm25_stats: Optional[Dict] = None
    ):
        """
        Args:
            classifier: 分类系统
            updater: 总结系统
            topic_name: topic名称
            max_depth: 最大深度
            bm25_stats: BM25统计信息 {'df': Dict[str, int], 'total_docs': int, 'avg_doc_length': float}
        """
        self.classifier = classifier
        self.updater = updater
        self.topic_name = topic_name
        self.max_depth = max_depth
        self.bm25_stats = bm25_stats
        
        # Worker队列支持（用于并行推理）
        self.use_workers = False
        self.classifier_prompt_queue = None
        self.classifier_result_queue = None
        self.updater_prompt_queue = None
        self.updater_result_queue = None
        self._prompt_counter = 0
        # 共享result_queue时，按prompt_id缓存错取结果，避免丢包
        self._pending_classifier_results = {}
        self._pending_updater_results = {}
        
        # 文章内容缓存（用于重新路由）
        self.articles_cache = {}
    
    # ==================== 结构操作：InsertParentPath ====================

    @staticmethod
    def _get_root(node: TreeNode) -> TreeNode:
        cur = node
        while cur.parent is not None:
            cur = cur.parent
        return cur

    @staticmethod
    def _recompute_depths(node: TreeNode, depth: int = 0):
        """递归刷新 depth 字段，避免结构调整后 depth 不一致。"""
        node.depth = depth
        for c in node.children:
            TreeBuilder._recompute_depths(c, depth + 1)

    def _insert_parent_path(self, parent: TreeNode, new_leaf: TreeNode, sibling: TreeNode) -> TreeNode:
        """
        InsertParentPath（受限版本）：
        - 只允许把“新创建的 leaf(new_leaf)”与“某个现存兄弟(sibling)”归拢到一个新父节点下
        - 不允许任意子树归拢/合并/拆分
        - 不回溯历史文章绑定
        """
        if new_leaf.parent != parent:
            raise ValueError("InsertParentPath: new_leaf 必须是 parent 的直接子节点")
        if sibling.parent != parent:
            raise ValueError("InsertParentPath: sibling 必须是 parent 的直接子节点")
        if new_leaf == sibling:
            raise ValueError("InsertParentPath: sibling 不能等于 new_leaf")

        # 从 parent.children 移除两者，其余保持顺序
        removed = {new_leaf, sibling}
        parent.children = [c for c in parent.children if c not in removed]

        # 创建新父节点（summary 留空，后续 bottom-up 更新）
        new_parent = TreeNode(summary="", citations=[], children=[])
        parent.add_child(new_parent)

        # new_parent 下挂两个孩子（稳定顺序）
        new_parent.add_child(sibling)
        new_parent.add_child(new_leaf)

        # 递归刷新 depth
        root = self._get_root(parent)
        self._recompute_depths(root, 0)
        return new_parent
    
    def _call_classifier(
        self,
        classification_input: ClassificationInput,
        n: int = 1
    ) -> List[ClassificationOutput]:
        """调用分类器（支持Worker队列或直接调用）"""
        if self.use_workers:
            # 使用Worker队列
            from summary_based_classifier.models.model_workers import PromptRequest
            import uuid
            
            prompt_id = str(uuid.uuid4())
            prompt = self.classifier.create_prompt(classification_input)
            
            # 发送请求
            self.classifier_prompt_queue.put(PromptRequest(
                prompt_id=prompt_id,
                prompt=prompt
            ))

            cached = self._pending_classifier_results.pop(prompt_id, None)
            if cached is not None:
                return cached.result[:n] if cached.result else []
            
            # 等待结果
            while True:
                try:
                    result = self.classifier_result_queue.get(timeout=30)
                    if result.prompt_id == prompt_id:
                        # result.result 已经是 ClassificationOutput 对象列表
                        return result.result[:n]
                    # 非本请求结果，缓存供对应调用读取
                    self._pending_classifier_results[result.prompt_id] = result
                except:
                    return []
        else:
            # 直接调用
            return self.classifier.classify_with_sampling(classification_input, n=n)
    
    def _call_updater(
        self,
        summary_input: SummaryInput,
        n: int = 1
    ) -> List[SummaryOutput]:
        """调用更新器（支持Worker队列或直接调用）"""
        if self.use_workers:
            # 使用Worker队列
            from summary_based_classifier.models.model_workers import PromptRequest
            import uuid
            
            prompt_id = str(uuid.uuid4())
            prompt = self.updater.create_prompt(summary_input)
            
            # 发送请求
            self.updater_prompt_queue.put(PromptRequest(
                prompt_id=prompt_id,
                prompt=prompt
            ))

            cached = self._pending_updater_results.pop(prompt_id, None)
            if cached is not None:
                return cached.result[:n] if cached.result else []
            
            # 等待结果
            while True:
                try:
                    result = self.updater_result_queue.get(timeout=30)
                    if result.prompt_id == prompt_id:
                        # result.result 已经是 SummaryOutput 对象列表
                        return result.result[:n]
                    # 非本请求结果，缓存供对应调用读取
                    self._pending_updater_results[result.prompt_id] = result
                except:
                    return []
        else:
            # 直接调用
            return self.updater.update_summary_with_sampling(summary_input, n=n)
    
    def classify_and_update(
        self,
        article_id: str,
        article_content: str,
        root: TreeNode
    ) -> List[List[TreeNode]]:
        """
        对一篇文章进行分类和更新（不采样，每次只用一个结果）
        
        Args:
            article_id: 文章ID
            article_content: 文章内容
            root: 根节点
            
        Returns:
            所有路径的列表（一篇文章可能被分到多个类别）
        """
        # 缓存文章内容（用于可能的重新路由）
        self.articles_cache[article_id] = article_content
        
        all_paths = []
        
        # Top-down分类（递归）
        self._classify_recursive(root, article_id, article_content, [root], all_paths)
        
        # Bottom-up更新（对每条路径）
        for path in all_paths:
            self._update_recursive(path, article_content)
        
        return all_paths
    
    def classify_and_update_with_sampling(
        self,
        article_id: str,
        article_content: str,
        root: TreeNode,
        sampling_num: int = 8,
        top_k: int = 4
    ) -> List[Trajectory]:
        """
        对一篇文章进行分类和更新，支持多结果采样产生多条轨迹
        
        新的简化方案：
        1. 主线：完整执行构建过程，每次采样sampling_num个结果，选top_k个，只用第一个继续，但记录所有决策点
        2. 支线：对每个决策点的"其他结果"（2到top_k），从该点继续完成剩余过程（不再分叉）
        
        Args:
            article_id: 文章ID
            article_content: 文章内容
            root: 根节点
            sampling_num: 每次采样的结果数
            top_k: 从采样结果中选择前k个
            
        Returns:
            轨迹列表
        """
        # 缓存文章内容（用于可能的重新路由）
        self.articles_cache[article_id] = article_content
        
        # 阶段1: 运行主线，记录所有决策点
        main_tree = root.clone()
        main_actions = []
        decision_points = []
        all_paths = []
        
        # Top-down分类阶段（主线）
        self._classify_main_trajectory(
            main_tree, article_id, article_content, [main_tree],
            all_paths, main_actions, decision_points, sampling_num, top_k, root_tree=main_tree
        )
        
        # Bottom-up更新阶段（主线）
        for path in all_paths:
            self._update_main_trajectory(
                path, article_content, main_actions, decision_points, sampling_num, top_k, main_tree
            )
        
        # 主线轨迹
        main_trajectory = Trajectory(
            actions=main_actions,
            final_tree=main_tree
        )
        trajectories = [main_trajectory]
        
        # 阶段2: 为每个决策点的"其他结果"生成支线
        for decision_point in decision_points:
            # 跳过第一个结果（已经是主线了）
            for alt_idx in range(1, len(decision_point.all_outputs)):
                branch_trajectory = self._create_branch_trajectory(
                    decision_point, alt_idx, article_id, article_content, all_paths
                )
                if branch_trajectory:
                    trajectories.append(branch_trajectory)
        
        return trajectories
    
    def _classify_recursive(
        self,
        current_node: TreeNode,
        article_id: str,
        article_content: str,
        path: List[TreeNode],
        all_paths: List[List[TreeNode]]
    ):
        """递归分类（不采样版本，每次只用一个结果）"""
            # 检查深度
        if current_node.depth >= self.max_depth:
            current_node.add_citation(article_id)
            all_paths.append(path.copy())
            return
            
        children = current_node.children
        child_summaries = [c.summary for c in children]
            
        # 计算结构特征
        child_num_children, child_max_depth = self._get_child_structure_features(children)
            
        # 调用分类系统（带结构特征）
        classification_input = ClassificationInput(
            article_content=article_content,
            current_node_summary=current_node.summary if current_node.summary else self.topic_name,
            child_summaries=child_summaries,
            topic_name=self.topic_name,
            child_num_children=child_num_children,
            child_max_depth=child_max_depth,
            current_depth=current_node.depth,
            num_children=len(children)
        )
        
        outputs = self._call_classifier(classification_input, n=1)
        
        if not outputs:
            current_node.add_citation(article_id)
            all_paths.append(path.copy())
            return
        
        action = outputs[0]
        
        # 处理Select+Merge场景：分类到已有类别后可能归拢
        if not action.need_new and action.merge_with is not None and len(action.selected_indices) > 0:
            # 获取选中的节点和归拢目标
            if 0 <= action.selected_indices[0] < len(children) and 0 <= action.merge_with < len(children):
                if action.selected_indices[0] != action.merge_with:
                    selected_node = children[action.selected_indices[0]]
                    merge_target = children[action.merge_with]
                    # 执行归拢
                    inserted_parent = self._insert_parent_path(current_node, selected_node, merge_target)
                    # 更新新父节点的summary
                    self._update_parent_based_on_children(inserted_parent, [selected_node, merge_target])
                    # 继续分类到selected_node
                    new_path = path + [inserted_parent, selected_node]
                    if len(selected_node.children) == 0:
                        selected_node.add_citation(article_id)
                        all_paths.append(new_path)
                    else:
                        self._classify_recursive(selected_node, article_id, article_content, new_path, all_paths)
                    return
        
        # 处理选中的已有类别（可能多个）
        for idx in action.selected_indices:
            if 0 <= idx < len(children):
                next_node = children[idx]
                new_path = path + [next_node]
                
                # 如果是叶子节点，添加文章引用并记录路径
                if len(next_node.children) == 0:
                    next_node.add_citation(article_id)
                    all_paths.append(new_path)
                else:
                    # 否则继续递归
                    self._classify_recursive(next_node, article_id, article_content, new_path, all_paths)
        
        # 处理NEW
        if action.need_new:
            # 创建新类别
            new_node = self._create_new_category(current_node, article_content, child_summaries)
            if new_node:
                # 新叶子添加文章引用
                new_node.add_citation(article_id)
                
                # 处理merge_with（InsertParentPath）
                if action.merge_with is not None:
                    # 获取旧的兄弟节点（不包括新创建的节点）
                    old_siblings = [c for c in current_node.children if c != new_node]
                    
                    if 0 <= action.merge_with < len(old_siblings):
                        sibling = old_siblings[action.merge_with]
                        # 执行InsertParentPath
                        inserted_parent = self._insert_parent_path(current_node, new_node, sibling)
                        
                        # Bottom-up更新新插入的父节点
                        # 使用子节点的summary来生成父节点的summary
                        self._update_parent_based_on_children(inserted_parent, [new_node, sibling])
                        
                        # 路径包含新插入的父节点
                        new_path = path + [inserted_parent, new_node]
                else:
                    new_path = path + [new_node]
                
                # 创建新叶子后流程结束，记录路径
                all_paths.append(new_path)
                return  # 重要：创建新叶子后立即返回，不继续递归
        
        # 如果没有任何选择
        if not action.selected_indices and not action.need_new:
            current_node.add_citation(article_id)
            all_paths.append(path.copy())

    def _create_new_category(
        self,
        parent_node: TreeNode,
        article_content: str,
        sibling_summaries: List[str]
    ) -> Optional[TreeNode]:
        """创建新类别（不采样版本）"""
        updater_input = SummaryInput(
            topic_name=self.topic_name,
            node_summary="",
            parent_summary=parent_node.summary if parent_node.summary else self.topic_name,
            sibling_summaries=sibling_summaries,
            new_content=article_content
        )
        
        # 根据是否使用Worker选择调用方式
        if self.use_workers:
            outputs = self._call_updater(updater_input, n=1)
        else:
            outputs = self.updater.update_summary(updater_input, n_samples=1, bm25_stats=self.bm25_stats)
        
        for output in outputs:
            # BOW模式：explanation是BOW JSON字符串，scope为空
            # Model模式：explanation和scope都是字符串
            if output.explanation:
                if self.updater.mode == 'bow':
                    # BOW模式：直接使用explanation作为summary（它已经是BOW JSON字符串）
                    new_summary = output.explanation
                else:
                    # Model模式：格式化为EXPLANATION/SCOPE格式
                    if output.scope:
                new_summary = f"EXPLANATION: {output.explanation}\nSCOPE: {output.scope}"
                    else:
                        # 如果没有scope，只用explanation
                        new_summary = output.explanation
                
                new_node = TreeNode(summary=new_summary, citations=[], children=[])
                parent_node.add_child(new_node)
                
                # 如果父节点是叶子，需要重新路由已有文章（仅推理时）
                if len(parent_node.children) == 1 and len(parent_node.citations) > 0:
                    new_children = self._reroute_articles(parent_node, article_content)
                    
                    # 如果重新路由创建了新的子节点，需要根据所有子节点更新父节点
                    # 包括触发重新路由的那个节点 (new_node) 和重新路由新创建的节点 (new_children)
                    if new_children:
                        all_new_children = [new_node] + new_children
                        self._update_parent_based_on_children(parent_node, all_new_children)
                
                return new_node
        
        return None
    
    def _update_recursive(
        self,
        path: List[TreeNode],
        article_content: str
    ):
        """Bottom-up更新（不采样版本）"""
        from summary_based_classifier.llm.prompts import PromptTemplates
        
        for node in reversed(path):
            if node.parent is None:
                break
            
            updater_input = SummaryInput(
                topic_name=self.topic_name,
                node_summary=node.summary,
                parent_summary=node.parent.summary if node.parent.summary else self.topic_name,
                sibling_summaries=[sib.summary for sib in node.get_siblings()],
                new_content=article_content
            )
            
            # 根据是否使用Worker选择调用方式
            if self.use_workers:
                outputs = self._call_updater(updater_input, n=1)
            else:
                outputs = self.updater.update_summary(updater_input, n_samples=1, bm25_stats=self.bm25_stats)
            
            for output in outputs:
                if output.needs_update and output.explanation:
                    # BOW模式：explanation是BOW JSON字符串
                    # Model模式：explanation和scope都是字符串
                    if self.updater.mode == 'bow':
                        node.summary = output.explanation
                    else:
                        if output.scope:
                    node.summary = f"EXPLANATION: {output.explanation}\nSCOPE: {output.scope}"
                        else:
                            node.summary = output.explanation
                    break
                else:
                    break
    
    def _update_parent_based_on_children(
        self,
        parent_node: TreeNode,
        new_children: List[TreeNode]
    ):
        """
        根据新子节点逐个更新父节点的summary
        
        当一个节点下创建了多个新子节点后（通过重新路由），父节点需要根据这些
        新子节点来更新自己的summary。每个新子节点作为"new_content"逐个输入。
        
        Args:
            parent_node: 需要更新的父节点
            new_children: 新创建的子节点列表
        """
        if not new_children:
            return
        
        # 逐个将新子节点作为"new_content"输入，判断是否需要更新父节点
        for new_child in new_children:
            updater_input = SummaryInput(
                topic_name=self.topic_name,
                node_summary=parent_node.summary,
                parent_summary=parent_node.parent.summary if parent_node.parent and parent_node.parent.summary else self.topic_name,
                sibling_summaries=[sib.summary for sib in parent_node.get_siblings()] if parent_node.parent else [],
                new_content=new_child.summary  # 使用子节点的summary作为新内容
            )
            
            # 根据是否使用Worker选择调用方式
            if self.use_workers:
                outputs = self._call_updater(updater_input, n=1)
            else:
                outputs = self.updater.update_summary(updater_input, n_samples=1, bm25_stats=self.bm25_stats)
            
            for output in outputs:
                if output.needs_update and output.explanation:
                    # BOW模式：explanation是BOW JSON字符串
                    # Model模式：explanation和scope都是字符串
                    if self.updater.mode == 'bow':
                        parent_node.summary = output.explanation
                    else:
                        if output.scope:
                    parent_node.summary = f"EXPLANATION: {output.explanation}\nSCOPE: {output.scope}"
                        else:
                            parent_node.summary = output.explanation
                    # 更新后继续处理下一个子节点
                break
    
    def _classify_main_trajectory(
        self,
        current_node: TreeNode,
        article_id: str,
        article_content: str,
        path: List[TreeNode],
        all_paths: List[List[TreeNode]],
        main_actions: List[Action],
        decision_points: List[DecisionPoint],
        sampling_num: int,
        top_k: int,
        root_tree: TreeNode  # 新增参数：根节点
    ):
        """主线的递归分类（采样multiple但只用第一个，记录所有决策点）"""
        # 检查深度
        if current_node.depth >= self.max_depth:
            current_node.add_citation(article_id)
            all_paths.append(path.copy())
            return
        
        children = current_node.children
        child_summaries = [c.summary for c in children]
        
        # 计算结构特征
        child_num_children, child_max_depth = self._get_child_structure_features(children)
        
        # 调用分类系统（采样sampling_num个，带结构特征）
        classification_input = ClassificationInput(
            article_content=article_content,
            current_node_summary=current_node.summary if current_node.summary else self.topic_name,
            child_summaries=child_summaries,
            topic_name=self.topic_name,
            child_num_children=child_num_children,
            child_max_depth=child_max_depth,
            current_depth=current_node.depth,
            num_children=len(children)
        )
        
        prompt = self.classifier.create_prompt(classification_input) if not self.use_workers else ""
        outputs = self._call_classifier(classification_input, n=sampling_num)
        
        if not outputs:
            current_node.add_citation(article_id)
            all_paths.append(path.copy())
            return
        
        # 过滤并选择top_k
        unique_outputs = self._filter_unique_classify_outputs(outputs)
        top_outputs = unique_outputs[:top_k]
        
        if not top_outputs:
            current_node.add_citation(article_id)
            all_paths.append(path.copy())
            return
        
        # 记录决策点（如果有多个结果）
        if len(top_outputs) > 1:
            decision_points.append(DecisionPoint(
                decision_type='classify',
                current_node=current_node,
                path=path.copy(),
                article_id=article_id,
                article_content=article_content,
                prompt=prompt,
                all_outputs=top_outputs,
                chosen_index=0,
                tree_snapshot=root_tree.clone(),  # 保存根节点而不是当前节点
                actions_so_far=main_actions.copy(),
                child_summaries=child_summaries
            ))
        
        # 使用第一个结果继续主线
        first_output = top_outputs[0]
        
        # 记录action
        action = Action(
            action_type='classify',
            system='classify_generator',
            node=current_node,
            prompt=prompt,
            completion=first_output.raw_response,
            selected_indices=first_output.selected_indices,
            need_new=first_output.need_new,
            merge_with=getattr(first_output, "merge_with", None)
        )
        main_actions.append(action)
        
        # 处理选中的已有类别
        for idx in first_output.selected_indices:
            if 0 <= idx < len(children):
                next_node = children[idx]
                new_path = path + [next_node]
                self._classify_main_trajectory(
                    next_node, article_id, article_content, new_path,
                    all_paths, main_actions, decision_points, sampling_num, top_k, root_tree
                )
        
        # 处理NEW（同一步可选决定 InsertParentPath 归拢对象）
        if first_output.need_new:
            self._handle_new_category_main(
                current_node,
                article_id,
                article_content,
                path,
                child_summaries,
                all_paths,
                main_actions,
                decision_points,
                sampling_num,
                top_k,
                root_tree,
                merge_with=getattr(first_output, "merge_with", None),
            )
        
        # 如果没有任何选择
        if not first_output.selected_indices and not first_output.need_new:
            current_node.add_citation(article_id)
            all_paths.append(path.copy())
    
    def _handle_new_category_main(
        self,
        current_node: TreeNode,
        article_id: str,
        article_content: str,
        path: List[TreeNode],
        child_summaries: List[str],
        all_paths: List[List[TreeNode]],
        main_actions: List[Action],
        decision_points: List[DecisionPoint],
        sampling_num: int,
        top_k: int,
        root_tree: TreeNode,  # 新增参数：根节点
        merge_with: Optional[int] = None,
    ):
        """主线处理NEW（CreateLeaf + 可选 InsertParentPath，然后交给 bottom-up 更新summary）"""
        from summary_based_classifier.llm.prompts import PromptTemplates
        
        updater_input = SummaryInput(
            topic_name=self.topic_name,
            node_summary="",
            parent_summary=current_node.summary if current_node.summary else self.topic_name,
            sibling_summaries=child_summaries,
            new_content=article_content
        )
                
        prompt = PromptTemplates.format_summary_prompt(
            topic_name=updater_input.topic_name,
            node_summary=updater_input.node_summary,
            parent_summary=updater_input.parent_summary,
            sibling_summaries=updater_input.sibling_summaries,
            new_content=updater_input.new_content
        )
        
        updater_outputs = self.updater.update_summary(updater_input, n_samples=sampling_num, bm25_stats=self.bm25_stats)
        
        # BOW模式和Model模式的判断不同
        if self.updater.mode == 'bow':
            valid_outputs = [o for o in updater_outputs if o.explanation]
        else:
        valid_outputs = [o for o in updater_outputs if o.explanation and o.scope]
        
        # 去重（根据explanation和scope）
        unique_outputs = []
        seen = set()
        for o in valid_outputs:
            if self.updater.mode == 'bow':
                key = o.explanation  # BOW模式：只用explanation作为key
            else:
            key = (o.explanation, o.scope)
            if key not in seen:
                seen.add(key)
                unique_outputs.append(o)
        
        top_summary_outputs = unique_outputs[:top_k]
        
        if not top_summary_outputs:
            return
        
        # 记录决策点
        if len(top_summary_outputs) > 1:
            decision_points.append(DecisionPoint(
                decision_type='generate_new',
                current_node=current_node,
                path=path.copy(),
                article_id=article_id,
                article_content=article_content,
                prompt=prompt,
                all_outputs=top_summary_outputs,
                chosen_index=0,
                tree_snapshot=root_tree.clone(),  # 保存根节点而不是当前节点
                actions_so_far=main_actions.copy(),
                child_summaries=child_summaries
            ))
        
        # 使用第一个结果
        first_summary = top_summary_outputs[0]
        if self.updater.mode == 'bow':
            new_summary = first_summary.explanation
        else:
            if first_summary.scope:
        new_summary = f"EXPLANATION: {first_summary.explanation}\nSCOPE: {first_summary.scope}"
            else:
                new_summary = first_summary.explanation
                    
        # CreateLeaf：创建新叶子节点
        new_leaf = TreeNode(summary=new_summary, citations=[], children=[])
        current_node.add_child(new_leaf)

        # 可选 InsertParentPath：只允许 new_leaf 与一个现存 sibling 归拢
        inserted_parent = None
        if merge_with is not None:
            old_children = [c for c in current_node.children if c != new_leaf]
            if 0 <= merge_with < len(old_children):
                sibling = old_children[merge_with]
                inserted_parent = self._insert_parent_path(current_node, new_leaf, sibling)
                    
        # 记录action
        action = Action(
            action_type='generate_new_summary',
            system='updater',
            node=new_leaf if inserted_parent is None else inserted_parent,
            prompt=prompt,
            completion=first_summary.raw_response if hasattr(first_summary, 'raw_response') else "",
            updated_summary=new_summary,
            explanation=first_summary.explanation,
            scope=first_summary.scope
        )
        main_actions.append(action)
        
        # 注意：采样时不进行重新路由，保持树的原始状态以记录真实决策路径
        
        # 继续递归分类新节点
        new_path = path + ([new_leaf] if inserted_parent is None else [inserted_parent, new_leaf])
        self._classify_main_trajectory(
            new_leaf, article_id, article_content, new_path,
            all_paths, main_actions, decision_points, sampling_num, top_k, root_tree
        )

    def _update_main_trajectory(
        self,
        path: List[TreeNode],
        article_content: str,
        main_actions: List[Action],
        decision_points: List[DecisionPoint],
        sampling_num: int,
        top_k: int,
        root_tree: TreeNode  # 新增参数：根节点
    ):
        """主线的bottom-up更新（采样multiple但只用第一个，记录所有决策点）"""
        from summary_based_classifier.llm.prompts import PromptTemplates
        
        for node in reversed(path):
            if node.parent is None:
                break
            
            updater_input = SummaryInput(
                topic_name=self.topic_name,
                node_summary=node.summary,
                parent_summary=node.parent.summary if node.parent.summary else self.topic_name,
                sibling_summaries=[sib.summary for sib in node.get_siblings()],
                new_content=article_content
            )
            
            prompt = PromptTemplates.format_summary_prompt(
                topic_name=updater_input.topic_name,
                node_summary=updater_input.node_summary,
                parent_summary=updater_input.parent_summary,
                sibling_summaries=updater_input.sibling_summaries,
                new_content=updater_input.new_content
            )
            
            # 采样sampling_num个输出
            updater_outputs = self.updater.update_summary(updater_input, n_samples=sampling_num, bm25_stats=self.bm25_stats)
            
            # 如果一个输出都没有（异常情况），跳过
            if not updater_outputs:
                break
            
            # 分成两组：需要更新的和不需要更新的
            if self.updater.mode == 'bow':
                update_outputs = [o for o in updater_outputs if o.needs_update and o.explanation]
            else:
            update_outputs = [o for o in updater_outputs if o.needs_update and o.explanation and o.scope]
            no_update_outputs = [o for o in updater_outputs if not o.needs_update]
            
            # 对需要更新的输出进行去重（根据explanation和scope）
            unique_update_outputs = []
            seen = set()
            for o in update_outputs:
                if self.updater.mode == 'bow':
                    key = o.explanation
                else:
                key = (o.explanation, o.scope)
                if key not in seen:
                    seen.add(key)
                    unique_update_outputs.append(o)
            
            # 取前top_k个
            top_update_outputs = unique_update_outputs[:top_k]
            
            # 确保至少有一个"不需要更新"的输出
            has_no_update = any(not o.needs_update for o in top_update_outputs)
            
            if not has_no_update:
                # 如果没有"不需要更新"的输出
                if no_update_outputs:
                    # 有现成的"不需要更新"输出，追加或替换
                    if len(top_update_outputs) < 2:
                        top_update_outputs.append(no_update_outputs[0])
                    else:
                        top_update_outputs[-1] = no_update_outputs[0]
                else:
                    # 没有"不需要更新"的输出，手动构造一个
                    no_update_sample = SummaryOutput(
                        needs_update=False,
                        explanation=None,
                        scope=None,
                        raw_response="NEEDS_UPDATE: No"
                    )
                    if len(top_update_outputs) < 2:
                        top_update_outputs.append(no_update_sample)
                    else:
                        top_update_outputs[-1] = no_update_sample
            
            # 如果所有输出都是"不需要更新"
            if not top_update_outputs and no_update_outputs:
                top_update_outputs = [no_update_outputs[0]]
            elif not top_update_outputs and not no_update_outputs:
                # 异常情况：一个有效输出都没有
                break
            
            # 记录决策点
            if len(top_update_outputs) > 1:
                decision_points.append(DecisionPoint(
                    decision_type='update',
                    current_node=node,
                    path=path.copy(),
                    article_id="",  # update阶段不需要article_id
                    article_content=article_content,
                    prompt=prompt,
                    all_outputs=top_update_outputs,
                    chosen_index=0,
                    tree_snapshot=root_tree.clone(),  # 保存根节点而不是当前节点
                    actions_so_far=main_actions.copy(),
                    parent_summary=updater_input.parent_summary,
                    sibling_summaries=updater_input.sibling_summaries
                ))
            
            # 使用第一个结果
            first_output = top_update_outputs[0]
            
            # 如果需要更新，更新summary并记录action
            if first_output.needs_update and first_output.explanation:
                if self.updater.mode == 'bow':
                    node.summary = first_output.explanation
                else:
                    if first_output.scope:
                node.summary = f"EXPLANATION: {first_output.explanation}\nSCOPE: {first_output.scope}"
                    else:
                        node.summary = first_output.explanation
                
                # 记录"需要更新"的action
                action = Action(
                    action_type='update_summary',
                    system='updater',
                    node=node,
                    prompt=prompt,
                    completion=first_output.raw_response if hasattr(first_output, 'raw_response') else "",
                    needs_update=True,
                    updated_summary=node.summary,
                    explanation=first_output.explanation,
                    scope=first_output.scope
                )
                main_actions.append(action)
            else:
                # 记录"不需要更新"的action（关键修复！）
                action = Action(
                    action_type='update_summary',
                    system='updater',
                    node=node,
                    prompt=prompt,
                    completion=first_output.raw_response if hasattr(first_output, 'raw_response') else "",
                    needs_update=False,
                    updated_summary=None,  # 不更新summary
                    explanation=None,
                    scope=None
                )
                main_actions.append(action)
                
                # 不需要更新，停止向上更新
                break
    
    def _create_branch_trajectory(
        self,
        decision_point: DecisionPoint,
        alt_idx: int,
        article_id: str,
        article_content: str,
        main_all_paths: List[List[TreeNode]]
    ) -> Optional[Trajectory]:
        """
        为决策点的某个替代结果创建支线轨迹（不再分叉，所有采样都用n=1）
        
        Args:
            decision_point: 决策点信息
            alt_idx: 替代结果的索引
            article_id: 文章ID
            article_content: 文章内容
            main_all_paths: 主线的all_paths（仅用于分类决策）
        """
        # 从决策点恢复状态（深拷贝）
        branch_tree = decision_point.tree_snapshot.clone()
        branch_actions = decision_point.actions_so_far.copy()
        
        alt_output = decision_point.all_outputs[alt_idx]
        
        if decision_point.decision_type == 'classify':
            return self._branch_from_classify(
                decision_point, alt_output, branch_tree, branch_actions, article_id, article_content
            )
        
        elif decision_point.decision_type == 'generate_new':
            return self._branch_from_generate_new(
                decision_point, alt_output, branch_tree, branch_actions, article_id, article_content
            )
        
        elif decision_point.decision_type == 'update':
            return self._branch_from_update(
                decision_point, alt_output, branch_tree, branch_actions, article_content
            )
        
        return None
    
    def _branch_from_classify(
        self,
        decision_point: DecisionPoint,
        alt_output: ClassificationOutput,
        branch_tree: TreeNode,
        branch_actions: List[Action],
        article_id: str,
        article_content: str
    ) -> Trajectory:
        """从分类决策点创建支线"""
        # 在branch_tree中找到对应节点
        current_node_in_branch = self._find_node_in_tree(branch_tree, decision_point.current_node)
        
        # 记录分类action
        action = Action(
            action_type='classify',
            system='classify_generator',
            node=current_node_in_branch,
            prompt=decision_point.prompt,
            completion=alt_output.raw_response,
            selected_indices=alt_output.selected_indices,
            need_new=alt_output.need_new,
            merge_with=getattr(alt_output, "merge_with", None)
        )
        branch_actions.append(action)
        
        # 继续完成分类（不再采样，n=1）
        branch_all_paths = []
        self._classify_branch(
            current_node_in_branch, alt_output, decision_point.article_id,
            decision_point.article_content, decision_point.path,
            decision_point.child_summaries, branch_all_paths, branch_actions
        )
        
        # Bottom-up更新（不再采样）
        for path in branch_all_paths:
            self._update_branch(path, decision_point.article_content, branch_actions)
        
        return Trajectory(actions=branch_actions, final_tree=branch_tree)
    
    def _branch_from_generate_new(
        self,
        decision_point: DecisionPoint,
        alt_output: SummaryOutput,
        branch_tree: TreeNode,
        branch_actions: List[Action],
        article_id: str,
        article_content: str
    ) -> Trajectory:
        """从生成新summary决策点创建支线"""
        if self.updater.mode == 'bow':
            new_summary = alt_output.explanation
        else:
            if alt_output.scope:
        new_summary = f"EXPLANATION: {alt_output.explanation}\nSCOPE: {alt_output.scope}"
            else:
                new_summary = alt_output.explanation
        
        # 在branch_tree中找到对应节点并创建新子节点
        current_node_in_branch = self._find_node_in_tree(branch_tree, decision_point.current_node)
        new_node = TreeNode(summary=new_summary, citations=[], children=[])
        current_node_in_branch.add_child(new_node)
        
        # 记录action
        action = Action(
            action_type='generate_new_summary',
            system='updater',
            node=new_node,
            prompt=decision_point.prompt,
            completion=alt_output.raw_response if hasattr(alt_output, 'raw_response') else "",
            updated_summary=new_summary,
            explanation=alt_output.explanation,
            scope=alt_output.scope
        )
        branch_actions.append(action)
        
        # 注意：采样时不进行重新路由，保持树的原始状态以记录真实决策路径
        
        # 继续分类新节点（不再采样）
        branch_all_paths = []
        new_path = decision_point.path + [new_node]
        self._classify_branch_simple(
            new_node, decision_point.article_id, decision_point.article_content,
            new_path, branch_all_paths, branch_actions
        )
        
        # Bottom-up更新
        for path in branch_all_paths:
            self._update_branch(path, decision_point.article_content, branch_actions)
        
        return Trajectory(actions=branch_actions, final_tree=branch_tree)
    
    def _branch_from_update(
        self,
        decision_point: DecisionPoint,
        alt_output: SummaryOutput,
        branch_tree: TreeNode,
        branch_actions: List[Action],
        article_content: str
    ) -> Trajectory:
        """从更新summary决策点创建支线"""
        # 在branch_tree中找到对应节点
        node_in_branch = self._find_node_in_tree(branch_tree, decision_point.current_node)
        
        # 检查是否需要更新
        if alt_output.needs_update and alt_output.explanation:
            if self.updater.mode == 'bow':
                node_in_branch.summary = alt_output.explanation
            else:
                if alt_output.scope:
            node_in_branch.summary = f"EXPLANATION: {alt_output.explanation}\nSCOPE: {alt_output.scope}"
                else:
                    node_in_branch.summary = alt_output.explanation
            
            # 记录"需要更新"的action
            action = Action(
                action_type='update_summary',
                system='updater',
                node=node_in_branch,
                prompt=decision_point.prompt,
                completion=alt_output.raw_response if hasattr(alt_output, 'raw_response') else "",
                needs_update=True,
                updated_summary=node_in_branch.summary,
                explanation=alt_output.explanation,
                scope=alt_output.scope
            )
            branch_actions.append(action)
            
            # 继续向上更新（不再采样）
            self._update_branch(decision_point.path, article_content, branch_actions, start_from=node_in_branch)
        else:
            # 记录"不需要更新"的action
            action = Action(
                action_type='update_summary',
                system='updater',
                node=node_in_branch,
                prompt=decision_point.prompt,
                completion=alt_output.raw_response if hasattr(alt_output, 'raw_response') else "",
                needs_update=False,
                updated_summary=None,
                explanation=None,
                scope=None
            )
            branch_actions.append(action)
            
            # 不需要更新，不再向上传播
        
        return Trajectory(actions=branch_actions, final_tree=branch_tree)
    
    def _classify_branch(
        self,
        current_node: TreeNode,
        classify_output: ClassificationOutput,
        article_id: str,
        article_content: str,
        path: List[TreeNode],
        child_summaries: List[str],
        branch_all_paths: List[List[TreeNode]],
        branch_actions: List[Action]
    ):
        """处理分类输出并继续完成分类（不再采样）"""
        children = current_node.children
        
        # 处理选中的已有类别
        for idx in classify_output.selected_indices:
            if 0 <= idx < len(children):
                next_node = children[idx]
                new_path = path + [next_node]
                self._classify_branch_simple(
                    next_node, article_id, article_content, new_path, branch_all_paths, branch_actions
                )
        
        # 处理NEW（同一步可选决定 InsertParentPath 归拢对象）
        if classify_output.need_new:
            new_node = self._create_new_category_branch(
                current_node,
                article_content,
                child_summaries,
                branch_actions,
                merge_with=getattr(classify_output, "merge_with", None),
            )
            if new_node:
                new_path = path + [new_node]
                self._classify_branch_simple(
                    new_node, article_id, article_content, new_path, branch_all_paths, branch_actions
                )
        
        # 如果没有任何选择
        if not classify_output.selected_indices and not classify_output.need_new:
            current_node.add_citation(article_id)
            branch_all_paths.append(path.copy())
    
    def _classify_branch_simple(
        self,
        current_node: TreeNode,
        article_id: str,
        article_content: str,
        path: List[TreeNode],
        branch_all_paths: List[List[TreeNode]],
        branch_actions: List[Action]
    ):
        """支线的递归分类（不采样，n=1）"""
        # 检查深度
        if current_node.depth >= self.max_depth:
            current_node.add_citation(article_id)
            branch_all_paths.append(path.copy())
            return
        
        children = current_node.children
        child_summaries = [c.summary for c in children]
        
        # 计算结构特征
        child_num_children, child_max_depth = self._get_child_structure_features(children)
        
        # 调用分类系统（不采样，带结构特征）
        classification_input = ClassificationInput(
            article_content=article_content,
            current_node_summary=current_node.summary if current_node.summary else self.topic_name,
            child_summaries=child_summaries,
            topic_name=self.topic_name,
            child_num_children=child_num_children,
            child_max_depth=child_max_depth,
            current_depth=current_node.depth,
            num_children=len(children)
        )
        
        prompt = self.classifier.create_prompt(classification_input) if not self.use_workers else ""
        outputs = self._call_classifier(classification_input, n=1)
        
        if not outputs:
            current_node.add_citation(article_id)
            branch_all_paths.append(path.copy())
            return
        
        action_output = outputs[0]
        
        # 记录action
        action = Action(
            action_type='classify',
            system='classify_generator',
            node=current_node,
            prompt=prompt,
            completion=action_output.raw_response,
            selected_indices=action_output.selected_indices,
            need_new=action_output.need_new,
            merge_with=getattr(action_output, "merge_with", None)
        )
        branch_actions.append(action)
        
        # 处理选中的已有类别
        for idx in action_output.selected_indices:
            if 0 <= idx < len(children):
                next_node = children[idx]
                new_path = path + [next_node]
                self._classify_branch_simple(
                    next_node, article_id, article_content, new_path, branch_all_paths, branch_actions
                )
        
        # 处理NEW（同一步可选决定 InsertParentPath 归拢对象）
        if action_output.need_new:
            new_node = self._create_new_category_branch(
                current_node,
                article_content,
                child_summaries,
                branch_actions,
                merge_with=getattr(action_output, "merge_with", None),
            )
            if new_node:
                new_path = path + [new_node]
                self._classify_branch_simple(
                    new_node, article_id, article_content, new_path, branch_all_paths, branch_actions
                )
        
        # 如果没有任何选择
        if not action_output.selected_indices and not action_output.need_new:
            current_node.add_citation(article_id)
            branch_all_paths.append(path.copy())
    
    def _create_new_category_branch(
        self,
        parent_node: TreeNode,
        article_content: str,
        sibling_summaries: List[str],
        branch_actions: List[Action],
        merge_with: Optional[int] = None
    ) -> Optional[TreeNode]:
        """支线创建新类别（不采样）"""
        from summary_based_classifier.llm.prompts import PromptTemplates
        
        updater_input = SummaryInput(
            topic_name=self.topic_name,
            node_summary="",
            parent_summary=parent_node.summary if parent_node.summary else self.topic_name,
            sibling_summaries=sibling_summaries,
            new_content=article_content
        )
        
        prompt = PromptTemplates.format_summary_prompt(
            topic_name=updater_input.topic_name,
            node_summary=updater_input.node_summary,
            parent_summary=updater_input.parent_summary,
            sibling_summaries=updater_input.sibling_summaries,
            new_content=updater_input.new_content
        )
        
        outputs = self.updater.update_summary(updater_input, n_samples=1, bm25_stats=self.bm25_stats)
        
        for output in outputs:
            if output.explanation:
                # BOW模式：explanation是BOW JSON字符串
                # Model模式：explanation和scope都是字符串
                if self.updater.mode == 'bow':
                    new_summary = output.explanation
                else:
                    if output.scope:
                new_summary = f"EXPLANATION: {output.explanation}\nSCOPE: {output.scope}"
                    else:
                        new_summary = output.explanation
                
                new_leaf = TreeNode(summary=new_summary, citations=[], children=[])
                parent_node.add_child(new_leaf)

                # 可选 InsertParentPath：只允许 new_leaf 与一个现存 sibling 归拢
                if merge_with is not None:
                    old_children = [c for c in parent_node.children if c != new_leaf]
                    if 0 <= merge_with < len(old_children):
                        sibling = old_children[merge_with]
                        self._insert_parent_path(parent_node, new_leaf, sibling)
                
                # 记录action
                action = Action(
                    action_type='generate_new_summary',
                    system='updater',
                    node=new_leaf,
                    prompt=prompt,
                    completion=output.raw_response if hasattr(output, 'raw_response') else "",
                    updated_summary=new_summary,
                    explanation=output.explanation,
                    scope=output.scope if output.scope else ""
                )
                branch_actions.append(action)
                
                # 注意：采样时不进行重新路由，保持树的原始状态以记录真实决策路径
                
                return new_leaf
        
        return None
    
    def _update_branch(
        self,
        path: List[TreeNode],
        article_content: str,
        branch_actions: List[Action],
        start_from: TreeNode = None
    ):
        """支线的bottom-up更新（不采样）"""
        from summary_based_classifier.llm.prompts import PromptTemplates
        
        started = (start_from is None)
        
        for node in reversed(path):
            if not started:
                if node == start_from:
                    started = True
                continue
            
            if node.parent is None:
                break
            
            updater_input = SummaryInput(
                topic_name=self.topic_name,
                node_summary=node.summary,
                parent_summary=node.parent.summary if node.parent.summary else self.topic_name,
                sibling_summaries=[sib.summary for sib in node.get_siblings()],
                new_content=article_content
            )
            
            prompt = PromptTemplates.format_summary_prompt(
                topic_name=updater_input.topic_name,
                node_summary=updater_input.node_summary,
                parent_summary=updater_input.parent_summary,
                sibling_summaries=updater_input.sibling_summaries,
                new_content=updater_input.new_content
            )
            
            outputs = self.updater.update_summary(updater_input, n_samples=1, bm25_stats=self.bm25_stats)
            
            for output in outputs:
                if output.needs_update and output.explanation:
                    # BOW模式：explanation是BOW JSON字符串
                    # Model模式：explanation和scope都是字符串
                    if self.updater.mode == 'bow':
                        node.summary = output.explanation
                    else:
                        if output.scope:
                    node.summary = f"EXPLANATION: {output.explanation}\nSCOPE: {output.scope}"
                        else:
                            node.summary = output.explanation
                    
                    # 记录"需要更新"的action
                    action = Action(
                        action_type='update_summary',
                        system='updater',
                        node=node,
                        prompt=prompt,
                        completion=output.raw_response if hasattr(output, 'raw_response') else "",
                        needs_update=True,
                        updated_summary=node.summary,
                        explanation=output.explanation,
                        scope=output.scope if output.scope else ""
                    )
                    branch_actions.append(action)
                    break
                else:
                    # 记录"不需要更新"的action（关键修复！）
                    action = Action(
                        action_type='update_summary',
                        system='updater',
                        node=node,
                        prompt=prompt,
                        completion=output.raw_response if hasattr(output, 'raw_response') else "",
                        needs_update=False,
                        updated_summary=None,
                        explanation=None,
                        scope=None
                    )
                    branch_actions.append(action)
                    
                    # 不需要更新，停止向上更新
                    break
    
    def _filter_unique_classify_outputs(self, outputs: List[ClassificationOutput]) -> List[ClassificationOutput]:
        """过滤重复的分类输出"""
        unique = []
        seen = set()
        
        for output in outputs:
            key = (tuple(sorted(output.selected_indices)), output.need_new)
            if key not in seen and (output.selected_indices or output.need_new):
                unique.append(output)
                seen.add(key)
        
        return unique
    
    def _find_node_in_tree(self, tree: TreeNode, target_node: TreeNode) -> TreeNode:
        """在树中找到对应节点（根据summary和depth匹配）"""
        if tree.summary == target_node.summary and tree.depth == target_node.depth:
            return tree
        for child in tree.children:
            result = self._find_node_in_tree(child, target_node)
            if result and result.summary == target_node.summary:
                return result
        return tree
    
    def _reroute_articles(self, parent_node: TreeNode, new_article_content: str) -> List[TreeNode]:
        """
        重新路由父节点中的已有文章
        
        当在一个节点下第一次创建新类别时调用，将该节点的文章重新分类到子节点，
        可能会创建更多新类别
        
        Args:
            parent_node: 父节点（原来是叶子节点，现在有了第一个子节点）
            new_article_content: 触发创建新类别的文章内容（用于日志）
            
        Returns:
            重新路由期间新创建的子节点列表
        """
        # 保留旧的citations
        old_citations = parent_node.citations.copy()
        
        if not old_citations:
            return []
        
        # 记录重新路由前的子节点
        old_children = parent_node.children.copy()
        
        # 清空父节点的citations
        parent_node.citations = []
        
        # 对每篇旧文章重新分类
        for old_article_id in old_citations:
            # 从缓存中获取文章内容
            old_article_content = self.articles_cache.get(old_article_id, "")
            
            if not old_article_content:
                # 如果找不到内容，保持在父节点
                parent_node.add_citation(old_article_id)
                continue
            
            # 在父节点下重新分类（可能创建新类别）
            self._reclassify_at_node(
                parent_node, 
                old_article_id, 
                old_article_content
            )
        
        # 找出新创建的子节点
        new_children = [child for child in parent_node.children if child not in old_children]
        return new_children
    
    def _reclassify_at_node(
        self,
        parent_node: TreeNode,
        article_id: str,
        article_content: str
    ):
        """
        在指定节点下重新分类一篇文章（可能创建新类别）
        
        Args:
            parent_node: 要在其下分类的节点
            article_id: 文章ID
            article_content: 文章内容
        """
        children = parent_node.children
        child_summaries = [c.summary for c in children]
        
        # 计算结构特征
        child_num_children, child_max_depth = self._get_child_structure_features(children)
        
        # 调用分类系统（带结构特征）
        classification_input = ClassificationInput(
            article_content=article_content,
            current_node_summary=parent_node.summary if parent_node.summary else self.topic_name,
            child_summaries=child_summaries,
            topic_name=self.topic_name,
            child_num_children=child_num_children,
            child_max_depth=child_max_depth,
            current_depth=parent_node.depth,
            num_children=len(children)
        )
        
        outputs = self._call_classifier(classification_input, n=1)
        
        if not outputs:
            # 分类失败，保持在父节点
            parent_node.add_citation(article_id)
            return
        
        action = outputs[0]
        
        # 处理选中的已有类别（可能多个）
        classified = False
        for idx in action.selected_indices:
            if 0 <= idx < len(children):
                children[idx].add_citation(article_id)
                classified = True
        
        # 处理NEW：创建新类别
        if action.need_new:
            new_node = self._create_new_category(
                parent_node, 
                article_content, 
                child_summaries
            )
            if new_node:
                new_node.add_citation(article_id)
                classified = True
        
        # 如果没有选择任何类别，保持在父节点
        if not classified:
            parent_node.add_citation(article_id)
    
    def build_tree_for_articles(
        self,
        articles: List[Dict],
        root: TreeNode
    ) -> TreeNode:
        """
        为多篇文章构建结构树（推理用）
        
        Args:
            articles: 文章列表 [{'id': ..., 'content': ...}, ...]
            root: 根节点
            
        Returns:
            更新后的根节点
        """
        from tqdm import tqdm
        
        i=0
        for article in tqdm(articles, desc="推理进度", disable=self.use_workers):
            self.classify_and_update(
                article_id=article['id'],
                article_content=article['content'],
                root=root
            )
            i+=1
            if i == 59:
                pass
        
        return root
    
    def tree_to_dict(self, node: TreeNode, level: int = 1) -> Dict:
        """
        将树节点转换为字典（用于保存）
        
        Args:
            node: 节点
            level: 层级
            
        Returns:
            字典
        """
        # 从summary中提取title
        title = self._extract_title_from_summary(node.summary)
        
        return {
            'title': title,
            'level': level,
            'summary': node.summary,
            'citations': node.citations,
            'children': [self.tree_to_dict(child, level + 1) for child in node.children]
        }
    
    def _extract_title_from_summary(self, summary: str) -> str:
        """从summary中提取title"""
        if not summary:
            return "Untitled"
        
        # 提取EXPLANATION部分作为title
        if 'EXPLANATION:' in summary:
            lines = summary.split('\n')
            for line in lines:
                if line.startswith('EXPLANATION:'):
                    title = line.replace('EXPLANATION:', '').strip()
                    # 截取前50字符
                    if len(title) > 50:
                        title = title[:50] + '...'
                    return title
        
        # 如果没有EXPLANATION，取第一行
        first_line = summary.split('\n')[0].strip()
        if len(first_line) > 50:
            first_line = first_line[:50] + '...'
        return first_line if first_line else "Untitled"
    
    def get_tree_depth(self, node: TreeNode) -> int:
        """
        计算树的最大深度
        
        Args:
            node: 根节点
            
        Returns:
            树的最大深度
        """
        if not node.children:
            return node.depth
        
        max_depth = node.depth
        for child in node.children:
            child_depth = self.get_tree_depth(child)
            max_depth = max(max_depth, child_depth)
        
        return max_depth
    
    def _get_child_structure_features(self, children: List[TreeNode]) -> Tuple[List[int], List[int]]:
        """
        获取每个子节点的结构特征
        
        Args:
            children: 子节点列表
            
        Returns:
            (child_num_children, child_max_depth) 两个列表
        """
        child_num_children = []
        child_max_depth = []
        
        for child in children:
            child_num_children.append(len(child.children))
            # 计算子树深度（相对深度）
            max_depth = self.get_tree_depth(child) - child.depth
            child_max_depth.append(max_depth)
        
        return child_num_children, child_max_depth

