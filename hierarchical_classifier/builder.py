"""
构建系统
递归地构建文章的层次化结构树
"""
import json
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, asdict
from pathlib import Path
from classifier import Classifier, ClassificationInput, ClassificationOutput


def jaccard_similarity(str1: str, str2: str) -> float:
    """
    计算两个字符串的Jaccard相似度（词级别）
    
    Args:
        str1: 第一个字符串
        str2: 第二个字符串
        
    Returns:
        Jaccard相似度 (0-1之间)
    """
    # 转为小写并分词
    words1 = set(str1.lower().split())
    words2 = set(str2.lower().split())
    
    if not words1 and not words2:
        return 0.0
    
    # 计算交集和并集
    intersection = words1 & words2
    union = words1 | words2
    
    if not union:
        return 0.0
    
    return len(intersection) / len(union)


@dataclass
class TreeNode:
    """树节点"""
    title: str  # 节点标题
    level: int  # 层级（1, 2, 3...）
    citations: List[str]  # 引用的reference_ids
    children: List['TreeNode']  # 子节点
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'title': self.title,
            'level': self.level,
            'citations': sorted(self.citations),
            'children': [child.to_dict() for child in self.children]
        }
    
    def add_citation(self, reference_id: str):
        """添加引用"""
        if reference_id not in self.citations:
            self.citations.append(reference_id)
    
    def get_child(self, title: str) -> Optional['TreeNode']:
        """获取子节点（不区分大小写）"""
        title_lower = title.lower()
        for child in self.children:
            if child.title.lower() == title_lower:
                return child
        return None
    
    def add_child(self, child: 'TreeNode'):
        """添加子节点"""
        self.children.append(child)


@dataclass
class PendingClassification:
    """待分类的项目"""
    reference_id: str
    node: TreeNode  # 当前节点
    path: str  # 当前路径


class TreeBuilder:
    """树构建器"""
    
    def __init__(
        self,
        classifier: Classifier,
        references_data: Dict,
        max_depth: int = 10,
        structures_file: Optional[str] = None,
        num_inference_constraint_leaves: int = 20
    ):
        """
        Args:
            classifier: 分类器
            references_data: references数据
            max_depth: 最大深度
            structures_file: 结构文件路径（用于初始化模式）
            num_inference_constraint_leaves: 推理时选择多少个叶子节点作为约束
        """
        self.classifier = classifier
        self.references_data = references_data
        self.max_depth = max_depth
        self.structures_file = structures_file
        self.structures_data = None
        self.num_inference_constraint_leaves = num_inference_constraint_leaves
        
        # 如果提供了结构文件，加载它
        if structures_file:
            print(f"加载结构文件: {structures_file}")
            with open(structures_file, 'r', encoding='utf-8') as f:
                self.structures_data = json.load(f)
            print(f"  - 加载 {len(self.structures_data)} 个topics的结构")
        
    def _load_structure_as_tree(self, structure_node: Dict) -> TreeNode:
        """
        从结构字典递归构建树节点（不包含citations）
        
        Args:
            structure_node: 结构节点字典
            
        Returns:
            TreeNode
        """
        node = TreeNode(
            title=structure_node['title'],
            level=structure_node.get('level', 0),
            citations=[],  # 初始时清空所有引用
            children=[]
        )
        
        # 递归处理子节点
        for child_dict in structure_node.get('children', []):
            child_node = self._load_structure_as_tree(child_dict)
            node.add_child(child_node)
        
        return node
    
    def _load_topic_structure(self, topic_key: str, topic_name: str) -> TreeNode:
        """
        只加载topic结构树的第一层节点
        
        Args:
            topic_key: topic键
            topic_name: topic名称
            
        Returns:
            根节点（level=0），包含第一层子节点（但子节点没有children）
        """
        # 创建根节点
        root = TreeNode(
            title=topic_name,
            level=0,
            citations=[],
            children=[]
        )
        
        # 从结构文件只加载第一层子节点
        if topic_key in self.structures_data:
            structure = self.structures_data[topic_key]
            # structure['structure'] 是一个列表，包含多个顶层节点
            for structure_node in structure.get('structure', []):
                # 只创建第一层节点，不递归加载它的子节点
                child_node = TreeNode(
                    title=structure_node['title'],
                    level=structure_node.get('level', 1),  # 第一层通常是level 1
                    citations=[],
                    children=[]  # 不加载子节点
                )
                root.add_child(child_node)
        
        return root
    
    def _remove_empty_nodes(self, node: TreeNode) -> bool:
        """
        递归删除没有引用的节点
        
        Args:
            node: 当前节点
            
        Returns:
            True表示这个节点应该被保留，False表示应该被删除
        """
        # 首先递归处理所有子节点
        node.children = [child for child in node.children if self._remove_empty_nodes(child)]
        
        # 如果这个节点有引用，或者有子节点，则保留
        # 根节点(level=0)总是保留
        if node.level == 0 or node.citations or node.children:
            return True
        
        # 否则删除
        return False
    
    def collect_sibling_leaf_nodes(self, root: TreeNode, current_path: str) -> List[tuple[TreeNode, str]]:
        """
        收集当前路径的兄弟节点及其所有后代叶子节点
        
        例如：root是整棵树，current_path = "The Hobbit - Narrative"
        返回：Concept and creation 和 Influences 分支下的所有叶子节点
        
        Args:
            root: 根节点（整棵树）
            current_path: 当前路径，如 "The Hobbit - Narrative"
            
        Returns:
            [(叶子节点, 完整路径), ...]
        """
        # 解析当前路径
        path_parts = [p.strip() for p in current_path.split(' - ')]
        
        # 如果是根节点（只有一层），没有兄弟节点
        if len(path_parts) == 1:
            return []
        
        sibling_leaves = []
        
        def collect_all_leaves(node: TreeNode, path: str):
            """从一个节点递归收集所有叶子"""
            if not node.children:
                sibling_leaves.append((node, path))
            else:
                for child in node.children:
                    child_path = f"{path} - {child.title}"
                    collect_all_leaves(child, child_path)
        
        def find_and_collect_siblings(nodes: List[TreeNode], path_index: int, accumulated_path: str) -> bool:
            """
            递归查找目标节点，并收集其兄弟节点的叶子
            
            Args:
                nodes: 当前层的节点列表
                path_index: 当前在path_parts中的索引（从1开始，因为0是root title）
                accumulated_path: 累积的路径（到当前层的父节点）
            
            Returns:
                True if found, False otherwise
            """
            if path_index >= len(path_parts):
                return False
            
            target_title = path_parts[path_index]
            
            # 遍历当前层的所有节点
            for node in nodes:
                if node.title == target_title:
                    # 找到了路径上的节点
                    if path_index == len(path_parts) - 1:
                        # 这是目标节点，现在收集其兄弟节点的叶子
                        for sibling in nodes:
                            if sibling.title != target_title:
                                # 这是兄弟节点
                                sibling_path = f"{accumulated_path} - {sibling.title}"
                                collect_all_leaves(sibling, sibling_path)
                        return True
                    else:
                        # 还需要继续向下查找
                        if node.children:
                            node_path = f"{accumulated_path} - {node.title}"
                            return find_and_collect_siblings(node.children, path_index + 1, node_path)
                        else:
                            return False
            
            return False
        
        # 从根节点的子节点开始查找（path_parts[1]对应root.children的第一层）
        find_and_collect_siblings(root.children, 1, root.title)
        
        return sibling_leaves
    
    def select_inference_constraints(
        self,
        root: TreeNode,
        current_path: str,
        existing_subtitles: List[str]
    ) -> List[str]:
        """
        选择推理时的约束路径
        
        约束来自当前节点的兄弟节点及其所有后代叶子节点
        
        Args:
            root: 根节点（整棵树）
            current_path: 当前路径，如 "T - A"
            existing_subtitles: 当前节点已有的子标题 [a1, a2]
            
        Returns:
            约束路径列表，如 ["T - B - b1", "T - B - b2", "T - C"]
        """
        # 收集兄弟节点的所有叶子
        sibling_leaves = self.collect_sibling_leaf_nodes(root, current_path)
        
        if not sibling_leaves:
            return []
        
        if not existing_subtitles:
            # 没有existing_subtitles，返回所有兄弟叶子（但通常不会到这里）
            return [path for _, path in sibling_leaves[:self.num_inference_constraint_leaves]]
        
        # 计算每个叶子节点与existing_subtitles的最高相似度得分
        leaf_scores = []
        for leaf_node, leaf_path in sibling_leaves:
            leaf_title = leaf_node.title
            
            # 计算与所有existing_subtitles的相似度，取最高分
            max_score = max(
                jaccard_similarity(leaf_title, subtitle)
                for subtitle in existing_subtitles
            )
            
            leaf_scores.append((leaf_path, max_score))
        
        # 按得分排序，选择前N个
        leaf_scores.sort(key=lambda x: x[1], reverse=True)
        constraint_paths = [path for path, score in leaf_scores[:self.num_inference_constraint_leaves]]
        
        return constraint_paths
    
    def build_tree(
        self,
        topic_key: str,
        reference_ids: List[str],
        record_mode: bool = False,
        use_structure_init: bool = False
    ) -> tuple[TreeNode, Optional[List[Dict]]]:
        """
        构建结构树（单topic内串行处理）
        
        Args:
            topic_key: topic键，例如 "Person:Albert Einstein"
            reference_ids: 要处理的reference id列表
            record_mode: 是否记录分类过程（用于生成训练数据）
            use_structure_init: 是否使用结构文件初始化
            
        Returns:
            root: 根节点
            records: 如果record_mode=True，返回记录列表
        """
        # 获取topic信息
        if topic_key not in self.references_data:
            raise ValueError(f"Topic {topic_key} 不存在")
        
        topic_data = self.references_data[topic_key]
        topic_name = topic_data['topic']
        
        # 创建或加载根节点
        if use_structure_init and self.structures_data:
            # 从结构文件加载完整结构树
            root = self._load_topic_structure(topic_key, topic_name)
        else:
            # 创建空根节点
            root = TreeNode(
                title=topic_name,
                level=0,
                citations=[],
                children=[]
            )
        
        # 记录列表（如果需要）
        records = [] if record_mode else None
        
        # 逐篇文章处理（串行，确保后面的文章能看到前面创建的节点）
        for ref_id in reference_ids:
            if ref_id not in topic_data['references']:
                continue
            
            reference = topic_data['references'][ref_id]
            content = reference.get('content', '')
            
            # 从根节点开始处理这篇文章（传递root用于约束计算）
            self._process_article_recursive(
                topic_key=topic_key,
                reference_id=ref_id,
                article_content=content,
                node=root,
                path=topic_name,
                current_depth=0,
                records=records,
                root=root
            )
        
        # 如果使用结构初始化模式，删除没有引用的空节点
        if use_structure_init:
            self._remove_empty_nodes(root)
        
        return root, records
    
    def _process_article_recursive(
        self,
        topic_key: str,
        reference_id: str,
        article_content: str,
        node: TreeNode,
        path: str,
        current_depth: int,
        records: Optional[List[Dict]],
        root: TreeNode = None
    ):
        """
        递归处理单篇文章在某个节点下的分类
        
        Args:
            topic_key: topic键
            reference_id: reference id
            article_content: 文章内容
            node: 当前节点
            path: 当前路径
            current_depth: 当前深度
            records: 记录列表（如果需要）
            root: 根节点（用于计算约束）
        """
        # 检查深度
        if current_depth >= self.max_depth:
            node.add_citation(reference_id)
            return
        
        # 获取当前节点的子节点标题
        existing_subtitles = [child.title for child in node.children]
        
        # 计算约束路径（如果提供了root）
        constraint_paths = []
        if root is not None and existing_subtitles:
            constraint_paths = self.select_inference_constraints(root, path, existing_subtitles)
        
        # 创建分类输入
        input_data = ClassificationInput(
            topic_key=topic_key,
            reference_id=reference_id,
            article_content=article_content,
            current_path=path,
            existing_subtitles=existing_subtitles,
            constraint_paths=constraint_paths
        )
        
        # 单个分类
        output, _ = self.classifier.classify_single(input_data)
        
        # 记录（如果需要）
        if records is not None:
            record = {
                'topic_key': topic_key,
                'reference_id': reference_id,
                'current_path': path,
                'existing_subtitles': existing_subtitles,
                'constraint_paths': constraint_paths,
                'article_content': article_content,
                'selected_existing': output.selected_existing,
                'new_subtitles': output.new_subtitles
            }
            records.append(record)
        
        # 合并所有标题（现有 + 新增）
        all_subtitles = output.selected_existing + output.new_subtitles
        
        if not all_subtitles:
            # 没有子标题，这是叶子节点
            node.add_citation(reference_id)
            return
        
        # 处理每个子标题
        for subtitle in all_subtitles:
            # 获取或创建子节点
            child_node = node.get_child(subtitle)
            if child_node is None:
                child_node = TreeNode(
                    title=subtitle,
                    level=node.level + 1,
                    citations=[],
                    children=[]
                )
                node.add_child(child_node)
            
            # 添加引用到子节点
            child_node.add_citation(reference_id)
            
            # 递归处理下一层（传递root）
            new_path = f"{path} - {subtitle}"
            self._process_article_recursive(
                topic_key=topic_key,
                reference_id=reference_id,
                article_content=article_content,
                node=child_node,
                path=new_path,
                current_depth=current_depth + 1,
                records=records,
                root=root
            )
    
    def build_trees_for_split_with_tracking(
        self,
        dataset_split: Dict,
        split_name: str,
        output_dir: str,
        record_mode: bool = False,
        use_structure_init: bool = False
    ) -> tuple[Dict, Dict]:
        """
        为数据集划分构建树（带错误跟踪的批处理版本）
        
        用于推理时统计模型输出解析的成功和失败情况
        
        Args:
            dataset_split: 数据集划分
            split_name: 'train', 'val', 或 'test'
            output_dir: 输出目录
            record_mode: 是否记录分类过程
            use_structure_init: 是否使用结构文件初始化
            
        Returns:
            (all_trees, error_stats)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        split_data = dataset_split.get(split_name, {})
        print(f"\n{'='*80}")
        print(f"构建 {split_name} 集的结构树（跨topic批处理模式 + 错误跟踪）")
        print(f"{'='*80}")
        
        # 统计信息
        total_topics = len(split_data)
        total_refs = sum(len(refs) for refs in split_data.values())
        print(f"总计: {total_topics} topics, {total_refs} references")
        
        all_records = [] if record_mode else None
        all_trees = {}
        
        # 错误统计
        error_stats = {
            'total_success': 0,
            'total_failed': 0,
            'by_topic': {}
        }
        
        # 为每个topic创建根节点和初始状态
        topic_states = {}
        for topic_key, reference_ids in split_data.items():
            if topic_key not in self.references_data:
                continue
            
            topic_data = self.references_data[topic_key]
            topic_name = topic_data['topic']
            
            # 创建或加载根节点
            if use_structure_init and self.structures_data:
                # 从结构文件加载完整结构树
                root = self._load_topic_structure(topic_key, topic_name)
            else:
                # 创建空根节点
                root = TreeNode(
                    title=topic_name,
                    level=0,
                    citations=[],
                    children=[]
                )
            
            # 准备该topic的所有文章内容
            articles = []
            for ref_id in reference_ids:
                if ref_id in topic_data['references']:
                    articles.append({
                        'reference_id': ref_id,
                        'content': topic_data['references'][ref_id].get('content', '')
                    })
            
            topic_states[topic_key] = {
                'root': root,
                'articles': articles,
                'current_article_idx': 0,
                'pending_requests': [],
                'records': [] if record_mode else None,
                'completed': False,
                'error_stats': {'success': 0, 'failed': 0}
            }
        
        # 初始化：每个topic的第一篇文章的第一个请求
        for topic_key, state in topic_states.items():
            if state['articles']:
                article = state['articles'][0]
                state['pending_requests'] = [{
                    'reference_id': article['reference_id'],
                    'content': article['content'],
                    'node': state['root'],
                    'path': state['root'].title,
                    'depth': 0
                }]
        
        # 循环处理，直到所有topic都完成
        iteration = 0
        while True:
            iteration += 1
            
            # 收集所有topic当前的待分类请求
            batch_inputs = []
            batch_metadata = []
            
            for topic_key, state in topic_states.items():
                if state['completed']:
                    continue
                
                # 取出当前topic的一个待处理请求（如果有）
                if state['pending_requests']:
                    req = state['pending_requests'][0]
                    
                    existing_subtitles = [child.title for child in req['node'].children]
                    
                    # 计算约束路径
                    constraint_paths = []
                    if existing_subtitles:
                        constraint_paths = self.select_inference_constraints(
                            state['root'], req['path'], existing_subtitles
                        )
                    
                    input_data = ClassificationInput(
                        topic_key=topic_key,
                        reference_id=req['reference_id'],
                        article_content=req['content'],
                        current_path=req['path'],
                        existing_subtitles=existing_subtitles,
                        constraint_paths=constraint_paths
                    )
                    
                    batch_inputs.append(input_data)
                    batch_metadata.append({
                        'topic_key': topic_key,
                        'request': req
                    })
            
            if not batch_inputs:
                break  # 所有topic都处理完了
            
            if iteration:  # 每10次迭代输出一次
                active_topics = sum(1 for s in topic_states.values() if not s['completed'])
                print(f"\r迭代 {iteration}: 批量处理 {len(batch_inputs)} 请求 (活跃topics: {active_topics}/{total_topics})", end='', flush=True)
            
            # 批量分类（启用错误跟踪）
            batch_outputs, batch_error_stats = self.classifier.classify_batch(batch_inputs, track_errors=True)
            
            # 处理分类结果并更新状态
            for input_data, output, metadata in zip(batch_inputs, batch_outputs, batch_metadata):
                topic_key = metadata['topic_key']
                req = metadata['request']
                state = topic_states[topic_key]
                
                # 移除已处理的请求
                state['pending_requests'].pop(0)
                
                # 记录
                if record_mode:
                    record = {
                        'topic_key': topic_key,
                        'reference_id': input_data.reference_id,
                        'current_path': input_data.current_path,
                        'existing_subtitles': input_data.existing_subtitles,
                        'constraint_paths': input_data.constraint_paths,
                        'article_content': input_data.article_content,
                        'selected_existing': output.selected_existing,
                        'new_subtitles': output.new_subtitles
                    }
                    state['records'].append(record)
                
                # 合并所有标题
                all_subtitles = output.selected_existing + output.new_subtitles
                
                if not all_subtitles:
                    # 叶子节点
                    req['node'].add_citation(req['reference_id'])
                else:
                    # 为每个子标题创建/获取节点，并加入待处理队列
                    for subtitle in all_subtitles:
                        # 获取或创建子节点
                        child_node = req['node'].get_child(subtitle)
                        if child_node is None:
                            child_node = TreeNode(
                                title=subtitle,
                                level=req['node'].level + 1,
                                citations=[],
                                children=[]
                            )
                            req['node'].add_child(child_node)
                        
                        child_node.add_citation(req['reference_id'])
                        
                        # 如果没超过最大深度，加入待处理队列
                        if req['depth'] + 1 < self.max_depth:
                            new_path = f"{req['path']} - {subtitle}"
                            state['pending_requests'].append({
                                'reference_id': req['reference_id'],
                                'content': req['content'],
                                'node': child_node,
                                'path': new_path,
                                'depth': req['depth'] + 1
                            })
                
                # 检查当前文章是否处理完
                if not state['pending_requests']:
                    # 当前文章处理完，移到下一篇
                    state['current_article_idx'] += 1
                    
                    if state['current_article_idx'] < len(state['articles']):
                        # 还有下一篇文章
                        next_article = state['articles'][state['current_article_idx']]
                        state['pending_requests'] = [{
                            'reference_id': next_article['reference_id'],
                            'content': next_article['content'],
                            'node': state['root'],
                            'path': state['root'].title,
                            'depth': 0
                        }]
                    else:
                        # 该topic所有文章都处理完了
                        state['completed'] = True
            
            # 更新错误统计
            if batch_error_stats:
                error_stats['total_success'] += batch_error_stats['success']
                error_stats['total_failed'] += batch_error_stats['failed']
        
        print()  # 换行
        
        # 收集结果和错误统计
        print(f"\n处理完成，收集结果...")
        
        total_records = 0
        for topic_key, state in topic_states.items():
            # 如果使用结构初始化模式，删除没有引用的空节点
            if use_structure_init:
                self._remove_empty_nodes(state['root'])
            
            all_trees[topic_key] = state['root'].to_dict()
            
            if record_mode and state['records']:
                all_records.extend(state['records'])
                total_records += len(state['records'])
            
            # 保存每个topic的错误统计
            error_stats['by_topic'][topic_key] = state['error_stats']
        
        print(f"  - 总topics: {len(all_trees)}")
        if record_mode:
            print(f"  - 总records: {total_records}")
        
        # 保存所有树
        trees_file = output_dir / f'{split_name}_trees.json'
        with open(trees_file, 'w', encoding='utf-8') as f:
            json.dump(all_trees, f, indent=2, ensure_ascii=False)
        print(f"  - 结构树: {trees_file}")
        
        # 保存记录（如果有）
        if record_mode and all_records:
            records_file = output_dir / f'{split_name}_records.json'
            with open(records_file, 'w', encoding='utf-8') as f:
                json.dump(all_records, f, indent=2, ensure_ascii=False)
            print(f"  - 分类记录: {records_file}")
        
        return all_trees, error_stats
    
    def build_trees_for_split_batch(
        self,
        dataset_split: Dict,
        split_name: str,
        output_dir: str,
        record_mode: bool = False
    ) -> Optional[List[Dict]]:
        """
        为数据集划分构建树（跨topic批处理版本）
        
        关键逻辑：
        1. 同一topic内的文章必须串行处理（第1篇完全处理完，第2篇才开始）
        2. 不同topic之间并行（批量调用分类器）
        
        Args:
            dataset_split: 数据集划分
            split_name: 'train', 'val', 或 'test'
            output_dir: 输出目录
            record_mode: 是否记录分类过程
            
        Returns:
            records: 如果record_mode=True，返回所有记录
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        split_data = dataset_split.get(split_name, {})
        print(f"\n{'='*80}")
        print(f"构建 {split_name} 集的结构树（跨topic批处理模式）")
        print(f"{'='*80}")
        
        # 统计信息
        total_topics = len(split_data)
        total_refs = sum(len(refs) for refs in split_data.values())
        print(f"总计: {total_topics} topics, {total_refs} references")
        
        all_records = [] if record_mode else None
        all_trees = {}
        
        # 为每个topic创建根节点和初始状态
        topic_states = {}
        for topic_key, reference_ids in split_data.items():
            if topic_key not in self.references_data:
                continue
            
            topic_data = self.references_data[topic_key]
            topic_name = topic_data['topic']
            
            root = TreeNode(
                title=topic_name,
                level=0,
                citations=[],
                children=[]
            )
            
            # 准备该topic的所有文章内容
            articles = []
            for ref_id in reference_ids:
                if ref_id in topic_data['references']:
                    articles.append({
                        'reference_id': ref_id,
                        'content': topic_data['references'][ref_id].get('content', '')
                    })
            
            topic_states[topic_key] = {
                'root': root,
                'articles': articles,
                'current_article_idx': 0,  # 当前处理到第几篇文章
                'pending_requests': [],  # 当前文章的待处理请求
                'records': [] if record_mode else None,
                'completed': False
            }
        
        # 初始化：每个topic的第一篇文章的第一个请求
        for topic_key, state in topic_states.items():
            if state['articles']:
                article = state['articles'][0]
                state['pending_requests'] = [{
                    'reference_id': article['reference_id'],
                    'content': article['content'],
                    'node': state['root'],
                    'path': state['root'].title,
                    'depth': 0
                }]
        
        # 循环处理，直到所有topic都完成
        iteration = 0
        while True:
            iteration += 1
            
            # 收集所有topic当前的待分类请求
            batch_inputs = []
            batch_metadata = []
            
            for topic_key, state in topic_states.items():
                if state['completed']:
                    continue
                
                # 取出当前topic的一个待处理请求（如果有）
                if state['pending_requests']:
                    req = state['pending_requests'][0]
                    
                    existing_subtitles = [child.title for child in req['node'].children]
                    
                    # 计算约束路径
                    constraint_paths = []
                    if existing_subtitles:
                        constraint_paths = self.select_inference_constraints(
                            state['root'], req['path'], existing_subtitles
                        )
                    
                    input_data = ClassificationInput(
                        topic_key=topic_key,
                        reference_id=req['reference_id'],
                        article_content=req['content'],
                        current_path=req['path'],
                        existing_subtitles=existing_subtitles,
                        constraint_paths=constraint_paths
                    )
                    
                    batch_inputs.append(input_data)
                    batch_metadata.append({
                        'topic_key': topic_key,
                        'request': req
                    })
            
            if not batch_inputs:
                break  # 所有topic都处理完了
            
            if iteration % 10 == 1:  # 每10次迭代输出一次
                active_topics = sum(1 for s in topic_states.values() if not s['completed'])
                print(f"\r迭代 {iteration}: 批量处理 {len(batch_inputs)} 请求 (活跃topics: {active_topics}/{total_topics})", end = '')
            
            # 批量分类
            batch_outputs, _ = self.classifier.classify_batch(batch_inputs)
            
            # 处理分类结果并更新状态
            for input_data, output, metadata in zip(batch_inputs, batch_outputs, batch_metadata):
                topic_key = metadata['topic_key']
                req = metadata['request']
                state = topic_states[topic_key]
                
                # 移除已处理的请求
                state['pending_requests'].pop(0)
                
                # 记录
                if record_mode:
                    record = {
                        'topic_key': topic_key,
                        'reference_id': input_data.reference_id,
                        'current_path': input_data.current_path,
                        'existing_subtitles': input_data.existing_subtitles,
                        'constraint_paths': input_data.constraint_paths,
                        'article_content': input_data.article_content,
                        'selected_existing': output.selected_existing,
                        'new_subtitles': output.new_subtitles
                    }
                    state['records'].append(record)
                
                # 合并所有标题
                all_subtitles = output.selected_existing + output.new_subtitles
                
                if not all_subtitles:
                    # 叶子节点
                    req['node'].add_citation(req['reference_id'])
                else:
                    # 为每个子标题创建/获取节点，并加入待处理队列
                    for subtitle in all_subtitles:
                        # 获取或创建子节点
                        child_node = req['node'].get_child(subtitle)
                        if child_node is None:
                            child_node = TreeNode(
                                title=subtitle,
                                level=req['node'].level + 1,
                                citations=[],
                                children=[]
                            )
                            req['node'].add_child(child_node)
                        
                        child_node.add_citation(req['reference_id'])
                        
                        # 如果没超过最大深度，加入待处理队列
                        if req['depth'] + 1 < self.max_depth:
                            new_path = f"{req['path']} - {subtitle}"
                            state['pending_requests'].append({
                                'reference_id': req['reference_id'],
                                'content': req['content'],
                                'node': child_node,
                                'path': new_path,
                                'depth': req['depth'] + 1
                            })
                
                # 检查当前文章是否处理完
                if not state['pending_requests']:
                    # 当前文章处理完，移到下一篇
                    state['current_article_idx'] += 1
                    
                    if state['current_article_idx'] < len(state['articles']):
                        # 还有下一篇文章
                        next_article = state['articles'][state['current_article_idx']]
                        state['pending_requests'] = [{
                            'reference_id': next_article['reference_id'],
                            'content': next_article['content'],
                            'node': state['root'],
                            'path': state['root'].title,
                            'depth': 0
                        }]
                    else:
                        # 该topic所有文章都处理完了
                        state['completed'] = True
        
        # 收集结果
        print(f"\n处理完成，收集结果...")
        
        # 统计
        total_records = 0
        for topic_key, state in topic_states.items():
            all_trees[topic_key] = state['root'].to_dict()
            
            if record_mode and state['records']:
                all_records.extend(state['records'])
                total_records += len(state['records'])
        
        print(f"  - 总topics: {len(all_trees)}")
        print(f"  - 总records: {total_records}")
        
        # 保存所有树
        trees_file = output_dir / f'{split_name}_trees.json'
        with open(trees_file, 'w', encoding='utf-8') as f:
            json.dump(all_trees, f, indent=2, ensure_ascii=False)
        print(f"  - 结构树: {trees_file}")
        
        # 保存记录（如果有）
        if record_mode and all_records:
            records_file = output_dir / f'{split_name}_records.json'
            with open(records_file, 'w', encoding='utf-8') as f:
                json.dump(all_records, f, indent=2, ensure_ascii=False)
            print(f"  - 分类记录: {records_file}")
        
        return all_records
    
    def build_trees_for_split(
        self,
        dataset_split: Dict,
        split_name: str,
        output_dir: str,
        record_mode: bool = False,
        use_batch: bool = True
    ) -> Optional[List[Dict]]:
        """
        为数据集划分构建树
        
        Args:
            dataset_split: 数据集划分
            split_name: 'train', 'val', 或 'test'
            output_dir: 输出目录
            record_mode: 是否记录分类过程
            use_batch: 是否使用跨topic批处理（推荐True）
            
        Returns:
            records: 如果record_mode=True，返回所有记录
        """
        if use_batch:
            return self.build_trees_for_split_batch(
                dataset_split, split_name, output_dir, record_mode
            )
        
        # 原来的逐个topic处理方式（不使用批处理）
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        split_data = dataset_split.get(split_name, {})
        print(f"\n{'='*80}")
        print(f"构建 {split_name} 集的结构树（单topic处理模式）")
        print(f"{'='*80}")
        
        # 统计信息
        total_topics = len(split_data)
        total_refs = sum(len(refs) for refs in split_data.values())
        print(f"总计: {total_topics} topics, {total_refs} references")
        
        all_records = [] if record_mode else None
        all_trees = {}
        
        for idx, (topic_key, reference_ids) in enumerate(split_data.items(), 1):
            if idx % 10 == 1 or idx == total_topics:  # 每10个topic输出一次
                print(f"  处理进度: {idx}/{total_topics}")
            
            # 构建树
            root, records = self.build_tree(
                topic_key=topic_key,
                reference_ids=reference_ids,
                record_mode=record_mode
            )
            
            # 保存树
            all_trees[topic_key] = root.to_dict()
            
            # 收集记录
            if record_mode and records:
                all_records.extend(records)
        
        # 保存所有树
        trees_file = output_dir / f'{split_name}_trees.json'
        with open(trees_file, 'w', encoding='utf-8') as f:
            json.dump(all_trees, f, indent=2, ensure_ascii=False)
        
        # 保存记录（如果有）
        if record_mode and all_records:
            records_file = output_dir / f'{split_name}_records.json'
            with open(records_file, 'w', encoding='utf-8') as f:
                json.dump(all_records, f, indent=2, ensure_ascii=False)
        
        print(f"\n处理完成:")
        print(f"  - 总topics: {len(all_trees)}")
        print(f"  - 总records: {len(all_records) if all_records else 0}")
        print(f"  - 结构树: {trees_file}")
        if record_mode:
            print(f"  - 分类记录: {records_file}")
        
        return all_records


# 测试代码
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='构建结构树')
    parser.add_argument(
        '--mode',
        type=str,
        default='ground_truth',
        choices=['ground_truth', 'model'],
        help='分类器模式'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='要处理的数据集划分'
    )
    parser.add_argument(
        '--record',
        action='store_true',
        help='记录分类过程'
    )
    parser.add_argument(
        '--references_file',
        type=str,
        default='/mnt/literism/tree/data/wikipedia_references_final.json',
        help='references文件路径'
    )
    parser.add_argument(
        '--split_file',
        type=str,
        default='/mnt/literism/tree/hierarchical_output/data/dataset_split.json',
        help='数据集划分文件路径'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./output',
        help='输出目录'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='/home/literism/model/Qwen3-8B',
        help='模型路径（model模式需要）'
    )
    
    args = parser.parse_args()
    
    # 加载references数据
    print("加载数据...")
    with open(args.references_file, 'r', encoding='utf-8') as f:
        references_data = json.load(f)
    
    # 加载数据集划分
    with open(args.split_file, 'r', encoding='utf-8') as f:
        split_info = json.load(f)
    dataset_split = split_info['dataset_split']
    
    # 创建分类器
    if args.mode == 'ground_truth':
        classifier = Classifier(
            mode='ground_truth',
            references_file=args.references_file
        )
    else:
        if not args.model_path:
            raise ValueError("model模式需要提供--model_path")
        classifier = Classifier(
            mode='model',
            model_path=args.model_path
        )
    
    # 创建构建器
    builder = TreeBuilder(
        classifier=classifier,
        references_data=references_data
    )
    
    # 构建树
    builder.build_trees_for_split(
        dataset_split=dataset_split,
        split_name=args.split,
        output_dir=args.output_dir,
        record_mode=args.record
    )
    
    print("\n构建完成！")

