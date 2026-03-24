"""
新的构建系统
按照classify_and_update伪代码实现
"""
from typing import List, Dict, Optional
from trajectory_sampler import TreeNode
from classify_generator import ClassifyGenerator, ClassificationInput
from updater import Updater, SummaryInput


class TreeBuilder:
    """结构树构建器"""
    
    def __init__(
        self,
        classifier: ClassifyGenerator,
        updater: Updater,
        topic_name: str,
        max_depth: int = 10
    ):
        """
        Args:
            classifier: 分类系统
            updater: 总结系统
            topic_name: topic名称
            max_depth: 最大深度
        """
        self.classifier = classifier
        self.updater = updater
        self.topic_name = topic_name
        self.max_depth = max_depth
    
    def classify_and_update(
        self,
        article_id: str,
        article_content: str,
        root: TreeNode
    ) -> List[TreeNode]:
        """
        对一篇文章进行分类和更新
        
        Args:
            article_id: 文章ID
            article_content: 文章内容
            root: 根节点
            
        Returns:
            path列表（从根到叶子的路径）
        """
        current_node = root
        path = [root]
        
        # ========== Top-down classification ==========
        while True:
            # 检查深度
            if current_node.depth >= self.max_depth:
                current_node.add_citation(article_id)
                break
            
            children = current_node.children
            
            # Step 2: classification decision
            child_summaries = [child.summary for child in children]
            classification_input = ClassificationInput(
                article_content=article_content,
                current_node_summary=current_node.summary if current_node.summary else self.topic_name,
                child_summaries=child_summaries,
                topic_name=self.topic_name
            )
            
            outputs = self.classifier.classify_with_sampling(classification_input, n=1)
            if not outputs:
                current_node.add_citation(article_id)
                break
            
            action = outputs[0]
            
            # Step 3: create new class if needed
            if action.need_new:
                # 生成新summary
                updater_input = SummaryInput(
                    topic_name=self.topic_name,
                    node_summary="",
                    parent_summary=current_node.summary if current_node.summary else self.topic_name,
                    sibling_summaries=child_summaries,
                    new_content=article_content
                )
                
                updater_outputs = self.updater.update_summary(updater_input, n_samples=1)
                
                if not updater_outputs or not (updater_outputs[0].explanation and updater_outputs[0].scope):
                    new_summary = None
                else:
                    output = updater_outputs[0]
                    new_summary = f"EXPLANATION: {output.explanation}\nSCOPE: {output.scope}"
                
                if new_summary:
                    
                    # 创建新节点
                    new_node = TreeNode(
                        summary=new_summary,
                        citations=[],
                        children=[]
                    )
                    current_node.add_child(new_node)
                    
                    # Step 4: re-route existing articles if current_node was leaf
                    if len(children) == 0 and len(current_node.citations) > 0:
                        # 重新路由已有文章
                        self._reroute_articles(current_node, article_content)
                    
                    next_node = new_node
                else:
                    # 生成失败，分到第一个已有子节点或停留在当前节点
                    if len(action.selected_indices) > 0:
                        idx = action.selected_indices[0]
                        if 0 <= idx < len(children):
                            next_node = children[idx]
                        else:
                            current_node.add_citation(article_id)
                            break
                    else:
                        current_node.add_citation(article_id)
                        break
            else:
                # 选择已有子节点
                if len(action.selected_indices) > 0:
                    # 选择第一个
                    idx = action.selected_indices[0]
                    if 0 <= idx < len(children):
                        next_node = children[idx]
                    else:
                        current_node.add_citation(article_id)
                        break
                else:
                    # 没有选择任何节点，停留在当前节点
                    current_node.add_citation(article_id)
                    break
            
            path.append(next_node)
            
            # Step 5: termination condition
            if len(next_node.children) == 0 and not action.need_new:
                next_node.add_citation(article_id)
                break
            
            current_node = next_node
        
        # ========== Bottom-up summary update ==========
        for node in reversed(path):
            if node.parent is None:
                break
            
            # 准备输入
            updater_input = SummaryInput(
                topic_name=self.topic_name,
                node_summary=node.summary,
                parent_summary=node.parent.summary if node.parent.summary else self.topic_name,
                sibling_summaries=[sib.summary for sib in node.get_siblings()],
                new_content=article_content
            )
            
            # 调用总结系统
            updater_outputs = self.updater.update_summary(updater_input, n_samples=1)
            
            if not updater_outputs or not updater_outputs[0].needs_update:
                break
            
            update_result = updater_outputs[0]
            if update_result.explanation and update_result.scope:
                node.summary = f"EXPLANATION: {update_result.explanation}\nSCOPE: {update_result.scope}"
        
        return path
    
    def _reroute_articles(self, node: TreeNode, new_article_content: str):
        """
        重新路由节点中的文章（Step 4）
        
        Args:
            node: 当前节点（之前是叶子，现在有了子节点）
            new_article_content: 新文章内容（用于参考，实际不处理）
        """
        # 获取节点中的文章
        articles_to_reroute = list(node.citations)
        node.citations.clear()
        
        # 对每篇文章进行一次分类
        for article_id in articles_to_reroute:
            # 注意：这里我们没有文章内容，所以简化处理
            # 实际应该保存文章内容或从references中读取
            
            # 简化：随机分配到一个子节点
            if node.children:
                child_summaries = [child.summary for child in node.children]
                
                # 这里需要文章内容才能真正分类
                # 简化版：分到第一个子节点
                node.children[0].add_citation(article_id)
    
    def build_tree_for_articles(
        self,
        articles: List[Dict],
        root: TreeNode
    ) -> TreeNode:
        """
        为多篇文章构建结构树
        
        Args:
            articles: 文章列表 [{'id': ..., 'content': ...}, ...]
            root: 根节点
            
        Returns:
            更新后的根节点
        """
        from tqdm import tqdm
        
        for article in tqdm(articles, desc="Building tree"):
            self.classify_and_update(
                article_id=article['id'],
                article_content=article['content'],
                root=root
            )
        
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
        
        # 默认使用前50字符
        if len(summary) > 50:
            return summary[:50] + '...'
        return summary

