"""
Oracle风格的推理处理器
使用与数据生成相同的推理逻辑，但使用训练好的模型
采用多进程Worker和多线程Topic处理实现并行推理
"""
import threading
import time
import uuid
from queue import Queue
from collections import deque
from tqdm import tqdm
from typing import Dict, List, Any, Tuple, Optional
import multiprocessing as mp

from summary_based_classifier.models.model_workers import (
    ClassifierWorker, UpdaterWorker, PromptRequest
)
from summary_based_classifier.core.trajectory.trajectory_sampler import TreeNode
from summary_based_classifier.llm.prompts import PromptTemplates


class OracleStyleInferenceProcessor:
    """Oracle风格的推理处理器"""
    
    def __init__(
        self,
        classify_generator_model: str,
        updater_model: str,
        max_depth: int,
        classify_generator_gpu_id: int = 0,
        updater_gpu_id: int = 1,
        classifier_batch_size: int = 8,
        updater_batch_size: int = 4,
        classifier_timeout: float = 1.0,
        updater_timeout: float = 2.0,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.85,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_workers: int = 4,
        max_content_length: int = 4000,
        tokenizer_name: str = "base_model"
    ):
        """
        Args:
            classify_generator_model: 分类生成模型路径
            updater_model: 总结更新模型路径
            max_depth: 最大树深度
            classify_generator_gpu_id: 分类生成模型GPU ID
            updater_gpu_id: 总结更新模型GPU ID
            classifier_batch_size: 分类器批次大小
            updater_batch_size: 更新器批次大小
            classifier_timeout: 分类器批次超时（秒）
            updater_timeout: 更新器批次超时（秒）
            max_model_len: 模型最大长度
            gpu_memory_utilization: GPU内存利用率
            temperature: 采样温度
            top_p: top_p采样
            max_workers: 最大并行topic数
            max_content_length: 文章最大token长度
            tokenizer_name: tokenizer名称
        """
        self.max_depth = max_depth
        self.max_workers = max_workers
        self.max_content_length = max_content_length
        
        # 创建tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # 创建multiprocessing队列
        mp_ctx = mp.get_context('spawn')
        self.classifier_prompt_queue = mp_ctx.Queue()
        self.classifier_result_queue = mp_ctx.Queue()
        self.updater_prompt_queue = mp_ctx.Queue()
        self.updater_result_queue = mp_ctx.Queue()
        
        # 创建Worker进程
        print(f"\n启动Worker进程...")
        self.classifier_worker = ClassifierWorker(
            model_path=classify_generator_model,
            prompt_queue=self.classifier_prompt_queue,
            result_queue=self.classifier_result_queue,
            gpu_id=classify_generator_gpu_id,
            batch_size=classifier_batch_size,
            timeout=classifier_timeout,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            temperature=temperature
        )
        self.classifier_worker.start()
        
        self.updater_worker = UpdaterWorker(
            model_path=updater_model,
            prompt_queue=self.updater_prompt_queue,
            result_queue=self.updater_result_queue,
            gpu_id=updater_gpu_id,
            batch_size=updater_batch_size,
            timeout=updater_timeout,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            temperature=temperature
        )
        self.updater_worker.start()
        
        print(f"  ✓ Worker进程已启动")
    
    def truncate_text(self, text: str) -> str:
        """根据token数截断文本"""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > self.max_content_length:
            tokens = tokens[:self.max_content_length]
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        return text
    
    def _classify_article(
        self,
        topic_name: str,
        article_content: str,
        parent_summary: str,
        current_node_summary: str,
        child_summaries: List[str],
        max_retries: int = 3
    ) -> Optional[Dict]:
        """
        使用分类模型对文章进行分类（带重试）
        
        Args:
            max_retries: 最大重试次数
        
        Returns:
            ClassificationOutput对象，如果失败则返回None
        """
        # 构造prompt
        prompt = PromptTemplates.format_classification_prompt(
            topic_name=topic_name,
            current_summary=current_node_summary,
            article_content=article_content,
            child_summaries=child_summaries,
            current_depth=0,  # 可以传递实际深度，但这里简化为0
            num_children=len(child_summaries)
        )
        
        # 重试逻辑
        for attempt in range(max_retries):
            # 生成唯一ID并提交到队列
            prompt_id = str(uuid.uuid4())
            self.classifier_prompt_queue.put(PromptRequest(
                prompt_id=prompt_id,
                prompt=prompt,
                context={'num_children': len(child_summaries), 'n': 1, 'temperature': 0.0}
            ))
            
            # 等待结果
            while True:
                result = self.classifier_result_queue.get()
                if result.prompt_id == prompt_id:
                    if result.result and len(result.result) > 0:
                        return result.result[0]  # ClassificationOutput对象
                    break  # 这次尝试失败，继续重试
            
            # 如果不是最后一次尝试，等待一小段时间再重试
            if attempt < max_retries - 1:
                import time
                time.sleep(0.1)
        
        # 所有重试都失败
        return None
    
    def _generate_summary(
        self,
        topic_name: str,
        article_content: str,
        parent_summary: str,
        current_summary: str,
        sibling_summaries: List[str]
    ) -> Optional[str]:
        """
        使用总结模型生成/更新summary
        
        Returns:
            更新后的summary，如果不需要更新则返回None
        """
        # 构造prompt（不包含target_label引导）
        prompt = PromptTemplates.format_summary_prompt(
            topic_name=topic_name,
            node_summary=current_summary,
            parent_summary=parent_summary if parent_summary else topic_name,
            sibling_summaries=sibling_summaries,
            new_content=article_content
        )
        
        # 生成唯一ID并提交到队列
        prompt_id = str(uuid.uuid4())
        self.updater_prompt_queue.put(PromptRequest(
            prompt_id=prompt_id,
            prompt=prompt,
            context={'n': 1, 'temperature': 0.0}
        ))
        
        # 等待结果
        while True:
            result = self.updater_result_queue.get()
            if result.prompt_id == prompt_id:
                if result.result and len(result.result) > 0:
                    summary_output = result.result[0]  # SummaryOutput对象
                    if summary_output.needs_update:
                        return f"EXPLANATION: {summary_output.explanation}\nSCOPE: {summary_output.scope}"
                return None
    
    def _merge_summaries(
        self,
        topic_name: str,
        child_summaries: List[str],
        parent_summary: str
    ) -> str:
        """
        归拢时，为新父节点生成summary
        
        Returns:
            新父节点的summary
        """
        # 构造一个虚拟文章内容（使用子节点的summaries）
        merge_content = "\n\n".join([f"- {s}" for s in child_summaries])
        
        # 使用总结模型
        prompt = PromptTemplates.format_summary_prompt(
            topic_name=topic_name,
            node_summary="",
            parent_summary=parent_summary if parent_summary else topic_name,
            sibling_summaries=[],
            new_content=merge_content
        )
        
        prompt_id = str(uuid.uuid4())
        self.updater_prompt_queue.put(PromptRequest(
            prompt_id=prompt_id,
            prompt=prompt,
            context={'n': 1, 'temperature': 0.0}
        ))
        
        while True:
            result = self.updater_result_queue.get()
            if result.prompt_id == prompt_id:
                if result.result and len(result.result) > 0:
                    summary_output = result.result[0]
                    if summary_output.needs_update and summary_output.scope:
                        return summary_output.scope
                
                # Fallback: 使用第一个子节点的summary
                return child_summaries[0] if child_summaries else ""
    
    def _route_article(
        self,
        topic_name: str,
        article: Dict,
        root: TreeNode,
        parent_summary: str = ""
    ) -> List[TreeNode]:
        """
        对单篇文章进行路由和分类
        类似prepare_dataset_oracle.py的逻辑，但使用训练好的模型
        
        Returns:
            所有被添加文章引用的叶子节点列表（用于后续的summary更新）
        """
        article_content = self.truncate_text(article['content'])
        article_id = article['id']
        
        # 使用队列进行广度优先处理
        queue = deque([(root, parent_summary)])
        
        # 记录所有被添加文章引用的叶子节点
        leaf_nodes = []
        
        while queue:
            cur_node, cur_parent_summary = queue.popleft()
            
            # 深度限制：到达深度限制时，直接将文章归入当前节点，不再创建新节点
            if cur_node.depth >= self.max_depth:
                if article_id not in cur_node.citations:
                    cur_node.citations.append(article_id)
                    leaf_nodes.append(cur_node)
                continue
            
            # 使用分类模型进行分类（无论当前节点是否有子节点都要调用）
            child_summaries = [child.summary for child in cur_node.children]
            classification_output = self._classify_article(
                topic_name=topic_name,
                article_content=article_content,
                parent_summary=cur_parent_summary,
                current_node_summary=cur_node.summary,
                child_summaries=child_summaries,
                max_retries=3
            )
            
            if not classification_output:
                # 分类失败（重试多次仍失败），直接结束该分支，不添加文章
                print(f"\n[警告] 文章 {article_id} 在节点（深度{cur_node.depth}）分类失败，跳过该分支")
                continue
            
            # 处理分类结果：可能同时有 need_new=True 和 selected_indices 非空
            # 这两者并不矛盾，需要都处理
            
            # 1. 处理创建新节点（如果need_new=True）
            new_node = None
            if classification_output.need_new:
                # 创建新节点
                sibling_summaries = []
                if cur_node.parent:
                    sibling_summaries = [c.summary for c in cur_node.parent.children if c != cur_node]
                
                new_summary = self._generate_summary(
                    topic_name=topic_name,
                    article_content=article_content,
                    parent_summary=cur_parent_summary,
                    current_summary="",
                    sibling_summaries=sibling_summaries
                )
                
                if not new_summary:
                    new_summary = article_content[:200]
                
                new_node = TreeNode(
                    summary=new_summary,
                    citations=[article_id],
                    children=[],
                    depth=cur_node.depth + 1,
                    parent=cur_node
                )
                cur_node.children.append(new_node)
                
                # 处理归拢（如果需要）
                if classification_output.merge_with is not None:
                    merge_idx = classification_output.merge_with
                    if isinstance(merge_idx, int) and 0 <= merge_idx < len(cur_node.children) - 1:
                        # 收集要归拢的节点（新节点 + 1个指定已有节点）
                        nodes_to_merge = [new_node, cur_node.children[merge_idx]]
                        # merge会让被归拢子树整体深度+1，因此需按“子树最大深度+1”校验
                        nodes_to_merge = [
                            n for n in nodes_to_merge
                            if n is not None and (self._get_subtree_max_depth(n) + 1) <= self.max_depth
                        ]
                        # 执行归拢
                        if len(nodes_to_merge) >= 2:
                            self._execute_merge(
                                topic_name=topic_name,
                                parent=cur_node,
                                nodes_to_merge=nodes_to_merge,
                                parent_summary=cur_parent_summary
                            )
                # 新节点已创建，是叶子节点，记录到leaf_nodes用于后续summary更新
                if new_node:
                    leaf_nodes.append(new_node)
            
            # 2. 处理选中的已有子节点（如果selected_indices非空）
            # 注意：所有被选中的子节点都需要继续向下分类，不是只处理第一个
            if classification_output.selected_indices:
                selected_indices = classification_output.selected_indices
                if not isinstance(selected_indices, list):
                    selected_indices = [selected_indices]
                
                # 对所有选中的子节点，继续向下分类
                for idx in selected_indices:
                    if isinstance(idx, int) and 0 <= idx < len(cur_node.children):
                        selected_child = cur_node.children[idx]
                        
                        # 检查是否是叶子节点（没有子节点）
                        if not selected_child.children:
                            # 是叶子节点，添加文章引用并记录
                            if article_id not in selected_child.citations:
                                selected_child.citations.append(article_id)
                                leaf_nodes.append(selected_child)
                        else:
                            # 不是叶子节点，继续向下分类
                            queue.append((selected_child, selected_child.summary))
            
            # 如果既没有创建新节点，也没有选中任何节点，说明分类结果有问题
            # 这种情况下，直接结束该分支，不添加文章
            if not classification_output.need_new and not classification_output.selected_indices:
                print(f"\n[警告] 文章 {article_id} 在节点（深度{cur_node.depth}）分类结果无效（既不创建新节点也不选择已有节点），跳过该分支")
        
        return leaf_nodes
    
    def _execute_merge(
        self,
        topic_name: str,
        parent: TreeNode,
        nodes_to_merge: List[TreeNode],
        parent_summary: str
    ):
        """执行归拢操作"""
        if len(nodes_to_merge) < 2:
            return
        
        # 生成新父节点的summary
        child_summaries = [node.summary for node in nodes_to_merge]
        new_parent_summary = self._merge_summaries(
            topic_name=topic_name,
            child_summaries=child_summaries,
            parent_summary=parent_summary
        )
        
        # 创建新父节点
        new_parent = TreeNode(
            summary=new_parent_summary,
            citations=[],
            children=nodes_to_merge.copy(),
            depth=parent.depth + 1,
            parent=parent
        )
        
        # 更新子节点的parent指针
        for node in nodes_to_merge:
            node.parent = new_parent
            # 调整深度
            self._adjust_depth(node, new_parent.depth + 1)
        
        # 从原parent移除被归拢的节点
        parent.children = [c for c in parent.children if c not in nodes_to_merge]
        
        # 添加新父节点
        parent.children.append(new_parent)
    
    def _adjust_depth(self, node: TreeNode, new_depth: int):
        """递归调整节点深度"""
        node.depth = new_depth
        for child in node.children:
            self._adjust_depth(child, new_depth + 1)

    def _get_subtree_max_depth(self, node: TreeNode) -> int:
        """获取以node为根的子树最大深度（按当前depth字段）"""
        if not node.children:
            return node.depth
        return max(self._get_subtree_max_depth(child) for child in node.children)
    
    def _propagate_summary_updates(
        self,
        topic_name: str,
        leaf_node: TreeNode,
        article_content: str
    ):
        """
        自底向上传播summary更新
        """
        current = leaf_node.parent
        
        while current is not None:
            # 获取父节点的summary
            parent_summary = current.parent.summary if current.parent else topic_name
            
            # 获取兄弟节点的summaries
            if current.parent:
                sibling_summaries = [
                    child.summary for child in current.parent.children
                    if child != current
                ]
            else:
                sibling_summaries = []
            
            # 尝试更新当前节点的summary
            updated_summary = self._generate_summary(
                topic_name=topic_name,
                article_content=article_content,
                parent_summary=parent_summary,
                current_summary=current.summary,
                sibling_summaries=sibling_summaries
            )
            
            if updated_summary:
                current.summary = updated_summary
            
            # 向上移动
            current = current.parent
    
    def process_topics(
        self,
        topics_data: Dict[str, Dict[str, Any]],
        max_refs: int = None
    ) -> Dict[str, Dict]:
        """
        并行处理多个topic的推理
        
        Args:
            topics_data: {topic_key: {'topic': topic_name, 'articles': [...]}}
            max_refs: 每个topic最多处理的文章数
            
        Returns:
            {topic_key: tree_dict}
        """
        print(f"\n{'='*60}")
        print(f"开始Oracle风格并行推理")
        print(f"  - Topic数量: {len(topics_data)}")
        print(f"  - 最大并行线程数: {self.max_workers}")
        print(f"{'='*60}")
        
        # 结果存储（线程安全）
        results = {}
        results_lock = threading.Lock()
        
        # 进度条
        pbar = tqdm(total=len(topics_data), desc="推理进度")
        
        def process_topic(topic_key: str, topic_info: Dict):
            """单个topic的推理线程"""
            try:
                topic_name = topic_info['topic']
                articles = topic_info['articles']
                
                if max_refs:
                    articles = articles[:max_refs]
                
                # 创建根节点
                root = TreeNode(summary="", citations=[], children=[], depth=0)
                
                # 逐篇文章处理
                for article in articles:
                    # 路由文章（就地修改树结构）
                    # 返回所有被添加文章引用的叶子节点
                    leaf_nodes = self._route_article(
                        topic_name=topic_name,
                        article=article,
                        root=root,
                        parent_summary=""
                    )
                    
                    # 对于被分类到已有节点的情况，需要自底向上更新summary
                    # 因为该节点现在包含了新的文章内容
                    article_content = self.truncate_text(article['content'])
                    for leaf_node in leaf_nodes:
                        # 只有当节点有多个引用时才更新（说明之前已有文章）
                        if len(leaf_node.citations) > 1:
                            self._propagate_summary_updates(
                                topic_name=topic_name,
                                leaf_node=leaf_node,
                                article_content=article_content
                            )
                
                # 转换为字典
                tree_dict = {
                    'topic': topic_name,
                    'structure': [self._tree_to_dict(child, level=2) for child in root.children]
                }
                
                # 保存结果（线程安全）
                with results_lock:
                    results[topic_key] = tree_dict
                
                pbar.update(1)
                
            except Exception as e:
                print(f"\n[错误] Topic {topic_key} 推理失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 使用线程池处理topics
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(process_topic, topic_key, topic_info): topic_key
                for topic_key, topic_info in topics_data.items()
            }
            
            # 等待所有任务完成
            for future in as_completed(futures):
                topic_key = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"\n[错误] Topic {topic_key} 处理异常: {e}")
        
        # 关闭进度条
        pbar.close()
        
        # 停止Worker进程
        print(f"\n停止Worker进程...")
        self.classifier_worker.stop()
        self.updater_worker.stop()
        
        # 清理资源
        print(f"清理资源...")
        for queue in [self.classifier_prompt_queue, self.classifier_result_queue,
                      self.updater_prompt_queue, self.updater_result_queue]:
            while not queue.empty():
                try:
                    queue.get_nowait()
                except:
                    break
            queue.close()
            queue.cancel_join_thread()
        
        # 强制释放显存
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        time.sleep(2)
        print(f"✓ Worker进程已停止，资源已清理")
        
        print(f"\n{'='*60}")
        print(f"✓ 推理完成")
        print(f"{'='*60}")
        
        return results
    
    def _tree_to_dict(self, node: TreeNode, level: int = 1) -> Dict:
        """将TreeNode转换为字典格式"""
        return {
            'level': level,
            'summary': node.summary,
            'citations': node.citations,
            'children': [self._tree_to_dict(child, level + 1) for child in node.children]
        }
