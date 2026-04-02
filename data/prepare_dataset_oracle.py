"""
Oracle π* 数据生成（简化版本 - 移除延迟归拢）

目标：
- 使用 Oracle 策略 π* 直接决定归拢（不延迟）
- 节点特征使用纯 summary 文本（API模型生成）
- 同时收集两类训练数据：
  1. 分类模型训练数据（prompt + completion）
  2. 总结模型训练数据（prompt + completion）
- 使用多进程Worker + 多线程Topic的架构
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import multiprocessing as mp
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from tqdm import tqdm

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
from summary_based_classifier.llm.deepseek_api import DeepSeekAPIClient, DeepSeekConfig


# ============================================================================
# 辅助函数
# ============================================================================

def collect_docs_in_subtree(node: TreeNode) -> List[str]:
    """收集子树中的所有文档"""
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
    """重新计算节点深度"""
    node.depth = depth
    for c in node.children:
        recompute_depths(c, depth + 1)


def get_root(node: TreeNode) -> TreeNode:
    """获取根节点"""
    cur = node
    while cur.parent is not None:
        cur = cur.parent
    return cur


def insert_parent_path(parent: TreeNode, new_leaf: TreeNode, sibling: TreeNode) -> TreeNode:
    """
    InsertParentPath（受限版本）：只允许把 new_leaf + sibling 归拢到新父节点下。
    返回新创建的父节点。
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


# ============================================================================
# 数据类
# ============================================================================

@dataclass
class OracleSample:
    """分类模型训练样本"""
    prompt: str
    completion: str
    
    topic: str = ""
    article_idx: int = -1
    node_summaries: List[str] = field(default_factory=list)
    merge_with_idx: Optional[int] = None
    target_label: Optional[str] = None


@dataclass
class SummaryGenerationSample:
    """总结模型训练样本"""
    prompt: str
    completion: str
    
    topic: str = ""
    article_idx: int = -1
    node_id: str = ""
    operation: str = ""  # "create", "update", "merge"


# ============================================================================
# Summary生成（使用DeepSeek API）
# ============================================================================
# 说明：使用 deepseek_api.py 的 run_prompts_to_texts 方法
# 该方法内部已实现多线程并行处理


# ============================================================================
# Oracle SFT 生成器（简化版本）
# ============================================================================

class OracleSFTGenerator:
    """
    Oracle策略指导的SFT数据生成器（简化版本）
    - 直接用策略π决定归拢（不延迟）
    - 节点特征=纯summary（API或模型生成）
    - 支持两种模式：
      * api: 使用线上API生成summary/补全
      * model: 使用本地模型Worker生成summary/补全
    两种模式除调用后端不同外，其余数据构建与记录逻辑保持一致。
    """
    
    def __init__(
        self,
        config: SummaryBasedConfig,
        api_config: DeepSeekConfig = None,
        mode: str = "api",
        summary_model_path: str = None
    ):
        self.config = config
        self.api_config = api_config
        self.mode = mode
        
        if mode == "api":
            if api_config is None:
                raise ValueError("mode=api 时必须提供 api_config")
            # 创建DeepSeek API客户端
            self.api_client = DeepSeekAPIClient(api_config)
            self._verify_api_client(self.api_client)
            print("✓ DeepSeek API客户端已创建（API模式）")
        elif mode == "model":
            # 创建模型Worker（稍后初始化）
            self.summary_model_path = summary_model_path
            self.updater_worker = None
            self.updater_prompt_queue = None
            self.updater_result_queue = None
            print("✓ 模型模式已配置（将使用指定模型进行总结与推理补全）")
        else:
            raise ValueError(f"未知的模式: {mode}，应为 'api' 或 'model'")
        
        # 加载tokenizer用于截断文本
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.path.base_model, trust_remote_code=True)
        self.max_content_tokens = config.summary.max_content_length  # 从config读取
        print(f"✓ Tokenizer已加载，最大内容长度: {self.max_content_tokens} tokens")
        
        # 训练数据收集
        self.classification_samples: List[OracleSample] = []
        self.summary_samples: List[SummaryGenerationSample] = []
        self.samples_lock = threading.Lock()
        
        # 统计信息
        self.stats = {
            'total_new': 0,
            'total_select': 0,
            'total_merge': 0,
        }
        self.stats_lock = threading.Lock()
        
        # 进度条（用于显示数据生成进度）
        self.progress_bar = None
        self.progress_lock = threading.Lock()
        
        # Prompt模板
        self.prompt_templates = PromptTemplates()

    @staticmethod
    def _verify_api_client(api_client: DeepSeekAPIClient):
        """启动时做一次最小API连通性检测，失败则立即报错。"""
        try:
            probe_prompt = "Health check. Reply with: OK"
            results = api_client.run_prompts_to_texts([probe_prompt], show_progress=False)
            ok = bool(results and isinstance(results[0], str) and results[0].strip())
            if not ok:
                raise RuntimeError("API返回为空")
            print(f"✓ API连通性检测成功")
        except Exception as e:
            raise RuntimeError(f"API客户端初始化失败或不可用: {e}") from e

    def _extract_json_after_labels(
        self,
        completion_text: str,
        labels: List[str],
        allow_global_fallback: bool = False,
    ) -> Dict:
        """从模型输出中按标签提取JSON字典（支持多行、嵌套、代码块）。"""
        if not isinstance(completion_text, str):
            return {}
        text = completion_text.strip()
        if not text:
            return {}

        def _strip_code_fence(s: str) -> str:
            s = s.strip()
            if s.startswith("```"):
                lines = s.splitlines()
                if lines and lines[0].strip().startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip().startswith("```"):
                    lines = lines[:-1]
                s = "\n".join(lines).strip()
            return s

        def _decode_first_json_obj(s: str):
            s = _strip_code_fence(s).lstrip()
            if not s:
                return None
            decoder = json.JSONDecoder()
            # 1) 先尝试从开头直接解
            try:
                obj, _ = decoder.raw_decode(s)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
            # 2) 再从首个'{'开始解（支持前置说明文本）
            brace = s.find("{")
            if brace != -1:
                try:
                    obj, _ = decoder.raw_decode(s[brace:])
                    if isinstance(obj, dict):
                        return obj
                except Exception:
                    pass
            return None

        # 优先解析 label 后的 JSON（支持多行、嵌套 JSON）
        up = text.upper()
        for label in labels:
            tag = f"{label.upper()}:"
            pos = up.find(tag)
            if pos != -1:
                colon = text.find(":", pos)
                if colon != -1:
                    remainder = text[colon + 1:].lstrip()
                    if remainder:
                        obj = _decode_first_json_obj(remainder)
                        if isinstance(obj, dict):
                            return obj

        # 可选回退：从全文中解析首个完整 JSON 对象（避免正则截断嵌套结构）
        if allow_global_fallback:
            obj = _decode_first_json_obj(text)
            if isinstance(obj, dict):
                return obj
        return {}

    def _extract_article_relevant_content(self, completion_text: str) -> Dict:
        return self._extract_json_after_labels(
            completion_text,
            ["ARTICLE_RELEVANT_CONTENT", "ARTICLE RELEVANT_CONTENT"],
            allow_global_fallback=True,
        )

    def _extract_new_node_direction(self, completion_text: str) -> Dict:
        return self._extract_json_after_labels(
            completion_text,
            ["NEW_NODE_DIRECTION", "NEW NODE DIRECTION"],
        )

    def _extract_merge_signal(self, completion_text: str) -> Dict:
        return self._extract_json_after_labels(
            completion_text,
            ["MERGE_SIGNAL", "MERGE SIGNAL"],
        )

    @staticmethod
    def _apply_thinking_mode(prompt: str) -> str:
        """
        为总结/补全模型开启thinking模式（可通过环境变量关闭）。
        默认开启：SBC_ENABLE_THINKING!=0
        """
        if os.environ.get("SBC_ENABLE_THINKING", "1") == "0":
            return prompt
        return (
            "[Thinking Mode: ON]\n"
            "First think carefully step by step, then output strictly in the required format.\n\n"
            f"{prompt}"
        )

    def _build_classification_completion_with_reasoning(
        self,
        classification_prompt: str,
        selected_indices: List[int],
        need_new: bool,
        merge_with_idx: Optional[int],
        num_categories: int,
        first_uncovered_path: Optional[Sequence[str]] = None,
    ) -> Tuple[str, Dict]:
        """根据策略π结果补充结构化字段，并输出completion与提取结果。"""
        valid_selected = sorted(set(i for i in selected_indices if 0 <= i < num_categories))
        valid_merge = merge_with_idx if (merge_with_idx is not None and 0 <= merge_with_idx < num_categories) else None
        reasoning_prompt = self.prompt_templates.format_classification_reasoning_prompt(
            classification_prompt=classification_prompt,
            matched_categories=valid_selected,
            need_new=need_new,
            merge_with=valid_merge,
            first_uncovered_path=(list(first_uncovered_path) if first_uncovered_path else None),
        )
        completion_from_model = self.generate_classification_completion(reasoning_prompt)
        relevant_content = self._extract_article_relevant_content(completion_from_model)
        new_node_direction = self._extract_new_node_direction(completion_from_model)
        merge_signal = self._extract_merge_signal(completion_from_model)
        
        # 兼容一种常见输出：模型把三段字段包在一个外层JSON里。
        # 例如：
        # ARTICLE_RELEVANT_CONTENT: {
        #   "ARTICLE_RELEVANT_CONTENT": {...},
        #   "NEW_NODE_DIRECTION": {...},
        #   "MERGE_SIGNAL": {...}
        # }
        # 这里做结构化拆包，避免 NEW_NODE_DIRECTION 被误判为空。
        if isinstance(relevant_content, dict):
            wrapper_arc = relevant_content.get("ARTICLE_RELEVANT_CONTENT")
            wrapper_nd = relevant_content.get("NEW_NODE_DIRECTION")
            wrapper_ms = relevant_content.get("MERGE_SIGNAL")
            has_wrapper = any(isinstance(x, dict) for x in [wrapper_arc, wrapper_nd, wrapper_ms])
            if has_wrapper:
                if isinstance(wrapper_arc, dict):
                    relevant_content = wrapper_arc
                if (not new_node_direction) and isinstance(wrapper_nd, dict):
                    new_node_direction = wrapper_nd
                if (not merge_signal) and isinstance(wrapper_ms, dict):
                    merge_signal = wrapper_ms

        if not merge_signal and isinstance(relevant_content, dict):
            nested = relevant_content.get("MERGE_SIGNAL")
            if isinstance(nested, dict):
                merge_signal = nested

        completion = self.prompt_templates.format_classification_completion(
            selected_indices=valid_selected,
            need_new=need_new,
            num_categories=num_categories,
            merge_with=valid_merge,
            relevant_content=relevant_content,
            new_node_direction=new_node_direction,
            merge_signal=merge_signal,
        )
        return completion, {
            "article_relevant_content": relevant_content,
            "new_node_direction": new_node_direction,
            "merge_signal": merge_signal,
        }

    def generate_classification_completion(self, prompt: str, max_retries: int = 3) -> str:
        """
        专用于“分类completion补全”的模型调用。
        注意：不走总结解析逻辑，只返回模型原始文本，避免影响summary生成/更新主流程。
        """
        prompt = self._apply_thinking_mode(prompt)

        if self.mode == "api":
            for attempt in range(max_retries):
                try:
                    results = self.api_client.run_prompts_to_texts([prompt], show_progress=False)
                    if results and results[0]:
                        return results[0]
                except Exception:
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(1)
                        continue
                    return ""
            return ""

        elif self.mode == "model":
            import uuid
            from summary_based_classifier.models.model_workers import PromptRequest

            for attempt in range(max_retries):
                try:
                    prompt_id = str(uuid.uuid4())
                    self.updater_prompt_queue.put(PromptRequest(
                        prompt_id=prompt_id,
                        prompt=prompt,
                        context={
                            'task': 'classification_completion',
                            'n': 1,
                            'temperature': 0.0,
                            'max_tokens': 2048,
                        }
                    ))

                    while True:
                        result = self.updater_result_queue.get(timeout=600)
                        if result.prompt_id == prompt_id:
                            if result.result and len(result.result) > 0:
                                completion_output = result.result[0]
                                return completion_output.raw_response if hasattr(completion_output, 'raw_response') else ""
                            break
                        self.updater_result_queue.put(result)
                        import time
                        time.sleep(0.01)
                except Exception:
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(0.1)
                        continue
                    return ""
            return ""

        return ""
    
    def truncate_text(self, text: str, max_tokens: int = None) -> str:
        """根据token数截断文本"""
        if max_tokens is None:
            max_tokens = self.max_content_tokens
        
        # 编码文本
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        # 如果超过最大长度，截断
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        
        return text
    
    def generate_summary(self, prompt: str, max_retries: int = 3) -> str:
        """
        生成summary（根据模式调用API或模型，带重试）
        """
        prompt = self._apply_thinking_mode(prompt)

        if self.mode == "api":
            for attempt in range(max_retries):
                try:
                    results = self.api_client.run_prompts_to_texts([prompt], show_progress=False)
                    if results and results[0]:
                        return results[0]
                except Exception as e:
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(1)  # 等待1秒后重试
                        continue
                    else:
                        # print(f"\n[警告] API调用失败（已重试{max_retries}次）: {e}")
                        return ""
            return ""
        
        elif self.mode == "model":
            # 使用模型Worker生成
            import uuid
            from summary_based_classifier.models.model_workers import PromptRequest
            
            for attempt in range(max_retries):
                try:
                    prompt_id = str(uuid.uuid4())
                    self.updater_prompt_queue.put(PromptRequest(
                        prompt_id=prompt_id,
                        prompt=prompt,
                        context={'n': 1, 'temperature': 0.0}
                    ))
                    
                    # 等待结果
                    while True:
                        result = self.updater_result_queue.get(timeout=600)
                        if result.prompt_id == prompt_id:
                            if result.result and len(result.result) > 0:
                                summary_output = result.result[0]
                                # 返回原始response文本
                                return summary_output.raw_response if hasattr(summary_output, 'raw_response') else ""
                            break
                        # 不是本请求的结果：放回队列，让对应请求线程消费
                        self.updater_result_queue.put(result)
                        import time
                        time.sleep(0.01)
                except Exception as e:
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(0.1)
                        continue
                    else:
                        # print(f"\n[警告] 模型推理失败（已重试{max_retries}次）: {e}")
                        return ""
            return ""
    
    def propagate_summary_updates(self, leaf_node: TreeNode, article: str, topic: str, article_idx: int):
        """
        自下而上传递summary更新
        
        从叶子节点开始，用文章更新叶子的summary，
        然后用更新后的子节点summary更新父节点summary，
        一直向上传递，直到判断不需要更新或到根节点为止。
        
        Args:
            leaf_node: 叶子节点
            article: 文章内容
            topic: topic名称
            article_idx: 文章索引
        """
        current_node = leaf_node
        new_content = article  # 第一次使用文章内容
        
        while current_node is not None:
            # 检查是否是根节点（没有parent）
            if current_node.parent is None:
                # 不更新根节点
                break
            
            # 获取父节点和兄弟节点信息
            parent_summary = ""
            sibling_summaries = []
            if current_node.parent:
                parent_summary = current_node.parent.summary
                sibling_summaries = [c.summary for c in current_node.parent.children if c != current_node]
            
            # 根据token数截断内容
            truncated_content = self.truncate_text(new_content)
            
            # 构建更新prompt
            base_prompt = self.prompt_templates.format_summary_prompt(
                topic_name=topic,
                node_summary=current_node.summary,
                parent_summary=parent_summary,
                sibling_summaries=sibling_summaries,
                new_content=truncated_content,
                target_label=None
            )
            
            # 调用API更新summary
            response = self.generate_summary(base_prompt)
            
            if not response:
                # API调用失败，停止传递
                break
            
            # 解析输出
            parsed = self.prompt_templates.parse_summary_output(response)
            
            if parsed:
                needs_update = parsed.get('needs_update', False)

                # 记录训练数据（API/Model统一记录）
                sample = SummaryGenerationSample(
                    prompt=base_prompt,
                    completion=response,
                    topic=topic,
                    article_idx=article_idx,
                    node_id=str(id(current_node)),
                    operation="update"
                )

                with self.samples_lock:
                    self.summary_samples.append(sample)
                    # 更新进度条
                    if self.progress_bar:
                        with self.progress_lock:
                            self.progress_bar.update(1)

                if needs_update:
                    # 需要更新
                    explanation = parsed.get('explanation', '')
                    scope = parsed.get('scope', '')
                    if explanation and scope:
                        summary = f"EXPLANATION: {explanation}\nSCOPE: {scope}"
                    elif explanation:
                        summary = explanation
                    else:
                        summary = scope
                    
                    # 更新节点summary
                    current_node.summary = summary
                    
                    # 准备向上传递：使用当前节点的summary作为新内容
                    new_content = summary
                    current_node = current_node.parent
                else:
                    # 不需要更新，停止传递（但已经记录了训练数据）
                    break
            else:
                # 解析失败，停止传递
                break
    
    def generate_summary_for_node(
        self,
        node: TreeNode,
        article: str,
        topic: str,
        article_idx: int,
        is_new_node: bool = False,
        new_node_direction: Optional[Dict] = None,
    ):
        """
        为节点生成/更新summary
        同时记录summary生成的训练数据
        
        Args:
            is_new_node: 如果是新创建的节点，则创建summary；否则更新summary
        """
        # 获取父节点和兄弟节点信息
        parent_summary = ""
        sibling_summaries = []
        if node.parent:
            parent_summary = node.parent.summary
            sibling_summaries = [c.summary for c in node.parent.children if c != node]
        
        # 使用prompts.py中的format_summary_prompt
        # 根据token数截断文章
        truncated_article = self.truncate_text(article)
        
        # 构建基础prompt（不包含guidance，用于记录训练数据）
        base_prompt = self.prompt_templates.format_summary_prompt(
            topic_name=topic,
            node_summary=node.summary if not is_new_node else "",
            parent_summary=parent_summary,
            sibling_summaries=sibling_summaries,
            new_content=truncated_article,
            target_label=None,  # 基础prompt不包含guidance
            new_node_direction=(new_node_direction if is_new_node else None),
        )
        
        # 在生成数据时，额外添加guidance给API（不记录到训练数据中）
        target_label = getattr(node, 'target_label', None) if is_new_node else None
        if target_label:
            # 提取target_label的最后部分作为引导
            guidance_text = f"\n\n**Guidance**: The correct classification path for the current node is {target_label}, and the summary you output should align closely with it."
            prompt_with_guidance = base_prompt + guidance_text
        else:
            prompt_with_guidance = base_prompt
        
        operation = "create" if is_new_node else "update"
        
        # 调用API生成summary（使用带guidance的prompt）
        response = self.generate_summary(prompt_with_guidance)
        
        if not response:
            # API调用失败，使用fallback
            if is_new_node:
                node.summary = ""
            # 如果是更新，保持原summary不变
            return
        
        # 解析输出
        parsed = self.prompt_templates.parse_summary_output(response)
        
        if parsed:
            needs_update = parsed.get('needs_update', False)
            
            # 记录summary训练数据（API/Model统一记录）
            sample = SummaryGenerationSample(
                prompt=base_prompt,  # 使用不带guidance的基础prompt
                completion=response,  # 保存原始输出，包含NEEDS_UPDATE等信息
                topic=topic,
                article_idx=article_idx,
                node_id=str(id(node)),
                operation=operation
            )
            
            with self.samples_lock:
                self.summary_samples.append(sample)
                # 更新进度条
                if self.progress_bar:
                    with self.progress_lock:
                        self.progress_bar.update(1)
            
            # 对于新节点，强制更新（不检查needs_update）
            if is_new_node or needs_update:
                # 需要更新：组合EXPLANATION和SCOPE作为summary
                explanation = parsed.get('explanation', '')
                scope = parsed.get('scope', '')
                if explanation and scope:
                    summary = f"EXPLANATION: {explanation}\nSCOPE: {scope}"
                elif explanation:
                    summary = explanation
                else:
                    summary = scope
                
                # 更新节点
                if summary:
                    node.summary = summary
                elif is_new_node:
                    # 新节点但没有生成有效summary，使用fallback
                    label_suffix = getattr(node, 'target_label', 'unknown')
                    if label_suffix:
                        label_parts = label_suffix.split(" - ")
                        label_suffix = label_parts[-1] if label_parts else label_suffix
                    article_snippet = self.truncate_text(article, max_tokens=100)
                    node.summary = f"[{label_suffix}] {article_snippet}"
                    # print(f"\n[警告] 新节点 {getattr(node, 'target_label', 'unknown')} 的summary生成失败，使用fallback")
            else:
                # 不需要更新，保持原summary不变（但已经记录了训练数据）
                pass
        else:
            # 解析失败
            if is_new_node:
                # 新节点解析失败，使用fallback
                label_suffix = getattr(node, 'target_label', 'unknown')
                if label_suffix:
                    label_parts = label_suffix.split(" - ")
                    label_suffix = label_parts[-1] if label_parts else label_suffix
                article_snippet = self.truncate_text(article, max_tokens=100)
                node.summary = f"[{label_suffix}] {article_snippet}"
                # print(f"\n[警告] 新节点 {getattr(node, 'target_label', 'unknown')} 的summary解析失败，使用fallback")
    
    def merge_summaries(self, children: List[TreeNode], parent_node: TreeNode, topic: str, article_idx: int) -> str:
        """
        合并多个子节点的summary
        同时记录训练数据
        """
        # 将所有子节点的summary拼接作为new_content
        child_summaries_text = "\n\n".join([f"Child {i+1}: {c.summary}" for i, c in enumerate(children)])
        
        # 获取父节点的兄弟节点
        sibling_summaries = []
        if parent_node.parent:
            sibling_summaries = [c.summary for c in parent_node.parent.children if c != parent_node]
        
        # 构建基础prompt（不包含guidance，用于记录训练数据）
        base_prompt = self.prompt_templates.format_summary_prompt(
            topic_name=topic,
            node_summary="",  # 新父节点，summary为空
            parent_summary=parent_node.parent.summary if parent_node.parent else "",
            sibling_summaries=sibling_summaries,
            new_content=child_summaries_text,
            target_label=None  # 基础prompt不包含guidance
        )
        
        # 在生成数据时，额外添加guidance给API（不记录到训练数据中）
        target_label = getattr(parent_node, 'target_label', None)
        if target_label:
            label_parts = target_label.split(" - ")
            current_label = label_parts[-1] if label_parts else target_label
            guidance_text = f"\n\n**Guidance**: The correct classification path for the current node is {target_label}, and the summary you output should align closely with it."
            prompt_with_guidance = base_prompt + guidance_text
        else:
            prompt_with_guidance = base_prompt
        
        # 调用API生成（使用带guidance的prompt）
        response = self.generate_summary(prompt_with_guidance)
        
        if not response:
            # API调用失败，使用fallback
            summary = ""
            return summary
        
        # 解析输出
        parsed = self.prompt_templates.parse_summary_output(response)
        
        if parsed:
            needs_update = parsed.get('needs_update', False)
            
            # 记录训练数据（API/Model统一记录）
            sample = SummaryGenerationSample(
                prompt=base_prompt,  # 使用不带guidance的基础prompt
                completion=response,
                topic=topic,
                article_idx=article_idx,
                node_id="merge",
                operation="merge"
            )
            
            with self.samples_lock:
                self.summary_samples.append(sample)
                # 更新进度条
                if self.progress_bar:
                    with self.progress_lock:
                        self.progress_bar.update(1)
            
            if needs_update:
                # 需要更新
                explanation = parsed.get('explanation', '')
                scope = parsed.get('scope', '')
                if explanation and scope:
                    summary = f"EXPLANATION: {explanation}\nSCOPE: {scope}"
                elif explanation:
                    summary = f"EXPLANATION: {explanation}"
                else:
                    summary = f"SCOPE: {scope}"
            else:
                # 不需要更新，使用fallback
                summary = children[0].summary if children else ""
        else:
            # 解析失败，使用fallback
            summary = children[0].summary if children else ""
        
        return summary
    
    def load_data(self):
        """加载数据集"""
        print("\n加载数据...")
        
        with open(self.config.path.references_file, 'r', encoding='utf-8') as f:
            self.references_data = json.load(f)
        
        with open(Path(self.config.path.data_dir) / 'dataset_split.json', 'r', encoding='utf-8') as f:
            split_data = json.load(f)
            self.dataset_split = split_data['dataset_split']
        
        # groundtruth就是structures_file（包含层级结构）
        with open(self.config.path.structures_file, 'r', encoding='utf-8') as f:
            self.groundtruth = json.load(f)
        
        print(f"  ✓ 加载完成")
        print(f"  - Train topics: {len(self.dataset_split.get('train', {}))}")
    
    def _oracle_route_multi(
        self,
        current: TreeNode,
        parts_list: List[Sequence[str]],
        topic: str,
        article_idx: int,
        article: str,
        ref_id: str
    ) -> List[TreeNode]:
        """
        多路径Oracle路由（使用队列处理）
        
        Args:
            current: 起始节点（通常是root）
            parts_list: 多条路径，例如 [['A','B','C'], ['A','B','D']]
            
        Returns:
            到达的所有叶子节点列表
        """
        from collections import deque
        
        # 队列：每个元素是 (当前节点, 剩余路径列表)
        # 初始化：从root开始，每条完整路径都要处理
        queue = deque([(current, parts_list)])
        final_leaves = []
        
        while queue:
            cur_node, remaining_paths = queue.popleft()
            
            if not remaining_paths or all(len(p) == 0 for p in remaining_paths):
                # 所有路径都处理完了，到达叶子
                if cur_node not in final_leaves:
                    final_leaves.append(cur_node)
                continue
            
            # 过滤掉空路径
            remaining_paths = [p for p in remaining_paths if len(p) > 0]
            if not remaining_paths:
                if cur_node not in final_leaves:
                    final_leaves.append(cur_node)
                continue
            
            # 设置当前节点的target_label（使用第一个路径的祖先）
            if not hasattr(cur_node, 'target_label') or cur_node.target_label is None:
                cur_node.target_label = anc(remaining_paths[0], 0)
            
            # 提取所有子节点的target_labels
            child_target_labels = [
                getattr(child, 'target_label', None) or ""
                for child in cur_node.children
            ]
            
            # 构造从根到当前节点的完整路径前缀
            if hasattr(cur_node, 'target_label') and cur_node.target_label and cur_node.target_label != 'ROOT':
                current_prefix = cur_node.target_label.split(" - ")
            else:
                current_prefix = []
            
            # 构造所有完整的article_parts（当前节点前缀 + 剩余路径）
            all_article_parts = [current_prefix + p for p in remaining_paths]
            
            # 直接把所有paths给Oracle策略π
            # 对于每个子节点，检查哪些paths应该分到它
            child_to_paths = {}  # {child_idx: [path_indices]}
            
            for child_idx, child_target_label in enumerate(child_target_labels):
                if not child_target_label:
                    continue
                
                # 检查哪些paths匹配这个子节点
                matched_path_indices = []
                for path_idx, article_parts in enumerate(all_article_parts):
                    # 使用Oracle策略判断
                    matched_indices = decide_top_down_child_by_target_label(
                        child_target_labels=[child_target_label],
                        article_parts=article_parts
                    )
                    if matched_indices:
                        matched_path_indices.append(path_idx)
                
                if matched_path_indices:
                    child_to_paths[child_idx] = matched_path_indices
            
            # 找出被覆盖的path索引和未被覆盖的path索引
            covered_path_indices = set()
            for matched_indices in child_to_paths.values():
                covered_path_indices.update(matched_indices)
            
            # 找到第一个未被覆盖的path
            first_uncovered_path = None
            first_uncovered_idx = None
            for idx, path in enumerate(remaining_paths):
                if idx not in covered_path_indices:
                    first_uncovered_path = path
                    first_uncovered_idx = idx
                    break
            
            # 记录分类样本
            # 判断是否需要创建新类：检查是否有未被覆盖的path
            if first_uncovered_path is not None:
                # 有未覆盖的path，需要创建新类
                # 使用第一个未覆盖的path来创建节点
                first_new_paths = [first_uncovered_path]

                # 在修改树结构前先确定 merge 目标并记录分类样本
                # 否则新建子节点会污染当前节点children，导致prompt/completion标签错位
                merge_idx = None
                k = cur_node.depth
                target_anc = anc(first_uncovered_path, k + 1)
                for idx, child in enumerate(cur_node.children):
                    child_label = getattr(child, 'target_label', None)
                    if not child_label:
                        continue
                    child_parts = child_label.split(" - ") if " - " in child_label else [child_label]
                    child_anc = anc(child_parts, k + 1)
                    if child_anc == target_anc:
                        merge_idx = idx
                        break

                all_selected = list(child_to_paths.keys())
                extracted_fields = self._record_classification_sample(
                    current_node=cur_node,
                    article=article,
                    topic=topic,
                    article_idx=article_idx,
                    selected_indices=all_selected,
                    need_new=True,
                    merge_with_idx=merge_idx,
                    first_uncovered_path=first_uncovered_path,
                )
                new_node_direction = extracted_fields.get("new_node_direction", {}) if isinstance(extracted_fields, dict) else {}
                
                # 创建新叶子
                new_leaf = TreeNode(summary="", citations=[], children=[])
                
                # 设置target_label：对齐到能覆盖这个path的、最深的节点
                # 当前节点的target_label已经是完整路径（如 "Charles Darwin - Biography"）
                # 需要拆分成parts，然后加上剩余路径
                if hasattr(cur_node, 'target_label') and cur_node.target_label and cur_node.target_label != 'ROOT':
                    # 拆分当前节点的target_label
                    current_parts = cur_node.target_label.split(" - ")
                    # 完整路径 = 当前节点的parts + 未覆盖path
                    full_path = current_parts + first_uncovered_path
                else:
                    # 根节点，直接使用未覆盖path
                    full_path = first_uncovered_path
                
                # 拼接成完整的target_label
                new_leaf.target_label = " - ".join(full_path) if full_path else first_uncovered_path[0]
                
                cur_node.add_child(new_leaf)
                
                # 添加文章引用
                new_leaf.citations.append(ref_id)
                
                # 为新节点生成初始summary（直接生成，不判断是否需要更新）
                self.generate_summary_for_node(
                    new_leaf,
                    article,
                    topic,
                    article_idx,
                    is_new_node=True,
                    new_node_direction=new_node_direction,
                )
                
                # 新节点创建后，从父节点开始向上传递更新
                # 因为新节点的内容会影响其父节点的summary
                if new_leaf.parent and new_leaf.parent.parent:  # 确保父节点不是根节点
                    # 从父节点开始向上传播，使用新节点的summary作为新内容
                    self.propagate_summary_updates(new_leaf.parent, new_leaf.summary, topic, article_idx)
                
                # 检查是否需要归拢
                # 使用Oracle策略：比较k+1层的祖先
                merge_indices = [merge_idx] if merge_idx is not None else []
                
                if merge_indices:
                    # 直接执行归拢
                    sibling = cur_node.children[merge_indices[0]]
                    new_parent = insert_parent_path(cur_node, new_leaf, sibling)
                    
                    # 生成merged summary
                    merged_summary = self.merge_summaries(
                        children=[new_leaf, sibling],
                        parent_node=new_parent,
                        topic=topic,
                        article_idx=article_idx
                    )
                    new_parent.summary = merged_summary
                    
                    # 设置target_label：新父节点对齐到两个子节点target_label的最长公共前缀
                    # 例如：new_leaf的target_label = "A-a1-a11"
                    #      sibling的target_label = "A-a1-a12"
                    #      最长公共前缀 = "A-a1"
                    new_leaf_parts = new_leaf.target_label.split(" - ") if " - " in new_leaf.target_label else [new_leaf.target_label]
                    sibling_parts = sibling.target_label.split(" - ") if " - " in sibling.target_label else [sibling.target_label]
                    
                    # 计算最长公共前缀
                    common_prefix = []
                    for i in range(min(len(new_leaf_parts), len(sibling_parts))):
                        if new_leaf_parts[i] == sibling_parts[i]:
                            common_prefix.append(new_leaf_parts[i])
                        else:
                            break
                    
                    # 设置新父节点的target_label
                    new_parent.target_label = " - ".join(common_prefix) if common_prefix else ""
                    
                    with self.stats_lock:
                        self.stats['total_merge'] += 1
                    
                    # 归拢后不需要继续处理
                    # 因为新叶子已经对齐到完整路径（例如 "A-a1-a11"）
                    # 归拢只是把两个已对齐的节点放到共同父节点下
                    # 不需要再递归创建节点
                else:
                    # 不归拢，直接到达新叶子
                    with self.stats_lock:
                        self.stats['total_new'] += 1
                    
                    # 新叶子已经对齐到完整路径，不需要继续处理
                    # 例如：new_leaf.target_label = "A-a1-a11"，已经是最深节点了
                    # 不再加入队列（因为已经到达叶子节点）
                    pass
                
            else:
                # 没有新类，只有已有类
                # 记录分类样本（SELECT，多分类）
                all_selected = list(child_to_paths.keys())
                self._record_classification_sample(
                    current_node=cur_node,
                    article=article,
                    topic=topic,
                    article_idx=article_idx,
                    selected_indices=all_selected,
                    need_new=False,
                    merge_with_idx=None,
                    first_uncovered_path=None,
                )
                
                with self.stats_lock:
                    self.stats['total_select'] += len(all_selected)
            
            # 对所有选中的已有类，递归处理
            for child_idx, path_indices in child_to_paths.items():
                selected_child = cur_node.children[child_idx]
                
                # 根据选中子节点的深度来剥离路径
                child_label = getattr(selected_child, 'target_label', None)
                if child_label and child_label != 'ROOT':
                    child_depth = len(child_label.split(" - "))
                else:
                    child_depth = 1
                
                # 获取对应的paths并剥离
                child_paths = [remaining_paths[idx] for idx in path_indices]
                next_paths = [p[child_depth:] for p in child_paths if len(p) > child_depth]
                
                if next_paths and any(len(p) > 0 for p in next_paths):
                    queue.append((selected_child, next_paths))
                else:
                    # 路径已经到达叶子节点
                    if selected_child not in final_leaves:
                        final_leaves.append(selected_child)
        
        # 对所有到达的叶子节点，添加文章引用并自下而上更新summary
        for leaf in final_leaves:
            if ref_id not in leaf.citations:
                leaf.citations.append(ref_id)
                # 只有选择了已有节点时才需要更新（新创建的节点已经生成过summary）
                if len(leaf.citations) > 1:
                    # 使用自下而上的summary更新传递
                    self.propagate_summary_updates(leaf, article, topic, article_idx)
        
        return final_leaves
    
    def _record_classification_sample(
        self,
        current_node: TreeNode,
        article: str,
        topic: str,
        article_idx: int,
        selected_indices: List[int],
        need_new: bool,
        merge_with_idx: Optional[int],
        first_uncovered_path: Optional[Sequence[str]] = None,
    ) -> Dict:
        """记录分类训练样本（支持多分类）"""
        # 根据token数截断文章（分类用较短的）
        truncated_article = self.truncate_text(article, max_tokens=1000)
        
        # 构造prompt
        prompt = self.prompt_templates.format_classification_prompt(
            topic_name=topic,
            current_summary=current_node.summary,
            article_content=truncated_article,
            child_summaries=[c.summary for c in current_node.children],
            current_depth=current_node.depth,
            num_children=len(current_node.children)
        )
        
        # 构造completion（推理 + JSON），JSON由Oracle结果固定
        completion, extracted_fields = self._build_classification_completion_with_reasoning(
            classification_prompt=prompt,
            selected_indices=selected_indices,
            need_new=need_new,
            merge_with_idx=merge_with_idx,
            num_categories=len(current_node.children),
            first_uncovered_path=first_uncovered_path,
        )
        
        # 创建样本
        sample = OracleSample(
            prompt=prompt,
            completion=completion,
            topic=topic,
            article_idx=article_idx,
            node_summaries=[c.summary for c in current_node.children],
            merge_with_idx=merge_with_idx,
            target_label=getattr(current_node, 'target_label', None)
        )
        
        with self.samples_lock:
            self.classification_samples.append(sample)
            # 更新进度条
            if self.progress_bar:
                with self.progress_lock:
                    self.progress_bar.update(1)
        return extracted_fields
    
    def _process_single_topic(self, topic_key: str, ref_ids: List[str], thread_id: int = 0) -> Dict:
        """处理单个topic（线程函数）"""
        try:
            topic_data = self.references_data[topic_key]
            topic_name = topic_data.get('topic', topic_key)
            
            # 打乱文章顺序
            shuffled_ref_ids = ref_ids.copy()
            random.shuffle(shuffled_ref_ids)
            
            # 创建根节点
            root = TreeNode(summary="", citations=[], children=[], depth=0)
            root.target_label = "ROOT"
            
            # 逐篇处理文章
            for article_idx, ref_id in enumerate(shuffled_ref_ids):
                ref = topic_data['references'].get(ref_id)
                if not ref:
                    continue
                
                article = ref.get('content', '')
                
                # 从ref中读取路径（已包含在references文件中）
                paths = ref.get('paths', [])
                if not paths:
                    continue
                
                # 将路径字符串分割成parts列表
                # paths是字符串列表，例如：['The Hobbit - Influences - Norse mythology']
                parts_list = []
                for path_str in paths:
                    parts = [p.strip() for p in path_str.split(' - ')]
                    parts_list.append(parts)
                
                # 使用多路径路由（一次性处理所有路径）
                leaves = self._oracle_route_multi(
                    current=root,
                    parts_list=parts_list,
                    topic=topic_name,
                    article_idx=article_idx,
                    article=article,
                    ref_id=ref_id
                )
            
            return {
                'topic_key': topic_key,
                'topic_name': topic_name,
                'num_articles': len(shuffled_ref_ids),
                'success': True
            }
            
        except Exception as e:
            print(f"\n[错误] Topic {topic_key} 处理失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                'topic_key': topic_key,
                'success': False,
                'error': str(e)
            }
    
    def generate(self, max_topics: int = None, num_workers: int = 4, max_samples_per_type: int = None, updater_gpus: str = "0,1"):
        """
        生成训练数据（多线程并行）
        
        Args:
            max_topics: 最多处理的topic数量
            num_workers: 并行线程数
            max_samples_per_type: 每种类型的最大样本数，达到后退出（None表示不限制）
            updater_gpus: 模型模式下使用的GPU列表（如"0,1"）
        """
        # 如果是模型模式，初始化UpdaterWorker
        if self.mode == "model":
            gpu_list = [g.strip() for g in str(updater_gpus).split(",") if g.strip()]
            if not gpu_list:
                gpu_list = ["0"]
            tp_size = len(gpu_list)
            visible_gpus = ",".join(gpu_list)
            print(f"\n{'='*60}")
            print("初始化总结模型Worker...")
            print(f"  - 模型路径: {self.summary_model_path}")
            print(f"  - GPUs: {visible_gpus}")
            print(f"  - Tensor Parallel Size: {tp_size}")
            print(f"{'='*60}")
            
            import multiprocessing as mp
            from summary_based_classifier.models.model_workers import UpdaterWorker, PromptRequest
            
            mp_ctx = mp.get_context('spawn')
            self.updater_prompt_queue = mp_ctx.Queue()
            self.updater_result_queue = mp_ctx.Queue()
            
            self.updater_worker = UpdaterWorker(
                model_path=self.summary_model_path,
                prompt_queue=self.updater_prompt_queue,
                result_queue=self.updater_result_queue,
                gpu_id=visible_gpus,
                tensor_parallel_size=tp_size,
                batch_size=32,
                timeout=0.1,
                max_model_len=self.config.inference.max_model_len if hasattr(self.config, 'inference') else 16384,
                gpu_memory_utilization=0.9,
                temperature=0.0
            )
            self.updater_worker.start()
            print("✓ 总结模型Worker已启动\n")
        
        train_topics = self.dataset_split.get('train', {})
        
        if max_topics:
            train_topics = dict(list(train_topics.items())[:max_topics])
        
        print(f"\n{'='*80}")
        print(f"开始生成Oracle训练数据 ({self.mode.upper()}模式)")
        print(f"  - Topic数量: {len(train_topics)}")
        print(f"  - 并行线程数: {num_workers}")
        if max_samples_per_type:
            print(f"  - 样本数目标: 每种类型 {max_samples_per_type} 条")
        print(f"{'='*80}\n")
        
        # 使用线程池并行处理topics
        results = []
        
        # 创建全局进度条（显示生成的数据条数）
        if max_samples_per_type:
            # 如果设置了目标样本数，使用目标值作为total
            # 分类 + Summary 各一份（API/Model保持一致）
            total_target = max_samples_per_type * 2
            
            self.progress_bar = tqdm(
                total=total_target,
                desc="生成数据",
                unit="条",
                ncols=100
            )
        else:
            # 如果没有设置目标，不显示total
            self.progress_bar = tqdm(
                desc="生成数据",
                unit="条",
                ncols=100
            )
        
        executor = ThreadPoolExecutor(max_workers=num_workers)
        should_exit = False
        try:
            # 为每个任务分配thread_id
            topic_items = list(train_topics.items())
            futures = {}
            
            for idx, (topic_key, ref_ids) in enumerate(topic_items):
                thread_id = idx % num_workers  # 循环分配thread_id
                future = executor.submit(self._process_single_topic, topic_key, ref_ids, thread_id)
                futures[future] = topic_key
            
            for future in as_completed(futures):
                if should_exit:
                    break
                
                topic_key = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # 打印当前样本统计（每处理完一个topic）
                    if result.get('success'):
                        cls_count = len(self.classification_samples)
                        sum_count = len(self.summary_samples)
                        tqdm.write(f"✓ {result.get('topic_name', topic_key)}: "
                                   f"分类样本={cls_count}, Summary样本={sum_count}")
                        
                        # 检查是否达到目标样本数（API/Model保持一致）
                        should_stop = False
                        if max_samples_per_type:
                            should_stop = (
                                cls_count >= max_samples_per_type
                                and sum_count >= max_samples_per_type
                            )
                        
                        if should_stop:
                            tqdm.write(f"\n{'='*80}")
                            tqdm.write(f"✓ 已达到目标样本数！提前退出。")
                            tqdm.write(f"  - 分类样本: {cls_count}/{max_samples_per_type}")
                            tqdm.write(f"  - Summary样本: {sum_count}/{max_samples_per_type}")
                            tqdm.write(f"{'='*80}\n")
                            self.progress_bar.close()
                            
                            # 设置退出标志
                            should_exit = True
                            
                            # 取消剩余的任务
                            for remaining_future in futures:
                                if not remaining_future.done():
                                    remaining_future.cancel()
                            
                            # 立即关闭executor，不等待剩余任务
                            executor.shutdown(wait=False)
                            break
                    
                except Exception as e:
                    tqdm.write(f"\n[错误] Topic {topic_key} 处理异常: {e}")
                    results.append({
                        'topic_key': topic_key,
                        'success': False,
                        'error': str(e)
                    })
        finally:
            # 确保executor被关闭
            executor.shutdown(wait=False)
            # 关闭进度条
            if not should_exit and self.progress_bar:
                self.progress_bar.close()
        
        # 如果提前退出，直接返回，不打印最终统计
        if should_exit:
            return
        
        # 统计
        success_count = sum(1 for r in results if r['success'])
        print(f"\n{'='*80}")
        print(f"数据生成完成")
        print(f"  - 成功处理: {success_count}/{len(train_topics)}")
        print(f"  - 分类样本数: {len(self.classification_samples)}")
        print(f"  - Summary样本数: {len(self.summary_samples)}")
        print(f"  - 统计:")
        print(f"    * NEW: {self.stats['total_new']}")
        print(f"    * SELECT: {self.stats['total_select']}")
        print(f"    * MERGE: {self.stats['total_merge']}")
        print(f"{'='*80}\n")
        
        # 清理模型Worker（如果是模型模式）
        if self.mode == "model" and self.updater_worker:
            print("停止总结模型Worker...")
            self.updater_worker.stop()
            
            # 清理队列
            for queue in [self.updater_prompt_queue, self.updater_result_queue]:
                if queue:
                    while not queue.empty():
                        try:
                            queue.get_nowait()
                        except:
                            break
                    queue.close()
                    queue.cancel_join_thread()
            
            print("✓ 总结模型Worker已停止\n")
    
    def save_data(self, output_dir: Path):
        """保存训练数据"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存分类数据
        classification_file = output_dir / 'classification_train.jsonl'
        with open(classification_file, 'w', encoding='utf-8') as f:
            for sample in self.classification_samples:
                f.write(json.dumps({
                    'prompt': sample.prompt,
                    'completion': sample.completion,
                    'metadata': {
                        'topic': sample.topic,
                        'article_idx': sample.article_idx,
                        'merge_with_idx': sample.merge_with_idx,
                        'target_label': sample.target_label
                    }
                }, ensure_ascii=False) + '\n')
        
        print(f"✓ 分类训练数据已保存: {classification_file}")
        
        # 保存summary数据
        summary_file = output_dir / 'summary_train.jsonl'
        with open(summary_file, 'w', encoding='utf-8') as f:
            for sample in self.summary_samples:
                f.write(json.dumps({
                    'prompt': sample.prompt,
                    'completion': sample.completion,
                    'metadata': {
                        'topic': sample.topic,
                        'article_idx': sample.article_idx,
                        'node_id': sample.node_id,
                        'operation': sample.operation
                    }
                }, ensure_ascii=False) + '\n')
        
        print(f"✓ Summary训练数据已保存: {summary_file}")


# ============================================================================
# 主程序
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='生成Oracle指导的SFT训练数据（支持API和模型两种模式）')
    parser.add_argument('--config', type=str, default='./configs/default.json')
    parser.add_argument('--mode', type=str, default='model', choices=['api', 'model'], help='生成模式：api（使用API）或model（使用训练好的模型）')
    
    # API模式参数
    parser.add_argument('--api_key', type=str, default="sk-3f2e7fe4ae6e4d588c619bbff9837dac", help='DeepSeek API Key（API模式）')
    parser.add_argument('--api_url', type=str, default='https://api.deepseek.com', help='API Base URL（API模式）')
    parser.add_argument('--api_model', type=str, default='deepseek-chat', help='API模型名称（API模式）')
    parser.add_argument('--temperature', type=float, default=0.1, help='生成温度')
    parser.add_argument('--max_output_tokens', type=int, default=4096, help='最大输出tokens（API模式）')
    parser.add_argument('--max_concurrent_jobs', type=int, default=8, help='API并发数（API模式）')
    
    # 模型模式参数
    parser.add_argument('--summary_model_path', type=str, default=None, help='总结模型路径（模型模式，默认使用config中的base_model）')
    parser.add_argument('--updater_gpu', type=int, default=1, help='总结模型使用的GPU ID（模型模式，兼容旧参数）')
    parser.add_argument('--updater_gpus', type=str, default='0,1,2,3,4,5,6,7', help='总结模型使用的GPU列表（模型模式，如0,1）')
    
    # 通用参数
    parser.add_argument('--max_topics', type=int, default=None, help='最多处理的topic数量')
    parser.add_argument('--num_workers', type=int, default=8, help='并行线程数')
    parser.add_argument('--max_samples_per_type', type=int, default=5000, help='每种类型的最大样本数（达到后退出）')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录')
    
    args = parser.parse_args()
    
    # 加载配置
    config = SummaryBasedConfig.from_json(args.config)
    
    # 输出目录
    output_dir = Path(args.output_dir) if args.output_dir else Path(config.path.data_dir) / f'oracle_data_{args.mode}'
    
    # 根据模式创建生成器
    if args.mode == "api":
        # 创建DeepSeek API配置
        api_config = DeepSeekConfig(
            api_key=args.api_key,
            base_url=args.api_url,
            model=args.api_model,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            max_concurrent_jobs=args.max_concurrent_jobs
        )
        
        # 创建生成器
        generator = OracleSFTGenerator(
            config=config,
            api_config=api_config,
            mode="api"
        )
    else:  # model mode
        summary_model_path = args.summary_model_path or config.path.base_model
        # 创建生成器
        generator = OracleSFTGenerator(
            config=config,
            mode="model",
            summary_model_path=summary_model_path
        )
    
    # 加载数据
    generator.load_data()
    
    # 生成数据
    generator.generate(
        max_topics=args.max_topics,
        num_workers=args.num_workers,
        max_samples_per_type=args.max_samples_per_type,
        updater_gpus=(args.updater_gpus if args.mode == "model" else str(args.updater_gpu))
    )
    
    # 保存数据
    generator.save_data(output_dir)
    
    print("\n✓ 全部完成！")


if __name__ == '__main__':
    main()
