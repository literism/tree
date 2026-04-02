"""
Prompt模板管理
所有的prompt模板集中在这里，方便统一修改和管理
"""
import json
import re
from typing import Dict, Optional, List


class PromptTemplates:
    """Prompt模板类"""
    
    # ==================== 分类系统Prompt ====================
    CLASSIFICATION_JSON_SEPARATOR = "<<<JSON>>>"
    
    CLASSIFICATION_PROMPT = """You are a hierarchical article classifier. Your task is to determine how an article should be integrated into the existing node hierarchy.

**Your Task**: Follow the reasoning steps below to decide classification actions.

** Step 1: Summarize Parent-Relevant Content  
- Summarize the parts of the article that are relevant to the Current Node Summary.  
- Do NOT quote long excerpts. Use concise semantic summaries.  
- Keep focus on classification-relevant information (theme, scope, focus, intent).  

** Step 2: Decompose into Overlap vs Residual  
- Compare Step 1 summary with each Existing Child Category.  
- Identify:
  1) content overlapping with existing child categories, and  
  2) residual content not covered by any child category.  
- Keep this step explicit and category-aware.
- There are several categories for Existing Child, so in CHILD_OVERLAP_ANALYSIS, several categories need to be output.
- If there are NO Existing Child Categories, set CHILD_OVERLAP_ANALYSIS = {{}}

** Step 3: Decide Matched Categories  
- Use the overlap analysis from Step 2.  
- Output indices of child categories with clear semantic overlap.  
- If none: output []

** Step 4: Decide NEED_NEW from Residual  
- Use only residual (non-overlapping) content from Step 2.  
- If residual forms a coherent and specific uncovered theme -> NEED_NEW = true  
- Otherwise -> NEED_NEW = false

** Step 5: Specify New Node Direction (if NEED_NEW=true)  
- Explicitly define what the new node should focus on, and what it should exclude.  
- This direction will be passed to the summary generator for new-node creation.

** Step 6: Analyze Merge Signal from Residual  
- Based on residual (non-overlapping) content and the new-node direction, analyze whether the new category is still highly related to one existing child category.  
- Provide concise evidence and counter-evidence instead of only giving a hard decision.  
- Keep this analysis short, specific, and category-aware.

** Step 7: Assess Need for Merge  
- Use Step 6 merge analysis to decide whether a newly created category should be merged with one existing child category.  
- Output: single index to merge into, or null.

**Output Format** (STRICT):
ARTICLE_RELEVANT_CONTENT: {{
  "PARENT_RELEVANT_SUMMARY": "string",
  "CHILD_OVERLAP_ANALYSIS": {{
    "Category i": "overlap summary or none"
  }},
  "RESIDUAL_NOVEL_POINTS": ["point1", "point2"]
}}
NEW_NODE_DIRECTION: {{
  "core_focus": "one-sentence direction for the new node",
  "in_scope_points": ["point1", "point2"],
  "out_of_scope_points": ["point1", "point2"],
  "anchor_terms": ["term1", "term2"]
}}
MERGE_SIGNAL: {{
  "highly_related_categories": ["Category i"],
  "evidence_for_merge": "short reason",
  "evidence_against_merge": "short reason",
  "merge_strength": 0.0
}}
MATCHED_CATEGORIES: [list from Step 3]
NEED_NEW: [true/false from Step 4]
MERGE_WITH: [int/null from Step 7]

## Input Content for Processing:
**Topic**: {topic_name}

**Current Node Summary**:
{current_summary}

**Article Content**:
{article_content}

**Existing Child Categories**:
{children_text}

Now classify the article:
"""

    @staticmethod
    def format_classification_prompt(
        topic_name: str,
        current_summary: str,
        article_content: str,
        child_summaries: List[str],
        child_num_children: List[int] = None,
        child_max_depth: List[int] = None,
        current_depth: int = 0,
        num_children: int = None
    ) -> str:
        """格式化分类系统prompt（简化版，无结构信息）"""
        # 构建子节点列表（简化版，不包含结构信息）
        if child_summaries:
            children_text = ""
            for i, summary in enumerate(child_summaries):
                # 直接使用summary文本，不解析复杂格式
                summary_text = str(summary).strip()
                children_text += f"Category {i}:\n{summary_text}\n\n"
        else:
            children_text = "No existing child categories.\n"
        
        # 如果current_summary为空，使用topic_name
        if not current_summary:
            current_summary = f"Root level - Topic: {topic_name}"
        
        return PromptTemplates.CLASSIFICATION_PROMPT.format(
            topic_name=topic_name,
            current_summary=current_summary,
            article_content=article_content[:3000],
            children_text=children_text
        )
    
    @staticmethod
    def parse_classification_output(response: str, num_categories: int) -> Optional[Dict]:
        """
        解析分类系统输出（兼容新四行格式和旧JSON格式）
        
        Args:
            response: 模型返回的文本
            num_categories: 现有类别数量
            
        Returns:
            {'selected_indices': [0, 2], 'need_new': True, 'merge_with': 1} 或 None
        """
        try:
            clean_response = response.strip().replace("```json", "").replace("```", "").strip()
            lines = [ln.strip() for ln in clean_response.splitlines() if ln.strip()]

            def _parse_json_after_label(text: str, label: str):
                up_text = text.upper()
                target = f"{label.upper()}:"
                pos = up_text.find(target)
                if pos == -1:
                    return None
                colon = text.find(":", pos)
                if colon == -1:
                    return None
                remainder = text[colon + 1:].lstrip()
                if not remainder:
                    return None
                try:
                    obj, _ = json.JSONDecoder().raw_decode(remainder)
                    return obj
                except Exception:
                    return None

            def _parse_json_after_any_label(text: str, labels: List[str]):
                for lb in labels:
                    obj = _parse_json_after_label(text, lb)
                    if isinstance(obj, dict):
                        return obj
                return {}

            def _normalize_merge_candidate_probs(raw_probs: Optional[Dict], n_cats: int) -> Dict[str, float]:
                if not isinstance(raw_probs, dict):
                    return {}
                normalized: Dict[str, float] = {}
                for k, v in raw_probs.items():
                    try:
                        prob = float(v)
                    except Exception:
                        continue
                    prob = max(0.0, min(1.0, prob))
                    key = str(k).strip()
                    low = key.lower()
                    if low in {"null", "none"}:
                        normalized["null"] = prob
                        continue
                    idx = None
                    if low.startswith("category"):
                        m = re.search(r"-?\d+", low)
                        if m:
                            idx = int(m.group(0))
                    elif low.isdigit() or (low.startswith("-") and low[1:].isdigit()):
                        idx = int(low)
                    if idx is not None and 0 <= idx < n_cats:
                        normalized[f"Category {idx}"] = prob
                if not normalized:
                    return {}
                total = sum(normalized.values())
                if total > 0:
                    normalized = {k: v / total for k, v in normalized.items()}
                return normalized

            def _extract_merge_candidate_probs(
                merge_signal_obj: Optional[Dict],
                merge_with_idx: Optional[int],
                n_cats: int
            ) -> Dict[str, float]:
                if not isinstance(merge_signal_obj, dict):
                    return {}

                # 兼容旧格式：直接读取candidate_probs
                probs = _normalize_merge_candidate_probs(merge_signal_obj.get("candidate_probs"), n_cats)
                if probs:
                    return probs

                # 新格式：用单一merge_strength构造可阈值化的二项分布
                raw_strength = merge_signal_obj.get("merge_strength")
                if raw_strength is None:
                    return {}
                try:
                    strength = float(raw_strength)
                except Exception:
                    return {}
                strength = max(0.0, min(1.0, strength))

                candidate_idx = merge_with_idx
                if candidate_idx is None:
                    raw_candidates = merge_signal_obj.get("highly_related_categories", [])
                    if isinstance(raw_candidates, list):
                        for c in raw_candidates:
                            key = str(c).strip().lower()
                            idx = None
                            if key.startswith("category"):
                                m = re.search(r"-?\d+", key)
                                if m:
                                    idx = int(m.group(0))
                            elif key.isdigit() or (key.startswith("-") and key[1:].isdigit()):
                                idx = int(key)
                            if idx is not None and 0 <= idx < n_cats:
                                candidate_idx = idx
                                break

                if isinstance(candidate_idx, int) and 0 <= candidate_idx < n_cats:
                    return {
                        f"Category {candidate_idx}": strength,
                        "null": 1.0 - strength,
                    }
                return {"null": 1.0}

            article_relevant_content = _parse_json_after_any_label(
                clean_response,
                ["ARTICLE_RELEVANT_CONTENT", "ARTICLE RELEVANT_CONTENT"],
            )
            new_node_direction = _parse_json_after_any_label(
                clean_response,
                ["NEW_NODE_DIRECTION", "NEW NODE DIRECTION"],
            )
            merge_signal = _parse_json_after_any_label(
                clean_response,
                ["MERGE_SIGNAL", "MERGE SIGNAL"],
            )

            # 兼容旧格式：MERGE_SIGNAL可能写在ARTICLE_RELEVANT_CONTENT内部
            if not merge_signal and isinstance(article_relevant_content, dict):
                nested = article_relevant_content.get("MERGE_SIGNAL")
                if isinstance(nested, dict):
                    merge_signal = nested

            merge_candidate_probs: Dict[str, float] = {}

            # 1) 优先解析当前prompt约定的四行格式
            if any("MATCHED_CATEGORIES:" in ln for ln in lines) and any("NEED_NEW:" in ln for ln in lines):
                matched_line = None
                need_new_line = None
                merge_with_line = None
                for ln in lines:
                    up = ln.upper()
                    if up.startswith("MATCHED_CATEGORIES:"):
                        matched_line = ln
                    elif up.startswith("NEED_NEW:"):
                        need_new_line = ln
                    elif up.startswith("MERGE_WITH:"):
                        merge_with_line = ln

                if matched_line is None or need_new_line is None:
                    return None

                selected_indices: List[int] = []
                matched_raw = matched_line.split(":", 1)[1].strip()
                try:
                    matched_obj = json.loads(matched_raw)
                    if isinstance(matched_obj, list):
                        for v in matched_obj:
                            if isinstance(v, bool):
                                continue
                            idx = int(v)
                            if 0 <= idx < num_categories:
                                selected_indices.append(idx)
                except Exception:
                    for m in re.findall(r"-?\d+", matched_raw):
                        try:
                            idx = int(m)
                        except Exception:
                            continue
                        if 0 <= idx < num_categories:
                            selected_indices.append(idx)
                selected_indices = sorted(set(selected_indices))

                need_new_raw = need_new_line.split(":", 1)[1].strip().lower()
                if need_new_raw in {"true", "yes", "1"}:
                    need_new = True
                elif need_new_raw in {"false", "no", "0"}:
                    need_new = False
                else:
                    return None

                merge_with: Optional[int] = None
                if merge_with_line is not None:
                    merge_raw = merge_with_line.split(":", 1)[1].strip()
                    if merge_raw.lower() not in {"none", "null", ""}:
                        try:
                            merge_idx = int(merge_raw)
                        except Exception:
                            return None
                        if 0 <= merge_idx < num_categories:
                            merge_with = merge_idx
                        else:
                            return None

                if isinstance(merge_signal, dict):
                    merge_candidate_probs = _extract_merge_candidate_probs(
                        merge_signal_obj=merge_signal,
                        merge_with_idx=merge_with,
                        n_cats=num_categories,
                    )

                return {
                    "selected_indices": selected_indices,
                    "need_new": need_new,
                    "merge_with": merge_with,
                    "article_relevant_content": article_relevant_content,
                    "new_node_direction": new_node_direction if isinstance(new_node_direction, dict) else {},
                    "merge_signal": merge_signal if isinstance(merge_signal, dict) else {},
                    "merge_candidate_probs": merge_candidate_probs,
                }

            # 2) 回退解析旧JSON格式
            marker = PromptTemplates.CLASSIFICATION_JSON_SEPARATOR

            candidates: List[str] = []
            if marker in clean_response:
                tail = clean_response.split(marker)[-1].strip()
                if tail:
                    candidates.append(tail)

            candidates.append(clean_response)

            parsed_obj = None
            for candidate in candidates:
                # 尝试直接解析
                try:
                    parsed_obj = json.loads(candidate)
                    break
                except Exception:
                    pass

                # 尝试提取最后一个JSON对象，兼容前置推理文本
                json_spans = re.findall(r"\{[\s\S]*?\}", candidate)
                for json_text in reversed(json_spans):
                    try:
                        parsed_obj = json.loads(json_text)
                        break
                    except Exception:
                        continue
                if parsed_obj is not None:
                    break

            if parsed_obj is None:
                return None

            if not isinstance(parsed_obj, dict):
                return None

            if "selected_indices" not in parsed_obj or "need_new" not in parsed_obj or "merge_with" not in parsed_obj:
                return None

            raw_selected = parsed_obj.get("selected_indices")
            if isinstance(raw_selected, list):
                selected_indices = []
                for v in raw_selected:
                    if isinstance(v, bool):
                        continue
                    try:
                        idx = int(v)
                    except (TypeError, ValueError):
                        continue
                    if 0 <= idx < num_categories:
                        selected_indices.append(idx)
                selected_indices = sorted(set(selected_indices))
            else:
                return None

            raw_need_new = parsed_obj.get("need_new")
            if isinstance(raw_need_new, bool):
                need_new = raw_need_new
            elif isinstance(raw_need_new, str):
                val = raw_need_new.strip().lower()
                if val in {"true", "yes", "1"}:
                    need_new = True
                elif val in {"false", "no", "0"}:
                    need_new = False
                else:
                    return None
            else:
                return None

            raw_merge = parsed_obj.get("merge_with")
            merge_with: Optional[int] = None
            if raw_merge is not None:
                if isinstance(raw_merge, str) and raw_merge.strip().lower() in {"none", "null", ""}:
                    raw_merge = None
                if isinstance(raw_merge, bool):
                    return None
                if raw_merge is not None:
                    try:
                        merge_idx = int(raw_merge)
                    except (TypeError, ValueError):
                        return None
                    if 0 <= merge_idx < num_categories:
                        merge_with = merge_idx
                    else:
                        return None

            article_relevant_content = parsed_obj.get("article_relevant_content", parsed_obj.get("ARTICLE_RELEVANT_CONTENT", {}))
            if not isinstance(article_relevant_content, dict):
                article_relevant_content = {}
            new_node_direction = parsed_obj.get("new_node_direction", parsed_obj.get("NEW_NODE_DIRECTION", {}))
            if not isinstance(new_node_direction, dict):
                new_node_direction = {}
            merge_signal = parsed_obj.get("merge_signal", parsed_obj.get("MERGE_SIGNAL", {}))
            if not isinstance(merge_signal, dict):
                merge_signal = article_relevant_content.get("MERGE_SIGNAL", {})
            merge_candidate_probs = {}
            if isinstance(merge_signal, dict):
                merge_candidate_probs = _extract_merge_candidate_probs(
                    merge_signal_obj=merge_signal,
                    merge_with_idx=merge_with,
                    n_cats=num_categories,
                )

            return {
                'selected_indices': selected_indices,
                'need_new': need_new,
                'merge_with': merge_with,
                'article_relevant_content': article_relevant_content,
                'new_node_direction': new_node_direction,
                'merge_signal': merge_signal if isinstance(merge_signal, dict) else {},
                'merge_candidate_probs': merge_candidate_probs,
            }
        except Exception as e:
            print(f"解析分类输出失败: {e}, response: {response}")
            return None
    
    @staticmethod
    def format_classification_completion(
        selected_indices: List[int],
        need_new: bool,
        num_categories: int,
        merge_with: Optional[int] = None,
        relevant_content: Optional[Dict] = None,
        new_node_direction: Optional[Dict] = None,
        merge_signal: Optional[Dict] = None,
    ) -> str:
        """
        格式化分类系统completion（当前四行格式）
        """
        valid_selected = sorted(set(i for i in selected_indices if 0 <= i < num_categories))
        valid_merge = merge_with if (merge_with is not None and 0 <= merge_with < num_categories) else None
        if not isinstance(relevant_content, dict):
            relevant_content = {}
        if not isinstance(new_node_direction, dict):
            new_node_direction = {}
        if not isinstance(merge_signal, dict):
            merge_signal = {}

        return (
            f"ARTICLE_RELEVANT_CONTENT: {json.dumps(relevant_content, ensure_ascii=False)}\n"
            f"NEW_NODE_DIRECTION: {json.dumps(new_node_direction, ensure_ascii=False)}\n"
            f"MERGE_SIGNAL: {json.dumps(merge_signal, ensure_ascii=False)}\n"
            f"MATCHED_CATEGORIES: {json.dumps(valid_selected, ensure_ascii=False)}\n"
            f"NEED_NEW: {'true' if bool(need_new) else 'false'}\n"
            f"MERGE_WITH: {('null' if valid_merge is None else str(valid_merge))}"
        )

    @staticmethod
    def format_classification_reasoning_prompt(
        classification_prompt: str,
        matched_categories: List[int],
        need_new: bool,
        merge_with: Optional[int],
        first_uncovered_path: Optional[List[str]] = None,
    ) -> str:
        """构造“基于Oracle结果补充推理”的数据生成prompt"""
        oracle_path_text = "null"
        if first_uncovered_path:
            oracle_path_text = json.dumps([str(x) for x in first_uncovered_path], ensure_ascii=False)

        return f"""You are reconstructing missing training data for a hierarchical classifier. Your task is to generate missing structured fields that are logically consistent with the given Oracle final decisions.

**Important**:
- Do NOT output chain-of-thought.
- Do NOT output MATCHED_CATEGORIES / NEED_NEW / MERGE_WITH.
- Do NOT output any extra text.
- Output ONLY the following three dictionaries: ARTICLE_RELEVANT_CONTENT, NEW_NODE_DIRECTION, MERGE_SIGNAL.

**Target Dictionary Schemas**:
{{
ARTICLE_RELEVANT_CONTENT:
{{
  "PARENT_RELEVANT_SUMMARY": "string",
  "CHILD_OVERLAP_ANALYSIS": {{
    "Category i": "overlap summary or none"
  }},
  "RESIDUAL_NOVEL_POINTS": ["point1", "point2"]
}},
NEW_NODE_DIRECTION:
{{
  "core_focus": "one-sentence direction for the new node",
  "in_scope_points": ["point1", "point2"],
  "out_of_scope_points": ["point1", "point2"],
  "anchor_terms": ["term1", "term2"]
}},
MERGE_SIGNAL:
{{
  "highly_related_categories": ["Category i"],
  "evidence_for_merge": "short reason",
  "evidence_against_merge": "short reason",
  "merge_strength": 0.0
}}
}}

**Answer-Guided Constraints (MUST satisfy)**:
1. The three dictionaries must support EXACTLY these Oracle outputs:
   - MATCHED_CATEGORIES: {json.dumps(sorted(set(matched_categories)), ensure_ascii=False)}
   - NEED_NEW: {"true" if bool(need_new) else "false"}
   - MERGE_WITH: {"null" if merge_with is None else merge_with}
   - ORACLE_FIRST_UNCOVERED_PATH: {oracle_path_text}
2. CHILD_OVERLAP_ANALYSIS:
   - Categories in MATCHED_CATEGORIES must have clear overlap evidence.
   - Categories NOT in MATCHED_CATEGORIES should be weak overlap or none.
   - If there are no existing child categories in the classification prompt, this field must be {{}}.
3. RESIDUAL_NOVEL_POINTS:
   - If NEED_NEW = true, include concrete residual points not covered by matched categories.
   - If NEED_NEW = false, residual points should be empty or clearly non-decisive.
4. NEW_NODE_DIRECTION requirements:
   - If NEED_NEW = true, provide specific direction for the newly created node.
   - If NEED_NEW = true, core_focus / in_scope_points must be semantically aligned with ORACLE_FIRST_UNCOVERED_PATH.
   - If NEED_NEW = false, NEW_NODE_DIRECTION can be empty or conservative.
5. Keep content concise, semantic, and classification-oriented (no long quoted excerpts).
6. MERGE_SIGNAL requirements:
   - Provide concise merge rationale (for/against) based on residual content.
   - highly_related_categories should include plausible merge candidates (can be empty).
   - merge_strength should be in [0,1] and reflect overall merge confidence.
   - If NEED_NEW = false, merge_strength should usually be low.
   - If MERGE_WITH is not null in Oracle outputs, merge rationale should support that category.

Classification task prompt:
==================================
{classification_prompt}
==================================

Now generate ONLY:
ARTICLE_RELEVANT_CONTENT:
NEW_NODE_DIRECTION:
MERGE_SIGNAL:
"""
    
    # ==================== 总结系统Prompt ====================
    SUMMARY_GENERATE_PROMPT = """You are tasked with generating a summary for a new node in a hierarchical classification system.

**Your Task**:
Follow the reasoning steps below to generate a detailed and specific summary.

** Step 1: Summarize Parent-Relevant Content
- Summarize the parts of the article that are relevant to the Parent Node Scope.
- Use concise semantic summaries (do not rely on clipped quote fragments).

** Step 2: Decompose into Sibling Overlap vs Residual
- Compare Step 1 summary with each Sibling Node Summary.
- Identify:
  1) content overlapping with siblings, and
  2) residual content not covered by siblings.
- If there are no sibling nodes, set SIBLING_OVERLAP_ANALYSIS = {{}} and treat all useful parent-relevant content as residual.

** Step 3: Generate New Node Summary from Residual
- Use residual (non-overlapping) content as the core evidence for the new node.
- Use the provided "New Node Direction" as hard guidance for what to include/exclude.
- The summary consists of:
  - OVERVIEW: concise description of what this node covers. (1-2 sentences)
  - SCOPE: explicit boundary of what belongs to this node, excluding sibling topics. (1-2 sentences)
- The generated summary must align with the direction's core_focus and in_scope_points, and avoid out_of_scope_points.
- The generated summary should not be general but rather focus on the details.
    - *Example Case*: Article -> This article discusses the casting process for the main character Harry in the movie Harry Potter.
        - "This node talks about the movie Harry Potter." (Incorrect)
        - "This node talks about the main character of the movie Harry Potter." (Incorrect)
        - "This node talks about the casting process of the main character Harry in the movie Harry Potter." (Correct)
-  The generated summary should focus on only **ONE** detail. For example:
    - *Example Case*: Article -> This article talks about a person's life story and family relationships.
        - This node talks about the person's life story and family relationships. (Incorrect)
        - This node talks about the person's life story. (Correct)

** Output Format (STRICT):
PARENT RELEVANT_CONTENT: {{
  "PARENT_RELEVANT_SUMMARY": "string",
  "SIBLING_OVERLAP_ANALYSIS": {{
    "Sibling i": "overlap summary or none"
  }},
  "RESIDUAL_NOVEL_POINTS": ["point1", "point2"]
}}
NON_OVERLAPPING: [residual points from Step 2]
OVERVIEW: [generated overview]
SCOPE: [generated scope]

## Input Content for Processing:
Topic: {topic_name}

Parent Node Summary:
{parent_summary}

Sibling Node Summaries:
{siblings_text}

New Node Direction (from classifier):
{new_node_direction_text}

Article:
{new_content}

Now perform the analysis:
"""

    SUMMARY_UPDATE_PROMPT = """You are tasked with determining whether an existing node's summary needs updating, and if so, produce an updated version incorporating new information.

**Your Task**:
Follow the reasoning steps below to assess and potentially update the current node's summary.

** Step 1: Summarize Parent-Relevant Content
- Summarize content from the new input (article or child summary) that is relevant to the Parent Node Scope.
- Use concise semantic summaries (not clipped quote snippets).

** Step 2: Decompose Overlap vs Residual
- Compare Step 1 summary against:
  1) the Current Node Summary, and
  2) all Sibling Node Summaries.
- Identify:
  - content already covered (current/sibling overlap), and
  - residual content not covered by either.

** Step 3: Decide Update
- If esidual content contains information related to the current node's height, but the current node does not cover such content -> NEEDS_UPDATE: Yes
- Otherwise -> NEEDS_UPDATE: No
- *Example Case A*:
    Article -> This article talks about a person's high school story.
    Current Node Summary -> This node talks about the person's primary school story.
    RESIDUAL_NOVEL_POINTS -> ["high school story"]
    - NEEDS_UPDATE: Yes (The stories of high school and those of junior high school belong to the same category of content and are highly related.)
- *Example Case B*:
    Article -> This article talks about a person's family relationships.
    Current Node Summary -> This node talks about the person's primary school story.
    RESIDUAL_NOVEL_POINTS -> ["family relationships"]
    - NEEDS_UPDATE: No (The content of family relationships is not related to the current node's height.)

** Step 4: Generate Updated Summary (Only if NEEDS_UPDATE = Yes)
- Integrate residual content into the current node summary.
- Output:
  - OVERVIEW: refreshed concise coverage. (1-2 sentences)
  - SCOPE: refined node boundary including new material while excluding sibling topics. (1-2 sentences)
- The updated summary should expand the scope based on the original summary rather than adding other theme.
    - *Example Case*: 
        Article -> This article talks about a person's high school story and family relationships.
        Original Summary -> This node talks about the person's primary school story.
        - This node talks about the person's primary school story, high school story and family relationships. (Incorrect, "family" is irrelevant with "school" and should not be included in new summary.)
        - This node talks about the person's school story. (Correct)

** Output Format (STRICT):
PARENT RELEVANT_CONTENT: {{
  "PARENT_RELEVANT_SUMMARY": "string",
  "CURRENT_NODE_OVERLAP": "what is already covered by current summary",
  "SIBLING_OVERLAP_ANALYSIS": {{
    "Sibling i": "overlap summary or none"
  }},
  "RESIDUAL_NOVEL_POINTS": ["point1", "point2"]
}}
NON_OVERLAPPING: [residual points from Step 2]
NEEDS_UPDATE: [Yes/No]
OVERVIEW: [if Yes]
SCOPE: [if Yes]

## Input Content for Processing:
Topic: {topic_name}

Current Node Summary:
{node_summary}

Parent Node Summary:
{parent_summary}

Sibling Node Summaries:
{siblings_text}

New Content (Article or Child Summary):
{new_content}

Now perform the analysis:
"""
    
    @staticmethod
    def format_summary_prompt(
        topic_name: str,
        node_summary: str,
        parent_summary: str,
        sibling_summaries: List[str],
        new_content: str,
        target_label: str = None,
        new_node_direction: Optional[Dict] = None,
    ) -> str:
        """格式化总结系统prompt"""
        # 构建兄弟节点列表
        if sibling_summaries:
            siblings_text = ""
            for i, summary in enumerate(sibling_summaries):
                if isinstance(summary, dict):
                    summary_text = summary.get('full', '')
                else:
                    summary_text = str(summary)
                siblings_text += f"Sibling {i+1}: {summary_text}\n"
        else:
            siblings_text = "No sibling nodes.\n"
        
        # 如果parent_summary为空，使用topic_name
        if not parent_summary:
            parent_summary = topic_name

        if isinstance(new_node_direction, dict) and new_node_direction:
            new_node_direction_text = json.dumps(new_node_direction, ensure_ascii=False)
        else:
            new_node_direction_text = "None"
        
        is_generate = not (node_summary and str(node_summary).strip())
        if is_generate:
            return PromptTemplates.SUMMARY_GENERATE_PROMPT.format(
                topic_name=topic_name,
                parent_summary=parent_summary,
                siblings_text=siblings_text,
                new_node_direction_text=new_node_direction_text,
                new_content=new_content,
            )

        return PromptTemplates.SUMMARY_UPDATE_PROMPT.format(
            topic_name=topic_name,
            node_summary=node_summary,
            parent_summary=parent_summary,
            siblings_text=siblings_text,
            new_content=new_content,
        )
    
    @staticmethod
    def parse_summary_output(response: str) -> Optional[Dict]:
        """
        解析总结系统输出（兼容生成与更新两种prompt）
        
        Returns:
            {'needs_update': True/False, 'explanation': '...', 'scope': '...', 
             'relevant_content': {...}, 'non_overlapping': [...]} 或 None
        """
        try:
            clean_response = response.strip().replace('```json', '').replace('```', '').replace('**', '')
            lines = clean_response.split('\n')
            needs_update: Optional[bool] = None
            overview = ""
            scope = ""
            relevant_content = {}
            non_overlapping = []
            current_field: Optional[str] = None

            def _parse_json_after_label(text: str, label: str):
                pos = text.upper().find(label.upper())
                if pos == -1:
                    return None
                colon = text.find(':', pos)
                if colon == -1:
                    return None
                remainder = text[colon + 1:].lstrip()
                if not remainder:
                    return None
                try:
                    obj, _end = json.JSONDecoder().raw_decode(remainder)
                    return obj
                except Exception:
                    return None

            # 新格式下 relevant_content 可能是多行JSON，优先用整体解析
            rc_obj = _parse_json_after_label(clean_response, 'PARENT RELEVANT_CONTENT')
            if rc_obj is None:
                rc_obj = _parse_json_after_label(clean_response, 'RELEVANT_CONTENT')
            if isinstance(rc_obj, dict):
                relevant_content = rc_obj

            # NON_OVERLAPPING 也可能是完整JSON数组（字符串点列表）
            no_obj = _parse_json_after_label(clean_response, 'NON_OVERLAPPING')
            if isinstance(no_obj, list):
                non_overlapping = no_obj
            
            for line in lines:
                line_stripped = line.strip()
                if not line_stripped:
                    continue

                # relevant_content/non_overlapping 已优先整体解析，这里不再按单行覆盖
                if line_stripped.startswith('PARENT RELEVANT_CONTENT:') or line_stripped.startswith('RELEVANT_CONTENT:'):
                    continue
                elif line_stripped.startswith('NON_OVERLAPPING:'):
                    continue
                elif line_stripped.startswith('NEEDS_UPDATE:'):
                    answer = line_stripped.split(':', 1)[1].strip().upper()
                    needs_update = (answer == 'YES')
                elif line_stripped.startswith('OVERVIEW:') or line_stripped.startswith('EXPLANATION:'):
                    current_field = 'overview'
                    overview = line_stripped.split(':', 1)[1].strip() if ':' in line_stripped else ""
                elif line_stripped.startswith('SCOPE:'):
                    current_field = 'scope'
                    scope = line_stripped.split(':', 1)[1].strip() if ':' in line_stripped else ""
                elif current_field == 'overview' and not line_stripped.startswith('SCOPE:'):
                    overview += ' ' + line_stripped
                elif current_field == 'scope':
                    scope += ' ' + line_stripped

            # 生成prompt通常不含NEEDS_UPDATE，默认视为需要生成
            if needs_update is None:
                needs_update = True

            if not needs_update:
                return {
                    'needs_update': False,
                    'explanation': None,
                    'scope': None,
                    'relevant_content': relevant_content,
                    'non_overlapping': non_overlapping
                }

            if overview and scope:
                return {
                    'needs_update': True,
                    'explanation': overview.strip(),
                    'scope': scope.strip(),
                    'relevant_content': relevant_content,
                    'non_overlapping': non_overlapping
                }

            # 如果没有overview或scope，但needs_update=True，尝试回退
            if needs_update and relevant_content:
                return {
                    'needs_update': True,
                    'explanation': overview.strip() if overview else "",
                    'scope': scope.strip() if scope else "",
                    'relevant_content': relevant_content,
                    'non_overlapping': non_overlapping
                }
            
            return None
        except Exception as e:
            print(f"解析总结输出失败: {e}, response: {response}")
            return None
    
    @staticmethod
    def format_summary_completion(
        needs_update: bool,
        explanation: Optional[str] = None,
        scope: Optional[str] = None
    ) -> str:
        """格式化总结系统completion（用于数据集构建）"""
        if not needs_update:
            return "NEEDS_UPDATE: No"
        
        return f"""NEEDS_UPDATE: Yes
OVERVIEW: {explanation}
SCOPE: {scope}"""
    
    # ==================== 标注系统Prompt ====================
    
    LABELING_PROMPT = """You are an expert quality checker and labeler for a hierarchical classification system.

**Topic**: {topic_name}

**Current Node Summary**:
{current_summary}

**Existing Child Categories**:
{children_text}

**Ground Truth Paths** (where this article actually belongs):
{ground_truth_paths}

=======================================
**Your Task**: Evaluate the child categories and provide the result in the specified format.

**Evaluation Criteria**:

1. **EXCEED_PARENT**: Check each child - does its summary describe content NOT covered by the parent's scope? List indices of children that exceed parent scope, or None if all are within scope.

2. **OVERLAPPING_PAIRS**: Check each pair of children - do they describe similar/redundant content? List [i,j] pairs that overlap, or None if no overlaps.

3. **CORRECT_INDICES**: Check which children match the ground truth paths. A child matches if its summary corresponds to the next level in any ground truth path. List matching indices, or [] if none match.

4. **NEED_NEW**: Is there a ground truth path relevant to the current node but NOT covered by any existing child? Answer Yes or No.

=======================================
**Output Format** (fill with actual values):

EXCEED_PARENT: [indices] or None
OVERLAPPING_PAIRS: [[i,j], ...] or None
CORRECT_INDICES: [indices] or []
NEED_NEW: Yes or No

=======================================
**Example 1**:

Current Node: "Biology - Plants"
Children:
  0: Trees (structure, types, growth)
  1: Animals and mammals
  2: Woody plants and trees
Ground Truth: ["Biology - Plants - Trees", "Biology - Plants - Grasses"]

Output:
EXCEED_PARENT: [1]
OVERLAPPING_PAIRS: [[0, 2]]
CORRECT_INDICES: [0, 2]
NEED_NEW: Yes

Explanation: Child 1 (Animals) exceeds parent scope (not about plants). Children 0 and 2 both describe trees (overlap). Both 0 and 2 match "Trees" in ground truth. "Grasses" is not covered, so need new.

=======================================
**Example 2**:

Current Node: "Technology - Software"
Children:
  0: Operating systems
  1: Applications
Ground Truth: ["Technology - Software - Operating systems"]

Output:
EXCEED_PARENT: None
OVERLAPPING_PAIRS: None
CORRECT_INDICES: [0]
NEED_NEW: No

Explanation: All children within scope. No overlaps. Child 0 matches ground truth. All paths covered.

=======================================
Now evaluate the input above and output the result directly (do not output reasoning or explanations):
"""

    @staticmethod
    def format_labeling_prompt(
        topic_name: str,
        current_summary: str,
        child_summaries: List[str],
        ground_truth_paths: List[str],
        article_content: str = None  # 保留参数以兼容旧代码，但不再使用
    ) -> str:
        """格式化标注系统prompt（不再需要文章内容）"""
        # 构建子节点列表
        if child_summaries:
            children_text = ""
            for i, summary in enumerate(child_summaries):
                if isinstance(summary, dict):
                    summary_text = summary.get('full', '')
                else:
                    summary_text = str(summary)
                children_text += f"Category {i}: {summary_text}\n"
        else:
            children_text = "No existing child categories.\n"
        
        # 构建ground truth paths
        if ground_truth_paths:
            paths_text = "\n".join([f"- {path}" for path in ground_truth_paths])
        else:
            paths_text = "(No ground truth paths provided)"
        
        # 如果current_summary为空，使用topic_name
        if not current_summary:
            current_summary = f"Root level - Topic: {topic_name}"
        
        return PromptTemplates.LABELING_PROMPT.format(
            topic_name=topic_name,
            current_summary=current_summary,
            children_text=children_text,
            ground_truth_paths=paths_text
        )
    
    @staticmethod
    def parse_labeling_output(response: str, num_children: int = None) -> Optional[Dict]:
        """
        解析标注系统输出
        
        Args:
            response: 模型输出
            num_children: 子类别数量（用于验证索引范围）
        
        Returns:
            {
                'exceed_parent': [1, 3] or None,
                'overlapping_pairs': [[0, 1], [2, 3]] or None,
                'correct_indices': [0, 2],
                'need_new': True
            } 或 None
        """
        try:
            lines = response.strip().split('\n')
            
            # 找到最后一组完整的输出（有可能前面有Q/A问答，或者有重复输出）
            # 策略：从后往前找，找到最后一个NEED_NEW，然后往前找其他三个字段
            
            last_exceed_idx = -1
            last_overlap_idx = -1
            last_correct_idx = -1
            last_need_idx = -1
            
            for i in range(len(lines) - 1, -1, -1):
                line = lines[i].strip()
                if last_need_idx == -1 and line.startswith('NEED_NEW:'):
                    last_need_idx = i
                if last_correct_idx == -1 and line.startswith('CORRECT_INDICES:'):
                    last_correct_idx = i
                if last_overlap_idx == -1 and line.startswith('OVERLAPPING_PAIRS:'):
                    last_overlap_idx = i
                if last_exceed_idx == -1 and line.startswith('EXCEED_PARENT:'):
                    last_exceed_idx = i
            
            # 如果没找到完整的四行，返回None
            if last_need_idx == -1:
                return None
            
            # 解析这些行
            exceed_parent = None
            overlapping_pairs = None
            correct_indices = []
            need_new = False
            
            # 解析EXCEED_PARENT
            if last_exceed_idx != -1:
                line = lines[last_exceed_idx].strip()
                value_str = line.split(':', 1)[1].strip()
                # 处理 [None] 的情况
                if value_str.lower() == '[none]':
                    exceed_parent = None
                elif value_str.lower() != 'none' and value_str != '[]':
                    exceed_parent = []
                    value_str = value_str.strip('[]')
                    for idx_str in value_str.split(','):
                        idx_str = idx_str.strip()
                        if idx_str and idx_str.lower() != 'none':
                            try:
                                idx = int(idx_str)
                                if num_children is None or (0 <= idx < num_children):
                                    exceed_parent.append(idx)
                            except ValueError:
                                continue
            
            # 解析OVERLAPPING_PAIRS
            if last_overlap_idx != -1:
                line = lines[last_overlap_idx].strip()
                value_str = line.split(':', 1)[1].strip()
                # 处理 [None] 的情况
                if value_str.lower() == '[none]':
                    overlapping_pairs = None
                elif value_str.lower() != 'none' and value_str != '[]':
                    overlapping_pairs = []
                    import re
                    pairs = re.findall(r'\[(\d+),\s*(\d+)\]', value_str)
                    for idx1_str, idx2_str in pairs:
                        try:
                            idx1 = int(idx1_str)
                            idx2 = int(idx2_str)
                            if num_children is None or (0 <= idx1 < num_children and 0 <= idx2 < num_children):
                                overlapping_pairs.append([idx1, idx2])
                        except ValueError:
                            continue
            
            # 解析CORRECT_INDICES
            if last_correct_idx != -1:
                line = lines[last_correct_idx].strip()
                indices_str = line.split(':', 1)[1].strip()
                if indices_str and indices_str != '[]':
                    indices_str = indices_str.strip('[]')
                    for idx_str in indices_str.split(','):
                        idx_str = idx_str.strip()
                        if idx_str:
                            try:
                                idx = int(idx_str)
                                if num_children is None or (0 <= idx < num_children):
                                    correct_indices.append(idx)
                            except ValueError:
                                continue
            
            # 解析NEED_NEW
            if last_need_idx != -1:
                line = lines[last_need_idx].strip()
                answer = line.split(':', 1)[1].strip().upper()
                need_new = (answer == 'YES')
            
            return {
                'exceed_parent': exceed_parent,
                'overlapping_pairs': overlapping_pairs,
                'correct_indices': correct_indices,
                'need_new': need_new
            }
        except Exception as e:
            print(f"解析标注输出失败: {e}, response: {response[:200]}")
            return None
    
    # ==================== Summary生成Prompt（为Wikipedia数据集生成summaries） ====================
    
    SUMMARY_GENERATION_PROMPT = """You are tasked with creating a hierarchical summary for a Wikipedia article section.

**Hierarchical Path**: {path}

**Content**:
{content}

**Task**: Generate a summary for the CURRENT NODE in this hierarchical structure.

**CRITICAL HIERARCHICAL RULES**:
1. **Focus on Current Level ONLY**: Your summary should ONLY describe what THIS specific node adds to its parent, not repeat the parent's content.
2. **Stay Within Parent's Scope**: All content must fall within the parent node's scope. Do NOT include unrelated topics.
3. **No Upward Repetition**: Do NOT repeat information from ancestor nodes (parent, grandparent, etc.).

**Your Summary Must Include TWO Parts**:

1. **EXPLANATION**: What does THIS specific subcategory discuss? 
   - Focus ONLY on what this node adds to its parent
   - Do NOT repeat the parent's description
   - Be specific to this level

2. **SCOPE**: What aspects or sub-topics does THIS subcategory include?
   - List ONLY the sub-topics within this node
   - Must be within the parent's scope
   - Do NOT include sibling or unrelated topics

**Requirements**:
- Concise (around 50-100 words per part)
- Hierarchically precise - each level adds specificity
- Clear boundaries - what IS and what IS NOT included
- No content duplication across levels

**Format your response as**:
EXPLANATION: [Your explanation here]
SCOPE: [Your scope definition here]
"""

    @staticmethod
    def format_summary_generation_prompt(path: str, content: str) -> str:
        """格式化summary生成prompt（用于生成Wikipedia节点的summaries）"""
        return PromptTemplates.SUMMARY_GENERATION_PROMPT.format(
            path=path,
            content=content
        )
