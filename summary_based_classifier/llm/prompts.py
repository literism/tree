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

** Step 1: Extract Article Relevant Content  
- Identify key themes from the article that indicate classification intent.  
- Output as a dictionary with numbered keys.  
- For each excerpt, show ONLY the first few words and last few words, separated by "...".  
- Merge related ideas; avoid fragmentation.  
- Each entry must represent a distinct thematic unit (e.g., topic, focus, scope).  
- Example: {{"0": "Chronicles: Art & Design ... film adaptation", "1": "concept art by Alan Lee ... Weta Workshop"}}

** Step 2: Match Against Existing Categories Using Step 1  
- Review each item in Step 1's output.  
- Determine whether it overlaps significantly with any existing child category.  
- Justify briefly for each match (or non-match).  
- Final output: List of indices where >0 excerpt from Step 1 clearly belongs.  
- If none: output []

** Step 3: Assess Need for New Category Based on Unmatched Content  
- Consider only the excerpts in Step 1 that were NOT covered by existing categories.  
- Ask: Does this residual content form a coherent, specific theme not currently represented?  
- If yes -> NEED_NEW = true  
- If no (general/redundant) -> NEED_NEW = false

** Step 4: Assess Need for Merge  
- Evaluate whether any two existing categories are semantically redundant given the full context.  
- Output: single index to merge into, or null

**Output Format** (STRICT):
ARTICLE RELEVANT_CONTENT: {{dictionary from Step 1}}
MATCHED_CATEGORIES: [list from Step 2]
NEED_NEW: [true/false from Step 3]
MERGE_WITH: [int/null from Step 4]

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

                return {
                    "selected_indices": selected_indices,
                    "need_new": need_new,
                    "merge_with": merge_with,
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
                    return {
                        'selected_indices': selected_indices,
                        'need_new': need_new,
                        'merge_with': None
                    }
                if isinstance(raw_merge, bool):
                    return None
                try:
                    merge_idx = int(raw_merge)
                except (TypeError, ValueError):
                    return None
                if 0 <= merge_idx < num_categories:
                    merge_with = merge_idx
                else:
                    return None

            return {
                'selected_indices': selected_indices,
                'need_new': need_new,
                'merge_with': merge_with
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
        relevant_content: Optional[Dict] = None
    ) -> str:
        """
        格式化分类系统completion（当前四行格式）
        """
        valid_selected = sorted(set(i for i in selected_indices if 0 <= i < num_categories))
        valid_merge = merge_with if (merge_with is not None and 0 <= merge_with < num_categories) else None
        if not isinstance(relevant_content, dict):
            relevant_content = {}

        return (
            f"ARTICLE RELEVANT_CONTENT: {json.dumps(relevant_content, ensure_ascii=False)}\n"
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
    ) -> str:
        """构造“基于Oracle结果补充推理”的数据生成prompt"""
        return f"""You are reconstructing missing training data for a hierarchical classifier. Your task is to generate the missing Step 1 output ("ARTICLE RELEVANT_CONTENT") that logically leads to the given final decisions.

**Important**: 
- Do NOT output reasoning.
- Do NOT modify or reformat the final JSON.
- The only thing you may generate is the "ARTICLE RELEVANT_CONTENT" dictionary.

**Rules for Reconstruction**:
1. Analyze the article and the final decisions to infer what key themes must have been extracted in Step 1.
2. Each theme should be a coherent idea relevant to classification (e.g., topic, focus, scope).
3. Format each as: "X": "first few words ... last few words"
   - Show only beginning and end of phrase, with "..." in between.
   - Total visible words (before/after "...") < 10.
   - Merge related ideas; avoid fragmentation.
4. Ensure the set explains:
   - Why certain categories were matched (or not)
   - Why a new category was needed (if YES)
   - Why no merge occurred (or which one was chosen)
5. The result must be sufficient and necessary to support the downstream outputs.

Classification task prompt:
{classification_prompt}

Oracle final outputs (MUST be preserved in logic):
MATCHED_CATEGORIES: {json.dumps(sorted(set(matched_categories)), ensure_ascii=False)}
NEED_NEW: {"true" if bool(need_new) else "false"}
MERGE_WITH: {"null" if merge_with is None else merge_with}

Now generate ONLY:
ARTICLE RELEVANT_CONTENT:
"""
    
    # ==================== 总结系统Prompt ====================
    SUMMARY_GENERATE_PROMPT = """You are tasked with generating a summary for a new node in a hierarchical classification system.

**Your Task**:
Follow the reasoning steps below to generate a detailed and specific summary.

** Step 1: Extract Parent Relevant Content
- Extract content from the article that is relevant to the Parent Node Scope. Output as a dictionary with numbered keys.
- For each excerpt, show ONLY the first few words and last few words, with "..." in between.
- Merge related content into longer excerpts (avoid fragmentation).
- Each excerpt should represent a coherent idea and be less than 10 words total (beginning + end), e.g.: {{"0": "Bilbo finds the ring ... changes his journey"}}
- Example format: {{"0": "Darwin's health declined ... requiring attention", "1": "He married Emma ... lived together until"}}

** Step 2: Identify Relevant Excerpts for This Node
- From Step 1 results, identify which items do NOT overlap with any Sibling Node Summary.
- Output: List of keys representing non-overlapping, relevant content. Example: ["0", "2"]

** Step 3: Generate Summary
- Use the selected excerpts to create a precise and focused summary.
- The summary consists of two parts:
    - OVERVIEW: Provide a concise description of what this node covers, based on the extracted content. (1-2 sentences)
    - SCOPE: Define clearly what belongs to this node — focus on specificity, not generality. Exclude topics handled by siblings. (1-2 sentences)
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
PARENT RELEVANT_CONTENT: {{dictionary from Step 1}}
NON_OVERLAPPING: [list from Step 2]
OVERVIEW: [generated overview]
SCOPE: [generated scope]

## Input Content for Processing:
Topic: {topic_name}

Parent Node Summary:
{parent_summary}

Sibling Node Summaries:
{siblings_text}

Article:
{new_content}

Now perform the analysis:
"""

    SUMMARY_UPDATE_PROMPT = """You are tasked with determining whether an existing node's summary needs updating, and if so, produce an updated version incorporating new information.

**Your Task**:
Follow the reasoning steps below to assess and potentially update the current node's summary.

** Step 1: Extract Parent Relevant Content
- Extract content from the new input (article or child summary) that is relevant to the Parent Node Scope. Output as a dictionary with numbered keys.
- For each excerpt, show ONLY the first few words and last few words, with "..." in between.
- Merge related content into coherent segments (not too fragmented).
- Each excerpt should be less than 10 words total (beginning + end), representing a longer section.
- Example: {{"0": "Bilbo meets Gollum ... wins the riddle game", "1": "The ring grants invisibility ... affects bearers over time"}}

** Step 2: Filter Non-Overlapping Content
- From Step 1 results, identify which items do NOT overlap with:
   - Any Sibling Node Summary  
   - The Current Node Summary
- Only when an item belongs to neither sibling nor current summary is it considered non-overlapping.
- Output: List of keys from Step 1 that are truly new and unique to this node. Example: ["0", "2"]

** Step 3: Decide Update
- If there is at least one non-overlapping item → Output "Yes"
- Otherwise → Output "No"

** Step 4: Generate Updated Summary (Only if Step 3 = "Yes")
- Integrate the non-overlapping content into the existing summary.
- The updated summary includes:
    - OVERVIEW: Refreshed to reflect all key aspects now covered by the node, including new insights. (1-2 sentences)
    - SCOPE: Refined to ensure boundaries remain clear and inclusive of new material, while still excluding sibling topics. (1-2 sentences)
- Maintain precision — avoid vagueness or repetition.
    - *Example Case*: If new info discusses “Gollum’s backstory influencing Bilbo,” then update accordingly.
    - "This node talks about characters in The Hobbit." (Incorrect)
    - "This node covers Bilbo’s encounter with Gollum and how it leads to acquiring the ring." (Correct)

** Output Format (STRICT):
PARENT RELEVANT_CONTENT: {{dictionary from Step 1}}
NON_OVERLAPPING: [list from Step 2]
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
        target_label: str = None
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
        
        is_generate = not (node_summary and str(node_summary).strip())
        if is_generate:
            return PromptTemplates.SUMMARY_GENERATE_PROMPT.format(
                topic_name=topic_name,
                parent_summary=parent_summary,
                siblings_text=siblings_text,
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
            lines = response.strip().split('\n')
            needs_update: Optional[bool] = None
            overview = ""
            scope = ""
            relevant_content = {}
            non_overlapping = []
            current_field: Optional[str] = None
            
            for line in lines:
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                
                line_stripped = line_stripped.replace('```json', '').replace('```', '').replace('**', '')
                
                if line_stripped.startswith('PARENT RELEVANT_CONTENT:') or line_stripped.startswith('RELEVANT_CONTENT:'):
                    # 解析JSON字典
                    content_str = line_stripped.split(':', 1)[1].strip()
                    try:
                        relevant_content = json.loads(content_str)
                    except:
                        relevant_content = {}
                elif line_stripped.startswith('NON_OVERLAPPING:'):
                    # 解析JSON列表
                    overlap_str = line_stripped.split(':', 1)[1].strip()
                    try:
                        non_overlapping = json.loads(overlap_str)
                    except:
                        non_overlapping = []
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
