"""
总结系统
判断是否需要更新节点summary，并生成更新后的summary
"""
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, Union
from summary_based_classifier.llm.prompts import PromptTemplates
import json
import math
import re
from collections import Counter
import os


def _pick_safe_vllm_dtype(requested_dtype):
    """Use float16 on legacy GPUs (e.g., V100) to avoid bf16 init errors."""
    if requested_dtype is not None:
        return requested_dtype
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        for idx in range(torch.cuda.device_count()):
            major, _minor = torch.cuda.get_device_capability(idx)
            if major < 8:
                return "half"
    except Exception:
        return None
    return None


@dataclass
class SummaryInput:
    """总结输入"""
    node_summary: str  # 当前节点的summary（可以为空，表示生成新节点）
    parent_summary: str  # 父节点的summary
    sibling_summaries: List[str]  # 兄弟节点的summary列表
    new_content: str  # 新内容（文章或子节点summary）
    topic_name: str  # topic名称


@dataclass
class SummaryOutput:
    """总结输出"""
    needs_update: bool  # 是否需要更新
    explanation: Optional[str]  # 更新后的explanation（如果needs_update=True）
    scope: Optional[str]  # 更新后的scope（如果needs_update=True）
    raw_response: str  # 原始响应


class Updater:
    """总结更新系统"""
    
    def __init__(self, mode: str = 'model', model_path: Optional[str] = None, **kwargs):
        """
        Args:
            mode: 模式 ('model' 或 'api' 或 'bow' 或 'hybrid')
            model_path: 模型路径（mode='model'时需要）
            **kwargs: 其他参数
        """
        self.mode = mode
        self.model_path = model_path
        self.bow_top_k = int(kwargs.get('bow_top_k', 30))
        self.bow_min_token_len = int(kwargs.get('bow_min_token_len', 2))
        
        # Hybrid模式参数
        self.hybrid_keywords_top_k = int(kwargs.get('hybrid_keywords_top_k', 10))
        self.hybrid_evidence_max_tokens = int(kwargs.get('hybrid_evidence_max_tokens', 200))
        self.hybrid_llm_model = kwargs.get('hybrid_llm_model', 'deepseek-chat')
        self.hybrid_api_key = kwargs.get('hybrid_api_key', '')
        self.hybrid_api_url = kwargs.get('hybrid_api_url', 'https://api.deepseek.com')
        
        if mode == 'model':
            from vllm import LLM, SamplingParams
            
            # 支持单卡ID（int）或多卡列表字符串（如"0,1"）
            gpu_id = kwargs.get('gpu_id', 0)
            tp_size = int(kwargs.get('tensor_parallel_size', 1))
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            llm_kwargs = dict(
                model=model_path,
                tensor_parallel_size=tp_size,
                max_model_len=kwargs.get('max_model_len', 16384),
                gpu_memory_utilization=kwargs.get('gpu_memory_utilization', 0.9),
                trust_remote_code=True,
                disable_log_stats=True,  # 禁用日志统计，减少进程间通信
                enforce_eager=True,
            )
            safe_dtype = _pick_safe_vllm_dtype(kwargs.get('dtype'))
            if safe_dtype is not None:
                llm_kwargs['dtype'] = safe_dtype

            self.llm = LLM(**llm_kwargs)
            
            self.sampling_params = SamplingParams(
                temperature=kwargs.get('temperature', 0.3),
                top_p=kwargs.get('top_p', 0.9),
                max_tokens=kwargs.get('max_tokens', 512),
                stop=[]
            )
        elif mode == 'api':
            # API模式配置 - 保存参数，延迟创建配置对象
            self.api_kwargs = {
                'api_key': kwargs.get('api_key', ''),
                'api_url': kwargs.get('api_url', 'https://api.deepseek.com'),
                'model_name': kwargs.get('model_name', 'deepseek-chat'),
                'temperature': kwargs.get('temperature', 0.3),
                'max_tokens': kwargs.get('max_tokens', 512),
                'max_workers': kwargs.get('max_workers', 5)
            }
        elif mode == 'bow':
            # 词频模式：不需要模型
            # BM25参数
            self.bm25_k1 = kwargs.get('bm25_k1', 1.5)
            self.bm25_b = kwargs.get('bm25_b', 0.75)
        elif mode == 'hybrid':
            # 混合模式：BM25关键词 + LLM提取的证据片段
            # BM25参数
            self.bm25_k1 = kwargs.get('bm25_k1', 1.5)
            self.bm25_b = kwargs.get('bm25_b', 0.75)
            
            # Tokenizer用于计数
            from transformers import AutoTokenizer
            tokenizer_name = kwargs.get('tokenizer_name', 'Qwen/Qwen2.5-7B-Instruct')
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
            
            # 初始化vLLM用于证据片段提取
            from vllm import LLM, SamplingParams
            
            # 获取GPU设备ID
            gpu_id = kwargs.get('gpu_id', 0)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            evidence_llm_kwargs = dict(
                model=model_path,  # 使用传入的model_path
                tensor_parallel_size=1,
                max_model_len=kwargs.get('max_model_len', 8192),
                gpu_memory_utilization=kwargs.get('gpu_memory_utilization', 0.5),
                trust_remote_code=True,
                disable_log_stats=True,
                enforce_eager=True,
            )
            safe_dtype = _pick_safe_vllm_dtype(kwargs.get('dtype'))
            if safe_dtype is not None:
                evidence_llm_kwargs['dtype'] = safe_dtype

            self.evidence_llm = LLM(**evidence_llm_kwargs)
            
            self.evidence_sampling_params = SamplingParams(
                temperature=0.3,
                top_p=0.9,
                max_tokens=250,
                stop=[]
            )
            
            # vLLM调用信号量（多线程保护）
            self.vllm_semaphore = kwargs.get('vllm_semaphore', None)
    
    def create_prompt(self, input_data: SummaryInput) -> str:
        """创建总结prompt"""
        if self.mode == 'bow':
            # 词频模式不依赖prompt，但为了接口一致仍返回可读的“伪prompt”
            return (
                f"[BOW_UPDATER]\n"
                f"Topic: {input_data.topic_name}\n"
                f"NodeSummary: {input_data.node_summary[:200]}\n"
                f"ParentSummary: {input_data.parent_summary[:200]}\n"
                f"Siblings: {len(input_data.sibling_summaries)}\n"
                f"NewContent: {input_data.new_content[:2000]}\n"
            )
        prompt = PromptTemplates.format_summary_prompt(
            topic_name=input_data.topic_name,
            node_summary=input_data.node_summary,
            parent_summary=input_data.parent_summary,
            sibling_summaries=input_data.sibling_summaries,
            new_content=input_data.new_content
        )
        if os.environ.get("SBC_ENABLE_THINKING", "1") != "0":
            prompt = (
                "[Thinking Mode: ON]\n"
                "First think carefully step by step, then output strictly in the required format.\n\n"
                f"{prompt}"
            )
        return prompt

    def complete_classification_prompts(
        self,
        prompts: List[str],
        n: int = 1,
        temperature: float = 0.0,
        max_tokens: int = 2048
    ) -> List[List[SummaryOutput]]:
        """
        专用于分类completion补全的推理接口。
        不做summary结构解析，仅回传raw_response。
        """
        if self.mode != 'model':
            raise ValueError("complete_classification_prompts 仅支持 model 模式")

        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=0.9,
            max_tokens=max_tokens,
            n=max(1, int(n)),
        )
        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)

        all_results: List[List[SummaryOutput]] = []
        for output in outputs:
            one_prompt_results: List[SummaryOutput] = []
            for out in output.outputs:
                one_prompt_results.append(SummaryOutput(
                    needs_update=False,
                    explanation="",
                    scope="",
                    raw_response=out.text
                ))
            all_results.append(one_prompt_results)

        return all_results
    
    def parse_output(self, response: str) -> Optional[SummaryOutput]:
        """
        解析总结输出
        
        Args:
            response: 模型返回的文本
            
        Returns:
            SummaryOutput对象，解析失败返回None
        """
        parsed = PromptTemplates.parse_summary_output(response)
        if parsed is None:
            return None
        
        return SummaryOutput(
            needs_update=parsed['needs_update'],
            explanation=parsed.get('explanation'),
            scope=parsed.get('scope'),
            raw_response=response
        )

    # ==================== BOW 词频模式实现 ====================

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        # 简单、稳定的 tokenizer：只保留字母数字，下采样噪声
        text = text.lower()
        return re.findall(r"[a-z0-9]+", text)

    def _bow_from_text(self, text: str) -> Counter:
        stop = {
            # 英文常见停用词（最小集合，稳定）
            "the","a","an","and","or","but","if","then","else","when","while","for","to","of","in","on","at","by","with","as",
            "is","are","was","were","be","been","being","this","that","these","those","it","its","they","them","their","we","you","your",
            "from","into","over","under","about","after","before","between","during","through","against","within","without",
        }
        # 极少量中文功能词（不做分词，只做极端过滤：对“英文tokenizer”影响很小）
        # 注：中文在本tokenizer里基本不会出现；保留这里是为了满足“去掉语气词/助词”的硬要求。
        stop_zh = {"的","了","啊","呀","呢","吧","吗","嘛","着","和","与","及","并","以及","或","而","但","是","在","对","于"}
        toks = []
        for t in self._tokenize(text):
            if len(t) < self.bow_min_token_len:
                continue
            if t in stop:
                continue
            if t in stop_zh:
                continue
            toks.append(t)
        return Counter(toks)

    def _bow_from_content(self, content: str) -> Counter:
        """
        new_content 可能是：
        - 原始文章文本：直接 tokenize 统计
        - 子节点 summary（包含 BOW_JSON）：解析并当作词频输入
        """
        parsed = self._extract_bow_json(content)
        if parsed:
            return Counter(parsed)
        return self._bow_from_text(content)

    def _compute_bm25_scores(
        self, 
        term_freqs: Counter, 
        doc_length: int,
        df_stats: Dict[str, int],  # 每个词的文档频率
        total_docs: int,           # 该topic总文档数
        avg_doc_length: float      # 该topic平均文档长度
    ) -> Dict[str, float]:
        """
        计算BM25得分
        BM25(w) = IDF(w) * (tf * (k1+1)) / (tf + k1 * (1 - b + b * dl/avgdl))
        IDF(w) = log((N - df + 0.5) / (df + 0.5) + 1)
        """
        scores = {}
        for term, tf in term_freqs.items():
            df = df_stats.get(term, 0)
            if df == 0:
                continue  # 该词不在统计中，跳过
            
            # IDF计算
            idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0)
            
            # BM25得分
            norm_factor = 1 - self.bm25_b + self.bm25_b * (doc_length / avg_doc_length) if avg_doc_length > 0 else 1.0
            score = idf * (tf * (self.bm25_k1 + 1)) / (tf + self.bm25_k1 * norm_factor)
            scores[term] = score
        
        return scores

    @staticmethod
    def _extract_bow_json(summary: str) -> Optional[Dict[str, int]]:
        """
        BOW模式的summary直接就是JSON词频字典
        """
        if not summary:
            return None
        try:
            obj = json.loads(summary.strip())
            if isinstance(obj, dict):
                return {str(k): int(v) for k, v in obj.items()}
        except Exception:
            return None
        return None

    def _format_bow_summary(self, bow: Union[Counter, Dict[str, float]]) -> str:
        """格式化BOW/BM25为summary（JSON格式，支持float得分）"""
        if isinstance(bow, Counter):
            top = bow.most_common(self.bow_top_k)
            return json.dumps(dict(top), ensure_ascii=False) if top else "{}"
        else:
            # BM25得分：按score降序排序，取top_k
            # 四舍五入到一位小数，减少prompt冗余
            sorted_items = sorted(bow.items(), key=lambda x: x[1], reverse=True)[:self.bow_top_k]
            rounded_items = [(word, round(score, 1)) for word, score in sorted_items]
            return json.dumps(dict(rounded_items), ensure_ascii=False) if rounded_items else "{}"
    
    def _extract_evidence_snippet(self, content: str, existing_evidence: str = "", parent_summary: str = "", sibling_summaries: List[str] = None) -> str:
        """
        使用LLM zero-shot提取区分性证据片段（≤200 tokens）
        
        Args:
            content: 新文章内容
            existing_evidence: 已有的证据片段
            parent_summary: 父节点summary
            sibling_summaries: 兄弟节点summaries
            
        Returns:
            提取的证据片段（严格≤200 tokens）
        """
        # 构建提示词：要求提取"区分性证据"
        prompt = f"""Extract distinguishing evidence from the article that best represents its unique characteristics.

Requirements:
1. Extract 2-4 key sentences or phrases that distinguish this content from siblings
2. Focus on specific facts, names, concepts that are distinctive
3. DO NOT summarize or generalize
4. STRICT LIMIT: Output must be ≤200 tokens

Parent context: {parent_summary[:200]}
Sibling categories: {len(sibling_summaries) if sibling_summaries else 0} others

Article content:
{content[:2000]}

Output the evidence snippets directly (no explanation):"""

        # 使用vLLM生成证据片段（信号量保护）
        if self.vllm_semaphore:
            with self.vllm_semaphore:
                outputs = self.evidence_llm.generate([prompt], self.evidence_sampling_params, use_tqdm=False)
        else:
            outputs = self.evidence_llm.generate([prompt], self.evidence_sampling_params, use_tqdm=False)
        evidence = outputs[0].outputs[0].text if outputs else ""
        
        # 严格限制到200 tokens
        tokens = self.tokenizer.encode(evidence)
        if len(tokens) > self.hybrid_evidence_max_tokens:
            tokens = tokens[:self.hybrid_evidence_max_tokens]
            evidence = self.tokenizer.decode(tokens, skip_special_tokens=True)
        
        # 如果有existing_evidence，合并并再次限制
        if existing_evidence:
            combined = existing_evidence + " " + evidence
            tokens = self.tokenizer.encode(combined)
            if len(tokens) > self.hybrid_evidence_max_tokens:
                tokens = tokens[:self.hybrid_evidence_max_tokens]
                evidence = self.tokenizer.decode(tokens, skip_special_tokens=True)
            else:
                evidence = combined
        
        return evidence
    
    def merge_hybrid_summaries(self, child_summaries: List[str], parent_summary: str = "") -> str:
        """
        归拢时合并多个子节点的hybrid summaries
        
        Args:
            child_summaries: 子节点的hybrid summaries
            parent_summary: 父节点summary（提供上下文）
            
        Returns:
            合并后的hybrid summary
        """
        # 1. 合并keywords（取并集并累加得分）
        from collections import defaultdict
        merged_keywords = defaultdict(float)
        all_evidences = []
        
        for summary in child_summaries:
            try:
                parsed = json.loads(summary.strip())
                if isinstance(parsed, dict):
                    keywords = parsed.get('keywords', {})
                    for word, score in keywords.items():
                        merged_keywords[word] += float(score)
                    evidence = parsed.get('evidence', '')
                    if evidence:
                        all_evidences.append(evidence)
            except:
                continue
        
        # 取top-10关键词
        sorted_items = sorted(merged_keywords.items(), key=lambda x: x[1], reverse=True)[:self.hybrid_keywords_top_k]
        keywords = {word: round(score, 1) for word, score in sorted_items}
        
        # 2. 合并evidences（使用LLM挑选最佳片段）
        if all_evidences:
            combined_evidences = "\n---\n".join(all_evidences)
            prompt = f"""Select and combine the most distinguishing evidence from the following child categories.

Requirements:
1. Pick the most representative and distinctive snippets
2. STRICT LIMIT: Output must be ≤200 tokens
3. DO NOT add new content, only select from existing evidence

Parent context: {parent_summary[:200]}

Child evidences:
{combined_evidences[:1500]}

Output the selected evidence directly (no explanation):"""
            
            # 使用vLLM生成（信号量保护）
            if self.vllm_semaphore:
                with self.vllm_semaphore:
                    outputs = self.evidence_llm.generate([prompt], self.evidence_sampling_params, use_tqdm=False)
            else:
                outputs = self.evidence_llm.generate([prompt], self.evidence_sampling_params, use_tqdm=False)
            evidence = outputs[0].outputs[0].text if outputs else all_evidences[0]
            
            # 严格限制到200 tokens
            tokens = self.tokenizer.encode(evidence)
            if len(tokens) > self.hybrid_evidence_max_tokens:
                tokens = tokens[:self.hybrid_evidence_max_tokens]
                evidence = self.tokenizer.decode(tokens, skip_special_tokens=True)
        else:
            evidence = ""
        
        # 3. 组合为summary
        hybrid_summary = {
            'keywords': keywords,
            'evidence': evidence
        }
        return json.dumps(hybrid_summary, ensure_ascii=False)

    def _bow_update(self, input_data: SummaryInput) -> SummaryOutput:
        # 读取已有 bow（若没有则视为空）
        existing = self._extract_bow_json(input_data.node_summary) or {}
        bow = Counter(existing)
        # new_content 可以是文章或子节点 summary：都当文本累加
        bow += self._bow_from_content(input_data.new_content)
        summary_str = self._format_bow_summary(bow)
        raw = f"NEEDS_UPDATE: Yes\nBOW_SUMMARY: {summary_str}\n"
        # 为了兼容接口，仍返回needs_update/explanation/scope，但BOW模式下这些字段不再有明确语义
        return SummaryOutput(needs_update=True, explanation=summary_str, scope="", raw_response=raw)
    
    def _bm25_update(
        self, 
        input_data: SummaryInput,
        df_stats: Dict[str, int],
        total_docs: int,
        avg_doc_length: float
    ) -> SummaryOutput:
        """BM25模式更新summary"""
        # 解析已有得分（若有）
        existing = self._extract_bow_json(input_data.node_summary) or {}
        
        # 从new_content提取词频
        new_bow = self._bow_from_content(input_data.new_content)
        if not new_bow:
            return SummaryOutput(needs_update=False, explanation=input_data.node_summary, scope="", raw_response="")
        
        # 计算新内容的BM25得分
        doc_length = sum(new_bow.values())
        new_scores = self._compute_bm25_scores(new_bow, doc_length, df_stats, total_docs, avg_doc_length)
        
        # 合并：累加得分
        from collections import defaultdict
        merged = defaultdict(float)
        for w, s in existing.items():
            merged[w] += float(s)
        for w, s in new_scores.items():
            merged[w] += s
        
        summary_str = self._format_bow_summary(dict(merged))
        raw = f"NEEDS_UPDATE: Yes\nBM25_SUMMARY: {summary_str}\n"
        return SummaryOutput(needs_update=True, explanation=summary_str, scope="", raw_response=raw)
    
    def _hybrid_update(
        self,
        input_data: SummaryInput,
        df_stats: Dict[str, int],
        total_docs: int,
        avg_doc_length: float
    ) -> SummaryOutput:
        """
        Hybrid模式更新：BM25关键词 + LLM证据片段
        
        Summary格式: {"keywords": {...}, "evidence": "..."}
        """
        # 1. 解析已有内容
        existing_keywords = {}
        existing_evidence = ""
        if input_data.node_summary:
            try:
                parsed = json.loads(input_data.node_summary.strip())
                if isinstance(parsed, dict):
                    existing_keywords = parsed.get('keywords', {})
                    existing_evidence = parsed.get('evidence', '')
            except:
                pass
        
        # 2. 更新BM25关键词（只取top-10）
        new_bow = self._bow_from_content(input_data.new_content)
        if new_bow:
            doc_length = sum(new_bow.values())
            new_scores = self._compute_bm25_scores(new_bow, doc_length, df_stats, total_docs, avg_doc_length)
            
            # 合并得分
            from collections import defaultdict
            merged = defaultdict(float)
            for w, s in existing_keywords.items():
                merged[w] += float(s)
            for w, s in new_scores.items():
                merged[w] += s
            
            # 只取top-10
            sorted_items = sorted(merged.items(), key=lambda x: x[1], reverse=True)[:self.hybrid_keywords_top_k]
            keywords = {word: round(score, 1) for word, score in sorted_items}
        else:
            keywords = existing_keywords
        
        # 3. 更新证据片段（LLM zero-shot）
        evidence = self._extract_evidence_snippet(
            content=input_data.new_content,
            existing_evidence=existing_evidence,
            parent_summary=input_data.parent_summary,
            sibling_summaries=input_data.sibling_summaries
        )
        
        # 4. 组合为summary
        hybrid_summary = {
            'keywords': keywords,
            'evidence': evidence
        }
        summary_str = json.dumps(hybrid_summary, ensure_ascii=False)
        
        raw = f"NEEDS_UPDATE: Yes\nHYBRID_SUMMARY: {summary_str}\n"
        return SummaryOutput(needs_update=True, explanation=summary_str, scope="", raw_response=raw)
    
    def update(self, input_data: SummaryInput) -> SummaryOutput:
        """
        执行总结更新
        
        Args:
            input_data: 总结输入
            
        Returns:
            SummaryOutput对象
        """
        if self.mode == 'bow':
            return self._bow_update(input_data)
        if self.mode == 'model':
            # 创建prompt
            prompt = self.create_prompt(input_data)
            
            # 调用模型
            outputs = self.llm.generate([prompt], self.sampling_params, use_tqdm=False)
            response = outputs[0].outputs[0].text
            
            # 解析输出
            result = self.parse_output(response)
            
            if result is None:
                # 解析失败，返回默认值（不更新）
                print(f"警告: 总结输出解析失败，使用默认值")
                return SummaryOutput(
                    needs_update=False,
                    explanation=None,
                    scope=None,
                    raw_response=response
                )
            
            return result
        elif self.mode == 'api':
            # API模式
            import sys
            from pathlib import Path
            sys.path.append(str(Path(__file__).parent.parent))
            from modeling.deepseek_api import DeepSeekAPIClient, DeepSeekConfig
            
            # 创建API配置
            api_config = DeepSeekConfig(
                api_key=self.api_kwargs['api_key'],
                base_url=self.api_kwargs['api_url'].replace('/chat/completions', ''),
                model=self.api_kwargs['model_name'],
                temperature=self.api_kwargs['temperature'],
                max_output_tokens=self.api_kwargs['max_tokens'],
                max_concurrent_jobs=self.api_kwargs['max_workers']
            )
            
            # 创建API客户端
            client = DeepSeekAPIClient(api_config)
            
            # 创建prompt
            prompt = self.create_prompt(input_data)
            
            # 调用API
            responses = client.run_prompts_to_texts([prompt], show_progress=False)
            response = responses[0] if responses else ""
            
            # 解析输出
            result = self.parse_output(response)
            
            if result is None:
                # 解析失败，返回默认值（不更新）
                print(f"警告: 总结输出解析失败，使用默认值")
                return SummaryOutput(
                    needs_update=False,
                    explanation=None,
                    scope=None,
                    raw_response=response
                )
            
            return result
        else:
            raise ValueError(f"不支持的模式: {self.mode}")
    
    def update_with_multiple_samples(
        self,
        inputs: List[SummaryInput],
        n: int = 1
    ) -> List[List[SummaryOutput]]:
        """
        批量更新，每个输入采样n个结果
        
        Args:
            inputs: 输入列表
            n: 每个输入采样的结果数
            
        Returns:
            List[List[SummaryOutput]]，外层list对应inputs，内层list是n个采样结果
        """
        if self.mode == 'model':
            # 创建prompts
            prompts = [self.create_prompt(inp) for inp in inputs]
            
            # 设置采样参数（采样n个结果）
            from vllm import SamplingParams
            sampling_params = SamplingParams(
                temperature=self.sampling_params.temperature,
                top_p=self.sampling_params.top_p,
                max_tokens=self.sampling_params.max_tokens,
                stop=self.sampling_params.stop,
                n=n  # 关键：采样n个结果
            )
            
            # 批量调用模型
            outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)
            
            # 解析输出
            results = []
            for output in outputs:
                sample_results = []
                
                # 处理n个采样结果
                for sample_output in output.outputs:
                    response = sample_output.text
                    result = self.parse_output(response)
                    
                    if result is None:
                        # 解析失败，使用默认值
                        result = SummaryOutput(
                            needs_update=False,
                            updated_summary=None,
                            raw_response=response
                        )
                    
                    sample_results.append(result)
                
                results.append(sample_results)
            
            return results
        else:
            raise ValueError(f"不支持的模式: {self.mode}")
    
    def update_batch(self, inputs: List[SummaryInput]) -> List[SummaryOutput]:
        """
        批量总结更新
        
        Args:
            inputs: 总结输入列表
            
        Returns:
            总结输出列表
        """
        if self.mode == 'model':
            # 创建prompts
            prompts = [self.create_prompt(inp) for inp in inputs]
            
            # 批量调用模型
            outputs = self.llm.generate(prompts, self.sampling_params, use_tqdm=False)
            
            # 解析输出
            results = []
            for output in outputs:
                response = output.outputs[0].text
                result = self.parse_output(response)
                
                if result is None:
                    # 解析失败，使用默认值
                    result = SummaryOutput(
                        needs_update=False,
                        explanation=None,
                        scope=None,
                        raw_response=response
                    )
                
                results.append(result)
            
            return results
        elif self.mode == 'api':
            # API模式
            import sys
            from pathlib import Path
            sys.path.append(str(Path(__file__).parent.parent))
            from modeling.deepseek_api import DeepSeekAPIClient, DeepSeekConfig
            
            # 创建API配置
            api_config = DeepSeekConfig(
                api_key=self.api_kwargs['api_key'],
                base_url=self.api_kwargs['api_url'].replace('/chat/completions', ''),
                model=self.api_kwargs['model_name'],
                temperature=self.api_kwargs['temperature'],
                max_output_tokens=self.api_kwargs['max_tokens'],
                max_concurrent_jobs=self.api_kwargs['max_workers']
            )
            
            # 创建API客户端
            client = DeepSeekAPIClient(api_config)
            
            # 创建prompts
            prompts = [self.create_prompt(inp) for inp in inputs]
            
            # 批量调用API
            responses = client.run_prompts_to_texts(prompts, show_progress=False)
            
            # 解析输出
            results = []
            for response in responses:
                result = self.parse_output(response)
                
                if result is None:
                    # 解析失败，使用默认值
                    result = SummaryOutput(
                        needs_update=False,
                        explanation=None,
                        scope=None,
                        raw_response=response
                    )
                
                results.append(result)
            
            return results
        else:
            raise ValueError(f"不支持的模式: {self.mode}")
    
    def update_summary(
        self,
        input_data: SummaryInput,
        n_samples: int = 1,
        temperature: float = 0.5,
        bm25_stats: Optional[Dict] = None
    ) -> List[SummaryOutput]:
        """
        执行总结更新（兼容性方法，与update_with_sampling相同）
        
        Args:
            input_data: 总结输入
            n_samples: 采样数量
            temperature: 采样温度
            bm25_stats: BM25统计信息 {'df': Dict[str, int], 'total_docs': int, 'avg_doc_length': float}
            
        Returns:
            SummaryOutput列表
        """
        return self.update_with_sampling(input_data, n_samples, temperature, bm25_stats)
    
    def update_with_sampling(
        self,
        input_data: SummaryInput,
        n: int = 1,
        temperature: float = 0.5,
        bm25_stats: Optional[Dict] = None
    ) -> List[SummaryOutput]:
        """
        执行总结更新并采样多个结果（用于轨迹采样）
        
        Args:
            input_data: 总结输入
            n: 采样数量
            temperature: 采样温度
            bm25_stats: BM25统计信息 {'df': Dict[str, int], 'total_docs': int, 'avg_doc_length': float}
            
        Returns:
            SummaryOutput列表
        """
        if self.mode == 'bow':
            # 如果提供了bm25_stats，使用BM25；否则回退到原始BOW
            if bm25_stats:
                out = self._bm25_update(
                    input_data,
                    bm25_stats['df'],
                    bm25_stats['total_docs'],
                    bm25_stats['avg_doc_length']
                )
            else:
                out = self._bow_update(input_data)
            return [out for _ in range(max(1, int(n)))]
        if self.mode == 'hybrid':
            # Hybrid模式需要bm25_stats
            if not bm25_stats:
                raise ValueError("Hybrid mode requires bm25_stats")
            out = self._hybrid_update(
                input_data,
                bm25_stats['df'],
                bm25_stats['total_docs'],
                bm25_stats['avg_doc_length']
            )
            return [out for _ in range(max(1, int(n)))]
        if self.mode == 'model':
            from vllm import SamplingParams
            
            # 创建prompt
            prompt = self.create_prompt(input_data)
            
            # 修改采样参数以支持采样
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=0.9,
                max_tokens=512,
                stop=["\n\n\n", "###"],
                n=n  # 采样n个结果
            )
            
            # 调用模型
            outputs = self.llm.generate([prompt], sampling_params, use_tqdm=False)
            
            # 解析所有采样结果
            results = []
            for output_choice in outputs[0].outputs:
                response = output_choice.text
                result = self.parse_output(response)
                
                if result is not None:
                    results.append(result)
            
            # 如果没有成功解析任何结果，返回默认值
            if not results:
                results = [SummaryOutput(
                    needs_update=False,
                    explanation=None,
                    scope=None,
                    raw_response=""
                )]
            
            return results
        elif self.mode == 'api':
            # API模式 - 采样多次
            import sys
            from pathlib import Path
            sys.path.append(str(Path(__file__).parent.parent))
            from modeling.deepseek_api import DeepSeekAPIClient, DeepSeekConfig
            
            # 创建API配置
            api_config = DeepSeekConfig(
                api_key=self.api_kwargs['api_key'],
                base_url=self.api_kwargs['api_url'].replace('/chat/completions', ''),
                model=self.api_kwargs['model_name'],
                temperature=self.api_kwargs['temperature'],
                max_output_tokens=self.api_kwargs['max_tokens'],
                max_concurrent_jobs=self.api_kwargs['max_workers']
            )
            
            # 创建API客户端
            client = DeepSeekAPIClient(api_config)
            
            # 创建prompt
            prompt = self.create_prompt(input_data)
            
            # 调用API多次以获得多个采样结果
            prompts = [prompt] * n
            responses = client.run_prompts_to_texts(prompts, show_progress=False)
            
            # 解析所有结果
            results = []
            for response in responses:
                result = self.parse_output(response)
                if result is not None:
                    results.append(result)
            
            # 如果没有成功解析任何结果，返回默认值
            if not results:
                results = [SummaryOutput(
                    needs_update=False,
                    explanation=None,
                    scope=None,
                    raw_response=""
                )]
            
            return results
        elif self.mode == 'api':
            # API 模式不支持 n 采样，这里退化为单次 update 并复制
            out = self.update(input_data)
            return [out for _ in range(max(1, int(n)))]
        else:
            raise ValueError(f"不支持的模式: {self.mode}")
    
    def generate_new_summary(
        self,
        parent_summary: str,
        sibling_summaries: List[str],
        new_content: str,
        topic_name: str
    ) -> Optional[Dict[str, str]]:
        """
        生成新节点的summary（当分类系统判断需要NEW时调用）
        
        Args:
            parent_summary: 父节点summary
            sibling_summaries: 兄弟节点summaries
            new_content: 新内容（文章）
            topic_name: topic名称
            
        Returns:
            {'explanation': '...', 'scope': '...'} 或 None
        """
        input_data = SummaryInput(
            node_summary="",  # 空summary表示生成新节点
            parent_summary=parent_summary,
            sibling_summaries=sibling_summaries,
            new_content=new_content,
            topic_name=topic_name
        )
        
        result = self.update(input_data)
        
        if result.needs_update and result.explanation and result.scope:
            return {
                'explanation': result.explanation,
                'scope': result.scope
            }
        
        return None
