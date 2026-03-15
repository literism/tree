#!/usr/bin/env python3
"""
结构树可视化程序
- 读取test_trees.json
- 使用Qwen3-8B翻译节点summary
- 可视化树结构
- 显示叶子节点文章数
"""
import json
import os
from pathlib import Path
from vllm import LLM, SamplingParams
from typing import Dict
# Graphviz是可选的
try:
    from graphviz import Digraph
    HAS_GRAPHVIZ = True
except ImportError:
    HAS_GRAPHVIZ = False
    print("警告：未安装graphviz，图形格式将不可用")

class TreeVisualizer:
    def __init__(self, model_path="/home/literism/model/Qwen3-8B"):
        """初始化可视化器和翻译模型"""
        print("使用vllm加载Qwen3-8B模型...")
        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            gpu_memory_utilization=0.85,
            max_model_len=4096,
            tensor_parallel_size=1
        )
        print("模型加载完成！")
        
        # 采样参数
        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.8,
            max_tokens=512,
            repetition_penalty=1.0
        )
        
        # 翻译缓存
        self.translation_cache = {}
    
    def clean_summary_text(self, text):
        """清理summary文本，去掉EXPLANATION、SCOPE等格式标记"""
        import re
        
        # 去掉EXPLANATION:、SCOPE:等前缀
        text = re.sub(r'EXPLANATION:\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'SCOPE:\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'KEYWORDS:\s*.*', '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # 分割成句子，只保留有意义的内容
        sentences = []
        for line in text.split('\n'):
            line = line.strip()
            if line and not line.startswith('KEYWORDS'):
                sentences.append(line)
        
        # 合并句子
        cleaned_text = ' '.join(sentences)
        
        # 限制长度
        if len(cleaned_text) > 5000:
            cleaned_text = cleaned_text[:5000] + "..."
        
        return cleaned_text

    def merge_single_child_nodes(self, tree_dict: Dict) -> Dict:
        """
        递归合并只有一个子节点的节点
        
        规则：如果一个节点只有一个子节点，将子节点合并到父节点
        - summary使用父节点的summary
        - 文章集合合并
        - 递归处理，例如 A->B->C 都是单子节点，最后合并成一个节点
        
        Args:
            tree_dict: 树的字典表示
            
        Returns:
            合并后的树字典
        """
        def merge_node(node: Dict) -> Dict:
            """
            递归合并节点
            """
            # 首先递归处理所有子节点
            if node.get('children'):
                merged_children = [merge_node(child) for child in node['children']]
                
                # 合并单子节点
                while len(merged_children) == 1:
                    # 只有一个子节点，合并它
                    only_child = merged_children[0]
                    
                    # 合并文章集合
                    merged_citations = list(set(node.get('citations', []) + only_child.get('citations', [])))
                    
                    # 使用当前节点的summary和其他属性，但合并文章和子节点
                    node = {
                        'title': node.get('title', ''),
                        'level': node.get('level', 0),
                        'summary': node.get('summary', {}),  # 保留父节点的summary
                        'citations': merged_citations,
                        'children': only_child.get('children', [])  # 继承孙子节点
                    }
                    
                    # 更新merged_children为新的子节点列表，继续检查
                    merged_children = node.get('children', [])
                
                # 更新节点的children
                node['children'] = merged_children
            
            return node
        
        # 处理真实数据格式（有structure字段）
        if 'structure' in tree_dict:
            merged_structure = [merge_node(node) for node in tree_dict['structure']]
            result = tree_dict.copy()
            result['structure'] = merged_structure
            return result
        else:
            # 推理结果格式（整个dict就是一个树节点）
            return merge_node(tree_dict)
    
    def collect_texts_to_translate(self, tree_dict):
        """收集所有需要翻译的文本"""
        texts_to_translate = []
        node_text_map = {}  # 记录每个节点对应的文本索引
        
        def traverse(node, node_id):
            summary = node.get('summary', '')
            title = node.get('title', '')
            
            # 优先翻译summary，否则翻译title
            if summary:
                # 清理summary
                text_to_translate = self.clean_summary_text(summary)
            else:
                text_to_translate = title
            
            if text_to_translate and text_to_translate not in self.translation_cache:
                if text_to_translate not in texts_to_translate:
                    texts_to_translate.append(text_to_translate)
                node_text_map[node_id] = text_to_translate
            elif text_to_translate in self.translation_cache:
                node_text_map[node_id] = text_to_translate
            
            # 递归处理子节点
            for i, child in enumerate(node.get('children', [])):
                traverse(child, f"{node_id}_c{i}")
        
        # 检查是否有structure字段
        if 'structure' in tree_dict:
            for i, root_node in enumerate(tree_dict['structure']):
                traverse(root_node, f"root_{i}")
        else:
            traverse(tree_dict, "root_0")
        
        return texts_to_translate, node_text_map
    
    def batch_translate(self, texts):
        """批量翻译文本"""
        if not texts:
            return {}
        
        print(f"批量翻译 {len(texts)} 个文本...")
        
        # 准备prompts - 使用更严格的提示词
        prompts = []
        for text in texts:
            prompt = f"""将下面的英文翻译成中文。只输出中文翻译，不要输出任何其他内容。

{text}

中文翻译："""
            prompts.append(prompt)
        
        # 批量推理
        outputs = self.llm.generate(prompts, self.sampling_params)
        
        # 处理结果
        translations = {}
        for text, output in zip(texts, outputs):
            generated_text = output.outputs[0].text
            chinese_text = self.clean_translation(generated_text)
            translations[text] = chinese_text
            self.translation_cache[text] = chinese_text
        
        print(f"翻译完成！")
        return translations
    
    def clean_translation(self, text):
        """清理翻译结果，去掉think标签和其他无关内容"""
        import re
        
        # 去掉<think>...</think>标签及其内容
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)
        text = re.sub(r'</think>.*', '', text, flags=re.DOTALL)
        
        # 去掉常见的无关前缀和标记
        prefixes_to_remove = [
            "中文翻译：",
            "中文：",
            "翻译：",
            "好的",
            "首先",
            "Okay",
            "Let's",
            "The translation",
            "Translation:",
        ]
        
        for prefix in prefixes_to_remove:
            if text.strip().startswith(prefix):
                text = text.strip()[len(prefix):].strip()
        
        # 如果翻译结果仍然包含大量英文，说明翻译失败，返回空
        english_chars = sum(1 for c in text if ord(c) < 128 and c.isalpha())
        total_chars = len(text.replace(' ', ''))
        if total_chars > 0 and english_chars / total_chars > 0.5:
            return ""
        
        # 清理多余的空白
        text = '\n'.join(line.strip() for line in text.split('\n') if line.strip())
        
        # 限制长度
        if len(text) > 300:
            text = text[:300] + "..."
        
        return text.strip()
    
    def count_citations(self, node):
        """统计节点的citations数量"""
        return len(node.get('citations', []))
    
    def is_leaf(self, node):
        """判断是否为叶子节点"""
        return len(node.get('children', [])) == 0
    
    def build_text_tree(self, node, prefix="", is_last=True, level=0):
        """构建文本形式的树结构"""
        lines = []
        
        # 节点信息
        title = node.get('title', 'NO_TITLE')
        summary = node.get('summary', '')
        num_citations = self.count_citations(node)
        node_level = node.get('level', -1)
        is_leaf_node = self.is_leaf(node)
        
        # 获取翻译（从缓存中）
        if summary:
            cleaned_summary = self.clean_summary_text(summary)
            chinese_text = self.translation_cache.get(cleaned_summary, "")
        else:
            chinese_text = self.translation_cache.get(title, "")
        
        # 构建节点显示文本 - 只显示中文翻译
        connector = "└── " if is_last else "├── "
        
        # 主节点行：显示中文翻译（如果有）或英文标题
        if chinese_text:
            display_text = chinese_text[:1000] + "..." if len(chinese_text) > 1000 else chinese_text
        else:
            display_text = title[:800] + "..." if len(title) > 800 else title
        
        node_text = f"{prefix}{connector}[L{node_level}] {display_text}"
        
        if is_leaf_node:
            node_text += f" 📄{num_citations}篇"
        else:
            node_text += f" ({len(node.get('children', []))}个子节点)"
        
        lines.append(node_text)
        
        # 递归处理子节点
        children = node.get('children', [])
        for i, child in enumerate(children):
            is_last_child = (i == len(children) - 1)
            new_prefix = prefix + ("    " if is_last else "│   ")
            child_lines = self.build_text_tree(child, new_prefix, is_last_child, level + 1)
            lines.extend(child_lines)
        
        return lines
    
    def build_graphviz_tree(self, tree_dict, topic_name):
        """使用graphviz构建可视化树"""
        if not HAS_GRAPHVIZ:
            print("错误：未安装graphviz，无法生成图形")
            return None
        
        dot = Digraph(comment=topic_name)
        dot.attr(rankdir='TB')
        dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')
        
        node_counter = [0]
        
        def add_node(node, parent_id=None):
            node_id = f"node_{node_counter[0]}"
            node_counter[0] += 1
            
            # 节点信息
            title = node.get('title', 'NO_TITLE')
            summary = node.get('summary', '')
            num_citations = self.count_citations(node)
            is_leaf_node = self.is_leaf(node)
            
            # 获取翻译（从缓存中）
            text_to_translate = summary if summary else title
            chinese_text = self.translation_cache.get(text_to_translate, "")
            
            # 构建标签
            # 缩短title
            display_title = title if len(title) <= 50 else title[:47] + "..."
            label = f"{display_title}\n"
            if chinese_text:
                # 限制长度
                short_text = chinese_text[:80] + "..." if len(chinese_text) > 80 else chinese_text
                label += f"💭 {short_text}\n"
            
            if is_leaf_node:
                label += f"📄 {num_citations}篇"
                dot.node(node_id, label, fillcolor='lightgreen')
            else:
                label += f"{len(node.get('children', []))}个子节点"
                dot.node(node_id, label, fillcolor='lightblue')
            
            # 添加边
            if parent_id is not None:
                dot.edge(parent_id, node_id)
            
            # 递归处理子节点
            for child in node.get('children', []):
                add_node(child, node_id)
        
        # 检查是否有structure字段
        if 'structure' in tree_dict:
            for root_node in tree_dict['structure']:
                add_node(root_node)
        else:
            add_node(tree_dict)
        
        return dot
    
    def visualize_topic(self, tree_dict, topic_name, output_dir, format='both'):
        """可视化单个topic的树
        
        Args:
            tree_dict: 树的字典表示
            topic_name: topic名称
            output_dir: 输出目录
            format: 'text', 'graph', 或 'both'
        """
        print(f"\n{'='*80}")
        print(f"正在可视化: {topic_name}")
        print(f"{'='*80}")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 第一步：收集所有需要翻译的文本
        print(f"\n收集需要翻译的文本...")
        texts_to_translate, node_text_map = self.collect_texts_to_translate(tree_dict)
        
        # 第二步：批量翻译
        if texts_to_translate:
            self.batch_translate(texts_to_translate)
        else:
            print("所有文本都已在缓存中")
        
        # 第三步：生成可视化
        # 文本格式
        if format in ['text', 'both']:
            print(f"\n生成文本树...")
            
            # 检查是否有structure字段
            if 'structure' in tree_dict:
                all_lines = [f"{topic_name}"]
                for root_node in tree_dict['structure']:
                    lines = self.build_text_tree(root_node)
                    all_lines.extend(lines)
            else:
                lines = self.build_text_tree(tree_dict)
                all_lines = [f"{topic_name}"] + lines
            
            # 保存到文件
            text_file = output_dir / f"{topic_name.replace(':', '_')}.txt"
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(all_lines))
            
            print(f"文本树已保存到: {text_file}")
            
            # 也打印到控制台（限制行数）
            # print("\n预览（前50行）:")
            # for line in all_lines[:50]:
            #     print(line)
            # if len(all_lines) > 50:
            #     print(f"... (还有 {len(all_lines) - 50} 行)")
        
        # 图形格式
        if format in ['graph', 'both']:
            if not HAS_GRAPHVIZ:
                print("跳过图形生成（未安装graphviz）")
            else:
                print(f"\n生成Graphviz图...")
                dot = self.build_graphviz_tree(tree_dict, topic_name)
                
                if dot is not None:
                    # 保存为PDF和PNG
                    graph_file = output_dir / f"{topic_name.replace(':', '_')}"
                    dot.render(graph_file, format='pdf', cleanup=True)
                    dot.render(graph_file, format='png', cleanup=True)
                    
                    print(f"图形已保存到: {graph_file}.pdf 和 {graph_file}.png")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='可视化结构树')
    parser.add_argument(
        '--input',
        type=str,
        default='/mnt/literism/tree/summary_output/inference/test_trees.json',
        help='输入JSON文件路径'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='/home/literism/tree/visualizations',
        help='输出目录'
    )
    parser.add_argument(
        '--topics',
        type=str,
        nargs='*',
        help='要可视化的topics（不指定则全部）'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['text', 'graph', 'both'],
        default='text',
        help='输出格式：text（文本）, graph（图形）, both（两者）'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='/home/literism/model/Qwen3-8B',
        help='Qwen模型路径'
    )
    
    args = parser.parse_args()
    
    # 加载数据
    print(f"加载数据: {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"找到 {len(data)} 个topics")
    
    # 确定要处理的topics
    if args.topics:
        topics_to_process = {k: v for k, v in data.items() if k in args.topics}
        print(f"将处理 {len(topics_to_process)} 个指定的topics")
    else:
        topics_to_process = data
        print(f"将处理所有 {len(topics_to_process)} 个topics")
    
    # 创建可视化器
    visualizer = TreeVisualizer(model_path=args.model_path)
    
    # 处理每个topic
    for topic_name, tree_dict in topics_to_process.items():
        try:
            tree_dict = visualizer.merge_single_child_nodes(tree_dict)
            visualizer.visualize_topic(tree_dict, topic_name, args.output, args.format)
        except Exception as e:
            print(f"处理 {topic_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*80}")
    print(f"完成！所有可视化文件已保存到: {args.output}")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()

