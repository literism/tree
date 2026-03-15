"""
准备训练数据集
基于路径和扰动策略生成三类数据
"""
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm


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


class DatasetPreparator:
    def __init__(
        self,
        references_file: str,
        structures_file: str,
        paraphrases_dir: str,
        dataset_split_file: str,
        output_dir: str,
        delete_prob: float = 0.1,
        replace_prob: float = 0.1,
        class_ratio: Tuple[int, int, int] = (2, 1, 1),
        train_size: int = 10000,
        val_ratio: float = 0.1,
        seed: int = 42,
        num_constraint_leaves: int = 10,
        type1_single_new_prob: float = 0.7,
        mix_output_to_constraint_prob: float = 0.1
    ):
        """
        Args:
            references_file: 引用文件路径
            structures_file: 结构文件路径
            paraphrases_dir: paraphrase目录
            dataset_split_file: 数据集划分文件
            output_dir: 输出目录
            delete_prob: 删除非输出title的概率
            replace_prob: 替换为paraphrase的概率
            class_ratio: 三类数据的比例 (有新标题, 只有existing, 都为空)
            train_size: 训练集总大小
            val_ratio: 验证集比例
            seed: 随机种子
            num_constraint_leaves: 选择多少个叶子节点作为约束
            type1_single_new_prob: 类型1数据中只放一个new_subtitle的概率
            mix_output_to_constraint_prob: 将一个输出混入约束的概率
        """
        self.references_file = references_file
        self.structures_file = structures_file
        self.paraphrases_dir = Path(paraphrases_dir)
        self.dataset_split_file = dataset_split_file
        self.output_dir = Path(output_dir)
        self.delete_prob = delete_prob
        self.replace_prob = replace_prob
        self.class_ratio = class_ratio
        self.train_size = train_size
        self.val_ratio = val_ratio
        self.seed = seed
        self.num_constraint_leaves = num_constraint_leaves
        self.type1_single_new_prob = type1_single_new_prob
        self.mix_output_to_constraint_prob = mix_output_to_constraint_prob
        
        random.seed(seed)
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据容器
        self.references = {}
        self.structures = {}
        self.paraphrases = {}
        self.dataset_split = {}
        
    def load_data(self):
        """加载所有数据"""
        print("=" * 80)
        print("加载数据...")
        print("=" * 80)
        
        # 加载引用
        print("\n加载引用数据...")
        with open(self.references_file, 'r', encoding='utf-8') as f:
            self.references = json.load(f)
        print(f"  - 加载 {len(self.references)} 个topics的引用")
        
        # 加载结构
        print("\n加载结构数据...")
        with open(self.structures_file, 'r', encoding='utf-8') as f:
            self.structures = json.load(f)
        print(f"  - 加载 {len(self.structures)} 个topics的结构")
        
        # 加载paraphrases（树结构）
        print("\n加载Paraphrase数据...")
        for topic_key in self.references.keys():
            para_file = self.paraphrases_dir / f"{topic_key.replace(':', '_')}_paraphrases.json"
            if para_file.exists():
                with open(para_file, 'r', encoding='utf-8') as f:
                    self.paraphrases[topic_key] = json.load(f)
            else:
                print(f"  警告: 未找到 {topic_key} 的paraphrase文件")
                self.paraphrases[topic_key] = {}
        print(f"  - 加载 {len(self.paraphrases)} 个topics的paraphrases")
        
        # 加载数据集划分
        print("\n加载数据集划分...")
        with open(self.dataset_split_file, 'r', encoding='utf-8') as f:
            self.dataset_split = json.load(f)
        print(f"  - 训练topics: {len(self.dataset_split['dataset_split']['train'])}")
        print(f"  - 简单测试topics: {len(self.dataset_split['dataset_split']['test_easy'])}")
        print(f"  - 困难测试topics: {len(self.dataset_split['dataset_split']['test_hard'])}")
        
    def build_structure_dict(self, topic_key: str) -> Dict[str, List[str]]:
        """
        构建结构字典，记录每个路径下的所有直接子节点
        
        Args:
            topic_key: topic键
            
        Returns:
            字典 {path: [child_titles]}
        """
        structure_dict = defaultdict(list)
        
        if topic_key not in self.structures:
            return structure_dict
        
        topic_data = self.structures[topic_key]
        topic_name = topic_data.get('topic', '')
        
        def traverse(node, current_path):
            """递归遍历结构树"""
            if 'title' in node and node['title']:
                title = node['title']
                # 记录当前路径下的这个子节点
                structure_dict[current_path].append(title)
                
                # 更新路径
                new_path = f"{current_path} - {title}" if current_path else title
                
                # 递归处理子节点
                if 'children' in node and node['children']:
                    for child in node['children']:
                        traverse(child, new_path)
        
        # 处理结构
        if 'structure' in topic_data:
            if isinstance(topic_data['structure'], list):
                # structure是列表，每个元素是根节点的直接子节点
                for child in topic_data['structure']:
                    traverse(child, topic_name)
            else:
                # structure是单个节点
                traverse(topic_data['structure'], "")
        
        return structure_dict
    
    def collect_all_leaf_paths(self, topic_key: str) -> List[str]:
        """
        收集该topic的所有叶子节点路径
        
        Args:
            topic_key: topic键
            
        Returns:
            所有叶子节点的完整路径列表
        """
        leaf_paths = []
        
        if topic_key not in self.structures:
            return leaf_paths
        
        topic_data = self.structures[topic_key]
        topic_name = topic_data.get('topic', '')
        
        def traverse(node, current_path):
            """递归遍历结构树收集叶子节点"""
            if 'title' in node and node['title']:
                title = node['title']
                new_path = f"{current_path} - {title}" if current_path else title
                
                # 检查是否是叶子节点
                if not node.get('children'):
                    leaf_paths.append(new_path)
                else:
                    # 递归处理子节点
                    for child in node['children']:
                        traverse(child, new_path)
        
        # 处理结构
        if 'structure' in topic_data:
            if isinstance(topic_data['structure'], list):
                for child in topic_data['structure']:
                    traverse(child, topic_name)
            else:
                traverse(topic_data['structure'], "")
        
        return leaf_paths
    
    def collect_sibling_leaf_paths(self, topic_key: str, current_path: str) -> List[str]:
        """
        收集当前路径的兄弟节点及其所有后代叶子节点的路径
        
        例如：current_path = "The Hobbit - Narrative"
        返回：Concept and creation 和 Influences 分支下的所有叶子节点
        
        Args:
            topic_key: topic键
            current_path: 当前路径，如 "The Hobbit - Narrative"
            
        Returns:
            兄弟分支的叶子节点路径列表
        """
        if topic_key not in self.structures:
            return []
        
        topic_data = self.structures[topic_key]
        topic_name = topic_data.get('topic', '')
        
        # 解析当前路径
        path_parts = [p.strip() for p in current_path.split(' - ')]
        
        # 如果是根节点，没有兄弟节点
        if len(path_parts) == 1:
            return []
        
        sibling_leaves = []
        
        def collect_all_leaves(node, node_path):
            """从一个节点收集所有叶子节点（递归）"""
            if not node.get('children'):
                # 是叶子节点
                sibling_leaves.append(node_path)
            else:
                # 非叶子节点，递归收集子节点的叶子
                for child in node['children']:
                    child_path = f"{node_path} - {child['title']}"
                    collect_all_leaves(child, child_path)
        
        def find_and_collect_siblings(nodes, path_index, accumulated_path):
            """
            递归查找目标节点，并收集其兄弟节点的叶子
            
            Args:
                nodes: 当前层的节点列表
                path_index: 当前在path_parts中的索引（从1开始，因为0是topic名）
                accumulated_path: 累积的路径（到当前层的父节点）
            
            Returns:
                True if found, False otherwise
            """
            if path_index >= len(path_parts):
                return False
            
            target_title = path_parts[path_index]
            
            # 遍历当前层的所有节点
            for node in nodes:
                if node['title'] == target_title:
                    # 找到了路径上的节点
                    if path_index == len(path_parts) - 1:
                        # 这是目标节点，现在收集其兄弟节点的叶子
                        for sibling in nodes:
                            if sibling['title'] != target_title:
                                # 这是兄弟节点
                                sibling_path = f"{accumulated_path} - {sibling['title']}"
                                collect_all_leaves(sibling, sibling_path)
                        return True
                    else:
                        # 还需要继续向下查找
                        children = node.get('children', [])
                        if children:
                            node_path = f"{accumulated_path} - {node['title']}"
                            return find_and_collect_siblings(children, path_index + 1, node_path)
                        else:
                            return False
            
            return False
        
        # 处理结构
        if 'structure' in topic_data:
            structure = topic_data['structure']
            if not isinstance(structure, list):
                structure = [structure]
            
            # 从第一层开始查找（path_parts[1]对应structure的第一层）
            find_and_collect_siblings(structure, 1, topic_name)
        
        return sibling_leaves
    
    def select_constraint_leaves(
        self,
        topic_key: str,
        current_path: str,
        existing_subtitles: List[str],
        output_titles: List[str] = None
    ) -> Tuple[List[str], Optional[str]]:
        """
        选择约束叶子节点
        
        约束来自当前节点的兄弟节点及其所有后代叶子节点
        另外，可能从output_titles中选择一个混入约束
        
        Args:
            topic_key: topic键
            current_path: 当前路径，如 "T - A"
            existing_subtitles: 当前节点已有的子标题 [a1, a2]
            output_titles: 输出标题列表（用于混入约束）
            
        Returns:
            (约束路径列表, 被混入约束的输出标题)
        """
        # 收集兄弟节点的所有叶子
        sibling_leaves = self.collect_sibling_leaf_paths(topic_key, current_path)
        
        # 按相似度排序（如果有existing_subtitles）
        if existing_subtitles and sibling_leaves:
            # 计算每个叶子节点与existing_subtitles的最高相似度得分
            leaf_scores = []
            for leaf_path in sibling_leaves:
                leaf_title = leaf_path.split(' - ')[-1]
                max_score = max(
                    jaccard_similarity(leaf_title, subtitle)
                    for subtitle in existing_subtitles
                )
                leaf_scores.append((leaf_path, max_score))
            
            # 按得分排序，选择前N个
            leaf_scores.sort(key=lambda x: x[1], reverse=True)
            constraint_paths = [path for path, score in leaf_scores[:self.num_constraint_leaves]]
        else:
            # 没有existing_subtitles，直接取前N个
            constraint_paths = sibling_leaves[:self.num_constraint_leaves] if sibling_leaves else []
        
        # 混入输出到约束（以一定概率）
        mixed_output = None
        if output_titles and random.random() < self.mix_output_to_constraint_prob:
            # 从输出中随机选择一个
            selected_output = random.choice(output_titles)
            mixed_output = selected_output
            
            # 构建跳过当前节点的路径
            # 例如：当前路径 "Albert Einstein - Early life"，输出 "Education"
            # 结果："Albert Einstein - Education"
            path_parts = current_path.split(' - ')
            topic_name = path_parts[0]
            
            # 跳过当前节点（最后一层），直接连接到topic
            mixed_path = f"{topic_name} - {selected_output}"
            
            # 将混入的路径添加到约束中
            constraint_paths.append(mixed_path)
            
            # 打乱顺序，避免混入的总是在最后
            random.shuffle(constraint_paths)
        
        return constraint_paths, mixed_output
    
    def get_paraphrase_by_path(self, topic_key: str, path: str) -> Dict[str, str]:
        """
        根据路径从树结构中获取paraphrase
        
        Args:
            topic_key: topic键
            path: 完整路径，如 "Albert Einstein - Early life - Education"
            
        Returns:
            paraphrases字典，如果未找到返回空字典
        """
        if topic_key not in self.paraphrases:
            return {}
        
        topic_data = self.paraphrases[topic_key]
        
        # 解析路径
        path_parts = [p.strip() for p in path.split(' - ')]
        
        # 第一部分应该是topic名称
        if len(path_parts) < 2:
            return {}
        
        # 从第二部分开始在structure中查找
        structure = topic_data.get('structure', [])
        if not structure:
            return {}
        
        # 如果structure是单个节点而非列表
        if not isinstance(structure, list):
            structure = [structure]
        
        # 递归查找
        def find_node(nodes, target_parts, depth):
            """在节点列表中查找目标路径"""
            if depth >= len(target_parts):
                return {}
            
            target_title = target_parts[depth]
            
            for node in nodes:
                if node.get('title') == target_title:
                    # 找到了目标节点
                    if depth == len(target_parts) - 1:
                        # 这是最后一层，返回paraphrase
                        return node.get('paraphrases', {})
                    else:
                        # 还需要继续往下找
                        children = node.get('children', [])
                        return find_node(children, target_parts, depth + 1)
            
            return {}
        
        # 从第二部分开始查找（跳过topic名称）
        return find_node(structure, path_parts[1:], 0)
    
    def parse_paths_to_classifications(self, paths: List[str]) -> List[Tuple[str, List[str]]]:
        """
        将路径列表解析为分类步骤列表
        
        Args:
            paths: 路径列表，如 ["T - A - a1", "T - A - a2", "T - B - b1"]
            
        Returns:
            分类步骤列表 [(current_path, output_titles), ...]
            例如: [("T", ["A", "B"]), ("T - A", ["a1", "a2"]), ("T - B", ["b1"])]
        """
        classifications = defaultdict(set)
        
        for path in paths:
            parts = [p.strip() for p in path.split(' - ')]
            
            # 对每一层都记录分类
            for i in range(1, len(parts)):
                # 当前路径
                current_path = ' - '.join(parts[:i])
                # 输出标题
                output_title = parts[i]
                classifications[current_path].add(output_title)
        
        # 转换为列表
        result = [(path, sorted(list(titles))) for path, titles in classifications.items()]
        # 按照路径深度排序
        result.sort(key=lambda x: x[0].count(' - '))
        
        return result
    
    def apply_perturbations(
        self,
        titles: List[str],
        topic_key: str,
        current_path: str
    ) -> Tuple[List[str], Dict[str, str]]:
        """
        对标题列表应用扰动（替换为paraphrase）
        
        Args:
            titles: 标题列表
            topic_key: topic键
            current_path: 当前路径（用于构建完整路径来查找paraphrase）
            
        Returns:
            (扰动后的标题列表, 映射字典 {原标题: 扰动后标题})
        """
        perturbed_titles = []
        mapping = {}
        
        for title in titles:
            # 构建完整路径
            full_path = f"{current_path} - {title}"
            
            # 以replace_prob的概率替换
            if random.random() < self.replace_prob:
                paras = self.get_paraphrase_by_path(topic_key, full_path)
                if paras:
                    # 随机选择一种paraphrase
                    para_type = random.choice(list(paras.keys()))
                    perturbed_title = paras[para_type]
                    perturbed_titles.append(perturbed_title)
                    mapping[title] = perturbed_title
                else:
                    # 没有有效的paraphrase，保持原样
                    perturbed_titles.append(title)
                    mapping[title] = title
            else:
                # 不替换
                perturbed_titles.append(title)
                mapping[title] = title
        
        return perturbed_titles, mapping
    
    def generate_type1_data(
        self,
        article: str,
        current_path: str,
        all_siblings: List[str],
        output_titles: List[str],
        topic_key: str
    ) -> Dict:
        """
        生成第一类数据：有新标题
        
        流程：
        1. 10%概率从output中选一个混入约束，从output中删除
        2. 从剩余output中：70%概率选1个作为new_subtitles，30%选多个
        3. 剩余output作为selected_existing
        4. 构建existing_subtitles：包含selected_existing + 其他siblings（以概率保留）
        5. 统一对所有涉及的标题应用paraphrase替换，确保一致性
        
        Args:
            article: 文章内容
            current_path: 当前路径
            all_siblings: 所有兄弟节点（从结构树获取）
            output_titles: 输出标题列表
            topic_key: topic键
            
        Returns:
            数据样本
        """
        # 步骤1：先选择兄弟节点的约束
        # 构建临时的existing_subtitles用于选择约束
        temp_existing = output_titles.copy()
        for title in all_siblings:
            if title not in output_titles:
                if random.random() > self.delete_prob:
                    temp_existing.append(title)
        
        # 选择约束，并可能混入一个output
        constraint_paths, mixed_output = self.select_constraint_leaves(
            topic_key, current_path, temp_existing, output_titles=output_titles
        )
        
        # 步骤2：从output中移除混入的（如果有）
        remaining_outputs = [t for t in output_titles if t != mixed_output] if mixed_output else output_titles.copy()
        
        if not remaining_outputs:
            # 所有output都被混入约束了，特殊处理
            remaining_outputs = []
        
        # 步骤3：从剩余output中决定new_subtitles和selected_existing
        if len(remaining_outputs) == 0:
            deleted_outputs = []
            kept_outputs = []
        elif len(remaining_outputs) == 1:
            deleted_outputs = remaining_outputs.copy()
            kept_outputs = []
        elif random.random() < self.type1_single_new_prob:
            # 70%：只选1个作为new
            deleted_outputs = random.sample(remaining_outputs, 1)
            kept_outputs = [t for t in remaining_outputs if t not in deleted_outputs]
        else:
            # 30%：选多个作为new
            num_to_delete = random.randint(2, len(remaining_outputs))
            deleted_outputs = random.sample(remaining_outputs, num_to_delete)
            kept_outputs = [t for t in remaining_outputs if t not in deleted_outputs]
        
        # 步骤4：构建final_existing（原始标题）
        # 包含kept_outputs + 其他siblings（以概率保留）
        final_existing = kept_outputs.copy()
        for title in all_siblings:
            if title not in output_titles and title != mixed_output:
                # 非输出的，以(1-delete_prob)概率保留
                if random.random() > self.delete_prob:
                    final_existing.append(title)
        
        # 步骤5：统一应用扰动（关键：对所有相关标题统一替换）
        # 收集所有需要替换的标题
        all_titles_to_replace = set(final_existing + kept_outputs + deleted_outputs)
        
        # 对所有标题统一应用paraphrase替换
        title_mapping = {}
        for title in all_titles_to_replace:
            # 构建完整路径
            full_path = f"{current_path} - {title}"
            
            # 以replace_prob的概率替换
            if random.random() < self.replace_prob:
                paras = self.get_paraphrase_by_path(topic_key, full_path)
                if paras:
                    para_type = random.choice(list(paras.keys()))
                    title_mapping[title] = paras[para_type]
                else:
                    title_mapping[title] = title
            else:
                title_mapping[title] = title
        
        # 应用映射到各个列表
        perturbed_existing = [title_mapping[t] for t in final_existing]
        perturbed_kept = [title_mapping[t] for t in kept_outputs]
        perturbed_deleted = [title_mapping[t] for t in deleted_outputs]
        
        return {
            'article': article,
            'current_path': current_path,
            'existing_subtitles': perturbed_existing,
            'selected_existing': perturbed_kept,
            'new_subtitles': perturbed_deleted,
            'constraint_paths': constraint_paths,
            'data_type': 1
        }
    
    def generate_type2_data(
        self,
        article: str,
        current_path: str,
        all_siblings: List[str],
        output_titles: List[str],
        topic_key: str
    ) -> Dict:
        """
        生成第二类数据：只有existing，没有新标题
        
        流程：
        1. 10%概率从output中选一个混入约束，从output中删除
        2. 剩余所有output作为selected_existing
        3. 构建existing_subtitles：selected_existing + 其他siblings（以概率保留）
        4. 统一对所有涉及的标题应用paraphrase替换
        5. new_subtitles为空
        
        Args:
            article: 文章内容
            current_path: 当前路径
            all_siblings: 所有兄弟节点
            output_titles: 输出标题列表
            topic_key: topic键
            
        Returns:
            数据样本
        """
        # 步骤1：构建临时existing用于选择约束
        temp_existing = output_titles.copy()
        for title in all_siblings:
            if title not in output_titles:
                if random.random() > self.delete_prob:
                    temp_existing.append(title)
        
        # 选择约束，并可能混入一个output
        constraint_paths, mixed_output = self.select_constraint_leaves(
            topic_key, current_path, temp_existing, output_titles=output_titles
        )
        
        # 步骤2：从output中移除混入的（如果有）
        remaining_outputs = [t for t in output_titles if t != mixed_output] if mixed_output else output_titles.copy()
        
        # 步骤3：构建existing_subtitles（原始标题）
        # 包含剩余的output + 其他siblings（以概率保留）
        existing_subtitles = remaining_outputs.copy()
        for title in all_siblings:
            if title not in output_titles and title != mixed_output:
                if random.random() > self.delete_prob:
                    existing_subtitles.append(title)
        
        # 步骤4：统一应用paraphrase替换
        all_titles_to_replace = set(existing_subtitles + remaining_outputs)
        
        title_mapping = {}
        for title in all_titles_to_replace:
            full_path = f"{current_path} - {title}"
            
            if random.random() < self.replace_prob:
                paras = self.get_paraphrase_by_path(topic_key, full_path)
                if paras:
                    para_type = random.choice(list(paras.keys()))
                    title_mapping[title] = paras[para_type]
                else:
                    title_mapping[title] = title
            else:
                title_mapping[title] = title
        
        # 应用映射
        perturbed_existing = [title_mapping[t] for t in existing_subtitles]
        perturbed_outputs = [title_mapping[t] for t in remaining_outputs]
        
        return {
            'article': article,
            'current_path': current_path,
            'existing_subtitles': perturbed_existing,
            'selected_existing': perturbed_outputs,
            'new_subtitles': [],
            'constraint_paths': constraint_paths,
            'data_type': 2
        }
    
    def generate_type3_data(
        self,
        article: str,
        leaf_path: str,
        topic_key: str = None
    ) -> Dict:
        """
        生成第三类数据：叶子节点，都为空
        
        修改：添加约束（来自兄弟节点的叶子，但叶子节点通常没有existing_subtitles）
        
        Args:
            article: 文章内容
            leaf_path: 叶子节点路径
            topic_key: topic键（用于选择约束）
            
        Returns:
            数据样本
        """
        # 叶子节点通常没有existing_subtitles，所以约束为空
        # 但为了完整性，还是调用一次
        constraint_paths, _ = self.select_constraint_leaves(
            topic_key, leaf_path, []
        ) if topic_key else ([], None)
        
        return {
            'article': article,
            'current_path': leaf_path,
            'existing_subtitles': [],
            'selected_existing': [],
            'new_subtitles': [],
            'constraint_paths': constraint_paths,
            'data_type': 3
        }
    
    def generate_data_for_reference(
        self,
        topic_key: str,
        ref_id: str,
        structure_dict: Dict[str, List[str]]
    ) -> List[Dict]:
        """
        为一个引用文章生成所有数据
        
        Args:
            topic_key: topic键
            ref_id: 引用ID
            structure_dict: 结构字典
            
        Returns:
            数据样本列表
        """
        reference = self.references[topic_key]['references'][ref_id]
        article = reference.get('content', '')
        paths = reference.get('paths', [])
        
        if not paths:
            return []
        
        data_samples = []
        
        # 解析路径为分类步骤
        classifications = self.parse_paths_to_classifications(paths)
        
        # 为每个分类步骤生成第一类和第二类数据
        for current_path, output_titles in classifications:
            # 获取所有兄弟节点
            all_siblings = structure_dict.get(current_path, [])
            
            if not all_siblings:
                # 如果结构字典中没有，可能是因为这个路径只出现在这篇文章中
                # 使用output_titles作为all_siblings
                all_siblings = output_titles
            
            # 生成第一类数据
            data1 = self.generate_type1_data(
                article, current_path, all_siblings, output_titles, topic_key
            )
            data_samples.append(data1)
            
            # 生成第二类数据
            data2 = self.generate_type2_data(
                article, current_path, all_siblings, output_titles, topic_key
            )
            data_samples.append(data2)
        
        # 为所有叶子节点生成第三类数据
        for path in paths:
            data3 = self.generate_type3_data(article, path, topic_key)
            data_samples.append(data3)
        
        return data_samples
    
    def generate_data_for_split(self, split_name: str) -> List[Dict]:
        """
        为一个数据集划分生成数据
        
        Args:
            split_name: 'train', 'test_easy', 'test_hard'
            
        Returns:
            数据样本列表
        """
        print(f"\n{'=' * 80}")
        print(f"生成 {split_name} 数据...")
        print(f"{'=' * 80}")
        
        all_data = []
        split_data = self.dataset_split['dataset_split'][split_name]
        
        for topic_key, ref_ids in tqdm(split_data.items(), desc=f"处理{split_name}的topics"):
            # 构建结构字典
            structure_dict = self.build_structure_dict(topic_key)
            
            # 为每个引用生成数据
            for ref_id in ref_ids:
                samples = self.generate_data_for_reference(topic_key, ref_id, structure_dict)
                all_data.extend(samples)
        
        # 统计各类数据的数量
        type1_count = sum(1 for d in all_data if d['data_type'] == 1)
        type2_count = sum(1 for d in all_data if d['data_type'] == 2)
        type3_count = sum(1 for d in all_data if d['data_type'] == 3)
        
        print(f"\n{split_name} 数据生成完成:")
        print(f"  - 第一类（有新标题）: {type1_count}")
        print(f"  - 第二类（只有existing）: {type2_count}")
        print(f"  - 第三类（都为空）: {type3_count}")
        print(f"  - 总计: {len(all_data)}")
        
        return all_data
    
    def sample_and_split(self, all_data: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        按照比例采样数据，并划分训练集和验证集
        
        Args:
            all_data: 所有数据
            
        Returns:
            (训练集, 验证集)
        """
        print(f"\n{'=' * 80}")
        print("采样和划分数据...")
        print(f"{'=' * 80}")
        
        # 按类型分组
        type1_data = [d for d in all_data if d['data_type'] == 1]
        type2_data = [d for d in all_data if d['data_type'] == 2]
        type3_data = [d for d in all_data if d['data_type'] == 3]
        
        print(f"\n原始数据统计:")
        print(f"  - 第一类: {len(type1_data)}")
        print(f"  - 第二类: {len(type2_data)}")
        print(f"  - 第三类: {len(type3_data)}")
        
        # 计算每类需要采样的数量
        ratio_sum = sum(self.class_ratio)
        type1_target = int(self.train_size * self.class_ratio[0] / ratio_sum)
        type2_target = int(self.train_size * self.class_ratio[1] / ratio_sum)
        type3_target = self.train_size - type1_target - type2_target
        
        print(f"\n目标数据量 (总计 {self.train_size}):")
        print(f"  - 第一类: {type1_target}")
        print(f"  - 第二类: {type2_target}")
        print(f"  - 第三类: {type3_target}")
        
        # 采样
        random.shuffle(type1_data)
        random.shuffle(type2_data)
        random.shuffle(type3_data)
        
        # 如果数据不够，就用全部
        sampled_type1 = type1_data[:type1_target] if len(type1_data) >= type1_target else type1_data
        sampled_type2 = type2_data[:type2_target] if len(type2_data) >= type2_target else type2_data
        sampled_type3 = type3_data[:type3_target] if len(type3_data) >= type3_target else type3_data
        
        # 合并
        sampled_data = sampled_type1 + sampled_type2 + sampled_type3
        random.shuffle(sampled_data)
        
        print(f"\n实际采样数量:")
        print(f"  - 第一类: {len(sampled_type1)}")
        print(f"  - 第二类: {len(sampled_type2)}")
        print(f"  - 第三类: {len(sampled_type3)}")
        print(f"  - 总计: {len(sampled_data)}")
        
        # 划分训练集和验证集
        val_size = int(len(sampled_data) * self.val_ratio)
        val_data = sampled_data[:val_size]
        train_data = sampled_data[val_size:]
        
        print(f"\n最终数据集:")
        print(f"  - 训练集: {len(train_data)}")
        print(f"  - 验证集: {len(val_data)}")
        
        return train_data, val_data
    
    def format_for_training(self, data_sample: Dict) -> Dict:
        """
        将数据样本格式化为训练格式
        
        Args:
            data_sample: 数据样本
            
        Returns:
            格式化后的样本
        """
        # 格式化existing subtitles
        existing_str = ", ".join(data_sample['existing_subtitles']) if data_sample['existing_subtitles'] else "None"
        
        # 格式化constraint paths
        constraint_paths = data_sample.get('constraint_paths', [])
        constraint_str = ", ".join(constraint_paths) if constraint_paths else "None"
        
        # 构建prompt（与classifier.py中的_create_prompt保持完全一致）
        prompt = f"""You are a hierarchical content classifier for Wikipedia articles.

TOPIC PATH: {data_sample['current_path']}
This is the current hierarchical path in the Wikipedia structure. Any subtitles you identify MUST be direct children of this topic path.

EXISTING SUBTITLES: {existing_str}
These are subtitles that already exist under the current topic path.

CONSTRAINT SUBTITLES: {constraint_str}
These are subtitles from other branches in the hierarchy. You should NOT output or create subtitles that are the same as or similar to these constraint subtitles, as they belong to different parts of the structure.

ARTICLE CONTENT:
{data_sample['article'][:3000]}

TASK:
1. If the article relates to any EXISTING subtitles, list them in "selected_existing"
2. If the article introduces NEW content that needs new subtitles under "{data_sample['current_path']}", list them in "new_subtitles"
3. All subtitles must be direct children of "{data_sample['current_path']}" in the Wikipedia hierarchy
4. Use concise, Wikipedia-style subtitle names (1-10 words)
5. If no subtitles apply, return empty arrays
6. IMPORTANT: Do NOT create subtitles similar to the CONSTRAINT SUBTITLES listed above

CRITICAL: Output ONLY a valid JSON object. No explanation, no additional text.

JSON OUTPUT:
{{
  "selected_existing": [],
  "new_subtitles": []
}}"""
        
        # 构建completion（注意：SFTTrainer期望的字段名是'completion'，不是'response'）
        completion = json.dumps({
            "selected_existing": data_sample['selected_existing'],
            "new_subtitles": data_sample['new_subtitles']
        }, ensure_ascii=False)
        
        return {
            'prompt': prompt,
            'completion': completion
        }
    
    def save_dataset(self, train_data: List[Dict], val_data: List[Dict]):
        """保存数据集"""
        print(f"\n{'=' * 80}")
        print("保存数据集...")
        print(f"{'=' * 80}")
        
        # 格式化数据
        train_formatted = [self.format_for_training(d) for d in train_data]
        val_formatted = [self.format_for_training(d) for d in val_data]
        
        # 保存
        train_file = self.output_dir / 'train_dataset.json'
        val_file = self.output_dir / 'val_dataset.json'
        
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_formatted, f, indent=2, ensure_ascii=False)
        print(f"\n训练集已保存到: {train_file}")
        
        with open(val_file, 'w', encoding='utf-8') as f:
            json.dump(val_formatted, f, indent=2, ensure_ascii=False)
        print(f"验证集已保存到: {val_file}")
        
        # 保存统计信息
        stats = {
            'train_size': len(train_data),
            'val_size': len(val_data),
            'class_ratio': self.class_ratio,
            'delete_prob': self.delete_prob,
            'replace_prob': self.replace_prob,
            'val_ratio': self.val_ratio,
            'seed': self.seed
        }
        
        stats_file = self.output_dir / 'dataset_stats.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"统计信息已保存到: {stats_file}")
    
    def run(self):
        """执行数据准备流程"""
        print("=" * 80)
        print("数据集准备")
        print("=" * 80)
        
        # 1. 加载数据
        self.load_data()
        
        # 2. 生成训练数据
        train_all_data = self.generate_data_for_split('train')
        
        # 3. 采样和划分
        train_data, val_data = self.sample_and_split(train_all_data)
        
        # 4. 保存数据集
        self.save_dataset(train_data, val_data)
        
        print("\n" + "=" * 80)
        print("数据集准备完成！")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='准备训练数据集')
    parser.add_argument(
        '--references_file',
        type=str,
        default='/mnt/literism/tree/data/wikipedia_references_final.json',
        help='引用文件路径'
    )
    parser.add_argument(
        '--structures_file',
        type=str,
        default='/mnt/literism/tree/data/wikipedia_structures_final.json',
        help='结构文件路径'
    )
    parser.add_argument(
        '--paraphrases_dir',
        type=str,
        default='./data/paraphrases',
        help='paraphrase目录'
    )
    parser.add_argument(
        '--dataset_split_file',
        type=str,
        default='./data/dataset_split.json',
        help='数据集划分文件'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./data/datasets',
        help='输出目录'
    )
    parser.add_argument(
        '--delete_prob',
        type=float,
        default=0.1,
        help='删除非输出title的概率'
    )
    parser.add_argument(
        '--replace_prob',
        type=float,
        default=0.1,
        help='替换为paraphrase的概率'
    )
    parser.add_argument(
        '--class_ratio',
        type=str,
        default='2:1:1',
        help='三类数据的比例，格式如"2:1:1"'
    )
    parser.add_argument(
        '--train_size',
        type=int,
        default=10000,
        help='训练集总大小'
    )
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.1,
        help='验证集比例'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子'
    )
    parser.add_argument(
        '--num_constraint_leaves',
        type=int,
        default=10,
        help='选择多少个叶子节点作为约束'
    )
    parser.add_argument(
        '--type1_single_new_prob',
        type=float,
        default=0.7,
        help='类型1数据中只放一个new_subtitle的概率'
    )
    parser.add_argument(
        '--mix_output_to_constraint_prob',
        type=float,
        default=0.1,
        help='将一个输出混入约束的概率'
    )
    
    args = parser.parse_args()
    
    # 解析比例
    class_ratio = tuple(map(int, args.class_ratio.split(':')))
    if len(class_ratio) != 3:
        raise ValueError("class_ratio必须是3个整数，如'2:1:1'")
    
    preparator = DatasetPreparator(
        references_file=args.references_file,
        structures_file=args.structures_file,
        paraphrases_dir=args.paraphrases_dir,
        dataset_split_file=args.dataset_split_file,
        output_dir=args.output_dir,
        delete_prob=args.delete_prob,
        replace_prob=args.replace_prob,
        class_ratio=class_ratio,
        train_size=args.train_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_constraint_leaves=args.num_constraint_leaves,
        type1_single_new_prob=args.type1_single_new_prob,
        mix_output_to_constraint_prob=args.mix_output_to_constraint_prob
    )
    
    preparator.run()


if __name__ == '__main__':
    main()
