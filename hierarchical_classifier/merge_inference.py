"""
合并系统推理：对推理生成的结构树进行合并
使用Leiden算法进行聚类
"""
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import networkx as nx
from tqdm import tqdm


class MergeInference:
    """合并系统推理器"""
    
    def __init__(
        self,
        base_model: str,
        lora_model: str,
        test_trees_file: str,
        output_file: str,
        resolution: float = 1.0
    ):
        """
        Args:
            base_model: 基础模型路径
            lora_model: LoRA模型路径
            test_trees_file: 测试树文件路径
            output_file: 输出文件路径
            resolution: Leiden算法的resolution参数（越大簇越小）
        """
        self.base_model = base_model
        self.lora_model = lora_model
        self.test_trees_file = test_trees_file
        self.output_file = output_file
        self.resolution = resolution
        
        self.model = None
        self.tokenizer = None
        self.test_trees = {}
    
    def load_model(self):
        """加载模型"""
        print("="*80)
        print("加载模型...")
        print("="*80)
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载基础模型
        print(f"加载基础模型: {self.base_model}")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            trust_remote_code=True
        )
        
        # 加载LoRA权重
        print(f"加载LoRA权重: {self.lora_model}")
        self.model = PeftModel.from_pretrained(
            base_model,
            self.lora_model,
            torch_dtype=torch.bfloat16
        )
        
        self.model.eval()
        print("模型加载完成")
    
    def load_test_trees(self):
        """加载测试树"""
        print("\n" + "="*80)
        print("加载测试树...")
        print("="*80)
        
        with open(self.test_trees_file, 'r', encoding='utf-8') as f:
            self.test_trees = json.load(f)
        
        print(f"加载了 {len(self.test_trees)} 个topics的树")
    
    def predict_merge(self, topic: str, summary1: str, summary2: str) -> bool:
        """
        预测两个类别是否应该合并
        
        Args:
            topic: Topic名称
            summary1: 第一个类别的summary
            summary2: 第二个类别的summary
            
        Returns:
            True表示应该合并，False表示不应该合并
        """
        # 构建prompt
        prompt = f"""You are tasked with determining whether two categories from a Wikipedia topic should be merged into one.

TOPIC: {topic}

CATEGORY 1 SUMMARY:
{summary1}

CATEGORY 2 SUMMARY:
{summary2}

TASK:
Determine if these two categories are similar enough that they should be merged into a single category. Consider:
1. Do they cover similar or overlapping content?
2. Would it make sense to combine them for better organization?
3. Are they at the same conceptual level?

Answer with ONLY "Yes" or "No".

ANSWER:"""
        
        # 编码
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=4096
        ).to(self.model.device)
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # 解码
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        # 判断
        response_lower = response.lower()
        if 'yes' in response_lower:
            return True
        else:
            return False
    
    def get_first_layer_nodes(self, tree: List[Dict]) -> List[Dict]:
        """
        获取树的第一层节点
        
        Args:
            tree: 树结构
            
        Returns:
            第一层节点列表
        """
        return tree if isinstance(tree, list) else [tree]
    
    def merge_tree_for_topic(self, topic_key: str, tree: List[Dict]) -> List[Dict]:
        """
        对一个topic的树进行合并
        
        Args:
            topic_key: Topic键
            tree: 树结构
            
        Returns:
            合并后的树
        """
        # 获取第一层节点
        first_layer = self.get_first_layer_nodes(tree)
        
        if len(first_layer) <= 1:
            # 只有一个或没有节点，无需合并
            return tree
        
        topic_name = self.test_trees.get(topic_key, {}).get('topic', topic_key)
        
        print(f"\n处理 {topic_name}:")
        print(f"  - 第一层节点数: {len(first_layer)}")
        
        # 两两比较，构建图
        G = nx.Graph()
        
        # 添加节点
        for i in range(len(first_layer)):
            G.add_node(i)
        
        # 添加边（判断为应该合并的）
        merge_count = 0
        for i in range(len(first_layer)):
            for j in range(i + 1, len(first_layer)):
                node_i = first_layer[i]
                node_j = first_layer[j]
                
                summary_i = node_i.get('summary', '')
                summary_j = node_j.get('summary', '')
                
                if not summary_i or not summary_j:
                    continue
                
                # 预测是否合并
                should_merge = self.predict_merge(topic_name, summary_i, summary_j)
                
                if should_merge:
                    G.add_edge(i, j)
                    merge_count += 1
        
        print(f"  - 判断应该合并的节点对数: {merge_count}")
        
        # 使用Leiden算法聚类
        try:
            import leidenalg
            import igraph as ig
            
            # 转换为igraph
            edges = list(G.edges())
            if not edges:
                # 没有边，不需要合并
                print(f"  - 无需合并")
                return tree
            
            ig_graph = ig.Graph()
            ig_graph.add_vertices(len(first_layer))
            ig_graph.add_edges(edges)
            
            # Leiden聚类
            partition = leidenalg.find_partition(
                ig_graph,
                leidenalg.ModularityVertexPartition,
                resolution_parameter=self.resolution
            )
            
            # 获取聚类结果
            clusters = {}
            for node_idx, cluster_id in enumerate(partition.membership):
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(node_idx)
            
            print(f"  - 聚类后簇数: {len(clusters)}")
            
            # 重构树
            new_tree = []
            for cluster_id, node_indices in clusters.items():
                if len(node_indices) == 1:
                    # 单个节点，保持不变
                    new_tree.append(first_layer[node_indices[0]])
                else:
                    # 多个节点，创建新父节点
                    merged_node = {
                        'summary': '',  # 新父节点summary留空
                        'children': [first_layer[idx] for idx in node_indices]
                    }
                    new_tree.append(merged_node)
                    print(f"  - 合并了 {len(node_indices)} 个节点到一个簇")
            
            return new_tree
            
        except ImportError:
            print("警告: 未安装leidenalg或igraph，使用连通分量代替")
            # 使用连通分量作为后备
            clusters_list = list(nx.connected_components(G))
            
            print(f"  - 连通分量数: {len(clusters_list)}")
            
            new_tree = []
            for cluster in clusters_list:
                node_indices = list(cluster)
                if len(node_indices) == 1:
                    new_tree.append(first_layer[node_indices[0]])
                else:
                    merged_node = {
                        'summary': '',
                        'children': [first_layer[idx] for idx in node_indices]
                    }
                    new_tree.append(merged_node)
                    print(f"  - 合并了 {len(node_indices)} 个节点")
            
            return new_tree
    
    def merge_all_trees(self):
        """对所有树进行合并"""
        print("\n" + "="*80)
        print("开始合并所有树...")
        print("="*80)
        
        merged_trees = {}
        
        for topic_key in tqdm(self.test_trees.keys(), desc="处理topics"):
            topic_data = self.test_trees[topic_key]
            tree = topic_data.get('tree', [])
            
            if not tree:
                merged_trees[topic_key] = topic_data
                continue
            
            # 合并树
            merged_tree = self.merge_tree_for_topic(topic_key, tree)
            
            # 更新
            merged_trees[topic_key] = {
                **topic_data,
                'tree': merged_tree
            }
        
        return merged_trees
    
    def save_results(self, merged_trees: Dict):
        """保存结果"""
        print("\n" + "="*80)
        print("保存结果...")
        print("="*80)
        
        output_path = Path(self.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_trees, f, ensure_ascii=False, indent=2)
        
        print(f"结果保存到: {output_path}")
    
    def run(self):
        """运行合并流程"""
        self.load_model()
        self.load_test_trees()
        merged_trees = self.merge_all_trees()
        self.save_results(merged_trees)
        
        print("\n" + "="*80)
        print("合并完成！")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description='合并系统推理')
    
    parser.add_argument(
        '--base_model',
        type=str,
        required=True,
        help='基础模型路径'
    )
    parser.add_argument(
        '--lora_model',
        type=str,
        required=True,
        help='LoRA模型路径'
    )
    parser.add_argument(
        '--test_trees',
        type=str,
        default='/mnt/literism/tree/summary_output/inference/test_trees.json',
        help='测试树文件路径'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='输出文件路径'
    )
    parser.add_argument(
        '--resolution',
        type=float,
        default=1.0,
        help='Leiden算法resolution参数'
    )
    
    args = parser.parse_args()
    
    inference = MergeInference(
        base_model=args.base_model,
        lora_model=args.lora_model,
        test_trees_file=args.test_trees,
        output_file=args.output,
        resolution=args.resolution
    )
    
    inference.run()


if __name__ == '__main__':
    main()

