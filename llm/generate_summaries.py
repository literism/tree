"""
生成结构树节点的summary
"""
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
sys.path.append(str(Path(__file__).parent.parent))
from modeling.deepseek_api import DeepSeekAPIClient, DeepSeekConfig
from summary_based_classifier.config import SummaryBasedConfig
from summary_based_classifier.llm.prompts import PromptTemplates


class SummaryGenerator:
    def __init__(self, config: SummaryBasedConfig):
        """
        Args:
            config: 配置对象
        """
        self.config = config
        self.output_dir = Path(config.path.summaries_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.structures_data = None
        
    def load_structures(self):
        """加载结构数据"""
        print(f"加载结构数据: {self.config.path.structures_file}")
        with open(self.config.path.structures_file, 'r', encoding='utf-8') as f:
            self.structures_data = json.load(f)
        print(f"  - 加载 {len(self.structures_data)} 个topics")
    
    def truncate_content(self, content: str, max_length: int) -> str:
        """截断内容到指定长度"""
        if len(content) <= max_length:
            return content
        return content[:max_length] + "..."
    
    def get_node_content_with_children(self, node: Dict) -> str:
        """
        获取节点的内容，如果有子节点则拼接子节点的内容
        
        如果所有子节点内容总长度超过max_content_length，则平均分配每个子节点的长度
        
        Args:
            node: 节点字典
            
        Returns:
            拼接后的内容字符串
        """
        max_length = self.config.summary.max_content_length
        
        # 获取当前节点的content
        node_content = node.get('content', '')
        
        # 如果有子节点，拼接子节点的title和content
        if 'children' in node and node['children']:
            children = node['children']
            
            # 计算所有子节点内容的总长度
            children_contents = []
            total_length = 0
            for child in children:
                child_content = child.get('content', '')
                children_contents.append((child['title'], child_content))
                total_length += len(child_content)
            
            # 如果总长度超过限制，平均分配长度
            if total_length > max_length:
                per_child_length = max_length // len(children)
                result_parts = []
                for title, content in children_contents:
                    truncated = self.truncate_content(content, per_child_length)
                    result_parts.append(f"**{title}**\n{truncated}")
                return "\n\n".join(result_parts)
            else:
                # 不超过限制，直接拼接
                result_parts = []
                for title, content in children_contents:
                    if content:
                        result_parts.append(f"**{title}**\n{content}")
                return "\n\n".join(result_parts)
        else:
            # 没有子节点，返回当前节点内容（可能需要截断）
            return self.truncate_content(node_content, max_length)
    
    def create_summary_prompt(self, path: str, content: str) -> str:
        """
        创建生成summary的prompt
        
        Args:
            path: 节点路径，如 "Anarchism - History - Origins"
            content: 节点内容（可能包含子节点内容）
            
        Returns:
            prompt字符串
        """
        return PromptTemplates.format_summary_generation_prompt(path, content)
    
    def parse_summary_response(self, response: str) -> Optional[Dict]:
        """
        解析API返回的summary
        
        Args:
            response: API返回的文本
            
        Returns:
            包含explanation和scope的字典，如果解析失败返回None
        """
        try:
            lines = response.strip().split('\n')
            explanation = ""
            scope = ""
            current_part = None
            
            for line in lines:
                line = line.strip()
                if line.startswith('EXPLANATION:'):
                    current_part = 'explanation'
                    explanation = line[len('EXPLANATION:'):].strip()
                elif line.startswith('SCOPE:'):
                    current_part = 'scope'
                    scope = line[len('SCOPE:'):].strip()
                elif current_part == 'explanation':
                    explanation += ' ' + line
                elif current_part == 'scope':
                    scope += ' ' + line
            
            if explanation and scope:
                return {
                    'explanation': explanation.strip(),
                    'scope': scope.strip(),
                    'full': f"{explanation.strip()}\n{scope.strip()}"
                }
            else:
                return None
        except Exception as e:
            print(f"解析summary失败: {e}")
            return None
    
    def generate_summaries_for_tree(self, topic_key: str, structure: Dict) -> Dict:
        """
        为一个结构树生成所有节点的summary
        
        Args:
            topic_key: topic键
            structure: 结构字典
            
        Returns:
            包含所有节点summary的字典，key为路径，value为summary
        """
        topic_name = structure.get('topic', topic_key)
        summaries = {}
        
        # 收集所有节点的prompt
        prompts_list = []
        paths_list = []
        
        def collect_node_prompts(node: Dict, path: str):
            """递归收集所有节点的prompts"""
            # 获取节点内容
            content = self.get_node_content_with_children(node)
            
            if content:  # 只处理有内容的节点
                # 创建prompt
                prompt = self.create_summary_prompt(path, content)
                prompts_list.append(prompt)
                paths_list.append(path)
            
            # 递归处理子节点
            if 'children' in node:
                for child in node['children']:
                    child_path = f"{path} - {child['title']}"
                    collect_node_prompts(child, child_path)
        
        # 收集所有节点
        for root_node in structure.get('structure', []):
            root_path = f"{topic_name} - {root_node['title']}"
            collect_node_prompts(root_node, root_path)
        
        print(f"  - 需要生成 {len(prompts_list)} 个节点的summary")
        
        if not prompts_list:
            return summaries
        
        # 批量调用API
        print(f"  - 调用DeepSeek API生成summaries...")
        
        # 创建API客户端
        api_config = DeepSeekConfig(
            api_key=self.config.summary.api_key,
            base_url=self.config.summary.api_url.replace('/chat/completions', ''),
            model=self.config.summary.model_name,
            temperature=self.config.summary.temperature,
            max_output_tokens=self.config.summary.max_tokens,
            max_concurrent_jobs=self.config.summary.max_workers
        )
        client = DeepSeekAPIClient(api_config)
        
        # 调用API
        jobs = client.run_prompts(prompts_list)
        
        # 解析responses
        success_count = 0
        for path, job in zip(paths_list, jobs):
            if job.status == 'completed' and job.result:
                response_text = client.extract_text(job.result)
                if response_text:
                    parsed = self.parse_summary_response(response_text)
                    if parsed:
                        summaries[path] = parsed
                        success_count += 1
                    else:
                        print(f"  警告: 解析失败 - {path}")
                else:
                    print(f"  警告: API返回空 - {path}")
            else:
                print(f"  警告: API调用失败 - {path}: {job.error}")
        
        print(f"  - 成功生成 {success_count}/{len(prompts_list)} 个summaries")
        
        return summaries
    
    def test_single_generation(self, topic_key: str, node_path: Optional[str] = None):
        """
        测试模式：为一个topic生成一个节点的summary
        
        Args:
            topic_key: 要测试的topic键
            node_path: 要测试的节点路径，如 "Topic - Title1 - Title2"
                      如果为None，则测试第一个根节点
        """
        print("="*80)
        print(f"测试模式：生成单个节点summary")
        print("="*80)
        
        if topic_key not in self.structures_data:
            print(f"错误: topic {topic_key} 不存在")
            return
        
        structure = self.structures_data[topic_key]
        topic_name = structure.get('topic', topic_key)
        
        if not structure.get('structure'):
            print(f"错误: topic {topic_key} 没有结构节点")
            return
        
        # 查找要测试的节点
        if node_path:
            # 解析路径
            path_parts = [p.strip() for p in node_path.split(' - ')]
            print(f"\n查找节点路径: {node_path}")
            
            # 递归查找节点
            current_nodes = structure['structure']
            target_node = None
            current_path = topic_name
            
            for i, target_title in enumerate(path_parts[1:], 1):  # 跳过topic名
                found = False
                for node in current_nodes:
                    if node['title'] == target_title:
                        target_node = node
                        current_path = f"{current_path} - {node['title']}"
                        current_nodes = node.get('children', [])
                        found = True
                        print(f"  找到第{i}层: {target_title}")
                        break
                
                if not found:
                    print(f"  错误: 找不到节点 '{target_title}' 在第{i}层")
                    return
            
            if target_node is None:
                print(f"错误: 无法定位到节点")
                return
            
            path = current_path
            content = self.get_node_content_with_children(target_node)
        else:
            # 默认测试第一个根节点
            target_node = structure['structure'][0]
            path = f"{topic_name} - {target_node['title']}"
            content = self.get_node_content_with_children(target_node)
        
        print(f"\n节点路径: {path}")
        print(f"节点层级: level {target_node.get('level', 'unknown')}")
        print(f"是否有子节点: {'是' if target_node.get('children') else '否'}")
        if target_node.get('children'):
            print(f"子节点数量: {len(target_node['children'])}")
            print(f"子节点: {[c['title'] for c in target_node['children'][:5]]}")
        
        print(f"\n节点内容 (前500字符):\n{content[:500]}...")
        
        # 创建prompt
        prompt = self.create_summary_prompt(path, content)
        print(f"\n生成的Prompt:\n{prompt}")
        
        # 调用API
        print(f"\n调用DeepSeek API...")
        
        # 创建API客户端
        api_config = DeepSeekConfig(
            api_key=self.config.summary.api_key,
            base_url=self.config.summary.api_url.replace('/chat/completions', ''),
            model=self.config.summary.model_name,
            temperature=self.config.summary.temperature,
            max_output_tokens=self.config.summary.max_tokens,
            max_concurrent_jobs=1
        )
        client = DeepSeekAPIClient(api_config)
        
        # 调用API
        jobs = client.run_prompts([prompt])
        
        if jobs and jobs[0].status == 'completed' and jobs[0].result:
            response_text = client.extract_text(jobs[0].result)
            print(f"\nAPI返回结果:\n{response_text}")
            
            # 解析
            parsed = self.parse_summary_response(response_text)
            if parsed:
                print(f"\n解析后的Summary:")
                print(f"  Explanation: {parsed['explanation'][:1000]}...")
                print(f"  Scope: {parsed['scope'][:1000]}...")
            else:
                print(f"\n解析失败！")
        else:
            print(f"\nAPI调用失败！错误: {jobs[0].error if jobs else 'Unknown'}")
        
        print("\n" + "="*80)
    
    def run(self, test_mode: bool = False, test_topic: Optional[str] = None, test_path: Optional[str] = None):
        """
        执行生成流程
        
        Args:
            test_mode: 是否为测试模式
            test_topic: 测试模式下的topic键
            test_path: 测试模式下的节点路径，如 "Topic - Title1 - Title2"
        """
        # 加载结构数据
        self.load_structures()
        
        if test_mode:
            # 测试模式
            if test_topic:
                self.test_single_generation(test_topic, test_path)
            else:
                # 随机选择一个topic测试
                import random
                test_topic = random.choice(list(self.structures_data.keys()))
                self.test_single_generation(test_topic, test_path)
        else:
            # 正式生成模式
            print("\n开始为所有topics生成summaries...")
            all_summaries = {}
            
            for i, (topic_key, structure) in enumerate(self.structures_data.items(), 1):
                print(f"\n处理 [{i}/{len(self.structures_data)}]: {topic_key}")
                
                try:
                    summaries = self.generate_summaries_for_tree(topic_key, structure)
                    all_summaries[topic_key] = summaries
                except Exception as e:
                    print(f"  错误: {e}")
                    continue
            
            # 保存结果
            output_file = self.output_dir / 'node_summaries.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_summaries, f, indent=2, ensure_ascii=False)
            
            print(f"\n所有summaries已保存到: {output_file}")
            
            # 统计
            total_nodes = sum(len(s) for s in all_summaries.values())
            print(f"总计: 为 {len(all_summaries)} 个topics生成了 {total_nodes} 个节点的summary")


def main():
    parser = argparse.ArgumentParser(description='生成结构树节点的summary')
    parser.add_argument(
        '--config',
        type=str,
        default='./configs/default.json',
        help='配置文件路径'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='测试模式：只生成一个节点的summary'
    )
    parser.add_argument(
        '--test_topic',
        type=str,
        default="Book:The Hobbit",
        help='测试模式下指定的topic键'
    )
    parser.add_argument(
        '--test_path',
        type=str,
        default="The Hobbit - Influences - 19th century fiction",
        help='测试模式下指定的节点路径，如 "Anarchism - History - Origins"'
    )
    
    args = parser.parse_args()
    
    # 加载配置
    config = SummaryBasedConfig.from_json(args.config)
    
    # 创建生成器并运行
    generator = SummaryGenerator(config)
    generator.run(test_mode=args.test, test_topic=args.test_topic, test_path=args.test_path)


if __name__ == '__main__':
    main()

