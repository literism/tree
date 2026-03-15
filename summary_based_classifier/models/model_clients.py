"""
HTTP客户端版本的分类生成系统和总结更新系统
通过HTTP API调用远程服务
"""
import requests
from typing import List
from summary_based_classifier.llm.classify_generator import ClassifyGenerateInput, ClassifyGenerateOutput
from summary_based_classifier.llm.updater import UpdateInput, UpdateOutput


class ClassifyGeneratorClient:
    """分类生成系统HTTP客户端（适配vLLM OpenAI API）"""
    
    def __init__(self, service_url: str = "http://localhost:8000"):
        """
        Args:
            service_url: vLLM服务URL（OpenAI兼容）
        """
        self.service_url = service_url.rstrip('/')
        self.mode = 'client'
        self.model_name = None
        
        # 检查服务是否可用，并获取模型名称
        try:
            response = requests.get(f"{self.service_url}/v1/models", timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                if models_data.get('data'):
                    self.model_name = models_data['data'][0]['id']
                    print(f"✓ 分类生成系统服务连接成功: {self.service_url}")
                    print(f"  模型: {self.model_name}")
                else:
                    print(f"⚠ 未找到可用模型")
            else:
                print(f"⚠ 分类生成系统服务响应异常: {response.status_code}")
        except Exception as e:
            print(f"✗ 无法连接到分类生成系统服务: {e}")
    
    def classify_generate(self, input_data: ClassifyGenerateInput) -> ClassifyGenerateOutput:
        """
        执行分类和生成（通过vLLM OpenAI API）
        
        Args:
            input_data: 分类生成输入
            
        Returns:
            ClassifyGenerateOutput对象
        """
        try:
            from summary_based_classifier.llm.prompts import PromptTemplates
            
            # 创建prompt
            prompt = PromptTemplates.format_classify_generator_prompt(
                topic_name=input_data.topic_name,
                current_summary=input_data.current_node_summary,
                article_content=input_data.article_content,
                child_summaries=input_data.child_summaries
            )
            
            # 调用vLLM OpenAI Chat API
            response = requests.post(
                f"{self.service_url}/v1/chat/completions",
                json={
                    "model": self.model_name or "default",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 4096,
                    "temperature": 0,
                    "top_p": 1,
                    "stop": ["\n\n\n", "###"]
                },
                timeout=120
            )
            response.raise_for_status()
            
            # 解析响应
            result = response.json()
            response_text = result['choices'][0]['message']['content']
            
            # 解析JSON输出
            parsed = PromptTemplates.parse_classify_generator_json_output(response_text)
            if parsed:
                return ClassifyGenerateOutput(
                    brother_node=parsed['brother_overlap'].get('brother_node', []),
                    new_categories=parsed.get('new_categories', []),
                    reasoning={
                        'brother_overlap': parsed.get('brother_overlap', {}),
                        'parent_general_relevance': parsed.get('parent_general_relevance', {}),
                        'candidate_for_new_node': parsed.get('candidate_for_new_node', {})
                    }
                )
            else:
                raise ValueError("无法解析模型输出")
        
        except Exception as e:
            print(f"分类生成请求失败: {e}")
            import traceback
            traceback.print_exc()
            # 返回默认值
            return ClassifyGenerateOutput(
                brother_node=[],
                new_categories=[],
                reasoning={}
            )


class UpdaterClient:
    """总结更新系统HTTP客户端"""
    
    def __init__(self, service_url: str = "http://localhost:5001"):
        """
        Args:
            service_url: 总结更新系统服务URL
        """
        self.service_url = service_url.rstrip('/')
        self.mode = 'client'
        
        # 检查服务是否可用
        try:
            response = requests.get(f"{self.service_url}/health", timeout=5)
            if response.status_code == 200:
                print(f"✓ 总结更新系统服务连接成功: {self.service_url}")
            else:
                print(f"⚠ 总结更新系统服务响应异常: {response.status_code}")
        except Exception as e:
            print(f"✗ 无法连接到总结更新系统服务: {e}")
    
    def update(self, input_data: UpdateInput) -> UpdateOutput:
        """
        执行更新判断
        
        Args:
            input_data: 更新输入
            
        Returns:
            UpdateOutput对象
        """
        try:
            # 准备请求数据
            data = {
                'topic_name': input_data.topic_name,
                'node_summary': input_data.node_summary,
                'parent_summary': input_data.parent_summary,
                'sibling_summaries': input_data.sibling_summaries,
                'new_content': input_data.new_content
            }
            
            # 发送请求
            response = requests.post(
                f"{self.service_url}/update",
                json=data,
                timeout=60
            )
            response.raise_for_status()
            
            # 解析响应
            result = response.json()
            return UpdateOutput(
                needs_update=result['needs_update'],
                updated_summary=result.get('updated_summary'),
                reasoning=result.get('reasoning', {})
            )
        
        except Exception as e:
            print(f"更新请求失败: {e}")
            # 返回默认值（不更新）
            return UpdateOutput(
                needs_update=False,
                updated_summary=None,
                reasoning={}
            )


