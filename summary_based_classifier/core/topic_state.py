"""
Topic状态管理
管理每个topic的树状态、文章处理进度等
"""
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict
from summary_based_classifier.core.trajectory.trajectory_sampler import TreeNode


@dataclass
class TopicState:
    """Topic状态"""
    topic_key: str
    topic_name: str
    current_tree: TreeNode
    article_order: List[str]  # 打乱后的文章ID顺序
    next_article_idx: int = 0  # 下一篇该处理的索引
    articles_processed: int = 0  # 已处理文章数
    
    def has_next_article(self) -> bool:
        """是否还有文章待处理"""
        return self.next_article_idx < len(self.article_order)
    
    def get_next_article_id(self) -> Optional[str]:
        """获取下一篇文章ID"""
        if not self.has_next_article():
            return None
        article_id = self.article_order[self.next_article_idx]
        self.next_article_idx += 1
        self.articles_processed += 1
        return article_id
    
    def reset_tree(self):
        """重置树（当所有文章处理完后）"""
        self.current_tree = TreeNode(
            summary=self.topic_name,
            citations=[],
            children=[],
            depth=0
        )
    
    def shuffle_articles(self, article_ids: List[str]):
        """打乱文章顺序并重置索引"""
        import random
        self.article_order = article_ids.copy()
        random.shuffle(self.article_order)
        self.next_article_idx = 0
    
    def save(self, save_dir: Path):
        """保存状态"""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存树（使用pickle）
        tree_file = save_dir / f"{self.topic_key}_tree.pkl"
        with open(tree_file, 'wb') as f:
            pickle.dump(self.current_tree, f)
        
        # 保存其他状态（使用JSON）
        state_file = save_dir / f"{self.topic_key}_state.json"
        state_data = {
            'topic_key': self.topic_key,
            'topic_name': self.topic_name,
            'article_order': self.article_order,
            'next_article_idx': self.next_article_idx,
            'articles_processed': self.articles_processed
        }
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def load(topic_key: str, save_dir: Path) -> Optional['TopicState']:
        """加载状态"""
        tree_file = save_dir / f"{topic_key}_tree.pkl"
        state_file = save_dir / f"{topic_key}_state.json"
        
        if not tree_file.exists() or not state_file.exists():
            return None
        
        # 加载树
        with open(tree_file, 'rb') as f:
            tree = pickle.load(f)
        
        # 加载状态
        with open(state_file, 'r', encoding='utf-8') as f:
            state_data = json.load(f)
        
        return TopicState(
            topic_key=state_data['topic_key'],
            topic_name=state_data['topic_name'],
            current_tree=tree,
            article_order=state_data['article_order'],
            next_article_idx=state_data['next_article_idx'],
            articles_processed=state_data['articles_processed']
        )


class TopicStateManager:
    """管理所有topic的状态"""
    
    def __init__(self, save_dir: Path):
        self.save_dir = save_dir
        self.states: Dict[str, TopicState] = {}
    
    def initialize_topic(
        self,
        topic_key: str,
        topic_name: str,
        article_ids: List[str],
        load_if_exists: bool = True
    ) -> TopicState:
        """初始化或加载topic状态"""
        # 尝试加载已有状态
        if load_if_exists:
            state = TopicState.load(topic_key, self.save_dir)
            if state is not None:
                self.states[topic_key] = state
                return state
        
        # 创建新状态
        state = TopicState(
            topic_key=topic_key,
            topic_name=topic_name,
            current_tree=TreeNode(
                summary=topic_name,
                citations=[],
                children=[],
                depth=0
            ),
            article_order=[]
        )
        state.shuffle_articles(article_ids)
        self.states[topic_key] = state
        return state
    
    def get_state(self, topic_key: str) -> Optional[TopicState]:
        """获取topic状态"""
        return self.states.get(topic_key)
    
    def save_all(self):
        """保存所有状态"""
        for state in self.states.values():
            state.save(self.save_dir)
    
    def get_topics_with_articles(self) -> List[str]:
        """获取还有文章待处理的topic列表"""
        return [
            topic_key for topic_key, state in self.states.items()
            if state.has_next_article()
        ]
    
    def reset_exhausted_topics(self, references_data: Dict):
        """重置已处理完所有文章的topic"""
        for topic_key, state in self.states.items():
            if not state.has_next_article():
                # 重置树
                state.reset_tree()
                
                # 重新打乱文章
                if topic_key in references_data:
                    article_ids = list(references_data[topic_key].get('references', {}).keys())
                    state.shuffle_articles(article_ids)
                
                print(f"  Topic {topic_key} 已处理完所有文章，重置树并重新打乱")

