"""
轨迹数据存储和加载
用于在DPO训练的采样阶段和标注阶段之间持久化数据
"""

import pickle
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
import gzip


@dataclass
class StoredDecisionPoint:
    """存储的决策点（用于标注）"""
    # 标识信息
    topic_key: str
    article_id: str
    decision_index: int  # 在该文章轨迹中的第几个决策点
    
    # 决策类型
    decision_type: str  # 'classify', 'generate_new', 'update'
    
    # 用于标注的信息
    topic_name: str
    current_summary: str  # 当前节点的summary
    child_summaries: List[str]  # 子节点summaries（用于分类系统）
    parent_summary: str  # 父节点summary（用于更新系统）
    sibling_summaries: List[str]  # 兄弟节点summaries（用于更新系统）
    ground_truth_paths: List[str]  # 真实路径
    
    # 用于构建数据集的信息
    prompt: str
    all_outputs: List[str]  # 所有采样输出的原始文本
    chosen_index: int  # 主线选择的索引


@dataclass
class StoredTrajectory:
    """存储的轨迹数据"""
    # 标识信息
    topic_key: str
    article_id: str
    trajectory_index: int  # 第几条轨迹
    
    # 轨迹信息
    actions: List[Dict[str, Any]]  # Action对象序列化后的字典列表
    
    # 决策点信息（用于标注）
    decision_points: List[StoredDecisionPoint]
    
    # 文章内容
    article_content: str
    
    # 其他元数据
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IterationTrajectoryData:
    """一个iteration的所有轨迹数据"""
    iteration: int
    trajectories: List[StoredTrajectory]
    
    # 统计信息
    num_topics: int = 0
    num_articles: int = 0
    num_trajectories: int = 0
    num_decision_points: int = 0


class TrajectoryStorage:
    """轨迹数据存储管理器"""
    
    def __init__(self, output_dir: str):
        """
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.trajectories_dir = self.output_dir / 'train_trajectories'
        self.trajectories_dir.mkdir(parents=True, exist_ok=True)
    
    def save_iteration_data(
        self,
        iteration: int,
        trajectories: List[StoredTrajectory],
        compress: bool = True
    ) -> str:
        """
        保存一个iteration的轨迹数据
        
        Args:
            iteration: iteration编号
            trajectories: 轨迹列表
            compress: 是否压缩（使用gzip）
        
        Returns:
            保存的文件路径
        """
        # 统计信息
        num_topics = len(set(t.topic_key for t in trajectories))
        num_articles = len(set((t.topic_key, t.article_id) for t in trajectories))
        num_trajectories = len(trajectories)
        num_decision_points = sum(len(t.decision_points) for t in trajectories)
        
        data = IterationTrajectoryData(
            iteration=iteration,
            trajectories=trajectories,
            num_topics=num_topics,
            num_articles=num_articles,
            num_trajectories=num_trajectories,
            num_decision_points=num_decision_points
        )
        
        # 保存路径
        if compress:
            filepath = self.trajectories_dir / f'iteration_{iteration}_trajectories.pkl.gz'
        else:
            filepath = self.trajectories_dir / f'iteration_{iteration}_trajectories.pkl'
        
        print(f"\n保存轨迹数据到: {filepath}")
        print(f"  - Topics: {num_topics}")
        print(f"  - Articles: {num_articles}")
        print(f"  - Trajectories: {num_trajectories}")
        print(f"  - Decision Points: {num_decision_points}")
        
        # 保存
        if compress:
            with gzip.open(filepath, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"✓ 轨迹数据已保存")
        return str(filepath)
    
    def load_iteration_data(self, iteration: int) -> Optional[IterationTrajectoryData]:
        """
        加载一个iteration的轨迹数据
        
        Args:
            iteration: iteration编号
        
        Returns:
            轨迹数据，如果文件不存在则返回None
        """
        # 尝试加载压缩文件
        filepath_gz = self.trajectories_dir / f'iteration_{iteration}_trajectories.pkl.gz'
        filepath = self.trajectories_dir / f'iteration_{iteration}_trajectories.pkl'
        
        if filepath_gz.exists():
            filepath = filepath_gz
        elif not filepath.exists():
            return None
        
        print(f"\n加载轨迹数据从: {filepath}")
        
        try:
            if str(filepath).endswith('.gz'):
                with gzip.open(filepath, 'rb') as f:
                    data = pickle.load(f)
            else:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
            
            print(f"✓ 轨迹数据已加载")
            print(f"  - Topics: {data.num_topics}")
            print(f"  - Articles: {data.num_articles}")
            print(f"  - Trajectories: {data.num_trajectories}")
            print(f"  - Decision Points: {data.num_decision_points}")
            
            return data
        
        except Exception as e:
            print(f"✗ 加载失败: {e}")
            return None
    
    def extract_decision_points_for_labeling(
        self,
        data: IterationTrajectoryData
    ) -> List[StoredDecisionPoint]:
        """
        从轨迹数据中提取所有需要标注的决策点
        
        Args:
            data: 轨迹数据
        
        Returns:
            决策点列表（展平的，用于批量标注）
        """
        all_decision_points = []
        
        for trajectory in data.trajectories:
            for dp in trajectory.decision_points:
                all_decision_points.append(dp)
        
        return all_decision_points
    
    def save_metadata(self, iteration: int, metadata: Dict[str, Any]):
        """
        保存额外的元数据（如配置、统计信息等）
        
        Args:
            iteration: iteration编号
            metadata: 元数据字典
        """
        filepath = self.trajectories_dir / f'iteration_{iteration}_metadata.json'
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 元数据已保存到: {filepath}")
    
    def load_metadata(self, iteration: int) -> Optional[Dict[str, Any]]:
        """
        加载元数据
        
        Args:
            iteration: iteration编号
        
        Returns:
            元数据字典，如果文件不存在则返回None
        """
        filepath = self.trajectories_dir / f'iteration_{iteration}_metadata.json'
        
        if not filepath.exists():
            return None
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)


def test_storage():
    """测试存储功能"""
    import tempfile
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = TrajectoryStorage(tmpdir)
        
        # 创建测试数据
        dp1 = StoredDecisionPoint(
            topic_key="topic1",
            article_id="article1",
            decision_index=0,
            decision_type="classify",
            topic_name="Biology",
            current_summary="Study of life",
            child_summaries=["Plants", "Animals"],
            parent_summary="",
            sibling_summaries=[],
            ground_truth_paths=["Biology - Plants - Trees"],
            prompt="Test prompt",
            all_outputs=["Output 1", "Output 2"],
            chosen_index=0
        )
        
        trajectory1 = StoredTrajectory(
            topic_key="topic1",
            article_id="article1",
            trajectory_index=0,
            actions=[{"action_type": "classify", "system": "classify_generator"}],
            decision_points=[dp1],
            article_content="Test article content",
            metadata={"test": True}
        )
        
        # 保存
        filepath = storage.save_iteration_data(1, [trajectory1], compress=True)
        print(f"\n保存成功: {filepath}")
        
        # 加载
        data = storage.load_iteration_data(1)
        assert data is not None
        assert data.num_trajectories == 1
        assert data.num_decision_points == 1
        
        # 提取决策点
        decision_points = storage.extract_decision_points_for_labeling(data)
        assert len(decision_points) == 1
        assert decision_points[0].topic_key == "topic1"
        
        print("\n✓ 所有测试通过")


if __name__ == '__main__':
    test_storage()
