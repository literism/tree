"""
实时状态监控
覆盖式输出，显示prompt池和结果池的状态
"""
import threading
import time
import sys
from typing import Dict, Any


class StatusMonitor:
    """状态监控器"""
    
    def __init__(self):
        self.stats = {
            'classifier_prompt_pool_size': 0,
            'classifier_result_pool_size': 0,
            'updater_prompt_pool_size': 0,
            'updater_result_pool_size': 0,
            'classifier_worker_status': 'idle',
            'updater_worker_status': 'idle',
            'total_articles_processed': 0,
            'last_update_time': time.time()
        }
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
    
    def update(self, key: str, value: Any):
        """更新状态"""
        with self.lock:
            self.stats[key] = value
            self.stats['last_update_time'] = time.time()
    
    def get_stats(self) -> Dict:
        """获取状态快照"""
        with self.lock:
            return self.stats.copy()
    
    def start(self):
        """启动监控线程"""
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """停止监控线程"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
    
    def _monitor_loop(self):
        """监控主循环"""
        print("\n" + "=" * 80)
        print("系统状态监控（实时更新）")
        print("=" * 80 + "\n")
        
        # 保留几行用于显示
        lines_to_show = 12
        
        while self.running:
            stats = self.get_stats()
            
            # 构建显示内容
            lines = [
                "┌─────────────────────────────────────────────────────────────────────────────┐",
                "│                          Prompt池 & 结果池状态                               │",
                "├─────────────────────────────────────────────────────────────────────────────┤",
                f"│ 分类系统 Prompt池: {stats['classifier_prompt_pool_size']:>3} 个待处理  │  结果池: {stats['classifier_result_pool_size']:>3} 个待取   │",
                f"│ 总结系统 Prompt池: {stats['updater_prompt_pool_size']:>3} 个待处理  │  结果池: {stats['updater_result_pool_size']:>3} 个待取   │",
                "├─────────────────────────────────────────────────────────────────────────────┤",
                "│                              Worker状态                                     │",
                "├─────────────────────────────────────────────────────────────────────────────┤",
                f"│ 分类Worker: {stats['classifier_worker_status']:>10}                                        │",
                f"│ 总结Worker: {stats['updater_worker_status']:>10}                                        │",
                "├─────────────────────────────────────────────────────────────────────────────┤",
                f"│ 已处理文章数: {stats['total_articles_processed']:>4}                                               │",
                "└─────────────────────────────────────────────────────────────────────────────┘",
            ]
            
            # 移动光标到开始位置并清空
            sys.stdout.write('\033[' + str(lines_to_show) + 'A')  # 向上移动N行
            
            # 输出所有行
            for line in lines:
                # 清空当前行并输出
                sys.stdout.write('\033[K' + line + '\n')
            
            sys.stdout.flush()
            
            # 等待一小段时间
            time.sleep(0.5)
        
        # 停止时输出最终状态
        print("\n监控已停止")


# 全局监控实例
_global_monitor = None


def get_monitor() -> StatusMonitor:
    """获取全局监控实例"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = StatusMonitor()
    return _global_monitor

