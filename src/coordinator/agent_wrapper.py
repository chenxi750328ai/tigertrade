"""
Agent包装器
提供简洁的协调接口
"""

from typing import Callable, Any
from .coordinator import AgentCoordinator
import time


class CoordinatedAgent:
    """
    协调的Agent包装器
    
    自动处理：
    - 心跳
    - 锁管理
    - 错误恢复
    
    使用示例：
        agent = CoordinatedAgent("agent1", "数据工程师")
        
        @agent.task("数据预处理")
        def preprocess_data():
            # 自动获取锁，自动更新状态
            return process()
        
        agent.run()
    """
    
    def __init__(self, agent_id: str, role: str):
        self.coordinator = AgentCoordinator(agent_id, role)
        self.tasks = []
        self._running = False
    
    def task(self, task_name: str, resources: list = None):
        """
        任务装饰器
        
        Args:
            task_name: 任务名称
            resources: 需要锁定的资源列表
        
        使用示例:
            @agent.task("训练模型", resources=["gpu", "train.csv"])
            def train():
                # 任务代码
                pass
        """
        def decorator(func: Callable):
            def wrapper(*args, **kwargs):
                # 更新状态
                self.coordinator.update_status("working", task_name, 0.0)
                
                # 获取资源锁
                if resources:
                    for resource in resources:
                        print(f"  🔒 获取锁: {resource}")
                        if not self.coordinator.acquire_lock(resource):
                            print(f"  ❌ 无法获取锁: {resource}")
                            self.coordinator.update_status("error", task_name, 0.0)
                            return None
                
                try:
                    # 执行任务
                    result = func(*args, **kwargs)
                    
                    # 完成
                    self.coordinator.update_status("idle", task_name, 1.0)
                    return result
                    
                except Exception as e:
                    print(f"  ❌ 任务失败: {e}")
                    self.coordinator.update_status("error", task_name, 0.0)
                    raise
                    
                finally:
                    # 释放锁
                    if resources:
                        for resource in resources:
                            print(f"  🔓 释放锁: {resource}")
                            self.coordinator.release_lock(resource)
            
            self.tasks.append((task_name, wrapper))
            return wrapper
        return decorator
    
    def run(self):
        """运行所有任务"""
        self._running = True
        
        try:
            for task_name, task_func in self.tasks:
                if not self._running:
                    break
                
                print(f"\n{'='*60}")
                print(f"📋 任务: {task_name}")
                print(f"{'='*60}")
                
                task_func()
                
                # 心跳
                self.coordinator.heartbeat()
        
        finally:
            self.coordinator.cleanup()
    
    def stop(self):
        """停止运行"""
        self._running = False
