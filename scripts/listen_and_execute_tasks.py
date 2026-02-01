#!/usr/bin/env python3
"""
监听并执行分配给我的任务
这个脚本会监听多AGENT系统中分配给我的任务，并执行它们
"""

import json
import time
from pathlib import Path
import random


class TaskExecutor:
    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.state_file = Path("/tmp/tigertrade_agent_state.json")
        self.running = True
        
    def read_state(self):
        """读取状态"""
        return json.loads(self.state_file.read_text())
    
    def write_state(self, state):
        """写入状态"""
        self.state_file.write_text(json.dumps(state, indent=2))
    
    def heartbeat(self):
        """心跳"""
        state = self.read_state()
        if self.worker_id in state["agents"]:
            state["agents"][self.worker_id]["last_heartbeat"] = time.time()
            state["agents"][self.worker_id]["status"] = "working"  # 更新状态为工作中
            self.write_state(state)
    
    def get_assigned_task(self):
        """获取分配给我的任务"""
        state = self.read_state()
        # 查找分配给我的任务
        assigned_msgs = [
            msg for msg in state["messages"] 
            if msg["to"] == self.worker_id and msg["type"] == "task_assign"
        ]
        
        if assigned_msgs:
            # 获取最新的任务
            latest_msg = assigned_msgs[-1]
            # 从消息列表中移除这个任务
            state["messages"] = [
                msg for msg in state["messages"] 
                if msg["id"] != latest_msg["id"]
            ]
            self.write_state(state)
            return latest_msg["data"]
        
        return None
    
    def report_completion(self, task_id, result):
        """报告任务完成"""
        state = self.read_state()
        completion_msg = {
            "id": f"msg_{time.time()}_{self.worker_id}_complete",
            "from": self.worker_id,
            "to": "master",
            "type": "task_complete",
            "data": {
                "task_id": task_id,
                "result": result,
                "completed_at": time.time()
            },
            "timestamp": time.time()
        }
        
        state["messages"].append(completion_msg)
        
        # 更新agent状态
        if self.worker_id in state["agents"]:
            state["agents"][self.worker_id]["status"] = "idle"
            state["agents"][self.worker_id]["task"] = None
            state["agents"][self.worker_id]["progress"] = 0.0
        
        self.write_state(state)
        print(f"✅ 任务 {task_id} 已完成并报告给master")
    
    def execute_task(self, task):
        """执行任务"""
        task_id = task.get("task_id", "unknown")
        task_type = task.get("type", "unknown")
        description = task.get("description", "No description")
        
        print(f"\n🚀 开始执行任务: {task_type}")
        print(f"   任务ID: {task_id}")
        print(f"   描述: {description}")
        
        # 更新状态
        state = self.read_state()
        if self.worker_id in state["agents"]:
            state["agents"][self.worker_id]["task"] = description
            state["agents"][self.worker_id]["progress"] = 0.0
        self.write_state(state)
        
        # 模拟任务执行，不同类型的任务有不同的执行逻辑
        total_steps = 10
        for i in range(total_steps):
            # 更新进度
            progress = (i + 1) / total_steps
            state = self.read_state()
            if self.worker_id in state["agents"]:
                state["agents"][self.worker_id]["progress"] = progress
            self.write_state(state)
            
            print(f"   执行进度: {progress*100:.1f}%")
            time.sleep(0.5)  # 模拟执行时间
        
        # 生成模拟结果
        result = {
            "status": "completed",
            "worker": self.worker_id,
            "task_id": task_id,
            "task_type": task_type,
            "execution_time": time.time(),
            "details": f"成功完成{task_type}任务: {description}",
            "random_factor": random.random()  # 添加随机因素以展示每次执行略有不同
        }
        
        print(f"✅ 任务 {task_id} 执行完成")
        return result
    
    def listen_for_tasks(self, max_duration=600):
        """监听任务并执行"""
        print(f"👂 开始监听分配给 {self.worker_id} 的任务...")
        print(f"   最大监听时间: {max_duration}秒")
        
        start_time = time.time()
        
        while time.time() - start_time < max_duration and self.running:
            self.heartbeat()
            
            # 检查是否有分配给我的任务
            task = self.get_assigned_task()
            if task:
                print(f"\n✅ 收到任务分配!")
                result = self.execute_task(task)
                self.report_completion(task.get("task_id"), result)
            
            time.sleep(2)  # 每2秒检查一次
        
        print(f"\n⏰ 监听时间结束，{self.worker_id} 完成任务监听")


def main():
    # 创建任务执行器
    executor = TaskExecutor("worker_lingma_enhanced")
    
    # 开始监听和执行任务
    executor.listen_for_tasks(max_duration=300)  # 监听5分钟


if __name__ == "__main__":
    main()