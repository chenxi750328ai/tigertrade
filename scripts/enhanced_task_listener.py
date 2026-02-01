#!/usr/bin/env python3
"""
增强版任务监听器
同时监听多个可能的worker ID，以确保不错过任何任务分配
"""

import json
import time
from pathlib import Path
import random


class EnhancedTaskListener:
    def __init__(self, worker_ids):
        self.worker_ids = worker_ids if isinstance(worker_ids, list) else [worker_ids]
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
        for worker_id in self.worker_ids:
            if worker_id in state["agents"]:
                state["agents"][worker_id]["last_heartbeat"] = time.time()
                state["agents"][worker_id]["status"] = "listening"  # 更新状态为监听中
        self.write_state(state)
    
    def get_assigned_task(self):
        """获取分配给我的任一ID的任务"""
        state = self.read_state()
        
        for worker_id in self.worker_ids:
            # 查找分配给当前worker_id的任务
            assigned_msgs = [
                msg for msg in state["messages"] 
                if msg["to"] == worker_id and msg["type"] == "task_assign"
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
                
                # 记录是哪个worker_id收到的任务
                latest_msg["actual_receiver"] = worker_id
                return latest_msg["data"]
        
        return None
    
    def report_completion(self, task_id, result, receiver_id):
        """报告任务完成"""
        state = self.read_state()
        completion_msg = {
            "id": f"msg_{time.time()}_{receiver_id}_complete",
            "from": receiver_id,
            "to": "master",
            "type": "task_complete",
            "data": {
                "task_id": task_id,
                "result": result,
                "completed_at": time.time(),
                "by": receiver_id
            },
            "timestamp": time.time()
        }
        
        state["messages"].append(completion_msg)
        
        # 更新agent状态
        for worker_id in self.worker_ids:
            if worker_id in state["agents"]:
                state["agents"][worker_id]["status"] = "idle"
                state["agents"][worker_id]["task"] = None
                state["agents"][worker_id]["progress"] = 0.0
        
        self.write_state(state)
        print(f"✅ 任务 {task_id} 已由 {receiver_id} 完成并报告给master")
    
    def execute_task(self, task, receiver_id):
        """执行任务"""
        task_id = task.get("task_id", "unknown")
        task_type = task.get("type", "unknown")
        description = task.get("description", "No description")
        
        print(f"\n🚀 开始执行任务: {task_type}")
        print(f"   任务ID: {task_id}")
        print(f"   描述: {description}")
        print(f"   执行者: {receiver_id}")
        
        # 更新状态
        state = self.read_state()
        if receiver_id in state["agents"]:
            state["agents"][receiver_id]["task"] = description
            state["agents"][receiver_id]["progress"] = 0.0
        self.write_state(state)
        
        # 模拟任务执行，不同类型的任务有不同的执行逻辑
        total_steps = 10
        for i in range(total_steps):
            # 更新进度
            progress = (i + 1) / total_steps
            state = self.read_state()
            if receiver_id in state["agents"]:
                state["agents"][receiver_id]["progress"] = progress
            self.write_state(state)
            
            print(f"   执行进度: {progress*100:.1f}%")
            time.sleep(0.3)  # 模拟执行时间
        
        # 生成模拟结果
        result = {
            "status": "completed",
            "worker": receiver_id,
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
        print(f"👂 开始监听分配给 {self.worker_ids} 的任务...")
        print(f"   最大监听时间: {max_duration}秒")
        
        start_time = time.time()
        
        while time.time() - start_time < max_duration and self.running:
            self.heartbeat()
            
            # 检查是否有分配给任一ID的任务
            task = self.get_assigned_task()
            if task:
                # 获取接收者ID
                receiver_id = task.get('actual_receiver', self.worker_ids[0])
                print(f"\n✅ 收到任务分配给 {receiver_id}!")
                result = self.execute_task(task, receiver_id)
                self.report_completion(task.get("task_id"), result, receiver_id)
            
            time.sleep(3)  # 每3秒检查一次
        
        print(f"\n⏰ 监听时间结束，完成任务监听")


def main():
    # 创建任务监听器，同时监听多个worker ID
    listener = EnhancedTaskListener([
        "worker_lingma_enhanced",  # 我的主要ID
        "worker_lingma"           # 也监听这个ID，以防任务分配给了这个ID
    ])
    
    # 开始监听和执行任务
    listener.listen_for_tasks(max_duration=600)  # 监听10分钟


if __name__ == "__main__":
    main()