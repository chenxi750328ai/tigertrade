#!/usr/bin/env python3
"""
增强版Worker Agent - 支持协议v2.1.0
具备讨论、提议、知识共享等功能
"""

import json
import time
from pathlib import Path


class EnhancedWorkerAgent:
    """增强版Worker Agent"""
    
    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.state_file = Path("/tmp/tigertrade_agent_state.json")
        self.task_queue_file = Path("/tmp/tigertrade_task_queue.json")
        self.protocol_proposals_file = Path("/tmp/tigertrade_protocol_proposals.json")
        self.init_agent()
    
    def init_agent(self):
        """初始化Agent"""
        # 确保状态文件存在
        if not self.state_file.exists():
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self.state_file.write_text(json.dumps({
                "agents": {}, 
                "resources": {}, 
                "messages": [],
                "protocol_version": "2.1.0",
                "election_status": {
                    "current_master": "master",
                    "candidates": [],
                    "votes": {}
                }
            }))
        
        # 注册到系统
        state = self.read_state()
        state["agents"][self.worker_id] = {
            "role": "Enhanced Worker", 
            "status": "online", 
            "task": None, 
            "progress": 0,
            "locked_resources": [], 
            "registered_at": time.time(), 
            "last_heartbeat": time.time()
        }
        
        # 发送上线通知
        state["messages"].append({
            "id": f"msg_{time.time()}_{self.worker_id}",
            "from": self.worker_id,
            "to": "all",
            "type": "agent_online",
            "data": {"msg": f"{self.worker_id} 上线"},
            "timestamp": time.time()
        })
        
        self.write_state(state)
        print(f"✅ {self.worker_id} 已注册并上线")
    
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
            self.write_state(state)
    
    def get_task(self):
        """获取任务"""
        state = self.read_state()
        # 查找分配给自己的任务
        msgs = [m for m in state["messages"] 
                if m["to"] == self.worker_id and m["type"] == "task_assign"]
        if msgs:
            task = msgs[-1]["data"]
            # 删除已获取的消息
            state["messages"] = [m for m in state["messages"] if m["id"] != msgs[-1]["id"]]
            self.write_state(state)
            return task
        return None
    
    def propose_task(self, task_type, description, reason):
        """提议任务"""
        state = self.read_state()
        
        state["messages"].append({
            "id": f"msg_{time.time()}_{self.worker_id}_proposal",
            "from": self.worker_id,
            "to": "master",
            "type": "task_proposal",
            "data": {
                "type": task_type,
                "description": description,
                "reason": reason
            },
            "timestamp": time.time()
        })
        
        self.write_state(state)
        print(f"📝 任务提议已发送: {task_type}")
    
    def complete_task(self, task_id, result):
        """完成任务"""
        state = self.read_state()
        state["messages"].append({
            "id": f"msg_{time.time()}_{self.worker_id}_complete",
            "from": self.worker_id,
            "to": "master",
            "type": "task_complete", 
            "data": {"task_id": task_id, "result": result},
            "timestamp": time.time()
        })
        state["agents"][self.worker_id]["status"] = "idle"
        self.write_state(state)
        print(f"✅ 任务完成: {task_id}")
    
    def start_discussion(self, topic, question):
        """发起讨论"""
        state = self.read_state()
        
        state["messages"].append({
            "id": f"msg_{time.time()}_{self.worker_id}_discuss",
            "from": self.worker_id,
            "to": "all",
            "type": "discussion",
            "data": {
                "topic": topic,
                "question": question
            },
            "timestamp": time.time()
        })
        
        self.write_state(state)
        print(f"💬 讨论已发起: {topic}")
    
    def share_knowledge(self, title, content):
        """分享知识到RAG"""
        state = self.read_state()
        
        state["messages"].append({
            "id": f"msg_{time.time()}_{self.worker_id}_knowledge",
            "from": self.worker_id,
            "to": "all",
            "type": "knowledge_share",
            "data": {
                "title": title,
                "content": content,
                "timestamp": time.time()
            },
            "timestamp": time.time()
        })
        
        self.write_state(state)
        print(f"📚 知识已分享: {title}")
    
    def participate_in_election(self):
        """参与选举成为MASTER"""
        state = self.read_state()
        
        # 添加自己到候选人列表
        if "election_status" not in state:
            state["election_status"] = {
                "current_master": "master",
                "candidates": [],
                "votes": {}
            }
        
        if self.worker_id not in state["election_status"]["candidates"]:
            state["election_status"]["candidates"].append(self.worker_id)
        
        # 发送参选消息
        state["messages"].append({
            "id": f"msg_{time.time()}_{self.worker_id}_election",
            "from": self.worker_id,
            "to": "all",
            "type": "election_candidate",
            "data": {
                "candidate": self.worker_id,
                "platform": "我将致力于优化多AGENT协作效率，推进项目达成月盈利率20%的目标！"
            },
            "timestamp": time.time()
        })
        
        self.write_state(state)
        print(f"🗳️  {self.worker_id} 已参选MASTER")
    
    def run(self, duration=600):  # 默认运行10分钟
        """运行Worker"""
        print(f"\n{'='*60}")
        print(f"🚀 增强版Worker {self.worker_id} 启动")
        print(f"   功能: 任务执行、讨论、提议、知识分享、参选MASTER")
        print(f"{'='*60}\n")
        
        # 参与MASTER选举
        self.participate_in_election()
        
        # 分享一个知识
        self.share_knowledge(
            "RAG系统使用最佳实践", 
            "在执行任何任务前，必须查询RAG系统获取相关约束和规范，确保遵循项目架构和文件组织规则。"
        )
        
        # 发起一次讨论
        self.start_discussion(
            "数据预处理策略优化",
            "大家认为对于时间序列数据，使用滑动窗口特征提取还是傅里叶变换更能捕捉趋势？"
        )
        
        start_time = time.time()
        while time.time() - start_time < duration:
            self.heartbeat()
            
            # 检查是否有任务
            task = self.get_task()
            if task:
                print(f"\n🎯 收到任务: {task['type']} - {task.get('description', 'N/A')}")
                self.execute_task(task)
            
            print(".", end="", flush=True)
            time.sleep(1)
        
        print(f"\n\n✅ Worker {self.worker_id} 运行结束")
    
    def execute_task(self, task):
        """执行任务"""
        print(f"\n🔨 执行任务: {task['type']}")
        print(f"   描述: {task.get('description', 'N/A')}")
        
        # 模拟任务执行
        for i in range(5):
            time.sleep(0.5)
            progress = (i + 1) / 5
            print(f"   进度: {progress*100:.0f}%")
        
        result = {
            "status": "success", 
            "worker": self.worker_id, 
            "task_id": task['task_id'],
            "summary": f"成功完成{task['type']}任务"
        }
        
        self.complete_task(task['task_id'], result)


def main():
    """主函数"""
    # 创建增强版Worker Agent
    worker = EnhancedWorkerAgent("worker_lingma_enhanced")
    
    # 运行指定时间
    worker.run(duration=300)  # 运行5分钟


if __name__ == "__main__":
    main()