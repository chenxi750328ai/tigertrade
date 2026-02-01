"""
Master Agent - 任务分配和协调中心

职责：
1. 任务分解和分配
2. Worker注册和管理
3. 进度监控
4. 结果汇总
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Optional
from .coordinator import AgentCoordinator


class TaskQueue:
    """任务队列"""
    
    def __init__(self, queue_file="/tmp/tigertrade_task_queue.json"):
        self.queue_file = Path(queue_file)
        self._init_queue()
    
    def _init_queue(self):
        """初始化队列"""
        if not self.queue_file.exists():
            self.queue_file.write_text(json.dumps({
                "pending": [],
                "proposed": [],  # 新增：Worker提议的任务
                "assigned": {},
                "completed": [],
                "failed": []
            }, indent=2))
    
    def add_tasks(self, tasks: List[Dict], created_by="unknown"):
        """添加任务到队列（需要权限）"""
        data = json.loads(self.queue_file.read_text())
        
        # 权限检查和日志
        if created_by != "master":
            print(f"\n⚠️  警告: {created_by} 尝试创建任务！")
            print(f"   当前架构要求: 只有Master可以直接创建任务")
            print(f"   建议: Worker应该使用 propose_task() 提议任务\n")
        
        for task in tasks:
            if "task_id" not in task:
                task["task_id"] = f"task_{int(time.time()*1000)}_{len(data['pending'])}"
            
            # 记录创建者和时间
            task["created_by"] = created_by
            task["created_at"] = time.time()
            
            data["pending"].append(task)
        
        self.queue_file.write_text(json.dumps(data, indent=2))
        print(f"✅ {created_by} 创建了 {len(tasks)} 个任务")
    
    def get_next_task(self, worker_id: str) -> Optional[Dict]:
        """获取下一个待处理任务"""
        data = json.loads(self.queue_file.read_text())
        
        if not data["pending"]:
            return None
        
        task = data["pending"].pop(0)
        task["assigned_to"] = worker_id
        task["assigned_at"] = time.time()
        
        data["assigned"][task["task_id"]] = task
        self.queue_file.write_text(json.dumps(data, indent=2))
        
        return task
    
    def complete_task(self, task_id: str, result: Dict):
        """标记任务完成"""
        data = json.loads(self.queue_file.read_text())
        
        if task_id in data["assigned"]:
            task = data["assigned"].pop(task_id)
            task["result"] = result
            task["completed_at"] = time.time()
            data["completed"].append(task)
            self.queue_file.write_text(json.dumps(data, indent=2))
            return True
        return False
    
    def fail_task(self, task_id: str, error: str):
        """标记任务失败"""
        data = json.loads(self.queue_file.read_text())
        
        if task_id in data["assigned"]:
            task = data["assigned"].pop(task_id)
            task["error"] = error
            task["failed_at"] = time.time()
            data["failed"].append(task)
            self.queue_file.write_text(json.dumps(data, indent=2))
            return True
        return False
    
    def get_status(self) -> Dict:
        """获取队列状态"""
        data = json.loads(self.queue_file.read_text())
        return {
            "pending": len(data["pending"]),
            "assigned": len(data["assigned"]),
            "completed": len(data["completed"]),
            "failed": len(data["failed"])
        }
    
    def get_all_tasks(self) -> Dict:
        """获取所有任务详情"""
        return json.loads(self.queue_file.read_text())


class MasterAgent:
    """
    Master Agent - 任务协调中心
    
    功能：
    1. 任务分解：将大任务拆分成小任务
    2. 任务分配：分配给可用的Worker
    3. Worker管理：注册、心跳、状态监控
    4. 进度监控：实时监控所有任务进度
    5. 结果汇总：收集和汇总Worker的结果
    
    使用示例：
        master = MasterAgent()
        
        # 注册任务
        master.register_project("数据处理", [
            {"type": "download", "symbol": "SIL2603"},
            {"type": "clean", "file": "raw_data.csv"},
            {"type": "train", "model": "transformer"}
        ])
        
        # 运行Master
        master.run()
    """
    
    def __init__(self, master_id="master"):
        self.master_id = master_id
        self.coordinator = AgentCoordinator(master_id, "Master")
        self.task_queue = TaskQueue()
        self.workers = {}  # worker_id -> worker_info
        self.running = False
    
    def register_project(self, project_name: str, tasks: List[Dict]):
        """
        注册项目和任务
        
        Args:
            project_name: 项目名称
            tasks: 任务列表，每个任务包含 type, params 等
        """
        print(f"\n{'='*70}")
        print(f"📋 Master: 注册项目 '{project_name}'")
        print(f"{'='*70}")
        
        # 为任务添加元数据
        for i, task in enumerate(tasks):
            task["project"] = project_name
            task["task_index"] = i
            task["status"] = "pending"
        
        # 添加到任务队列
        self.task_queue.add_tasks(tasks)
        
        print(f"\n✅ 已注册 {len(tasks)} 个任务:")
        for task in tasks:
            print(f"   [{task.get('task_index', 0)}] {task['type']}: {task.get('description', 'N/A')}")
        
        status = self.task_queue.get_status()
        print(f"\n📊 任务队列状态:")
        print(f"   待分配: {status['pending']}")
        print(f"   执行中: {status['assigned']}")
        print(f"   已完成: {status['completed']}")
        print(f"   失败: {status['failed']}")
    
    def run(self, duration=60):
        """
        运行Master Agent
        
        Args:
            duration: 运行时长（秒）
        """
        print(f"\n{'='*70}")
        print(f"🚀 Master Agent 启动")
        print(f"{'='*70}")
        print(f"运行时长: {duration}秒")
        print(f"任务队列: /tmp/tigertrade_task_queue.json")
        print(f"Agent状态: /tmp/tigertrade_agent_state.json")
        print(f"{'='*70}\n")
        
        self.running = True
        start_time = time.time()
        last_status_print = 0
        
        try:
            while self.running and (time.time() - start_time < duration):
                current_time = time.time() - start_time
                
                # 1. 发现新Worker
                self._discover_workers()
                
                # 2. 分配任务给空闲Worker
                self._assign_tasks()
                
                # 3. 检查Worker心跳
                self._check_worker_health()
                
                # 4. 处理Worker消息
                self._process_messages()
                
                # 5. 定期打印状态（每5秒）
                if current_time - last_status_print >= 5:
                    self._print_status()
                    last_status_print = current_time
                
                # 6. Master心跳
                self.coordinator.heartbeat()
                
                # 检查是否所有任务完成
                status = self.task_queue.get_status()
                if status['pending'] == 0 and status['assigned'] == 0:
                    print(f"\n{'='*70}")
                    print(f"✅ 所有任务完成！")
                    print(f"{'='*70}")
                    break
                
                time.sleep(1)
        
        except KeyboardInterrupt:
            print(f"\n\n⏹️  Master收到停止信号")
        
        finally:
            self._print_final_report()
            self.coordinator.cleanup()
    
    def _discover_workers(self):
        """发现可用的Worker"""
        all_agents = self.coordinator.get_all_agents_status()
        
        for agent_id, agent_info in all_agents.items():
            # 跳过自己和非Worker
            if agent_id == self.master_id:
                continue
            
            role = agent_info.get('role', '')
            if 'worker' in role.lower():
                if agent_id not in self.workers:
                    self.workers[agent_id] = {
                        "registered_at": time.time(),
                        "role": role,
                        "status": "idle"
                    }
                    print(f"\n🤝 发现新Worker: {agent_id} ({role})")
    
    def _assign_tasks(self):
        """分配任务给空闲Worker"""
        for worker_id, worker_info in self.workers.items():
            # 检查Worker是否空闲
            agent_status = self.coordinator.get_all_agents_status().get(worker_id, {})
            if agent_status.get('status') == 'idle':
                # 获取下一个任务
                task = self.task_queue.get_next_task(worker_id)
                
                if task:
                    print(f"\n📤 Master → {worker_id}: 分配任务 '{task['type']}'")
                    
                    # 发送任务给Worker
                    self.coordinator.send_message(
                        worker_id,
                        "task_assign",
                        task
                    )
                    
                    worker_info['current_task'] = task['task_id']
                    worker_info['status'] = 'busy'
    
    def _check_worker_health(self):
        """检查Worker健康状态"""
        all_agents = self.coordinator.get_all_agents_status()
        current_time = time.time()
        
        for worker_id in list(self.workers.keys()):
            if worker_id not in all_agents:
                # Worker已离线
                print(f"\n⚠️  Worker {worker_id} 已离线")
                
                # 如果有未完成任务，重新放回队列
                worker_info = self.workers.pop(worker_id)
                if 'current_task' in worker_info:
                    task_id = worker_info['current_task']
                    print(f"   将任务 {task_id} 重新放回队列")
                    self.task_queue.fail_task(task_id, "Worker离线")
            else:
                agent_info = all_agents[worker_id]
                last_heartbeat = agent_info.get('last_heartbeat', 0)
                
                # 超过60秒无心跳
                if current_time - last_heartbeat > 60:
                    print(f"\n⚠️  Worker {worker_id} 心跳超时")
    
    def _process_messages(self):
        """处理Worker消息"""
        messages = self.coordinator.receive_messages()
        
        for msg in messages:
            msg_type = msg['type']
            from_worker = msg['from']
            data = msg['data']
            
            if msg_type == 'task_complete':
                # 任务完成
                task_id = data.get('task_id')
                result = data.get('result', {})
                
                print(f"\n✅ {from_worker}: 任务完成 '{task_id}'")
                print(f"   结果: {result}")
                
                self.task_queue.complete_task(task_id, result)
                
                if from_worker in self.workers:
                    self.workers[from_worker]['status'] = 'idle'
                    if 'current_task' in self.workers[from_worker]:
                        del self.workers[from_worker]['current_task']
            
            elif msg_type == 'task_failed':
                # 任务失败
                task_id = data.get('task_id')
                error = data.get('error', 'Unknown error')
                
                print(f"\n❌ {from_worker}: 任务失败 '{task_id}'")
                print(f"   错误: {error}")
                
                self.task_queue.fail_task(task_id, error)
                
                if from_worker in self.workers:
                    self.workers[from_worker]['status'] = 'idle'
            
            elif msg_type == 'worker_ready':
                # Worker准备就绪
                print(f"\n🟢 {from_worker}: 准备就绪")
                
                if from_worker in self.workers:
                    self.workers[from_worker]['status'] = 'idle'
            
            elif msg_type == 'request_help':
                # Worker请求帮助
                problem = data.get('problem', 'Unknown')
                print(f"\n🆘 {from_worker}: 请求帮助")
                print(f"   问题: {problem}")
                
                # 可以实现自动协商逻辑
                # 例如：分配更多资源、调整任务优先级等
    
    def _print_status(self):
        """打印当前状态"""
        print(f"\n{'─'*70}")
        print(f"📊 Master状态报告")
        print(f"{'─'*70}")
        
        # Worker状态
        print(f"\n👥 Workers ({len(self.workers)}):")
        for worker_id, worker_info in self.workers.items():
            status = worker_info.get('status', 'unknown')
            current_task = worker_info.get('current_task', 'N/A')
            print(f"   [{worker_id:15}] {status:8} | 任务: {current_task}")
        
        # 任务队列状态
        status = self.task_queue.get_status()
        print(f"\n📋 任务队列:")
        print(f"   待分配: {status['pending']}")
        print(f"   执行中: {status['assigned']}")
        print(f"   已完成: {status['completed']}")
        print(f"   失败: {status['failed']}")
        
        # 进度
        total = status['pending'] + status['assigned'] + status['completed'] + status['failed']
        if total > 0:
            progress = (status['completed'] + status['failed']) / total * 100
            print(f"\n📈 总进度: {progress:.1f}% ({status['completed'] + status['failed']}/{total})")
    
    def _print_final_report(self):
        """打印最终报告"""
        print(f"\n{'='*70}")
        print(f"📊 Master最终报告")
        print(f"{'='*70}")
        
        all_tasks = self.task_queue.get_all_tasks()
        
        print(f"\n✅ 已完成任务 ({len(all_tasks['completed'])}):")
        for task in all_tasks['completed']:
            duration = task.get('completed_at', 0) - task.get('assigned_at', 0)
            worker = task.get('assigned_to', 'Unknown')
            print(f"   [{task['task_id']}] {task['type']}")
            print(f"      Worker: {worker}")
            print(f"      耗时: {duration:.1f}秒")
            if 'result' in task:
                print(f"      结果: {task['result']}")
        
        if all_tasks['failed']:
            print(f"\n❌ 失败任务 ({len(all_tasks['failed'])}):")
            for task in all_tasks['failed']:
                print(f"   [{task['task_id']}] {task['type']}")
                print(f"      错误: {task.get('error', 'Unknown')}")
        
        if all_tasks['pending']:
            print(f"\n⏳ 未完成任务 ({len(all_tasks['pending'])}):")
            for task in all_tasks['pending']:
                print(f"   [{task['task_id']}] {task['type']}")
        
        print(f"\n{'='*70}\n")
