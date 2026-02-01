"""
Agent协调器核心实现
基于文件锁的消息队列和资源管理
"""

import json
import fcntl
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


class AgentCoordinator:
    """
    多Agent协调器
    
    功能：
    - 资源锁管理（互斥）
    - 消息队列（通信）
    - 状态同步
    - 心跳检测
    
    使用示例：
        coordinator = AgentCoordinator("agent1")
        
        # 获取锁
        if coordinator.acquire_lock("train.csv"):
            try:
                # 处理数据
                process_data()
            finally:
                coordinator.release_lock("train.csv")
        
        # 发送消息
        coordinator.send_message("agent2", "task_complete", {"task": "preprocessing"})
        
        # 接收消息
        messages = coordinator.receive_messages()
    """
    
    STATE_FILE = Path("/tmp/tigertrade_agent_state.json")
    HEARTBEAT_TIMEOUT = 60  # 心跳超时（秒）
    
    def __init__(self, agent_id: str, role: str = "worker"):
        """
        初始化协调器
        
        Args:
            agent_id: Agent唯一标识
            role: Agent角色（用于显示）
        """
        self.agent_id = agent_id
        self.role = role
        self._init_state()
        self._register_agent()
    
    def _init_state(self):
        """初始化状态文件"""
        if not self.STATE_FILE.exists():
            self.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            self.STATE_FILE.write_text(json.dumps({
                "agents": {},
                "resources": {},
                "messages": [],
                "created_at": time.time()
            }, indent=2))
    
    def _register_agent(self):
        """注册Agent"""
        with self._file_lock():
            state = self._read_state()
            state["agents"][self.agent_id] = {
                "role": self.role,
                "status": "idle",
                "task": None,
                "progress": 0.0,
                "locked_resources": [],
                "registered_at": time.time(),
                "last_heartbeat": time.time()
            }
            self._write_state(state)
    
    # ==================== 资源锁管理 ====================
    
    def acquire_lock(self, resource: str, timeout: float = 30.0) -> bool:
        """
        获取资源锁（阻塞等待）
        
        Args:
            resource: 资源名称（如 "train.csv", "gpu", "model_file"）
            timeout: 超时时间（秒）
        
        Returns:
            bool: 是否成功获取锁
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self._file_lock():
                state = self._read_state()
                
                # 清理过期锁（死锁恢复）
                self._cleanup_stale_locks(state)
                
                # 检查资源是否被锁定
                if resource in state["resources"]:
                    locked_by = state["resources"][resource].get("locked_by")
                    if locked_by and locked_by != self.agent_id:
                        # 资源被占用
                        pass
                    else:
                        # 资源可用或被自己锁定
                        state["resources"][resource] = {
                            "locked_by": self.agent_id,
                            "locked_at": time.time()
                        }
                        # 更新Agent的锁列表
                        if resource not in state["agents"][self.agent_id]["locked_resources"]:
                            state["agents"][self.agent_id]["locked_resources"].append(resource)
                        self._write_state(state)
                        return True
                else:
                    # 资源未被锁定
                    state["resources"][resource] = {
                        "locked_by": self.agent_id,
                        "locked_at": time.time()
                    }
                    if resource not in state["agents"][self.agent_id]["locked_resources"]:
                        state["agents"][self.agent_id]["locked_resources"].append(resource)
                    self._write_state(state)
                    return True
            
            # 等待一会再试
            time.sleep(0.5)
        
        return False
    
    def release_lock(self, resource: str):
        """
        释放资源锁
        
        Args:
            resource: 资源名称
        """
        with self._file_lock():
            state = self._read_state()
            
            # 删除资源锁
            if resource in state["resources"]:
                locked_by = state["resources"][resource].get("locked_by")
                if locked_by == self.agent_id:
                    del state["resources"][resource]
            
            # 更新Agent的锁列表
            if self.agent_id in state["agents"]:
                locked_res = state["agents"][self.agent_id]["locked_resources"]
                if resource in locked_res:
                    locked_res.remove(resource)
            
            self._write_state(state)
    
    def release_all_locks(self):
        """释放所有锁"""
        with self._file_lock():
            state = self._read_state()
            
            if self.agent_id not in state["agents"]:
                return
            
            # 获取所有锁定的资源
            locked_resources = state["agents"][self.agent_id]["locked_resources"].copy()
            
            # 释放每个资源
            for resource in locked_resources:
                if resource in state["resources"]:
                    locked_by = state["resources"][resource].get("locked_by")
                    if locked_by == self.agent_id:
                        del state["resources"][resource]
            
            # 清空锁列表
            state["agents"][self.agent_id]["locked_resources"] = []
            self._write_state(state)
    
    def _cleanup_stale_locks(self, state: Dict):
        """清理过期锁（死锁恢复）"""
        current_time = time.time()
        stale_resources = []
        
        for resource, lock_info in state["resources"].items():
            locked_by = lock_info.get("locked_by")
            locked_at = lock_info.get("locked_at", 0)
            
            # 检查锁定的Agent是否还活着
            if locked_by in state["agents"]:
                last_heartbeat = state["agents"][locked_by].get("last_heartbeat", 0)
                if current_time - last_heartbeat > self.HEARTBEAT_TIMEOUT:
                    # Agent超时，释放锁
                    stale_resources.append(resource)
            else:
                # Agent不存在，释放锁
                stale_resources.append(resource)
        
        # 删除过期锁
        for resource in stale_resources:
            del state["resources"][resource]
    
    # ==================== 消息队列 ====================
    
    def send_message(self, to_agent: str, msg_type: str, data: Any):
        """
        发送消息给另一个Agent
        
        Args:
            to_agent: 目标Agent ID
            msg_type: 消息类型（如 "task_complete", "error", "request"）
            data: 消息数据
        """
        with self._file_lock():
            state = self._read_state()
            state["messages"].append({
                "id": f"msg_{time.time()}",
                "from": self.agent_id,
                "to": to_agent,
                "type": msg_type,
                "data": data,
                "timestamp": time.time()
            })
            self._write_state(state)
    
    def broadcast_message(self, msg_type: str, data: Any):
        """
        广播消息给所有Agent
        
        Args:
            msg_type: 消息类型
            data: 消息数据
        """
        with self._file_lock():
            state = self._read_state()
            for agent_id in state["agents"]:
                if agent_id != self.agent_id:
                    state["messages"].append({
                        "id": f"msg_{time.time()}_{agent_id}",
                        "from": self.agent_id,
                        "to": agent_id,
                        "type": msg_type,
                        "data": data,
                        "timestamp": time.time()
                    })
            self._write_state(state)
    
    def receive_messages(self, msg_type: Optional[str] = None) -> List[Dict]:
        """
        接收发给自己的消息
        
        Args:
            msg_type: 消息类型过滤（None=接收所有）
        
        Returns:
            List[Dict]: 消息列表
        """
        with self._file_lock():
            state = self._read_state()
            
            # 筛选消息
            messages = [
                msg for msg in state["messages"]
                if msg["to"] == self.agent_id and (msg_type is None or msg["type"] == msg_type)
            ]
            
            # 删除已读消息
            state["messages"] = [
                msg for msg in state["messages"]
                if msg["to"] != self.agent_id or (msg_type and msg["type"] != msg_type)
            ]
            
            self._write_state(state)
            return messages
    
    def wait_for_message(self, msg_type: str, timeout: float = 60.0) -> Optional[Dict]:
        """
        等待特定类型的消息
        
        Args:
            msg_type: 消息类型
            timeout: 超时时间（秒）
        
        Returns:
            Dict | None: 消息或None
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            messages = self.receive_messages(msg_type)
            if messages:
                return messages[0]
            time.sleep(1)
        
        return None
    
    # ==================== 状态管理 ====================
    
    def update_status(self, status: str, task: Optional[str] = None, progress: Optional[float] = None):
        """
        更新Agent状态
        
        Args:
            status: 状态（"idle", "working", "waiting", "error"）
            task: 当前任务
            progress: 进度（0.0-1.0）
        """
        with self._file_lock():
            state = self._read_state()
            
            if self.agent_id not in state["agents"]:
                self._register_agent()
                state = self._read_state()
            
            agent_state = state["agents"][self.agent_id]
            agent_state["status"] = status
            if task is not None:
                agent_state["task"] = task
            if progress is not None:
                agent_state["progress"] = progress
            agent_state["last_heartbeat"] = time.time()
            
            self._write_state(state)
    
    def get_all_agents_status(self) -> Dict[str, Dict]:
        """获取所有Agent的状态"""
        with self._file_lock():
            state = self._read_state()
            return state["agents"]
    
    def heartbeat(self):
        """发送心跳（证明还活着）"""
        with self._file_lock():
            state = self._read_state()
            if self.agent_id in state["agents"]:
                state["agents"][self.agent_id]["last_heartbeat"] = time.time()
                self._write_state(state)
    
    # ==================== 内部方法 ====================
    
    def _file_lock(self):
        """文件锁上下文管理器"""
        class FileLock:
            def __init__(self, file_path):
                self.file_path = file_path
                self.lock_file = None
            
            def __enter__(self):
                lock_path = f"{self.file_path}.lock"
                self.lock_file = open(lock_path, "w")
                fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX)
                return self
            
            def __exit__(self, *args):
                fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)
                self.lock_file.close()
        
        return FileLock(self.STATE_FILE)
    
    def _read_state(self) -> Dict:
        """读取状态（需要在锁内调用）"""
        try:
            return json.loads(self.STATE_FILE.read_text())
        except Exception as e:
            print(f"⚠️ 读取状态失败: {e}")
            return {"agents": {}, "resources": {}, "messages": []}
    
    def _write_state(self, state: Dict):
        """写入状态（需要在锁内调用）"""
        self.STATE_FILE.write_text(json.dumps(state, indent=2))
    
    def cleanup(self):
        """清理资源（退出时调用）"""
        self.release_all_locks()
        with self._file_lock():
            state = self._read_state()
            if self.agent_id in state["agents"]:
                state["agents"][self.agent_id]["status"] = "offline"
            self._write_state(state)
