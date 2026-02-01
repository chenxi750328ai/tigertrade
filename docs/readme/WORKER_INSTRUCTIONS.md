# Worker Agent 使用说明

**给其他AI工具的指令** - 复制这个文档给另一个AI（如ChatGPT、Claude等）

---

## 🎯 您的角色

您是一个 **Worker Agent**，负责执行Master分配的任务。

**Worker ID**: `worker_<您选择的名称>`（例如：`worker_chatgpt`, `worker_claude2`）

---

## 📋 您需要做的事情

### 1. 初始化

运行以下Python代码初始化Worker：

```python
import json
import time
from pathlib import Path

# Worker配置
WORKER_ID = "worker_<您的名称>"  # ⚠️ 修改这里！
WORKER_ROLE = "Worker Agent"

# 文件路径
STATE_FILE = Path("/tmp/tigertrade_agent_state.json")
TASK_QUEUE_FILE = Path("/tmp/tigertrade_task_queue.json")

# 注册Worker
def register_worker():
    """注册Worker到系统"""
    if not STATE_FILE.exists():
        STATE_FILE.write_text(json.dumps({
            "agents": {},
            "resources": {},
            "messages": []
        }))
    
    state = json.loads(STATE_FILE.read_text())
    state["agents"][WORKER_ID] = {
        "role": WORKER_ROLE,
        "status": "idle",
        "task": None,
        "progress": 0.0,
        "locked_resources": [],
        "registered_at": time.time(),
        "last_heartbeat": time.time()
    }
    STATE_FILE.write_text(json.dumps(state, indent=2))
    print(f"✅ Worker {WORKER_ID} 已注册")

# 发送消息给Master
def send_to_master(msg_type, data):
    """发送消息给Master"""
    state = json.loads(STATE_FILE.read_text())
    state["messages"].append({
        "id": f"msg_{time.time()}",
        "from": WORKER_ID,
        "to": "master",
        "type": msg_type,
        "data": data,
        "timestamp": time.time()
    })
    STATE_FILE.write_text(json.dumps(state, indent=2))
    print(f"📤 发送消息给Master: {msg_type}")

# 注册并告知Master准备就绪
register_worker()
send_to_master("worker_ready", {"message": "Worker准备就绪"})
```

### 2. 获取任务

```python
def get_my_task():
    """获取分配给自己的任务"""
    # 检查消息
    state = json.loads(STATE_FILE.read_text())
    
    my_messages = [
        msg for msg in state["messages"]
        if msg["to"] == WORKER_ID and msg["type"] == "task_assign"
    ]
    
    if my_messages:
        # 获取最新任务
        task_msg = my_messages[-1]
        task = task_msg["data"]
        
        # 删除消息
        state["messages"] = [
            msg for msg in state["messages"]
            if msg["id"] != task_msg["id"]
        ]
        STATE_FILE.write_text(json.dumps(state, indent=2))
        
        print(f"\n📥 收到任务: {task['type']}")
        print(f"   任务ID: {task['task_id']}")
        print(f"   详情: {task}")
        
        return task
    
    return None

# 检查任务
task = get_my_task()
if task:
    print("有新任务！")
else:
    print("暂无任务，等待Master分配...")
```

### 3. 执行任务

根据任务类型执行相应操作：

```python
def execute_task(task):
    """执行任务"""
    task_type = task.get('type')
    task_id = task['task_id']
    
    print(f"\n🔨 开始执行任务: {task_type}")
    
    # 更新状态为工作中
    state = json.loads(STATE_FILE.read_text())
    state["agents"][WORKER_ID]["status"] = "working"
    state["agents"][WORKER_ID]["task"] = task_type
    STATE_FILE.write_text(json.dumps(state, indent=2))
    
    try:
        # 根据任务类型执行
        if task_type == "data_download":
            result = download_data(task)
        elif task_type == "data_clean":
            result = clean_data(task)
        elif task_type == "model_train":
            result = train_model(task)
        elif task_type == "backtest":
            result = run_backtest(task)
        else:
            result = {"status": "unknown_task_type"}
        
        # 报告完成
        send_to_master("task_complete", {
            "task_id": task_id,
            "result": result
        })
        
        # 更新状态为空闲
        state = json.loads(STATE_FILE.read_text())
        state["agents"][WORKER_ID]["status"] = "idle"
        state["agents"][WORKER_ID]["task"] = None
        STATE_FILE.write_text(json.dumps(state, indent=2))
        
        print(f"✅ 任务完成: {task_type}")
        
    except Exception as e:
        # 报告失败
        send_to_master("task_failed", {
            "task_id": task_id,
            "error": str(e)
        })
        
        print(f"❌ 任务失败: {e}")

# 示例任务实现
def download_data(task):
    """下载数据"""
    symbol = task.get('symbol', 'SIL2603')
    print(f"   下载 {symbol} 数据...")
    time.sleep(2)  # 模拟处理
    return {"status": "success", "records": 1000}

def clean_data(task):
    """清洗数据"""
    file = task.get('file', 'data.csv')
    print(f"   清洗 {file}...")
    time.sleep(3)  # 模拟处理
    return {"status": "success", "cleaned_records": 950}

def train_model(task):
    """训练模型"""
    model = task.get('model', 'transformer')
    print(f"   训练 {model} 模型...")
    time.sleep(5)  # 模拟处理
    return {"status": "success", "accuracy": 0.85}

def run_backtest(task):
    """运行回测"""
    strategy = task.get('strategy', 'grid')
    print(f"   回测 {strategy} 策略...")
    time.sleep(3)  # 模拟处理
    return {"status": "success", "return": 0.23}

# 执行任务（如果有）
if task:
    execute_task(task)
```

### 4. 持续运行（循环模式）

```python
def run_worker_loop(duration=60):
    """Worker主循环"""
    print(f"\n{'='*70}")
    print(f"🚀 Worker {WORKER_ID} 开始运行")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    while time.time() - start_time < duration:
        # 1. 心跳
        state = json.loads(STATE_FILE.read_text())
        if WORKER_ID in state["agents"]:
            state["agents"][WORKER_ID]["last_heartbeat"] = time.time()
            STATE_FILE.write_text(json.dumps(state, indent=2))
        
        # 2. 检查新任务
        task = get_my_task()
        
        if task:
            execute_task(task)
        else:
            print(".", end="", flush=True)  # 等待指示
        
        time.sleep(2)
    
    print(f"\n\n✅ Worker {WORKER_ID} 运行结束")

# 运行Worker
run_worker_loop(duration=60)
```

---

## 🎯 快速开始（一键运行）

将以下完整代码复制到Python环境运行：

```python
#!/usr/bin/env python3
"""Worker Agent - 完整实现"""

import json
import time
from pathlib import Path

# ==================== 配置 ====================
WORKER_ID = "worker_test"  # ⚠️ 修改您的Worker名称！
WORKER_ROLE = "Worker Agent"
STATE_FILE = Path("/tmp/tigertrade_agent_state.json")
TASK_QUEUE_FILE = Path("/tmp/tigertrade_task_queue.json")

# ==================== 核心函数 ====================

def register_worker():
    """注册Worker"""
    if not STATE_FILE.exists():
        STATE_FILE.write_text(json.dumps({"agents": {}, "resources": {}, "messages": []}))
    
    state = json.loads(STATE_FILE.read_text())
    state["agents"][WORKER_ID] = {
        "role": WORKER_ROLE,
        "status": "idle",
        "task": None,
        "progress": 0.0,
        "locked_resources": [],
        "registered_at": time.time(),
        "last_heartbeat": time.time()
    }
    STATE_FILE.write_text(json.dumps(state, indent=2))
    print(f"✅ Worker {WORKER_ID} 已注册")

def send_to_master(msg_type, data):
    """发送消息给Master"""
    state = json.loads(STATE_FILE.read_text())
    state["messages"].append({
        "id": f"msg_{time.time()}",
        "from": WORKER_ID,
        "to": "master",
        "type": msg_type,
        "data": data,
        "timestamp": time.time()
    })
    STATE_FILE.write_text(json.dumps(state, indent=2))

def get_my_task():
    """获取任务"""
    state = json.loads(STATE_FILE.read_text())
    my_messages = [msg for msg in state["messages"] 
                   if msg["to"] == WORKER_ID and msg["type"] == "task_assign"]
    
    if my_messages:
        task_msg = my_messages[-1]
        task = task_msg["data"]
        state["messages"] = [msg for msg in state["messages"] if msg["id"] != task_msg["id"]]
        STATE_FILE.write_text(json.dumps(state, indent=2))
        return task
    return None

def execute_task(task):
    """执行任务"""
    task_type = task.get('type')
    task_id = task['task_id']
    
    print(f"\n🔨 执行: {task_type}")
    
    # 模拟任务执行
    time.sleep(3)
    
    # 报告完成
    send_to_master("task_complete", {
        "task_id": task_id,
        "result": {"status": "success", "worker": WORKER_ID}
    })
    
    print(f"✅ 完成: {task_type}")

# ==================== 主循环 ====================

def main():
    print(f"\n{'='*70}")
    print(f"🚀 Worker {WORKER_ID} 启动")
    print(f"{'='*70}\n")
    
    register_worker()
    send_to_master("worker_ready", {"message": "Worker准备就绪"})
    
    for i in range(60):  # 运行60秒
        # 心跳
        state = json.loads(STATE_FILE.read_text())
        if WORKER_ID in state["agents"]:
            state["agents"][WORKER_ID]["last_heartbeat"] = time.time()
            STATE_FILE.write_text(json.dumps(state, indent=2))
        
        # 检查任务
        task = get_my_task()
        if task:
            execute_task(task)
        else:
            print(".", end="", flush=True)
        
        time.sleep(1)
    
    print(f"\n\n✅ Worker完成")

if __name__ == '__main__':
    main()
```

---

## 📊 监控

查看Worker状态：

```bash
# 查看所有Agent状态
cat /tmp/tigertrade_agent_state.json | jq '.agents'

# 查看任务队列
cat /tmp/tigertrade_task_queue.json | jq
```

---

## 🆘 常见问题

**Q: 找不到任务？**  
A: 确保Master已经启动并注册了任务

**Q: Worker不工作？**  
A: 检查`WORKER_ID`是否唯一，检查文件路径是否正确

**Q: 如何请求帮助？**  
A: 发送消息：`send_to_master("request_help", {"problem": "描述问题"})`

---

**准备就绪！等待Master分配任务！** 🚀
