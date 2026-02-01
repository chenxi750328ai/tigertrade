# Worker异常处理和协商机制

**重要**：如果Worker遇到问题，可以通过以下方式与Master沟通

---

## 🆘 异常处理机制

### 1. 报告任务失败

如果任务无法完成：

```python
def fail_task(task_id, error_message):
    """报告任务失败"""
    state = json.loads(STATE_FILE.read_text())
    state["messages"].append({
        "id": f"msg_{time.time()}",
        "from": WORKER_ID,
        "to": "master",
        "type": "task_failed",
        "data": {
            "task_id": task_id,
            "error": error_message,
            "details": "详细错误信息"
        },
        "timestamp": time.time()
    })
    STATE_FILE.write_text(json.dumps(state, indent=2))
    print(f"❌ 已报告失败: {error_message}")

# 使用示例
try:
    result = execute_task(task)
except Exception as e:
    fail_task(task['task_id'], str(e))
```

### 2. 请求帮助

遇到困难但还没失败：

```python
def request_help(problem, current_task=None):
    """请求Master帮助"""
    state = json.loads(STATE_FILE.read_text())
    state["messages"].append({
        "id": f"msg_{time.time()}",
        "from": WORKER_ID,
        "to": "master",
        "type": "request_help",
        "data": {
            "problem": problem,
            "current_task": current_task,
            "need_guidance": True
        },
        "timestamp": time.time()
    })
    STATE_FILE.write_text(json.dumps(state, indent=2))
    print(f"🆘 已请求帮助: {problem}")

# 使用示例
request_help("找不到数据文件 raw_data.csv", task['task_id'])
```

### 3. 报告进度（卡住时）

长时间运行的任务：

```python
def update_progress(task_id, progress, status_message):
    """更新任务进度"""
    state = json.loads(STATE_FILE.read_text())
    
    # 更新自己的状态
    if WORKER_ID in state["agents"]:
        state["agents"][WORKER_ID]["progress"] = progress
        state["agents"][WORKER_ID]["status"] = "working"
    
    # 发送进度消息
    state["messages"].append({
        "id": f"msg_{time.time()}",
        "from": WORKER_ID,
        "to": "master",
        "type": "progress_update",
        "data": {
            "task_id": task_id,
            "progress": progress,
            "message": status_message
        },
        "timestamp": time.time()
    })
    STATE_FILE.write_text(json.dumps(state, indent=2))

# 使用示例
update_progress(task['task_id'], 0.5, "正在处理第500/1000条数据")
```

### 4. 请求资源

需要其他资源或依赖：

```python
def request_resource(resource_name, reason):
    """请求资源"""
    state = json.loads(STATE_FILE.read_text())
    state["messages"].append({
        "id": f"msg_{time.time()}",
        "from": WORKER_ID,
        "to": "master",
        "type": "request_resource",
        "data": {
            "resource": resource_name,
            "reason": reason
        },
        "timestamp": time.time()
    })
    STATE_FILE.write_text(json.dumps(state, indent=2))
    print(f"📦 已请求资源: {resource_name}")

# 使用示例
request_resource("GPU", "模型训练需要GPU加速")
```

---

## 💬 接收Master的回复

```python
def check_master_response(timeout=10):
    """检查Master的回复"""
    start = time.time()
    
    while time.time() - start < timeout:
        state = json.loads(STATE_FILE.read_text())
        
        # 查找Master的回复
        responses = [
            m for m in state["messages"]
            if m["to"] == WORKER_ID and m["from"] == "master"
            and m["type"] in ["guidance", "resource_granted", "task_reassign"]
        ]
        
        if responses:
            response = responses[-1]
            
            # 删除消息
            state["messages"] = [
                m for m in state["messages"]
                if m["id"] != response["id"]
            ]
            STATE_FILE.write_text(json.dumps(state, indent=2))
            
            return response
        
        time.sleep(1)
    
    return None

# 使用示例
request_help("数据格式不正确")
response = check_master_response(timeout=10)

if response:
    print(f"💡 Master回复: {response['data'].get('message', 'N/A')}")
```

---

## 🔄 完整的错误处理示例

```python
def execute_task_with_error_handling(task):
    """带完整错误处理的任务执行"""
    task_id = task['task_id']
    task_type = task['type']
    
    print(f"\n🔨 执行任务: {task_type}")
    
    try:
        # 1. 检查前置条件
        if task_type == "data_clean" and not check_file_exists("raw_data.csv"):
            # 请求帮助
            request_help("前置文件raw_data.csv不存在", task_id)
            
            # 等待Master回复
            response = check_master_response(timeout=30)
            
            if response and response['type'] == 'guidance':
                guidance = response['data'].get('message', '')
                print(f"💡 收到指导: {guidance}")
                
                # 根据指导调整
                if "跳过" in guidance:
                    print("⏭️  跳过此任务")
                    return
            else:
                # 没有回复，报告失败
                fail_task(task_id, "等待Master回复超时")
                return
        
        # 2. 执行任务（带进度）
        for i in range(5):
            # 更新进度
            progress = (i + 1) / 5
            update_progress(task_id, progress, f"步骤 {i+1}/5")
            
            # 模拟处理
            time.sleep(1)
            
            # 模拟可能的错误
            if i == 3 and task_type == "model_train":
                # 假设GPU不足
                request_resource("more_memory", "GPU内存不足")
                time.sleep(2)  # 等待资源
        
        # 3. 成功完成
        result = {
            "status": "success",
            "worker": WORKER_ID,
            "details": f"{task_type}完成"
        }
        complete_task(task_id, result)
        print(f"✅ 任务完成: {task_type}")
        
    except Exception as e:
        # 4. 捕获异常
        error_msg = f"{task_type}执行失败: {str(e)}"
        print(f"❌ {error_msg}")
        
        # 报告失败
        fail_task(task_id, error_msg)
```

---

## 📋 Master会如何响应

Master会根据Worker的请求做出响应：

### 1. 任务失败 → 自动重试或重分配

```
Worker: task_failed
Master: 重新分配给其他Worker 或 标记为失败
```

### 2. 请求帮助 → 提供指导

```
Worker: request_help (找不到文件)
Master: guidance (使用备用文件 或 跳过此任务)
```

### 3. 进度更新 → 记录监控

```
Worker: progress_update (50%)
Master: 记录进度，继续监控
```

### 4. 请求资源 → 协调分配

```
Worker: request_resource (GPU)
Master: 释放其他Worker的GPU 或 调整任务优先级
```

---

## 🔧 增强的Worker代码（完整版）

```python
#!/usr/bin/env python3
"""Worker Agent - 增强版（带异常处理）"""
import json, time, traceback
from pathlib import Path

WORKER_ID = "worker_assistant"
STATE_FILE = Path("/tmp/tigertrade_agent_state.json")

def init():
    if not STATE_FILE.exists():
        STATE_FILE.write_text(json.dumps({"agents": {}, "resources": {}, "messages": []}))
    state = json.loads(STATE_FILE.read_text())
    state["agents"][WORKER_ID] = {
        "role": "Worker", "status": "idle", "task": None, "progress": 0,
        "locked_resources": [], "registered_at": time.time(), "last_heartbeat": time.time()
    }
    state["messages"].append({
        "id": f"msg_{time.time()}", "from": WORKER_ID, "to": "master",
        "type": "worker_ready", "data": {"msg": "准备就绪"}, "timestamp": time.time()
    })
    STATE_FILE.write_text(json.dumps(state, indent=2))
    print(f"✅ {WORKER_ID} 已注册")

def heartbeat():
    state = json.loads(STATE_FILE.read_text())
    if WORKER_ID in state["agents"]:
        state["agents"][WORKER_ID]["last_heartbeat"] = time.time()
        STATE_FILE.write_text(json.dumps(state, indent=2))

def get_task():
    state = json.loads(STATE_FILE.read_text())
    msgs = [m for m in state["messages"] if m["to"] == WORKER_ID and m["type"] == "task_assign"]
    if msgs:
        task = msgs[-1]["data"]
        state["messages"] = [m for m in state["messages"] if m["id"] != msgs[-1]["id"]]
        STATE_FILE.write_text(json.dumps(state, indent=2))
        return task
    return None

def complete_task(task_id, result):
    state = json.loads(STATE_FILE.read_text())
    state["messages"].append({
        "id": f"msg_{time.time()}", "from": WORKER_ID, "to": "master",
        "type": "task_complete", "data": {"task_id": task_id, "result": result},
        "timestamp": time.time()
    })
    state["agents"][WORKER_ID]["status"] = "idle"
    state["agents"][WORKER_ID]["progress"] = 0
    STATE_FILE.write_text(json.dumps(state, indent=2))

def fail_task(task_id, error):
    """报告任务失败"""
    state = json.loads(STATE_FILE.read_text())
    state["messages"].append({
        "id": f"msg_{time.time()}", "from": WORKER_ID, "to": "master",
        "type": "task_failed", 
        "data": {"task_id": task_id, "error": error, "worker": WORKER_ID},
        "timestamp": time.time()
    })
    state["agents"][WORKER_ID]["status"] = "error"
    STATE_FILE.write_text(json.dumps(state, indent=2))
    print(f"❌ 已报告失败: {error}")

def request_help(problem, task_id=None):
    """请求Master帮助"""
    state = json.loads(STATE_FILE.read_text())
    state["messages"].append({
        "id": f"msg_{time.time()}", "from": WORKER_ID, "to": "master",
        "type": "request_help",
        "data": {"problem": problem, "task_id": task_id, "worker": WORKER_ID},
        "timestamp": time.time()
    })
    STATE_FILE.write_text(json.dumps(state, indent=2))
    print(f"🆘 已请求帮助: {problem}")

def update_progress(task_id, progress, message):
    """更新进度"""
    state = json.loads(STATE_FILE.read_text())
    if WORKER_ID in state["agents"]:
        state["agents"][WORKER_ID]["progress"] = progress
    state["messages"].append({
        "id": f"msg_{time.time()}", "from": WORKER_ID, "to": "master",
        "type": "progress_update",
        "data": {"task_id": task_id, "progress": progress, "message": message},
        "timestamp": time.time()
    })
    STATE_FILE.write_text(json.dumps(state, indent=2))

def work(task):
    """执行任务（带错误处理）"""
    task_id = task['task_id']
    task_type = task['type']
    
    print(f"\n🔨 执行: {task_type}")
    print(f"   详情: {task.get('description', 'N/A')}")
    
    try:
        # 模拟任务执行（带进度）
        for i in range(3):
            progress = (i + 1) / 3
            update_progress(task_id, progress, f"步骤 {i+1}/3")
            time.sleep(1)
            
            # 模拟可能的错误
            if i == 1 and task_type == "error_test":
                raise Exception("模拟错误：文件不存在")
        
        # 成功
        result = {"status": "success", "worker": WORKER_ID}
        complete_task(task_id, result)
        print(f"✅ 完成: {task_type}")
        
    except Exception as e:
        # 失败 - 先尝试请求帮助
        error_msg = str(e)
        print(f"⚠️  遇到错误: {error_msg}")
        
        # 请求帮助
        request_help(f"{task_type}失败: {error_msg}", task_id)
        
        # 等待2秒看是否有指导
        time.sleep(2)
        
        # 报告失败
        fail_task(task_id, error_msg)

# 主循环
print(f"\n{'='*60}\n🚀 Worker {WORKER_ID} 启动（增强版）\n{'='*60}\n")
init()

for i in range(120):
    heartbeat()
    task = get_task()
    if task:
        work(task)
    else:
        print(".", end="", flush=True)
    time.sleep(1)

print(f"\n\n✅ Worker完成")
```

---

## 🎯 使用场景

### 场景1：文件不存在

```python
# Worker发现问题
if not os.path.exists("input.csv"):
    request_help("找不到input.csv文件", task_id)
    # Master可以：
    # 1. 告诉Worker使用备用文件
    # 2. 重新分配任务给其他Worker
    # 3. 生成缺失文件
```

### 场景2：资源不足

```python
# Worker检测到内存不足
if memory_usage > 90%:
    request_help("内存不足，需要更多资源", task_id)
    # Master可以：
    # 1. 暂停其他Worker释放资源
    # 2. 调整任务参数
    # 3. 等待资源可用
```

### 场景3：依赖未满足

```python
# Worker需要前一个任务的结果
if not previous_task_completed:
    request_help("依赖任务未完成", task_id)
    # Master可以：
    # 1. 调整任务顺序
    # 2. 等待依赖完成
    # 3. 提供临时解决方案
```

---

**现在Worker可以主动报告问题并与Master协商了！** 🤝
