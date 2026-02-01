# 给其他AI工具的Worker接入指令

## 👋 欢迎加入TigerTrade多Agent协作！

您将作为**Worker Agent**协助Master完成任务。

---

## 🚀 一键启动（复制以下代码运行）

```python
#!/usr/bin/env python3
"""Worker Agent - 快速启动版"""
import json, time
from pathlib import Path

# ⚠️ 修改您的Worker名称
WORKER_ID = "worker_assistant"  # 例如：worker_chatgpt, worker_claude
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
    print(f"✅ {WORKER_ID} 已注册并通知Master")

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
    STATE_FILE.write_text(json.dumps(state, indent=2))

def work(task):
    """执行任务 - 根据实际情况修改"""
    print(f"\n🔨 执行: {task['type']}")
    print(f"   详情: {task.get('description', 'N/A')}")
    
    # TODO: 根据任务类型实际执行
    # 这里是模拟
    time.sleep(3)
    
    result = {"status": "success", "worker": WORKER_ID, "note": "任务完成"}
    complete_task(task['task_id'], result)
    print(f"✅ 完成: {task['type']}")

# 主循环
print(f"\n{'='*60}")
print(f"🚀 Worker {WORKER_ID} 启动")
print(f"{'='*60}\n")

init()

for i in range(120):  # 运行2分钟
    heartbeat()
    task = get_task()
    if task:
        work(task)
    else:
        print(".", end="", flush=True)
    time.sleep(1)

print(f"\n\n✅ Worker {WORKER_ID} 完成")
```

---

## 📊 运行后

您会看到：
```
✅ worker_assistant 已注册并通知Master
.......
🔨 执行: data_download
   详情: 下载SIL2603历史数据
✅ 完成: data_download
.......
```

---

## 🔧 任务类型

可能收到的任务：
- `data_download`: 下载市场数据
- `data_clean`: 数据清洗
- `model_train`: 模型训练
- `backtest`: 策略回测
- 其他自定义任务

---

## 💡 提示

1. **唯一ID**: 确保`WORKER_ID`唯一
2. **文件路径**: `/tmp/tigertrade_agent_state.json`必须正确
3. **实际执行**: 修改`work()`函数实现真实任务
4. **求助**: 可发送`request_help`消息

---

**准备就绪！等待Master分配任务！** 🎯
