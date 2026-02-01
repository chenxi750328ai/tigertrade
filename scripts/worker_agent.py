#!/usr/bin/env python3
"""Worker Agent"""
import json, time
from pathlib import Path

# ⚠️ 修改你的Worker名称
WORKER_ID = "worker_lingma"  # 改成：worker_chatgpt, worker_claude等
STATE_FILE = Path("/tmp/tigertrade_agent_state.json")

def init():
    if not STATE_FILE.exists():
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
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
    STATE_FILE.write_text(json.dumps(state, indent=2))

def work(task):
    print(f"\n🔨 执行: {task['type']}")
    print(f"   详情: {task.get('description', 'N/A')}")
    time.sleep(3)
    result = {"status": "success", "worker": WORKER_ID, "task_id": task['task_id']}
    complete_task(task['task_id'], result)
    print(f"✅ 完成: {task['type']}")

print(f"\n{'='*60}\n🚀 Worker {WORKER_ID} 启动\n{'='*60}\n")
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