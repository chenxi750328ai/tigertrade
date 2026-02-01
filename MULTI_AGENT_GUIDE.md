# 🤝 TigerTrade 多Agent协作指南

**支持跨平台AI协作** - Master在Cursor，Worker可以是任何AI工具

---

## 📋 架构

```
┌──────────────────┐
│  Master Agent    │  ← Cursor (当前)
│  - 任务分配       │
│  - 进度监控       │
│  - 结果汇总       │
└────────┬─────────┘
         │
    ┌────┴────┬────────┬────────┐
    ↓         ↓        ↓        ↓
 Worker1   Worker2  Worker3  Worker4
(ChatGPT) (Claude)  (Gemini) (其他AI)
```

---

## 🚀 启动步骤

### 步骤1: 启动Master（当前Cursor）

```bash
cd /home/cx/tigertrade
python run_master.py
```

**Master会自动**：
- ✅ 注册4个示例任务
- ✅ 等待Worker连接（2分钟）
- ✅ 分配任务给空闲Worker
- ✅ 实时监控进度
- ✅ 生成最终报告

### 步骤2: 通知其他AI启动Worker

**📋 复制以下内容发送给ChatGPT/Claude/Gemini等**：

---

## 📨 给其他AI的通知（复制发送）

```
你好！我正在运行一个TigerTrade多Agent协作项目，需要你作为Worker Agent协助完成任务。

请运行以下Python代码（2分钟）：

[完整代码见下方]

重要提醒：
1. 修改 WORKER_ID = "worker_你的名字"（例如：worker_chatgpt）
2. 这个代码会连接到共享的任务队列
3. 你会接收Master分配的任务（数据处理、模型训练等）
4. 完成后自动报告给Master

运行环境要求：
- Python 3.x
- 可以访问 /tmp/tigertrade_agent_state.json

准备好了吗？运行后等待Master分配任务！
```

**完整Worker代码**：

```python
#!/usr/bin/env python3
"""Worker Agent - 快速启动"""
import json, time
from pathlib import Path

# ⚠️ 修改你的Worker名称
WORKER_ID = "worker_assistant"  # 改成：worker_chatgpt, worker_claude等
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
    STATE_FILE.write_text(json.dumps(state, indent=2))

def work(task):
    print(f"\n🔨 执行: {task['type']}")
    print(f"   详情: {task.get('description', 'N/A')}")
    time.sleep(3)  # 模拟任务执行
    result = {"status": "success", "worker": WORKER_ID}
    complete_task(task['task_id'], result)
    print(f"✅ 完成: {task['type']}")

# 主循环
print(f"\n{'='*60}")
print(f"🚀 Worker {WORKER_ID} 启动")
print(f"{'='*60}\n")

init()

for i in range(120):  # 2分钟
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

## 📊 示例任务

Master会分配以下任务：

| 任务类型 | 描述 | 预计耗时 |
|---------|------|---------|
| `data_download` | 下载SIL2603历史数据 | 3秒 |
| `data_clean` | 清洗和标准化数据 | 3秒 |
| `model_train` | 训练Transformer模型 | 3秒 |
| `backtest` | 回测交易策略 | 3秒 |

---

## 🔍 监控

### 查看所有Agent状态

```bash
cat /tmp/tigertrade_agent_state.json | jq '.agents'
```

### 查看任务队列

```bash
cat /tmp/tigertrade_task_queue.json | jq
```

### 实时监控

```bash
watch -n 1 'cat /tmp/tigertrade_agent_state.json | jq ".agents"'
```

---

## 💡 协作特性

| 特性 | 说明 |
|------|------|
| **自动发现** | Master自动发现新注册的Worker |
| **智能分配** | 优先分配给空闲Worker |
| **容错机制** | Worker掉线自动重分配任务 |
| **心跳检测** | 60秒无响应判定离线 |
| **进度监控** | 实时查看所有任务进度 |
| **结果汇总** | 自动收集并汇总Worker结果 |

---

## 🎯 预期输出

### Master端

```
📋 Master: 注册项目 'TigerTrade数据处理'

✅ 已注册 4 个任务

🚀 Master Agent 启动
⏰ Master将运行120秒，等待Worker连接...

🤝 发现新Worker: worker_chatgpt (Worker)
📤 Master → worker_chatgpt: 分配任务 'data_download'

✅ worker_chatgpt: 任务完成 'task_xxx'
   结果: {'status': 'success', 'worker': 'worker_chatgpt'}

📊 Master状态报告
👥 Workers (1):
   [worker_chatgpt] idle     | 任务: N/A

📋 任务队列:
   待分配: 0
   执行中: 0
   已完成: 4
   失败: 0

✅ 所有任务完成！
```

### Worker端

```
🚀 Worker worker_chatgpt 启动

✅ worker_chatgpt 已注册
.........
🔨 执行: data_download
   详情: 下载SIL2603历史数据
✅ 完成: data_download
.........
✅ Worker完成
```

---

## 📚 完整文档

- **Worker快速指令**: `WORKER_QUICK_START.md`
- **Worker完整文档**: `WORKER_INSTRUCTIONS.md`
- **架构设计**: `docs/并发架构最终方案.md`
- **协调器API**: `src/coordinator/master_agent.py`

---

## 🆘 常见问题

**Q: Worker找不到文件？**  
A: 确保路径为 `/tmp/tigertrade_agent_state.json`

**Q: Worker不接收任务？**  
A: 检查`WORKER_ID`是否唯一，确保Master已启动

**Q: 如何自定义任务？**  
A: 修改`run_master.py`中的`register_project()`

**Q: 跨机器协作？**  
A: 将`/tmp/tigertrade_*.json`放到共享存储（如NFS、S3）

---

**准备开始多Agent协作！** 🚀
