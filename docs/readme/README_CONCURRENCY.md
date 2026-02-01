# TigerTrade 并发架构速查

## 🎯 核心问题

**Q1**: 如何避免Agent冲突？  
**A1**: 用协调器（`src/coordinator/`）实现锁、消息、状态同步

**Q2**: 如何避免串行等待？  
**A2**: 用流水线（Queue）或完全并行，避免依赖链

## 📊 三种模式

| 模式 | 适用 | 耗时 | 加速 |
|------|------|------|------|
| 串行 | 单次流程 | Σt | 1x |
| 流水线 | 持续数据 | max(t) | 1.6x |
| 完全并行 | 独立任务 | max(t) | nx |

## 🚀 快速开始

```bash
# 测试协调器
python tests/test_coordinator.py

# 查看流水线演示
python examples/pipeline_quick_demo.py

# 实际应用（待实现）
python src/realtime_pipeline.py
```

## 📚 文档

- `docs/并发架构最终方案.md` - 完整方案 ⭐
- `docs/流水线并发vs依赖链串行.md` - 性能对比
- `docs/协作机制总结.md` - 协作机制

## 💡 关键代码

```python
# 协调器（避免冲突）
from src.coordinator import AgentCoordinator
coord = AgentCoordinator("agent1")
coord.acquire_lock("resource")
coord.send_message("agent2", "ready", {})

# 流水线（避免串行）
from queue import Queue
queue = Queue()
# Producer: queue.put(item)
# Consumer: item = queue.get()
```

**实测**: 流水线比串行快1.6x，CPU利用率从33%→100% ✅
