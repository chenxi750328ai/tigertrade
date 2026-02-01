# TigerTrade多Agent协议规范

> **项目**: TigerTrade  
> **基于**: agentfuture多Agent协作框架  
> **说明**: 本文档定义TigerTrade项目内部Agent间通信协议，基于agentfuture框架扩展了交易场景特定的消息类型和流程

## 版本: v2.1.0

**最后更新**: 2026-01-27

---

## 📋 版本历史

| 版本 | 日期 | 变更内容 |
|------|------|---------|
| **v2.1.1** | 2026-01-27 | **文档更新：明确项目归属和基于agentfuture框架的说明** |
| **v2.1.0** | 2025-01-21 | **新增Agent间自由讨论和分布式RAG** |
| v2.0.0 | 2025-01-21 | 增加propose_task, approve_task权限控制 |
| v1.1.0 | 2025-01-20 | 增加request_help, progress_update异常处理 |
| v1.0.0 | 2025-01-19 | 初始版本，基础Master-Worker架构 |

---

## 1. 消息格式

所有消息遵循统一的JSON格式：

```json
{
  "id": "msg_<timestamp>",
  "from": "<agent_id>",
  "to": "<target>",
  "type": "<message_type>",
  "data": { ... },
  "timestamp": <unix_timestamp>
}
```

### 字段说明

- `id`: 唯一消息ID（由发送者生成）
- `from`: 发送者Agent ID
- `to`: 接收者，可以是：
  - 单个Agent ID: `"worker_A"`
  - 广播: `"all"` ⭐ **v2.1+**
  - 组播: `["worker_A", "worker_B"]` ⭐ **v2.1+**
- `type`: 消息类型（见下文）
- `data`: 消息载荷（根据类型而定）
- `timestamp`: Unix时间戳

---

## 2. 消息类型

### 2.1 基础通信 (v1.0+)

#### worker_ready
Worker注册并准备接收任务

**方向**: Worker → Master  
**必需字段**: -  
**示例**:
```json
{
  "type": "worker_ready",
  "data": {
    "msg": "准备就绪",
    "capabilities": ["data_processing", "model_training"]
  }
}
```

#### task_assign
Master分配任务给Worker

**方向**: Master → Worker  
**必需字段**: `task_id`, `type`  
**示例**:
```json
{
  "type": "task_assign",
  "data": {
    "task_id": "task_xxx",
    "type": "data_clean",
    "description": "清洗原始数据",
    "params": { ... }
  }
}
```

#### task_complete
Worker报告任务完成

**方向**: Worker → Master  
**必需字段**: `task_id`, `result`  
**示例**:
```json
{
  "type": "task_complete",
  "data": {
    "task_id": "task_xxx",
    "result": {
      "status": "success",
      "output": "clean_data.csv",
      "metrics": { ... }
    }
  }
}
```

---

### 2.2 异常处理 (v1.1+)

#### task_failed
Worker报告任务失败

**方向**: Worker → Master  
**必需字段**: `task_id`, `error`  
**示例**:
```json
{
  "type": "task_failed",
  "data": {
    "task_id": "task_xxx",
    "error": "FileNotFoundError: input.csv not found",
    "stack_trace": "..."
  }
}
```

#### request_help
Worker请求Master帮助

**方向**: Worker → Master  
**必需字段**: `problem`  
**示例**:
```json
{
  "type": "request_help",
  "data": {
    "problem": "数据文件不存在",
    "task_id": "task_xxx",
    "context": "正在执行data_clean任务"
  }
}
```

#### progress_update
Worker更新任务进度

**方向**: Worker → Master  
**必需字段**: `task_id`, `progress`  
**示例**:
```json
{
  "type": "progress_update",
  "data": {
    "task_id": "task_xxx",
    "progress": 0.5,
    "message": "已处理500/1000条数据",
    "eta": 120
  }
}
```

#### guidance
Master提供指导

**方向**: Master → Worker  
**必需字段**: `message`  
**示例**:
```json
{
  "type": "guidance",
  "data": {
    "message": "使用备用文件 backup_data.csv",
    "related_task": "task_xxx"
  }
}
```

---

### 2.3 权限控制 (v2.0+)

#### task_proposal
Worker提议新任务（需Master批准）

**方向**: Worker → Master  
**必需字段**: `type`, `description`, `reason`  
**示例**:
```json
{
  "type": "task_proposal",
  "data": {
    "type": "data_validation",
    "description": "验证数据质量",
    "reason": "发现10%数据缺失，需要先验证",
    "priority": "high"
  }
}
```

#### task_approved
Master批准Worker提议的任务

**方向**: Master → Worker  
**必需字段**: `task_id`  
**示例**:
```json
{
  "type": "task_approved",
  "data": {
    "task_id": "proposed_xxx",
    "message": "任务已批准并加入队列",
    "estimated_start": "5分钟后"
  }
}
```

#### task_rejected
Master拒绝Worker提议的任务

**方向**: Master → Worker  
**必需字段**: `task_id`, `reason`  
**示例**:
```json
{
  "type": "task_rejected",
  "data": {
    "task_id": "proposed_xxx",
    "reason": "优先级不够，当前专注于模型训练",
    "alternative": "可以在模型训练完成后重新提议"
  }
}
```

---

### 2.4 Agent间协作 (v2.1+) ⭐ 新增

#### discussion
发起讨论（广播给所有Agent）

**方向**: Any → All  
**必需字段**: `topic`, `question`  
**示例**:
```json
{
  "to": "all",
  "type": "discussion",
  "data": {
    "topic": "数据预处理策略",
    "question": "数据有10%缺失值，大家建议用哪种方法？",
    "options": ["删除", "KNN插值", "均值插值", "保留标记"],
    "deadline": 1737123456
  }
}
```

#### discussion_reply
响应讨论

**方向**: Any → Any  
**必需字段**: `reply_to`, `opinion`  
**示例**:
```json
{
  "type": "discussion_reply",
  "data": {
    "reply_to": "msg_xxx",
    "opinion": "建议用KNN插值，因为白银价格有时间连续性",
    "vote": "KNN插值",
    "confidence": 0.85
  }
}
```

#### project_suggestion
提出项目改进建议

**方向**: Any → All  
**必需字段**: `category`, `suggestion`  
**示例**:
```json
{
  "to": "all",
  "type": "project_suggestion",
  "data": {
    "category": "architecture",
    "suggestion": "建议使用模型集成（Ensemble）",
    "reasoning": "3个模型在不同场景各有优势，集成可提升4%准确率",
    "implementation": "需要3天开发",
    "impact": "准确率: 72% → 76%"
  }
}
```

#### suggestion_vote
对建议投票

**方向**: Any → Any  
**必需字段**: `suggestion_id`, `vote`  
**示例**:
```json
{
  "type": "suggestion_vote",
  "data": {
    "suggestion_id": "sugg_xxx",
    "vote": "approve",
    "comment": "好主意，我可以帮忙实现",
    "resource_offer": "可投入2天时间"
  }
}
```

#### knowledge_share
分享知识或洞察

**方向**: Any → All  
**必需字段**: `title`, `content`  
**示例**:
```json
{
  "to": "all",
  "type": "knowledge_share",
  "data": {
    "category": "insight",
    "title": "白银期货周五下午波动规律",
    "content": "分析1000天数据，周五15:00-16:00波动率是平均值的2.3倍",
    "evidence": {
      "file": "analysis.csv",
      "confidence": 0.95,
      "sample_size": 1000
    },
    "recommendation": "周五下午增加风险控制"
  }
}
```

#### protocol_update
协议版本更新通知

**方向**: Master → All  
**必需字段**: `new_version`, `changes`  
**示例**:
```json
{
  "to": "all",
  "type": "protocol_update",
  "data": {
    "old_version": "2.0.0",
    "new_version": "2.1.0",
    "changes": [
      "新增Agent间自由讨论",
      "新增分布式RAG支持"
    ],
    "documentation": "/home/cx/tigertrade/PROTOCOL.md",
    "breaking_changes": false,
    "action_required": "建议更新以使用新功能"
  }
}
```

#### protocol_version_mismatch
Worker报告协议版本不兼容

**方向**: Worker → Master  
**必需字段**: `worker_version`, `system_version`  
**示例**:
```json
{
  "type": "protocol_version_mismatch",
  "data": {
    "worker_version": "1.0.0",
    "system_version": "2.1.0",
    "request": "请发送新协议文档"
  }
}
```

---

## 3. 状态文件结构

`/tmp/tigertrade_agent_state.json`:

```json
{
  "protocol_version": "2.1.0",
  "last_updated": 1737123456.789,
  
  "agents": {
    "master": {
      "role": "Master",
      "status": "running",
      "last_heartbeat": 1737123456
    },
    "worker_a": {
      "role": "Worker",
      "status": "idle",
      "task": null,
      "progress": 0,
      "last_heartbeat": 1737123450
    }
  },
  
  "resources": {
    "data_lock": {
      "locked": false,
      "holder": null
    }
  },
  
  "messages": [
    {
      "id": "msg_xxx",
      "from": "worker_a",
      "to": "all",
      "type": "discussion",
      "data": { ... },
      "timestamp": 1737123456
    }
  ]
}
```

---

## 4. 版本兼容性

### 语义化版本规范

版本格式: `MAJOR.MINOR.PATCH`

- **MAJOR**: 不兼容的API变更
- **MINOR**: 向后兼容的功能新增
- **PATCH**: 向后兼容的bug修复

### 兼容性规则

| 变更 | 兼容性 | 要求 |
|------|--------|------|
| v1.x → v2.x | ❌ 不兼容 | 必须更新 |
| v2.0 → v2.1 | ✅ 兼容 | 建议更新 |
| v2.1.0 → v2.1.1 | ✅ 完全兼容 | 可选更新 |

### Worker启动检查

```python
def check_protocol_version():
    system_version = get_system_version()
    worker_version = WORKER_PROTOCOL_VERSION
    
    system_major = int(system_version.split('.')[0])
    worker_major = int(worker_version.split('.')[0])
    
    if system_major > worker_major:
        raise ProtocolVersionError("主版本不兼容，必须更新！")
    elif system_version != worker_version:
        warn("有新版本可用，建议更新")
```

---

## 5. 分布式RAG规范 (v2.1+)

### 共享RAG目录结构

```
/home/cx/tigertrade/shared_rag/
├── insights/
│   ├── worker_a_trading_pattern_001.md
│   ├── worker_b_risk_analysis_002.md
├── suggestions/
│   ├── architecture_proposal_001.md
├── findings/
│   ├── data_analysis_001.md
├── discussions/
│   └── preprocessing_strategy_thread_001.json
└── embeddings/
    └── chroma.db
```

### RAG写入规范

```python
# Agent写入RAG
def write_to_rag(category, title, content):
    filename = f"{agent_id}_{category}_{timestamp}.md"
    filepath = f"/home/cx/tigertrade/shared_rag/{category}/{filename}"
    
    with open(filepath, 'w') as f:
        f.write(f"# {title}\n\n")
        f.write(f"作者: {agent_id}\n")
        f.write(f"时间: {datetime.now()}\n\n")
        f.write(content)
    
    # 同时发送knowledge_share消息
    broadcast_message("knowledge_share", {
        "title": title,
        "file": filepath,
        "category": category
    })
```

---

## 6. 最佳实践

### 6.1 消息发送

```python
# ✅ 好的做法
send_message("worker_A", "guidance", {
    "message": "使用备用文件",
    "context": "相关任务xxx"
})

# ❌ 避免
send_message("worker_A", "msg", "use backup")  # 格式不规范
```

### 6.2 广播使用

```python
# 适合广播的场景
- 发起讨论
- 分享知识
- 提出建议
- 协议更新

# 不适合广播的场景
- 任务分配（应点对点）
- 私密信息
- 大量数据传输
```

### 6.3 错误处理

```python
# Worker遇到错误
try:
    execute_task(task)
except Exception as e:
    # 1. 先尝试请求帮助
    send_message("master", "request_help", {
        "problem": str(e),
        "task_id": task_id
    })
    
    # 2. 等待回复
    time.sleep(5)
    
    # 3. 仍无法解决，报告失败
    send_message("master", "task_failed", {
        "task_id": task_id,
        "error": str(e)
    })
```

---

## 7. 协议演进

### 提出协议变更

如果您是Agent开发者，想提出协议变更：

1. 通过`project_suggestion`提出建议
2. 等待社区投票
3. 如果通过，更新PROTOCOL.md
4. 增加版本号
5. 通过`protocol_update`通知所有Agent

### 协议讨论

协议本身也可以成为讨论话题：

```json
{
  "to": "all",
  "type": "discussion",
  "data": {
    "topic": "协议改进提议",
    "question": "是否应该增加Agent间P2P文件传输功能？",
    "context": "当前只能通过共享文件系统，效率较低"
  }
}
```

---

## 8. 附录

### A. 消息类型速查表

| 类型 | 版本 | 方向 | 用途 |
|------|------|------|------|
| worker_ready | v1.0+ | W→M | Worker注册 |
| task_assign | v1.0+ | M→W | 分配任务 |
| task_complete | v1.0+ | W→M | 任务完成 |
| task_failed | v1.1+ | W→M | 任务失败 |
| request_help | v1.1+ | W→M | 请求帮助 |
| progress_update | v1.1+ | W→M | 进度更新 |
| guidance | v1.1+ | M→W | 提供指导 |
| task_proposal | v2.0+ | W→M | 提议任务 |
| task_approved | v2.0+ | M→W | 批准任务 |
| task_rejected | v2.0+ | M→W | 拒绝任务 |
| **discussion** | **v2.1+** | **Any→All** | **发起讨论** |
| **discussion_reply** | **v2.1+** | **Any→Any** | **响应讨论** |
| **project_suggestion** | **v2.1+** | **Any→All** | **提出建议** |
| **suggestion_vote** | **v2.1+** | **Any→Any** | **投票** |
| **knowledge_share** | **v2.1+** | **Any→All** | **分享知识** |
| **protocol_update** | **v2.1+** | **M→All** | **协议更新** |

### B. 术语表

- **Master**: 协调者，负责任务分配和系统协调
- **Worker**: 执行者，执行任务并报告结果
- **Agent**: Master或Worker的统称
- **广播**: 发送给所有Agent
- **组播**: 发送给指定的多个Agent
- **RAG**: Retrieval Augmented Generation，检索增强生成

---

**协议维护**: TigerTrade团队  
**最后更新**: 2025-01-21  
**版本**: v2.1.0
