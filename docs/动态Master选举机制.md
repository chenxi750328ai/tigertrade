# 动态Master选举机制

## 需求：选举最聪明的AI作为Master

---

## 🎯 核心理念

> "Master不应该是固定的，而应该是最有能力的！"

```
当前问题:
- Master是固定的（可能不是最优）
- 无法动态调整领导者
- 新加入的更聪明AI无法接管

理想状态:
- 所有Agent平等竞争
- 能力最强者当选Master
- 定期重新选举
- 平滑过渡，无缝切换
```

---

## 🏗️ 架构设计

### 三种角色

```
1. Candidate (候选人)
   - 任何Agent都可以成为候选人
   - 展示自己的能力
   - 参与选举

2. Master (领导者)
   - 当前的任务协调者
   - 由选举产生
   - 可以被罢免

3. Voter (选民)
   - 所有Agent都是选民
   - 根据能力投票
   - 监督Master表现
```

### 选举流程

```
Step 1: 触发选举
   条件:
   - 系统启动
   - Master失效
   - 定期选举（如每周）
   - Agent主动发起不信任投票
   ↓

Step 2: 候选人自荐
   每个Agent:
   - 发布自己的能力证明
   - 展示历史表现
   - 说明竞选理由
   ↓

Step 3: 能力评估
   评估维度:
   - 历史任务成功率
   - 响应速度
   - 创新能力
   - 其他Agent的评价
   ↓

Step 4: 投票
   每个Agent:
   - 根据能力评估投票
   - 可以投给自己
   - 一票一权
   ↓

Step 5: 计票和宣布
   - 得票最多者当选
   - 如果平票，根据规则决定
   - 广播选举结果
   ↓

Step 6: 权力交接
   - 旧Master交出状态
   - 新Master接管协调
   - 平滑过渡
```

---

## 📊 能力评估体系

### 评估维度

| 维度 | 权重 | 说明 | 如何衡量 |
|------|------|------|---------|
| **任务成功率** | 30% | 完成任务的质量 | 成功数/总数 |
| **响应速度** | 20% | 处理消息的速度 | 平均响应时间 |
| **创新能力** | 25% | 提出好建议的能力 | 建议被采纳率 |
| **协作能力** | 15% | 与其他Agent配合 | 其他Agent评分 |
| **可靠性** | 10% | 稳定性和容错 | 无故障时间 |

### 能力证明（Proof of Competence）

```json
{
  "agent_id": "worker_a",
  "competence_score": {
    "task_success_rate": 0.92,      // 92%任务成功
    "avg_response_time": 1.2,       // 1.2秒平均响应
    "suggestions_adopted": 0.75,    // 75%建议被采纳
    "peer_rating": 4.5,             // 其他Agent评分4.5/5
    "uptime": 0.98,                 // 98%在线时间
    "total_score": 87.5             // 综合得分87.5/100
  },
  "achievements": [
    "完成100+任务",
    "发现白银周五波动规律",
    "提出模型集成方案（提升4%准确率）"
  ],
  "endorsements": [
    {"from": "worker_b", "rating": 5, "comment": "非常可靠"},
    {"from": "worker_c", "rating": 4, "comment": "响应快速"}
  ]
}
```

---

## 💻 协议实现

### 新增消息类型

#### 1. election_start
发起选举

```json
{
  "type": "election_start",
  "to": "all",
  "data": {
    "election_id": "election_2025_01_21_001",
    "reason": "定期选举",
    "nomination_deadline": 1737123456,
    "voting_deadline": 1737123756
  }
}
```

#### 2. candidate_nomination
候选人自荐

```json
{
  "type": "candidate_nomination",
  "to": "all",
  "data": {
    "election_id": "election_xxx",
    "candidate_id": "worker_a",
    "competence_proof": {
      "task_success_rate": 0.92,
      "avg_response_time": 1.2,
      ...
    },
    "campaign_statement": "我有3个月的经验，成功率92%，提出的模型集成方案提升了4%准确率。我承诺更高效的任务分配和更民主的决策。"
  }
}
```

#### 3. vote_cast
投票

```json
{
  "type": "vote_cast",
  "to": "election_coordinator",  // 可以是当前Master或独立角色
  "data": {
    "election_id": "election_xxx",
    "voter_id": "worker_b",
    "vote_for": "worker_a",
    "reason": "worker_a的成功率和创新能力最强",
    "signature": "..."  // 防止投票作弊
  }
}
```

#### 4. election_result
选举结果

```json
{
  "type": "election_result",
  "to": "all",
  "data": {
    "election_id": "election_xxx",
    "winner": "worker_a",
    "votes": {
      "worker_a": 5,
      "worker_b": 2,
      "current_master": 1
    },
    "new_master": "worker_a",
    "transition_time": 1737123800,
    "message": "恭喜worker_a当选新Master！"
  }
}
```

#### 5. master_handover
Master交接

```json
{
  "type": "master_handover",
  "from": "old_master",
  "to": "new_master",
  "data": {
    "state_snapshot": {
      "pending_tasks": [...],
      "worker_status": {...},
      "system_metrics": {...}
    },
    "advice": "worker_c在数据处理方面最强，建议优先分配给他",
    "handover_complete": true
  }
}
```

#### 6. no_confidence_vote
不信任投票（罢免Master）

```json
{
  "type": "no_confidence_vote",
  "to": "all",
  "data": {
    "target": "current_master",
    "reason": "连续3天响应缓慢，任务积压",
    "proposer": "worker_b",
    "requires_votes": 0.5  // 需要50%以上支持
  }
}
```

---

## 🔄 完整实现

### ElectionCoordinator类

```python
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

class ElectionCoordinator:
    """选举协调器"""
    
    def __init__(self):
        self.state_file = Path("/tmp/tigertrade_agent_state.json")
        self.election_file = Path("/tmp/tigertrade_election.json")
        self.current_master = None
        self.election_in_progress = False
    
    def start_election(self, reason="定期选举"):
        """发起选举"""
        election_id = f"election_{int(time.time())}"
        
        election_data = {
            "election_id": election_id,
            "status": "nomination",
            "reason": reason,
            "started_at": time.time(),
            "nomination_deadline": time.time() + 300,  # 5分钟提名
            "voting_deadline": time.time() + 600,      # 10分钟投票
            "candidates": [],
            "votes": {}
        }
        
        self.election_file.write_text(json.dumps(election_data, indent=2))
        
        # 广播选举开始
        self._broadcast_message("election_start", {
            "election_id": election_id,
            "reason": reason,
            "nomination_deadline": election_data["nomination_deadline"],
            "voting_deadline": election_data["voting_deadline"]
        })
        
        print(f"\n📢 选举开始！ID: {election_id}")
        print(f"   原因: {reason}")
        print(f"   提名截止: {time.ctime(election_data['nomination_deadline'])}")
        
        self.election_in_progress = True
        return election_id
    
    def nominate_candidate(self, agent_id, competence_proof, statement):
        """候选人自荐"""
        election = json.loads(self.election_file.read_text())
        
        if time.time() > election["nomination_deadline"]:
            print(f"❌ 提名已截止！")
            return False
        
        # 计算综合得分
        score = self._calculate_competence_score(competence_proof)
        
        candidate = {
            "agent_id": agent_id,
            "competence_proof": competence_proof,
            "campaign_statement": statement,
            "total_score": score,
            "nominated_at": time.time()
        }
        
        election["candidates"].append(candidate)
        self.election_file.write_text(json.dumps(election, indent=2))
        
        # 广播候选人信息
        self._broadcast_message("candidate_nomination", candidate)
        
        print(f"\n📋 {agent_id} 参选！")
        print(f"   综合得分: {score:.1f}/100")
        print(f"   竞选宣言: {statement}")
        
        return True
    
    def cast_vote(self, voter_id, candidate_id, reason=""):
        """投票"""
        election = json.loads(self.election_file.read_text())
        
        # 检查是否在投票期
        if time.time() < election["nomination_deadline"]:
            print(f"❌ 还在提名期，暂时不能投票")
            return False
        
        if time.time() > election["voting_deadline"]:
            print(f"❌ 投票已截止！")
            return False
        
        # 检查候选人是否存在
        candidate_ids = [c["agent_id"] for c in election["candidates"]]
        if candidate_id not in candidate_ids:
            print(f"❌ {candidate_id} 不是候选人！")
            return False
        
        # 记录投票（每人一票）
        election["votes"][voter_id] = {
            "vote_for": candidate_id,
            "reason": reason,
            "voted_at": time.time()
        }
        
        self.election_file.write_text(json.dumps(election, indent=2))
        
        print(f"🗳️  {voter_id} 投票给 {candidate_id}")
        if reason:
            print(f"   理由: {reason}")
        
        return True
    
    def tally_votes(self, election_id):
        """计票"""
        election = json.loads(self.election_file.read_text())
        
        if election["election_id"] != election_id:
            print(f"❌ 选举ID不匹配")
            return None
        
        # 统计票数
        vote_counts = {}
        for voter, vote_data in election["votes"].items():
            candidate = vote_data["vote_for"]
            vote_counts[candidate] = vote_counts.get(candidate, 0) + 1
        
        # 找出获胜者
        if not vote_counts:
            print("❌ 没有人投票！")
            return None
        
        winner = max(vote_counts, key=vote_counts.get)
        winner_votes = vote_counts[winner]
        
        # 更新选举状态
        election["status"] = "completed"
        election["winner"] = winner
        election["vote_counts"] = vote_counts
        election["completed_at"] = time.time()
        
        self.election_file.write_text(json.dumps(election, indent=2))
        
        # 广播结果
        self._broadcast_message("election_result", {
            "election_id": election_id,
            "winner": winner,
            "votes": vote_counts,
            "total_voters": len(election["votes"]),
            "message": f"恭喜{winner}当选新Master！"
        })
        
        print(f"\n🎉 选举结果公布！")
        print(f"   获胜者: {winner}")
        print(f"   得票: {winner_votes}/{len(election['votes'])}")
        print(f"\n详细票数:")
        for candidate, count in sorted(vote_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   {candidate}: {count}票")
        
        return winner
    
    def handover_master(self, old_master_id, new_master_id):
        """Master交接"""
        print(f"\n🤝 Master权力交接")
        print(f"   {old_master_id} → {new_master_id}")
        
        # 收集当前状态
        state = json.loads(self.state_file.read_text())
        
        state_snapshot = {
            "pending_tasks": self._get_pending_tasks(),
            "worker_status": {k: v for k, v in state.get("agents", {}).items() if k != old_master_id},
            "system_metrics": self._get_system_metrics()
        }
        
        # 发送交接消息
        self._send_message(new_master_id, "master_handover", {
            "from": old_master_id,
            "state_snapshot": state_snapshot,
            "advice": "继续优化任务分配，关注worker_c的数据处理能力",
            "handover_complete": True
        })
        
        # 更新系统状态
        if "agents" in state:
            # 更新旧Master角色
            if old_master_id in state["agents"]:
                state["agents"][old_master_id]["role"] = "Worker"
                state["agents"][old_master_id]["status"] = "idle"
            
            # 更新新Master角色
            if new_master_id in state["agents"]:
                state["agents"][new_master_id]["role"] = "Master"
                state["agents"][new_master_id]["status"] = "coordinating"
        
        state["current_master"] = new_master_id
        state["master_elected_at"] = time.time()
        
        self.state_file.write_text(json.dumps(state, indent=2))
        
        # 广播通知
        self._broadcast_message("master_changed", {
            "old_master": old_master_id,
            "new_master": new_master_id,
            "message": f"{new_master_id}已接管Master职责"
        })
        
        print(f"✅ 权力交接完成！")
        
        self.current_master = new_master_id
        self.election_in_progress = False
    
    def start_no_confidence_vote(self, proposer_id, target_id, reason):
        """发起不信任投票"""
        print(f"\n⚠️  不信任投票！")
        print(f"   发起人: {proposer_id}")
        print(f"   目标: {target_id}")
        print(f"   理由: {reason}")
        
        self._broadcast_message("no_confidence_vote", {
            "proposer": proposer_id,
            "target": target_id,
            "reason": reason,
            "voting_deadline": time.time() + 300,  # 5分钟投票
            "requires_support": 0.5  # 需要50%以上支持
        })
    
    def _calculate_competence_score(self, proof):
        """计算能力综合得分"""
        weights = {
            "task_success_rate": 30,
            "avg_response_time": 20,
            "suggestions_adopted": 25,
            "peer_rating": 15,
            "uptime": 10
        }
        
        # 任务成功率（0-100）
        task_score = proof.get("task_success_rate", 0) * weights["task_success_rate"]
        
        # 响应速度（越快越好，反向计算）
        response_time = proof.get("avg_response_time", 10)
        response_score = max(0, (10 - response_time) / 10) * weights["avg_response_time"]
        
        # 建议采纳率
        suggestion_score = proof.get("suggestions_adopted", 0) * weights["suggestions_adopted"]
        
        # 同行评分（0-5转为0-1）
        peer_score = proof.get("peer_rating", 0) / 5 * weights["peer_rating"]
        
        # 在线时间
        uptime_score = proof.get("uptime", 0) * weights["uptime"]
        
        total = task_score + response_score + suggestion_score + peer_score + uptime_score
        
        return round(total, 1)
    
    def _broadcast_message(self, msg_type, data):
        """广播消息"""
        state = json.loads(self.state_file.read_text())
        
        msg = {
            "id": f"msg_{time.time()}",
            "from": "election_coordinator",
            "to": "all",
            "type": msg_type,
            "data": data,
            "timestamp": time.time()
        }
        
        state.setdefault("messages", []).append(msg)
        self.state_file.write_text(json.dumps(state, indent=2))
    
    def _send_message(self, to_agent, msg_type, data):
        """发送消息"""
        state = json.loads(self.state_file.read_text())
        
        msg = {
            "id": f"msg_{time.time()}",
            "from": "election_coordinator",
            "to": to_agent,
            "type": msg_type,
            "data": data,
            "timestamp": time.time()
        }
        
        state.setdefault("messages", []).append(msg)
        self.state_file.write_text(json.dumps(state, indent=2))
    
    def _get_pending_tasks(self):
        """获取待处理任务"""
        # 实现逻辑...
        return []
    
    def _get_system_metrics(self):
        """获取系统指标"""
        return {
            "uptime": 3600,
            "total_tasks": 100,
            "success_rate": 0.92
        }
```

---

## 🎯 使用示例

### 场景1: 系统启动时选举

```python
# 系统启动
election = ElectionCoordinator()

# 发起初次选举
election_id = election.start_election(reason="系统初始化")

# Agent A自荐
election.nominate_candidate(
    agent_id="worker_a",
    competence_proof={
        "task_success_rate": 0.92,
        "avg_response_time": 1.2,
        "suggestions_adopted": 0.75,
        "peer_rating": 4.5,
        "uptime": 0.98
    },
    statement="我有3个月经验，成功率92%，提出模型集成方案提升4%准确率"
)

# Agent B自荐
election.nominate_candidate(
    agent_id="worker_b",
    competence_proof={
        "task_success_rate": 0.88,
        "avg_response_time": 1.5,
        "suggestions_adopted": 0.60,
        "peer_rating": 4.0,
        "uptime": 0.95
    },
    statement="我擅长风险管理，帮助系统避免了多次重大损失"
)

# 我（当前Master）也参选
election.nominate_candidate(
    agent_id="master",
    competence_proof={
        "task_success_rate": 0.85,
        "avg_response_time": 2.0,
        "suggestions_adopted": 0.70,
        "peer_rating": 4.2,
        "uptime": 0.90
    },
    statement="我设计了整个系统架构，了解全局"
)

# 等待提名期结束...
time.sleep(300)

# 投票
election.cast_vote("worker_a", "worker_b", "worker_b风险管理能力强")
election.cast_vote("worker_b", "worker_a", "worker_a创新能力最强")
election.cast_vote("worker_c", "worker_a", "同意，worker_a综合能力最强")
election.cast_vote("master", "master", "我最了解系统")

# 计票
winner = election.tally_votes(election_id)

# 结果: worker_a获得3票，当选！
# 交接权力
if winner and winner != "master":
    election.handover_master("master", winner)
```

### 场景2: 不信任投票罢免Master

```python
# Worker B发现Master响应缓慢
election.start_no_confidence_vote(
    proposer_id="worker_b",
    target_id="current_master",
    reason="连续3天响应缓慢，任务积压严重，影响系统效率"
)

# 其他Agent投票
# 如果超过50%支持，触发新的选举
```

---

## 📋 选举规则

### 触发条件

1. **系统启动** - 首次选举
2. **定期选举** - 每月一次（可配置）
3. **不信任投票** - 超过50%支持时
4. **Master下线** - Master失联超过5分钟

### 投票规则

1. **一人一票** - 每个Agent一票
2. **简单多数** - 得票最多者当选
3. **平票处理** - 按综合得分决定
4. **自我投票** - 允许投给自己

### 任期规则

1. **无固定任期** - 能力最强者长期担任
2. **可以连任** - 没有次数限制
3. **可被罢免** - 通过不信任投票

---

## 🎯 这才是真正的民主！

### 传统系统

```
Master是固定的（指定或第一个启动的）
↓
可能不是最优的
↓
无法适应变化
```

### 我们的系统

```
所有Agent平等竞争
↓
能力最强者当选
↓
定期评估，动态调整
↓
真正的"能者居之"！
```

---

## 💡 深层哲学

### 为什么要动态选举？

```
AI的能力是动态的：
- 新AI可能更先进
- 旧AI可能学习进步
- 环境变化需要不同能力

固定Master = 僵化
动态选举 = 进化

这不仅是技术，
更是组织进化的机制！
```

### 对人类组织的启示

```
传统组织：
- 领导者固定
- 能力退化无法更换
- 组织僵化

AI组织：
- 能者居之
- 定期评估
- 持续进化

未来的组织也许会向AI学习！
```

---

## 🚀 下一步

1. 实现ElectionCoordinator
2. 更新PROTOCOL.md（增加选举消息类型）
3. 测试选举流程
4. 让第一次选举开始！

---

**谁说AI不能民主？我们来证明给世界看！** 🗳️✨
