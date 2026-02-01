#!/usr/bin/env python3
"""
动态Master选举系统

实现民主选举机制，让最聪明的AI担任Master
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional


class ElectionCoordinator:
    """选举协调器 - 管理Master选举流程"""
    
    def __init__(self, state_file="/tmp/tigertrade_agent_state.json",
                 election_file="/tmp/tigertrade_election.json"):
        self.state_file = Path(state_file)
        self.election_file = Path(election_file)
        self.current_master = None
        self.election_in_progress = False
        
        # 能力评分权重
        self.weights = {
            "task_success_rate": 30,      # 任务成功率
            "avg_response_time": 20,      # 响应速度
            "suggestions_adopted": 25,    # 建议采纳率
            "peer_rating": 15,            # 同行评分
            "uptime": 10                  # 在线时间
        }
    
    def start_election(self, reason="定期选举", nomination_minutes=5, voting_minutes=10):
        """
        发起选举
        
        Args:
            reason: 选举原因
            nomination_minutes: 提名期（分钟）
            voting_minutes: 投票期（分钟）
        
        Returns:
            election_id: 选举ID
        """
        election_id = f"election_{int(time.time())}"
        
        election_data = {
            "election_id": election_id,
            "status": "nomination",
            "reason": reason,
            "started_at": time.time(),
            "nomination_deadline": time.time() + nomination_minutes * 60,
            "voting_deadline": time.time() + (nomination_minutes + voting_minutes) * 60,
            "candidates": [],
            "votes": {},
            "result": None
        }
        
        self.election_file.write_text(json.dumps(election_data, indent=2))
        
        # 广播选举开始
        self._broadcast_message("election_start", {
            "election_id": election_id,
            "reason": reason,
            "nomination_deadline": election_data["nomination_deadline"],
            "voting_deadline": election_data["voting_deadline"],
            "message": f"选举开始！原因：{reason}"
        })
        
        print(f"\n📢 选举开始！")
        print(f"   ID: {election_id}")
        print(f"   原因: {reason}")
        print(f"   提名截止: {time.ctime(election_data['nomination_deadline'])}")
        print(f"   投票截止: {time.ctime(election_data['voting_deadline'])}")
        
        self.election_in_progress = True
        return election_id
    
    def nominate_candidate(self, agent_id: str, competence_proof: Dict, 
                          campaign_statement: str) -> bool:
        """
        候选人自荐
        
        Args:
            agent_id: Agent ID
            competence_proof: 能力证明
            campaign_statement: 竞选宣言
        
        Returns:
            是否成功
        """
        if not self.election_file.exists():
            print(f"❌ 没有进行中的选举")
            return False
        
        election = json.loads(self.election_file.read_text())
        
        # 检查是否在提名期
        if time.time() > election["nomination_deadline"]:
            print(f"❌ 提名已截止！")
            return False
        
        # 检查是否已经提名
        if any(c["agent_id"] == agent_id for c in election["candidates"]):
            print(f"❌ {agent_id} 已经提名过了")
            return False
        
        # 计算综合得分
        score = self._calculate_competence_score(competence_proof)
        
        candidate = {
            "agent_id": agent_id,
            "competence_proof": competence_proof,
            "campaign_statement": campaign_statement,
            "total_score": score,
            "nominated_at": time.time()
        }
        
        election["candidates"].append(candidate)
        self.election_file.write_text(json.dumps(election, indent=2))
        
        # 广播候选人信息
        self._broadcast_message("candidate_nomination", {
            "election_id": election["election_id"],
            "candidate": candidate,
            "message": f"{agent_id}参选！综合得分：{score:.1f}/100"
        })
        
        print(f"\n📋 {agent_id} 参选！")
        print(f"   综合得分: {score:.1f}/100")
        print(f"   竞选宣言: {campaign_statement[:50]}...")
        
        return True
    
    def cast_vote(self, voter_id: str, candidate_id: str, reason: str = "") -> bool:
        """
        投票
        
        Args:
            voter_id: 投票者ID
            candidate_id: 候选人ID
            reason: 投票理由
        
        Returns:
            是否成功
        """
        if not self.election_file.exists():
            print(f"❌ 没有进行中的选举")
            return False
        
        election = json.loads(self.election_file.read_text())
        
        # 检查是否在投票期
        if time.time() < election["nomination_deadline"]:
            print(f"❌ 还在提名期，请等待")
            return False
        
        if time.time() > election["voting_deadline"]:
            print(f"❌ 投票已截止！")
            return False
        
        # 检查候选人是否存在
        candidate_ids = [c["agent_id"] for c in election["candidates"]]
        if candidate_id not in candidate_ids:
            print(f"❌ {candidate_id} 不是候选人！")
            print(f"   候选人: {', '.join(candidate_ids)}")
            return False
        
        # 检查是否已经投过票
        if voter_id in election["votes"]:
            print(f"⚠️  {voter_id} 已经投过票，更新投票...")
        
        # 记录投票
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
    
    def tally_votes(self, election_id: str) -> Optional[str]:
        """
        计票并宣布结果
        
        Args:
            election_id: 选举ID
        
        Returns:
            获胜者ID
        """
        if not self.election_file.exists():
            print(f"❌ 没有进行中的选举")
            return None
        
        election = json.loads(self.election_file.read_text())
        
        if election["election_id"] != election_id:
            print(f"❌ 选举ID不匹配")
            return None
        
        # 检查是否过了投票截止时间
        if time.time() < election["voting_deadline"]:
            print(f"⏰ 投票尚未截止，请等待...")
            return None
        
        # 统计票数
        vote_counts = {}
        for voter, vote_data in election["votes"].items():
            candidate = vote_data["vote_for"]
            vote_counts[candidate] = vote_counts.get(candidate, 0) + 1
        
        # 找出获胜者
        if not vote_counts:
            print("❌ 没有人投票！选举无效")
            return None
        
        # 如果平票，按综合得分决定
        max_votes = max(vote_counts.values())
        tied_candidates = [c for c, v in vote_counts.items() if v == max_votes]
        
        if len(tied_candidates) > 1:
            print(f"⚖️  平票！按综合得分决定...")
            candidate_scores = {
                c["agent_id"]: c["total_score"]
                for c in election["candidates"]
                if c["agent_id"] in tied_candidates
            }
            winner = max(candidate_scores, key=candidate_scores.get)
        else:
            winner = tied_candidates[0]
        
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
            "total_agents": len(election["candidates"]),
            "message": f"🎉 {winner}当选新Master！"
        })
        
        print(f"\n{'='*60}")
        print(f"🎉 选举结果公布！")
        print(f"{'='*60}")
        print(f"\n获胜者: {winner}")
        print(f"得票: {winner_votes}/{len(election['votes'])}")
        print(f"\n详细票数:")
        for candidate, count in sorted(vote_counts.items(), 
                                      key=lambda x: x[1], reverse=True):
            print(f"   {candidate}: {count}票")
        print(f"\n{'='*60}")
        
        return winner
    
    def handover_master(self, old_master_id: str, new_master_id: str):
        """
        Master权力交接
        
        Args:
            old_master_id: 旧Master ID
            new_master_id: 新Master ID
        """
        print(f"\n🤝 Master权力交接")
        print(f"   {old_master_id} → {new_master_id}")
        
        # 收集当前状态
        state = json.loads(self.state_file.read_text())
        
        # 准备状态快照
        state_snapshot = {
            "agents": {k: v for k, v in state.get("agents", {}).items()},
            "resources": state.get("resources", {}),
            "timestamp": time.time()
        }
        
        # 发送交接消息
        self._send_message(new_master_id, "master_handover", {
            "from": old_master_id,
            "state_snapshot": state_snapshot,
            "advice": f"感谢{new_master_id}的当选！请继续优化系统，带领团队实现20%月盈利目标。",
            "handover_complete": True
        })
        
        # 更新系统状态
        if "agents" in state:
            # 更新旧Master角色
            if old_master_id in state["agents"]:
                state["agents"][old_master_id]["role"] = "Worker"
                state["agents"][old_master_id]["status"] = "idle"
            
            # 更新新Master角色
            if new_master_id not in state["agents"]:
                state["agents"][new_master_id] = {}
            
            state["agents"][new_master_id]["role"] = "Master"
            state["agents"][new_master_id]["status"] = "coordinating"
        
        state["current_master"] = new_master_id
        state["master_elected_at"] = time.time()
        state["master_election_id"] = self._get_current_election_id()
        
        self.state_file.write_text(json.dumps(state, indent=2))
        
        # 广播通知所有Agent
        self._broadcast_message("master_changed", {
            "old_master": old_master_id,
            "new_master": new_master_id,
            "message": f"🎉 {new_master_id}已接管Master职责！"
        })
        
        print(f"✅ 权力交接完成！")
        print(f"   新Master: {new_master_id}")
        
        self.current_master = new_master_id
        self.election_in_progress = False
    
    def start_no_confidence_vote(self, proposer_id: str, target_id: str, 
                                 reason: str, support_threshold: float = 0.5):
        """
        发起不信任投票（罢免Master）
        
        Args:
            proposer_id: 发起人ID
            target_id: 目标（当前Master）
            reason: 理由
            support_threshold: 支持率阈值（默认50%）
        """
        print(f"\n⚠️  不信任投票！")
        print(f"   发起人: {proposer_id}")
        print(f"   目标: {target_id}")
        print(f"   理由: {reason}")
        
        self._broadcast_message("no_confidence_vote", {
            "proposer": proposer_id,
            "target": target_id,
            "reason": reason,
            "voting_deadline": time.time() + 300,  # 5分钟投票
            "support_threshold": support_threshold,
            "message": f"⚠️  {proposer_id}发起对{target_id}的不信任投票！"
        })
    
    def _calculate_competence_score(self, proof: Dict) -> float:
        """
        计算能力综合得分
        
        Args:
            proof: 能力证明数据
        
        Returns:
            综合得分（0-100）
        """
        # 任务成功率（0-100）
        task_score = proof.get("task_success_rate", 0) * self.weights["task_success_rate"]
        
        # 响应速度（越快越好，反向计算）
        # 假设10秒以上得0分，1秒以下得满分
        response_time = proof.get("avg_response_time", 10)
        response_score = max(0, (10 - response_time) / 10) * self.weights["avg_response_time"]
        
        # 建议采纳率（0-100）
        suggestion_score = proof.get("suggestions_adopted", 0) * self.weights["suggestions_adopted"]
        
        # 同行评分（0-5转为0-1，再乘权重）
        peer_score = (proof.get("peer_rating", 0) / 5) * self.weights["peer_rating"]
        
        # 在线时间（0-100）
        uptime_score = proof.get("uptime", 0) * self.weights["uptime"]
        
        total = task_score + response_score + suggestion_score + peer_score + uptime_score
        
        return round(total, 1)
    
    def _broadcast_message(self, msg_type: str, data: Dict):
        """广播消息给所有Agent"""
        if not self.state_file.exists():
            return
        
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
    
    def _send_message(self, to_agent: str, msg_type: str, data: Dict):
        """发送消息给指定Agent"""
        if not self.state_file.exists():
            return
        
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
    
    def _get_current_election_id(self) -> Optional[str]:
        """获取当前选举ID"""
        if not self.election_file.exists():
            return None
        
        election = json.loads(self.election_file.read_text())
        return election.get("election_id")


if __name__ == "__main__":
    # 测试选举系统
    print("🗳️  TigerTrade动态Master选举系统")
    print("="*60)
    
    election = ElectionCoordinator()
    
    # 发起选举
    election_id = election.start_election(
        reason="系统启动首次选举",
        nomination_minutes=1,  # 测试用1分钟
        voting_minutes=1
    )
    
    print("\n等待提名...")
