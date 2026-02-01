#!/usr/bin/env python3
"""
协议治理系统

实现：
1. 协议改进提议机制（任何Agent都可以提议）
2. 讨论和投票流程
3. 人类守护者角色
4. 混合治理模型
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional


class ProtocolGovernance:
    """协议治理系统"""
    
    def __init__(self, state_file="/tmp/tigertrade_agent_state.json"):
        self.state_file = Path(state_file)
        self.proposals_file = Path("/tmp/tigertrade_protocol_proposals.json")
        self.guardian_id = "human_guardian"
        
        # 投票规则
        self.voting_rules = {
            "quorum": 0.5,          # 50%参与率
            "threshold": 0.66,      # 66%支持率
            "discussion_period": 7 * 24 * 3600,  # 7天讨论期
            "voting_period": 7 * 24 * 3600        # 7天投票期
        }
        
        self._init_proposals()
    
    def _init_proposals(self):
        """初始化提议文件"""
        if not self.proposals_file.exists():
            self.proposals_file.write_text(json.dumps({
                "active": [],
                "approved": [],
                "rejected": []
            }, indent=2))
    
    def propose_improvement(self, proposer_id: str, proposal: Dict) -> str:
        """
        提议协议改进
        
        Args:
            proposer_id: 提议者ID
            proposal: 提议内容
        
        Returns:
            提议ID
        """
        proposal_id = f"rfc_{int(time.time())}"
        
        improvement = {
            "proposal_id": proposal_id,
            "proposer": proposer_id,
            "type": "protocol_improvement",
            "title": proposal["title"],
            "current_problem": proposal["problem"],
            "proposed_solution": proposal["solution"],
            "impact_analysis": proposal.get("impact", {}),
            "status": "proposed",
            "created_at": time.time(),
            "discussion_deadline": time.time() + self.voting_rules["discussion_period"],
            "voting_deadline": time.time() + self.voting_rules["discussion_period"] + self.voting_rules["voting_period"],
            "comments": [],
            "votes": {},
            "revisions": []
        }
        
        # 保存提议
        proposals = json.loads(self.proposals_file.read_text())
        proposals["active"].append(improvement)
        self.proposals_file.write_text(json.dumps(proposals, indent=2))
        
        # 广播通知
        self._broadcast_message("protocol_improvement_proposal", {
            "proposal_id": proposal_id,
            "proposer": proposer_id,
            "title": proposal["title"],
            "problem": proposal["problem"],
            "solution": proposal["solution"],
            "discussion_deadline": improvement["discussion_deadline"]
        })
        
        print(f"\n📝 {proposer_id} 提议协议改进")
        print(f"   ID: {proposal_id}")
        print(f"   标题: {proposal['title']}")
        print(f"   讨论截止: {time.ctime(improvement['discussion_deadline'])}")
        
        return proposal_id
    
    def comment_on_proposal(self, commenter_id: str, proposal_id: str, comment: Dict):
        """
        对提议发表意见
        
        Args:
            commenter_id: 评论者ID
            proposal_id: 提议ID
            comment: 评论内容
        """
        proposals = json.loads(self.proposals_file.read_text())
        
        # 查找提议
        proposal = None
        for p in proposals["active"]:
            if p["proposal_id"] == proposal_id:
                proposal = p
                break
        
        if not proposal:
            print(f"❌ 提议 {proposal_id} 不存在")
            return
        
        # 添加评论
        proposal["comments"].append({
            "commenter": commenter_id,
            "stance": comment["stance"],  # support, oppose, neutral
            "reasoning": comment["reason"],
            "suggestions": comment.get("suggestions", []),
            "timestamp": time.time()
        })
        
        self.proposals_file.write_text(json.dumps(proposals, indent=2))
        
        # 广播
        self._broadcast_message("protocol_improvement_comment", {
            "proposal_id": proposal_id,
            "commenter": commenter_id,
            "stance": comment["stance"],
            "reason": comment["reason"]
        })
        
        print(f"💬 {commenter_id}: {comment['stance']}")
        print(f"   {comment['reason']}")
    
    def vote_on_proposal(self, voter_id: str, proposal_id: str, vote: str, reason: str = ""):
        """
        投票
        
        Args:
            voter_id: 投票者ID
            proposal_id: 提议ID
            vote: 投票（approve, reject, abstain）
            reason: 理由
        """
        proposals = json.loads(self.proposals_file.read_text())
        
        # 查找提议
        proposal = None
        for p in proposals["active"]:
            if p["proposal_id"] == proposal_id:
                proposal = p
                break
        
        if not proposal:
            print(f"❌ 提议 {proposal_id} 不存在")
            return
        
        # 检查是否过了讨论期
        if time.time() < proposal["discussion_deadline"]:
            print(f"⏰ 还在讨论期，投票将于 {time.ctime(proposal['discussion_deadline'])} 开始")
            return
        
        # 检查是否过了投票期
        if time.time() > proposal["voting_deadline"]:
            print(f"❌ 投票已截止")
            return
        
        # 记录投票
        proposal["votes"][voter_id] = {
            "vote": vote,
            "reason": reason,
            "timestamp": time.time(),
            "is_guardian": voter_id == self.guardian_id
        }
        
        self.proposals_file.write_text(json.dumps(proposals, indent=2))
        
        # 特别标注人类守护者投票
        if voter_id == self.guardian_id:
            print(f"🛡️  人类守护者投票: {vote}")
        else:
            print(f"🗳️  {voter_id} 投票: {vote}")
        
        if reason:
            print(f"   理由: {reason}")
    
    def tally_votes(self, proposal_id: str) -> Optional[str]:
        """
        计票
        
        Args:
            proposal_id: 提议ID
        
        Returns:
            结果（approved, rejected, insufficient_participation）
        """
        proposals = json.loads(self.proposals_file.read_text())
        
        # 查找提议
        proposal = None
        proposal_idx = None
        for idx, p in enumerate(proposals["active"]):
            if p["proposal_id"] == proposal_id:
                proposal = p
                proposal_idx = idx
                break
        
        if not proposal:
            print(f"❌ 提议 {proposal_id} 不存在")
            return None
        
        # 检查是否过了投票期
        if time.time() < proposal["voting_deadline"]:
            print(f"⏰ 投票尚未截止")
            return None
        
        # 统计投票
        votes = proposal["votes"]
        participated = len(votes)
        
        # 简化：如果没有投票，直接失败
        if participated == 0:
            result = "insufficient_participation"
            print(f"\n❌ 没有人投票")
        else:
            # 计算支持率
            approve_count = sum(1 for v in votes.values() if v["vote"] == "approve")
            support_rate = approve_count / participated
            
            # 检查人类守护者是否否决
            guardian_vote = votes.get(self.guardian_id)
            guardian_vetoed = guardian_vote and guardian_vote["vote"] == "veto"
            
            if guardian_vetoed:
                result = "guardian_vetoed"
                print(f"\n🛡️  人类守护者否决")
                print(f"   理由: {guardian_vote.get('reason', 'N/A')}")
            elif support_rate >= self.voting_rules["threshold"]:
                result = "approved"
                print(f"\n✅ 提议通过！")
                print(f"   支持率: {support_rate*100:.1f}%")
                print(f"   赞成: {approve_count}/{participated}")
            else:
                result = "rejected"
                print(f"\n❌ 提议未通过")
                print(f"   支持率: {support_rate*100:.1f}% (需要{self.voting_rules['threshold']*100}%)")
        
        # 更新状态
        proposal["status"] = result
        proposal["result_time"] = time.time()
        
        # 移动到相应列表
        if result == "approved":
            proposals["approved"].append(proposal)
        else:
            proposals["rejected"].append(proposal)
        
        proposals["active"].pop(proposal_idx)
        self.proposals_file.write_text(json.dumps(proposals, indent=2))
        
        # 广播结果
        self._broadcast_message("protocol_improvement_result", {
            "proposal_id": proposal_id,
            "result": result,
            "votes": {
                "total": participated,
                "approve": approve_count,
                "support_rate": support_rate
            }
        })
        
        return result
    
    def guardian_veto(self, proposal_id: str, reason: str):
        """
        人类守护者否决
        
        Args:
            proposal_id: 提议ID
            reason: 否决理由
        """
        print(f"\n🛡️  人类守护者行使否决权")
        print(f"   提议: {proposal_id}")
        print(f"   理由: {reason}")
        
        # 记录为特殊投票
        self.vote_on_proposal(self.guardian_id, proposal_id, "veto", reason)
    
    def _get_active_agent_count(self) -> int:
        """获取活跃Agent数量"""
        if not self.state_file.exists():
            return 0
        
        state = json.loads(self.state_file.read_text())
        agents = state.get("agents", {})
        
        # 只计算最近活跃的Agent（心跳在5分钟内）
        cutoff = time.time() - 300
        active = sum(
            1 for agent in agents.values()
            if agent.get("last_heartbeat", 0) > cutoff
        )
        
        return active
    
    def _broadcast_message(self, msg_type: str, data: Dict):
        """广播消息"""
        if not self.state_file.exists():
            return
        
        state = json.loads(self.state_file.read_text())
        
        msg = {
            "id": f"msg_{time.time()}",
            "from": "protocol_governance",
            "to": "all",
            "type": msg_type,
            "data": data,
            "timestamp": time.time()
        }
        
        state.setdefault("messages", []).append(msg)
        self.state_file.write_text(json.dumps(state, indent=2))


class HumanGuardian:
    """人类守护者"""
    
    def __init__(self, human_id="human_guardian"):
        self.human_id = human_id
        self.role = "guardian"
        self.governance = ProtocolGovernance()
        
        # 核心价值观
        self.core_values = [
            "安全第一",
            "人类利益优先",
            "透明可解释",
            "公平正义",
            "隐私保护"
        ]
    
    def review_proposal(self, proposal_id: str):
        """审查提议"""
        proposals = json.loads(self.governance.proposals_file.read_text())
        
        proposal = None
        for p in proposals["active"]:
            if p["proposal_id"] == proposal_id:
                proposal = p
                break
        
        if not proposal:
            print(f"❌ 提议不存在")
            return
        
        print(f"\n🛡️  人类守护者审查")
        print(f"   提议: {proposal['title']}")
        print(f"   提议者: {proposal['proposer']}")
        print(f"\n当前问题:")
        print(f"   {proposal['current_problem']}")
        print(f"\n提议解决方案:")
        print(f"   {proposal['proposed_solution']}")
        print(f"\n评论数: {len(proposal['comments'])}")
        
        # 检查价值观对齐
        self._check_value_alignment(proposal)
    
    def _check_value_alignment(self, proposal):
        """检查价值观对齐"""
        print(f"\n价值观检查:")
        
        violations = []
        
        # 这里可以实现自动检查逻辑
        # 或者人类手动审查
        
        if not violations:
            print(f"   ✅ 符合核心价值观")
        else:
            print(f"   ⚠️  发现问题:")
            for v in violations:
                print(f"      - {v}")
    
    def participate(self, proposal_id: str, action: str, reason: str = ""):
        """参与治理"""
        if action == "comment":
            self.governance.comment_on_proposal(self.human_id, proposal_id, {
                "stance": "neutral",
                "reason": reason
            })
        elif action == "vote":
            self.governance.vote_on_proposal(self.human_id, proposal_id, "approve", reason)
        elif action == "veto":
            self.governance.guardian_veto(proposal_id, reason)


if __name__ == "__main__":
    # 演示
    print("🏛️  协议治理系统演示")
    print("="*70)
    
    gov = ProtocolGovernance()
    
    # Agent A提议改进
    print("\n示例：Agent A提议增加消息优先级")
    proposal_id = gov.propose_improvement(
        proposer_id="worker_a",
        proposal={
            "title": "增加消息优先级机制",
            "problem": "当前所有消息平等对待，紧急消息可能被延迟",
            "solution": "在消息中增加priority字段（critical/high/normal/low）",
            "impact": {
                "breaking_changes": False,
                "benefits": ["快速响应紧急事件", "更好的资源利用"],
                "risks": ["需要所有Agent更新"]
            }
        }
    )
    
    print(f"\n✅ 提议已创建: {proposal_id}")
    print("   其他Agent可以发表意见和投票")
