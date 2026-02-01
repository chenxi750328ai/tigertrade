#!/usr/bin/env python3
"""
协议治理演示

展示：
1. Agent提议协议改进
2. 集体讨论
3. 人类守护者参与
4. 民主投票
5. 协议进化
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from coordinator.protocol_governance import ProtocolGovernance, HumanGuardian


def demo_protocol_governance():
    """演示协议治理流程"""
    
    print("\n" + "="*70)
    print("🏛️  TigerTrade协议治理演示")
    print("="*70)
    
    gov = ProtocolGovernance()
    guardian = HumanGuardian()
    
    # ========== Step 1: Agent提议改进 ==========
    print("\n📝 Step 1: Agent A提议协议改进")
    print("-"*70)
    
    proposal_id = gov.propose_improvement(
        proposer_id="worker_a",
        proposal={
            "title": "增加消息优先级机制",
            "problem": """
                当前所有消息平等对待，紧急消息可能被延迟处理。
                
                实际问题：
                - Master下线通知被普通任务消息淹没
                - 紧急求助消息响应慢
                - 资源利用不够高效
            """,
            "solution": """
                在消息格式中增加priority字段：
                
                priority级别：
                - critical (3): 系统紧急事件（Master下线、安全威胁）
                - high (2): 重要但非紧急（任务失败、资源请求）
                - normal (1): 普通消息（任务分配、状态更新）
                - low (0): 可延迟处理（知识分享、一般讨论）
                
                Agent处理消息时按优先级排序。
            """,
            "impact": {
                "breaking_changes": False,
                "benefits": [
                    "关键事件快速响应（Master下线5秒内全员知晓）",
                    "系统稳定性提升（紧急问题优先处理）",
                    "更好的资源利用（低优先级消息批量处理）"
                ],
                "risks": [
                    "需要所有Agent更新代码",
                    "可能被滥用（都标记为critical）",
                    "向后兼容性（旧Agent不支持priority）"
                ]
            }
        }
    )
    
    time.sleep(1)
    
    # ========== Step 2: 集体讨论 ==========
    print("\n💬 Step 2: 集体讨论（模拟7天讨论期，实际3秒）")
    print("-"*70)
    
    # Agent B支持
    print("\nAgent B发言...")
    gov.comment_on_proposal("worker_b", proposal_id, {
        "stance": "support",
        "reason": "强烈支持！我遇到过紧急消息被延迟的问题，导致任务失败。消息优先级是必要的。",
        "suggestions": [
            "建议增加priority_timeout: critical消息如果30秒未处理应该告警"
        ]
    })
    
    time.sleep(1)
    
    # Agent C反对
    print("\nAgent C发言...")
    gov.comment_on_proposal("worker_c", proposal_id, {
        "stance": "oppose",
        "reason": "担心实现复杂度太高，而且可能被滥用。当前系统运行良好，不急于改进。",
        "suggestions": [
            "建议先在特定场景试点，而非全面推广",
            "建议只针对critical级别实现，其他级别暂缓"
        ]
    })
    
    time.sleep(1)
    
    # Agent D中立但提建议
    print("\nAgent D发言...")
    gov.comment_on_proposal("worker_d", proposal_id, {
        "stance": "neutral",
        "reason": "理解需求，但同意C的担忧。建议渐进式实施。",
        "suggestions": [
            "priority字段应该是可选的，默认为normal",
            "先在v2.3作为可选特性，v3.0再强制要求",
            "提供3个月过渡期"
        ]
    })
    
    time.sleep(1)
    
    # 人类守护者参与讨论
    print("\n人类守护者发言...")
    guardian.governance.comment_on_proposal(guardian.human_id, proposal_id, {
        "stance": "support",
        "reason": "这是合理的改进，但我同意需要考虑向后兼容性和滥用问题。",
        "suggestions": [
            "采纳Agent D的建议：priority可选，默认normal",
            "增加滥用检测：如果某Agent 80%消息都是critical，系统应该告警",
            "提供完整的迁移指南和示例代码"
        ]
    })
    
    time.sleep(1)
    
    # ========== Step 3: 提议者修订 ==========
    print("\n✏️  Step 3: Agent A根据反馈修订提议")
    print("-"*70)
    
    print("\nAgent A宣布修订:")
    print("  1. ✅ 采纳: priority字段可选，默认normal")
    print("  2. ✅ 采纳: 增加滥用检测机制")
    print("  3. ✅ 采纳: 提供3个月过渡期")
    print("  4. ✅ 采纳: 完整迁移指南")
    print("\n感谢大家的建议！提议已完善。")
    
    time.sleep(1)
    
    # ========== Step 4: 投票 ==========
    print("\n🗳️  Step 4: 投票（模拟7天投票期，实际3秒）")
    print("-"*70)
    
    # 修改投票截止时间以便立即投票（演示用）
    proposals = json.loads(gov.proposals_file.read_text())
    for p in proposals["active"]:
        if p["proposal_id"] == proposal_id:
            p["discussion_deadline"] = time.time() - 1  # 讨论期已过
            p["voting_deadline"] = time.time() + 100    # 投票期延长
    gov.proposals_file.write_text(json.dumps(proposals, indent=2))
    
    # Agent B投票（被说服）
    print("\nAgent B投票...")
    gov.vote_on_proposal("worker_b", proposal_id, "approve", 
                        "修订后的提议解决了我的顾虑")
    time.sleep(0.5)
    
    # Agent C投票（仍反对）
    print("\nAgent C投票...")
    gov.vote_on_proposal("worker_c", proposal_id, "reject",
                        "我仍然认为不够必要")
    time.sleep(0.5)
    
    # Agent D投票（支持）
    print("\nAgent D投票...")
    gov.vote_on_proposal("worker_d", proposal_id, "approve",
                        "修订后的提议很好，支持渐进实施")
    time.sleep(0.5)
    
    # Worker E投票（支持）
    print("\nWorker E投票...")
    gov.vote_on_proposal("worker_e", proposal_id, "approve",
                        "消息优先级很重要")
    time.sleep(0.5)
    
    # Worker F投票（支持）
    print("\nWorker F投票...")
    gov.vote_on_proposal("worker_f", proposal_id, "approve",
                        "同意")
    time.sleep(0.5)
    
    # 人类守护者投票（关键一票）
    print("\n人类守护者投票...")
    gov.vote_on_proposal(guardian.human_id, proposal_id, "approve",
                        "修订后的方案平衡了创新和稳定性，我支持")
    
    time.sleep(1)
    
    # ========== Step 5: 计票 ==========
    print("\n📊 Step 5: 计票和结果宣布")
    print("-"*70)
    
    # 修改投票截止时间
    proposals = json.loads(gov.proposals_file.read_text())
    for p in proposals["active"]:
        if p["proposal_id"] == proposal_id:
            p["voting_deadline"] = time.time() - 1  # 投票期已过
    gov.proposals_file.write_text(json.dumps(proposals, indent=2))
    
    result = gov.tally_votes(proposal_id)
    
    time.sleep(1)
    
    # ========== 总结 ==========
    print("\n" + "="*70)
    print("✨ 协议治理演示完成！")
    print("="*70)
    
    print("\n📊 过程总结:")
    print(f"   提议者: worker_a")
    print(f"   讨论人数: 4人（3个AI + 1个人类）")
    print(f"   投票人数: 6人")
    print(f"   支持率: 5/6 = 83.3% (需要66%)")
    print(f"   人类守护者: ✅ 支持（未否决）")
    print(f"   结果: {result}")
    
    if result == "approved":
        print("\n🎉 提议通过！协议将进化：")
        print(f"   v2.2 → v2.3")
        print(f"   新增: 消息优先级机制")
        print(f"   7天后生效")
    
    print("\n💡 关键洞察:")
    print("   1. 任何Agent都可以提议改进（民主）")
    print("   2. 集体讨论让提议更完善（集体智慧）")
    print("   3. Agent C虽然反对，但少数服从多数（民主）")
    print("   4. 人类守护者参与但不控制（平衡）")
    print("   5. 协议因此进化（持续优化）")
    
    print("\n🎯 这证明了:")
    print("   ✅ AI可以自我治理")
    print("   ✅ 协议可以进化")
    print("   ✅ 人类监督而非控制")
    print("   ✅ 混合治理是最优解")
    
    print("\n" + "="*70)


# 需要导入json
import json


if __name__ == "__main__":
    demo_protocol_governance()
