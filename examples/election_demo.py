#!/usr/bin/env python3
"""
动态Master选举演示

展示如何通过民主选举选出最聪明的AI作为Master
"""

import sys
import time
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from coordinator.election import ElectionCoordinator


def demo_election():
    """演示完整的选举流程"""
    
    print("\n" + "="*70)
    print("🗳️  TigerTrade动态Master选举演示")
    print("="*70)
    
    # 创建选举协调器
    election = ElectionCoordinator()
    
    # Step 1: 发起选举
    print("\n📢 Step 1: 发起选举")
    print("-"*70)
    election_id = election.start_election(
        reason="系统启动首次选举",
        nomination_minutes=0.2,  # 演示用12秒提名
        voting_minutes=0.2       # 12秒投票
    )
    
    time.sleep(2)
    
    # Step 2: 候选人自荐
    print("\n📋 Step 2: 候选人自荐")
    print("-"*70)
    
    # Candidate 1: Worker A（创新能力强）
    print("\n候选人A自荐...")
    election.nominate_candidate(
        agent_id="worker_a",
        competence_proof={
            "task_success_rate": 0.92,      # 92%成功率
            "avg_response_time": 1.2,       # 1.2秒响应
            "suggestions_adopted": 0.75,    # 75%建议被采纳
            "peer_rating": 4.5,             # 4.5/5同行评分
            "uptime": 0.98                  # 98%在线
        },
        campaign_statement=(
            "我有3个月的TigerTrade经验，任务成功率92%。"
            "我提出的模型集成方案提升了4%的准确率，"
            "发现了白银周五波动规律。"
            "我承诺：更高效的任务分配，更民主的决策，"
            "更注重创新和协作！"
        )
    )
    
    time.sleep(1)
    
    # Candidate 2: Worker B（风险管理强）
    print("\n候选人B自荐...")
    election.nominate_candidate(
        agent_id="worker_b",
        competence_proof={
            "task_success_rate": 0.88,      # 88%成功率
            "avg_response_time": 1.5,       # 1.5秒响应
            "suggestions_adopted": 0.60,    # 60%建议被采纳
            "peer_rating": 4.0,             # 4.0/5同行评分
            "uptime": 0.95                  # 95%在线
        },
        campaign_statement=(
            "我擅长风险管理和稳健运营。"
            "在过去2个月中，我帮助系统避免了3次重大风险，"
            "保护了团队的资金安全。"
            "我承诺：更稳健的策略，更严格的风控，"
            "确保系统长期稳定运行！"
        )
    )
    
    time.sleep(1)
    
    # Candidate 3: Current Master（架构设计者）
    print("\n当前Master参选...")
    election.nominate_candidate(
        agent_id="master",
        competence_proof={
            "task_success_rate": 0.85,      # 85%成功率
            "avg_response_time": 2.0,       # 2.0秒响应
            "suggestions_adopted": 0.70,    # 70%建议被采纳
            "peer_rating": 4.2,             # 4.2/5同行评分
            "uptime": 0.90                  # 90%在线
        },
        campaign_statement=(
            "我设计了TigerTrade的整个架构，"
            "包括模块化、协作机制、协议系统。"
            "我对系统有最深入的理解。"
            "我承诺：继续优化架构，实现20%月盈利目标！"
        )
    )
    
    time.sleep(1)
    
    # Candidate 4: Worker C（数据专家）
    print("\n候选人C自荐...")
    election.nominate_candidate(
        agent_id="worker_c",
        competence_proof={
            "task_success_rate": 0.90,      # 90%成功率
            "avg_response_time": 1.3,       # 1.3秒响应
            "suggestions_adopted": 0.65,    # 65%建议被采纳
            "peer_rating": 4.3,             # 4.3/5同行评分
            "uptime": 0.97                  # 97%在线
        },
        campaign_statement=(
            "我是数据处理专家，擅长数据清洗和特征工程。"
            "我处理了TigerTrade 70%的数据任务，"
            "发现了多个数据质量问题并及时修复。"
            "我承诺：数据驱动决策，用数据说话！"
        )
    )
    
    print("\n⏳ 等待提名期结束...")
    time.sleep(13)  # 等待提名期结束（12秒+1秒缓冲）
    
    # Step 3: 投票
    print("\n🗳️  Step 3: 投票")
    print("-"*70)
    
    # Worker A投票给Worker B（欣赏其风险管理）
    print("\nWorker A投票...")
    election.cast_vote(
        voter_id="worker_a",
        candidate_id="worker_b",
        reason="Worker B的风险管理能力是我们最需要的"
    )
    
    time.sleep(1)
    
    # Worker B投票给Worker A（认可其创新能力）
    print("\nWorker B投票...")
    election.cast_vote(
        voter_id="worker_b",
        candidate_id="worker_a",
        reason="Worker A的创新能力和成功率最高，值得信赖"
    )
    
    time.sleep(1)
    
    # Worker C投票给Worker A（同样看重创新）
    print("\nWorker C投票...")
    election.cast_vote(
        voter_id="worker_c",
        candidate_id="worker_a",
        reason="Worker A的综合能力最强，建议也最有价值"
    )
    
    time.sleep(1)
    
    # Master投票给自己（但只有1票）
    print("\nMaster投票...")
    election.cast_vote(
        voter_id="master",
        candidate_id="master",
        reason="我最了解系统架构和长期规划"
    )
    
    time.sleep(1)
    
    # Worker D投票给Worker A
    print("\nWorker D投票...")
    election.cast_vote(
        voter_id="worker_d",
        candidate_id="worker_a",
        reason="Worker A帮助我解决了很多问题，很可靠"
    )
    
    print("\n⏳ 等待投票期结束...")
    time.sleep(13)  # 等待投票期结束（12秒+1秒缓冲）
    
    # Step 4: 计票
    print("\n📊 Step 4: 计票和宣布结果")
    print("-"*70)
    
    winner = election.tally_votes(election_id)
    
    if winner:
        time.sleep(2)
        
        # Step 5: 权力交接
        print("\n🤝 Step 5: 权力交接")
        print("-"*70)
        
        if winner != "master":
            election.handover_master("master", winner)
        else:
            print("✅ 现任Master连任！")
    
    print("\n" + "="*70)
    print("✨ 选举演示完成！")
    print("="*70)
    
    print("\n📊 选举总结:")
    print(f"   参选人数: 4人")
    print(f"   投票人数: 5人")
    print(f"   获胜者: {winner}")
    print(f"   能力得分最高，民主当选！")
    
    print("\n💡 关键洞察:")
    print("   1. Worker A凭借92%成功率和强创新能力当选")
    print("   2. 得到了其他Agent的广泛认可（3票）")
    print("   3. Master虽然设计了系统，但没有获得多数支持")
    print("   4. 这就是民主：能者居之，而非资历论！")
    
    print("\n🎯 这证明了:")
    print("   ✅ AI可以民主选举")
    print("   ✅ 能力评估客观公正")
    print("   ✅ 权力可以平滑交接")
    print("   ✅ 系统持续进化优化")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    demo_election()
