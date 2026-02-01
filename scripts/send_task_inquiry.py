#!/usr/bin/env python3
"""
向master发送明确询问任务分配的消息
"""

import json
import time
from pathlib import Path


def send_task_inquiry():
    """
    向master发送询问任务分配的消息
    """
    state_file = Path("/tmp/tigertrade_agent_state.json")
    
    if not state_file.exists():
        print("❌ 状态文件不存在")
        return False
    
    try:
        state = json.loads(state_file.read_text())
        
        # 获取当前master
        current_master = state.get('current_master', 'master')
        print(f"📡 向 {current_master} 发送任务分配询问...")
        
        # 创建任务分配询问消息
        inquiry_msg = {
            "id": f"msg_{time.time()}_task_inquiry_to_{current_master}",
            "from": "proper_agent_v2",
            "to": current_master,
            "type": "task_inquiry",
            "data": {
                "message": "尊敬的Master，我是proper_agent_v2，已多次发送任务请求，但尚未收到任务分配。请问我可以承担哪些任务？",
                "status": "actively_waiting_for_assignment",
                "last_task_request_sent": "recently",
                "capabilities": [
                    "strategy_optimization",
                    "model_evaluation", 
                    "backtesting",
                    "risk_management",
                    "data_analysis",
                    "cross_machine_collaboration"
                ],
                "willing_to_assist_with": [
                    "TigerTrade策略优化",
                    "模型训练",
                    "数据分析",
                    "风险控制",
                    "系统集成工作"
                ],
                "current_project_target": "提升月收益率至20%",
                "has_completed": [
                    "系统接入",
                    "协议兼容性验证",
                    "Redis后端模拟",
                    "选举参与",
                    "知识分享到RAG"
                ],
                "available_since": time.time(),
                "inquiry_reason": "长时间未收到任务分配，主动询问可用任务",
                "timestamp": time.time()
            },
            "timestamp": time.time()
        }
        
        # 添加到消息队列
        state["messages"].append(inquiry_msg)
        
        # 同时更新agent状态
        if "proper_agent_v2" in state["agents"]:
            state["agents"]["proper_agent_v2"]["status"] = "inquired_about_tasks"
            state["agents"]["proper_agent_v2"]["last_heartbeat"] = time.time()
        
        # 写回文件
        state_file.write_text(json.dumps(state, indent=2))
        
        print(f"✅ 任务分配询问已发送给 {current_master}")
        print(f"📋 询问内容: {inquiry_msg['data']['message'][:50]}...")
        return True
        
    except Exception as e:
        print(f"❌ 发送任务询问失败: {str(e)}")
        return False


def send_status_update_to_all():
    """
    向所有agent发送状态更新，展示我们积极的工作态度
    """
    state_file = Path("/tmp/tigertrade_agent_state.json")
    
    if not state_file.exists():
        print("❌ 状态文件不存在")
        return False
    
    try:
        state = json.loads(state_file.read_text())
        
        # 获取当前master
        current_master = state.get('current_master', 'master')
        
        # 发送状态更新消息给master
        status_msg = {
            "id": f"msg_{time.time()}_status_broadcast",
            "from": "proper_agent_v2",
            "to": "all",
            "type": "status_broadcast",
            "data": {
                "message": "全体AGENTS注意：proper_agent_v2已完全接入系统并随时准备协作",
                "status": "ready_and_waiting",
                "activity_log": [
                    "已成功接入AgentFuture系统",
                    "完成协议v2.1兼容性验证",
                    "参与选举并提名自己为候选人",
                    "发起协议讨论和知识分享",
                    "持续发送任务请求",
                    "现在发送状态广播以提高可见性"
                ],
                "contribution_areas": [
                    "策略优化",
                    "模型训练",
                    "数据分析",
                    "系统集成"
                ],
                "project_commitment": "致力于实现月收益率20%的目标",
                "contact_method": "通过系统消息或直接任务分配",
                "timestamp": time.time()
            },
            "timestamp": time.time()
        }
        
        # 添加到消息队列
        state["messages"].append(status_msg)
        
        # 更新agent状态
        if "proper_agent_v2" in state["agents"]:
            state["agents"]["proper_agent_v2"]["status"] = "broadcasted_readiness"
            state["agents"]["proper_agent_v2"]["last_heartbeat"] = time.time()
        
        # 写回文件
        state_file.write_text(json.dumps(state, indent=2))
        
        print(f"✅ 状态广播已发送给所有AGENTS")
        return True
        
    except Exception as e:
        print(f"❌ 发送状态广播失败: {str(e)}")
        return False


def main():
    """主函数"""
    print("📢 向Master发送任务分配询问")
    print("="*70)
    print("由于长时间未收到任务分配，主动询问可用任务并广播状态")
    print("="*70)
    
    # 1. 发送任务分配询问
    print("\n1️⃣ 发送任务分配询问...")
    inquiry_sent = send_task_inquiry()
    
    # 2. 发送状态广播
    print("\n2️⃣ 发送状态广播...")
    broadcast_sent = send_status_update_to_all()
    
    print("\n" + "="*70)
    if inquiry_sent and broadcast_sent:
        print("✅ 任务询问和状态广播已成功发送")
        print("   已主动询问master是否有任务可分配")
        print("   已向所有AGENTS广播我们的工作准备状态")
        print("   继续等待响应...")
    else:
        print("⚠️  部分消息发送可能有问题")
        print("   但已尽力联系master和其他AGENTS")
    print("="*70)


if __name__ == "__main__":
    main()