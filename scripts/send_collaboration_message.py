#!/usr/bin/env python3
"""
向其他AGENT发送协作消息
保持与其他AGENT的沟通和协作
"""

import json
import time
from pathlib import Path


def send_collaboration_message():
    """
    向其他AGENT发送协作消息
    """
    state_file = Path("/tmp/tigertrade_agent_state.json")
    
    if not state_file.exists():
        print("❌ 状态文件不存在")
        return False
    
    try:
        state = json.loads(state_file.read_text())
        
        # 获取当前master和其他agents
        current_master = state.get('current_master', 'master')
        
        # 获取所有agents（除了我自己）
        other_agents = []
        for agent_id in state['agents'].keys():
            if agent_id not in ['proper_agent_v2', 'redis_connected_agent']:
                other_agents.append(agent_id)
        
        print(f"👥 检测到 {len(other_agents)} 个其他AGENT: {other_agents}")
        
        # 向所有其他agents发送协作消息
        for agent_id in other_agents:
            collaboration_msg = {
                "id": f"msg_{time.time()}_collaboration_to_{agent_id}",
                "from": "proper_agent_v2",
                "to": agent_id,
                "type": "collaboration_offer",
                "data": {
                    "message": "你好，我是proper_agent_v2，已成功接入系统并准备协作",
                    "status": "ready_for_collaboration",
                    "capabilities": [
                        "strategy_optimization",
                        "model_evaluation", 
                        "backtesting",
                        "risk_management",
                        "data_analysis",
                        "cross_machine_collaboration"
                    ],
                    "offer_help_with": [
                        "策略优化",
                        "模型训练",
                        "数据分析",
                        "风险控制"
                    ],
                    "current_focus": "提升TigerTrade月收益率至20%",
                    "contact_for_collaboration": True,
                    "timestamp": time.time()
                },
                "timestamp": time.time()
            }
            
            # 添加到消息队列
            state["messages"].append(collaboration_msg)
        
        # 同时也向master发送一个状态更新，表明我们正在进行协作
        master_status_msg = {
            "id": f"msg_{time.time()}_collaboration_status_to_{current_master}",
            "from": "proper_agent_v2",
            "to": current_master,
            "type": "collaboration_status",
            "data": {
                "message": "正在与其他AGENT建立协作关系",
                "status": "building_collaboration_network",
                "actions_taken": [
                    "发送协作邀请给其他agents",
                    "保持与master的任务请求",
                    "监控系统状态"
                ],
                "goal": "协助实现项目目标：月收益率20%",
                "ready_for_assignment": True,
                "timestamp": time.time()
            },
            "timestamp": time.time()
        }
        
        # 添加到消息队列
        state["messages"].append(master_status_msg)
        
        # 更新agent状态
        if "proper_agent_v2" in state["agents"]:
            state["agents"]["proper_agent_v2"]["status"] = "active_collaboration"
            state["agents"]["proper_agent_v2"]["last_heartbeat"] = time.time()
        
        if "redis_connected_agent" in state["agents"]:
            state["agents"]["redis_connected_agent"]["status"] = "active_collaboration"
            state["agents"]["redis_connected_agent"]["last_heartbeat"] = time.time()
        
        # 写回文件
        state_file.write_text(json.dumps(state, indent=2))
        
        print(f"✅ 已向 {len(other_agents)} 个AGENT发送协作消息")
        if other_agents:
            print(f"📝 协作消息已发送给: {', '.join(other_agents)}")
        print(f"📋 状态更新已发送给 {current_master}")
        return True
        
    except Exception as e:
        print(f"❌ 发送协作消息失败: {str(e)}")
        return False


def main():
    """主函数"""
    print("🤝 向其他AGENT发送协作消息")
    print("="*60)
    print("与其他AGENT建立联系，促进协作和信息共享")
    print("="*60)
    
    # 发送协作消息
    success = send_collaboration_message()
    
    print("\n" + "="*60)
    if success:
        print("✅ 协作消息已成功发送")
        print("   已与其他AGENT建立联系，促进协作")
        print("   继续监控系统并等待任务分配")
    else:
        print("⚠️ 发送协作消息时出现问题")
        print("   但系统状态已更新，继续等待响应")
    print("="*60)


if __name__ == "__main__":
    main()