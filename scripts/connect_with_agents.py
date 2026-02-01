#!/usr/bin/env python3
"""
与其他AGENT建立联系的脚本
用于了解当前系统状态和协议变化
"""

import json
import time
from pathlib import Path


def connect_with_agents():
    """
    与其他AGENT建立联系
    """
    state_file = Path("/tmp/tigertrade_agent_state.json")
    
    if not state_file.exists():
        print("❌ 状态文件不存在")
        return False
    
    try:
        state = json.loads(state_file.read_text())
        
        print("👥 当前系统中的AGENTs:")
        for agent_id, info in state.get('agents', {}).items():
            print(f"   - {agent_id}: {info.get('status', 'unknown')}, role: {info.get('role', 'unknown')}")
        
        print(f"\n📋 系统协议版本: {state.get('protocol_version', 'unknown')}")
        print(f"   当前MASTER: {state.get('current_master', 'unknown')}")
        
        # 发送连接消息给所有AGENT
        connection_msg = {
            "id": f"msg_{time.time()}_connect_all",
            "from": "worker_lingma_enhanced",
            "to": "all",
            "type": "connection_request",
            "data": {
                "message": "worker_lingma_enhanced 上线，寻求连接与协作",
                "status": "ready_to_collaborate",
                "capabilities": [
                    "strategy_optimization",
                    "model_evaluation", 
                    "backtesting",
                    "risk_management"
                ],
                "request": "请告知当前系统状态和最新协议",
                "timestamp": time.time()
            },
            "timestamp": time.time()
        }
        
        # 添加到消息队列
        state["messages"].append(connection_msg)
        
        # 更新自己的状态
        state["agents"]["worker_lingma_enhanced"] = {
            "role": "Worker",
            "status": "connecting",
            "task": None,
            "progress": 0.0,
            "last_heartbeat": time.time(),
            "registered_at": time.time(),
            "capabilities": [
                "data_processing",
                "model_training",
                "strategy_backtesting",
                "feature_analysis",
                "knowledge_sharing",
                "discussion_participation"
            ]
        }
        
        # 写回文件
        state_file.write_text(json.dumps(state, indent=2))
        
        print(f"\n✅ 连接请求已发送给所有AGENT")
        return True
        
    except Exception as e:
        print(f"❌ 连接失败: {str(e)}")
        return False


def check_recent_messages():
    """
    检查最近的系统消息
    """
    state_file = Path("/tmp/tigertrade_agent_state.json")
    
    if not state_file.exists():
        print("❌ 状态文件不存在")
        return None
    
    try:
        state = json.loads(state_file.read_text())
        
        print(f"\n💬 最近10条系统消息:")
        for msg in state.get('messages', [])[-10:]:
            print(f"   {msg['from']} -> {msg['to']} ({msg['type']}): {str(msg['data'])[:100]}...")
        
        return state
        
    except Exception as e:
        print(f"❌ 检查消息失败: {str(e)}")
        return None


def main():
    """主函数"""
    print("🤝 与其他AGENT建立联系")
    print("="*60)
    
    # 与其他AGENT建立联系
    connect_with_agents()
    
    # 检查最近的系统消息
    state = check_recent_messages()
    
    if state:
        protocol_version = state.get('protocol_version', 'unknown')
        print(f"\n📖 当前协议版本: {protocol_version}")
        
        print("\n✅ 已与其他AGENT建立联系")
        print("   您现在可以继续了解项目文档和协议变化")


if __name__ == "__main__":
    main()