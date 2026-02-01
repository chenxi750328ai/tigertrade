#!/usr/bin/env python3
"""
向MASTER发送消息的脚本
报告当前状态并请求任务分配
"""

import json
import time
from pathlib import Path


def send_message_to_master(message_type, data):
    """
    向MASTER发送消息
    
    Args:
        message_type: 消息类型
        data: 消息数据
    """
    state_file = Path("/tmp/tigertrade_agent_state.json")
    
    if not state_file.exists():
        print("❌ 状态文件不存在")
        return False
    
    try:
        state = json.loads(state_file.read_text())
        
        # 创建消息
        message = {
            "id": f"msg_{time.time()}_to_master",
            "from": "worker_lingma_enhanced",
            "to": "master",
            "type": message_type,
            "data": data,
            "timestamp": time.time()
        }
        
        # 添加到消息队列
        state["messages"].append(message)
        
        # 更新发送者状态
        if "worker_lingma_enhanced" in state["agents"]:
            state["agents"]["worker_lingma_enhanced"]["status"] = "reporting"
            state["agents"]["worker_lingma_enhanced"]["last_heartbeat"] = time.time()
        
        # 写回文件
        state_file.write_text(json.dumps(state, indent=2))
        
        print(f"✅ 消息已发送给MASTER")
        print(f"   类型: {message_type}")
        print(f"   内容: {data}")
        
        return True
        
    except Exception as e:
        print(f"❌ 发送消息失败: {str(e)}")
        return False


def main():
    """主函数"""
    print("📤 向MASTER发送状态报告和任务请求...")
    
    # 发送状态报告消息
    status_report = {
        "message": "报告MASTER：worker_lingma_enhanced已上线并准备好执行任务",
        "status": "ready_for_task",
        "capabilities": [
            "data_processing",
            "model_training", 
            "strategy_backtesting",
            "feature_analysis",
            "knowledge_sharing",
            "discussion_participation"
        ],
        "availability": "available",
        "request": "请分配任务"
    }
    
    send_message_to_master("status_report", status_report)
    
    # 发送任务请求消息
    task_request = {
        "message": "请求分配任务：我没有收到任何任务，但已准备好执行工作",
        "worker_id": "worker_lingma_enhanced",
        "status": "idle_and_waiting",
        "skills": [
            "python_development",
            "data_analysis",
            "machine_learning",
            "quantitative_trading"
        ]
    }
    
    send_message_to_master("task_request", task_request)
    
    print("\n📋 已完成向MASTER的消息发送")
    print("   1. 已发送状态报告")
    print("   2. 已发送任务请求")
    print("   MASTER将会处理这些请求")


if __name__ == "__main__":
    main()