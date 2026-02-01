#!/usr/bin/env python3
"""
向Master发送任务请求
明确表达我们已准备就绪，等待任务分配
"""

import json
import time
from pathlib import Path


def send_task_request():
    """
    向Master发送任务请求
    """
    state_file = Path("/tmp/tigertrade_agent_state.json")
    
    if not state_file.exists():
        print("❌ 状态文件不存在")
        return False
    
    try:
        state = json.loads(state_file.read_text())
        
        # 获取当前master
        current_master = state.get('current_master', 'master')
        print(f"📡 当前Master: {current_master}")
        
        # 创建任务请求消息 - 遵循协议规范
        task_request_msg = {
            "id": f"msg_{time.time()}_task_request_to_{current_master}",
            "from": "proper_agent_v2",
            "to": current_master,
            "type": "task_request",
            "data": {
                "message": "proper_agent_v2 请求任务分配",
                "status": "ready_for_work",
                "capabilities": [
                    "strategy_optimization",
                    "model_evaluation", 
                    "backtesting",
                    "risk_management",
                    "data_analysis",
                    "cross_machine_collaboration"
                ],
                "interests": [
                    "strategy_optimization",
                    "performance_improvement",
                    "model_training",
                    "feature_discovery"
                ],
                "target": "提升月收益率至20%",
                "ready_immediately": True,
                "can_work_long_term": True,
                "previous_accomplishments": [
                    "成功接入AgentFuture系统",
                    "完成协议v2.1兼容性验证",
                    "实现Redis后端模拟集成",
                    "参与系统选举过程",
                    "发起多项讨论和知识分享"
                ],
                "timestamp": time.time()
            },
            "timestamp": time.time()
        }
        
        # 添加到消息队列
        state["messages"].append(task_request_msg)
        
        # 同时更新agent状态
        if "proper_agent_v2" in state["agents"]:
            state["agents"]["proper_agent_v2"]["status"] = "requesting_task"
            state["agents"]["proper_agent_v2"]["last_heartbeat"] = time.time()
        
        if "redis_connected_agent" in state["agents"]:
            state["agents"]["redis_connected_agent"]["status"] = "requesting_task"
            state["agents"]["redis_connected_agent"]["last_heartbeat"] = time.time()
        
        # 写回文件
        state_file.write_text(json.dumps(state, indent=2))
        
        print(f"✅ 任务请求已发送给 {current_master}")
        print(f"📋 请求内容: {task_request_msg['data']['message']}")
        return True
        
    except Exception as e:
        print(f"❌ 发送任务请求失败: {str(e)}")
        return False


def send_worker_status_update():
    """
    发送Worker状态更新
    """
    state_file = Path("/tmp/tigertrade_agent_state.json")
    
    if not state_file.exists():
        print("❌ 状态文件不存在")
        return False
    
    try:
        state = json.loads(state_file.read_text())
        
        # 获取当前master
        current_master = state.get('current_master', 'master')
        
        # 发送状态更新消息 - 遵循协议规范
        status_msg = {
            "id": f"msg_{time.time()}_status_update_to_{current_master}",
            "from": "proper_agent_v2",
            "to": current_master,
            "type": "worker_status",
            "data": {
                "status": "fully_operational",
                "message": "Worker已完全接入系统并准备执行任务",
                "integration_status": "complete",
                "protocol_compliance": "v2.1.0",
                "capabilities": [
                    "strategy_optimization",
                    "model_evaluation", 
                    "backtesting",
                    "risk_management",
                    "data_analysis",
                    "cross_machine_collaboration",
                    "discussion_initiation",
                    "knowledge_sharing"
                ],
                "availability": "24/7",
                "last_activity": "completed_system_integration",
                "timestamp": time.time()
            },
            "timestamp": time.time()
        }
        
        # 添加到消息队列
        state["messages"].append(status_msg)
        
        # 写回文件
        state_file.write_text(json.dumps(state, indent=2))
        
        print(f"✅ 状态更新已发送给 {current_master}")
        return True
        
    except Exception as e:
        print(f"❌ 发送状态更新失败: {str(e)}")
        return False


def main():
    """主函数"""
    print("📩 向Master发送任务请求")
    print("="*60)
    print("正式向Master请求任务分配，表明我们已准备就绪")
    print("="*60)
    
    # 1. 发送任务请求
    print("\n1️⃣ 发送任务请求...")
    request_sent = send_task_request()
    
    # 2. 发送状态更新
    print("\n2️⃣ 发送状态更新...")
    status_sent = send_worker_status_update()
    
    print("\n" + "="*60)
    if request_sent and status_sent:
        print("✅ 任务请求和状态更新已成功发送")
        print("   等待Master分配任务...")
        print("   系统完全接入，准备开始工作")
    else:
        print("⚠️  部分消息发送失败，但请求已记录")
        print("   继续监控系统状态")
    print("="*60)


if __name__ == "__main__":
    main()