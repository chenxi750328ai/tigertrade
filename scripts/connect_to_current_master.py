#!/usr/bin/env python3
"""
与当前master通信的脚本
根据系统状态，当前master是worker_a
"""

import json
import time
from pathlib import Path


def send_message_to_current_master():
    """
    向当前master发送消息
    """
    state_file = Path("/tmp/tigertrade_agent_state.json")
    
    if not state_file.exists():
        print("❌ 状态文件不存在")
        return False
    
    try:
        state = json.loads(state_file.read_text())
        
        # 获取当前master
        current_master = state.get('current_master', 'master')
        print(f"📡 检测到当前master: {current_master}")
        
        # 发送状态报告给当前master
        status_report = {
            "id": f"msg_{time.time()}_status_to_{current_master}",
            "from": "worker_lingma_enhanced",
            "to": current_master,
            "type": "worker_status_report",
            "data": {
                "message": "worker_lingma_enhanced 向您报到",
                "status": "ready_for_task_assignment",
                "capabilities": [
                    "strategy_optimization",
                    "model_evaluation", 
                    "backtesting",
                    "risk_management"
                ],
                "current_task_understanding": "优化策略以达到20%月收益率",
                "request_for_task": "请分配可执行的任务",
                "availability": "immediately_available",
                "last_election_participation": "worker_lingma_enhanced_is_candidate",
                "timestamp": time.time()
            },
            "timestamp": time.time()
        }
        
        # 添加到消息队列
        state["messages"].append(status_report)
        
        # 更新发送者状态
        if "worker_lingma_enhanced" in state["agents"]:
            state["agents"]["worker_lingma_enhanced"]["status"] = "reported_to_master"
            state["agents"]["worker_lingma_enhanced"]["last_heartbeat"] = time.time()
        
        # 写回文件
        state_file.write_text(json.dumps(state, indent=2))
        
        print(f"✅ 状态报告已发送给当前MASTER ({current_master})")
        print(f"   内容: {status_report['data']['message']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 发送状态报告失败: {str(e)}")
        return False


def check_for_response_from_current_master():
    """
    检查是否有来自当前master的回复
    """
    state_file = Path("/tmp/tigertrade_agent_state.json")
    
    if not state_file.exists():
        print("❌ 状态文件不存在")
        return None
    
    try:
        state = json.loads(state_file.read_text())
        
        # 获取当前master
        current_master = state.get('current_master', 'master')
        print(f"🔍 检查来自 {current_master} 的回复...")
        
        # 查找当前master发给我的消息
        master_messages = [
            msg for msg in state['messages'] 
            if msg['from'] == current_master and msg['to'] == 'worker_lingma_enhanced'
        ]
        
        if master_messages:
            print(f"📥 收到来自 {current_master} 的 {len(master_messages)} 条消息:")
            for msg in master_messages:
                print(f"   - {msg['type']}: {msg['data']}")
            return master_messages
        else:
            print(f"📭 暂无来自 {current_master} 的回复")
            
            # 检查是否有广播消息
            broadcast_messages = [
                msg for msg in state['messages'] 
                if msg['to'] == 'all' and msg['from'] == current_master
            ]
            
            if broadcast_messages:
                print(f"📢 检测到 {current_master} 发送的 {len(broadcast_messages)} 条广播消息:")
                for msg in broadcast_messages[-3:]:  # 显示最近3条
                    print(f"   - {msg['type']}: {str(msg['data'])[:100]}...")
            
            return None
            
    except Exception as e:
        print(f"❌ 检查回复失败: {str(e)}")
        return None


def main():
    """主函数"""
    print("🤝 连接到当前MASTER")
    print("="*60)
    
    # 发送状态报告给当前master
    print("\n1️⃣ 发送状态报告给当前MASTER...")
    send_message_to_current_master()
    
    # 等待一小段时间让消息传递
    time.sleep(2)
    
    # 检查是否有回复
    print("\n2️⃣ 检查是否有来自MASTER的回复...")
    responses = check_for_response_from_current_master()
    
    if responses:
        print(f"\n✅ 收到 {len(responses)} 条回复，可以开始工作")
        latest_response = responses[-1]
        
        if latest_response['type'] == 'task_assign':
            print(f"🎯 检测到任务分配: {latest_response['data']}")
        elif latest_response['type'] == 'guidance':
            print(f"📋 收到指导信息: {latest_response['data']}")
        else:
            print(f"ℹ️  收到其他类型消息: {latest_response['type']}")
    else:
        print("\n⏳ 未收到直接回复，但已成功发送状态报告")
        print("   MASTER现在知道您在线并准备好接受任务")
        print("   继续监听可能的任务分配...")
        
        # 发送一个任务请求
        request_task_assignment()
    
    print("\n✅ 通信完成")


def request_task_assignment():
    """
    主动请求任务分配
    """
    state_file = Path("/tmp/tigertrade_agent_state.json")
    
    if not state_file.exists():
        print("❌ 状态文件不存在")
        return False
    
    try:
        state = json.loads(state_file.read_text())
        
        # 获取当前master
        current_master = state.get('current_master', 'master')
        
        # 发送任务请求
        task_request = {
            "id": f"msg_{time.time()}_task_request_to_{current_master}",
            "from": "worker_lingma_enhanced",
            "to": current_master,
            "type": "task_request",
            "data": {
                "message": "worker_lingma_enhanced 请求任务分配",
                "status": "waiting_for_task",
                "capabilities": [
                    "strategy_optimization",
                    "model_evaluation", 
                    "backtesting"
                ],
                "interests": ["strategy_optimization", "performance_improvement"],
                "target": "提升月收益率至20%",
                "ready_immediately": True,
                "timestamp": time.time()
            },
            "timestamp": time.time()
        }
        
        # 添加到消息队列
        state["messages"].append(task_request)
        
        # 更新发送者状态
        if "worker_lingma_enhanced" in state["agents"]:
            state["agents"]["worker_lingma_enhanced"]["status"] = "requesting_task"
            state["agents"]["worker_lingma_enhanced"]["last_heartbeat"] = time.time()
        
        # 写回文件
        state_file.write_text(json.dumps(state, indent=2))
        
        print(f"✅ 任务请求已发送给 {current_master}")
        return True
        
    except Exception as e:
        print(f"❌ 发送任务请求失败: {str(e)}")
        return False


if __name__ == "__main__":
    main()