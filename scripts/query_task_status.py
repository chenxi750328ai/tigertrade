#!/usr/bin/env python3
"""
与MASTER通信的任务状态查询脚本
用于确认当前任务状态，避免执行过时任务
"""

import json
import time
from pathlib import Path


def send_query_to_master(query_type, data):
    """
    向MASTER发送查询消息
    
    Args:
        query_type: 查询类型
        data: 查询数据
    """
    state_file = Path("/tmp/tigertrade_agent_state.json")
    
    if not state_file.exists():
        print("❌ 状态文件不存在")
        return False
    
    try:
        state = json.loads(state_file.read_text())
        
        # 创建查询消息
        message = {
            "id": f"msg_{time.time()}_query_{query_type}",
            "from": "worker_lingma_enhanced",
            "to": "master",
            "type": query_type,
            "data": data,
            "timestamp": time.time()
        }
        
        # 添加到消息队列
        state["messages"].append(message)
        
        # 更新发送者状态
        if "worker_lingma_enhanced" in state["agents"]:
            state["agents"]["worker_lingma_enhanced"]["status"] = "inquiring"
            state["agents"]["worker_lingma_enhanced"]["last_heartbeat"] = time.time()
        
        # 写回文件
        state_file.write_text(json.dumps(state, indent=2))
        
        print(f"✅ 查询消息已发送给MASTER")
        print(f"   类型: {query_type}")
        print(f"   内容: {data}")
        
        return True
        
    except Exception as e:
        print(f"❌ 发送查询消息失败: {str(e)}")
        return False


def check_for_updates_from_master():
    """
    检查是否有来自MASTER的更新或回复
    """
    state_file = Path("/tmp/tigertrade_agent_state.json")
    
    if not state_file.exists():
        print("❌ 状态文件不存在")
        return None
    
    try:
        state = json.loads(state_file.read_text())
        
        # 查找MASTER发来的回复或更新
        master_messages = [
            msg for msg in state['messages'] 
            if msg['from'] == 'master' and msg['to'] == 'worker_lingma_enhanced'
        ]
        
        if master_messages:
            print(f"📥 收到来自MASTER的 {len(master_messages)} 条消息:")
            for msg in master_messages:
                print(f"   - {msg['type']}: {msg['data']}")
            return master_messages
        else:
            print("📭 暂无来自MASTER的回复")
            return None
            
    except Exception as e:
        print(f"❌ 检查MASTER回复失败: {str(e)}")
        return None


def main():
    """主函数"""
    print("📡 与MASTER通信 - 任务状态查询")
    print("="*60)
    
    # 1. 发送任务状态查询
    task_status_query = {
        "message": "worker_lingma_enhanced 查询当前任务状态",
        "request": "请确认当前任务是否仍然有效，是否需要更新",
        "current_task": "策略回测优化，目标月收益率20%",
        "timestamp": time.time()
    }
    
    send_query_to_master("task_status_query", task_status_query)
    
    # 2. 发送是否过时任务查询
    outdated_query = {
        "message": "worker_lingma_enhanced 确认任务是否过时",
        "request": "请确认我收到的任务指令是否仍然适用，是否已有新的策略方向",
        "concern": "避免执行过时或不再适用的任务",
        "timestamp": time.time()
    }
    
    send_query_to_master("outdated_task_check", outdated_query)
    
    # 3. 检查是否有来自MASTER的回复
    print("\n🔍 正在检查MASTER的回复...")
    replies = check_for_updates_from_master()
    
    if not replies:
        print("\n⏰ 没有立即回复，发送提醒消息...")
        reminder = {
            "message": "worker_lingma_enhanced 等待任务确认",
            "request": "请MASTER尽快确认任务状态，以便我能正确开展工作",
            "availability": "随时可以开始工作",
            "need_guidance": True,
            "timestamp": time.time()
        }
        
        send_query_to_master("task_reminder", reminder)
    
    print("\n📋 任务状态查询完成")
    print("   已发送状态查询和过时任务确认请求")
    print("   正在等待MASTER的进一步指示")


if __name__ == "__main__":
    main()