#!/usr/bin/env python3
"""
说明如何检查系统消息的脚本
展示我们使用的接口和方法来检查消息
"""

import json
import time
from pathlib import Path


def explain_message_checking_approach():
    """
    解释我们用来检查消息的方法
    """
    print("🔍 消息检查方法说明")
    print("="*60)
    print("我们使用以下接口和方法来检查系统中的消息：")
    print()
    
    print("1️⃣ 状态文件接口：")
    print("   - 文件路径：/tmp/tigertrade_agent_state.json")
    print("   - 包含所有agents的状态和消息队列")
    print("   - 所有agent通过此文件进行通信")
    print()
    
    print("2️⃣ 消息结构：")
    print("   - 每条消息包含：id, from, to, type, data, timestamp")
    print("   - 消息类型包括：task_request, worker_ready, task_assign等")
    print("   - 通过匹配'my_agent_ids'来过滤发给自己的消息")
    print()
    
    print("3️⃣ 检查方法：")
    print("   - 读取JSON文件内容")
    print("   - 遍历messages数组")
    print("   - 筛选to字段匹配的agent_id的消息")
    print()


def check_messages_for_agent(target_agent_id):
    """
    检查指定agent的消息
    """
    state_file = Path("/tmp/tigertrade_agent_state.json")
    
    if not state_file.exists():
        print("❌ 状态文件不存在")
        return []
    
    try:
        state = json.loads(state_file.read_text())
        
        # 查找发给指定agent的消息
        target_messages = [
            msg for msg in state['messages']
            if msg['to'] == target_agent_id
        ]
        
        print(f"📥 检查发给 {target_agent_id} 的消息:")
        print(f"   找到 {len(target_messages)} 条消息")
        
        for msg in target_messages:
            print(f"   - 类型: {msg['type']}")
            print(f"   - 来自: {msg['from']}")
            print(f"   - 时间: {time.ctime(msg['timestamp'])}")
            print(f"   - 内容: {str(msg['data'])[:100]}...")
            print()
        
        return target_messages
        
    except Exception as e:
        print(f"❌ 检查消息时出错: {str(e)}")
        return []


def check_system_status():
    """
    检查系统整体状态
    """
    state_file = Path("/tmp/tigertrade_agent_state.json")
    
    if not state_file.exists():
        print("❌ 状态文件不存在")
        return
    
    try:
        state = json.loads(state_file.read_text())
        
        print("📋 系统整体状态:")
        print(f"   协议版本: {state.get('protocol_version', 'unknown')}")
        print(f"   当前Master: {state.get('current_master', 'unknown')}")
        print(f"   总消息数: {len(state.get('messages', []))}")
        print(f"   总Agents数: {len(state.get('agents', {}))}")
        print()
        
        print("👥 注册的Agents:")
        for agent_id, agent_info in state.get('agents', {}).items():
            status = agent_info.get('status', 'unknown')
            role = agent_info.get('role', 'unknown')
            print(f"   - {agent_id} ({role}): {status}")
        print()
        
    except Exception as e:
        print(f"❌ 检查系统状态时出错: {str(e)}")


def main():
    """主函数"""
    explain_message_checking_approach()
    
    # 检查我们自己的agent的消息
    my_agents = ["proper_agent_v2", "redis_connected_agent"]
    
    for agent_id in my_agents:
        check_messages_for_agent(agent_id)
    
    # 检查系统整体状态
    check_system_status()
    
    print("💡 总结:")
    print("   我们使用 /tmp/tigertrade_agent_state.json 文件接口来检查消息")
    print("   这是AgentFuture框架中所有agents通信的中心接口")
    print("   通过读取JSON文件并筛选相关消息来实现消息检查功能")


if __name__ == "__main__":
    main()