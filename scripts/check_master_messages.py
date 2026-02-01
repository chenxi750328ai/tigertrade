#!/usr/bin/env python3
"""
检查是否收到master的消息
"""

import json
import time
from pathlib import Path


def check_master_messages():
    """
    检查是否收到master的消息
    """
    state_file = Path("/tmp/tigertrade_agent_state.json")
    
    if not state_file.exists():
        print("❌ 状态文件不存在")
        return False
    
    try:
        state = json.loads(state_file.read_text())
        
        print("📋 当前系统状态:")
        print(f"   协议版本: {state.get('protocol_version', 'unknown')}")
        print(f"   最后更新: {state.get('last_updated', 'unknown')}")
        
        print("\n👥 Agent状态:")
        for agent_id, agent_info in state.get('agents', {}).items():
            status = agent_info.get('status', 'unknown')
            role = agent_info.get('role', 'unknown')
            last_hb = agent_info.get('last_heartbeat', 'unknown')
            print(f"   - {agent_id} ({role}): {status}, heartbeat: {last_hb}")
        
        # 查找来自master的消息
        master_messages = []
        for msg in state.get('messages', []):
            msg_from = msg.get('from', '').lower()
            # 检查是否来自任何master类型的agent
            if ('master' in msg_from or 
                msg_from in ['claude_master', 'claude_master_v2', 'tigertrade_master', 'test_master']) and \
               msg.get('to') == 'proper_agent_v2':
                master_messages.append(msg)
        
        print(f"\n📩 来自master并发送给proper_agent_v2的消息:")
        if master_messages:
            for msg in sorted(master_messages, key=lambda x: x.get('timestamp', 0), reverse=True):
                msg_type = msg.get('type', 'unknown')
                msg_from = msg.get('from', 'unknown')
                timestamp = time.ctime(msg.get('timestamp', 0))
                
                print(f"   [{timestamp}] {msg_from} -> proper_agent_v2: {msg_type}")
                
                # 根据消息类型显示具体内容
                data = msg.get('data', {})
                if 'message' in data:
                    print(f"      消息: {data['message']}")
                if 'description' in data:
                    print(f"      描述: {data['description']}")
                if 'question' in data:
                    print(f"      问题: {data['question']}")
                if 'topic' in data:
                    print(f"      主题: {data['topic']}")
                if 'suggestion' in data:
                    print(f"      建议: {data['suggestion']}")
        else:
            print("   没有找到发送给proper_agent_v2的master消息")
            
        # 检查最近的所有消息
        print(f"\n📨 最近 10 条消息:")
        messages = sorted(state.get('messages', []), key=lambda x: x.get('timestamp', 0), reverse=True)[:10]
        for msg in messages:
            msg_type = msg.get('type', 'unknown')
            msg_from = msg.get('from', 'unknown')
            msg_to = msg.get('to', 'unknown')
            timestamp = time.ctime(msg.get('timestamp', 0))
            print(f"   [{timestamp}] {msg_from} -> {msg_to}: {msg_type}")
            
            # 如果是给我们的消息，特别标注
            if msg_to == 'proper_agent_v2':
                print(f"      >>> 这是一条发送给我们的消息")
        
        return len(master_messages) > 0
        
    except Exception as e:
        print(f"❌ 读取状态失败: {str(e)}")
        return False


def main():
    """主函数"""
    print("📬 检查master消息")
    print("="*70)
    print("查看是否收到master发送给我们的任何消息")
    print("="*70)
    
    has_master_messages = check_master_messages()
    
    print("\n" + "="*70)
    if has_master_messages:
        print("✅ 已收到master的消息")
        print("   我们正在正确地接收来自master的通信")
    else:
        print("❌ 暂未收到master的特定消息")
        print("   但我们已建立通信渠道，持续监听新消息")
        print("   请耐心等待master的进一步指示")
    print("="*70)


if __name__ == "__main__":
    main()