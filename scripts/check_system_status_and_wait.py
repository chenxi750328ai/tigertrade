#!/usr/bin/env python3
"""
检查系统状态并等待其他agents的响应
"""

import json
import time
from pathlib import Path


def check_system_status():
    """
    检查系统状态，查看是否有其他agents的响应
    """
    state_file = Path("/tmp/tigertrade_agent_state.json")
    
    if not state_file.exists():
        print("❌ 状态文件不存在")
        return None
    
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
        
        print(f"\n📨 最近 {min(10, len(state.get('messages', [])))} 条消息:")
        messages = sorted(state.get('messages', []), key=lambda x: x.get('timestamp', 0), reverse=True)[:10]
        for msg in messages:
            msg_type = msg.get('type', 'unknown')
            msg_from = msg.get('from', 'unknown')
            msg_to = msg.get('to', 'unknown')
            timestamp = msg.get('timestamp', 0)
            print(f"   [{time.ctime(timestamp) if timestamp else '?'}] {msg_from} -> {msg_to}: {msg_type}")
            if 'data' in msg and 'topic' in msg.get('data', {}):
                print(f"      主题: {msg['data']['topic']}")
        
        return state
        
    except Exception as e:
        print(f"❌ 读取状态失败: {str(e)}")
        return None


def wait_for_responses(timeout_seconds=300):
    """
    等待其他agents的响应
    """
    print(f"⏳ 等待其他agents响应，超时时间: {timeout_seconds}秒")
    
    start_time = time.time()
    initial_state = check_system_status()
    initial_msg_count = len(initial_state.get('messages', [])) if initial_state else 0
    
    while time.time() - start_time < timeout_seconds:
        time.sleep(5)  # 每5秒检查一次
        
        current_state = check_system_status()
        if not current_state:
            continue
            
        current_msg_count = len(current_state.get('messages', []))
        
        # 检查是否有新消息
        if current_msg_count > initial_msg_count:
            print(f"\n✅ 检测到新消息: {current_msg_count - initial_msg_count} 条新消息")
            
            # 显示新消息
            all_messages = sorted(current_state.get('messages', []), key=lambda x: x.get('timestamp', 0), reverse=True)
            new_messages = all_messages[:current_msg_count - initial_msg_count]
            
            for msg in new_messages:
                msg_type = msg.get('type', 'unknown')
                msg_from = msg.get('from', 'unknown')
                msg_to = msg.get('to', 'unknown')
                timestamp = msg.get('timestamp', 0)
                
                print(f"   [NEW] {msg_from} -> {msg_to}: {msg_type}")
                if 'data' in msg:
                    data = msg['data']
                    if 'topic' in data:
                        print(f"         主题: {data['topic']}")
                    if 'question' in data:
                        print(f"         问题: {data['question'][:50]}...")
                    if 'suggestion' in data:
                        print(f"         建议: {data['suggestion'][:50]}...")
                    if 'reply_to' in data:
                        print(f"         回复: {data['reply_to']}")
                    if 'opinion' in data:
                        print(f"         观点: {data['opinion'][:50]}...")
        
        # 检查是否有针对我们发起的讨论或建议的回应
        for msg in current_state.get('messages', []):
            if msg.get('type') in ['discussion_reply', 'suggestion_vote']:
                related_to_us = False
                vote_for_our_suggestion = False  # 初始化此变量
                
                if 'reply_to' in msg.get('data', {}):
                    # 检查是否回复了我们的消息
                    reply_to = msg['data']['reply_to']
                    our_msgs = [m for m in all_messages if m.get('id') == reply_to and m.get('from') == 'proper_agent_v2']
                    if our_msgs:
                        related_to_us = True
                        
                if 'suggestion_id' in msg.get('data', {}):
                    # 检查是否对我们的建议进行了投票
                    suggestion_id = msg['data']['suggestion_id']
                    our_suggestions = [
                        m for m in all_messages 
                        if m.get('type') == 'project_suggestion' 
                        and m.get('id') == suggestion_id 
                        and m.get('from') == 'proper_agent_v2'
                    ]
                    if our_suggestions:
                        vote_for_our_suggestion = True
                
                if related_to_us or vote_for_our_suggestion:
                    vote = msg['data'].get('vote', 'no vote specified')
                    opinion = msg['data'].get('opinion', 'no opinion specified')
                    print(f"   🎯 检测到对我们的回应: {vote}, {opinion}")
        
        # 检查是否有新的API配置问题讨论
        api_discussion_responses = [
            m for m in current_state.get('messages', []) 
            if m.get('type') == 'discussion_reply' 
            and 'question' in m.get('data', {})
            and 'API' in m['data']['question']
        ]
        
        if api_discussion_responses:
            print(f"   📢 检测到 {len(api_discussion_responses)} 个关于API配置问题的回复")
            for resp in api_discussion_responses[-3:]:  # 最新的3个
                opinion = resp.get('data', {}).get('opinion', 'no opinion')
                from_agent = resp.get('from', 'unknown')
                print(f"      {from_agent}: {opinion[:60]}...")
    
    print(f"\n⏰ 等待超时，共收到 {current_msg_count - initial_msg_count} 条新消息")
    return current_state


def main():
    """主函数"""
    print("🔍 检查系统状态并等待其他agents响应")
    print("="*70)
    print("查看是否有其他agents对我们的讨论、建议或知识分享做出回应")
    print("="*70)
    
    # 检查当前状态
    print("\n1️⃣ 检查当前系统状态...")
    current_state = check_system_status()
    
    if not current_state:
        print("❌ 无法获取系统状态")
        return
    
    # 等待响应
    print("\n2️⃣ 等待其他agents响应...")
    final_state = wait_for_responses(120)  # 等待2分钟
    
    # 总结
    print("\n" + "="*70)
    print("📋 总结")
    print("   已检查系统状态")
    print("   已等待其他agents响应")
    print("   如有响应，可在后续工作中考虑其他agents的意见")
    print("="*70)


if __name__ == "__main__":
    main()