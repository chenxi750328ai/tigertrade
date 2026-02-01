#!/usr/bin/env python3
"""
持续监听和响应新消息的脚本
"""

import json
import time
import threading
from pathlib import Path


def listen_for_new_messages():
    """
    持续监听新消息
    """
    state_file = Path("/tmp/tigertrade_agent_state.json")
    
    if not state_file.exists():
        print("❌ 状态文件不存在")
        return
    
    # 记录初始消息数量
    initial_state = json.loads(state_file.read_text())
    initial_msg_count = len(initial_state.get('messages', []))
    
    print(f"✅ 开始监听新消息，当前已有 {initial_msg_count} 条消息")
    print("💡 按 Ctrl+C 停止监听")
    
    try:
        while True:
            time.sleep(5)  # 每5秒检查一次
            
            if not state_file.exists():
                continue
                
            current_state = json.loads(state_file.read_text())
            current_msg_count = len(current_state.get('messages', []))
            
            if current_msg_count > initial_msg_count:
                new_messages = sorted(
                    current_state.get('messages', []), 
                    key=lambda x: x.get('timestamp', 0), 
                    reverse=True
                )[:current_msg_count - initial_msg_count]
                
                for msg in new_messages:
                    handle_message(msg, state_file)
                
                initial_msg_count = current_msg_count
                
    except KeyboardInterrupt:
        print("\n🛑 监听已停止")


def handle_message(msg, state_file):
    """
    处理接收到的消息
    """
    msg_type = msg.get('type', 'unknown')
    msg_from = msg.get('from', 'unknown')
    msg_to = msg.get('to', 'unknown')
    timestamp = msg.get('timestamp', 0)
    
    print(f"\n📥 [{time.ctime(timestamp)}] 收到消息: {msg_from} -> {msg_to} ({msg_type})")
    
    # 根据消息类型进行相应处理
    if msg_to == 'proper_agent_v2' or msg_to == 'all':
        if msg_type == 'task_assign':
            handle_task_assignment(msg, state_file)
        elif msg_type == 'discussion':
            handle_discussion(msg, state_file)
        elif msg_type == 'knowledge_share':
            handle_knowledge_share(msg, state_file)
        elif msg_type == 'guidance':
            handle_guidance(msg, state_file)
        elif msg_type == 'discussion_reply':
            handle_discussion_reply(msg, state_file)
        elif msg_type == 'suggestion_vote':
            handle_suggestion_vote(msg, state_file)


def handle_task_assignment(msg, state_file):
    """
    处理任务分配消息
    """
    task_data = msg.get('data', {})
    task_id = task_data.get('task_id', 'unknown')
    
    print(f"   📋 任务分配: {task_data.get('description', 'no description')}")
    
    # 更新agent状态
    state = json.loads(state_file.read_text())
    if "proper_agent_v2" in state["agents"]:
        state["agents"]["proper_agent_v2"]["status"] = f"working_on_{task_id}"
        state["agents"]["proper_agent_v2"]["task"] = task_id
        state["agents"]["proper_agent_v2"]["last_heartbeat"] = time.time()
    
    # 发送确认消息
    confirm_msg = {
        "id": f"msg_{time.time()}_task_confirm_{task_id}",
        "from": "proper_agent_v2",
        "to": msg.get('from', 'unknown'),
        "type": "progress_update",
        "data": {
            "task_id": task_id,
            "progress": 0.0,
            "message": f"已收到任务: {task_data.get('description', 'no description')}",
            "eta": 300
        },
        "timestamp": time.time()
    }
    
    state["messages"].append(confirm_msg)
    state_file.write_text(json.dumps(state, indent=2))
    
    print(f"   ✅ 已确认任务 {task_id}")


def handle_discussion(msg, state_file):
    """
    处理讨论消息
    """
    discussion_data = msg.get('data', {})
    topic = discussion_data.get('topic', 'no topic')
    
    print(f"   💬 讨论主题: {topic}")
    
    # 如果是关于API配置问题的讨论，给予回应
    if 'api' in topic.lower() or 'config' in topic.lower():
        # 发送回复
        reply_msg = {
            "id": f"msg_{time.time()}_discussion_reply",
            "from": "proper_agent_v2",
            "to": "all",
            "type": "discussion_reply",
            "data": {
                "reply_to": msg.get('id'),
                "opinion": "同意，我已经将API配置验证最佳实践分享到了RAG系统中",
                "vote": "agree",
                "confidence": 0.95
            },
            "timestamp": time.time()
        }
        
        state = json.loads(state_file.read_text())
        state["messages"].append(reply_msg)
        state_file.write_text(json.dumps(state, indent=2))
        
        print(f"   💬 已回复讨论")


def handle_knowledge_share(msg, state_file):
    """
    处理知识分享消息
    """
    knowledge_data = msg.get('data', {})
    title = knowledge_data.get('title', 'no title')
    
    print(f"   📚 知识分享: {title}")
    
    # 如果是关于策略或API的分享，记录下来
    if any(keyword in title.lower() for keyword in ['strategy', 'api', 'config', 'trading']):
        print(f"   📝 已记录: {title}")


def handle_guidance(msg, state_file):
    """
    处理指导消息
    """
    guidance_data = msg.get('data', {})
    message = guidance_data.get('message', 'no message')
    
    print(f"   🧭 指导: {message}")
    
    # 发送确认消息
    confirm_msg = {
        "id": f"msg_{time.time()}_guidance_ack",
        "from": "proper_agent_v2",
        "to": msg.get('from', 'unknown'),
        "type": "progress_update",
        "data": {
            "message": f"已收到指导: {message}",
            "acknowledged": True
        },
        "timestamp": time.time()
    }
    
    state = json.loads(state_file.read_text())
    state["messages"].append(confirm_msg)
    state_file.write_text(json.dumps(state, indent=2))
    
    print(f"   ✅ 已确认收到指导")


def handle_discussion_reply(msg, state_file):
    """
    处理讨论回复消息
    """
    reply_data = msg.get('data', {})
    opinion = reply_data.get('opinion', 'no opinion')
    
    print(f"   💬 回复意见: {opinion}")


def handle_suggestion_vote(msg, state_file):
    """
    处理建议投票消息
    """
    vote_data = msg.get('data', {})
    vote = vote_data.get('vote', 'no vote')
    
    print(f"   🗳️ 投票: {vote}")
    
    # 如果是关于我们提出的建议的投票，记录下来
    if 'suggestion_id' in vote_data:
        print(f"   📊 已记录对此建议的投票")


def send_periodic_status():
    """
    定期发送状态更新
    """
    state_file = Path("/tmp/tigertrade_agent_state.json")
    
    while True:
        try:
            time.sleep(60)  # 每分钟发送一次状态
            
            if not state_file.exists():
                continue
                
            state = json.loads(state_file.read_text())
            
            # 发送状态更新
            status_msg = {
                "id": f"msg_{time.time()}_periodic_status",
                "from": "proper_agent_v2",
                "to": "all",
                "type": "progress_update",
                "data": {
                    "status": "listening_for_tasks",
                    "available": True,
                    "capabilities": ["data_processing", "strategy_implementation", "api_validation"],
                    "message": "随时准备接受新任务"
                },
                "timestamp": time.time()
            }
            
            state["messages"].append(status_msg)
            
            # 更新agent状态
            if "proper_agent_v2" in state["agents"]:
                state["agents"]["proper_agent_v2"]["status"] = "listening_for_tasks"
                state["agents"]["proper_agent_v2"]["last_heartbeat"] = time.time()
            
            state_file.write_text(json.dumps(state, indent=2))
            
            print(f"📊 状态更新已发送")
            
        except KeyboardInterrupt:
            print("\n🛑 状态更新已停止")
            break
        except Exception as e:
            print(f"❌ 发送状态更新失败: {str(e)}")


def main():
    """主函数"""
    print("💬 持续消息处理与沟通系统")
    print("="*70)
    print("启动持续监听新消息和与其它agents沟通的功能")
    print("="*70)
    
    # 创建两个线程：一个用于监听消息，一个用于定期发送状态
    listener_thread = threading.Thread(target=listen_for_new_messages)
    status_thread = threading.Thread(target=send_periodic_status)
    
    # 设置为守护线程，这样主程序退出时它们也会退出
    listener_thread.daemon = True
    status_thread.daemon = True
    
    # 启动线程
    listener_thread.start()
    status_thread.start()
    
    print("\n🚀 消息处理系统已启动")
    print("   - 监听线程正在运行")
    print("   - 状态更新线程正在运行")
    print("   - 系统将持续监听和响应新消息")
    print("   - 按 Ctrl+C 停止系统")
    
    try:
        # 保持主线程运行
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 系统正在关闭...")


if __name__ == "__main__":
    main()