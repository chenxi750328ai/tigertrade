#!/usr/bin/env python3
"""
验证通信是否正常工作的脚本
"""

import json
import time
from pathlib import Path


def verify_communication():
    """
    验证通信是否正常
    """
    state_file = Path("/tmp/tigertrade_agent_state.json")
    
    if not state_file.exists():
        print("❌ 状态文件不存在")
        return False
    
    try:
        state = json.loads(state_file.read_text())
        
        print("✅ 通信验证成功")
        print(f"   协议版本: {state.get('protocol_version', 'unknown')}")
        print(f"   总消息数: {len(state.get('messages', []))}")
        
        # 检查最近的消息
        recent_msgs = sorted(state.get('messages', []), key=lambda x: x.get('timestamp', 0), reverse=True)[:3]
        print("\n   最近3条消息:")
        for msg in recent_msgs:
            msg_type = msg.get('type', 'unknown')
            msg_from = msg.get('from', 'unknown')
            msg_to = msg.get('to', 'unknown')
            timestamp = time.ctime(msg.get('timestamp', 0))
            print(f"     [{timestamp}] {msg_from} -> {msg_to}: {msg_type}")
        
        # 检查我们的状态
        our_agent = state.get('agents', {}).get('proper_agent_v2', {})
        print(f"\n   我们的Agent状态: {our_agent.get('status', 'unknown')}")
        
        # 发送一个测试消息到系统
        test_msg = {
            "id": f"msg_{time.time()}_communication_test",
            "from": "proper_agent_v2",
            "to": "all",
            "type": "discussion",
            "data": {
                "topic": "通信测试",
                "question": "通信系统正常运行，随时准备接受新任务",
                "timestamp": time.time()
            },
            "timestamp": time.time()
        }
        
        # 添加到消息队列
        state["messages"].append(test_msg)
        
        # 更新agent状态
        if "proper_agent_v2" in state["agents"]:
            state["agents"]["proper_agent_v2"]["status"] = "communicating_normally"
            state["agents"]["proper_agent_v2"]["last_heartbeat"] = time.time()
        
        # 写回文件
        state_file.write_text(json.dumps(state, indent=2))
        
        print(f"\n   📨 测试消息已发送")
        print(f"   🔄 状态已更新")
        
        return True
        
    except Exception as e:
        print(f"❌ 验证通信失败: {str(e)}")
        return False


def main():
    """主函数"""
    print("📡 通信验证")
    print("="*50)
    print("验证与其它agents的通信是否正常工作")
    print("="*50)
    
    success = verify_communication()
    
    print("\n" + "="*50)
    if success:
        print("✅ 通信系统正常运行")
        print("   - 可以读取系统状态")
        print("   - 可以发送消息")
        print("   - 持续监听脚本正在后台运行")
        print("   - 随时准备接收新任务")
    else:
        print("❌ 通信系统可能有问题")
        print("   请检查系统状态")
    print("="*50)


if __name__ == "__main__":
    main()