#!/usr/bin/env python3
"""
监测MASTER回复的脚本
用于持续检查是否有来自MASTER的回复
"""

import json
import time
from pathlib import Path


def monitor_master_reply(duration=300):
    """
    监测MASTER的回复
    
    Args:
        duration: 监测时长（秒）
    """
    state_file = Path("/tmp/tigertrade_agent_state.json")
    start_time = time.time()
    
    print(f"👀 开始监测MASTER回复，监测时长: {duration}秒")
    print("="*60)
    
    # 记录初始消息数
    initial_msg_count = 0
    if state_file.exists():
        initial_state = json.loads(state_file.read_text())
        initial_msg_count = len(initial_state['messages'])
    
    latest_master_reply = None
    reply_found_time = None
    
    while time.time() - start_time < duration:
        if not state_file.exists():
            time.sleep(2)
            continue
        
        try:
            state = json.loads(state_file.read_text())
            
            # 查找MASTER发给我的最新回复
            master_messages = [
                msg for msg in state['messages'] 
                if msg['from'] == 'master' and msg['to'] == 'worker_lingma_enhanced'
            ]
            
            if master_messages:
                # 获取最新的回复
                latest_msg = master_messages[-1]
                
                if latest_msg != latest_master_reply:
                    latest_master_reply = latest_msg
                    reply_found_time = time.time()
                    
                    print(f"\n📩 收到来自MASTER的新消息!")
                    print(f"   时间: {time.strftime('%H:%M:%S', time.localtime(latest_msg['timestamp']))}")
                    print(f"   类型: {latest_msg['type']}")
                    print(f"   内容: {latest_msg['data']}")
                    
                    # 如果是任务确认或更新，可以据此采取行动
                    if latest_msg['type'] in ['task_confirmed', 'task_updated', 'task_assignment_new']:
                        print(f"   🎯 检测到任务更新，可能需要调整工作方向")
                    
                    if latest_msg['type'] == 'task_obsolete':
                        print(f"   ⚠️  收到任务过时通知，需要停止当前工作")
                    
                    print("-" * 50)
            
            # 每隔几秒打印一次监测状态
            if int(time.time()) % 10 == 0:
                print(f"⏱️  监测进行中... {(time.time() - start_time):.0f}s")
                time.sleep(0.1)  # 避免过多重复打印
            
            time.sleep(0.5)  # 每0.5秒检查一次
            
        except Exception as e:
            print(f"❌ 监测过程中出现错误: {str(e)}")
            time.sleep(2)
    
    print(f"\n✅ 监测完成")
    
    if latest_master_reply:
        print(f"📋 总结: 在监测期间收到了 {len([msg for msg in state['messages'] if msg['from'] == 'master' and msg['to'] == 'worker_lingma_enhanced'])} 条来自MASTER的消息")
        print(f"   最新回复时间: {time.strftime('%H:%M:%S', time.localtime(reply_found_time))}")
        print(f"   最新回复类型: {latest_master_reply['type']}")
        print(f"   最新回复内容: {latest_master_reply['data']}")
    else:
        print("   总结: 监测期间未收到MASTER的回复")
        
    return latest_master_reply


def main():
    """主函数"""
    print("📡 监测MASTER回复")
    print("等待MASTER确认任务状态...")
    
    # 监测3分钟
    latest_reply = monitor_master_reply(duration=180)
    
    if latest_reply:
        print(f"\n🎯 已收到MASTER回复，可根据内容调整工作方向")
        if latest_reply['type'] == 'task_obsolete':
            print("⚠️  收到任务过时通知，请勿继续执行原有任务")
        else:
            print("✅ 可以根据最新指示继续执行任务")
    else:
        print(f"\n⏰ 监测结束，未收到MASTER明确回复")
        print("ℹ️  建议继续执行既定任务，但保持对系统消息的监听")


if __name__ == "__main__":
    main()