#!/usr/bin/env python3
"""
等待任务分配监控脚本
持续监控系统状态，等待Master分配任务
"""

import json
import time
import sys
from pathlib import Path


def monitor_system_for_assignments(agent_id="proper_agent_v2"):
    """
    监控系统状态，查找分配给我们的任务
    """
    print(f"👀 监控系统状态，等待分配给 {agent_id} 的任务...")
    
    state_file = Path("/tmp/tigertrade_agent_state.json")
    
    # 任务队列文件
    queue_file = Path("/tmp/tigertrade_task_queue.json")
    
    while True:
        try:
            # 检查状态文件
            if state_file.exists():
                state = json.loads(state_file.read_text())
                
                # 检查是否有发给我们的消息
                our_messages = [
                    msg for msg in state['messages']
                    if msg['to'] == agent_id and msg['type'] in ['task_assign', 'guidance', 'task_approved']
                ]
                
                if our_messages:
                    print(f"\n📬 检测到 {len(our_messages)} 条发给我们的消息:")
                    for msg in our_messages:
                        print(f"   📌 {msg['type']}: {msg['data']}")
                        
                        # 如果是任务分配，退出监控
                        if msg['type'] == 'task_assign':
                            print(f"\n✅ 检测到任务分配! 退出监控...")
                            return msg
                        
                        # 如果是指导消息，也显示出来
                        if msg['type'] == 'guidance':
                            print(f"💡 指导信息: {msg['data'].get('message', 'N/A')}")
                
                # 检查任务队列
                if queue_file.exists():
                    queue = json.loads(queue_file.read_text())
                    
                    # 检查分配给我们的任务
                    assigned_to_us = {}
                    for task_id, task in queue.get('assigned', {}).items():
                        if task.get('assigned_to') == agent_id:
                            assigned_to_us[task_id] = task
                    
                    if assigned_to_us:
                        print(f"\n✅ 检测到 {len(assigned_to_us)} 个分配给我们的任务:")
                        for task_id, task in assigned_to_us.items():
                            print(f"   📋 任务 {task_id}: {task['description']}")
                            print(f"      状态: {task.get('status', 'N/A')}, 进度: {task.get('progress', 0)*100}%")
                            
                        return assigned_to_us
            
            print(f"⏳ {time.strftime('%H:%M:%S')} - 未检测到分配给我们的任务，继续监控...")
            time.sleep(5)  # 每5秒检查一次
            
        except KeyboardInterrupt:
            print("\n🛑 用户中断监控")
            break
        except Exception as e:
            print(f"⚠️ 检查时出现错误: {str(e)}")
            time.sleep(5)


def update_heartbeat(agent_id="proper_agent_v2"):
    """
    更新心跳，表明我们仍然在线并准备好接收任务
    """
    state_file = Path("/tmp/tigertrade_agent_state.json")
    
    if state_file.exists():
        try:
            state = json.loads(state_file.read_text())
            
            # 更新agent的心跳
            if agent_id in state["agents"]:
                state["agents"][agent_id]["last_heartbeat"] = time.time()
                state["agents"][agent_id]["status"] = "waiting_for_task"
                
                state_file.write_text(json.dumps(state, indent=2))
                
        except Exception as e:
            print(f"⚠️ 更新心跳失败: {str(e)}")


def main():
    """主函数"""
    print("👀 等待任务分配监控系统")
    print("="*60)
    print("持续监控系统状态，等待Master分配任务")
    print("按 Ctrl+C 停止监控")
    print("="*60)
    
    agent_id = "proper_agent_v2"
    
    # 发送一个状态更新消息，表明我们正在等待任务
    state_file = Path("/tmp/tigertrade_agent_state.json")
    if state_file.exists():
        try:
            state = json.loads(state_file.read_text())
            
            # 获取当前master
            current_master = state.get('current_master', 'master')
            
            # 发送状态更新消息
            status_msg = {
                "id": f"msg_{time.time()}_status_update",
                "from": agent_id,
                "to": current_master,
                "type": "status_update",
                "data": {
                    "status": "waiting_for_task_assignment",
                    "capabilities": [
                        "strategy_optimization",
                        "model_evaluation", 
                        "backtesting",
                        "risk_management",
                        "cross_machine_collaboration"
                    ],
                    "ready_immediately": True,
                    "last_integration_step": "system_verification_completed",
                    "timestamp": time.time()
                },
                "timestamp": time.time()
            }
            
            # 添加到消息队列
            state["messages"].append(status_msg)
            
            # 更新agent状态
            if agent_id in state["agents"]:
                state["agents"][agent_id]["status"] = "waiting_for_task"
                state["agents"][agent_id]["last_heartbeat"] = time.time()
            
            # 写回文件
            state_file.write_text(json.dumps(state, indent=2))
            
            print(f"✅ 状态更新消息已发送，表明我们正在等待任务分配")
            
        except Exception as e:
            print(f"❌ 发送状态更新消息失败: {str(e)}")
    
    # 开始监控
    print(f"\n🚀 开始监控任务分配...")
    assignment = monitor_system_for_assignments(agent_id)
    
    if assignment:
        print("\n🎉 成功检测到任务分配!")
        print("="*60)
        print("现在可以开始处理分配的任务")
        print("="*60)
    else:
        print("\nℹ️  监控已停止，没有检测到任务分配")
        print("系统将继续保持在线状态，随时准备接收任务")


if __name__ == "__main__":
    main()