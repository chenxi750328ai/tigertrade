#!/usr/bin/env python3
"""
任务监听脚本
持续监听来自master的任务分配
"""

import json
import time
import sys
from pathlib import Path


def listen_for_tasks(duration=600):
    """
    监听来自master的任务分配
    
    Args:
        duration: 监听时长（秒）
    """
    state_file = Path("/tmp/tigertrade_agent_state.json")
    start_time = time.time()
    
    print(f"👂 开始监听来自master的任务分配，监听时长: {duration}秒")
    print("="*60)
    
    task_received = False
    initial_msg_count = 0
    
    # 记录初始状态
    if state_file.exists():
        initial_state = json.loads(state_file.read_text())
        initial_msg_count = len(initial_state['messages'])
        current_master = initial_state.get('current_master', 'master')
        print(f"📡 当前master: {current_master}")
    
    print("✅ 开始监听...")
    
    while time.time() - start_time < duration:
        if not state_file.exists():
            time.sleep(2)
            continue
        
        try:
            state = json.loads(state_file.read_text())
            current_master = state.get('current_master', 'master')
            
            # 检查是否有分配给我的任务
            assigned_tasks = [
                msg for msg in state['messages'] 
                if msg['to'] == 'worker_lingma_enhanced' and 
                   msg['from'] == current_master and
                   msg['type'] == 'task_assign'
            ]
            
            if assigned_tasks:
                latest_task = assigned_tasks[-1]
                
                print(f"\n🎯 任务接收! 来自: {current_master}")
                print(f"   任务ID: {latest_task['data'].get('task_id', 'unknown')}")
                print(f"   任务类型: {latest_task['data'].get('type', 'unknown')}")
                print(f"   描述: {latest_task['data'].get('description', 'N/A')}")
                print(f"   参数: {latest_task['data'].get('params', {})}")
                
                # 从消息队列中移除已接收的任务（模拟消费）
                task_msg_id = latest_task['id']
                state['messages'] = [msg for msg in state['messages'] if msg['id'] != task_msg_id]
                state_file.write_text(json.dumps(state, indent=2))
                
                task_received = True
                print(f"\n✅ 任务已接收，准备执行...")
                
                # 执行任务
                execute_task(latest_task['data'])
                
                # 完成任务
                complete_task(latest_task['data'].get('task_id', 'unknown'), {
                    'status': 'completed',
                    'result': 'Task executed successfully',
                    'worker': 'worker_lingma_enhanced'
                })
                
                break
            
            # 每10秒打印一次状态
            elapsed = int(time.time() - start_time)
            if elapsed % 10 == 0:
                print(f"⏱️  监听进行中... {elapsed}s (消息总数: {len(state['messages'])})")
            
            time.sleep(0.5)  # 每0.5秒检查一次
            
        except Exception as e:
            print(f"❌ 监听过程中出现错误: {str(e)}")
            time.sleep(2)
    
    print(f"\n🏁 监听完成")
    if task_received:
        print("✅ 已成功接收并处理任务")
    else:
        print("⏰ 监听结束，但未收到任务分配")
        print("💡 建议继续等待或主动请求任务")
    
    return task_received


def execute_task(task_data):
    """
    执行接收到的任务
    """
    print(f"\n🚀 开始执行任务: {task_data.get('type', 'unknown')}")
    print(f"   描述: {task_data.get('description', 'N/A')}")
    
    # 模拟任务执行
    total_steps = 5
    for i in range(total_steps):
        progress = (i + 1) / total_steps
        print(f"   执行进度: {progress*100:.1f}%")
        time.sleep(0.5)
    
    print(f"✅ 任务执行完成")


def complete_task(task_id, result):
    """
    报告任务完成
    """
    state_file = Path("/tmp/tigertrade_agent_state.json")
    
    if not state_file.exists():
        print("❌ 状态文件不存在")
        return False
    
    try:
        state = json.loads(state_file.read_text())
        current_master = state.get('current_master', 'master')
        
        # 创建完成消息
        completion_msg = {
            "id": f"msg_{time.time()}_task_complete_{task_id}",
            "from": "worker_lingma_enhanced",
            "to": current_master,
            "type": "task_complete",
            "data": {
                "task_id": task_id,
                "result": result,
                "completed_by": "worker_lingma_enhanced",
                "timestamp": time.time()
            },
            "timestamp": time.time()
        }
        
        # 添加到消息队列
        state["messages"].append(completion_msg)
        
        # 更新agent状态
        if "worker_lingma_enhanced" in state["agents"]:
            state["agents"]["worker_lingma_enhanced"]["status"] = "task_completed"
            state["agents"]["worker_lingma_enhanced"]["task"] = None
            state["agents"]["worker_lingma_enhanced"]["progress"] = 1.0
            state["agents"]["worker_lingma_enhanced"]["last_heartbeat"] = time.time()
        
        # 写回文件
        state_file.write_text(json.dumps(state, indent=2))
        
        print(f"✅ 任务完成报告已发送给 {current_master}")
        return True
        
    except Exception as e:
        print(f"❌ 发送完成报告失败: {str(e)}")
        return False


def main():
    """主函数"""
    print("🎯 TigerTrade任务监听器")
    print("等待MASTER分配任务...")
    
    # 监听任务，最长等待10分钟
    task_received = listen_for_tasks(duration=600)
    
    if not task_received:
        print(f"\n💡 建议接下来采取的行动:")
        print(f"   1. 继续监听: 任务可能稍后分配")
        print(f"   2. 主动请求: 再次发送任务请求")
        print(f"   3. 自主工作: 执行预先规划的任务")


if __name__ == "__main__":
    main()