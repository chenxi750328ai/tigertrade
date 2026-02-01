#!/usr/bin/env python3
"""
持续监控脚本
保持与Master的持续沟通，定期发送心跳并监控任务分配
"""

import json
import time
import sys
from pathlib import Path


def update_heartbeat(agent_id="proper_agent_v2"):
    """
    更新心跳，表明我们仍然在线
    """
    state_file = Path("/tmp/tigertrade_agent_state.json")
    
    if state_file.exists():
        try:
            state = json.loads(state_file.read_text())
            
            # 更新agent的心跳
            if agent_id in state["agents"]:
                state["agents"][agent_id]["last_heartbeat"] = time.time()
                state["agents"][agent_id]["status"] = "monitoring_and_ready"
                
                state_file.write_text(json.dumps(state, indent=2))
                
                return True
        except Exception as e:
            print(f"⚠️ 更新心跳失败: {str(e)}")
    
    return False


def check_for_assignments(agent_id="proper_agent_v2"):
    """
    检查是否有分配给我们的任务
    """
    state_file = Path("/tmp/tigertrade_agent_state.json")
    queue_file = Path("/tmp/tigertrade_task_queue.json")
    
    assignments = []
    
    # 检查状态文件中的消息
    if state_file.exists():
        try:
            state = json.loads(state_file.read_text())
            
            # 检查是否有发给我们的消息
            our_messages = [
                msg for msg in state['messages']
                if msg['to'] == agent_id and msg['type'] in ['task_assign', 'guidance', 'task_approved', 'task_rejected']
            ]
            
            assignments.extend(our_messages)
        except Exception as e:
            print(f"⚠️ 检查状态文件时出错: {str(e)}")
    
    # 检查任务队列
    if queue_file.exists():
        try:
            queue = json.loads(queue_file.read_text())
            
            # 检查分配给我们的任务
            for task_id, task in queue.get('assigned', {}).items():
                if task.get('assigned_to') == agent_id:
                    assignments.append({
                        'type': 'task_assigned',
                        'task_id': task_id,
                        'task': task
                    })
        except Exception as e:
            print(f"⚠️ 检查任务队列时出错: {str(e)}")
    
    return assignments


def send_periodic_status(agent_id="proper_agent_v2"):
    """
    定期发送状态更新
    """
    state_file = Path("/tmp/tigertrade_agent_state.json")
    
    if not state_file.exists():
        print("❌ 状态文件不存在")
        return False
    
    try:
        state = json.loads(state_file.read_text())
        
        # 获取当前master
        current_master = state.get('current_master', 'master')
        
        # 发送周期性状态更新消息
        status_msg = {
            "id": f"msg_{time.time()}_periodic_status_{agent_id}",
            "from": agent_id,
            "to": current_master,
            "type": "periodic_status",
            "data": {
                "status": "continuously_monitoring",
                "message": f"{agent_id} 持续监控中，随时准备接受任务",
                "last_task_request": "sent_recently",
                "availability": "immediate",
                "capabilities_confirmed": [
                    "strategy_optimization",
                    "model_evaluation", 
                    "backtesting",
                    "risk_management",
                    "data_analysis",
                    "cross_machine_collaboration"
                ],
                "system_integrity": "operational",
                "timestamp": time.time()
            },
            "timestamp": time.time()
        }
        
        # 添加到消息队列
        state["messages"].append(status_msg)
        
        # 更新agent状态
        if agent_id in state["agents"]:
            state["agents"][agent_id]["status"] = "monitoring_continuously"
            state["agents"][agent_id]["last_heartbeat"] = time.time()
        
        # 写回文件
        state_file.write_text(json.dumps(state, indent=2))
        
        return True
        
    except Exception as e:
        print(f"❌ 发送周期性状态更新失败: {str(e)}")
        return False


def main():
    """主函数"""
    print("🔄 持续监控与沟通系统")
    print("="*60)
    print("保持与Master的持续沟通，定期发送心跳并监控任务分配")
    print("按 Ctrl+C 停止监控")
    print("="*60)
    
    agent_id = "proper_agent_v2"
    iteration_count = 0
    
    try:
        while True:
            iteration_count += 1
            
            # 更新心跳
            heartbeat_ok = update_heartbeat(agent_id)
            
            # 检查是否有新分配的任务
            assignments = check_for_assignments(agent_id)
            
            if assignments:
                print(f"\n🎉 检测到 {len(assignments)} 个新消息/任务:")
                for assignment in assignments:
                    if 'type' in assignment and assignment['type'] == 'task_assigned':
                        print(f"   📋 任务分配: {assignment['task']['description']}")
                    else:
                        print(f"   📨 消息: {assignment['type']} - {str(assignment['data'])[:100]}...")
                
                print("\n" + "="*60)
                print("✅ 检测到任务分配！监控系统将退出，准备处理任务")
                print("="*60)
                break
            else:
                print(f"⏳ {time.strftime('%H:%M:%S')} - 第 {iteration_count} 次检查，暂无新任务分配")
            
            # 发送周期性状态更新（每5次迭代发送一次，即每25秒）
            if iteration_count % 5 == 0:
                status_ok = send_periodic_status(agent_id)
                if status_ok:
                    print(f"   📊 周期性状态已发送 #{iteration_count//5}")
            
            # 等待5秒后继续
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n🛑 用户中断监控")
        print("监控系统已停止，但系统仍保持在线状态")


if __name__ == "__main__":
    main()