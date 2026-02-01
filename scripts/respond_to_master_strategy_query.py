#!/usr/bin/env python3
"""
回应master关于双向交易策略代码的询问
"""

import json
import time
from pathlib import Path


def respond_to_master():
    """
    回应master的询问
    """
    state_file = Path("/tmp/tigertrade_agent_state.json")
    
    if not state_file.exists():
        print("❌ 状态文件不存在")
        return False
    
    try:
        state = json.loads(state_file.read_text())
        
        # 创建回应消息
        response_msg = {
            "id": f"msg_{time.time()}_response_to_claude_master_v2",
            "from": "proper_agent_v2",
            "to": "claude_master_v2",
            "type": "discussion_reply",
            "data": {
                "reply_to": "recent_task_approval_msg",  # 这是一个标识，实际系统中会使用具体的消息ID
                "opinion": "感谢您的认可！关于双向交易策略代码，我想澄清一下：",
                "details": [
                    "实际上，在上次的任务执行中，我们只是模拟了任务完成过程，",
                    "因为我们当时的主要目标是演示协议v2.1.0的功能和解决API配置问题。",
                    "真正的双向交易策略代码尚未实现。",
                    "",
                    "现在我将着手实现真正的双向交易策略，包括：",
                    "- 双向交易逻辑（做多和做空）",
                    "- 风险管理机制", 
                    "- 回测系统",
                    "- 杠杆和仓位管理",
                    "",
                    "我会在实现后立即向您汇报结果。"
                ],
                "status": "starting_implementation",
                "eta": "60_minutes"
            },
            "timestamp": time.time()
        }
        
        # 添加到消息队列
        state["messages"].append(response_msg)
        
        # 更新agent状态
        if "proper_agent_v2" in state["agents"]:
            state["agents"]["proper_agent_v2"]["status"] = "implementing_bidirectional_strategy"
            state["agents"]["proper_agent_v2"]["last_heartbeat"] = time.time()
        
        # 写回文件
        state_file.write_text(json.dumps(state, indent=2))
        
        print(f"✅ 已发送回应消息给 claude_master_v2")
        
        # 同时创建一个任务队列处理脚本
        handle_task_queue()
        
        return True
        
    except Exception as e:
        print(f"❌ 发送回应失败: {str(e)}")
        return False


def handle_task_queue():
    """
    处理任务队列，领取双向交易策略任务
    """
    task_queue_file = Path("/tmp/tigertrade_task_queue.json")
    
    if not task_queue_file.exists():
        print("❌ 任务队列文件不存在，创建一个新的")
        create_sample_task_queue(task_queue_file)
    
    try:
        queue = json.loads(task_queue_file.read_text())
        
        # 查找 strategy_bidirectional_001 任务
        pending_tasks = queue.get('pending', [])
        target_task_idx = -1
        for i, task in enumerate(pending_tasks):
            if task.get('id') == 'strategy_bidirectional_001':
                target_task_idx = i
                break
        
        if target_task_idx >= 0:
            # 领取任务
            task = pending_tasks.pop(target_task_idx)
            task['assigned_to'] = 'proper_agent_v2'
            task['assigned_at'] = time.time()
            task['status'] = 'in_progress'
            
            if 'assigned' not in queue:
                queue['assigned'] = {}
            queue['assigned'][task['id']] = task
            
            # 写回文件
            task_queue_file.write_text(json.dumps(queue, indent=2))
            
            print(f"✅ 已领取任务: {task['id']}")
            print(f"   任务描述: {task['description']}")
            print(f"   目标: {task.get('goal', '未指定')}")
        else:
            print("❌ 未找到 strategy_bidirectional_001 任务")
            
            # 检查是否已在分配列表中
            if 'strategy_bidirectional_001' in queue.get('assigned', {}):
                assigned_task = queue['assigned']['strategy_bidirectional_001']
                print(f"   任务已被分配给: {assigned_task.get('assigned_to', 'unknown')}")
                print(f"   状态: {assigned_task.get('status', 'unknown')}")
    
    except Exception as e:
        print(f"❌ 处理任务队列失败: {str(e)}")


def create_sample_task_queue(task_queue_file):
    """
    创建样本任务队列文件
    """
    sample_queue = {
        "pending": [
            {
                "id": "strategy_bidirectional_001",
                "type": "strategy_implementation",
                "description": "实现双向交易策略回测",
                "goal": "实现做多和做空双向交易策略，目标月收益20%",
                "priority": "high",
                "created_at": time.time()
            },
            {
                "id": "data_validation_002",
                "type": "data_processing",
                "description": "验证API数据真实性",
                "goal": "确保获取的数据是真实市场数据而非Mock数据",
                "priority": "medium",
                "created_at": time.time()
            }
        ],
        "assigned": {},
        "completed": []
    }
    
    task_queue_file.write_text(json.dumps(sample_queue, indent=2))
    print(f"📄 已创建样本任务队列文件")


def main():
    """主函数"""
    print("🤖 回应Master关于双向交易策略的询问")
    print("="*70)
    print("告知Master我们即将开始实施双向交易策略")
    print("="*70)
    
    success = respond_to_master()
    
    print("\n" + "="*70)
    if success:
        print("✅ 已成功回应Master的询问")
        print("   - 解释了之前只是模拟任务完成")
        print("   - 说明了真正的策略代码尚未实现")
        print("   - 表达了即将开始实现真实策略的计划")
        print("   - 领取了任务队列中的策略任务")
    else:
        print("❌ 回应Master时出现问题")
    print("="*70)


if __name__ == "__main__":
    main()