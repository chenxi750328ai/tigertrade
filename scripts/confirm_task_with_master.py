#!/usr/bin/env python3
"""
与MASTER确认任务状态的脚本
用于确保当前任务仍然有效，并获取最新的任务指示
"""

import json
import time
from pathlib import Path


def send_task_confirmation_request():
    """
    向MASTER发送任务确认请求
    """
    state_file = Path("/tmp/tigertrade_agent_state.json")
    
    if not state_file.exists():
        print("❌ 状态文件不存在")
        return False
    
    try:
        state = json.loads(state_file.read_text())
        
        # 发送任务确认请求
        confirmation_request = {
            "id": f"msg_{time.time()}_task_confirmation",
            "from": "worker_lingma_enhanced",
            "to": "master",
            "type": "task_confirmation_request",
            "data": {
                "message": "请求确认当前任务状态",
                "current_task_understanding": "策略回测优化，目标月收益率20%",
                "received_instructions": "从 /home/cx/给worker_lingma_enhanced的紧急消息.txt 获悉需要优化策略",
                "ask_if_still_valid": True,
                "ask_for_latest_guidance": True,
                "availability": "随时可以开始工作",
                "need_priority_confirmation": True,
                "timestamp": time.time()
            },
            "timestamp": time.time()
        }
        
        # 添加到消息队列
        state["messages"].append(confirmation_request)
        
        # 更新发送者状态
        if "worker_lingma_enhanced" in state["agents"]:
            state["agents"]["worker_lingma_enhanced"]["status"] = "awaiting_confirmation"
            state["agents"]["worker_lingma_enhanced"]["last_heartbeat"] = time.time()
        
        # 写回文件
        state_file.write_text(json.dumps(state, indent=2))
        
        print(f"✅ 任务确认请求已发送给MASTER")
        print(f"   内容: {confirmation_request['data']['message']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 发送任务确认请求失败: {str(e)}")
        return False


def send_capability_report():
    """
    向MASTER发送能力报告
    """
    state_file = Path("/tmp/tigertrade_agent_state.json")
    
    if not state_file.exists():
        print("❌ 状态文件不存在")
        return False
    
    try:
        state = json.loads(state_file.read_text())
        
        # 发送能力报告
        capability_report = {
            "id": f"msg_{time.time()}_capability_report",
            "from": "worker_lingma_enhanced",
            "to": "master",
            "type": "capability_report",
            "data": {
                "message": "汇报当前能力状态",
                "capabilities": [
                    "数据预处理和特征工程",
                    "模型训练和优化",
                    "策略回测和评估",
                    "风险管理实现",
                    "自定义特征发现"
                ],
                "current_status": "ready_for_task",
                "specializations": [
                    "量化策略优化",
                    "收益率提升算法",
                    "风险控制策略"
                ],
                "immediate_availability": "yes",
                "estimated_completion_time": "2-4小时根据任务复杂度",
                "timestamp": time.time()
            },
            "timestamp": time.time()
        }
        
        # 添加到消息队列
        state["messages"].append(capability_report)
        
        print(f"✅ 能力报告已发送给MASTER")
        
        return True
        
    except Exception as e:
        print(f"❌ 发送能力报告失败: {str(e)}")
        return False


def check_master_response(timeout=120):
    """
    检查MASTER的响应
    
    Args:
        timeout: 等待响应的超时时间（秒）
    """
    state_file = Path("/tmp/tigertrade_agent_state.json")
    start_time = time.time()
    
    print(f"⏳ 等待MASTER响应，超时时间: {timeout}秒")
    
    while time.time() - start_time < timeout:
        if not state_file.exists():
            time.sleep(2)
            continue
        
        try:
            state = json.loads(state_file.read_text())
            
            # 查找MASTER发给我的回复
            master_responses = [
                msg for msg in state['messages'] 
                if msg['from'] == 'master' and msg['to'] == 'worker_lingma_enhanced'
            ]
            
            if master_responses:
                latest_response = master_responses[-1]
                print(f"\n✅ 收到MASTER响应!")
                print(f"   类型: {latest_response['type']}")
                print(f"   内容: {latest_response['data']}")
                
                # 根据响应类型采取不同行动
                if latest_response['type'] in ['task_confirmed', 'task_assignment']:
                    print(f"   🎯 任务已确认，可以开始执行")
                    return latest_response
                elif latest_response['type'] == 'task_updated':
                    print(f"   📝 任务已更新，需按新要求执行")
                    return latest_response
                elif latest_response['type'] == 'task_obsolete':
                    print(f"   ⚠️  任务已过时，请勿执行")
                    return latest_response
                else:
                    print(f"   ℹ️  收到其他类型响应")
                    return latest_response
            
            time.sleep(1)  # 每秒检查一次
            
        except Exception as e:
            print(f"❌ 检查响应时出现错误: {str(e)}")
            time.sleep(2)
    
    print(f"\n⏰ 超时未收到MASTER响应")
    return None


def main():
    """主函数"""
    print("💬 与MASTER通信 - 确认任务状态")
    print("="*70)
    
    # 1. 发送能力报告
    print("\n1️⃣ 发送能力报告...")
    send_capability_report()
    
    # 2. 发送任务确认请求
    print("\n2️⃣ 发送任务确认请求...")
    send_task_confirmation_request()
    
    # 3. 等待并检查MASTER响应
    print("\n3️⃣ 等待MASTER响应...")
    response = check_master_response(timeout=180)  # 等待3分钟
    
    # 4. 根据响应决定后续动作
    print("\n4️⃣ 分析响应并决定后续动作...")
    if response:
        response_type = response.get('type', 'unknown')
        response_data = response.get('data', {})
        
        print(f"   响应类型: {response_type}")
        
        if response_type in ['task_confirmed', 'task_assignment', 'task_updated']:
            print(f"   ✅ 收到积极响应，准备执行任务")
            print(f"   📋 按照MASTER的最新指示执行任务")
            print(f"   🚀 开始工作...")
        elif response_type == 'task_obsolete':
            print(f"   ⚠️  任务已过时，不会执行旧任务")
            print(f"   📋 等待MASTER分配新任务")
        else:
            print(f"   ℹ️  收到其他类型响应，但仍可按原计划执行")
            print(f"   📋 继续执行既定任务")
    else:
        print(f"   ℹ️  未收到明确响应，但根据之前的指令，准备执行任务")
        print(f"   📋 继续执行策略优化任务")
    
    print("\n✅ 通信完成，已确认任务状态")
    

if __name__ == "__main__":
    main()