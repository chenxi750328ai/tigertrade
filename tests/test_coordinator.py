#!/usr/bin/env python3
"""
测试协调器
验证锁、消息、状态同步
"""

import sys
import os
import time
import threading
from src.coordinator import AgentCoordinator


def test_basic_lock():
    """测试基础锁功能"""
    print("\n" + "="*80)
    print("测试1: 基础锁功能")
    print("="*80)
    
    coord1 = AgentCoordinator("agent1", "测试Agent1")
    coord2 = AgentCoordinator("agent2", "测试Agent2")
    
    # Agent 1获取锁
    print("\n[Agent 1] 获取锁...")
    assert coord1.acquire_lock("test_resource", timeout=5.0)
    print("[Agent 1] ✅ 获取成功")
    
    # Agent 2尝试获取（应该失败）
    print("\n[Agent 2] 尝试获取同一资源...")
    assert not coord2.acquire_lock("test_resource", timeout=2.0)
    print("[Agent 2] ✅ 正确阻塞（资源被占用）")
    
    # Agent 1释放锁
    print("\n[Agent 1] 释放锁...")
    coord1.release_lock("test_resource")
    print("[Agent 1] ✅ 释放成功")
    
    # Agent 2再次尝试（应该成功）
    print("\n[Agent 2] 再次尝试...")
    assert coord2.acquire_lock("test_resource", timeout=5.0)
    print("[Agent 2] ✅ 获取成功")
    
    coord2.release_lock("test_resource")
    
    print("\n✅ 测试1通过！\n")


def test_message_queue():
    """测试消息队列"""
    print("\n" + "="*80)
    print("测试2: 消息队列")
    print("="*80)
    
    coord1 = AgentCoordinator("agent1", "发送者")
    coord2 = AgentCoordinator("agent2", "接收者")
    
    # 发送消息
    print("\n[Agent 1] 发送消息...")
    coord1.send_message("agent2", "test_message", {"data": "Hello Agent 2!"})
    print("[Agent 1] ✅ 消息已发送")
    
    # 接收消息
    print("\n[Agent 2] 接收消息...")
    messages = coord2.receive_messages("test_message")
    print(f"[Agent 2] ✅ 收到 {len(messages)} 条消息")
    
    assert len(messages) == 1
    assert messages[0]["type"] == "test_message"
    assert messages[0]["data"]["data"] == "Hello Agent 2!"
    print(f"[Agent 2] 消息内容: {messages[0]['data']}")
    
    print("\n✅ 测试2通过！\n")


def test_concurrent_access():
    """测试并发访问"""
    print("\n" + "="*80)
    print("测试3: 并发访问（模拟真实场景）")
    print("="*80)
    
    results = {"agent1": None, "agent2": None}
    
    def agent1_task():
        coord = AgentCoordinator("agent1", "Agent1")
        coord.update_status("working", "获取资源")
        
        print("\n[Agent 1] 尝试获取资源...")
        if coord.acquire_lock("shared_file", timeout=10.0):
            print("[Agent 1] ✅ 获取成功，处理中...")
            time.sleep(2)  # 模拟处理
            coord.release_lock("shared_file")
            print("[Agent 1] ✅ 处理完成，释放资源")
            
            # 通知Agent 2
            coord.send_message("agent2", "task_complete", {"task": "agent1_work"})
            results["agent1"] = "success"
        else:
            print("[Agent 1] ❌ 获取失败")
            results["agent1"] = "failed"
    
    def agent2_task():
        time.sleep(0.5)  # 延迟启动
        
        coord = AgentCoordinator("agent2", "Agent2")
        coord.update_status("waiting", "等待Agent1")
        
        print("\n[Agent 2] 尝试获取资源（Agent1持有中）...")
        if coord.acquire_lock("shared_file", timeout=10.0):
            print("[Agent 2] ✅ 获取成功（Agent1已释放）")
            time.sleep(1)
            coord.release_lock("shared_file")
            results["agent2"] = "success"
        else:
            print("[Agent 2] ❌ 超时")
            results["agent2"] = "failed"
        
        # 检查消息
        print("\n[Agent 2] 检查消息...")
        messages = coord.receive_messages("task_complete")
        if messages:
            print(f"[Agent 2] ✅ 收到Agent1的完成通知")
    
    # 启动两个线程
    t1 = threading.Thread(target=agent1_task)
    t2 = threading.Thread(target=agent2_task)
    
    t1.start()
    t2.start()
    
    t1.join()
    t2.join()
    
    assert results["agent1"] == "success"
    assert results["agent2"] == "success"
    
    print("\n✅ 测试3通过！\n")


def test_status_sync():
    """测试状态同步"""
    print("\n" + "="*80)
    print("测试4: 状态同步")
    print("="*80)
    
    coord1 = AgentCoordinator("agent1", "Worker")
    coord2 = AgentCoordinator("agent2", "Monitor")
    
    # Agent1更新状态
    print("\n[Agent 1] 更新状态...")
    coord1.update_status("working", "数据处理", 0.5)
    
    # Agent2查看所有状态
    print("\n[Agent 2] 查看所有Agent状态...")
    all_status = coord2.get_all_agents_status()
    
    print("\n所有Agent状态:")
    for agent_id, status in all_status.items():
        print(f"  {agent_id}: {status['status']} - {status.get('task', 'N/A')} ({status.get('progress', 0)*100:.0f}%)")
    
    assert "agent1" in all_status
    assert all_status["agent1"]["status"] == "working"
    assert all_status["agent1"]["task"] == "数据处理"
    
    print("\n✅ 测试4通过！\n")


def main():
    """运行所有测试"""
    print("\n" + "="*80)
    print("🧪 Agent协调器测试套件")
    print("="*80)
    
    try:
        test_basic_lock()
        test_message_queue()
        test_concurrent_access()
        test_status_sync()
        
        print("\n" + "="*80)
        print("🎉 所有测试通过！")
        print("="*80)
        print("\n协调器功能验证：")
        print("  ✅ 资源锁（互斥）")
        print("  ✅ 消息队列（通信）")
        print("  ✅ 并发访问（同步）")
        print("  ✅ 状态管理（监控）")
        print("\n🚀 可以安全用于多Agent协作！\n")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
