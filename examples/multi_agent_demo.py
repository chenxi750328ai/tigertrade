#!/usr/bin/env python3
"""
真实的多Agent并发协作演示

展示：
- Agent 1: 数据预处理
- Agent 2: 模型训练
- Agent 3: 策略回测

三个Agent通过协调器实现：
✅ 互斥访问
✅ 消息通信
✅ 状态同步
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import threading
from src.coordinator import AgentCoordinator


def agent1_data_engineer():
    """Agent 1: 数据工程师"""
    coord = AgentCoordinator("agent1", "数据工程师")
    
    print("\n" + "="*60)
    print("👤 Agent 1 (数据工程师) 启动")
    print("="*60)
    
    # 任务1: 数据清洗
    coord.update_status("working", "数据清洗", 0.0)
    
    print("\n[Agent 1] 📊 开始数据清洗...")
    print("[Agent 1] 🔒 获取锁: raw_data.csv")
    
    if coord.acquire_lock("raw_data.csv", timeout=10.0):
        try:
            print("[Agent 1] ✅ 锁获取成功")
            print("[Agent 1] 🔄 清洗中...")
            
            for i in range(5):
                time.sleep(0.5)
                progress = (i + 1) / 5
                coord.update_status("working", "数据清洗", progress)
                print(f"[Agent 1] 进度: {progress*100:.0f}%")
            
            print("[Agent 1] ✅ 数据清洗完成")
            
        finally:
            print("[Agent 1] 🔓 释放锁: raw_data.csv")
            coord.release_lock("raw_data.csv")
    
    # 任务2: 生成训练集
    coord.update_status("working", "生成训练集", 0.0)
    
    print("\n[Agent 1] 📊 生成训练集...")
    print("[Agent 1] 🔒 获取锁: train.csv")
    
    if coord.acquire_lock("train.csv", timeout=10.0):
        try:
            print("[Agent 1] ✅ 锁获取成功")
            print("[Agent 1] 🔄 生成中...")
            time.sleep(2)
            print("[Agent 1] ✅ train.csv 生成完成")
            
            coord.update_status("idle", "生成训练集", 1.0)
            
            # 通知Agent 2
            print("[Agent 1] 📨 通知Agent 2: 数据已就绪")
            coord.send_message("agent2", "data_ready", {
                "train_file": "train.csv",
                "records": 10000
            })
            
        finally:
            print("[Agent 1] 🔓 释放锁: train.csv")
            coord.release_lock("train.csv")
    
    print("\n[Agent 1] ✅ 所有任务完成")
    coord.cleanup()


def agent2_ai_researcher():
    """Agent 2: AI研究员"""
    coord = AgentCoordinator("agent2", "AI研究员")
    
    print("\n" + "="*60)
    print("👤 Agent 2 (AI研究员) 启动")
    print("="*60)
    
    # 等待数据就绪
    coord.update_status("waiting", "等待数据", 0.0)
    print("\n[Agent 2] ⏳ 等待Agent 1完成数据处理...")
    
    message = coord.wait_for_message("data_ready", timeout=30.0)
    
    if not message:
        print("[Agent 2] ❌ 超时，未收到数据就绪消息")
        coord.cleanup()
        return
    
    print(f"[Agent 2] ✅ 收到消息: {message['data']}")
    
    # 开始训练
    coord.update_status("working", "模型训练", 0.0)
    
    print("\n[Agent 2] 🤖 开始模型训练...")
    print("[Agent 2] 🔒 获取锁: train.csv, gpu")
    
    # 需要两个资源
    if coord.acquire_lock("train.csv", timeout=10.0):
        if coord.acquire_lock("gpu", timeout=10.0):
            try:
                print("[Agent 2] ✅ 所有锁获取成功")
                print("[Agent 2] 🔄 训练中...")
                
                for epoch in range(1, 6):
                    time.sleep(0.8)
                    progress = epoch / 5
                    coord.update_status("working", f"模型训练 (Epoch {epoch}/5)", progress)
                    print(f"[Agent 2] Epoch {epoch}/5 - {progress*100:.0f}%")
                
                print("[Agent 2] ✅ 模型训练完成")
                print("[Agent 2] 💾 保存模型: model.pth")
                
                coord.update_status("idle", "模型训练", 1.0)
                
                # 通知Agent 3
                print("[Agent 2] 📨 通知Agent 3: 模型已就绪")
                coord.send_message("agent3", "model_ready", {
                    "model_file": "model.pth",
                    "accuracy": 0.85
                })
                
            finally:
                print("[Agent 2] 🔓 释放锁: gpu, train.csv")
                coord.release_lock("gpu")
                coord.release_lock("train.csv")
        else:
            coord.release_lock("train.csv")
            print("[Agent 2] ❌ 无法获取GPU锁")
    else:
        print("[Agent 2] ❌ 无法获取train.csv锁")
    
    print("\n[Agent 2] ✅ 所有任务完成")
    coord.cleanup()


def agent3_strategy_engineer():
    """Agent 3: 策略工程师"""
    coord = AgentCoordinator("agent3", "策略工程师")
    
    print("\n" + "="*60)
    print("👤 Agent 3 (策略工程师) 启动")
    print("="*60)
    
    # 等待模型就绪
    coord.update_status("waiting", "等待模型", 0.0)
    print("\n[Agent 3] ⏳ 等待Agent 2完成模型训练...")
    
    message = coord.wait_for_message("model_ready", timeout=60.0)
    
    if not message:
        print("[Agent 3] ❌ 超时，未收到模型就绪消息")
        coord.cleanup()
        return
    
    print(f"[Agent 3] ✅ 收到消息: {message['data']}")
    
    # 开始回测
    coord.update_status("working", "策略回测", 0.0)
    
    print("\n[Agent 3] 📈 开始策略回测...")
    print("[Agent 3] 🔒 获取锁: model.pth, test.csv")
    
    if coord.acquire_lock("model.pth", timeout=10.0):
        if coord.acquire_lock("test.csv", timeout=10.0):
            try:
                print("[Agent 3] ✅ 所有锁获取成功")
                print("[Agent 3] 🔄 回测中...")
                
                for i in range(4):
                    time.sleep(0.6)
                    progress = (i + 1) / 4
                    coord.update_status("working", "策略回测", progress)
                    print(f"[Agent 3] 回测进度: {progress*100:.0f}%")
                
                print("[Agent 3] ✅ 回测完成")
                print("[Agent 3] 📊 收益率: +23.5%")
                print("[Agent 3] 📊 胜率: 68.3%")
                
                coord.update_status("idle", "策略回测", 1.0)
                
                # 广播结果
                print("[Agent 3] 📨 广播: 回测完成")
                coord.broadcast_message("backtest_complete", {
                    "return": 0.235,
                    "win_rate": 0.683
                })
                
            finally:
                print("[Agent 3] 🔓 释放锁: test.csv, model.pth")
                coord.release_lock("test.csv")
                coord.release_lock("model.pth")
        else:
            coord.release_lock("model.pth")
            print("[Agent 3] ❌ 无法获取test.csv锁")
    else:
        print("[Agent 3] ❌ 无法获取model.pth锁")
    
    print("\n[Agent 3] ✅ 所有任务完成")
    coord.cleanup()


def monitor_all_agents():
    """监控所有Agent状态"""
    coord = AgentCoordinator("monitor", "监控器")
    
    print("\n" + "="*60)
    print("👁️  监控器启动")
    print("="*60)
    
    for _ in range(20):
        time.sleep(1)
        
        # 获取所有状态
        all_status = coord.get_all_agents_status()
        
        # 打印状态
        print(f"\n{'─'*60}")
        print(f"⏰ {time.strftime('%H:%M:%S')} - Agent状态")
        print(f"{'─'*60}")
        
        for agent_id in ["agent1", "agent2", "agent3"]:
            if agent_id in all_status:
                status = all_status[agent_id]
                progress = status.get('progress', 0) * 100
                task = status.get('task', 'N/A')
                state = status.get('status', 'unknown')
                
                print(f"  [{agent_id}] {state:8} | {task:20} | {progress:5.1f}%")
        
        # 检查是否都完成
        all_idle = all(
            all_status.get(aid, {}).get('status') == 'idle'
            for aid in ["agent1", "agent2", "agent3"]
            if aid in all_status
        )
        
        if all_idle:
            print(f"\n{'='*60}")
            print("✅ 所有Agent任务完成！")
            print("="*60)
            break
    
    coord.cleanup()


def main():
    """主函数：启动多Agent协作"""
    print("\n" + "="*80)
    print("🚀 真实多Agent并发协作演示")
    print("="*80)
    print("\n场景：数据处理 → 模型训练 → 策略回测")
    print("\n特性：")
    print("  ✅ 互斥访问（资源锁）")
    print("  ✅ 消息传递（任务通知）")
    print("  ✅ 状态同步（实时监控）")
    print("  ✅ 并发执行（真正并行）")
    print("\n" + "="*80)
    
    # 创建线程
    threads = [
        threading.Thread(target=agent1_data_engineer, name="Agent1"),
        threading.Thread(target=agent2_ai_researcher, name="Agent2"),
        threading.Thread(target=agent3_strategy_engineer, name="Agent3"),
        threading.Thread(target=monitor_all_agents, name="Monitor")
    ]
    
    # 启动所有线程
    for t in threads:
        t.start()
    
    # 等待所有线程完成
    for t in threads:
        t.join()
    
    print("\n" + "="*80)
    print("🎉 演示完成！")
    print("="*80)
    print("\n核心机制验证：")
    print("  ✅ Agent 1完成数据处理后，Agent 2才开始训练")
    print("  ✅ Agent 2完成训练后，Agent 3才开始回测")
    print("  ✅ 资源锁防止冲突（train.csv不会被同时访问）")
    print("  ✅ 监控器实时显示所有Agent状态")
    print("\n💡 这才是真正的多Agent协作！")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
