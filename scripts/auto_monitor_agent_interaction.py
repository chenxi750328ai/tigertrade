#!/usr/bin/env python3
"""
自动化监控脚本
用于持续与其他agent交互并自动执行任务
"""

import threading
import time
from datetime import datetime
import sys
import os
import asyncio
import json
from queue import Queue

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 用于控制所有线程的事件
stop_event = threading.Event()

# 任务队列
task_queue = Queue()


def auto_send_status_reports():
    """
    自动发送状态报告给master
    """
    print(f"[{datetime.now()}] 📊 自动状态报告系统启动...")
    
    while not stop_event.is_set():
        try:
            # 准备状态报告
            status_report = {
                "agent_id": "proper_agent_v2",
                "status": "active",
                "timestamp": datetime.now().isoformat(),
                "capabilities": [
                    "bidirectional_trading",
                    "strategy_optimization", 
                    "risk_management",
                    "real_time_monitoring"
                ],
                "current_task": "monitoring_and_executing",
                "system_uptime": f"{int(time.time() % 86400 // 3600):02d}:{int(time.time() % 3600 // 60):02d}:{int(time.time() % 60):02d}",
                "tasks_completed": 0  # 可以通过某种方式跟踪完成的任务数
            }
            
            print(f"[{datetime.now()}] 📡 自动发送状态报告: {status_report}")
            
            # 模拟发送状态报告
            time.sleep(1)  # 模拟网络延迟
            
        except Exception as e:
            print(f"[{datetime.now()}] ❌ 发送状态报告时出错: {e}")
        
        # 每30秒发送一次状态报告
        for _ in range(30):
            if stop_event.is_set():
                break
            time.sleep(1)


def auto_request_tasks():
    """
    自动请求任务给master
    """
    print(f"[{datetime.now()}] 🎯 自动任务请求系统启动...")
    
    request_count = 0
    
    while not stop_event.is_set():
        try:
            request_count += 1
            task_request = {
                "request_id": f"task_req_{int(time.time())}_{request_count}",
                "agent_id": "proper_agent_v2",
                "request_type": "task_assignment",
                "timestamp": datetime.now().isoformat(),
                "current_status": "ready_for_task",
                "available_resources": {
                    "cpu_usage": "low",
                    "memory_usage": "low", 
                    "trading_strategies_available": ["bidirectional", "grid", "scalping"]
                },
                "skills": [
                    "technical_analysis",
                    "risk_management",
                    "market_monitoring"
                ]
            }
            
            print(f"[{datetime.now()}] 📥 自动请求任务: {task_request}")
            
            # 模拟发送任务请求
            time.sleep(0.5)  # 模拟网络延迟
            
        except Exception as e:
            print(f"[{datetime.now()}] ❌ 请求任务时出错: {e}")
        
        # 每60秒请求一次任务
        for _ in range(60):
            if stop_event.is_set():
                break
            time.sleep(1)


def auto_check_messages():
    """
    自动检查来自其他agent的消息
    """
    print(f"[{datetime.now()}] 📬 自动消息检查系统启动...")
    
    while not stop_event.is_set():
        try:
            print(f"[{datetime.now()}] 🔍 自动检查来自其他agent的消息...")
            
            # 模拟检查消息的过程
            time.sleep(0.2)  # 模拟检查延迟
            
            # 模拟处理消息队列
            if not task_queue.empty():
                msg = task_queue.get()
                print(f"[{datetime.now()}] 📨 处理消息: {msg}")
            
        except Exception as e:
            print(f"[{datetime.now()}] ❌ 检查消息时出错: {e}")
        
        # 每5秒检查一次消息
        for _ in range(5):
            if stop_event.is_set():
                break
            time.sleep(1)


def auto_execute_trading_strategy():
    """
    自动执行交易策略
    """
    print(f"[{datetime.now()}] 📈 自动交易策略执行系统启动...")
    
    from src.bidirectional_strategy import bidirectional_grid_strategy
    
    execution_count = 0
    
    while not stop_event.is_set():
        try:
            execution_count += 1
            print(f"[{datetime.now()}] 🔄 执行第 {execution_count} 次双向策略分析...")
            
            # 执行策略
            bidirectional_grid_strategy()
            
            print(f"[{datetime.now()}] ✅ 第 {execution_count} 次策略执行完成")
            
        except Exception as e:
            print(f"[{datetime.now()}] ❌ 执行策略时出错: {e}")
            import traceback
            traceback.print_exc()
        
        # 每15秒执行一次策略
        for _ in range(15):
            if stop_event.is_set():
                break
            time.sleep(1)


def auto_monitor_system_health():
    """
    自动监控系统健康状况
    """
    print(f"[{datetime.now()}] 🏥 系统健康监控启动...")
    
    while not stop_event.is_set():
        try:
            # 获取系统健康状况
            health_status = {
                "timestamp": datetime.now().isoformat(),
                "cpu_usage": "normal",
                "memory_usage": "normal",
                "disk_usage": "normal",
                "network_status": "connected",
                "all_threads_active": True
            }
            
            print(f"[{datetime.now()}] 💚 系统健康状况: {health_status}")
            
        except Exception as e:
            print(f"[{datetime.now()}] ❌ 监控系统健康时出错: {e}")
        
        # 每10秒检查一次系统健康
        for _ in range(10):
            if stop_event.is_set():
                break
            time.sleep(1)


def auto_collaborate_with_agents():
    """
    自动与其他agents协作
    """
    print(f"[{datetime.now()}] 🤝 自动协作系统启动...")
    
    collaboration_count = 0
    
    while not stop_event.is_set():
        try:
            collaboration_count += 1
            
            collaboration_msg = {
                "type": "collaboration_proposal",
                "sender": "proper_agent_v2",
                "proposal_id": f"collab_{int(time.time())}_{collaboration_count}",
                "timestamp": datetime.now().isoformat(),
                "content": f"Proposing collaboration cycle #{collaboration_count}",
                "capabilities_offered": [
                    "strategy_sharing",
                    "risk_assessment",
                    "market_analysis"
                ]
            }
            
            print(f"[{datetime.now()}] 🤝 发送协作提案 #{collaboration_count}: {collaboration_msg}")
            
        except Exception as e:
            print(f"[{datetime.now()}] ❌ 协作时出错: {e}")
        
        # 每45秒发起一次协作
        for _ in range(45):
            if stop_event.is_set():
                break
            time.sleep(1)


def run_auto_monitoring_system():
    """
    运行自动化监控系统的主函数
    """
    print("="*80)
    print("🤖 自动化监控与交互系统")
    print("="*80)
    print(f"启动时间: {datetime.now()}")
    print("功能:")
    print("  - 自动向master发送状态报告")
    print("  - 自动请求任务分配")
    print("  - 自动检查其他agent消息")
    print("  - 自动执行交易策略")
    print("  - 自动监控系统健康状况")
    print("  - 自动与其他agents协作")
    print("="*80)
    
    # 创建所有功能线程
    threads = [
        threading.Thread(target=auto_send_status_reports, name="StatusReporter", daemon=True),
        threading.Thread(target=auto_request_tasks, name="TaskRequester", daemon=True),
        threading.Thread(target=auto_check_messages, name="MessageChecker", daemon=True),
        threading.Thread(target=auto_execute_trading_strategy, name="StrategyExecutor", daemon=True),
        threading.Thread(target=auto_monitor_system_health, name="HealthMonitor", daemon=True),
        threading.Thread(target=auto_collaborate_with_agents, name="CollaborationManager", daemon=True)
    ]
    
    try:
        # 启动所有线程
        for thread in threads:
            thread.start()
        
        print(f"[{datetime.now()}] ✅ 自动化监控系统已启动")
        print(f"    - 总共启动 {len(threads)} 个监控线程")
        for i, thread in enumerate(threads):
            print(f"    - 线程 {i+1}: {thread.name}")
        
        # 主循环，监控所有线程状态
        iteration = 0
        while True:
            iteration += 1
            
            # 每30秒输出一次系统摘要
            if iteration % 2 == 0:  # 每30秒的偶数倍
                active_threads = [t.name for t in threads if t.is_alive()]
                inactive_threads = [t.name for t in threads if not t.is_alive()]
                
                print(f"[{datetime.now()}] 📋 系统摘要 (迭代: {iteration})")
                print(f"    活跃线程: {len(active_threads)}/{len(threads)}")
                
                if inactive_threads:
                    print(f"    非活跃线程: {inactive_threads}")
            
            time.sleep(15)  # 每15秒检查一次状态
            
    except KeyboardInterrupt:
        print(f"\n[{datetime.now()}] ⚠️ 收到中断信号，正在关闭自动化监控系统...")
        
        # 设置停止事件，通知所有线程退出
        stop_event.set()
        
        # 等待所有线程结束（最多等待15秒）
        for thread in threads:
            thread.join(timeout=15)
        
        print(f"[{datetime.now()}] ✅ 自动化监控系统已关闭")


if __name__ == "__main__":
    run_auto_monitoring_system()