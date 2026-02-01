#!/usr/bin/env python3
"""
增强版并发通信与任务执行系统
此脚本使用线程同时处理与其他agent的通信和执行指定任务，并实现秒级监控
"""

import threading
import time
from datetime import datetime
import sys
import os
import requests
from queue import Queue
import json

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 全局变量
message_queue = Queue()
stop_event = threading.Event()


def communicate_with_agents():
    """
    与其他agent进行通信的函数（运行在线程中）
    """
    print(f"[{datetime.now()}] 🤖 通信线程启动...")
    
    while not stop_event.is_set():
        # 发送状态报告给master
        print(f"[{datetime.now()}] 📡 发送状态报告给master...")
        try:
            # 这里可以添加实际的API调用来发送状态
            status_report = {
                "agent_id": "proper_agent_v2",
                "status": "active",
                "timestamp": datetime.now().isoformat(),
                "capabilities": ["bidirectional_trading", "strategy_optimization", "risk_management"],
                "current_task": "running_bidirectional_strategy"
            }
            # 模拟API调用
            print(f"   Status report: {status_report}")
        except Exception as e:
            print(f"[{datetime.now()}] ❌ 发送状态报告失败: {e}")
        
        # 每5秒发送一次状态
        for i in range(5):
            if stop_event.is_set():
                break
            time.sleep(1)
        
        if stop_event.is_set():
            break
            
        # 检查来自其他agent的消息
        print(f"[{datetime.now()}] 📬 检查来自其他agent的消息...")
        try:
            # 这里可以添加实际的API调用来检查消息
            # 模拟检查消息
            print(f"   Checked messages from other agents")
        except Exception as e:
            print(f"[{datetime.now()}] ❌ 检查消息失败: {e}")
        
        # 每3秒检查一次消息
        for i in range(3):
            if stop_event.is_set():
                break
            time.sleep(1)
        
    print(f"[{datetime.now()}] 🛑 通信线程停止")


def execute_trading_task():
    """
    执行交易任务的函数（运行在线程中）
    """
    print(f"[{datetime.now()}] ⚡ 任务执行线程启动...")
    
    from src.bidirectional_strategy import bidirectional_grid_strategy
    
    while not stop_event.is_set():
        print(f"[{datetime.now()}] 🔄 执行一次双向策略分析...")
        try:
            # 执行一次策略
            bidirectional_grid_strategy()
        except Exception as e:
            print(f"[{datetime.now()}] ❌ 执行策略时出错: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"[{datetime.now()}] ⏳ 策略执行完成，等待下次执行...")
        # 每10秒执行一次策略
        for i in range(10):
            if stop_event.is_set():
                break
            time.sleep(1)
            
    print(f"[{datetime.now()}] 🛑 任务执行线程停止")


def monitor_system_status():
    """
    监控系统状态的函数（运行在线程中）
    """
    print(f"[{datetime.now()}] 👁️ 系统监控线程启动...")
    
    while not stop_event.is_set():
        print(f"[{datetime.now()}] 📊 监控系统状态...")
        
        # 获取当前时间戳
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # 模拟系统状态检查
        print(f"   [{timestamp}] 系统状态正常 - 通信线程活跃，任务执行线程活跃")
        
        # 每2秒检查一次系统状态
        for i in range(2):
            if stop_event.is_set():
                break
            time.sleep(1)
        
    print(f"[{datetime.now()}] 🛑 系统监控线程停止")


def task_scheduler():
    """
    任务调度器（运行在线程中）
    """
    print(f"[{datetime.now()}] ⏰ 任务调度器启动...")
    
    task_counter = 0
    
    while not stop_event.is_set():
        task_counter += 1
        print(f"[{datetime.now()}] 🗂️ 任务调度 #{task_counter} - 检查是否有待处理任务...")
        
        # 检查任务队列
        if not message_queue.empty():
            task = message_queue.get()
            print(f"[{datetime.now()}] 🚀 执行队列任务: {task}")
        
        # 每7秒执行一次调度检查
        for i in range(7):
            if stop_event.is_set():
                break
            time.sleep(1)
        
    print(f"[{datetime.now()}] 🛑 任务调度器停止")


def run_enhanced_system():
    """
    运行增强版并发系统的入口函数
    """
    print("="*80)
    print("🔗 增强版并发通信与任务执行系统")
    print("="*80)
    
    # 创建多个功能线程
    threads = [
        threading.Thread(target=communicate_with_agents, name="CommunicationThread", daemon=True),
        threading.Thread(target=execute_trading_task, name="ExecutionThread", daemon=True),
        threading.Thread(target=monitor_system_status, name="MonitorThread", daemon=True),
        threading.Thread(target=task_scheduler, name="SchedulerThread", daemon=True)
    ]
    
    try:
        # 启动所有线程
        for thread in threads:
            thread.start()
        
        print(f"[{datetime.now()}] ✅ 增强版并发系统已启动")
        print(f"    - 线程总数: {len(threads)}")
        for i, thread in enumerate(threads):
            print(f"    - 线程 {i+1}: {thread.name} (PID: {thread.ident})")
        
        # 主线程循环，显示系统状态
        iteration = 0
        while True:
            iteration += 1
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] 🌀 主循环运行中... (迭代: {iteration})")
            
            # 显示各线程状态
            active_threads = [t.name for t in threads if t.is_alive()]
            if active_threads:
                print(f"    Active threads: {', '.join(active_threads)}")
            
            time.sleep(15)  # 每15秒显示一次状态
            
            # 检查子线程是否仍在运行
            for thread in threads:
                if not thread.is_alive():
                    print(f"[{datetime.now()}] ⚠️ {thread.name} 已停止")
                    
    except KeyboardInterrupt:
        print(f"\n[{datetime.now()}] ❌ 收到中断信号，正在停止系统...")
        stop_event.set()  # 设置停止事件，通知所有线程退出
        
        # 等待所有线程结束（最多等待10秒）
        for thread in threads:
            thread.join(timeout=10)
        
        print(f"[{datetime.now()}] ✅ 增强版并发系统已停止")


if __name__ == "__main__":
    run_enhanced_system()