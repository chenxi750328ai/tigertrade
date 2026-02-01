#!/usr/bin/env python3
"""
线程化并发通信与任务执行系统
此脚本使用线程同时处理与其他agent的通信和执行指定任务
"""

import threading
import time
from datetime import datetime
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def communicate_with_agents(stop_event):
    """
    与其他agent进行通信的函数（运行在线程中）
    """
    print(f"[{datetime.now()}] 🤖 通信线程启动...")
    
    while not stop_event.is_set():
        print(f"[{datetime.now()}] 📡 发送状态报告给master...")
        # 模拟向master发送状态
        time.sleep(5)  # 每5秒发送一次状态
        
        if stop_event.is_set():
            break
            
        print(f"[{datetime.now()}] 📬 检查来自其他agent的消息...")
        # 模拟检查其他agent的消息
        time.sleep(3)  # 每3秒检查一次消息
        
    print(f"[{datetime.now()}] 🛑 通信线程停止")


def execute_trading_task(stop_event):
    """
    执行交易任务的函数（运行在线程中）
    """
    print(f"[{datetime.now()}] ⚡ 任务执行线程启动...")
    
    from src.bidirectional_strategy import bidirectional_grid_strategy
    
    while not stop_event.is_set():
        print(f"[{datetime.now()}] 🔄 执行一次双向策略分析...")
        # 执行一次策略
        bidirectional_grid_strategy()
        
        print(f"[{datetime.now()}] ⏳ 策略执行完成，等待下次执行...")
        time.sleep(10)  # 每10秒执行一次策略
        
        if stop_event.is_set():
            break
            
    print(f"[{datetime.now()}] 🛑 任务执行线程停止")


def run_concurrent_system():
    """
    运行并发系统的入口函数
    """
    print("="*80)
    print("🔗 线程化并发通信与任务执行系统")
    print("="*80)
    
    # 创建一个事件来控制线程的停止
    stop_event = threading.Event()
    
    # 创建通信线程
    communication_thread = threading.Thread(
        target=communicate_with_agents, 
        args=(stop_event,),
        name="CommunicationThread",
        daemon=True
    )
    
    # 创建任务执行线程
    execution_thread = threading.Thread(
        target=execute_trading_task,
        args=(stop_event,),
        name="ExecutionThread",
        daemon=True
    )
    
    try:
        # 启动线程
        communication_thread.start()
        execution_thread.start()
        
        print(f"[{datetime.now()}] ✅ 并发系统已启动")
        print(f"    - 通信线程 PID: {communication_thread.ident}")
        print(f"    - 任务执行线程 PID: {execution_thread.ident}")
        
        # 主线程循环，显示系统状态
        iteration = 0
        while True:
            iteration += 1
            print(f"[{datetime.now()}] 🌀 主循环运行中... (迭代: {iteration})")
            time.sleep(15)  # 每15秒显示一次状态
            
            # 检查子线程是否仍在运行
            if not communication_thread.is_alive():
                print(f"[{datetime.now()}] ⚠️ 通信线程已停止")
                
            if not execution_thread.is_alive():
                print(f"[{datetime.now()}] ⚠️ 任务执行线程已停止")
                
    except KeyboardInterrupt:
        print(f"\n[{datetime.now()}] ❌ 收到中断信号，正在停止系统...")
        stop_event.set()  # 设置停止事件，通知线程退出
        
        # 等待线程结束（最多等待10秒）
        communication_thread.join(timeout=10)
        execution_thread.join(timeout=10)
        
        print(f"[{datetime.now()}] ✅ 系统已停止")


if __name__ == "__main__":
    run_concurrent_system()