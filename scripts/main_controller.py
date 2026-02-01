#!/usr/bin/env python3
"""
主控制器脚本
协调所有并发任务和agent交互
"""

import threading
import time
from datetime import datetime
import sys
import os
import subprocess
import signal
from queue import Queue

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 全局变量
stop_event = threading.Event()
active_processes = []


def start_auto_monitoring():
    """
    启动自动化监控系统
    """
    print(f"[{datetime.now()}] 🚀 启动自动化监控系统...")
    
    try:
        # 启动自动化监控脚本
        cmd = [sys.executable, "-u", os.path.join(os.path.dirname(__file__), "auto_monitor_agent_interaction.py")]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        active_processes.append(process)
        
        print(f"[{datetime.now()}] ✅ 自动化监控系统已启动 (PID: {process.pid})")
        
        return process
    except Exception as e:
        print(f"[{datetime.now()}] ❌ 启动自动化监控系统失败: {e}")
        return None


def start_enhanced_concurrent_system():
    """
    启动增强版并发系统
    """
    print(f"[{datetime.now()}] 🚀 启动增强版并发系统...")
    
    try:
        # 启动增强版并发系统
        cmd = [sys.executable, "-u", os.path.join(os.path.dirname(__file__), "enhanced_concurrent_system.py")]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        active_processes.append(process)
        
        print(f"[{datetime.now()}] ✅ 增强版并发系统已启动 (PID: {process.pid})")
        
        return process
    except Exception as e:
        print(f"[{datetime.now()}] ❌ 启动增强版并发系统失败: {e}")
        return None


def monitor_active_processes():
    """
    监控活跃进程
    """
    print(f"[{datetime.now()}] 👁️ 启动进程监控...")
    
    while not stop_event.is_set():
        # 检查活跃进程
        still_active = []
        for proc in active_processes:
            if proc.poll() is None:  # 进程仍在运行
                still_active.append(proc)
            else:
                print(f"[{datetime.now()}] ⚠️ 进程 {proc.pid} 已退出")
        
        # 更新活跃进程列表
        active_processes.clear()
        active_processes.extend(still_active)
        
        # 每10秒检查一次
        time.sleep(10)


def run_main_controller():
    """
    运行主控制器
    """
    print("="*80)
    print("🎛️ 主控制器 - 并发任务与agent交互协调系统")
    print("="*80)
    print(f"启动时间: {datetime.now()}")
    print("功能:")
    print("  - 启动自动化监控系统")
    print("  - 启动增强版并发系统")
    print("  - 监控所有活跃进程")
    print("  - 管理系统生命周期")
    print("="*80)
    
    try:
        # 启动自动化监控系统
        monitor_proc = start_auto_monitoring()
        
        # 启动增强版并发系统
        concurrent_proc = start_enhanced_concurrent_system()
        
        # 启动进程监控线程
        monitor_thread = threading.Thread(target=monitor_active_processes, daemon=True)
        monitor_thread.start()
        
        print(f"[{datetime.now()}] ✅ 主控制器已启动并运行")
        print(f"    - 自动监控系统: {'运行中' if monitor_proc and monitor_proc.poll() is None else '未运行'}")
        print(f"    - 增强并发系统: {'运行中' if concurrent_proc and concurrent_proc.poll() is None else '未运行'}")
        
        # 主循环
        iteration = 0
        while True:
            iteration += 1
            
            # 每30秒输出一次系统状态
            if iteration % 2 == 0:  # 每30秒的偶数倍
                active_count = len([p for p in active_processes if p.poll() is None])
                print(f"[{datetime.now()}] 📊 系统状态 (迭代: {iteration})")
                print(f"    活跃进程数: {active_count}/{len(active_processes)}")
                
                for proc in active_processes:
                    if proc.poll() is None:
                        print(f"    - 进程 {proc.pid}: 运行中")
                    else:
                        print(f"    - 进程 {proc.pid}: 已退出 (返回码: {proc.returncode})")
            
            time.sleep(15)  # 每15秒检查一次
            
    except KeyboardInterrupt:
        print(f"\n[{datetime.now()}] ⚠️ 收到中断信号，正在关闭主控制器...")
        
        # 设置停止事件
        stop_event.set()
        
        # 终止所有活跃进程
        print(f"[{datetime.now()}] 🛑 正在终止所有子进程...")
        for proc in active_processes:
            try:
                proc.terminate()  # 尝试优雅终止
                try:
                    proc.wait(timeout=5)  # 等待5秒让进程退出
                except subprocess.TimeoutExpired:
                    proc.kill()  # 如果进程未在5秒内退出，则强制杀死
                    print(f"[{datetime.now()}] ⚠️ 进程 {proc.pid} 未能优雅退出，已强制终止")
            except Exception as e:
                print(f"[{datetime.now()}] 终止进程 {proc.pid} 时出错: {e}")
        
        # 等待监控线程结束
        monitor_thread.join(timeout=5)
        
        print(f"[{datetime.now()}] ✅ 主控制器已关闭")


if __name__ == "__main__":
    run_main_controller()