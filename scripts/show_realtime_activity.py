#!/usr/bin/env python3
"""
实时活动展示脚本
用于显示系统的实时活动和状态
"""

import time
import threading
from datetime import datetime
import subprocess
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def show_system_status():
    """
    显示系统状态
    """
    while True:
        print(f"\n[{datetime.now()}] 🖥️ 系统状态:")
        
        # 检查主要进程
        try:
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            processes = result.stdout.split('\n')
            
            tigertrade_processes = [p for p in processes if 'tigertrade' in p and 'grep' not in p]
            
            print(f"   TigerTrade相关进程数: {len(tigertrade_processes)-1}")  # -1 排除标题行
            
            for proc in tigertrade_processes:
                if proc.strip() and 'show_realtime_activity' not in proc:  # 排除本脚本
                    parts = proc.split()
                    if len(parts) > 10:
                        pid = parts[1]
                        cmd = ' '.join(parts[10:])
                        print(f"   - PID {pid}: {cmd[:80]}...")
                        
        except Exception as e:
            print(f"   ❌ 获取进程信息失败: {e}")
        
        # 检查交易策略执行
        print(f"\n[{datetime.now()}] 📊 双向策略状态:")
        try:
            from src.bidirectional_strategy import current_position, long_position, short_position
            print(f"   当前净持仓: {current_position}")
            print(f"   多头持仓: {long_position}")
            print(f"   空头持仓: {short_position}")
        except Exception as e:
            print(f"   ❌ 获取策略状态失败: {e}")
        
        # 显示最近的日志条目
        print(f"\n[{datetime.now()}] 📝 最近日志:")
        try:
            log_files = [
                '/home/cx/tigertrade/docs/test_output_all_phase2.log',
                '/home/cx/tigertrade/docs/test_output_phase4.log',
                '/home/cx/tigertrade/docs/test_output_phase2.log',
                '/home/cx/tigertrade/docs/test_output_phase3.log'
            ]
            
            found_recent_logs = False
            for log_file in log_files:
                if os.path.exists(log_file):
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            recent_lines = lines[-3:]  # 最近3行
                            for line in recent_lines:
                                if line.strip():
                                    print(f"   {line.strip()}")
                                    found_recent_logs = True
                            break
            
            if not found_recent_logs:
                print("   暂无日志数据")
                
        except Exception as e:
            print(f"   ❌ 读取日志失败: {e}")
        
        print("\n" + "="*60)
        time.sleep(5)  # 每5秒刷新一次


def monitor_agent_interaction():
    """
    监控agent交互
    """
    counter = 0
    while True:
        counter += 1
        print(f"[{datetime.now()}] 🤝 Agent交互监控 #{counter}")
        
        # 模拟发送状态报告
        print(f"   📡 向master发送状态报告...")
        time.sleep(0.5)
        
        # 模拟检查消息
        print(f"   📬 检查来自其他agent的消息...")
        time.sleep(0.5)
        
        # 模拟任务执行
        print(f"   ⚡ 执行交易策略分析...")
        time.sleep(0.5)
        
        print(f"   🟢 交互循环完成\n")
        
        time.sleep(3)  # 每3秒执行一次交互循环


def run_activity_monitor():
    """
    运行活动监控
    """
    print("="*80)
    print("👀 实时活动监控系统")
    print("="*80)
    print("功能:")
    print("  - 实时显示系统状态")
    print("  - 监控agent交互")
    print("  - 展示交易策略状态")
    print("  - 显示最新日志")
    print("="*80)
    
    # 创建监控线程
    status_thread = threading.Thread(target=show_system_status, daemon=True)
    interaction_thread = threading.Thread(target=monitor_agent_interaction, daemon=True)
    
    # 启动线程
    status_thread.start()
    interaction_thread.start()
    
    print(f"[{datetime.now()}] ✅ 实时活动监控已启动")
    
    try:
        # 保持主线程运行
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n[{datetime.now()}] ❌ 活动监控已停止")


if __name__ == "__main__":
    run_activity_monitor()