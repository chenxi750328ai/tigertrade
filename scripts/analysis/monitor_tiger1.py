#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
监控tiger1策略运行状态
"""

import os
import subprocess
from datetime import datetime

def monitor_tiger1():
    """监控tiger1运行状态"""
    print("📊 tiger1策略监控")
    print("="*60)
    
    # 1. 检查进程
    pid_file = '/tmp/tiger1_demo.pid'
    if os.path.exists(pid_file):
        with open(pid_file, 'r') as f:
            pid = f.read().strip()
        print(f"✅ 找到PID文件: {pid}")
        
        # 检查进程是否在运行
        try:
            result = subprocess.run(['ps', '-p', pid], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ 进程正在运行 (PID: {pid})")
            else:
                print(f"⚠️ 进程不存在 (PID: {pid})")
        except:
            print("⚠️ 无法检查进程状态")
    else:
        print("⚠️ 未找到PID文件")
    
    print("\n" + "-"*60)
    
    # 2. 查找日志文件
    log_dir = '/home/cx/tigertrade/logs'
    if os.path.exists(log_dir):
        log_files = [f for f in os.listdir(log_dir) if f.startswith('tiger1_demo_') and f.endswith('.log')]
        if log_files:
            log_files.sort(reverse=True)
            latest_log = os.path.join(log_dir, log_files[0])
            print(f"📄 最新日志文件: {latest_log}")
            
            # 显示最后20行
            try:
                with open(latest_log, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    print(f"\n   最后20行日志:")
                    for line in lines[-20:]:
                        print(f"   {line.rstrip()}")
            except Exception as e:
                print(f"   ⚠️ 读取日志失败: {e}")
        else:
            print("⚠️ 未找到日志文件")
    else:
        print("⚠️ 日志目录不存在")
    
    print("\n" + "-"*60)
    
    # 3. 检查数据收集
    data_dir = '/home/cx/trading_data'
    today = datetime.now().strftime('%Y-%m-%d')
    today_data_dir = os.path.join(data_dir, today)
    
    if os.path.exists(today_data_dir):
        csv_files = [f for f in os.listdir(today_data_dir) if f.endswith('.csv')]
        print(f"📂 今日数据目录: {today_data_dir}")
        print(f"   数据文件数: {len(csv_files)}")
        if csv_files:
            latest_csv = max([os.path.join(today_data_dir, f) for f in csv_files], 
                            key=os.path.getmtime)
            print(f"   最新文件: {os.path.basename(latest_csv)}")
            # 检查文件大小
            size = os.path.getsize(latest_csv)
            print(f"   文件大小: {size / 1024:.2f} KB")
    else:
        print(f"⚠️ 今日数据目录不存在: {today_data_dir}")
    
    print("\n" + "="*60)
    print("💡 提示:")
    print("   - 查看实时日志: tail -f /home/cx/tigertrade/logs/tiger1_demo_*.log")
    print("   - 停止策略: kill \$(cat /tmp/tiger1_demo.pid)")
    print("="*60)

if __name__ == "__main__":
    monitor_tiger1()
