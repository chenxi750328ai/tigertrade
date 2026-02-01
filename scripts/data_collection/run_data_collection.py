#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
持续运行数据收集脚本，用于收集实时交易数据
"""

import sys
import os
import time
from datetime import datetime
import threading
import signal
import traceback

# 添加tigertrade目录到路径
sys.path.insert(0, '/home/cx/tigertrade')

from data_collector_analyzer import enhance_strategy_with_logging

# 控制脚本运行的标志
running = True

def signal_handler(sig, frame):
    """处理中断信号"""
    global running
    print(f"\n🛑 收到中断信号 ({sig})，正在停止数据收集...")
    running = False
    print("✅ 数据收集已停止")
    sys.exit(0)

def run_data_collection():
    """运行数据收集"""
    global running
    
    print("🚀 启动数据收集系统...")
    print(f"⏰ 启动时间: {datetime.now()}")
    
    # 增强策略函数以支持数据收集
    enhance_strategy_with_logging()
    print("✅ 策略函数已增强，支持数据收集")
    
    # 导入tiger1模块
    from src import tiger1 as t1
    
    print("🏃‍♂️ 开始运行网格交易策略（数据收集模式）...")
    
    iteration = 0
    last_run_time = None
    
    while running:
        try:
            current_time = datetime.now()
            
            # 每分钟运行一次
            if last_run_time is None or (current_time - last_run_time).seconds >= 60:
                print(f"\n🔄 第 {iteration+1} 次运行 - {current_time.strftime('%H:%M:%S')}")
                
                # 运行增强版网格交易策略（会自动记录数据）
                t1.grid_trading_strategy_pro1()
                
                last_run_time = current_time
                iteration += 1
                
                print(f"✅ 第 {iteration} 次运行完成")
            
            # 每秒检查一次运行标志
            time.sleep(1)
            
        except KeyboardInterrupt:
            print("\n🛑 用户中断")
            break
        except Exception as e:
            print(f"💥 运行过程中发生错误: {e}")
            print(traceback.format_exc())
            time.sleep(5)  # 出错后等待5秒再继续
    
    print(f"\n🏁 数据收集完成，总共运行了 {iteration} 次")
    print(f"⏰ 结束时间: {datetime.now()}")


if __name__ == "__main__":
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 运行数据收集
    run_data_collection()