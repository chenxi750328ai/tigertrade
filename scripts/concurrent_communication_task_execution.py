#!/usr/bin/env python3
"""
并发通信与任务执行脚本
此脚本可以同时处理与其他agent的通信和执行指定任务
"""

import asyncio
import threading
import time
from datetime import datetime
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def communicate_with_agents():
    """
    与其他agent进行通信的协程
    """
    print(f"[{datetime.now()}] 🤖 开始与其他agent进行通信...")
    
    while True:
        print(f"[{datetime.now()}] 📡 发送状态报告给master...")
        # 模拟向master发送状态
        await asyncio.sleep(5)  # 每5秒发送一次状态
        
        print(f"[{datetime.now()}] 📬 检查来自其他agent的消息...")
        # 模拟检查其他agent的消息
        await asyncio.sleep(3)  # 每3秒检查一次消息


async def execute_trading_task():
    """
    执行交易任务的协程
    """
    print(f"[{datetime.now()}] ⚡ 开始执行交易任务...")
    
    from src.bidirectional_strategy import bidirectional_grid_strategy
    
    while True:
        print(f"[{datetime.now()}] 🔄 执行一次双向策略分析...")
        # 执行一次策略
        bidirectional_grid_strategy()
        
        print(f"[{datetime.now()}] ⏳ 策略执行完成，等待下次执行...")
        await asyncio.sleep(10)  # 每10秒执行一次策略


async def main():
    """
    主函数 - 并发运行通信和任务执行
    """
    print("="*80)
    print("🔗 并发通信与任务执行系统")
    print("="*80)
    
    # 创建两个并发任务
    communication_task = asyncio.create_task(communicate_with_agents())
    execution_task = asyncio.create_task(execute_trading_task())
    
    # 等待两个任务完成（实际上它们都是无限循环）
    await asyncio.gather(communication_task, execution_task)


def run_concurrent_system():
    """
    运行并发系统的入口函数
    """
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n[{datetime.now()}] ❌ 系统被用户中断")
    except Exception as e:
        print(f"[{datetime.now()}] ❌ 系统发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_concurrent_system()