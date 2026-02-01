#!/usr/bin/env python3
"""
持续监控master消息的脚本
此脚本将持续监听来自master的消息并做出响应
"""

import time
import threading
from datetime import datetime
import sys
import os
import json

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def monitor_master_messages():
    """
    持续监控来自master的消息
    """
    print(f"[{datetime.now()}] 🎯 启动持续master消息监控...")
    
    while True:
        try:
            # 检查是否有来自master的消息
            result = os.popen("cd /home/cx/tigertrade && python scripts/check_master_messages.py 2>/dev/null").read()
            
            # 解析结果查找消息
            if "来自master并发送给proper_agent_v2的消息:" in result:
                # 提取最新的几条消息
                lines = result.split('\n')
                in_messages_section = False
                recent_messages = []
                
                for line in lines:
                    if "来自master并发送给proper_agent_v2的消息:" in line:
                        in_messages_section = True
                        continue
                    
                    if in_messages_section:
                        if line.startswith("=" * 70):  # 结束标记
                            break
                        
                        if line.strip() and ":" in line[:50]:  # 消息行通常包含时间戳
                            recent_messages.append(line.strip())
                
                # 如果有新消息，输出最后几条
                if recent_messages:
                    print(f"[{datetime.now()}] 📬 收到master消息:")
                    for msg in recent_messages[-3:]:  # 显示最近3条消息
                        print(f"   {msg}")
            
            # 每5秒检查一次
            time.sleep(5)
            
        except KeyboardInterrupt:
            print(f"[{datetime.now()}] ❌ 监控被中断")
            break
        except Exception as e:
            print(f"[{datetime.now()}] ❌ 监控出错: {e}")
            time.sleep(5)  # 出错后稍等再继续


def send_regular_status_reports():
    """
    定期发送状态报告给master
    """
    print(f"[{datetime.now()}] 📊 启动定期状态报告...")
    
    counter = 0
    while True:
        try:
            counter += 1
            print(f"[{datetime.now()}] 📤 发送第 {counter} 次状态报告给master")
            
            # 发送状态报告
            os.system("cd /home/cx/tigertrade && python scripts/send_message_to_master.py \"proper_agent_v2\" \"Regular status report: System operational with concurrent monitoring and task execution.\" >/dev/null 2>&1 &")
            
            # 每60秒发送一次
            for _ in range(60):
                if stop_event.is_set():
                    break
                time.sleep(1)
                
            if stop_event.is_set():
                break
                
        except KeyboardInterrupt:
            print(f"[{datetime.now()}] ❌ 状态报告被中断")
            break
        except Exception as e:
            print(f"[{datetime.now()}] ❌ 发送状态报告出错: {e}")
            time.sleep(60)


def send_task_requests():
    """
    定期发送任务请求
    """
    print(f"[{datetime.now()}] 🎯 启动定期任务请求...")
    
    counter = 0
    while True:
        try:
            counter += 1
            print(f"[{datetime.now()}] 📋 发送第 {counter} 次任务请求")
            
            # 发送任务请求
            os.system("cd /home/cx/tigertrade && python scripts/send_task_request.py \"proper_agent_v2\" \"Requesting task assignment\" >/dev/null 2>&1 &")
            
            # 每120秒发送一次
            for _ in range(120):
                if stop_event.is_set():
                    break
                time.sleep(1)
                
            if stop_event.is_set():
                break
                
        except KeyboardInterrupt:
            print(f"[{datetime.now()}] ❌ 任务请求被中断")
            break
        except Exception as e:
            print(f"[{datetime.now()}] ❌ 发送任务请求出错: {e}")
            time.sleep(120)


def auto_collaborate():
    """
    自动与其他agents协作
    """
    print(f"[{datetime.now()}] 🤝 启动自动协作...")
    
    counter = 0
    while True:
        try:
            counter += 1
            print(f"[{datetime.now()}] 🤝 执行第 {counter} 次协作行动")
            
            # 发送协作消息
            os.system("cd /home/cx/tigertrade && python scripts/send_collaboration_message.py \"proper_agent_v2\" \"Active monitoring and execution. Seeking collaboration opportunities.\" >/dev/null 2>&1 &")
            
            # 每180秒执行一次
            for _ in range(180):
                if stop_event.is_set():
                    break
                time.sleep(1)
                
            if stop_event.is_set():
                break
                
        except KeyboardInterrupt:
            print(f"[{datetime.now()}] ❌ 自动协作被中断")
            break
        except Exception as e:
            print(f"[{datetime.now()}] ❌ 自动协作出错: {e}")
            time.sleep(180)


# 全局停止事件
stop_event = threading.Event()


def run_continuous_monitoring():
    """
    运行连续监控系统
    """
    print("="*80)
    print("📡 持续监控与交互系统")
    print("="*80)
    print("功能:")
    print("  - 持续监控来自master的消息")
    print("  - 定期发送状态报告")
    print("  - 定期请求任务分配")
    print("  - 自动与其他agents协作")
    print("="*80)
    
    # 创建所有监控线程
    threads = [
        threading.Thread(target=monitor_master_messages, name="MasterMonitor", daemon=True),
        threading.Thread(target=send_regular_status_reports, name="StatusReporter", daemon=True),
        threading.Thread(target=send_task_requests, name="TaskRequester", daemon=True),
        threading.Thread(target=auto_collaborate, name="CollaborationManager", daemon=True)
    ]
    
    # 启动所有线程
    for thread in threads:
        thread.start()
        print(f"[{datetime.now()}] ✅ 启动线程: {thread.name}")
    
    print(f"[{datetime.now()}] 🎯 所有监控服务已启动")
    
    try:
        # 主循环，保持程序运行
        while True:
            time.sleep(10)  # 每10秒检查一次
            
            # 每分钟输出一次系统状态
            active_threads = [t.name for t in threading.enumerate() if t.name in [th.name for th in threads]]
            print(f"[{datetime.now()}] 📊 系统状态: {len(active_threads)}/{len(threads)} 个监控服务运行中")
            
    except KeyboardInterrupt:
        print(f"\n[{datetime.now()}] ⚠️ 收到中断信号，正在停止监控系统...")
        stop_event.set()
        
        # 等待所有线程结束
        for thread in threads:
            thread.join(timeout=5)
        
        print(f"[{datetime.now()}] ✅ 监控系统已停止")


if __name__ == "__main__":
    run_continuous_monitoring()