#!/usr/bin/env python3
"""
可视化仪表板
实时显示系统活动和状态
"""

import time
import threading
from datetime import datetime
import sys
import os
import subprocess
import curses
from curses import wrapper

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_system_status():
    """获取系统状态信息"""
    try:
        # 获取进程信息
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        processes = result.stdout.split('\n')
        
        tigertrade_processes = [p for p in processes if 'tigertrade' in p and 'grep' not in p and 'visual_dashboard' not in p]
        
        # 获取最近的消息
        msg_result = subprocess.run(
            ['python', 'scripts/check_master_messages.py'], 
            cwd='/home/cx/tigertrade', 
            capture_output=True, 
            text=True
        )
        
        # 计算活跃进程数
        active_count = len([p for p in tigertrade_processes if p.strip()])
        
        return {
            'processes': active_count,
            'messages': msg_result.stdout[-500:],  # 取最后500个字符
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'process_details': tigertrade_processes
        }
    except Exception as e:
        return {
            'processes': 0,
            'messages': f"Error getting status: {e}",
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'process_details': []
        }


def draw_dashboard(stdscr):
    """绘制仪表板界面"""
    curses.curs_set(0)  # 隐藏光标
    stdscr.nodelay(True)  # 非阻塞输入
    
    while True:
        # 清屏
        stdscr.clear()
        
        # 获取系统状态
        status = get_system_status()
        
        # 获取屏幕尺寸
        height, width = stdscr.getmaxyx()
        
        # 绘制标题
        title = "🚀 TigerTrade 可视化监控仪表板"
        stdscr.addstr(0, max(0, (width - len(title)) // 2), title, curses.A_BOLD)
        
        # 绘制时间
        time_str = f"🕒 时间: {status['timestamp']}"
        stdscr.addstr(1, 2, time_str)
        
        # 绘制系统状态
        status_line = f"📊 系统状态: {status['processes']} 个 TigerTrade 进程运行中"
        stdscr.addstr(2, 2, status_line, curses.A_BOLD)
        
        # 绘制活跃组件
        stdscr.addstr(4, 2, "🔄 活跃组件:", curses.A_UNDERLINE)
        
        y_pos = 5
        for i, proc in enumerate(status['process_details'][:min(10, height-10)]):  # 最多显示10个进程
            if proc.strip():
                # 截断过长的行
                proc_display = proc[:width-4]
                stdscr.addstr(y_pos + i, 4, proc_display)
        
        # 绘制说明
        stdscr.addstr(height-3, 2, "按 'q' 键退出监控", curses.A_DIM)
        
        # 刷新屏幕
        stdscr.refresh()
        
        # 检查退出键
        key = stdscr.getch()
        if key == ord('q') or key == ord('Q'):
            break
            
        # 等待1秒
        time.sleep(1)


def run_visual_dashboard():
    """运行可视化仪表板"""
    print("启动可视化仪表板...")
    print("按 'q' 键退出监控")
    wrapper(draw_dashboard)
    print("可视化仪表板已退出")


if __name__ == "__main__":
    run_visual_dashboard()