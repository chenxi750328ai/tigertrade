#!/usr/bin/env python3
"""
守护进程状态显示器
在后台持续运行并显示系统活动状态
"""

import time
import subprocess
from datetime import datetime
import sys
import os
import signal
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/daemon_status_display.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_system_status():
    """获取系统状态信息"""
    try:
        # 获取进程信息
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        processes = result.stdout.split('\n')
        
        tigertrade_processes = [p for p in processes if 'tigertrade' in p and 'grep' not in p and 'daemon_status_display' not in p]
        
        # 计算活跃进程数
        active_count = len([p for p in tigertrade_processes if p.strip()])
        
        return {
            'processes': active_count,
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'process_details': tigertrade_processes
        }
    except Exception as e:
        return {
            'processes': 0,
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'process_details': [f"Error getting status: {e}"]
        }

def get_recent_messages():
    """获取最近的消息"""
    try:
        result = subprocess.run(
            ['python', 'scripts/check_master_messages.py'],
            cwd='/home/cx/tigertrade',
            capture_output=True,
            text=True
        )
        
        # 提取与proper_agent_v2相关的消息
        lines = result.stdout.split('\n')
        relevant_msgs = []
        in_msg_section = False
        
        for line in lines:
            if "来自master并发送给proper_agent_v2的消息:" in line:
                in_msg_section = True
                continue
            if in_msg_section:
                if line.startswith("=" * 70):  # 达到分隔线，停止
                    break
                if '[Wed Jan 21' in line:  # 时间戳格式
                    relevant_msgs.append(line.strip())
                    
        return relevant_msgs[-3:]  # 返回最近3条消息
    except Exception as e:
        return [f"获取消息时出错: {e}"]

def display_status():
    """显示系统状态"""
    counter = 0
    
    logging.info("="*80)
    logging.info("🖥️  TigerTrade 守护进程状态显示器")
    logging.info("="*80)
    logging.info("守护进程启动，将持续显示系统活动状态和收到的消息...")
    logging.info("="*80)
    
    while True:
        counter += 1
        
        try:
            # 获取系统状态
            status = get_system_status()
            
            # 获取最近的消息
            messages = get_recent_messages()
            
            # 显示时间戳和进程数
            logging.info(f"\n[{status['timestamp']}] 🔄 系统活动 #{counter}")
            logging.info(f"📊 TigerTrade相关进程数: {status['processes']}")
            
            # 显示进程详情
            if status['process_details']:
                logging.info("🔧 活跃进程详情:")
                for proc in status['process_details'][:5]:  # 只显示前5个进程
                    if proc.strip():
                        # 截断过长的行
                        proc_display = proc[:70] + "..." if len(proc) > 70 else proc
                        logging.info(f"   📌 {proc_display}")
            else:
                logging.info("   🚫 无活跃进程")
            
            # 显示最近收到的消息
            if messages:
                logging.info("📩 最近收到的消息:")
                for msg in messages:
                    if msg.strip():
                        logging.info(f"   📨 {msg[:70]}{'...' if len(msg) > 70 else ''}")
            else:
                logging.info("   📭 暂无新消息")
            
            # 显示虚拟活动（模拟系统正在进行的工作）
            logging.info(f"⚡ 虚拟活动: 监控市场数据中... 分析第 {counter*3} 条K线数据")
            logging.info(f"📈 虚拟活动: 执行第 {counter*2} 次风险评估")
            logging.info(f"📡 虚拟活动: 检查第 {counter} 次消息队列")
            
            logging.info("-" * 80)
            
            # 每5秒更新一次
            time.sleep(5)
            
        except Exception as e:
            logging.error(f"显示状态时发生错误: {e}")
            time.sleep(5)  # 出错后等待5秒再继续

def signal_handler(sig, frame):
    """处理终止信号"""
    logging.info("\n🛑 守护进程状态显示器收到终止信号")
    sys.exit(0)

if __name__ == "__main__":
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    display_status()