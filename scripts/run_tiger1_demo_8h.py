#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
在DEMO账户运行tiger1策略8小时
"""

import sys
import os
import time
import signal
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# 运行配置
RUN_DURATION_HOURS = 20
LOG_DIR = str(_REPO_ROOT / "logs")
LOG_FILE = os.path.join(LOG_DIR, f'tiger1_demo_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
PID_FILE = '/tmp/tiger1_demo.pid'

# 创建日志目录
os.makedirs(LOG_DIR, exist_ok=True)

def signal_handler(sig, frame):
    """处理中断信号"""
    print("\n⚠️ 收到中断信号，正在优雅退出...")
    # 读取PID并尝试终止子进程
    if os.path.exists(PID_FILE):
        with open(PID_FILE, 'r') as f:
            try:
                pid = int(f.read().strip())
                os.kill(pid, signal.SIGTERM)
                print(f"✅ 已终止子进程 (PID: {pid})")
            except:
                pass
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def main():
    """主函数"""
    print("="*60)
    print("🚀 启动tiger1策略（DEMO账户，运行20小时）")
    print("="*60)
    print(f"📅 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏰ 预计结束时间: {(datetime.now() + timedelta(hours=RUN_DURATION_HOURS)).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📝 日志文件: {LOG_FILE}")
    print("="*60)
    
    # 切换到仓库根目录
    os.chdir(str(_REPO_ROOT))
    
    # 构建运行命令（使用'd'参数表示DEMO/sandbox模式）
    cmd = [
        sys.executable,
        'src/tiger1.py',
        'd'  # 'd'表示demo/sandbox模式
    ]
    
    print(f"🔧 运行命令: {' '.join(cmd)}")
    print("="*60)
    print("💡 提示: 按 Ctrl+C 可以优雅退出")
    print("="*60)
    print()
    
    # 计算结束时间
    end_time = datetime.now() + timedelta(hours=RUN_DURATION_HOURS)
    
    try:
        # 打开日志文件
        with open(LOG_FILE, 'w', encoding='utf-8') as log_file:
            # 启动子进程
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # 保存PID
            with open(PID_FILE, 'w') as f:
                f.write(str(process.pid))
            
            print(f"✅ 策略已启动 (PID: {process.pid})")
            print(f"📝 日志实时输出到: {LOG_FILE}")
            print()
            
            # 实时输出日志
            start_time = datetime.now()
            while True:
                # 检查是否超过运行时间
                if datetime.now() >= end_time:
                    print(f"\n⏰ 已达到运行时间限制（{RUN_DURATION_HOURS}小时），正在停止...")
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    break
                
                # 检查进程是否还在运行
                if process.poll() is not None:
                    print(f"\n⚠️ 进程已退出 (返回码: {process.returncode})")
                    break
                
                # 读取输出
                output = process.stdout.readline()
                if output:
                    line = output.strip()
                    # 同时输出到控制台和日志文件
                    print(line)
                    log_file.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {line}\n")
                    log_file.flush()
                
                # 短暂休眠避免CPU占用过高
                time.sleep(0.1)
            
            # 等待进程完全结束
            if process.poll() is None:
                process.wait()
            
            # 读取剩余输出
            remaining_output, _ = process.communicate()
            if remaining_output:
                for line in remaining_output.split('\n'):
                    if line.strip():
                        print(line)
                        log_file.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {line}\n")
            
            # 计算运行时间
            elapsed = datetime.now() - start_time
            print(f"\n{'='*60}")
            print(f"✅ 运行完成")
            print(f"⏱️  总运行时间: {elapsed}")
            print(f"📝 日志文件: {LOG_FILE}")
            print(f"{'='*60}")
            
    except KeyboardInterrupt:
        print("\n⚠️ 收到中断信号，正在停止...")
        if 'process' in locals() and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理PID文件
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)

if __name__ == "__main__":
    main()
