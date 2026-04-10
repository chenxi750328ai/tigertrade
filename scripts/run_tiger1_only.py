#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
独立运行tiger1策略的脚本
此脚本只负责运行tiger1策略，不会进行数据分析
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

def run_tiger1_continuous():
    """连续运行tiger1策略"""
    print("🚀 启动tiger1策略独立运行程序...")
    
    while True:
        try:
            print(f"\n🕒 [{datetime.now().strftime('%H:%M:%S')}] 开始运行tiger1策略...")
            result = subprocess.run(
                [sys.executable, str(_REPO_ROOT / "src" / "tiger1.py"), "d"],
                cwd=str(_REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=60
            )
            
            print(f"✅ 策略执行完成，返回码: {result.returncode}")
            
            if result.returncode != 0:
                print(f"⚠️  策略执行有错误，stderr: {result.stderr[-500:]}")  # 只显示最后500个字符
            
            # 等待一定时间后继续
            print("⏸️  等待下次运行...")
            time.sleep(60)  # 每分钟运行一次
            
        except KeyboardInterrupt:
            print("\n🛑 用户中断，退出程序")
            break
        except subprocess.TimeoutExpired:
            print("⚠️ 策略执行超时")
            time.sleep(10)
        except Exception as e:
            print(f"❌ 运行过程中出现异常: {e}")
            time.sleep(30)  # 出错后等待30秒再继续

if __name__ == "__main__":
    run_tiger1_continuous()