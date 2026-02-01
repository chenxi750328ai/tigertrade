#!/usr/bin/env python3
"""
双向交易策略主执行脚本
支持做多和做空的双向交易，使用多种技术指标和风险控制
"""

import sys
import os
import time
import threading
import argparse
from datetime import datetime
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.bidirectional_strategy import bidirectional_grid_strategy, current_position, long_position, short_position
from src.api_adapter import api_manager
from src.data_collector import RealTimeDataCollector


def run_strategy():
    """
    运行双向策略
    """
    print("="*80)
    print("🚀 TigerTrade - 双向交易策略")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"合约: SIL2603")
    print(f"策略: 双向网格")
    print("="*80)
    
    iteration = 0
    try:
        while True:
            iteration += 1
            print(f"\n{'─'*80}")
            print(f"第 {iteration} 轮 | {datetime.now().strftime('%H:%M:%S')}")
            print(f"{'─'*80}")
            
            # 执行双向策略
            bidirectional_grid_strategy()
            
            # 显示当前状态
            print(f"\n{'─'*40}")
            print(f"💼 当前状态")
            print(f"{'─'*40}")
            print(f"净持仓: {current_position}")
            print(f"多头持仓: {long_position}")
            print(f"空头持仓: {short_position}")
            print(f"{'─'*40}")
            
            # 等待5秒后执行下一轮
            print(f"\n⏳ 等待 5 秒...")
            time.sleep(5)
            
    except KeyboardInterrupt:
        print(f"\n\n⏹️  用户中断程序")
    except Exception as e:
        print(f"\n\n❌ 程序异常: {e}")
        import traceback
        traceback.print_exc()


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='TigerTrade 双向交易策略')
    parser.add_argument('--mode', choices=['demo', 'live'], default='demo',
                        help='运行模式: demo(模拟) 或 live(实盘)')
    parser.add_argument('--interval', type=int, default=5,
                        help='执行间隔（秒）')
    
    args = parser.parse_args()
    
    # 设置运行模式
    if args.mode == 'demo':
        print("🧪 运行在模拟模式下")
        api_manager.initialize_mock_apis()
    else:
        print("💰 运行在实盘模式下")
        # 注意：实盘模式需要正确的API密钥配置
        # api_manager.initialize_real_apis(quote_client, trade_client)
    
    # 启动策略
    run_strategy()


if __name__ == '__main__':
    main()