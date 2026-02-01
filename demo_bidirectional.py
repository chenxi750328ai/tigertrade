#!/usr/bin/env python3
"""
双向交易策略演示脚本
"""

import sys
import os
import time
import threading
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the strategy function directly
from src.bidirectional_strategy import bidirectional_grid_strategy


def main():
    """
    主函数 - 演示双向策略
    """
    print("="*80)
    print("🚀 TigerTrade - 双向交易策略演示")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"合约: SIL2603")
    print(f"策略: 双向网格策略（支持做多和做空）")
    print("="*80)
    
    print("\n策略特点:")
    print("• 支持双向交易（做多和做空）")
    print("• 使用多种技术指标（RSI、BOLL、ATR、MACD）")
    print("• 风险控制机制（止损、止盈、仓位限制）")
    print("• 市场趋势判断")
    print("• 自动平仓逻辑")
    
    print(f"\n{'─'*80}")
    print("执行一次策略分析...")
    print(f"{'─'*80}")
    
    # 执行一次策略
    bidirectional_grid_strategy()
    
    print(f"\n{'─'*40}")
    print(f"📊 执行后状态")
    print(f"{'─'*40}")
    print("状态信息已在策略执行中显示")
    print(f"{'─'*40}")
    
    print("\n💡 策略说明:")
    print("1. 做多条件：价格接近布林下轨且RSI超卖（≤30）")
    print("2. 做空条件：价格接近布林上轨且RSI超买（≥70）")
    print("3. 止损：基于ATR计算，控制单笔风险")
    print("4. 止盈：基于ATR计算，锁定利润")
    print("5. 风控：限制最大持仓、日亏损上限")
    
    print("\n🎯 优化方向:")
    print("- 参数优化（ATR倍数、RSI阈值等）")
    print("- 多时间周期确认")
    print("- 机器学习模型增强信号判断")
    print("- 更复杂的风险管理规则")
    print("- 回测框架集成")
    
    print("\n✅ 演示完成")


if __name__ == '__main__':
    main()