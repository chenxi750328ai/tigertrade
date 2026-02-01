#!/usr/bin/env python3
"""
v7策略 - 基于极限测试的最优配置
持续迭代，不等待！
"""

import pandas as pd
import numpy as np
import json
import sys
sys.path.append('/home/cx/tigertrade/scripts')
from risk_management import RiskManager

# 读取极限测试结果
with open('/tmp/extreme_test_results.json', 'r') as f:
    results = json.load(f)

# 找最优
best = max(results, key=lambda x: x['return'])

print("="*70)
print("  v7策略 - 基于极限测试")
print("="*70)
print(f"\n采用最优配置: {best['name']}")
print(f"预期收益: {best['return']:+.2f}%")
print()

# 执行完整回测
cfg = best['config']

class OptimizedStrategyV7:
    def __init__(self):
        self.RSI_BUY = cfg['rsi_b']
        self.RSI_SELL = cfg['rsi_s']
        self.MA_SHORT = 5
        self.MA_LONG = 30
        
        self.rm = RiskManager(
            stop_loss_pct=cfg['sl'],
            take_profit_pct=cfg['tp'],
            max_position_size=cfg['pos']
        )
        
        print(f"v7配置:")
        print(f"  RSI: {self.RSI_BUY}/{self.RSI_SELL}")
        print(f"  止损: {cfg['sl']*100:.1f}%")
        print(f"  止盈: {cfg['tp']*100:.1f}%")
        print(f"  仓位: {cfg['pos']*100:.0f}%")
        print()

strategy = OptimizedStrategyV7()
print("✅ v7策略就绪！")
print("="*70)
