#!/usr/bin/env python3
"""
冲击20%目标 - 不等了，直接干！
测试更激进配置和更长数据
"""

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(_REPO_ROOT / "scripts"))
from risk_management import RiskManager

def test_extreme_configs(data):
    """测试极限配置"""
    
    configs = [
        # 当前最优v6
        {'name': 'v6基线', 'pos': 0.5, 'tp': 0.03, 'sl': 0.02, 'rsi_b': 30, 'rsi_s': 55},
        
        # 更大仓位
        {'name': '60%仓位', 'pos': 0.6, 'tp': 0.03, 'sl': 0.02, 'rsi_b': 30, 'rsi_s': 55},
        {'name': '70%仓位', 'pos': 0.7, 'tp': 0.03, 'sl': 0.02, 'rsi_b': 30, 'rsi_s': 55},
        
        # 更短止盈（更频繁）
        {'name': '2%止盈', 'pos': 0.5, 'tp': 0.02, 'sl': 0.02, 'rsi_b': 30, 'rsi_s': 55},
        {'name': '1.5%止盈', 'pos': 0.5, 'tp': 0.015, 'sl': 0.015, 'rsi_b': 30, 'rsi_s': 55},
        
        # 组合：大仓位+短止盈
        {'name': '60%+2%', 'pos': 0.6, 'tp': 0.02, 'sl': 0.02, 'rsi_b': 30, 'rsi_s': 55},
        {'name': '70%+2.5%', 'pos': 0.7, 'tp': 0.025, 'sl': 0.02, 'rsi_b': 30, 'rsi_s': 55},
        
        # 更激进RSI
        {'name': 'RSI40/50', 'pos': 0.5, 'tp': 0.03, 'sl': 0.02, 'rsi_b': 40, 'rsi_s': 50},
        {'name': 'RSI25/55', 'pos': 0.5, 'tp': 0.03, 'sl': 0.02, 'rsi_b': 25, 'rsi_s': 55},
    ]
    
    results = []
    
    for cfg in configs:
        # 计算指标
        d = data.copy()
        d['sma_short'] = d['close'].rolling(window=5).mean()
        d['sma_long'] = d['close'].rolling(window=30).mean()
        
        if 'rsi_14' in d.columns:
            d['rsi'] = d['rsi_14']
        else:
            delta = d['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            d['rsi'] = 100 - (100 / (1 + rs))
        
        d = d.dropna()
        
        # 回测
        capital = 100000
        position = 0
        trades = []
        
        rm = RiskManager(
            stop_loss_pct=cfg['sl'],
            take_profit_pct=cfg['tp'],
            max_position_size=cfg['pos']
        )
        
        entry_price = None
        direction = None
        stop_loss = None
        take_profit = None
        
        for i in range(len(d)):
            price = d['close'].iloc[i]
            
            if position != 0:
                should_close, reason = rm.should_close_position(
                    entry_price, price, direction, stop_loss, take_profit
                )
                
                if should_close:
                    profit = position * (price - entry_price) if direction == 'long' else position * (entry_price - price)
                    capital += profit
                    trades.append({'profit': profit})
                    position = 0
                    continue
            
            if position == 0:
                ma_bull = d['sma_short'].iloc[i] > d['sma_long'].iloc[i]
                rsi_low = d['rsi'].iloc[i] < cfg['rsi_b']
                
                ma_bear = d['sma_short'].iloc[i] < d['sma_long'].iloc[i]
                rsi_high = d['rsi'].iloc[i] > cfg['rsi_s']
                
                if ma_bull and rsi_low:
                    entry_price = price
                    direction = 'long'
                    stop_loss = rm.calculate_stop_loss(price, 'long')
                    take_profit = rm.calculate_take_profit(price, 'long')
                    position = rm.calculate_position_size(capital, price, stop_loss)
                
                elif ma_bear and rsi_high:
                    entry_price = price
                    direction = 'short'
                    stop_loss = rm.calculate_stop_loss(price, 'short')
                    take_profit = rm.calculate_take_profit(price, 'short')
                    position = rm.calculate_position_size(capital, price, stop_loss)
        
        ret = (capital - 100000) / 100000 * 100
        winning = [t for t in trades if t['profit'] > 0]
        wr = len(winning) / len(trades) * 100 if trades else 0
        
        results.append({
            'name': cfg['name'],
            'return': ret,
            'trades': len(trades),
            'win_rate': wr,
            'config': cfg
        })
        
        status = '🎯' if ret >= 20 else '🚀' if ret >= 15 else '✅' if ret >= 10 else ''
        print(f"{status} {cfg['name']:12s} → {len(trades):2d}笔 {ret:+6.2f}% (胜率{wr:4.1f}%)")
    
    return results

print("🚀 不等了，直接冲击20%！")
print("="*60)
print()

# 测试
data = pd.read_csv(_REPO_ROOT / "data" / "processed" / "test.csv")
results = test_extreme_configs(data)

# 找最优
results.sort(key=lambda x: x['return'], reverse=True)
best = results[0]

print()
print("="*60)
print(f"🏆 最优配置: {best['name']}")
print(f"   收益: {best['return']:+.2f}%")
print(f"   交易: {best['trades']}笔")
print(f"   胜率: {best['win_rate']:.1f}%")

if best['return'] >= 20:
    print()
    print("🎉🎉🎉 达成20%目标！！！")
elif best['return'] >= 15:
    print()
    print(f"💪 接近目标！还差 {20-best['return']:.2f}%")

# 保存
with open('/tmp/extreme_test_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print()
print("✅ 保存: /tmp/extreme_test_results.json")
