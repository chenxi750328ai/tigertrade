#!/usr/bin/env python3
"""
快速优化版本 - Master的测试
放宽入场条件，快速验证可行性
"""

import pandas as pd
import numpy as np
import json

def quick_backtest():
    """快速回测 - 放宽条件"""
    
    # 读取数据
    data = pd.read_csv('/home/cx/tigertrade/data/processed/test.csv')
    
    # 计算指标
    data['sma_short'] = data['close'].rolling(window=10).mean()
    data['sma_long'] = data['close'].rolling(window=30).mean()
    
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))
    
    data = data.dropna()
    
    # 优化后的条件（放宽阈值 + OR逻辑）
    # 做多：MA金叉 OR RSI<40（从30放宽到40）
    long_signal = (data['sma_short'] > data['sma_long']) | (data['rsi'] < 40)
    
    # 做空：MA死叉 OR RSI>60（从70放宽到60）
    short_signal = (data['sma_short'] < data['sma_long']) | (data['rsi'] > 60)
    
    # 简单回测
    capital = 100000
    position = 0
    trades = []
    
    for i in range(len(data)):
        if position == 0:  # 没有仓位
            if long_signal.iloc[i]:
                position = capital / data['close'].iloc[i]
                entry_price = data['close'].iloc[i]
                trades.append({
                    'type': 'LONG',
                    'entry': entry_price,
                    'time': i
                })
            elif short_signal.iloc[i]:
                position = -capital / data['close'].iloc[i]
                entry_price = data['close'].iloc[i]
                trades.append({
                    'type': 'SHORT',
                    'entry': entry_price,
                    'time': i
                })
        elif position > 0:  # 多头仓位
            # 简单止盈/止损
            if data['close'].iloc[i] > entry_price * 1.05:  # 止盈5%
                capital = position * data['close'].iloc[i]
                trades[-1]['exit'] = data['close'].iloc[i]
                trades[-1]['profit'] = capital - 100000
                position = 0
            elif data['close'].iloc[i] < entry_price * 0.98:  # 止损2%
                capital = position * data['close'].iloc[i]
                trades[-1]['exit'] = data['close'].iloc[i]
                trades[-1]['profit'] = capital - 100000
                position = 0
        elif position < 0:  # 空头仓位
            if data['close'].iloc[i] < entry_price * 0.95:  # 止盈5%
                capital = 100000 + (-position) * (entry_price - data['close'].iloc[i])
                trades[-1]['exit'] = data['close'].iloc[i]
                trades[-1]['profit'] = capital - 100000
                position = 0
            elif data['close'].iloc[i] > entry_price * 1.02:  # 止损2%
                capital = 100000 + (-position) * (entry_price - data['close'].iloc[i])
                trades[-1]['exit'] = data['close'].iloc[i]
                trades[-1]['profit'] = capital - 100000
                position = 0
    
    # 结果
    result = {
        'initial_capital': 100000,
        'final_capital': capital,
        'total_return_pct': (capital - 100000) / 100000 * 100,
        'num_trades': len([t for t in trades if 'exit' in t]),
        'trades': trades[:10]  # 只显示前10笔
    }
    
    return result

if __name__ == '__main__':
    result = quick_backtest()
    print("\n=== 快速优化测试结果 ===")
    print(f"初始资金: {result['initial_capital']:,.0f}")
    print(f"最终资金: {result['final_capital']:,.0f}")
    print(f"收益率: {result['total_return_pct']:.2f}%")
    print(f"交易次数: {result['num_trades']}")
    
    # 保存结果
    with open('/tmp/master_quick_test_result.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n结果已保存: /tmp/master_quick_test_result.json")

