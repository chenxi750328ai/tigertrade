#!/usr/bin/env python3
"""继续参数搜索 - 第2批"""

import pandas as pd
import numpy as np
import json
import sys
sys.path.append('/home/cx/tigertrade/scripts')
from risk_management import RiskManager

def backtest_with_params(data, params):
    """回测"""
    capital = 100000
    position = 0
    trades = []
    
    rsi_buy = params['rsi_buy']
    rsi_sell = params['rsi_sell']
    ma_short = params['ma_short']
    ma_long = params['ma_long']
    
    # 计算指标
    data['sma_short'] = data['close'].rolling(window=ma_short).mean()
    data['sma_long'] = data['close'].rolling(window=ma_long).mean()
    
    if 'rsi_14' in data.columns:
        data['rsi'] = data['rsi_14']
    else:
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
    
    data = data.dropna()
    
    rm = RiskManager(
        stop_loss_pct=params.get('stop_loss', 0.02),
        take_profit_pct=params.get('take_profit', 0.05)
    )
    
    entry_price = None
    direction = None
    stop_loss = None
    take_profit = None
    
    for i in range(len(data)):
        price = data['close'].iloc[i]
        
        if position != 0:
            should_close, reason = rm.should_close_position(
                entry_price, price, direction, stop_loss, take_profit
            )
            
            if should_close:
                profit = position * (price - entry_price) if direction == 'long' else position * (entry_price - price)
                capital += profit
                trades.append({'profit': profit, 'reason': reason})
                position = 0
                continue
        
        if position == 0:
            ma_bull = data['sma_short'].iloc[i] > data['sma_long'].iloc[i]
            rsi_low = data['rsi'].iloc[i] < rsi_buy
            
            ma_bear = data['sma_short'].iloc[i] < data['sma_long'].iloc[i]
            rsi_high = data['rsi'].iloc[i] > rsi_sell
            
            # 使用AND逻辑（已证实更优）
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
    
    completed = [t for t in trades if 'profit' in t]
    winning = [t for t in completed if t['profit'] > 0]
    
    return {
        'params': params,
        'capital': capital,
        'return_pct': (capital - 100000) / 100000 * 100,
        'num_trades': len(completed),
        'win_rate': len(winning) / len(completed) * 100 if completed else 0
    }

def test_batch2():
    """测试第2批：优化止损止盈"""
    print("🔍 第2批：优化止损止盈参数\n")
    
    data = pd.read_csv('/home/cx/tigertrade/data/processed/test.csv')
    
    # 基于最优RSI/MA，测试不同止损止盈
    test_configs = [
        # 止损2%, 止盈变化
        {'rsi_buy': 30, 'rsi_sell': 55, 'ma_short': 5, 'ma_long': 30, 
         'stop_loss': 0.02, 'take_profit': 0.03},
        {'rsi_buy': 30, 'rsi_sell': 55, 'ma_short': 5, 'ma_long': 30, 
         'stop_loss': 0.02, 'take_profit': 0.04},
        {'rsi_buy': 30, 'rsi_sell': 55, 'ma_short': 5, 'ma_long': 30, 
         'stop_loss': 0.02, 'take_profit': 0.05},
        {'rsi_buy': 30, 'rsi_sell': 55, 'ma_short': 5, 'ma_long': 30, 
         'stop_loss': 0.02, 'take_profit': 0.06},
        {'rsi_buy': 30, 'rsi_sell': 55, 'ma_short': 5, 'ma_long': 30, 
         'stop_loss': 0.02, 'take_profit': 0.08},
        
        # 止损变化, 止盈5%
        {'rsi_buy': 30, 'rsi_sell': 55, 'ma_short': 5, 'ma_long': 30, 
         'stop_loss': 0.015, 'take_profit': 0.05},
        {'rsi_buy': 30, 'rsi_sell': 55, 'ma_short': 5, 'ma_long': 30, 
         'stop_loss': 0.025, 'take_profit': 0.05},
        {'rsi_buy': 30, 'rsi_sell': 55, 'ma_short': 5, 'ma_long': 30, 
         'stop_loss': 0.03, 'take_profit': 0.05},
        
        # 测试其他有潜力的RSI组合
        {'rsi_buy': 25, 'rsi_sell': 55, 'ma_short': 5, 'ma_long': 30, 
         'stop_loss': 0.02, 'take_profit': 0.05},
        {'rsi_buy': 35, 'rsi_sell': 55, 'ma_short': 5, 'ma_long': 30, 
         'stop_loss': 0.02, 'take_profit': 0.05},
        {'rsi_buy': 30, 'rsi_sell': 50, 'ma_short': 5, 'ma_long': 30, 
         'stop_loss': 0.02, 'take_profit': 0.05},
        {'rsi_buy': 30, 'rsi_sell': 60, 'ma_short': 5, 'ma_long': 30, 
         'stop_loss': 0.02, 'take_profit': 0.05},
    ]
    
    results = []
    for idx, config in enumerate(test_configs, 1):
        print(f"[{idx}/{len(test_configs)}] RSI({config['rsi_buy']}/{config['rsi_sell']}) "
              f"止损{config['stop_loss']*100:.1f}% 止盈{config['take_profit']*100:.1f}%", 
              end='')
        
        result = backtest_with_params(data.copy(), config)
        results.append(result)
        
        print(f" → {result['num_trades']}笔 {result['return_pct']:+.2f}% "
              f"胜率{result['win_rate']:.0f}%")
    
    # 排序
    results.sort(key=lambda x: x['return_pct'], reverse=True)
    
    print(f"\n{'='*70}")
    print("🏆 Top 5 最优配置:")
    print(f"{'='*70}")
    
    for i, r in enumerate(results[:5], 1):
        p = r['params']
        print(f"\n{i}. RSI({p['rsi_buy']}/{p['rsi_sell']}) "
              f"止损{p['stop_loss']*100:.1f}% 止盈{p['take_profit']*100:.1f}%")
        print(f"   收益: {r['return_pct']:+.2f}%")
        print(f"   交易: {r['num_trades']}笔")
        print(f"   胜率: {r['win_rate']:.1f}%")
        print(f"   风险收益比: {p['take_profit']/p['stop_loss']:.1f}:1")
    
    # 保存
    with open('/tmp/batch2_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ 第2批完成！结果: /tmp/batch2_results.json")
    
    return results[0]  # 返回最优

if __name__ == '__main__':
    best = test_batch2()
