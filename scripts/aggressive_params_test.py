#!/usr/bin/env python3
"""
激进参数测试 - 追求更高收益
目标: 突破10%收益
"""

import pandas as pd
import numpy as np
import json
import sys
sys.path.append('/home/cx/tigertrade/scripts')
from risk_management import RiskManager

def backtest_config(data, config):
    """回测单个配置"""
    capital = 100000
    position = 0
    trades = []
    
    # 计算指标
    data['sma_short'] = data['close'].rolling(window=config['ma_short']).mean()
    data['sma_long'] = data['close'].rolling(window=config['ma_long']).mean()
    
    if 'rsi_14' in data.columns:
        data['rsi'] = data['rsi_14']
    else:
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
    
    data = data.dropna()
    
    # 风险管理
    rm = RiskManager(
        stop_loss_pct=config['stop_loss'],
        take_profit_pct=config['take_profit'],
        max_position_size=config.get('max_position', 0.3)
    )
    
    entry_price = None
    direction = None
    stop_loss = None
    take_profit = None
    
    for i in range(len(data)):
        price = data['close'].iloc[i]
        
        # 平仓
        if position != 0:
            should_close, reason = rm.should_close_position(
                entry_price, price, direction, stop_loss, take_profit
            )
            
            if should_close:
                profit = position * (price - entry_price) if direction == 'long' else position * (entry_price - price)
                capital += profit
                trades.append({
                    'profit': profit,
                    'profit_pct': profit / (position * entry_price) * 100,
                    'reason': reason
                })
                position = 0
                continue
        
        # 开仓
        if position == 0:
            ma_bull = data['sma_short'].iloc[i] > data['sma_long'].iloc[i]
            rsi_low = data['rsi'].iloc[i] < config['rsi_buy']
            
            ma_bear = data['sma_short'].iloc[i] < data['sma_long'].iloc[i]
            rsi_high = data['rsi'].iloc[i] > config['rsi_sell']
            
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
    
    # 统计
    completed = [t for t in trades if 'profit' in t]
    winning = [t for t in completed if t['profit'] > 0]
    
    return {
        'config': config,
        'capital': capital,
        'return_pct': (capital - 100000) / 100000 * 100,
        'num_trades': len(completed),
        'win_rate': len(winning) / len(completed) * 100 if completed else 0,
        'avg_profit': np.mean([t['profit'] for t in completed]) if completed else 0
    }

def test_aggressive():
    """测试激进配置"""
    print("🚀 激进参数测试 - 追求突破10%\n")
    
    data = pd.read_csv('/home/cx/tigertrade/data/processed/test.csv')
    
    # 激进配置：更宽松的入场条件，但严格止盈
    configs = [
        # 基于最优配置的变体
        {'name': '最优v5', 'rsi_buy': 30, 'rsi_sell': 55, 'ma_short': 5, 'ma_long': 30, 
         'stop_loss': 0.02, 'take_profit': 0.03, 'max_position': 0.3},
        
        # 更激进的RSI阈值
        {'name': '激进RSI', 'rsi_buy': 35, 'rsi_sell': 50, 'ma_short': 5, 'ma_long': 30, 
         'stop_loss': 0.02, 'take_profit': 0.03, 'max_position': 0.3},
        
        # 更短的止盈（更频繁锁定）
        {'name': '超短止盈', 'rsi_buy': 30, 'rsi_sell': 55, 'ma_short': 5, 'ma_long': 30, 
         'stop_loss': 0.02, 'take_profit': 0.025, 'max_position': 0.3},
        
        {'name': '极短止盈', 'rsi_buy': 30, 'rsi_sell': 55, 'ma_short': 5, 'ma_long': 30, 
         'stop_loss': 0.02, 'take_profit': 0.02, 'max_position': 0.3},
        
        # 更大仓位
        {'name': '大仓位', 'rsi_buy': 30, 'rsi_sell': 55, 'ma_short': 5, 'ma_long': 30, 
         'stop_loss': 0.02, 'take_profit': 0.03, 'max_position': 0.4},
        
        {'name': '超大仓位', 'rsi_buy': 30, 'rsi_sell': 55, 'ma_short': 5, 'ma_long': 30, 
         'stop_loss': 0.02, 'take_profit': 0.03, 'max_position': 0.5},
        
        # 组合：激进RSI + 短止盈
        {'name': '组合1', 'rsi_buy': 35, 'rsi_sell': 50, 'ma_short': 5, 'ma_long': 30, 
         'stop_loss': 0.02, 'take_profit': 0.025, 'max_position': 0.3},
        
        # 组合：大仓位 + 短止盈
        {'name': '组合2', 'rsi_buy': 30, 'rsi_sell': 55, 'ma_short': 5, 'ma_long': 30, 
         'stop_loss': 0.02, 'take_profit': 0.025, 'max_position': 0.4},
        
        # 更短MA
        {'name': '超短MA', 'rsi_buy': 30, 'rsi_sell': 55, 'ma_short': 3, 'ma_long': 20, 
         'stop_loss': 0.02, 'take_profit': 0.03, 'max_position': 0.3},
        
        # 更长MA（稳健）
        {'name': '长MA', 'rsi_buy': 30, 'rsi_sell': 55, 'ma_short': 10, 'ma_long': 40, 
         'stop_loss': 0.02, 'take_profit': 0.03, 'max_position': 0.3},
    ]
    
    results = []
    for cfg in configs:
        print(f"测试: {cfg['name']:12s}", end=' ')
        
        result = backtest_config(data.copy(), cfg)
        results.append(result)
        
        print(f"→ {result['num_trades']:2d}笔 {result['return_pct']:+6.2f}% "
              f"胜率{result['win_rate']:4.1f}%")
    
    # 排序
    results.sort(key=lambda x: x['return_pct'], reverse=True)
    
    print(f"\n{'='*70}")
    print("🏆 Top 5 最佳配置:")
    print(f"{'='*70}")
    
    for i, r in enumerate(results[:5], 1):
        c = r['config']
        print(f"\n{i}. {c['name']} - {r['return_pct']:+.2f}%")
        print(f"   RSI: {c['rsi_buy']}/{c['rsi_sell']}, MA: {c['ma_short']}/{c['ma_long']}")
        print(f"   止损: {c['stop_loss']*100:.1f}%, 止盈: {c['take_profit']*100:.2f}%")
        print(f"   仓位: {c['max_position']*100:.0f}%")
        print(f"   交易: {r['num_trades']}笔, 胜率: {r['win_rate']:.1f}%")
    
    # 保存
    with open('/tmp/aggressive_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ 结果保存: /tmp/aggressive_test_results.json")
    
    # 如果有突破10%的，高亮显示
    best = results[0]
    if best['return_pct'] > 10:
        print(f"\n🎉 突破10%！最佳收益: {best['return_pct']:.2f}%")
    
    return results

if __name__ == '__main__':
    results = test_aggressive()
