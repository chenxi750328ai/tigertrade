#!/usr/bin/env python3
"""
参数网格搜索 - 系统测试不同参数组合
目标：找到最优参数配置
"""

import pandas as pd
import numpy as np
import json
from itertools import product
import sys
sys.path.append('/home/cx/tigertrade/scripts')
from risk_management import RiskManager

def backtest_with_params(data, params):
    """使用指定参数回测"""
    capital = 100000
    position = 0
    trades = []
    
    # 解包参数
    rsi_buy = params['rsi_buy']
    rsi_sell = params['rsi_sell']
    ma_short = params['ma_short']
    ma_long = params['ma_long']
    use_or = params['use_or']
    
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
    
    # 风险管理
    rm = RiskManager()
    entry_price = None
    direction = None
    stop_loss = None
    take_profit = None
    
    for i in range(len(data)):
        price = data['close'].iloc[i]
        
        # 平仓检查
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
        
        # 入场检查
        if position == 0:
            ma_bull = data['sma_short'].iloc[i] > data['sma_long'].iloc[i]
            ma_bear = data['sma_short'].iloc[i] < data['sma_long'].iloc[i]
            rsi_low = data['rsi'].iloc[i] < rsi_buy
            rsi_high = data['rsi'].iloc[i] > rsi_sell
            
            if use_or:
                long_signal = ma_bull or rsi_low
                short_signal = ma_bear or rsi_high
            else:
                long_signal = ma_bull and rsi_low
                short_signal = ma_bear and rsi_high
            
            if long_signal and not short_signal:
                entry_price = price
                direction = 'long'
                stop_loss = rm.calculate_stop_loss(price, 'long')
                take_profit = rm.calculate_take_profit(price, 'long')
                position = rm.calculate_position_size(capital, price, stop_loss)
                trades.append({'type': 'LONG', 'entry': price})
            
            elif short_signal and not long_signal:
                entry_price = price
                direction = 'short'
                stop_loss = rm.calculate_stop_loss(price, 'short')
                take_profit = rm.calculate_take_profit(price, 'short')
                position = rm.calculate_position_size(capital, price, stop_loss)
                trades.append({'type': 'SHORT', 'entry': price})
    
    # 统计
    completed = [t for t in trades if 'profit' in t]
    winning = [t for t in completed if t['profit'] > 0]
    # 单笔收益占初始资金比例(%)，用于报告：单笔平均、单笔TOP
    profit_pcts = [(t['profit'] / 100000.0) * 100.0 for t in completed] if completed else []
    avg_per_trade_pct = sum(profit_pcts) / len(profit_pcts) if profit_pcts else 0.0
    top_per_trade_pct = max(profit_pcts) if profit_pcts else 0.0

    return {
        'params': params,
        'capital': capital,
        'return_pct': (capital - 100000) / 100000 * 100,
        'num_trades': len(completed),
        'win_rate': len(winning) / len(completed) * 100 if completed else 0,
        'avg_per_trade_pct': round(avg_per_trade_pct, 2),
        'top_per_trade_pct': round(top_per_trade_pct, 2),
        'trades': completed
    }

def grid_search():
    """网格搜索"""
    print("🔍 参数网格搜索开始...\n")
    
    # 读取数据
    data = pd.read_csv('/home/cx/tigertrade/data/processed/test.csv')
    
    # 定义参数网格
    param_grid = {
        'rsi_buy': [30, 35, 40, 45],
        'rsi_sell': [55, 60, 65, 70],
        'ma_short': [5, 10, 15],
        'ma_long': [20, 30, 40],
        'use_or': [True, False]
    }
    
    # 生成所有组合
    keys = param_grid.keys()
    combinations = list(product(*param_grid.values()))
    
    print(f"总共 {len(combinations)} 种参数组合\n")
    
    results = []
    for idx, values in enumerate(combinations[:20], 1):  # 先测试20组
        params = dict(zip(keys, values))
        
        print(f"[{idx}/20] 测试: RSI({params['rsi_buy']}/{params['rsi_sell']}) "
              f"MA({params['ma_short']}/{params['ma_long']}) "
              f"{'OR' if params['use_or'] else 'AND'}", end='')
        
        result = backtest_with_params(data.copy(), params)
        results.append(result)
        
        print(f" → {result['num_trades']}笔 {result['return_pct']:.2f}% "
              f"胜率{result['win_rate']:.1f}%")
    
    # 排序找最优
    results.sort(key=lambda x: x['return_pct'], reverse=True)
    
    print(f"\n{'='*60}")
    print("Top 5 最优参数组合:")
    print(f"{'='*60}")
    
    for i, r in enumerate(results[:5], 1):
        p = r['params']
        print(f"\n{i}. RSI({p['rsi_buy']}/{p['rsi_sell']}) "
              f"MA({p['ma_short']}/{p['ma_long']}) "
              f"{'OR' if p['use_or'] else 'AND'}")
        print(f"   收益: {r['return_pct']:.2f}%")
        print(f"   交易: {r['num_trades']}笔")
        print(f"   胜率: {r['win_rate']:.1f}%")
    
    # 保存结果
    with open('/tmp/grid_search_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ 结果已保存: /tmp/grid_search_results.json")
    return results


def grid_search_optimal_params(strategy_name):
    """
    供 optimize_algorithm_and_profitability 调用：对 grid/boll 做网格搜索，返回最优参数及回测效果。
    返回 dict：若成功含 params, return_pct, win_rate, num_trades；无数据或失败返回 {}。
    """
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(root, 'data', 'processed', 'test.csv')
    if not os.path.isfile(data_path):
        return {}
    try:
        data = pd.read_csv(data_path)
    except Exception:
        return {}
    if data is None or len(data) < 100:
        return {}
    param_grid = {
        'rsi_buy': [30, 35, 40],
        'rsi_sell': [60, 65, 70],
        'ma_short': [5, 10],
        'ma_long': [20, 30],
        'use_or': [True, False],
    }
    keys = list(param_grid.keys())
    combinations = list(product(*param_grid.values()))
    results = []
    for values in combinations[:15]:
        params = dict(zip(keys, values))
        try:
            result = backtest_with_params(data.copy(), params)
            results.append(result)
        except Exception:
            continue
    if not results:
        return {}
    # 最优选取规则：优先 num_trades>=2 再按收益；故报告可能看到 48 笔而非 1 笔（1 笔是另一组参数按纯收益最高）。若需纯按收益排序改为 key=lambda x: -x['return_pct']
    results.sort(key=lambda x: (-1 if x.get('num_trades', 0) >= 2 else 0, -x['return_pct']))
    best = results[0]
    return {
        'params': best['params'],
        'return_pct': best['return_pct'],
        'win_rate': best['win_rate'],
        'num_trades': best['num_trades'],
        'avg_per_trade_pct': best.get('avg_per_trade_pct'),
        'top_per_trade_pct': best.get('top_per_trade_pct'),
    }


if __name__ == '__main__':
    results = grid_search()
