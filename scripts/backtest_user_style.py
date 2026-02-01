#!/usr/bin/env python3
"""
回测用户风格策略
使用真实的SIL白银期货历史数据
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from datetime import datetime
from strategy_user_style import UserStyleStrategy

def load_historical_data():
    """加载历史数据（从之前采集的数据）"""
    data_path = '/home/cx/tigertrade/data/large_dataset_real.csv'
    
    if os.path.exists(data_path):
        print(f"📊 加载历史数据: {data_path}")
        df = pd.read_csv(data_path)
        print(f"   共 {len(df)} 条记录")
        return df
    else:
        print(f"❌ 数据文件不存在: {data_path}")
        print("   请先运行数据采集脚本")
        return None

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算技术指标"""
    print("\n🔧 计算技术指标...")
    
    # EMA
    df['ema_20'] = df['price_current'].ewm(span=20, adjust=False).mean()
    
    # RSI (简化版)
    delta = df['price_current'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'].fillna(50, inplace=True)
    
    # BOLL
    df['boll_mid'] = df['price_current'].rolling(window=20).mean()
    df['boll_std'] = df['price_current'].rolling(window=20).std()
    df['boll_upper'] = df['boll_mid'] + 2 * df['boll_std']
    df['boll_lower'] = df['boll_mid'] - 2 * df['boll_std']
    
    # 填充NaN
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)
    
    print(f"✅ 技术指标计算完成")
    return df

def run_backtest(df: pd.DataFrame, initial_capital: float = 100000):
    """
    运行回测
    
    参数:
        df: 历史数据
        initial_capital: 初始资金
    """
    print("\n" + "=" * 80)
    print("🚀 开始回测 - 用户风格策略")
    print("=" * 80)
    
    # 初始化策略
    strategy = UserStyleStrategy()
    
    # 回测状态
    capital = initial_capital
    position = 0  # 当前持仓
    trades = []  # 交易记录
    
    print(f"\n初始资金: ${capital:,.2f}\n")
    
    # 遍历数据
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"进度: {idx}/{len(df)} ({idx/len(df)*100:.1f}%)")
        
        price = row['price_current']
        timestamp = pd.to_datetime(row['timestamp']) if 'timestamp' in row.index and pd.notna(row['timestamp']) else datetime.now()
        
        # 准备指标
        indicators = {
            'ema_20': row.get('ema_20', price),
            'rsi': row.get('rsi', 50),
            'boll_upper': row.get('boll_upper', price * 1.02),
            'boll_lower': row.get('boll_lower', price * 0.98),
            'boll_mid': row.get('boll_mid', price)
        }
        
        # 获取交易信号
        signal = strategy.get_signal(price, indicators, timestamp)
        
        # 执行交易
        if signal['action'] == 'BUY':
            qty = signal['quantity']
            cost = price * qty * 1000  # 白银合约价值
            
            if capital >= cost:
                strategy.open_position(price, timestamp) if position == 0 else strategy.add_position(price, timestamp)
                position += qty
                capital -= cost
                
                trades.append({
                    'timestamp': timestamp,
                    'action': 'BUY',
                    'price': price,
                    'quantity': qty,
                    'capital': capital,
                    'position': position,
                    'reason': signal['reason']
                })
        
        elif signal['action'] == 'SELL' and position > 0:
            qty = min(signal['quantity'], position)
            revenue = price * qty * 1000
            
            pnl = strategy.close_position(price, qty, timestamp)
            position -= qty
            capital += revenue
            
            trades.append({
                'timestamp': timestamp,
                'action': 'SELL',
                'price': price,
                'quantity': qty,
                'capital': capital,
                'position': position,
                'pnl': pnl,
                'reason': signal['reason']
            })
    
    # 强制平仓剩余持仓
    if position > 0:
        final_price = df.iloc[-1]['price_current']
        pnl = strategy.close_position(final_price, position, datetime.now())
        capital += final_price * position * 1000
        print(f"\n⚠️  强制平仓: {position}手 @ ${final_price:.2f}, 盈亏=${pnl:.2f}")
    
    # 计算结果
    final_capital = capital
    total_return = ((final_capital - initial_capital) / initial_capital) * 100
    
    # 策略统计
    stats = strategy.get_stats()
    
    # 打印结果
    print("\n" + "=" * 80)
    print("📊 回测结果")
    print("=" * 80)
    
    print(f"\n【资金变化】")
    print(f"  初始资金: ${initial_capital:,.2f}")
    print(f"  最终资金: ${final_capital:,.2f}")
    print(f"  净盈亏: ${final_capital - initial_capital:,.2f}")
    print(f"  收益率: {total_return:.2f}%")
    
    print(f"\n【交易统计】")
    print(f"  总交易数: {len(trades)}笔")
    print(f"  买入次数: {len([t for t in trades if t['action'] == 'BUY'])}次")
    print(f"  卖出次数: {len([t for t in trades if t['action'] == 'SELL'])}次")
    
    print(f"\n【策略统计】")
    print(f"  完成轮次: {stats['total_rounds']}轮")
    print(f"  总盈亏: ${stats['total_pnl']:,.2f}")
    print(f"  平均每轮: ${stats['avg_pnl']:,.2f}")
    print(f"  胜率: {stats['win_rate']:.1f}%")
    print(f"  平均开仓次数: {stats['avg_entries']:.1f}次")
    print(f"  最大盈利: ${stats['max_pnl']:,.2f}")
    print(f"  最大亏损: ${stats['min_pnl']:,.2f}")
    
    # 对比用户真实表现
    print(f"\n【与用户真实表现对比】")
    print(f"  用户19天收益: 71.37% ($50,801)")
    print(f"  策略回测收益: {total_return:.2f}% (${final_capital - initial_capital:,.2f})")
    
    if total_return > 0:
        print(f"  ✅ 策略盈利！")
    else:
        print(f"  ⚠️  策略亏损，需要调整")
    
    print("\n" + "=" * 80)
    
    # 保存交易记录
    if trades:
        df_trades = pd.DataFrame(trades)
        output_path = '/home/cx/tigertrade/backtest_results/user_style_trades.csv'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_trades.to_csv(output_path, index=False)
        print(f"💾 交易记录已保存: {output_path}")
    
    return {
        'initial_capital': initial_capital,
        'final_capital': final_capital,
        'total_return': total_return,
        'trades': trades,
        'stats': stats
    }

def main():
    print("=" * 80)
    print("🎯 用户风格策略回测")
    print("=" * 80)
    
    # 加载数据
    df = load_historical_data()
    if df is None:
        return
    
    # 计算指标
    df = calculate_indicators(df)
    
    # 运行回测
    results = run_backtest(df, initial_capital=100000)
    
    print("\n✅ 回测完成！")
    print("\n💡 下一步:")
    print("  1. 查看交易记录分析哪里可以改进")
    print("  2. 调整策略参数（加仓间距、止盈目标等）")
    print("  3. 或者实盘小规模测试")

if __name__ == '__main__':
    main()
