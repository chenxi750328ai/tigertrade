#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析真实采集的SIL2603数据并给出tiger1.py参数优化建议
"""

import pandas as pd
import numpy as np
from datetime import datetime

def analyze_real_data():
    """分析真实交易数据"""
    print("🔍 开始分析真实交易数据...")
    
    try:
        # 读取数据
        df = pd.read_csv('trading_data.csv')
        print(f"📊 已加载 {len(df)} 条交易数据")
        
        # 筛选出有效的数值数据
        df_numeric = df[['price_current', 'grid_lower', 'grid_upper', 'atr', 'rsi_1m', 'rsi_5m', 'buffer']].dropna()
        
        if len(df_numeric) == 0:
            print("❌ 没有有效的数值数据")
            return
        
        print(f"📊 有效数值数据记录数: {len(df_numeric)}")
        
        # 计算基本统计信息
        print("\n📈 SIL2603真实市场特征分析:")
        print("-" * 50)
        
        # 价格统计
        price_stats = {
            'min': df_numeric['price_current'].min(),
            'max': df_numeric['price_current'].max(),
            'mean': df_numeric['price_current'].mean(),
            'std': df_numeric['price_current'].std()
        }
        
        print(f"价格范围: {price_stats['min']:.3f} - {price_stats['max']:.3f}")
        print(f"平均价格: {price_stats['mean']:.3f}")
        print(f"价格标准差: {price_stats['std']:.3f}")
        
        # ATR统计
        atr_stats = {
            'min': df_numeric['atr'].min(),
            'max': df_numeric['atr'].max(),
            'mean': df_numeric['atr'].mean()
        }
        
        print(f"ATR范围: {atr_stats['min']:.3f} - {atr_stats['max']:.3f}")
        print(f"平均ATR: {atr_stats['mean']:.3f}")
        
        # RSI统计
        rsi_1m_stats = {
            'mean': df_numeric['rsi_1m'].mean(),
            'min': df_numeric['rsi_1m'].min(),
            'max': df_numeric['rsi_1m'].max()
        }
        
        rsi_5m_stats = {
            'mean': df_numeric['rsi_5m'].mean(),
            'min': df_numeric['rsi_5m'].min(),
            'max': df_numeric['rsi_5m'].max()
        }
        
        print(f"平均RSI(1m): {rsi_1m_stats['mean']:.2f} (范围: {rsi_1m_stats['min']:.2f}-{rsi_1m_stats['max']:.2f})")
        print(f"平均RSI(5m): {rsi_5m_stats['mean']:.2f} (范围: {rsi_5m_stats['min']:.2f}-{rsi_5m_stats['max']:.2f})")
        
        # 波动率分析 (ATR/价格)
        avg_atr_price_ratio = atr_stats['mean'] / price_stats['mean']
        print(f"平均ATR/价格比率: {avg_atr_price_ratio:.4f}")
        
        # 计算每日价格变化（如果数据足够）
        if len(df_numeric) > 1:
            price_changes = df_numeric['price_current'].pct_change().abs()
            avg_daily_change = price_changes.mean() * 100  # 转换为百分比
            print(f"平均价格变化幅度: {avg_daily_change:.3f}%")
        
        # RSI超买超卖情况
        rsi_1m_overbought = (df_numeric['rsi_1m'] > 70).sum() / len(df_numeric) * 100
        rsi_1m_oversold = (df_numeric['rsi_1m'] < 30).sum() / len(df_numeric) * 100
        rsi_5m_overbought = (df_numeric['rsi_5m'] > 70).sum() / len(df_numeric) * 100
        rsi_5m_oversold = (df_numeric['rsi_5m'] < 30).sum() / len(df_numeric) * 100
        
        print(f"RSI(1m)超买(>70)比例: {rsi_1m_overbought:.2f}%")
        print(f"RSI(1m)超卖(<30)比例: {rsi_1m_oversold:.2f}%")
        print(f"RSI(5m)超买(>70)比例: {rsi_5m_overbought:.2f}%")
        print(f"RSI(5m)超卖(<30)比例: {rsi_5m_oversold:.2f}%")
        
        # 分析网格参数
        grid_stats = {
            'lower_min': df_numeric['grid_lower'].min(),
            'lower_max': df_numeric['grid_lower'].max(),
            'upper_min': df_numeric['grid_upper'].min(),
            'upper_max': df_numeric['grid_upper'].max(),
            'lower_mean': df_numeric['grid_lower'].mean(),
            'upper_mean': df_numeric['grid_upper'].mean()
        }
        
        print(f"网格下轨范围: {grid_stats['lower_min']:.3f} - {grid_stats['lower_max']:.3f}")
        print(f"网格上轨范围: {grid_stats['upper_min']:.3f} - {grid_stats['upper_max']:.3f}")
        
        # 基于真实数据分析参数优化
        print(f"\n💡 基于真实数据的参数优化建议:")
        print("-" * 50)
        
        print(f"📊 基于平均价格 ({price_stats['mean']:.2f}) 的建议:")
        print(f"📊 基于平均ATR ({atr_stats['mean']:.4f}) 的建议:")
        print(f"📊 ATR/价格比率: {avg_atr_price_ratio:.4f}")
        
        # 根据ATR/价格比率判断市场波动性
        if avg_atr_price_ratio > 0.015:
            print("⚠️ 市场波动较高，建议增加止损ATR乘数至: 1.5")
            suggested_stop_loss_mult = 1.5
        elif avg_atr_price_ratio < 0.005:
            print("⚠️ 市场波动较低，建议减少止损ATR乘数至: 0.8")
            suggested_stop_loss_mult = 0.8
        else:
            print("✅ 当前市场波动适中，维持止损ATR乘数: 1.2")
            suggested_stop_loss_mult = 1.2
        
        # 止盈参数建议
        if avg_atr_price_ratio > 0.015:
            print("⚠️ 市场波动较高，建议增加止盈参数: ATR偏移0.2→0.25, 最小偏移0.02→0.025")
            suggested_tp_atr = 0.25
            suggested_tp_min = 0.025
        else:
            print("✅ 当前止盈参数适合当前市场波动")
            suggested_tp_atr = 0.2
            suggested_tp_min = 0.02
        
        # RSI参数建议
        rsi_1m_var = rsi_1m_stats['max'] - rsi_1m_stats['min']
        rsi_5m_var = rsi_5m_stats['max'] - rsi_5m_stats['min']
        print(f"📊 RSI变异性 - 1m: {rsi_1m_var:.2f}, 5m: {rsi_5m_var:.2f}")
        
        if rsi_1m_var > 60:
            print("⚠️ 1分钟RSI变化剧烈，可能过于敏感，建议适当调整RSI阈值范围")
        else:
            print("✅ RSI变化在合理范围内")
        
        # 网格参数建议
        avg_grid_width = grid_stats['upper_mean'] - grid_stats['lower_mean']
        print(f"📊 平均网格宽度: {avg_grid_width:.3f}")
        
        if avg_grid_width > 2:
            print("⚠️ 网格较宽，建议适当增加网格间隔以适应波动")
        elif avg_grid_width < 0.5:
            print("⚠️ 网格较窄，可考虑减小网格间隔以增加交易机会")
        else:
            print("✅ 网格宽度适合当前市场")
        
        print(f"📊 平均日价格变化: {avg_daily_change:.3f}%")
        if avg_daily_change > 2.0:
            print("⚠️ 价格波动较大，可能需要更保守的风险控制")
        elif avg_daily_change < 0.5:
            print("⚠️ 价格波动较小，可适当增加交易频率")
        else:
            print("✅ 价格波动适中，当前风险控制参数合适")
        
        print(f"\n🎯 综合参数优化建议:")
        print(f"  - STOP_LOSS_MULTIPLIER: 1.2 → {suggested_stop_loss_mult}")
        print(f"  - TAKE_PROFIT_ATR_OFFSET: 0.2 → {suggested_tp_atr}")
        print(f"  - TAKE_PROFIT_MIN_OFFSET: 0.02 → {suggested_tp_min}")
        print(f"  - GRID_MAX_POSITION: 3 (保持不变，根据账户资金调整)")
        print(f"  - DAILY_LOSS_LIMIT: $1200 (根据账户规模调整)")
        
        print(f"\n🔧 额外建议:")
        print(f"  - 考虑调整RSI阈值以适应当前市场波动")
        print(f"  - 根据ATR/价格比率({avg_atr_price_ratio:.4f})，可考虑动态调整网格参数")
        print(f"  - 建议在高波动时期降低单笔交易量以控制风险")
        print(f"  - 根据日变化率({avg_daily_change:.3f}%)，可考虑调整交易频率")
        
        print(f"\n📊 建议定期监控的指标:")
        print(f"  - ATR/价格比率: 当前 {avg_atr_price_ratio:.4f}")
        print(f"  - RSI 1分钟和5分钟的变异性")
        print(f"  - 网格宽度变化")
        print(f"  - 价格日变化率")
        print(f"  - 成交量变化趋势")
        
        print(f"\n✅ 真实数据分析完成！")
        print(f"📈 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except FileNotFoundError:
        print("❌ 未找到 trading_data.csv 文件")
    except Exception as e:
        print(f"❌ 分析过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_real_data()