#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
精确调试BOLL计算和near_lower逻辑
"""

import sys
sys.path.insert(0, '/home/cx/tigertrade')

from src import tiger1 as t1
import pandas as pd
import numpy as np
import talib

def simulate_boll_calculation():
    """模拟BOLL指标计算"""
    print("🔍 模拟BOLL指标计算...")
    
    # 构建与问题场景相似的数据
    # 根据日志：90.500 90.645 90.415 90.615 486 2026-01-16 12:35:00+08:00 90.620 90.670 90.560 90.600 192
    # 我们需要构造一个包含这些数据点的序列，使得最后的BOLL下轨大约是某个值
    
    # 创建一个更接近实际的数据集
    np.random.seed(42)  # 固定随机种子以获得一致的结果
    
    # 构造一个价格序列，使其最后几个点接近观察到的值
    base_prices = 90.5 + 0.2 * np.sin(np.linspace(0, 4*np.pi, 50)) + 0.05 * np.random.randn(50)
    
    # 强制最后几个数据点接近观察到的值
    base_prices[-5:] = [90.55, 90.58, 90.615, 90.620, 90.600]
    
    # 创建5分钟数据
    df_5m = pd.DataFrame({
        'time': pd.date_range('2026-01-16 10:00', periods=50, freq='5min'),
        'open': base_prices,
        'high': base_prices + 0.15,
        'low': base_prices - 0.15,
        'close': base_prices,
        'volume': [200] * 50
    })
    df_5m.set_index('time', inplace=True)
    
    # 创建1分钟数据（更细粒度）
    minute_base_prices = 90.5 + 0.1 * np.sin(np.linspace(0, 20*np.pi, 150)) + 0.02 * np.random.randn(150)
    minute_base_prices[-10:] = [90.58, 90.59, 90.595, 90.605, 90.610, 90.612, 90.615, 90.620, 90.610, 90.600]
    
    df_1m = pd.DataFrame({
        'time': pd.date_range('2026-01-16 12:00', periods=150, freq='1min'),
        'open': minute_base_prices,
        'high': minute_base_prices + 0.08,
        'low': minute_base_prices - 0.08,
        'close': minute_base_prices,
        'volume': [50] * 150
    })
    df_1m.set_index('time', inplace=True)
    
    print(f"📊 构造的5分钟数据最后几行:")
    print(df_5m[['close']].tail())
    
    print(f"\n📊 构造的1分钟数据最后几行:")
    print(df_1m[['close']].tail())
    
    try:
        # 计算指标
        indicators = t1.calculate_indicators(df_1m, df_5m)
        
        print(f"\n📈 计算出的指标:")
        if '5m' in indicators:
            print(f"   5m指标: {indicators['5m']}")
        if '1m' in indicators:
            print(f"   1m指标: {indicators['1m']}")
        
        # 获取当前价格
        current_price = indicators['1m']['close'] if '1m' in indicators and 'close' in indicators['1m'] else 90.600
        atr_value = indicators['5m']['atr'] if '5m' in indicators and 'atr' in indicators['5m'] and indicators['5m']['atr'] is not None else 0.1
        
        print(f"\n🔧 计算near_lower逻辑:")
        print(f"   当前价格: {current_price}")
        print(f"   ATR值: {atr_value}")
        
        # 执行adjust_grid_interval
        t1.adjust_grid_interval("osc_normal", indicators)
        calculated_grid_lower = t1.grid_lower
        calculated_grid_upper = t1.grid_upper
        
        print(f"   调整后的grid_lower: {calculated_grid_lower}")
        print(f"   调整后的grid_upper: {calculated_grid_upper}")
        
        # 计算buffer
        buffer = max(0.5 * (atr_value if atr_value else 0), 0.02)
        print(f"   计算buffer: max(0.5 * {atr_value}, 0.02) = {buffer}")
        
        # 计算near_lower
        threshold = calculated_grid_lower + buffer
        near_lower = current_price <= threshold
        print(f"   阈值: {calculated_grid_lower} + {buffer} = {threshold}")
        print(f"   near_lower: {current_price} <= {threshold} = {near_lower}")
        
        print(f"\n💡 分析:")
        if near_lower:
            print(f"   ✓ 价格{current_price} <= 阈值{threshold}，所以near_lower=True")
        else:
            print(f"   ✗ 价格{current_price} > 阈值{threshold}，所以near_lower=False")
        
        # 如果near_lower是False，但你认为应该是True，那说明实际的grid_lower值计算有问题
        if not near_lower:
            required_grid_lower = current_price - buffer
            print(f"\n🔍 如果需要near_lower=True，则grid_lower需要 <= {required_grid_lower}")
            print(f"   但实际计算出的grid_lower是: {calculated_grid_lower}")
            print(f"   差距: {calculated_grid_lower - required_grid_lower}")
        
    except Exception as e:
        print(f"❌ 模拟计算出错: {e}")
        import traceback
        traceback.print_exc()


def analyze_log_scenario():
    """分析日志中的场景"""
    print(f"\n🔍 分析日志场景:")
    print(f"日志显示: '90.600不是靠近下限90.620'")
    print(f"但实际上: 90.600 < 90.620")
    print(f"所以near_lower应该为True")
    print(f"\n可能的原因:")
    print(f"1. 实际的grid_lower值不是90.620，而是通过adjust_grid_interval函数计算的BOLL下轨")
    print(f"2. BOLL下轨的计算可能因为数据不同而产生了不同于90.620的值")
    print(f"3. 也可能是其他指标影响了判断")
    
    print(f"\n🔧 验证计算公式:")
    print(f"near_lower = price_current <= (grid_lower + buffer)")
    print(f"其中 buffer = max(0.5 * atr, 0.02)")
    print(f"如果 price_current = 90.600, grid_lower = 90.620, atr = 0.04 (举例)")
    print(f"则 buffer = max(0.5 * 0.04, 0.02) = 0.02")
    print(f"阈值 = 90.620 + 0.02 = 90.640")
    print(f"判断: 90.600 <= 90.640 = True")
    print(f"所以near_lower应该为True")


if __name__ == "__main__":
    simulate_boll_calculation()
    analyze_log_scenario()