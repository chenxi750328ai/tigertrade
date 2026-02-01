#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
调试计算逻辑，验证near_lower计算是否正确
"""

import sys
sys.path.insert(0, '/home/cx/tigertrade')

from src import tiger1 as t1
import pandas as pd
import numpy as np
import talib

def debug_with_real_values():
    """使用实际值进行调试"""
    print("🔍 使用实际值调试计算逻辑...")
    
    # 手动设置tiger1模块的全局变量
    t1.FUTURE_SYMBOL = "SIL.COMEX.202603"
    t1.GRID_BOLL_PERIOD = 20
    t1.GRID_BOLL_STD = 2
    t1.GRID_ATR_PERIOD = 14
    t1.GRID_RSI_PERIOD_1M = 14
    t1.GRID_RSI_PERIOD_5M = 14
    
    # 从日志中提取的信息
    # 90.500 90.645 90.415 90.615 486 2026-01-16 12:35:00+08:00 90.620 90.670 90.560 90.600 192
    # 12:40的时间点，价格是90.600
    
    print("🔍 重现问题场景:")
    print("   根据日志: near_lower=False, rsi_ok=False, trend_check=False, rebound=False, vol_ok=False")
    print("   这意味着虽然价格90.600可能低于下轨，但其他条件不满足")
    
    # 尝试创建符合实际的日志数据
    print(f"\n🔍 分析near_lower计算逻辑:")
    print(f"   问题: 90.600应该低于下轨90.620，所以near_lower应该是True")
    
    # 让我们反向推导
    price_current = 90.600
    grid_lower_from_log = 90.620  # 根据日志推测
    actual_near_lower_from_log = False  # 根据日志显示
    
    print(f"\n💡 反向推理:")
    print(f"   观察到: price_current={price_current}, near_lower={actual_near_lower_from_log}")
    print(f"   推测的grid_lower: {grid_lower_from_log}")
    
    # 如果near_lower是False，那么:
    # price_current > (grid_lower + buffer)
    # 90.600 > (grid_lower + buffer)
    # 所以 grid_lower < 90.600 - buffer
    
    # 计算最小可能的grid_lower
    test_atr_values = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    print(f"\n🧪 计算不同ATR值下的grid_lower阈值:")
    for test_atr in test_atr_values:
        buffer = max(0.5 * (test_atr if test_atr else 0), 0.02)
        max_grid_lower = price_current - buffer
        print(f"   ATR={test_atr:4.2f} -> buffer={buffer:.3f}, grid_lower必须<{max_grid_lower:.3f}才能使near_lower=False")
    
    # 尝试重现实际计算过程
    print(f"\n🔧 重现实际计算过程...")
    
    # 创建模拟数据来匹配日志
    # 12:35数据: 90.500 90.645 90.415 90.615
    # 12:40数据: 90.620 90.670 90.560 90.600
    data_1m = {
        'open': [90.500, 90.620],
        'high': [90.645, 90.670], 
        'low': [90.415, 90.560],
        'close': [90.615, 90.600],  # 最后一个是90.600
        'volume': [486, 192]
    }
    
    # 创建足够长的数据来计算指标
    close_extended = [90.5 + 0.1*np.sin(i/5) for i in range(30)]  # 模拟30个数据点
    close_extended[-2] = 90.615  # 倒数第二个是12:35的价格
    close_extended[-1] = 90.600  # 最后一个是12:40的价格
    
    df_1m = pd.DataFrame({
        'time': pd.date_range('2026-01-16 12:00', periods=30, freq='1min'),
        'open': [90.5] * 30,
        'high': [90.7] * 30,
        'low': [90.4] * 30,
        'close': close_extended,
        'volume': [100] * 30
    })
    df_1m.set_index('time', inplace=True)
    
    # 类似地创建5分钟数据
    close_5m_extended = [90.5 + 0.1*np.sin(i/2) for i in range(15)]  # 模拟15个5分钟数据点
    close_5m_extended[-1] = 90.600  # 最后一个close是90.600
    
    df_5m = pd.DataFrame({
        'time': pd.date_range('2026-01-16 11:00', periods=15, freq='5min'),
        'open': [90.5] * 15,
        'high': [90.7] * 15,
        'low': [90.4] * 15,
        'close': close_5m_extended,
        'volume': [200] * 15
    })
    df_5m.set_index('time', inplace=True)
    
    try:
        # 计算指标
        indicators = t1.calculate_indicators(df_1m, df_5m)
        
        print(f"   计算出的指标:")
        if '5m' in indicators:
            print(f"     5m指标: {indicators['5m']}")
        if '1m' in indicators:
            print(f"     1m指标: {indicators['1m']}")
        
        # 获取实际值
        if indicators and '1m' in indicators:
            actual_price = indicators['1m']['close'] if 'close' in indicators['1m'] else 90.600
            actual_atr = indicators['5m']['atr'] if 'atr' in indicators['5m'] and indicators['5m']['atr'] is not None else 0.1
        else:
            actual_price = 90.600
            actual_atr = 0.1
            
        print(f"\n   实际计算使用的值:")
        print(f"     实际价格: {actual_price}")
        print(f"     实际ATR: {actual_atr}")
        
        # 现在根据grid_trading_strategy_pro1的逻辑计算
        rsi_low = 25  # 默认值
        
        # 计算buffer
        buffer = max(0.5 * (actual_atr if actual_atr else 0), 0.02)
        print(f"     计算buffer: max(0.5 * {actual_atr}, 0.02) = {buffer}")
        
        # 需要知道实际的grid_lower值，让我们使用adjust_grid_interval来获取
        t1.adjust_grid_interval("osc_normal", indicators)
        actual_grid_lower = t1.grid_lower
        actual_grid_upper = t1.grid_upper
        
        print(f"     实际grid_lower (经adjust_grid_interval调整): {actual_grid_lower}")
        print(f"     实际grid_upper: {actual_grid_upper}")
        
        # 重新计算near_lower
        actual_threshold = actual_grid_lower + buffer
        actual_near_lower = actual_price <= actual_threshold
        
        print(f"     阈值: {actual_grid_lower} + {buffer} = {actual_threshold}")
        print(f"     near_lower: {actual_price} <= {actual_threshold} = {actual_near_lower}")
        
        if actual_near_lower != actual_near_lower_from_log:
            print(f"\n🚨 发现不一致!")
            print(f"   日志显示: near_lower={actual_near_lower_from_log}")
            print(f"   计算得出: near_lower={actual_near_lower}")
            print(f"   这表明实际的grid_lower值与我们计算的不同")
        
    except Exception as e:
        print(f"   ❌ 模拟计算出错: {e}")
        import traceback
        traceback.print_exc()


def verify_calculation_logic():
    """验证near_lower计算逻辑是否正确"""
    print(f"\n🔍 验证near_lower计算逻辑...")
    
    # 按照代码中的实际逻辑进行计算
    price_current = 90.600
    grid_lower = 90.620  # 假设这是正确的值
    
    # 这是代码中的实际计算步骤
    # 1) 计算buffer
    atr = 0.05  # 假设ATR值
    buffer = max(0.5 * (atr if atr else 0), 0.02)
    print(f"   1) buffer = max(0.5 * {atr}, 0.02) = {buffer}")
    
    # 2) 计算near_lower
    near_lower = price_current <= (grid_lower + buffer)
    calculated_threshold = grid_lower + buffer
    print(f"   2) near_lower = {price_current} <= ({grid_lower} + {buffer}) = {price_current} <= {calculated_threshold} = {near_lower}")
    
    print(f"\n💡 问题分析:")
    if near_lower:
        print(f"   根据计算: {price_current} <= {calculated_threshold} 是 True，所以near_lower应该是True")
        print(f"   但日志显示near_lower是False，说明实际的grid_lower > {price_current - buffer} = {price_current - buffer}")
    else:
        print(f"   计算结果与日志一致")


if __name__ == "__main__":
    debug_with_real_values()
    verify_calculation_logic()