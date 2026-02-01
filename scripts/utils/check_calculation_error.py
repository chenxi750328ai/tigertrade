#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查计算过程中的错误
"""

import sys
import os
import pandas as pd
import numpy as np

# 添加tigertrade目录到路径
sys.path.insert(0, '/home/cx/tigertrade')

from src import tiger1 as t1


def debug_calculation_process():
    """调试计算过程"""
    print("🔍 调试计算过程...")
    
    # 创建符合您数据的测试数据
    # 从您的日志可以看到：
    # 2026-01-16 13:10:00+08:00  90.570  90.605  90.235  90.375     845
    # 2026-01-16 13:15:00+08:00  90.370  90.420  90.290  90.305     133
    # 结果: near_lower=False, 但理论上90.305比90.375更低，应该更接近下轨
    
    # 创建测试数据
    np.random.seed(42)
    
    # 创建5分钟数据
    base_prices_5m = []
    for i in range(45):  # 前45个点用于建立趋势
        base_price = 90.0 + 0.3 * np.sin(i/10) + 0.1 * np.random.randn()
        base_prices_5m.append(base_price)
    
    # 添加关键数据点
    base_prices_5m.extend([90.375, 90.305])  # 13:10: 90.375, 13:15: 90.305
    
    df_5m = pd.DataFrame({
        'time': pd.date_range('2026-01-16 10:00', periods=50, freq='5min'),
        'open': base_prices_5m,
        'high': [p + 0.15 for p in base_prices_5m],
        'low': [p - 0.15 for p in base_prices_5m],
        'close': base_prices_5m,
        'volume': [200] * 50
    })
    df_5m.set_index('time', inplace=True)
    
    # 创建1分钟数据
    minute_prices = []
    for i in range(155):  # 与时间索引长度一致
        minute_price = 90.0 + 0.15 * np.sin(i/20) + 0.05 * np.random.randn()
        minute_prices.append(minute_price)
    
    # 最后几个点接近您提到的值
    minute_prices[-10:] = [90.370, 90.372, 90.368, 90.365, 90.360, 90.355, 90.340, 90.320, 90.310, 90.305]
    
    df_1m = pd.DataFrame({
        'time': pd.date_range('2026-01-16 12:00', periods=155, freq='1min'),
        'open': minute_prices,
        'high': [p + 0.08 for p in minute_prices],
        'low': [p - 0.08 for p in minute_prices],
        'close': minute_prices,
        'volume': [50] * 155
    })
    df_1m.set_index('time', inplace=True)
    
    try:
        # 计算指标
        print("📊 计算技术指标...")
        indicators = t1.calculate_indicators(df_1m, df_5m)
        
        print(f"5m指标: {indicators['5m']}")
        print(f"1m指标: {indicators['1m']}")
        
        # 获取当前价格和ATR
        current_price = indicators['1m']['close']
        atr_value = indicators['5m']['atr']
        
        print(f"\n🔧 当前状态:")
        print(f"   当前价格: {current_price}")
        print(f"   ATR值: {atr_value}")
        
        # 调整网格
        original_lower = t1.grid_lower
        original_upper = t1.grid_upper
        
        t1.adjust_grid_interval("osc_normal", indicators)
        actual_grid_lower = t1.grid_lower
        actual_grid_upper = t1.grid_upper
        
        print(f"   调整后网格下轨: {actual_grid_lower}")
        print(f"   调整后网格上轨: {actual_grid_upper}")
        
        # 使用当前参数计算
        buffer = max(0.05 * (atr_value if atr_value else 0), 0.0025)
        threshold = actual_grid_lower + buffer
        near_lower = current_price <= threshold
        
        print(f"\n📈 计算过程:")
        print(f"   当前价格: {current_price}")
        print(f"   网格下轨: {actual_grid_lower}")
        print(f"   ATR: {atr_value}")
        print(f"   Buffer计算: max(0.05 * {atr_value}, 0.0025) = {buffer}")
        print(f"   阈值计算: {actual_grid_lower} + {buffer} = {threshold}")
        print(f"   near_lower计算: {current_price} <= {threshold} = {near_lower}")
        
        print(f"\n💡 问题分析:")
        print(f"   价格{current_price}与下轨{actual_grid_lower}的差值: {current_price - actual_grid_lower}")
        print(f"   如果差值为负，说明价格低于下轨，应该触发near_lower=True")
        print(f"   但目前near_lower={near_lower}，说明阈值{threshold}仍然大于当前价格{current_price}")
        
        # 恢复原始值
        t1.grid_lower = original_lower
        t1.grid_upper = original_upper
        
        return {
            'current_price': current_price,
            'grid_lower': actual_grid_lower,
            'buffer': buffer,
            'threshold': threshold,
            'near_lower': near_lower
        }
        
    except Exception as e:
        print(f"❌ 计算出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_detailed_log():
    """打印详细日志"""
    print(f"\n📋 详细日志输出:")
    print(f"当策略执行时，应该输出类似这样的日志:")
    print(f"   🔧 grid_trading_strategy_pro1: near_lower=T/F, rsi_ok=T/F, trend_check=T/F, rebound=T/F, vol_ok=T/F")
    print(f"   价格: 90.305, 网格下轨: X.XXX, buffer: X.XXX, 阈值: X.XXX, near_lower: False")
    print(f"   如果价格 < 阈值，near_lower应该是True")


def analyze_issue():
    """分析问题"""
    print(f"\n🔍 问题分析:")
    print(f"   从您的日志来看: near_lower=False")
    print(f"   但价格从90.375下降到90.305，理论上应该更接近下轨")
    print(f"   这意味着计算过程中可能存在以下问题:")
    print(f"   1. BOLL下轨也随着价格下降而下降")
    print(f"   2. ATR值较大，导致buffer过大")
    print(f"   3. 计算公式有误")
    print(f"   4. 数据获取不准确")
    
    print(f"\n🔧 验证计算逻辑:")
    print(f"   正确的计算应该是:")
    print(f"   - 获取当前价格")
    print(f"   - 获取网格下轨值")
    print(f"   - 计算buffer = max(0.05 * atr, 0.0025)")
    print(f"   - 计算阈值 = grid_lower + buffer")
    print(f"   - 判断 near_lower = current_price <= threshold")
    
    # 示例计算
    print(f"\n📝 示例计算:")
    examples = [
        {"price": 90.305, "lower": 90.290, "atr": 0.1},
        {"price": 90.305, "lower": 90.200, "atr": 0.2},
        {"price": 90.305, "lower": 90.300, "atr": 0.05}
    ]
    
    for ex in examples:
        buffer = max(0.05 * ex["atr"], 0.0025)
        threshold = ex["lower"] + buffer
        result = ex["price"] <= threshold
        
        print(f"   价格:{ex['price']}, 下轨:{ex['lower']}, ATR:{ex['atr']}")
        print(f"   => buffer={buffer:.4f}, 阈值={threshold:.4f}, near_lower={result}")


if __name__ == "__main__":
    print("🚀 开始检查计算过程错误...\n")
    
    result = debug_calculation_process()
    print_detailed_log()
    analyze_issue()
    
    if result:
        print(f"\n✅ 调试完成!")
        print(f"   当前价格: {result['current_price']:.3f}")
        print(f"   网格下轨: {result['grid_lower']:.3f}")
        print(f"   Buffer: {result['buffer']:.4f}")
        print(f"   阈值: {result['threshold']:.4f}")
        print(f"   near_lower: {result['near_lower']}")
    else:
        print(f"❌ 调试失败")