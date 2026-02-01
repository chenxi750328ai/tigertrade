#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证修复效果的精确测试
"""

import sys
import os
import pandas as pd
import numpy as np
import talib

# 添加tigertrade目录到路径
sys.path.insert(0, '/home/cx/tigertrade')

from src import tiger1 as t1


def create_test_data_like_original_problem():
    """创建类似原始问题的数据"""
    print("🔍 创建类似原始问题的测试数据...")
    
    # 根据原始日志："90.500 90.645 90.415 90.615 486 2026-01-16 12:35:00+08:00 90.620 90.670 90.560 90.600 192"
    # 构造符合这一序列的数据
    np.random.seed(42)
    
    # 创建5分钟数据，确保BOLL指标计算出接近90.620的下轨
    base_prices_5m = 90.5 + 0.2 * np.sin(np.linspace(0, 4*np.pi, 50)) + 0.05 * np.random.randn(50)
    # 强制最后几个数据点接近观察到的值
    base_prices_5m[-5:] = [90.55, 90.58, 90.615, 90.620, 90.600]

    df_5m = pd.DataFrame({
        'time': pd.date_range('2026-01-16 10:00', periods=50, freq='5min'),
        'open': base_prices_5m,
        'high': base_prices_5m + 0.15,
        'low': base_prices_5m - 0.15,
        'close': base_prices_5m,
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

    return df_1m, df_5m


def test_before_after_fix():
    """测试修复前后的对比"""
    print(f"\n🔧 测试修复前后的对比...")
    
    df_1m, df_5m = create_test_data_like_original_problem()
    
    # 计算指标
    indicators = t1.calculate_indicators(df_1m, df_5m)
    
    if indicators and '1m' in indicators:
        current_price = indicators['1m']['close']
        atr_value = indicators['5m']['atr'] if '5m' in indicators and 'atr' in indicators['5m'] and indicators['5m']['atr'] is not None else 0.1
    else:
        current_price = 90.600
        atr_value = 0.1
    
    print(f"📊 测试数据:")
    print(f"   当前价格: {current_price}")
    print(f"   ATR值: {atr_value}")
    
    # 获取调整后的网格值
    t1.adjust_grid_interval("osc_normal", indicators)
    grid_lower = t1.grid_lower
    grid_upper = t1.grid_upper
    
    print(f"   调整后网格下轨: {grid_lower}")
    print(f"   调整后网格上轨: {grid_upper}")
    
    # 计算修复前的逻辑
    old_buffer = max(0.5 * (atr_value if atr_value else 0), 0.02)
    old_threshold = grid_lower + old_buffer
    old_near_lower = current_price <= old_threshold
    
    # 计算修复后的逻辑
    new_buffer = max(0.2 * (atr_value if atr_value else 0), 0.01)
    new_threshold = grid_lower + new_buffer
    new_near_lower = current_price <= new_threshold
    
    print(f"\n📈 修复前后对比:")
    print(f"   修复前 - buffer: {old_buffer:.4f}, 阈值: {old_threshold:.4f}, near_lower: {old_near_lower}")
    print(f"   修复后 - buffer: {new_buffer:.4f}, 阈值: {new_threshold:.4f}, near_lower: {new_near_lower}")
    
    print(f"\n💡 修复效果:")
    if new_near_lower != old_near_lower:
        if new_near_lower and not old_near_lower:
            print(f"   ✅ 修复成功! 修复后能够正确识别价格接近下轨的情况")
        else:
            print(f"   ❌ 修复方向可能有误")
    else:
        print(f"   修复未改变结果，可能需要进一步调整参数")
    
    return {
        'old_result': old_near_lower,
        'new_result': new_near_lower,
        'improved': new_near_lower and not old_near_lower
    }


def run_simulation_test():
    """运行仿真测试"""
    print(f"\n🔍 运行仿真测试...")
    
    # 测试多种市场情况
    scenarios = [
        {"price": 90.600, "grid_lower": 90.200, "atr": 0.310, "desc": "原始问题场景"},
        {"price": 89.500, "grid_lower": 89.000, "atr": 0.100, "desc": "低价格低波动场景"},
        {"price": 100.00, "grid_lower": 99.500, "atr": 0.200, "desc": "高价格中波动场景"},
        {"price": 95.000, "grid_lower": 94.800, "atr": 0.050, "desc": "低波动场景"},
        {"price": 92.000, "grid_lower": 91.000, "atr": 0.500, "desc": "高波动场景"},
    ]
    
    improvements = 0
    total_tests = len(scenarios)
    
    print(f"{'场景':<15} {'价格':<8} {'下轨':<8} {'ATR':<6} {'旧结果':<8} {'新结果':<8} {'改善':<6}")
    print("-" * 70)
    
    for scenario in scenarios:
        old_buffer = max(0.5 * scenario['atr'], 0.02)
        old_threshold = scenario['grid_lower'] + old_buffer
        old_result = scenario['price'] <= old_threshold
        
        new_buffer = max(0.2 * scenario['atr'], 0.01)
        new_threshold = scenario['grid_lower'] + new_buffer
        new_result = scenario['price'] <= new_threshold
        
        improved = new_result and not old_result
        if improved:
            improvements += 1
        
        improvement_str = "✅" if improved else ""
        
        print(f"{scenario['desc']:<15} {scenario['price']:<8.3f} {scenario['grid_lower']:<8.3f} {scenario['atr']:<6.3f} "
              f"{str(old_result):<8} {str(new_result):<8} {improvement_str:<6}")
    
    print(f"\n📊 测试结果:")
    print(f"   总测试数: {total_tests}")
    print(f"   改善数量: {improvements}")
    print(f"   改善比例: {improvements/total_tests*100:.1f}%")
    
    return improvements > 0


def detailed_debug_of_original_case():
    """详细调试原始案例"""
    print(f"\n🔍 详细调试原始案例...")
    
    # 使用原始日志中的数据
    # 从日志 "90.600不是靠近下限90.620" 推断
    # 实际价格: 90.600, 声称下限: 90.620, 但near_lower=False
    
    print(f"原始场景分析:")
    print(f"  声称价格: 90.600")
    print(f"  声称下限: 90.620")
    print(f"  实际情况: near_lower=False (但应该为True)")
    
    # 实际上，grid_lower是通过BOLL计算的，不是90.620
    # 让我们反推实际的grid_lower值
    price_current = 90.600
    
    # 假设ATR为0.3（从之前的调试中看到）
    atr = 0.3
    
    # 修复前的参数
    old_buffer = max(0.5 * atr, 0.02)  # = max(0.15, 0.02) = 0.15
    old_threshold = price_current - 0.001  # 我们知道near_lower=False，所以阈值必须小于price_current
    actual_old_grid_lower = old_threshold - old_buffer
    
    # 修复后的参数
    new_buffer = max(0.2 * atr, 0.01)  # = max(0.06, 0.01) = 0.06
    new_threshold = actual_old_grid_lower + new_buffer
    
    print(f"\n反推计算:")
    print(f"  假设ATR: {atr}")
    print(f"  修复前buffer: max(0.5 * {atr}, 0.02) = {old_buffer}")
    print(f"  要使near_lower=False，需要grid_lower < {price_current - old_buffer:.3f}")
    print(f"  假设实际grid_lower = {actual_old_grid_lower:.3f}")
    print(f"  验证: {price_current} <= ({actual_old_grid_lower:.3f} + {old_buffer}) = {price_current <= (actual_old_grid_lower + old_buffer)}")
    
    print(f"\n修复后效果:")
    print(f"  修复后buffer: max(0.2 * {atr}, 0.01) = {new_buffer}")
    print(f"  新阈值: {actual_old_grid_lower:.3f} + {new_buffer} = {actual_old_grid_lower + new_buffer:.3f}")
    print(f"  新结果: {price_current} <= {actual_old_grid_lower + new_buffer:.3f} = {price_current <= (actual_old_grid_lower + new_buffer)}")
    
    if price_current <= (actual_old_grid_lower + new_buffer) and not (price_current <= (actual_old_grid_lower + old_buffer)):
        print(f"  ✅ 修复成功! 从False变为True")
    else:
        print(f"  📊 结果: 修复前后结果相同")


def create_corrected_log_output():
    """创建正确的日志输出"""
    print(f"\n🔧 创建正确的日志输出...")
    
    print(f"修复后的near_lower计算逻辑:")
    print(f"  buffer = max(0.2 * atr, 0.01)  # 从原来的max(0.5 * atr, 0.02)调整而来")
    print(f"  near_lower = price_current <= (grid_lower + buffer)")
    print(f"  ")
    print(f"这个调整的目的:")
    print(f"  1. 减少ATR对buffer的影响 (0.5 -> 0.2)，使信号更敏感")
    print(f"  2. 降低最小buffer值 (0.02 -> 0.01)，增加精细度")
    print(f"  3. 在高波动市场中仍能及时捕捉价格接近下轨的信号")
    print(f"  ")
    print(f"参数选择依据:")
    print(f"  - 0.2系数：在保持稳定性的同时提高敏感度")
    print(f"  - 0.01最小值：适用于大多数市场价格水平")
    print(f"  - 平衡了误报率和漏报率")


if __name__ == "__main__":
    print("🚀 开始验证修复效果...")
    
    result = test_before_after_fix()
    improvement_found = run_simulation_test()
    detailed_debug_of_original_case()
    create_corrected_log_output()
    
    print(f"\n✅ 验证完成!")
    print(f"   修复前near_lower: {result['old_result']}")
    print(f"   修复后near_lower: {result['new_result']}")
    print(f"   是否改善: {result['improved']}")
    print(f"   仿真测试改善: {improvement_found}")
    
    if result['improved'] or improvement_found:
        print(f"\n🎉 修复验证成功！新的参数设置能够更好地识别价格接近下轨的情况。")
    else:
        print(f"\n🤔 修复可能需要进一步调整。")