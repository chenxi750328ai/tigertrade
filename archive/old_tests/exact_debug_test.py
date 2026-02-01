#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
精确调试测试
"""

import sys
import os
import pandas as pd
import numpy as np

# 添加tigertrade目录到路径
sys.path.insert(0, '/home/cx/tigertrade')

from src import tiger1 as t1


def find_actual_grid_lower():
    """找出实际的grid_lower值"""
    print("🔍 找出实际的grid_lower值...")
    
    # 创建一个包含日志中提到数据的数据集
    # 从日志 "90.500 90.645 90.415 90.615 486 2026-01-16 12:35:00+08:00 90.620 90.670 90.560 90.600 192"
    # 这表示在12:35有价格90.500-90.645-90.415-90.615，在12:40有价格90.620-90.670-90.560-90.600
    
    # 创建5分钟K线数据，确保最后几个点接近观察到的值
    # 为了复现原始问题，我们需要确保BOLL指标计算出合适的下轨
    np.random.seed(42)
    
    # 创建一段价格走势，使得BOLL下轨接近某个值
    base_prices = []
    
    # 前45个数据点 - 模拟一段时间的价格走势
    for i in range(45):
        base_price = 90.0 + 0.3 * np.sin(i/5) + 0.1 * np.random.randn()
        base_prices.append(base_price)
    
    # 最后5个数据点 - 接近期望的值
    base_prices.extend([90.55, 90.58, 90.615, 90.620, 90.600])
    
    df_5m = pd.DataFrame({
        'time': pd.date_range('2026-01-16 10:00', periods=50, freq='5min'),
        'open': base_prices,
        'high': [p + 0.15 for p in base_prices],
        'low': [p - 0.15 for p in base_prices],
        'close': base_prices,
        'volume': [200] * 50
    })
    df_5m.set_index('time', inplace=True)
    
    # 1分钟数据
    minute_prices = []
    for i in range(145):
        minute_price = 90.0 + 0.1 * np.sin(i/10) + 0.05 * np.random.randn()
        minute_prices.append(minute_price)
    
    # 最后几个点接近观察到的值
    minute_prices.extend([90.58, 90.59, 90.595, 90.605, 90.610, 90.612, 90.615, 90.620, 90.610, 90.600])
    
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
        indicators = t1.calculate_indicators(df_1m, df_5m)
        
        print(f"📊 计算出的指标:")
        if '5m' in indicators:
            print(f"   5m指标: {indicators['5m']}")
        if '1m' in indicators:
            print(f"   1m指标: {indicators['1m']}")
        
        # 获取当前价格
        current_price = indicators['1m']['close'] if '1m' in indicators and 'close' in indicators['1m'] else 90.600
        atr_value = indicators['5m']['atr'] if '5m' in indicators and 'atr' in indicators['5m'] and indicators['5m']['atr'] is not None else 0.1
        
        print(f"\n🔧 实际计算过程:")
        print(f"   当前价格: {current_price}")
        print(f"   ATR值: {atr_value}")
        
        # 执行adjust_grid_interval
        original_lower = t1.grid_lower
        original_upper = t1.grid_upper
        
        t1.adjust_grid_interval("osc_normal", indicators)
        actual_grid_lower = t1.grid_lower
        actual_grid_upper = t1.grid_upper
        
        print(f"   调整后的grid_lower: {actual_grid_lower}")
        print(f"   调整后的grid_upper: {actual_grid_upper}")
        
        # 计算新旧参数
        old_buffer = max(0.5 * (atr_value if atr_value else 0), 0.02)
        old_threshold = actual_grid_lower + old_buffer
        old_result = current_price <= old_threshold
        
        new_buffer = max(0.1 * (atr_value if atr_value else 0), 0.005)
        new_threshold = actual_grid_lower + new_buffer
        new_result = current_price <= new_threshold
        
        print(f"\n📈 参数对比:")
        print(f"   旧参数: buffer={old_buffer:.4f}, 阈值={old_threshold:.4f}, near_lower={old_result}")
        print(f"   新参数: buffer={new_buffer:.4f}, 阈值={new_threshold:.4f}, near_lower={new_result}")
        
        # 恢复原始值
        t1.grid_lower = original_lower
        t1.grid_upper = original_upper
        
        return {
            'current_price': current_price,
            'atr_value': atr_value,
            'grid_lower': actual_grid_lower,
            'old_result': old_result,
            'new_result': new_result
        }
        
    except Exception as e:
        print(f"❌ 计算出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_various_scenarios():
    """测试各种场景"""
    print(f"\n🔍 测试各种场景...")
    
    # 测试不同参数组合的影响
    scenarios = [
        {"price": 90.60, "grid_lower": 90.55, "atr": 0.10},
        {"price": 90.60, "grid_lower": 90.58, "atr": 0.10},
        {"price": 90.60, "grid_lower": 90.59, "atr": 0.10},
        {"price": 90.60, "grid_lower": 90.595, "atr": 0.10},
        {"price": 90.60, "grid_lower": 90.599, "atr": 0.10},
    ]
    
    print(f"{'价格':<8} {'下轨':<8} {'ATR':<6} {'旧缓存':<8} {'旧阈值':<8} {'旧结果':<8} {'新缓存':<8} {'新阈值':<8} {'新结果':<8} {'改善':<6}")
    print("-" * 85)
    
    improvements = 0
    
    for scenario in scenarios:
        # 旧参数
        old_buffer = max(0.5 * scenario['atr'], 0.02)
        old_threshold = scenario['grid_lower'] + old_buffer
        old_result = scenario['price'] <= old_threshold
        
        # 新参数
        new_buffer = max(0.1 * scenario['atr'], 0.005)
        new_threshold = scenario['grid_lower'] + new_buffer
        new_result = scenario['price'] <= new_threshold
        
        improved = new_result and not old_result
        if improved:
            improvements += 1
        
        improvement_str = "✅" if improved else ""
        
        print(f"{scenario['price']:<8.3f} {scenario['grid_lower']:<8.3f} {scenario['atr']:<6.3f} "
              f"{old_buffer:<8.3f} {old_threshold:<8.3f} {str(old_result):<8} "
              f"{new_buffer:<8.3f} {new_threshold:<8.3f} {str(new_result):<8} {improvement_str:<6}")
    
    print(f"\n📊 测试结果: {improvements}/{len(scenarios)} 个场景得到改善")


def verify_fix_effectiveness():
    """验证修复的有效性"""
    print(f"\n🔧 验证修复有效性...")
    
    # 如果grid_lower接近90.60，而价格是90.60，那么应该触发near_lower
    # 我们需要找到一个临界点
    print("假设当前价格是90.600，我们来测试不同grid_lower值的效果:")
    
    price = 90.600
    atr = 0.10  # 假设ATR是0.10
    
    print(f"ATR = {atr}")
    print(f"\n{'grid_lower':<12} {'旧阈值':<10} {'旧结果':<8} {'新阈值':<10} {'新结果':<8} {'改善':<6}")
    print("-" * 60)
    
    improvements = 0
    for grid_lower in np.arange(90.50, 90.61, 0.01):
        # 旧参数
        old_buffer = max(0.5 * atr, 0.02)  # max(0.05, 0.02) = 0.05
        old_threshold = grid_lower + old_buffer
        old_result = price <= old_threshold
        
        # 新参数
        new_buffer = max(0.1 * atr, 0.005)  # max(0.01, 0.005) = 0.01
        new_threshold = grid_lower + new_buffer
        new_result = price <= new_threshold
        
        improved = new_result and not old_result
        if improved:
            improvements += 1
        
        improvement_str = "✅" if improved else ""
        
        print(f"{grid_lower:<12.3f} {old_threshold:<10.3f} {str(old_result):<8} {new_threshold:<10.3f} {str(new_result):<8} {improvement_str:<6}")
    
    print(f"\n📊 在grid_lower从90.500到90.600的范围内，共有 {improvements} 个点得到改善")
    

if __name__ == "__main__":
    print("🚀 开始精确调试测试...\n")
    
    result = find_actual_grid_lower()
    test_various_scenarios()
    verify_fix_effectiveness()
    
    print(f"\n✅ 测试完成!")
    
    if result:
        print(f"\n实际测试结果:")
        print(f"  当前价格: {result['current_price']:.3f}")
        print(f"  ATR值: {result['atr_value']:.3f}")
        print(f"  实际grid_lower: {result['grid_lower']:.3f}")
        print(f"  修复前near_lower: {result['old_result']}")
        print(f"  修复后near_lower: {result['new_result']}")
        
        if result['new_result'] and not result['old_result']:
            print(f"  🎯 修复成功! 从False变为True")
        elif result['new_result'] == result['old_result']:
            print(f"  📊 结果相同，可能需要其他调整")
        else:
            print(f"  ⚠️ 结果变化，但不是预期方向")
    else:
        print(f"❌ 无法完成实际测试")