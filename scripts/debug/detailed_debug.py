#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
详细调试当前计算过程
"""

import sys
from pathlib import Path
_R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_R))

from src import tiger1 as t1
import pandas as pd
import numpy as np
import talib


def detailed_calculation_debug():
    """详细调试计算过程"""
    print("🔍 详细调试计算过程...")
    
    # 创建符合您数据的测试数据
    # 时间段从13:10到13:15，价格下降
    np.random.seed(42)
    
    # 5分钟数据 - 包含您提到的数据点
    base_prices_5m = []
    for i in range(45):  # 前45个点用于建立趋势
        base_price = 90.0 + 0.3 * np.sin(i/10) + 0.1 * np.random.randn()
        base_prices_5m.append(base_price)
    
    # 添加您提到的数据点
    base_prices_5m.extend([90.375, 90.305])  # 13:10: 90.375, 13:15: 90.305
    
    df_5m = pd.DataFrame({
        'time': pd.date_range('2026-01-16 10:00', periods=50, freq='5min'),  # 修正时间范围
        'open': base_prices_5m,
        'high': [p + 0.15 for p in base_prices_5m],
        'low': [p - 0.15 for p in base_prices_5m],
        'close': base_prices_5m,
        'volume': [200] * 50
    })
    df_5m.set_index('time', inplace=True)
    
    # 1分钟数据 - 确保长度匹配
    minute_prices = []
    for i in range(155):  # 与时间索引长度一致
        minute_price = 90.0 + 0.15 * np.sin(i/20) + 0.05 * np.random.randn()
        minute_prices.append(minute_price)
    
    # 最后几个点接近您提到的值
    minute_prices[-10:] = [90.370, 90.372, 90.368, 90.365, 90.360, 90.355, 90.340, 90.320, 90.310, 90.305]
    
    df_1m = pd.DataFrame({
        'time': pd.date_range('2026-01-16 12:00', periods=155, freq='1min'),  # 修正时间范围
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
        buffer = max(0.1 * (atr_value if atr_value else 0), 0.005)
        threshold = actual_grid_lower + buffer
        near_lower = current_price <= threshold
        
        print(f"\n📈 当前参数计算:")
        print(f"   buffer = max(0.1 * {atr_value}, 0.005) = {buffer}")
        print(f"   阈值 = {actual_grid_lower} + {buffer} = {threshold}")
        print(f"   near_lower = {current_price} <= {threshold} = {near_lower}")
        
        # 检查其他条件
        rsi_1m = indicators['1m']['rsi']
        rsi_5m = indicators['5m']['rsi']
        
        print(f"\n📊 其他指标:")
        print(f"   RSI 1m: {rsi_1m}")
        print(f"   RSI 5m: {rsi_5m}")
        
        # 计算RSI相关条件
        rsi_low_map = {
            'boll_divergence_down': 15,
            'osc_bear': 22,
            'osc_bull': 55,
            'bull_trend': 50,
            'osc_normal': 25
        }
        
        trend = t1.judge_market_trend(indicators)
        rsi_low = rsi_low_map.get(trend, 25)
        print(f"   市场趋势: {trend}")
        print(f"   RSI低阈值: {rsi_low}")
        
        # RSI条件检查
        oversold_ok = (rsi_1m is not None) and (rsi_1m <= (rsi_low + 5))
        print(f"   oversold_ok: {rsi_1m} <= {rsi_low + 5} = {oversold_ok}")
        
        # 趋势检查
        trend_check = (trend in ['osc_bull', 'bull_trend'] and rsi_5m > 45) or \
                      (trend in ['osc_bear', 'boll_divergence_down'] and rsi_5m < 55)
        print(f"   trend_check: {trend_check}")
        
        # 动量检查
        closes = df_1m['close'].dropna()
        last = float(closes.iloc[-1])
        prev = float(closes.iloc[-2]) if len(closes) >= 2 else None
        rebound = (prev is not None and last > prev)
        print(f"   rebound: {prev} < {last} = {rebound}")
        
        # RSI OK检查
        rsi_ok = oversold_ok
        print(f"   rsi_ok: {rsi_ok}")
        
        # 最终决策
        final_decision = near_lower and rsi_ok and (trend_check or rebound)
        print(f"\n🎯 最终决策: near_lower({near_lower}) AND rsi_ok({rsi_ok}) AND (trend_check({trend_check}) OR rebound({rebound})) = {final_decision}")
        
        # 恢复原始值
        t1.grid_lower = original_lower
        t1.grid_upper = original_upper
        
        return {
            'current_price': current_price,
            'atr_value': atr_value,
            'grid_lower': actual_grid_lower,
            'buffer': buffer,
            'threshold': threshold,
            'near_lower': near_lower,
            'rsi_1m': rsi_1m,
            'rsi_5m': rsi_5m,
            'rsi_ok': rsi_ok,
            'trend_check': trend_check,
            'rebound': rebound,
            'final_decision': final_decision
        }
        
    except Exception as e:
        print(f"❌ 计算出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_old_vs_new_parameters():
    """比较新旧参数"""
    print(f"\n🔧 比较新旧参数效果...")
    
    # 使用一些典型值进行比较
    test_scenarios = [
        {"price": 90.305, "grid_lower": 90.0, "atr": 0.2, "desc": "低价格接近下轨"},
        {"price": 90.305, "grid_lower": 89.5, "atr": 0.3, "desc": "中等波动"},
        {"price": 90.305, "grid_lower": 90.2, "atr": 0.1, "desc": "高价格接近下轨"},
    ]
    
    print(f"{'场景':<15} {'价格':<8} {'下轨':<8} {'ATR':<6} {'旧缓冲':<8} {'旧阈值':<8} {'旧结果':<8} {'新缓冲':<8} {'新阈值':<8} {'新结果':<8}")
    print("-" * 90)
    
    for scenario in test_scenarios:
        # 旧参数
        old_buffer = max(0.5 * scenario['atr'], 0.02)
        old_threshold = scenario['grid_lower'] + old_buffer
        old_result = scenario['price'] <= old_threshold
        
        # 新参数
        new_buffer = max(0.1 * scenario['atr'], 0.005)
        new_threshold = scenario['grid_lower'] + new_buffer
        new_result = scenario['price'] <= new_threshold
        
        print(f"{scenario['desc']:<15} {scenario['price']:<8.3f} {scenario['grid_lower']:<8.3f} {scenario['atr']:<6.3f} "
              f"{old_buffer:<8.3f} {old_threshold:<8.3f} {str(old_result):<8} "
              f"{new_buffer:<8.3f} {new_threshold:<8.3f} {str(new_result):<8}")


def check_real_data_scenario():
    """检查实际数据场景"""
    print(f"\n🔍 检查实际数据场景...")
    
    # 根据您提供的数据：价格从90.375下降到90.305
    # 这种下降趋势可能意味着价格更接近下轨了，但其他条件可能阻止了交易
    print(f"场景分析:")
    print(f"  - 13:10 价格: 90.375")
    print(f"  - 13:15 价格: 90.305")
    print(f"  - 价格下降了: 90.375 - 90.305 = 0.07")
    print(f"  - 价格变得更低，应该更接近下轨")
    print(f"  - 但near_lower=False可能是因为:")
    print(f"    1. BOLL下轨也随价格下降而移动")
    print(f"    2. RSI条件不满足")
    print(f"    3. 其他条件不满足")
    
    print(f"\n💡 建议改进:")
    print(f"  1. 可以考虑相对位置而非绝对位置判断是否接近下轨")
    print(f"  2. 检查RSI条件是否过于严格")
    print(f"  3. 考虑加入价格动量指标")


if __name__ == "__main__":
    print("🚀 开始详细调试当前计算过程...\n")
    
    result = detailed_calculation_debug()
    compare_old_vs_new_parameters()
    check_real_data_scenario()
    
    if result:
        print(f"\n✅ 调试完成!")
        print(f"   当前价格: {result['current_price']:.3f}")
        print(f"   网格下轨: {result['grid_lower']:.3f}")
        print(f"   near_lower: {result['near_lower']}")
        print(f"   rsi_ok: {result['rsi_ok']}")
        print(f"   trend_check: {result['trend_check']}")
        print(f"   rebound: {result['rebound']}")
        print(f"   最终决策: {result['final_decision']}")
    else:
        print(f"❌ 调试失败")