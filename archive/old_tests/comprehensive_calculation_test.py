#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
全面测试计算过程，验证所有参数
"""

import sys
import os
import pandas as pd
import numpy as np
import talib
import math
from datetime import datetime, timedelta

# 添加tigertrade目录到路径
sys.path.insert(0, '/home/cx/tigertrade')

from src import tiger1 as t1


def test_near_lower_calculation():
    """测试near_lower计算过程"""
    print("🔍 测试near_lower计算过程...")
    
    # 测试不同的ATR值
    atr_values = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    price_current = 90.60
    grid_lower = 90.20
    
    print(f"📊 固定参数: price_current={price_current}, grid_lower={grid_lower}")
    print(f"   旧参数: buffer = max(0.5 * atr, 0.02)")
    print(f"   新参数: buffer = max(0.2 * atr, 0.01)")
    print(f"\n📈 测试不同ATR值的影响:")
    print(f"{'ATR':<8} {'旧buffer':<10} {'旧阈值':<10} {'旧结果':<8} {'新buffer':<10} {'新阈值':<10} {'新结果':<8}")
    print("-" * 70)
    
    for atr in atr_values:
        # 旧参数计算
        old_buffer = max(0.5 * atr, 0.02)
        old_threshold = grid_lower + old_buffer
        old_result = price_current <= old_threshold
        
        # 新参数计算
        new_buffer = max(0.2 * atr, 0.01)
        new_threshold = grid_lower + new_buffer
        new_result = price_current <= new_threshold
        
        print(f"{atr:<8.3f} {old_buffer:<10.3f} {old_threshold:<10.3f} {str(old_result):<8} {new_buffer:<10.3f} {new_threshold:<10.3f} {str(new_result):<8}")


def test_rsi_calculation():
    """测试RSI计算过程"""
    print(f"\n🔍 测试RSI计算过程...")
    
    # 创建测试数据
    prices = np.array([90.0, 90.1, 89.9, 90.2, 89.8, 90.3, 89.7, 90.4, 89.6, 90.5])
    
    # 计算RSI
    rsi = talib.RSI(prices, timeperiod=14)
    
    print(f"📊 价格序列: {prices}")
    print(f"📈 RSI值: {rsi[-1]:.2f}")  # 只显示最后一个值
    
    # 测试不同的RSI阈值
    rsi_values = [20, 25, 30, 35, 40, 45, 50, 55, 60, 70]
    print(f"\n📈 不同RSI值对判断结果的影响:")
    print(f"{'RSI值':<8} {'oversold_ok':<12} {'阈值(25)':<10}")
    print("-" * 35)
    
    for rsi_val in rsi_values:
        oversold_ok = rsi_val <= (25 + 5)  # 默认rsi_low是25
        print(f"{rsi_val:<8.2f} {str(oversold_ok):<12} {'<=30':<10}")


def test_grid_adjustment():
    """测试网格调整过程"""
    print(f"\n🔍 测试网格调整过程...")
    
    # 模拟不同市场情况
    market_scenarios = [
        {"boll_lower": 90.0, "boll_upper": 91.0, "last_price": 90.5},
        {"boll_lower": 89.5, "boll_upper": 90.8, "last_price": 90.2},
        {"boll_lower": 90.2, "boll_upper": 91.5, "last_price": 90.8},
        {"boll_lower": None, "boll_upper": None, "last_price": 90.5},  # 没有BOLL指标的情况
    ]
    
    for i, scenario in enumerate(market_scenarios):
        print(f"\n📊 场景 {i+1}: BOLL下轨={scenario['boll_lower']}, 上轨={scenario['boll_upper']}, 最新价格={scenario['last_price']}")
        
        # 模拟indicators
        indicators = {
            '5m': {},
            '1m': {'close': scenario['last_price']}
        }
        
        if scenario['boll_lower'] is not None and scenario['boll_upper'] is not None:
            indicators['5m'] = {
                'boll_lower': scenario['boll_lower'],
                'boll_upper': scenario['boll_upper']
            }
        
        # 保存原来的值
        original_lower, original_upper = t1.grid_lower, t1.grid_upper
        
        # 调整网格
        t1.adjust_grid_interval("osc_normal", indicators)
        
        print(f"   调整后: grid_lower={t1.grid_lower:.3f}, grid_upper={t1.grid_upper:.3f}")
        
        # 恢复原始值
        t1.grid_lower, t1.grid_upper = original_lower, original_upper


def test_full_calculation_process():
    """测试完整计算过程"""
    print(f"\n🔍 测试完整计算过程...")
    
    # 构造测试数据
    base_prices = 90.0 + 0.5 * np.sin(np.linspace(0, 4*np.pi, 50)) + 0.1 * np.random.randn(50)
    
    df_5m = pd.DataFrame({
        'time': pd.date_range('2026-01-16 10:00', periods=50, freq='5min'),
        'open': base_prices,
        'high': base_prices + 0.2,
        'low': base_prices - 0.2,
        'close': base_prices,
        'volume': [200] * 50
    })
    df_5m.set_index('time', inplace=True)
    
    minute_base_prices = 90.0 + 0.1 * np.sin(np.linspace(0, 20*np.pi, 150)) + 0.05 * np.random.randn(150)
    df_1m = pd.DataFrame({
        'time': pd.date_range('2026-01-16 12:00', periods=150, freq='1min'),
        'open': minute_base_prices,
        'high': minute_base_prices + 0.1,
        'low': minute_base_prices - 0.1,
        'close': minute_base_prices,
        'volume': [50] * 150
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
        
        print(f"\n🔧 完整计算过程:")
        print(f"   当前价格: {current_price:.3f}")
        print(f"   ATR值: {atr_value:.3f}")
        
        # 执行adjust_grid_interval
        t1.adjust_grid_interval("osc_normal", indicators)
        calculated_grid_lower = t1.grid_lower
        calculated_grid_upper = t1.grid_upper
        
        print(f"   调整后的grid_lower: {calculated_grid_lower:.3f}")
        print(f"   调整后的grid_upper: {calculated_grid_upper:.3f}")
        
        # 使用新参数计算buffer
        new_buffer = max(0.2 * (atr_value if atr_value else 0), 0.01)
        new_threshold = calculated_grid_lower + new_buffer
        new_near_lower = current_price <= new_threshold
        print(f"   新参数buffer: max(0.2 * {atr_value:.3f}, 0.01) = {new_buffer:.3f}")
        print(f"   新参数阈值: {calculated_grid_lower:.3f} + {new_buffer:.3f} = {new_threshold:.3f}")
        print(f"   新参数near_lower: {current_price:.3f} <= {new_threshold:.3f} = {new_near_lower}")
        
        # 使用旧参数计算buffer
        old_buffer = max(0.5 * (atr_value if atr_value else 0), 0.02)
        old_threshold = calculated_grid_lower + old_buffer
        old_near_lower = current_price <= old_threshold
        print(f"   旧参数buffer: max(0.5 * {atr_value:.3f}, 0.02) = {old_buffer:.3f}")
        print(f"   旧参数阈值: {calculated_grid_lower:.3f} + {old_buffer:.3f} = {old_threshold:.3f}")
        print(f"   旧参数near_lower: {current_price:.3f} <= {old_threshold:.3f} = {old_near_lower}")
        
        print(f"\n💡 比较结果:")
        print(f"   新参数使near_lower从{old_near_lower}变为{new_near_lower}")
        
        # 计算改善百分比
        diff = new_threshold - old_threshold
        print(f"   阈值变化: {diff:+.3f} ({diff/old_threshold*100:+.2f}%)")
        
    except Exception as e:
        print(f"❌ 完整计算过程出错: {e}")
        import traceback
        traceback.print_exc()


def test_risk_control_calculation():
    """测试风险控制计算"""
    print(f"\n🔍 测试风险控制计算...")
    
    # 测试不同的价格和ATR组合
    test_cases = [
        {"price": 90.0, "atr": 0.1, "side": "BUY"},
        {"price": 91.0, "atr": 0.2, "side": "BUY"},
        {"price": 90.5, "atr": 0.05, "side": "BUY"},
        {"price": 89.5, "atr": 0.15, "side": "BUY"}
    ]
    
    for case in test_cases:
        print(f"\n📊 测试案例: price={case['price']}, atr={case['atr']}, side={case['side']}")
        
        # 计算潜在止损价格
        estimated_stop_loss_price = case['price'] - (case['atr'] * t1.STOP_LOSS_ATR_FACTOR)
        potential_loss_per_unit = case['price'] - estimated_stop_loss_price
        potential_total_loss = potential_loss_per_unit * t1.FUTURE_MULTIPLIER
        
        print(f"   预估止损价: {case['price']} - ({case['atr']} * {t1.STOP_LOSS_ATR_FACTOR}) = {estimated_stop_loss_price:.3f}")
        print(f"   潜在单位损失: {case['price']} - {estimated_stop_loss_price:.3f} = {potential_loss_per_unit:.3f}")
        print(f"   潜在总损失: {potential_loss_per_unit:.3f} * {t1.FUTURE_MULTIPLIER} = {potential_total_loss:.3f}")
        print(f"   最大单笔损失限制: {t1.MAX_SINGLE_LOSS}")
        print(f"   是否超过限制: {potential_total_loss > t1.MAX_SINGLE_LOSS}")


def test_take_profit_calculation():
    """测试止盈计算"""
    print(f"\n🔍 测试止盈计算...")
    
    # 测试不同的价格和ATR组合
    test_cases = [
        {"price": 90.0, "atr": 0.1, "grid_upper": 91.0},
        {"price": 91.0, "atr": 0.2, "grid_upper": 92.0},
        {"price": 90.5, "atr": 0.05, "grid_upper": 91.2}
    ]
    
    for case in test_cases:
        print(f"\n📊 测试案例: price={case['price']}, atr={case['atr']}, grid_upper={case['grid_upper']}")
        
        # 计算止盈价格
        min_tick = 0.01
        try:
            min_tick = float(t1.FUTURE_TICK_SIZE)
        except Exception:
            pass
        
        tp_offset = max(t1.TAKE_PROFIT_ATR_OFFSET * (case['atr'] if case['atr'] else 0), t1.TAKE_PROFIT_MIN_OFFSET)
        take_profit_price = max(case['price'] + min_tick, case['grid_upper'] - tp_offset if case['grid_upper'] is not None else case['price'] + min_tick)
        
        print(f"   最小刻度: {min_tick}")
        print(f"   止盈偏移: max({t1.TAKE_PROFIT_ATR_OFFSET} * {case['atr']}, {t1.TAKE_PROFIT_MIN_OFFSET}) = {tp_offset:.3f}")
        print(f"   止盈价格: max({case['price']} + {min_tick}, {case['grid_upper']} - {tp_offset}) = {take_profit_price:.3f}")


def run_all_tests():
    """运行所有测试"""
    print("🚀 开始运行所有计算过程测试...")
    
    test_near_lower_calculation()
    test_rsi_calculation()
    test_grid_adjustment()
    test_full_calculation_process()
    test_risk_control_calculation()
    test_take_profit_calculation()
    
    print(f"\n✅ 所有测试完成！")


def create_enhanced_logging_strategy():
    """创建增强日志输出的策略函数"""
    print(f"\n🔧 创建增强日志输出的策略函数...")
    
    def enhanced_grid_trading_strategy_pro1():
        """增强版网格交易策略，带有详细日志"""
        global current_position

        # 获取市场数据
        df_1m = t1.get_kline_data([t1.FUTURE_SYMBOL], '1min', count=30)
        df_5m = t1.get_kline_data([t1.FUTURE_SYMBOL], '5min', count=50)
        if df_1m.empty or df_5m.empty:
            print("⚠️ 数据不足，跳过 enhanced_grid_trading_strategy_pro1")
            return

        indicators = t1.calculate_indicators(df_1m, df_5m)
        if not indicators or '5m' not in indicators or '1m' not in indicators:
            print("⚠️ 指标计算失败，跳过 enhanced_grid_trading_strategy_pro1")
            return

        trend = t1.judge_market_trend(indicators)
        t1.adjust_grid_interval(trend, indicators)

        price_current = indicators['1m']['close']
        rsi_1m = indicators['1m']['rsi']
        rsi_5m = indicators['5m']['rsi']
        atr = indicators['5m']['atr']

        # 详细日志输出
        print(f"\n📋 增强版策略计算详情:")
        print(f"   当前价格: {price_current:.3f}")
        print(f"   1分钟RSI: {rsi_1m:.3f}")
        print(f"   5分钟RSI: {rsi_5m:.3f}")
        print(f"   ATR: {atr:.3f}")
        print(f"   市场趋势: {trend}")
        print(f"   调整后网格下轨: {t1.grid_lower:.3f}")
        print(f"   调整后网格上轨: {t1.grid_upper:.3f}")

        rsi_low_map = {
            'boll_divergence_down': 15,
            'osc_bear': 22,
            'osc_bull': 55,
            'bull_trend': 50,
            'osc_normal': 25
        }
        rsi_low = rsi_low_map.get(trend, 25)
        print(f"   RSI低阈值: {rsi_low} (基于趋势: {trend})")

        # 1) buffer above lower band (using improved parameters)
        buffer = max(0.2 * (atr if atr else 0), 0.01)
        near_lower = price_current <= (t1.grid_lower + buffer)
        print(f"   计算buffer: max(0.2 * {atr:.3f}, 0.01) = {buffer:.3f}")
        print(f"   网格下轨 + buffer: {t1.grid_lower:.3f} + {buffer:.3f} = {t1.grid_lower + buffer:.3f}")
        print(f"   near_lower: {price_current:.3f} <= {t1.grid_lower + buffer:.3f} = {near_lower}")

        # 2) RSI acceptance: oversold OR reversal OR bullish divergence
        oversold_ok = False
        rsi_rev_ok = False
        rsi_div_ok = False
        try:
            oversold_ok = (rsi_1m is not None) and (rsi_1m <= (rsi_low + 5))
            print(f"   oversold_ok: {rsi_1m:.3f} <= {rsi_low + 5} = {oversold_ok}")

            # build recent RSI series (prefer precomputed, else compute)
            try:
                rsis = df_1m['rsi']
            except Exception:
                rsis = talib.RSI(df_1m['close'], timeperiod=t1.GRID_RSI_PERIOD_1M)

            rsis = rsis.dropna() if hasattr(rsis, 'dropna') else rsis
            rsi_prev = float(rsis.iloc[-2]) if hasattr(rsis, 'iloc') and len(rsis) >= 2 else None
            rsi_cap = (rsi_low + 12)

            # reversal: RSI crosses above 50 from below
            if (rsi_prev is not None) and (rsi_1m is not None):
                rsi_rev_ok = (rsi_prev < 50) and (rsi_1m >= 50)
                print(f"   rsi_rev_ok: {rsi_prev:.3f} < 50 AND {rsi_1m:.3f} >= 50 = {rsi_rev_ok}")
            else:
                print(f"   rsi_rev_ok: 无法计算 (rsi_prev={rsi_prev}, rsi_1m={rsi_1m})")

            # bullish divergence: price makes lower low while RSI makes higher low
            try:
                lows = df_1m['low'].dropna()
                low_prev = float(lows.iloc[-2]) if len(lows) >= 2 else None
                low_cur = float(lows.iloc[-1]) if len(lows) >= 1 else None
                rsi_div_ok = (low_cur is not None and low_prev is not None and rsi_prev is not None and
                              (low_cur < low_prev) and (rsi_1m is not None) and (rsi_1m > rsi_prev) and (rsi_1m <= rsi_cap))
                print(f"   rsi_div_ok: 价格创新低({low_cur:.3f} < {low_prev:.3f}) AND RSI未创新低({rsi_1m:.3f} > {rsi_prev:.3f}) = {rsi_div_ok}")
            except Exception as e:
                rsi_div_ok = False
                print(f"   rsi_div_ok: 计算出错 - {e}")
        except Exception as e:
            oversold_ok = False
            rsi_rev_ok = False
            rsi_div_ok = False
            print(f"   RSI计算出错: {e}")

        rsi_ok = oversold_ok or rsi_rev_ok or rsi_div_ok
        print(f"   rsi_ok: {oversold_ok} OR {rsi_rev_ok} OR {rsi_div_ok} = {rsi_ok}")

        # 3) relaxed trend check
        trend_check = (trend in ['osc_bull', 'bull_trend'] and rsi_5m > 45) or \
                      (trend in ['osc_bear', 'boll_divergence_down'] and rsi_5m < 55)
        print(f"   trend_check: ({trend in ['osc_bull', 'bull_trend']} AND {rsi_5m:.3f} > 45) OR ({trend in ['osc_bear', 'boll_divergence_down']} AND {rsi_5m:.3f} < 55) = {trend_check}")

        # 4) momentum / volume backups
        rebound = False
        vol_ok = False
        try:
            closes = df_1m['close'].dropna()
            last = float(closes.iloc[-1])
            prev = float(closes.iloc[-2]) if len(closes) >= 2 else None
            rebound = (prev is not None and last > prev)
            print(f"   rebound: {prev:.3f} < {last:.3f} = {rebound}")

            vols = df_1m['volume'].dropna()
            if len(vols) >= 6:
                window = vols.iloc[-6:-1]
                recent_mean = window.mean()
                recent_median = window.median()
                rmax = window.max()
                mean_up = recent_mean * 1.05
                med_up = recent_median * 1.01
                max_up = rmax * 0.95
                threshold = max(mean_up, med_up, max_up)
                vol_ok = vols.iloc[-1] >= max(threshold, 0)
                print(f"   vol_ok: {vols.iloc[-1]} >= max({mean_up:.2f}, {med_up:.2f}, {max_up:.2f}) = {vol_ok}")
            else:
                print(f"   vol_ok: 成交量数据不足 (只有{len(vols)}个数据点)")
        except Exception as e:
            rebound = False
            vol_ok = False
            print(f"   动量/成交量计算出错: {e}")

        print(f"   最终条件: near_lower={near_lower} AND rsi_ok={rsi_ok} AND ({trend_check} OR {rebound} OR {vol_ok})")
        final_condition = near_lower and rsi_ok and (trend_check or rebound or vol_ok)
        print(f"   最终条件: {final_condition}")
        
        if final_condition:
            risk_ok = t1.check_risk_control(price_current, 'BUY')
            print(f"   风控检查: {risk_ok}")
            final_buy = final_condition and risk_ok
            print(f"   最终买入: {final_buy}")
        else:
            print(f"   无需风控检查 (前置条件不满足)")
    
    # 返回增强版函数
    return enhanced_grid_trading_strategy_pro1


if __name__ == "__main__":
    run_all_tests()
    
    # 创建并展示增强版策略
    enhanced_strategy = create_enhanced_logging_strategy()
    print(f"\n✅ 增强版策略函数已创建，可提供详细的计算日志")