#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证计算逻辑并提出修复方案
"""

import sys
sys.path.insert(0, '/home/cx/tigertrade')

from src import tiger1 as t1
import pandas as pd
import numpy as np
import talib

def verify_calculation_correctness():
    """验证计算逻辑的正确性"""
    print("🔍 验证计算逻辑的正确性...")
    
    print(f"\n📋 当前near_lower计算逻辑:")
    print(f"   buffer = max(0.5 * atr, 0.02)")
    print(f"   near_lower = price_current <= (grid_lower + buffer)")
    
    # 根据我们的调试，实际值是这样的：
    actual_grid_lower = 90.20  # 从调试中得出约90.20
    current_price = 90.60
    atr_value = 0.31  # 从调试中得出约0.31
    
    buffer = max(0.5 * atr_value, 0.02)
    threshold = actual_grid_lower + buffer
    near_lower = current_price <= threshold
    
    print(f"\n📊 实际计算过程:")
    print(f"   grid_lower: {actual_grid_lower}")
    print(f"   current_price: {current_price}")
    print(f"   atr_value: {atr_value}")
    print(f"   buffer = max(0.5 * {atr_value}, 0.02) = {buffer}")
    print(f"   threshold = {actual_grid_lower} + {buffer} = {threshold}")
    print(f"   near_lower = {current_price} <= {threshold} = {near_lower}")
    
    print(f"\n💡 逻辑分析:")
    print(f"   这个逻辑的目的是: 判断价格是否接近网格下轨")
    print(f"   使用ATR作为波动性调整: 波动大的时候，'接近'的定义要放宽")
    print(f"   buffer = max(0.5 * atr, 0.02) 确保了最小的容忍度0.02")
    
    print(f"\n🤔 是否存在问题?")
    print(f"   从算法角度看，这个逻辑是合理的：")
    print(f"   - 当ATR较高（市场波动大）时，需要更大的buffer")
    print(f"   - 当ATR较低（市场平稳）时，使用较小的buffer")
    print(f"   - 至少0.02的buffer确保了微小的价格差异不会触发信号")
    
    print(f"\n🔍 但可能的问题是参数设置:")
    print(f"   - buffer系数0.5可能过大，导致在高波动时期过于宽松")
    print(f"   - 最小buffer 0.02可能不适合所有市场")
    
    return {
        'grid_lower': actual_grid_lower,
        'current_price': current_price,
        'atr_value': atr_value,
        'buffer': buffer,
        'threshold': threshold,
        'near_lower': near_lower
    }


def propose_fix_options():
    """提出修复选项"""
    print(f"\n🔧 提出修复选项:")
    
    current_price = 90.60
    actual_grid_lower = 90.20
    atr_value = 0.31
    
    print(f"\n选项1: 调整buffer计算系数")
    print(f"   当前: buffer = max(0.5 * atr, 0.02)")
    for factor in [0.1, 0.2, 0.3, 0.4]:
        buffer = max(factor * atr_value, 0.02)
        threshold = actual_grid_lower + buffer
        result = current_price <= threshold
        print(f"   系数{factor}: buffer={buffer:.3f}, threshold={threshold:.3f}, near_lower={result}")
    
    print(f"\n选项2: 调整最小buffer值")
    buffer_factor = 0.5  # 当前值
    for min_buf in [0.005, 0.01, 0.015]:
        buffer = max(buffer_factor * atr_value, min_buf)
        threshold = actual_grid_lower + buffer
        result = current_price <= threshold
        print(f"   最小值{min_buf}: buffer={buffer:.3f}, threshold={threshold:.3f}, near_lower={result}")
    
    print(f"\n选项3: 使用相对百分比而非绝对数值")
    print(f"   这样可以根据价格水平自适应调整阈值")
    
    print(f"\n💡 建议:")
    print(f"   从测试结果看，使用系数0.1或0.2可能会更合理")
    print(f"   这样可以确保在价格真正接近下轨时触发信号")


def implement_improved_calculation():
    """实施改进的计算方法"""
    print(f"\n🛠️ 实施改进的计算方法:")
    
    # 保存原始函数以备恢复
    original_grid_trading_strategy_pro1 = t1.grid_trading_strategy_pro1
    
    # 创建改进版本
    def improved_grid_trading_strategy_pro1():
        """改进版的网格交易策略"""
        global current_position

        # Track whether we executed a sell in this iteration to prevent multiple sells in one cycle
        initial_position = current_position
        sold_this_iteration = False

        # Fetch market data
        df_1m = t1.get_kline_data([t1.FUTURE_SYMBOL], '1min', count=30)
        df_5m = t1.get_kline_data([t1.FUTURE_SYMBOL], '5min', count=50)
        if df_1m.empty or df_5m.empty:
            print("⚠️ 数据不足，跳过 improved_grid_trading_strategy_pro1")
            return

        indicators = t1.calculate_indicators(df_1m, df_5m)
        if not indicators or '5m' not in indicators or '1m' not in indicators:
            print("⚠️ 指标计算失败，跳过 improved_grid_trading_strategy_pro1")
            return

        trend = t1.judge_market_trend(indicators)
        t1.adjust_grid_interval(trend, indicators)

        price_current = indicators['1m']['close']
        rsi_1m = indicators['1m']['rsi']
        rsi_5m = indicators['5m']['rsi']
        atr = indicators['5m']['atr']

        rsi_low_map = {
            'boll_divergence_down': 15,
            'osc_bear': 22,
            'osc_bull': 55,
            'bull_trend': 50,
            'osc_normal': 25
        }
        rsi_low = rsi_low_map.get(trend, 25)

        # 改进：使用更小的buffer系数
        buffer = max(0.2 * (atr if atr else 0), 0.01)  # 从0.5降到0.2，最小值从0.02降到0.01
        near_lower = price_current <= (t1.grid_lower + buffer)

        # 2) RSI acceptance: oversold OR reversal OR bullish divergence
        oversold_ok = False
        rsi_rev_ok = False
        rsi_div_ok = False
        try:
            oversold_ok = (rsi_1m is not None) and (rsi_1m <= (rsi_low + 5))

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

            # bullish divergence: price makes lower low while RSI makes higher low
            try:
                lows = df_1m['low'].dropna()
                low_prev = float(lows.iloc[-2]) if len(lows) >= 2 else None
                low_cur = float(lows.iloc[-1]) if len(lows) >= 1 else None
                rsi_div_ok = (low_cur is not None and low_prev is not None and rsi_prev is not None and
                              (low_cur < low_prev) and (rsi_1m is not None) and (rsi_1m > rsi_prev) and (rsi_1m <= rsi_cap))
            except Exception:
                rsi_div_ok = False
        except Exception:
            oversold_ok = False
            rsi_rev_ok = False
            rsi_div_ok = False

        rsi_ok = oversold_ok or rsi_rev_ok or rsi_div_ok

        # 3) relaxed trend check
        trend_check = (trend in ['osc_bull', 'bull_trend'] and rsi_5m > 45) or \
                      (trend in ['osc_bear', 'boll_divergence_down'] and rsi_5m < 55)

        # 4) momentum / volume backups
        rebound = False
        vol_ok = False
        try:
            closes = df_1m['close'].dropna()
            last = float(closes.iloc[-1])
            prev = float(closes.iloc[-2]) if len(closes) >= 2 else None
            rebound = (prev is not None and last > prev)
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
        except Exception:
            rebound = False
            vol_ok = False

        # Debug logging
        print(f"🔧 improved_grid_trading_strategy_pro1: near_lower={near_lower}, rsi_ok={rsi_ok}, trend_check={trend_check}, rebound={rebound}, vol_ok={vol_ok}")
        print(f"    price={price_current:.3f}, grid_lower={t1.grid_lower:.3f}, buffer={buffer:.3f}, atr={atr}")

        # Final buy decision: near_lower + rsi_ok + (trend_check or rebound or vol_ok)
        if near_lower and rsi_ok and (trend_check or rebound or vol_ok) and t1.check_risk_control(price_current, 'BUY'):
            stop_loss_price, projected_loss = t1.compute_stop_loss(price_current, atr, t1.grid_lower)
            if stop_loss_price is None or not math.isfinite(projected_loss):
                print("⚠️ 止损计算异常，跳过买入(improved)")
                return
            # compute TP with buffer below grid_upper
            import math
            min_tick = 0.01
            try:
                min_tick = float(t1.FUTURE_TICK_SIZE)
            except Exception:
                pass
            tp_offset = max(t1.TAKE_PROFIT_ATR_OFFSET * (atr if atr else 0), t1.TAKE_PROFIT_MIN_OFFSET)
            take_profit_price = max(price_current + min_tick, (t1.grid_upper - tp_offset) if t1.grid_upper is not None else price_current + min_tick)
            print(
                f"🎯 improved_grid_trading_strategy_pro1: 触发买入条件 -> price={price_current:.4f}, "
                f"rsi_1m={rsi_1m}, rsi_5m={rsi_5m}, atr={atr}, buffer={buffer:.4f}, near_lower={near_lower}, "
                f"rsi_ok={rsi_ok}, trend_check={trend_check}, rebound={rebound}, vol_ok={vol_ok}, "
                f"grid_lower={t1.grid_lower}, grid_upper={t1.grid_upper}, stop_loss={stop_loss_price:.4f}, tp={take_profit_price:.4f}"
            )
            t1.place_tiger_order('BUY', 1, price_current, stop_loss_price)
            try:
                t1.place_take_profit_order('BUY', 1, take_profit_price)
            except Exception:
                pass
    
    # 应用改进版本
    t1.improved_grid_trading_strategy_pro1 = improved_grid_trading_strategy_pro1
    
    print(f"✅ 已创建改进版的策略函数")
    print(f"   原来的buffer计算: max(0.5 * atr, 0.02)")
    print(f"   改进的buffer计算: max(0.2 * atr, 0.01)")


if __name__ == "__main__":
    results = verify_calculation_correctness()
    propose_fix_options()
    implement_improved_calculation()
    
    print(f"\n📋 总结:")
    print(f"1. 原始计算逻辑本身没有错误，但参数可能过于保守")
    print(f"2. 当前参数在高波动市场中可能导致信号延迟")
    print(f"3. 提出的改进版本使用更敏感的参数")
    print(f"4. 用户可以根据市场特性进一步调整参数")