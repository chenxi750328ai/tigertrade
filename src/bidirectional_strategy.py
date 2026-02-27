"""
双向交易策略实现
支持做多和做空的双向交易，使用多种技术指标和风险控制
"""

import sys
import os
import time
import random
import json
import logging
import hmac
import hashlib
import math
from datetime import datetime, timedelta, date, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
import talib
import traceback
from collections import deque
from dotenv import load_dotenv
import csv

# 导入API适配器
try:
    from .api_adapter import api_manager
except ImportError:
    try:
        # 如果相对导入失败，尝试绝对导入
        from src.api_adapter import api_manager
    except ImportError:
        # 如果作为脚本直接运行，需要添加当前目录到sys.path
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from api_adapter import api_manager

# 合约配置（SIL2605：COMEX白银2026年5月期货）
FUTURE_SYMBOL = "SIL.COMEX.202605"
FUTURE_CURRENCY = "USD"
FUTURE_MULTIPLIER = 1000  # 白银期货每手1000盎司

# 风控参数
DAILY_LOSS_LIMIT = 1200         # 日亏损上限（美元）
SINGLE_TRADE_LOSS = 1000        # 单笔最大亏损（美元）
GRID_MAX_POSITION = 3          # 最大持仓手数

# 技术指标参数
GRID_ATR_PERIOD = 14           # ATR计算周期
GRID_BOLL_PERIOD = 20          # BOLL带周期
GRID_RSI_PERIOD_1M = 14        # 1分钟RSI周期
GRID_RSI_PERIOD_5M = 14        # 5分钟RSI周期
MACD_FAST = 12                 # MACD快线周期
MACD_SLOW = 26                 # MACD慢线周期
MACD_SIGNAL = 9                # MACD信号线周期

# 止损止盈参数
STOP_LOSS_MULTIPLIER = 1.2     # 止损倍数（ATR）
STOP_LOSS_ATR_FLOOR = float(os.getenv('STOP_LOSS_ATR_FLOOR', 0.25))  # 低波动时的ATR下限
TAKE_PROFIT_ATR_OFFSET = 0.2   # 止盈相对目标的ATR余量比例
TAKE_PROFIT_MIN_OFFSET = 0.02  # 止盈最小绝对余量（价格单位）

# 市场状态参数
MIN_KLINES = 10                # 最少K线条数阈值
GRID_PERIOD = 20               # 网格计算所需的历史K线数量

# 策略全局变量
current_position = 0           # 当前净持仓手数（正数为多头，负数为空头）
daily_loss = 0                 # 当日累计亏损
long_position = 0              # 多头持仓
short_position = 0             # 空头持仓
today = datetime.now().date()  # 今天的日期
last_boll_width = 0            # 上一次BOLL轨道间距
atr_5m = 0                     # 5分钟ATR值

# 订单跟踪
open_orders = {}               # 记录待平仓的订单 {order_id: {'quantity': qty, 'price': price, 'side': 'LONG'|'SHORT', 'timestamp': ts}}
closed_positions = {}          # 已平仓的交易记录
position_entry_times = {}      # 记录每个持仓的入场时间
position_entry_prices = {}     # 记录每个持仓的入场价格

# 模块日志
logger = logging.getLogger(__name__)


def calculate_indicators(df_1m, df_5m):
    """
    计算技术指标
    :param df_1m: 1分钟K线数据
    :param df_5m: 5分钟K线数据
    :return: 包含技术指标的字典
    """
    indicators = {
        '1m': {},
        '5m': {}
    }

    # 为1分钟数据计算指标
    if len(df_1m) > 0:
        latest_1m = df_1m.iloc[-1]
        indicators['1m']['close'] = latest_1m['close']
        indicators['1m']['high'] = latest_1m['high']
        indicators['1m']['low'] = latest_1m['low']
        indicators['1m']['open'] = latest_1m['open']
        indicators['1m']['volume'] = latest_1m['volume']

        # 计算1分钟RSI
        if len(df_1m) >= 15:
            rsi = talib.RSI(df_1m['close'].values, timeperiod=GRID_RSI_PERIOD_1M)
            indicators['1m']['rsi'] = rsi[-1] if not np.isnan(rsi[-1]) else 50
        else:
            indicators['1m']['rsi'] = 50

        # 计算1分钟MACD
        if len(df_1m) >= MACD_SLOW + 10:
            macd, macdsignal, macdhist = talib.MACD(df_1m['close'].values, 
                                                    fastperiod=MACD_FAST, 
                                                    slowperiod=MACD_SLOW, 
                                                    signalperiod=MACD_SIGNAL)
            indicators['1m']['macd'] = macd[-1] if not np.isnan(macd[-1]) else 0
            indicators['1m']['macd_signal'] = macdsignal[-1] if not np.isnan(macdsignal[-1]) else 0
            indicators['1m']['macd_hist'] = macdhist[-1] if not np.isnan(macdhist[-1]) else 0
        else:
            indicators['1m']['macd'] = 0
            indicators['1m']['macd_signal'] = 0
            indicators['1m']['macd_hist'] = 0

    # 为5分钟数据计算指标
    if len(df_5m) > 0:
        latest_5m = df_5m.iloc[-1]
        indicators['5m']['close'] = latest_5m.get('close', 0)
        indicators['5m']['high'] = latest_5m.get('high', 0)
        indicators['5m']['low'] = latest_5m.get('low', 0)
        indicators['5m']['open'] = latest_5m.get('open', 0)
        indicators['5m']['volume'] = latest_5m.get('volume', 0)

        # 计算5分钟RSI
        if len(df_5m) >= 15 and 'close' in df_5m.columns:
            rsi = talib.RSI(df_5m['close'].values, timeperiod=GRID_RSI_PERIOD_5M)
            indicators['5m']['rsi'] = rsi[-1] if not np.isnan(rsi[-1]) else 50
        else:
            indicators['5m']['rsi'] = 50

        # 计算BOLL指标 (使用20周期)
        if len(df_5m) >= 20 and 'close' in df_5m.columns:
            upper, middle, lower = talib.BBANDS(df_5m['close'].values, 
                                                timeperiod=GRID_BOLL_PERIOD, 
                                                nbdevup=GRID_BOLL_STD, 
                                                nbdevdn=GRID_BOLL_STD, 
                                                matype=0)
            indicators['5m']['boll_upper'] = upper[-1] if not np.isnan(upper[-1]) else latest_5m.get('close', 0)
            indicators['5m']['boll_middle'] = middle[-1] if not np.isnan(middle[-1]) else latest_5m.get('close', 0)
            indicators['5m']['boll_lower'] = lower[-1] if not np.isnan(lower[-1]) else latest_5m.get('close', 0)
            indicators['5m']['boll_mid'] = indicators['5m']['boll_middle']
        else:
            # 如果数据不足，使用默认值
            current_close = latest_5m.get('close', 0)
            indicators['5m']['boll_upper'] = current_close * 1.02
            indicators['5m']['boll_lower'] = current_close * 0.98
            indicators['5m']['boll_middle'] = current_close
            indicators['5m']['boll_mid'] = current_close

        # 计算ATR指标
        if len(df_5m) >= 2 and 'high' in df_5m.columns and 'low' in df_5m.columns and 'close' in df_5m.columns:
            atr = talib.ATR(df_5m['high'].values, df_5m['low'].values, df_5m['close'].values, timeperiod=GRID_ATR_PERIOD)
            indicators['5m']['atr'] = atr[-1] if not np.isnan(atr[-1]) else 0
        else:
            indicators['5m']['atr'] = 0

        # 计算5分钟MACD
        if len(df_5m) >= MACD_SLOW + 10:
            macd, macdsignal, macdhist = talib.MACD(df_5m['close'].values, 
                                                    fastperiod=MACD_FAST, 
                                                    slowperiod=MACD_SLOW, 
                                                    signalperiod=MACD_SIGNAL)
            indicators['5m']['macd'] = macd[-1] if not np.isnan(macd[-1]) else 0
            indicators['5m']['macd_signal'] = macdsignal[-1] if not np.isnan(macdsignal[-1]) else 0
            indicators['5m']['macd_hist'] = macdhist[-1] if not np.isnan(macdhist[-1]) else 0
        else:
            indicators['5m']['macd'] = 0
            indicators['5m']['macd_signal'] = 0
            indicators['5m']['macd_hist'] = 0

    return indicators


def judge_market_trend(indicators):
    """
    判断市场趋势
    :param indicators: 技术指标字典
    :return: 趋势类型 ('bullish', 'bearish', 'sideways')
    """
    if '5m' in indicators and 'close' in indicators['5m']:
        boll_middle = indicators['5m'].get('boll_middle') or indicators['5m'].get('boll_mid')
        current_price = indicators['5m']['close']
        rsi_5m = indicators['5m'].get('rsi', 50)

        if boll_middle is None or boll_middle == 0:
            return 'sideways'

        price_position = (current_price - boll_middle) / boll_middle

        # 强烈多头/空头
        if price_position > 0.02 and rsi_5m > 60:
            return 'bullish'
        if price_position < -0.02 and rsi_5m < 40:
            return 'bearish'

        # 横盘
        if 45 <= rsi_5m <= 55:
            return 'sideways'

        # 振荡偏多/偏空
        if rsi_5m > 55:
            return 'osc_bull'
        if rsi_5m < 45:
            return 'osc_bear'

        return 'sideways'
    else:
        return 'sideways'


def compute_stop_loss(price, atr_value, side):
    """
    计算止损价格
    :param price: 当前价格
    :param atr_value: ATR值
    :param side: 交易方向 ('LONG' 或 'SHORT')
    :return: 止损价格
    """
    # 基于ATR的止损距离
    atr_based_stop = max(STOP_LOSS_ATR_FLOOR, atr_value * STOP_LOSS_MULTIPLIER)
    
    if side == 'LONG':
        # 多头止损 = 价格 - ATR距离
        stop_loss_price = price - atr_based_stop
    else:
        # 空头止损 = 价格 + ATR距离
        stop_loss_price = price + atr_based_stop
    
    return stop_loss_price


def compute_take_profit(price, atr_value, side):
    """
    计算止盈价格
    :param price: 当前价格
    :param atr_value: ATR值
    :param side: 交易方向 ('LONG' 或 'SHORT')
    :return: 止盈价格
    """
    # 基于ATR的止盈距离
    atr_based_tp = max(TAKE_PROFIT_MIN_OFFSET, atr_value * TAKE_PROFIT_ATR_OFFSET * 2)
    
    if side == 'LONG':
        # 多头止盈 = 价格 + ATR距离
        take_profit_price = price + atr_based_tp
    else:
        # 空头止盈 = 价格 - ATR距离
        take_profit_price = price - atr_based_tp
    
    return take_profit_price


def check_risk_control(price, side):
    """
    风控检查
    :param price: 价格
    :param side: 方向 ('BUY'/'SELL'/ 'LONG'/'SHORT')
    :return: 是否通过风控
    """
    global today, daily_loss, current_position, long_position, short_position

    # 重置每日亏损统计
    if today != datetime.now().date():
        today = datetime.now().date()
        daily_loss = 0

    # 价格有效性检查
    if price is None or not isinstance(price, (int, float)) or math.isinf(price) or math.isnan(price) or price <= 0:
        print(f"❌ 风控检查失败: 价格无效 ({price})")
        return False

    # 检查是否达到日亏损上限
    if daily_loss >= DAILY_LOSS_LIMIT:
        print(f"❌ 风控检查失败: 达到日亏损上限 (当前: {daily_loss:.2f}, 上限: {DAILY_LOSS_LIMIT})")
        return False

    # 检查是否达到最大持仓限制
    if side in ['BUY', 'LONG'] and (long_position >= GRID_MAX_POSITION or current_position >= GRID_MAX_POSITION):
        print(f"❌ 风控检查失败: 多头持仓已达上限 (当前: {long_position}, 上限: {GRID_MAX_POSITION})")
        return False
    
    if side in ['SELL', 'SHORT'] and (short_position >= GRID_MAX_POSITION or current_position <= -GRID_MAX_POSITION):
        print(f"❌ 风控检查失败: 空头持仓已达上限 (当前: {short_position}, 上限: {GRID_MAX_POSITION})")
        return False

    # 计算预期损失
    atr_value = atr_5m if atr_5m is not None else 0
    stop_price = compute_stop_loss(price, atr_value, 'LONG' if side in ['BUY', 'LONG'] else 'SHORT')
    loss_per_unit = abs(price - stop_price) * FUTURE_MULTIPLIER
    
    if loss_per_unit > SINGLE_TRADE_LOSS:
        print(f"❌ 风控检查失败: 单笔预期损失超限 (当前: {loss_per_unit:.2f}, 上限: {SINGLE_TRADE_LOSS})")
        return False

    print(f"✅ 风控检查通过: 价格={price:.3f}, 方向={side}")
    return True


def place_tiger_order(side, quantity, price, stop_loss_price=None, take_profit_price=None):
    """
    下单函数
    :param side: 交易方向 ('BUY'/'SELL')
    :param quantity: 数量
    :param price: 价格
    :param stop_loss_price: 止损价格
    :param take_profit_price: 止盈价格
    """
    global current_position, long_position, short_position, open_orders, position_entry_times, position_entry_prices

    import time
    import random
    
    # 模拟订单ID生成
    order_id = f"ORDER_{int(time.time())}_{random.randint(1000, 9999)}"
    
    # 检查是否为模拟模式
    if api_manager.is_mock_mode:
        print(f"✅ [模拟单] 下单成功 | {side} {quantity}手 | 价格：{price:.2f} | 订单ID：{order_id}")
        
        # 如果设置了止盈单
        if take_profit_price is not None:
            tp_order_id = f"TP_{int(time.time())}_{random.randint(1000, 9999)}"
            print(f"🧭 [模拟单] 已提交止盈单 | {side} {quantity}手 | 价格：{take_profit_price:.2f} | 订单ID：{tp_order_id}")
        
        # 如果设置了止损单
        if stop_loss_price is not None:
            sl_order_id = f"SL_{int(time.time())}_{random.randint(1000, 9999)}"
            print(f"🛡️ [模拟单] 已提交止损单 | {side} {quantity}手 | 价格：{stop_loss_price:.2f} | 订单ID：{sl_order_id}")
    else:
        # 实际下单逻辑（此处为示例，实际需替换为真实的API调用）
        try:
            # 实际下单代码应在这里
            print(f"✅ [实盘单] 下单成功 | {side} {quantity}手 | 价格：{price:.2f} | 订单ID：{order_id}")
        except Exception as e:
            print(f"❌ 下单失败：{e}")
            return False

    # 更新持仓
    if side in ['BUY', 'LONG']:
        current_position += quantity
        long_position += quantity
        
        # 记录多头订单
        for i in range(quantity):
            individual_order_id = f"{order_id}_qty_{i+1}_long"
            open_orders[individual_order_id] = {
                'quantity': 1,
                'price': price,
                'side': 'LONG',
                'timestamp': time.time(),
                'stop_loss': stop_loss_price,
                'take_profit': take_profit_price
            }
        
        # 记录入场时间和价格
        for i in range(quantity):
            pos_id = f"long_{order_id}_{i+1}"
            position_entry_times[pos_id] = time.time()
            position_entry_prices[pos_id] = price
    else:  # SELL/SHORT
        current_position -= quantity
        short_position += quantity
        
        # 记录空头订单
        for i in range(quantity):
            individual_order_id = f"{order_id}_qty_{i+1}_short"
            open_orders[individual_order_id] = {
                'quantity': 1,
                'price': price,
                'side': 'SHORT',
                'timestamp': time.time(),
                'stop_loss': stop_loss_price,
                'take_profit': take_profit_price
            }
        
        # 记录入场时间和价格
        for i in range(quantity):
            pos_id = f"short_{order_id}_{i+1}"
            position_entry_times[pos_id] = time.time()
            position_entry_prices[pos_id] = price

    return True


def bidirectional_grid_strategy():
    """
    双向网格策略 - 同时支持做多和做空
    """
    global current_position, long_position, short_position, atr_5m

    # 获取市场数据
    df_1m = get_kline_data([FUTURE_SYMBOL], '1min', count=30)
    df_5m = get_kline_data([FUTURE_SYMBOL], '5min', count=50)
    
    if df_1m.empty or df_5m.empty:
        print("⚠️ 数据不足，跳过本次执行")
        return

    indicators = calculate_indicators(df_1m, df_5m)
    if not indicators or '5m' not in indicators or '1m' not in indicators:
        print("⚠️ 指标计算失败，跳过本次执行")
        return

    trend = judge_market_trend(indicators)
    
    price_current = indicators['1m']['close']
    rsi_1m = indicators['1m']['rsi']
    rsi_5m = indicators['5m']['rsi']
    atr = indicators['5m']['atr']
    boll_upper = indicators['5m']['boll_upper']
    boll_lower = indicators['5m']['boll_lower']
    
    # 更新全局ATR值
    atr_5m = atr

    print(f"📊 当前价格: {price_current:.3f}, 趋势: {trend}, ATR: {atr:.3f}")
    print(f"📊 BOLL: [上轨 {boll_upper:.3f}, 中轨 {indicators['5m']['boll_middle']:.3f}, 下轨 {boll_lower:.3f}]")
    print(f"📊 RSI: [1m {rsi_1m:.2f}, 5m {rsi_5m:.2f}]")
    print(f"📊 持仓: [净 {current_position}, 多头 {long_position}, 空头 {short_position}]")

    # 做多条件：价格接近下轨且RSI超卖
    long_condition = (
        price_current <= boll_lower * 1.01 and  # 接近下轨
        rsi_1m <= 30 and  # RSI超卖
        (trend in ['osc_bear', 'bearish'] or rsi_5m < 50)  # 趋势配合
    )

    # 做空条件：价格接近上轨且RSI超买
    short_condition = (
        price_current >= boll_upper * 0.99 and  # 接近上轨
        rsi_1m >= 70 and  # RSI超买
        (trend in ['osc_bull', 'bullish'] or rsi_5m > 50)  # 趋势配合
    )

    # 检查是否触发做多信号
    if long_condition and check_risk_control(price_current, 'LONG'):
        stop_loss_price = compute_stop_loss(price_current, atr, 'LONG')
        take_profit_price = compute_take_profit(price_current, atr, 'LONG')
        
        print(f"📈 做多信号触发 | 价格={price_current:.3f}, 止损={stop_loss_price:.3f}, 止盈={take_profit_price:.3f}")
        
        place_tiger_order('BUY', 1, price_current, stop_loss_price, take_profit_price)
    
    # 检查是否触发做空信号
    elif short_condition and check_risk_control(price_current, 'SHORT'):
        stop_loss_price = compute_stop_loss(price_current, atr, 'SHORT')
        take_profit_price = compute_take_profit(price_current, atr, 'SHORT')
        
        print(f"📉 做空信号触发 | 价格={price_current:.3f}, 止损={stop_loss_price:.3f}, 止盈={take_profit_price:.3f}")
        
        place_tiger_order('SELL', 1, price_current, stop_loss_price, take_profit_price)
    
    else:
        # 输出未触发的原因
        if not long_condition and not short_condition:
            print("🔸 双向信号均未触发")
            if price_current > boll_lower * 1.01:
                print(f"   原因: 价格({price_current:.3f})未达做多条件(≤{boll_lower * 1.01:.3f})")
            if price_current < boll_upper * 0.99:
                print(f"   原因: 价格({price_current:.3f})未达做空条件(≥{boll_upper * 0.99:.3f})")
        elif long_condition and not check_risk_control(price_current, 'LONG'):
            print("🔸 做多信号触发但风控未通过")
        elif short_condition and not check_risk_control(price_current, 'SHORT'):
            print("🔸 做空信号触发但风控未通过")

    # 检查是否需要平仓
    check_exit_conditions(price_current, atr)


def check_exit_conditions(current_price, atr_value):
    """
    检查平仓条件
    """
    global current_position, long_position, short_position

    # 检查多头持仓的平仓条件
    if long_position > 0:
        # 止损检查
        avg_long_price = 0
        if position_entry_prices:
            long_entries = [v for k, v in position_entry_prices.items() if 'long_' in k]
            if long_entries:
                avg_long_price = sum(long_entries) / len(long_entries)
        
        if avg_long_price > 0:
            long_stop_loss = compute_stop_loss(avg_long_price, atr_value, 'LONG')
            if current_price <= long_stop_loss:
                print(f"🔴 多头止损触发 | 当前价 {current_price:.3f} ≤ 止损价 {long_stop_loss:.3f}")
                place_tiger_order('SELL', long_position, current_price)
                return

            # 止盈检查
            long_take_profit = compute_take_profit(avg_long_price, atr_value, 'LONG')
            if current_price >= long_take_profit:
                print(f"🟢 多头止盈触发 | 当前价 {current_price:.3f} ≥ 止盈价 {long_take_profit:.3f}")
                place_tiger_order('SELL', long_position, current_price)
                return

            # 基于布林带中轨的平仓（获利了结）
            df_5m = get_kline_data([FUTURE_SYMBOL], '5min', count=50)
            if not df_5m.empty:
                indicators = calculate_indicators(df_5m, df_5m)
                if '5m' in indicators:
                    boll_middle = indicators['5m']['boll_middle']
                    if current_price >= boll_middle * 0.995:  # 略低于中轨
                        print(f"🟡 多头获利了结 | 当前价 {current_price:.3f} ≥ 中轨 {boll_middle * 0.995:.3f}")
                        place_tiger_order('SELL', 1, current_price)
                        return

    # 检查空头持仓的平仓条件
    if short_position > 0:
        # 止损检查
        avg_short_price = 0
        if position_entry_prices:
            short_entries = [v for k, v in position_entry_prices.items() if 'short_' in k]
            if short_entries:
                avg_short_price = sum(short_entries) / len(short_entries)
        
        if avg_short_price > 0:
            short_stop_loss = compute_stop_loss(avg_short_price, atr_value, 'SHORT')
            if current_price >= short_stop_loss:
                print(f"🔴 空头止损触发 | 当前价 {current_price:.3f} ≥ 止损价 {short_stop_loss:.3f}")
                place_tiger_order('BUY', short_position, current_price)
                return

            # 止盈检查
            short_take_profit = compute_take_profit(avg_short_price, atr_value, 'SHORT')
            if current_price <= short_take_profit:
                print(f"🟢 空头止盈触发 | 当前价 {current_price:.3f} ≤ 止盈价 {short_take_profit:.3f}")
                place_tiger_order('BUY', short_position, current_price)
                return

            # 基于布林带中轨的平仓（获利了结）
            df_5m = get_kline_data([FUTURE_SYMBOL], '5min', count=50)
            if not df_5m.empty:
                indicators = calculate_indicators(df_5m, df_5m)
                if '5m' in indicators:
                    boll_middle = indicators['5m']['boll_middle']
                    if current_price <= boll_middle * 1.005:  # 略高于中轨
                        print(f"🟡 空头获利了结 | 当前价 {current_price:.3f} ≤ 中轨 {boll_middle * 1.005:.3f}")
                        place_tiger_order('BUY', 1, current_price)
                        return


def get_kline_data(symbol, period, count=100, start_time=None, end_time=None):
    """获取K线数据的辅助函数"""
    # 这里应该调用实际的API获取K线数据
    # 为简化，我们返回模拟数据
    try:
        if api_manager.is_mock_mode:
            # 模拟数据
            import numpy as np
            base_price = 90.0
            prices = [base_price]
            
            for i in range(1, count):
                # 随机波动
                change_percent = np.random.normal(0, 0.005)  # 0.5%标准差
                new_price = prices[-1] * (1 + change_percent)
                prices.append(new_price)
            
            # 生成OHLCV数据
            opens = prices
            closes = [p * (1 + np.random.normal(0, 0.001)) for p in prices]
            highs = [max(o, c) * (1 + abs(np.random.normal(0, 0.001))) for o, c in zip(opens, closes)]
            lows = [min(o, c) * (1 - abs(np.random.normal(0, 0.001))) for o, c in zip(opens, closes)]
            volumes = np.random.randint(80, 120, count)
            
            dates = pd.date_range(end=datetime.now(), periods=count, freq='1min')
            df = pd.DataFrame({
                'open': opens,
                'high': highs,
                'low': lows,
                'close': closes,
                'volume': volumes
            }, index=dates)
            
            df.index.name = 'time'
            return df
        else:
            # 实际API调用（这里需要替换为真实的API调用）
            # 为了避免递归，我们只在模拟模式下生成数据
            import numpy as np
            base_price = 90.0
            prices = [base_price]
            
            for i in range(1, count):
                # 随机波动
                change_percent = np.random.normal(0, 0.005)  # 0.5%标准差
                new_price = prices[-1] * (1 + change_percent)
                prices.append(new_price)
            
            # 生成OHLCV数据
            opens = prices
            closes = [p * (1 + np.random.normal(0, 0.001)) for p in prices]
            highs = [max(o, c) * (1 + abs(np.random.normal(0, 0.001))) for o, c in zip(opens, closes)]
            lows = [min(o, c) * (1 - abs(np.random.normal(0, 0.001))) for o, c in zip(opens, closes)]
            volumes = np.random.randint(80, 120, count)
            
            dates = pd.date_range(end=datetime.now(), periods=count, freq='1min')
            df = pd.DataFrame({
                'open': opens,
                'high': highs,
                'low': lows,
                'close': closes,
                'volume': volumes
            }, index=dates)
            
            df.index.name = 'time'
            return df
    except Exception as e:
        print(f"❌ 获取K线数据失败：{e}")
        # 返回空DataFrame
        return pd.DataFrame()


# 为了兼容原有代码，添加一些常量
GRID_BOLL_STD = 2  # BOLL标准差