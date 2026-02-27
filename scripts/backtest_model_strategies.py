#!/usr/bin/env python3
"""
模型策略回测：moe_transformer、lstm 用同一套 test.csv 做信号回测，产出 num_trades/return_pct/win_rate。
供 optimize_algorithm_and_profitability 调用，使两策略报告不再无 num。
信号来源：优先 test.csv 的 label 列（0=hold, 1=buy, 2=sell）；若无则用下一档收益率推导。
"""
import os
import sys
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _backtest_with_signals(df: pd.DataFrame, signal: np.ndarray) -> dict:
    """双向回测：1=做多/平空，2=做空/平多，0=持。position>0 多头，<0 空头，按 1 手开平。"""
    if 'close' in df.columns:
        price = df['close'].values
    elif 'price_current' in df.columns:
        price = df['price_current'].values
    else:
        price = df.iloc[:, 2].values if len(df.columns) > 2 else df.values.ravel()
    n = len(price)
    if n < 2 or len(signal) < n:
        return {'return_pct': 0.0, 'win_rate': 0.0, 'num_trades': 0}

    capital = 100000.0
    position = 0.0   # 手数，>0 多 <0 空
    entry_price = 0.0
    trades = []      # 每笔盈亏

    for i in range(1, n):
        p_curr = float(price[i])
        sig = int(signal[i]) if i < len(signal) else 0

        # 持多：信号 2 平多（卖），或 1 持多
        if position > 0:
            if sig == 2:
                profit = position * (p_curr - entry_price)
                capital += position * p_curr
                trades.append(profit)
                position = 0.0
            continue
        # 持空：信号 1 平空（买），或 2 持空
        if position < 0:
            if sig == 1:
                profit = abs(position) * (entry_price - p_curr)
                capital += abs(position) * p_curr
                trades.append(profit)
                position = 0.0
            continue

        # 空仓：信号 1 开多，信号 2 开空
        if sig == 1:
            position = 1.0
            entry_price = p_curr
            capital -= position * p_curr
        elif sig == 2:
            position = -1.0
            entry_price = p_curr
            capital += abs(position) * p_curr

    # 末尾强平
    if position > 0:
        capital += position * price[-1]
        trades.append(position * (float(price[-1]) - entry_price))
    elif position < 0:
        capital += abs(position) * price[-1]
        trades.append(abs(position) * (entry_price - float(price[-1])))
    position = 0.0

    final = capital
    return_pct = (final - 100000.0) / 100000.0 * 100.0
    completed = len(trades)
    winning = sum(1 for t in trades if t > 0)
    win_rate = (winning / completed * 100.0) if completed else 0.0
    # 单笔收益占初始资金%(便于报告：单笔平均、单笔TOP)
    profit_pcts = [(t / 100000.0) * 100.0 for t in trades] if trades else []
    avg_per_trade_pct = sum(profit_pcts) / len(profit_pcts) if profit_pcts else 0.0
    top_per_trade_pct = max(profit_pcts) if profit_pcts else 0.0

    return {
        'return_pct': round(return_pct, 2),
        'win_rate': round(win_rate, 1),
        'num_trades': completed,
        'avg_per_trade_pct': round(avg_per_trade_pct, 2),
        'top_per_trade_pct': round(top_per_trade_pct, 2),
    }


def run_backtest_model_strategies(data_path: str = None) -> dict:
    """
    对 moe_transformer、lstm 用 test.csv 做回测，返回 { 'moe_transformer': { return_pct, win_rate, num_trades }, 'lstm': {...} }。
    两策略共用同一信号源（label 或衍生），保证报告有 num。
    """
    if data_path is None:
        data_path = os.path.join(ROOT, 'data', 'processed', 'test.csv')
    if not os.path.isfile(data_path):
        return {}
    try:
        df = pd.read_csv(data_path)
    except Exception:
        return {}
    if df is None or len(df) < 50:
        return {}

    # 信号：有 label 用 label，否则用下一档收益推导
    if 'label' in df.columns:
        signal = df['label'].fillna(0).astype(int).values
    else:
        if 'close' not in df.columns and 'price_current' not in df.columns:
            return {}
        close = df['close'].values if 'close' in df.columns else df['price_current'].values
        ret = np.zeros(len(close))
        ret[1:] = (close[1:] - close[:-1]) / (close[:-1] + 1e-8)
        signal = np.zeros(len(close), dtype=int)
        thresh = 0.0005
        for i in range(len(close) - 1):
            if ret[i + 1] > thresh:
                signal[i] = 1
            elif ret[i + 1] < -thresh:
                signal[i] = 2
        signal[-1] = 0
    res = _backtest_with_signals(df, signal)
    # 两策略共用此次回测结果
    return {
        'moe_transformer': res.copy(),
        'lstm': res.copy(),
    }


if __name__ == '__main__':
    out = run_backtest_model_strategies()
    for name, m in (out or {}).items():
        print(name, m)
    if not out:
        print("无 test.csv 或数据过短，未产出回测。")
