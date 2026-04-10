#!/usr/bin/env python3
"""快速收益率分析：基于 test.csv 的 label 信号做基线回测，无需加载模型。"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

def main():
    data_path = _REPO_ROOT / "data" / "processed" / "test.csv"
    if not data_path.exists():
        print(f"❌ 数据不存在: {data_path}")
        return 1

    df = pd.read_csv(data_path)
    if 'price_current' not in df.columns:
        df['price_current'] = df.get('close', df.iloc[:, 2])  # fallback

    price = df['price_current'].values
    label = df['label'].values if 'label' in df.columns else np.zeros(len(df))
    # label: 0=hold, 1=buy, 2=sell

    capital = 100000.0
    position = 0.0
    equity_curve = [capital]
    trades = []

    for i in range(1, len(price)):
        p_prev, p_curr = price[i-1], price[i]
        ret = (p_curr - p_prev) / p_prev if p_prev else 0
        signal = int(label[i]) if i < len(label) else 0

        # 按信号交易：1=buy, 2=sell
        if signal == 1 and position <= 0:
            amt = capital * 0.3 / p_curr if p_curr else 0
            position = amt
            capital -= amt * p_curr
        elif signal == 2 and position > 0:
            capital += position * p_curr
            trades.append(position * (p_curr - p_prev))
            position = 0

        # 更新权益
        equity = capital + position * p_curr
        equity_curve.append(equity)

    # 平仓
    if position > 0:
        capital += position * price[-1]
        position = 0
    final_equity = capital

    total_return = (final_equity - 100000) / 100000
    eq = np.array(equity_curve)
    returns = np.diff(eq) / eq[:-1]
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24 * 60) if np.std(returns) > 0 else 0
    cummax = np.maximum.accumulate(eq)
    dd = (eq - cummax) / np.where(cummax > 0, cummax, 1)
    max_dd = np.min(dd)

    print("=" * 60)
    print("📊 快速收益率分析（基于 label 信号基线）")
    print("=" * 60)
    print(f"数据点数: {len(df)}")
    print(f"初始资金: $100,000")
    print(f"最终权益: ${final_equity:,.2f}")
    print(f"总收益率: {total_return*100:+.2f}%")
    print(f"Sharpe: {sharpe:.2f}")
    print(f"最大回撤: {max_dd*100:.2f}%")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
