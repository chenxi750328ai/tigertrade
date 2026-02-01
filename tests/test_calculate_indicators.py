import sys
import pandas as pd
import numpy as np

sys.argv = ['pytest']

sys.path.insert(0, '/home/cx/tigertrade')

from src import tiger1 as t1


def make_df(rows, freq, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end=pd.Timestamp('2026-01-01'), periods=rows, freq=freq)
    return pd.DataFrame({
        'open': rng.normal(100, 1, rows),
        'high': rng.normal(101, 1, rows),
        'low': rng.normal(99, 1, rows),
        'close': rng.normal(100, 1, rows),
        'volume': rng.integers(50, 150, rows)
    }, index=idx)


def test_calculate_indicators_returns_expected_keys():
    df_1m = make_df(30, '1min')
    df_5m = make_df(40, '5min')

    indicators = t1.calculate_indicators(df_1m, df_5m)

    assert set(indicators.keys()) == {'1m', '5m'}
    for frame in ('1m', '5m'):
        for key in ('close', 'high', 'low', 'open', 'volume'):
            assert key in indicators[frame]
    assert 'rsi' in indicators['1m'] and 'rsi' in indicators['5m']
    assert 'boll_upper' in indicators['5m'] and 'boll_lower' in indicators['5m']
    assert 'atr' in indicators['5m']


def test_calculate_indicators_uses_fallbacks_when_insufficient_data():
    df_1m = make_df(5, '1min')
    df_5m = make_df(5, '5min')

    indicators = t1.calculate_indicators(df_1m, df_5m)

    # With fewer than 15 rows, RSI falls back to 50 for 1m and 5m
    assert indicators['1m']['rsi'] == 50
    assert indicators['5m']['rsi'] == 50
    # With fewer than 20 rows, bollinger defaults are used
    close_val = indicators['5m']['close']
    assert indicators['5m']['boll_upper'] == close_val * 1.02
    assert indicators['5m']['boll_lower'] == close_val * 0.98
    # ATR also falls back to 0 when not enough rows
    assert indicators['5m']['atr'] == 0


def test_judge_market_trend_handles_missing_middle():
    indicators = {'5m': {'close': 100.0, 'rsi': 60, 'boll_middle': 0}}
    assert t1.judge_market_trend(indicators) == 'osc_normal'


def test_judge_market_trend_classification():
    base = {'boll_middle': 100.0, 'close': 103.0, 'rsi': 70}
    assert t1.judge_market_trend({'5m': base}) == 'bull_trend'
    bear = {'boll_middle': 100.0, 'close': 96.0, 'rsi': 30}
    assert t1.judge_market_trend({'5m': bear}) == 'bear_trend'
    neutral = {'boll_middle': 100.0, 'close': 100.5, 'rsi': 44}
    assert t1.judge_market_trend({'5m': neutral}) == 'osc_bear'
def test_atr_prev_zero_handled():
    df5 = make_df(t1.GRID_BOLL_PERIOD, '5min', seed=42)
    df1 = make_df(5, '1min', seed=24)

    t1.last_boll_width = 0.2
    t1.is_boll_divergence = False

    indicators = t1.calculate_indicators(df1, df5)

    assert '5m' in indicators
    assert indicators['5m']['atr'] >= 0