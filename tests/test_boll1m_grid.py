import sys
import os
import pandas as pd
import pytest
from types import SimpleNamespace

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tigertrade import tiger1 as t1


def test_boll1m_grid_buys_at_lower(monkeypatch):
    monkeypatch.setattr(t1, 'get_kline_data', lambda *a, **k: pd.DataFrame({'time': pd.date_range(end=pd.Timestamp.utcnow(), periods=30, freq='min'), 'open': range(30), 'high': range(30), 'low': range(30), 'close': range(30), 'volume': range(30)}).set_index('time'))

    indicators = {
        '5m': {'boll_mid': 100.0, 'boll_upper': 110.0, 'boll_lower': 90.0, 'rsi': 60.0, 'atr': 1.0},
        '1m': {'rsi': 20.0, 'close': 89.0, 'volume': 100}
    }
    monkeypatch.setattr(t1, 'calculate_indicators', lambda a, b: indicators)

    calls = {}
    monkeypatch.setattr(t1, 'place_tiger_order', lambda *a, **k: calls.update({'called': True, 'side': a[0]}))
    monkeypatch.setattr(t1, 'check_risk_control', lambda price, side: True)

    t1.boll1m_grid_strategy()

    assert calls.get('called', False) is True
    assert calls['side'] == 'BUY'


def test_boll1m_grid_sells_at_mid(monkeypatch):
    monkeypatch.setattr(t1, 'get_kline_data', lambda *a, **k: pd.DataFrame({'time': pd.date_range(end=pd.Timestamp.utcnow(), periods=30, freq='min'), 'open': range(30), 'high': range(30), 'low': range(30), 'close': range(30), 'volume': range(30)}).set_index('time'))

    indicators = {
        '5m': {'boll_mid': 100.0, 'boll_upper': 110.0, 'boll_lower': 90.0, 'rsi': 60.0, 'atr': 1.0},
        '1m': {'rsi': 70.0, 'close': 101.0, 'volume': 100}
    }
    monkeypatch.setattr(t1, 'calculate_indicators', lambda a, b: indicators)

    # start with a position
    monkeypatch.setattr(t1, 'place_tiger_order', lambda *a, **k: (_ for _ in ()).throw(SystemExit('SELL_CALLED')))
    t1.current_position = 1

    with pytest.raises(SystemExit):
        t1.boll1m_grid_strategy()
