import sys
import os
import pandas as pd
import pytest
from types import SimpleNamespace

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src import tiger1 as t1


def test_boll1m_grid_buys_at_lower():
    # local simple monkeypatch replacement (pytest fixture not available in our runner)
    class MP:
        def __init__(self):
            self._changes = []
        def setattr(self, obj, name, value):
            existed = hasattr(obj, name)
            old = getattr(obj, name) if existed else None
            setattr(obj, name, value)
            self._changes.append((obj, name, old, existed))
        def undo(self):
            for obj, name, old, existed in reversed(self._changes):
                if existed:
                    setattr(obj, name, old)
                else:
                    delattr(obj, name)

    mp = MP()
    try:
        # recent closes: dip to 89 then rebound to 90 (lower = 90)
        # create 30 rows where the last 3 are the dip+rebound
        times = pd.date_range(end=pd.Timestamp.utcnow(), periods=30, freq='min')
        opens = [95]*27 + [95, 89, 90]
        highs = opens.copy()
        lows = opens.copy()
        closes = opens.copy()
        vols = [100]*30
        df = pd.DataFrame({'time': times, 'open': opens, 'high': highs, 'low': lows, 'close': closes, 'volume': vols}).set_index('time')
        mp.setattr(t1, 'get_kline_data', lambda *a, **k: df)

        indicators = {
            '5m': {'boll_mid': 95.0, 'boll_upper': 110.0, 'boll_lower': 90.0, 'rsi': 60.0, 'atr': 1.0},
            '1m': {'rsi': 20.0, 'close': 90.0, 'volume': 100}
        }
        mp.setattr(t1, 'calculate_indicators', lambda a, b: indicators)

        calls = {}
        mp.setattr(t1, 'place_tiger_order', lambda *a, **k: calls.update({'called': True, 'side': a[0]}))
        mp.setattr(t1, 'check_risk_control', lambda price, side: True)

        t1.boll1m_grid_strategy()

        assert calls.get('called', False) is True
        assert calls['side'] == 'BUY'
    finally:
        mp.undo()


def test_boll1m_grid_sells_at_mid():
    class MP:
        def __init__(self):
            self._changes = []
        def setattr(self, obj, name, value):
            existed = hasattr(obj, name)
            old = getattr(obj, name) if existed else None
            setattr(obj, name, value)
            self._changes.append((obj, name, old, existed))
        def undo(self):
            for obj, name, old, existed in reversed(self._changes):
                if existed:
                    setattr(obj, name, old)
                else:
                    delattr(obj, name)

    mp = MP()
    try:
        mp.setattr(t1, 'get_kline_data', lambda *a, **k: pd.DataFrame({'time': pd.date_range(end=pd.Timestamp.utcnow(), periods=30, freq='min'), 'open': range(30), 'high': range(30), 'low': range(30), 'close': range(30), 'volume': range(30)}).set_index('time'))

        indicators = {
            '5m': {'boll_mid': 100.0, 'boll_upper': 110.0, 'boll_lower': 90.0, 'rsi': 60.0, 'atr': 1.0},
            '1m': {'rsi': 70.0, 'close': 101.0, 'volume': 100}
        }
        mp.setattr(t1, 'calculate_indicators', lambda a, b: indicators)

        # start with a position
        mp.setattr(t1, 'place_tiger_order', lambda *a, **k: (_ for _ in ()).throw(SystemExit('SELL_CALLED')))
        t1.current_position = 1

        with pytest.raises(SystemExit):
            t1.boll1m_grid_strategy()
    finally:
        mp.undo()
