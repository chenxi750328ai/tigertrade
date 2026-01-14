import sys
import os
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tigertrade import tiger1 as t1


def test_grid_trading_strategy_pro1_triggers_on_near_lower_with_rebound():
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
        # Prepare 30 1-min bars with last 2 showing a rebound (prev 89 -> last 90.5)
        times = pd.date_range(end=pd.Timestamp.utcnow(), periods=30, freq='min')
        opens = [95]*28 + [89, 90]
        closes = opens.copy()
        vols = [50]*26 + [200, 250, 200, 250]
        df_1m = pd.DataFrame({'time': times, 'open': opens, 'high': opens, 'low': opens, 'close': closes, 'volume': vols}).set_index('time')

        mp.setattr(t1, 'get_kline_data', lambda *a, **k: df_1m if a[1] == '1min' else df_1m)

        # indicators: boll_lower=90, atr=2, set 1m close to 90.5 (within buffer 0.6)
        indicators = {
            '5m': {'boll_mid': 95.0, 'boll_upper': 110.0, 'boll_lower': 90.0, 'rsi': 60.0, 'atr': 2.0},
            '1m': {'rsi': 20.0, 'close': 90.5, 'volume': 200}
        }
        mp.setattr(t1, 'calculate_indicators', lambda a, b: indicators)

        calls = {}
        mp.setattr(t1, 'place_tiger_order', lambda *a, **k: calls.update({'called': True, 'side': a[0], 'price': a[2]}))
        mp.setattr(t1, 'check_risk_control', lambda price, side: True)

        # Ensure starting state
        t1.current_position = 0

        t1.grid_trading_strategy_pro1()

        assert calls.get('called', False) is True
        assert calls['side'] == 'BUY'
        # price should be the indicator close
        assert abs(calls['price'] - 90.5) < 1e-6
    finally:
        mp.undo()
