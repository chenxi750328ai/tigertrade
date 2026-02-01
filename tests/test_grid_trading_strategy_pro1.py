import sys
import os
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src import tiger1 as t1


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

        # get_kline_data 先 1min 再 5min
        mp.setattr(t1, 'get_kline_data', lambda *a, **k: df_1m)

        indicators = {
            '5m': {'boll_mid': 95.0, 'boll_upper': 110.0, 'boll_lower': 90.0, 'rsi': 60.0, 'atr': 2.0},
            '1m': {'rsi': 20.0, 'close': 90.5, 'volume': 200}
        }
        mp.setattr(t1, 'calculate_indicators', lambda a, b: indicators)
        mp.setattr(t1, 'judge_market_trend', lambda ind: 'osc_bear')
        mp.setattr(t1, 'adjust_grid_interval', lambda trend, ind: None)
        t1.grid_lower = 90.0
        t1.grid_upper = 95.0

        calls = {}
        mp.setattr(t1, 'place_tiger_order', lambda *a, **k: calls.update({'called': True, 'side': a[0], 'price': a[2]}))
        mp.setattr(t1, 'check_risk_control', lambda price, side: True)

        t1.current_position = 0

        t1.grid_trading_strategy_pro1()

        assert calls.get('called', False) is True
        assert calls['side'] == 'BUY'
        # price should be the indicator close
        assert abs(calls['price'] - 90.5) < 1e-6
    finally:
        mp.undo()


def test_grid_trading_strategy_pro1_triggers_on_volume_spike_even_without_rebound():
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
        times = pd.date_range(end=pd.Timestamp.utcnow(), periods=30, freq='min')
        opens = [95]*28 + [90, 90]  # last==prev -> no rebound
        closes = opens.copy()
        # vol_ok 使用 iloc[-6:-1] 的均值/最大值，需在倒数第2～6 根放量
        vols = [50]*24 + [80, 80, 80, 80, 400, 80]
        df_1m = pd.DataFrame({'time': times, 'open': opens, 'high': opens, 'low': opens, 'close': closes, 'volume': vols}).set_index('time')

        mp.setattr(t1, 'get_kline_data', lambda *a, **k: df_1m)
        mp.setattr(t1, 'judge_market_trend', lambda ind: 'osc_normal')
        mp.setattr(t1, 'adjust_grid_interval', lambda trend, ind: None)
        t1.grid_lower = 90.0
        t1.grid_upper = 95.0

        indicators = {
            '5m': {'boll_mid': 95.0, 'boll_upper': 110.0, 'boll_lower': 90.0, 'rsi': 55.0, 'atr': 2.0},
            '1m': {'rsi': 22.0, 'close': 90.2, 'volume': 400}
        }
        mp.setattr(t1, 'calculate_indicators', lambda a, b: indicators)

        calls = {}
        mp.setattr(t1, 'place_tiger_order', lambda *a, **k: calls.update({'called': True, 'side': a[0]}))
        mp.setattr(t1, 'check_risk_control', lambda price, side: True)

        t1.current_position = 0

        t1.grid_trading_strategy_pro1()

        assert calls.get('called', False) is True
        assert calls['side'] == 'BUY'
    finally:
        mp.undo()


def test_grid_trading_strategy_pro1_skips_when_not_near_lower():
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
        times = pd.date_range(end=pd.Timestamp.utcnow(), periods=30, freq='min')
        opens = [100]*30  # well above lower band
        closes = opens.copy()
        vols = [50]*30
        df_1m = pd.DataFrame({'time': times, 'open': opens, 'high': opens, 'low': opens, 'close': closes, 'volume': vols}).set_index('time')

        mp.setattr(t1, 'get_kline_data', lambda *a, **k: df_1m)
        mp.setattr(t1, 'judge_market_trend', lambda ind: 'osc_normal')
        mp.setattr(t1, 'adjust_grid_interval', lambda trend, ind: None)
        t1.grid_lower = 90.0
        t1.grid_upper = 95.0

        indicators = {
            '5m': {'boll_mid': 95.0, 'boll_upper': 110.0, 'boll_lower': 90.0, 'rsi': 55.0, 'atr': 2.0},
            '1m': {'rsi': 20.0, 'close': 101.0, 'volume': 50}
        }
        mp.setattr(t1, 'calculate_indicators', lambda a, b: indicators)

        calls = {}
        mp.setattr(t1, 'place_tiger_order', lambda *a, **k: calls.update({'called': True, 'side': a[0]}))
        mp.setattr(t1, 'check_risk_control', lambda price, side: True)

        t1.current_position = 0

        t1.grid_trading_strategy_pro1()

        assert calls.get('called', False) is False
    finally:
        mp.undo()


def test_grid_trading_strategy_pro1_triggers_on_volume_spike():
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
        times = pd.date_range(end=pd.Timestamp.utcnow(), periods=30, freq='min')
        opens = [95]*28 + [89, 90]
        closes = opens.copy()
        # heavy last-volume spike
        vols = [50]*29 + [1000]
        df_1m = pd.DataFrame({'time': times, 'open': opens, 'high': opens, 'low': opens, 'close': closes, 'volume': vols}).set_index('time')

        mp.setattr(t1, 'get_kline_data', lambda *a, **k: df_1m)
        mp.setattr(t1, 'judge_market_trend', lambda ind: 'osc_normal')
        mp.setattr(t1, 'adjust_grid_interval', lambda trend, ind: None)
        t1.grid_lower = 90.0
        t1.grid_upper = 95.0

        indicators = {
            '5m': {'boll_mid': 95.0, 'boll_upper': 110.0, 'boll_lower': 90.0, 'rsi': 60.0, 'atr': 2.0},
            '1m': {'rsi': 20.0, 'close': 90.5, 'volume': 1000}
        }
        mp.setattr(t1, 'calculate_indicators', lambda a, b: indicators)

        calls = {}
        mp.setattr(t1, 'place_tiger_order', lambda *a, **k: calls.update({'called': True, 'side': a[0]}))
        mp.setattr(t1, 'check_risk_control', lambda price, side: True)

        t1.current_position = 0
        t1.grid_trading_strategy_pro1()

        assert calls.get('called', False) is True
        assert calls['side'] == 'BUY'
    finally:
        mp.undo()


def test_grid_trading_strategy_pro1_does_not_trigger_on_high_rsi():
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
        times = pd.date_range(end=pd.Timestamp.utcnow(), periods=30, freq='min')
        opens = [95]*28 + [89, 90]
        closes = opens.copy()
        vols = [50]*30
        df_1m = pd.DataFrame({'time': times, 'open': opens, 'high': opens, 'low': opens, 'close': closes, 'volume': vols}).set_index('time')

        mp.setattr(t1, 'get_kline_data', lambda *a, **k: df_1m)
        mp.setattr(t1, 'judge_market_trend', lambda ind: 'osc_normal')
        mp.setattr(t1, 'adjust_grid_interval', lambda trend, ind: None)
        t1.grid_lower = 90.0
        t1.grid_upper = 95.0

        # set rsi_1m higher than rsi_low + 5 so rsi_ok False
        indicators = {
            '5m': {'boll_mid': 95.0, 'boll_upper': 110.0, 'boll_lower': 90.0, 'rsi': 60.0, 'atr': 2.0},
            '1m': {'rsi': 40.0, 'close': 90.5, 'volume': 50}
        }
        mp.setattr(t1, 'calculate_indicators', lambda a, b: indicators)

        calls = {}
        mp.setattr(t1, 'place_tiger_order', lambda *a, **k: calls.update({'called': True, 'side': a[0]}))
        mp.setattr(t1, 'check_risk_control', lambda price, side: True)

        t1.current_position = 0
        t1.grid_trading_strategy_pro1()

        assert calls.get('called', False) is False
    finally:
        mp.undo()


def test_grid_trading_strategy_pro1_blocked_by_risk_control():
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
        times = pd.date_range(end=pd.Timestamp.utcnow(), periods=30, freq='min')
        opens = [95]*28 + [89, 90]
        closes = opens.copy()
        vols = [50]*30
        df_1m = pd.DataFrame({'time': times, 'open': opens, 'high': opens, 'low': opens, 'close': closes, 'volume': vols}).set_index('time')

        mp.setattr(t1, 'get_kline_data', lambda *a, **k: df_1m)
        mp.setattr(t1, 'judge_market_trend', lambda ind: 'osc_normal')
        mp.setattr(t1, 'adjust_grid_interval', lambda trend, ind: None)
        t1.grid_lower = 90.0
        t1.grid_upper = 95.0

        indicators = {
            '5m': {'boll_mid': 95.0, 'boll_upper': 110.0, 'boll_lower': 90.0, 'rsi': 60.0, 'atr': 2.0},
            '1m': {'rsi': 20.0, 'close': 90.5, 'volume': 50}
        }
        mp.setattr(t1, 'calculate_indicators', lambda a, b: indicators)

        calls = {}
        mp.setattr(t1, 'place_tiger_order', lambda *a, **k: calls.update({'called': True, 'side': a[0]}))
        mp.setattr(t1, 'check_risk_control', lambda price, side: False)

        t1.current_position = 0
        t1.grid_trading_strategy_pro1()

        assert calls.get('called', False) is False
    finally:
        mp.undo()


def test_grid_trading_strategy_pro1_trend_allows_without_rebound():
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
        times = pd.date_range(end=pd.Timestamp.utcnow(), periods=30, freq='min')
        opens = [95]*30
        closes = opens.copy()
        vols = [50]*30
        df_1m = pd.DataFrame({'time': times, 'open': opens, 'high': opens, 'low': opens, 'close': closes, 'volume': vols}).set_index('time')

        mp.setattr(t1, 'get_kline_data', lambda *a, **k: df_1m)
        mp.setattr(t1, 'judge_market_trend', lambda ind: 'bull_trend')
        mp.setattr(t1, 'adjust_grid_interval', lambda trend, ind: None)
        # near_lower: price_current 70 <= grid_lower + buffer(0.6)，故 grid_lower >= 69.4
        t1.grid_lower = 69.5
        t1.grid_upper = 75.0

        # bull_trend: trend_check=True(rsi_5m>45), near_lower=True(70<=70.1), rsi_ok(24<=30)
        indicators = {
            '5m': {'boll_mid': 80.0, 'boll_upper': 110.0, 'boll_lower': 70.0, 'rsi': 60.0, 'atr': 2.0},
            '1m': {'rsi': 24.0, 'close': 70.0, 'volume': 50}
        }
        mp.setattr(t1, 'calculate_indicators', lambda a, b: indicators)

        calls = {}
        mp.setattr(t1, 'place_tiger_order', lambda *a, **k: calls.update({'called': True, 'side': a[0]}))
        mp.setattr(t1, 'check_risk_control', lambda price, side: True)

        t1.current_position = 0
        t1.grid_trading_strategy_pro1()

        assert calls.get('called', False) is True
        assert calls['side'] == 'BUY'
    finally:
        mp.undo()
