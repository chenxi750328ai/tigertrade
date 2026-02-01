import pytest
import importlib
import types
import sys
from types import SimpleNamespace
import pandas as pd
from datetime import datetime, timedelta, timezone

# Ensure safe argv
sys.argv = ['pytest', 'd']

# Provide lightweight fake `tigeropen` modules so tests can import `tigertrade.tiger1` without the real SDK.
import types as _types
import sys as _sys
_sys.path.insert(0, '/home/cx/tigertrade')

def _inject_stubs():
    _consts = _types.SimpleNamespace(
        Language=None,
        Market=None,
        BarPeriod=_types.SimpleNamespace(ONE_MINUTE='ONE_MINUTE', THREE_MINUTES='THREE_MINUTES', FIVE_MINUTES='FIVE_MINUTES', TEN_MINUTES='TEN_MINUTES', FIFTEEN_MINUTES='FIFTEEN_MINUTES', HALF_HOUR='HALF_HOUR', FORTY_FIVE_MINUTES='FORTY_FIVE_MINUTES', ONE_HOUR='ONE_HOUR', TWO_HOURS='TWO_HOURS', THREE_HOURS='THREE_HOURS', FOUR_HOURS='FOUR_HOURS', SIX_HOURS='SIX_HOURS', DAY='DAY', WEEK='WEEK', MONTH='MONTH', YEAR='YEAR'),
        QuoteRight=None,
        Currency=_types.SimpleNamespace(USD='USD'),
        OrderStatus=_types.SimpleNamespace(FILLED='FILLED'),
        OrderType=_types.SimpleNamespace(MARKET='MARKET', LIMIT='LIMIT', LMT='LMT')
    )
    _mod = _types.ModuleType('tigeropen.common.consts')
    for k, v in _consts.__dict__.items():
        setattr(_mod, k, v)
    _sys.modules.setdefault('tigeropen', _types.ModuleType('tigeropen'))
    _sys.modules.setdefault('tigeropen.common', _types.ModuleType('tigeropen.common'))
    _sys.modules['tigeropen.common.consts'] = _mod
    _util_mod = _types.ModuleType('tigeropen.common.util')
    _sig_mod = _types.ModuleType('tigeropen.common.util.signature_utils')
    setattr(_sig_mod, 'read_private_key', lambda path=None: 'FAKE_PRIVATE_KEY')
    _cu_mod = _types.ModuleType('tigeropen.common.util.contract_utils')
    setattr(_cu_mod, 'stock_contract', lambda *a, **k: None)
    setattr(_cu_mod, 'future_contract', lambda *a, **k: None)
    _sys.modules['tigeropen.common.util'] = _util_mod
    _sys.modules['tigeropen.common.util.signature_utils'] = _sig_mod
    _sys.modules['tigeropen.common.util.contract_utils'] = _cu_mod
    setattr(_util_mod, 'contract_utils', _cu_mod)
    _conf_mod = _types.ModuleType('tigeropen.tiger_open_config')
    setattr(_conf_mod, 'TigerOpenClientConfig', lambda props_path=None: _types.SimpleNamespace(account='SIM', tiger_id='SIM'))
    _sys.modules['tigeropen.tiger_open_config'] = _conf_mod
    _qmod = _types.ModuleType('tigeropen.quote.quote_client')
    setattr(_qmod, 'QuoteClient', lambda cfg: _types.SimpleNamespace(get_future_bars=lambda *a, **k: []))
    _sys.modules['tigeropen.quote.quote_client'] = _qmod
    _tmod = _types.ModuleType('tigeropen.trade.trade_client')
    setattr(_tmod, 'TradeClient', lambda cfg: _types.SimpleNamespace(place_order=lambda req: _types.SimpleNamespace(order_id='SIM')))
    _sys.modules['tigeropen.trade.trade_client'] = _tmod

try:
    tiger1 = importlib.reload(importlib.import_module('src.tiger1'))
    _tiger1_import_error = None
except Exception as e1:
    _inject_stubs()
    try:
        tiger1 = importlib.reload(importlib.import_module('src.tiger1'))
        _tiger1_import_error = None
    except Exception as e2:
        tiger1 = None
        _tiger1_import_error = e2

def _require_tiger1():
    if tiger1 is None:
        pytest.fail(f"can't import src.tiger1: {_tiger1_import_error}")


def make_df_close_rsi(close_vals, rsi_vals, times=None):
    if times is None:
        times = pd.date_range(end=pd.Timestamp.utcnow(), periods=len(close_vals), freq='min')
    return pd.DataFrame({'time': times, 'open': close_vals, 'high': close_vals, 'low': close_vals, 'close': close_vals, 'volume': [100]*len(close_vals)})


def test_grid_trading_places_buy_when_conditions_met(monkeypatch):
    _require_tiger1()
    # 确保持仓为 0，避免先触发止损/卖出路径
    monkeypatch.setattr(tiger1, 'current_position', 0)
    monkeypatch.setattr(tiger1, 'position_entry_prices', {})
    # Prepare dummy kline data so get_kline_data doesn't block
    monkeypatch.setattr(tiger1, 'get_kline_data', lambda *a, **k: pd.DataFrame({'time': pd.date_range(end=pd.Timestamp.utcnow(), periods=20, freq='min'), 'open': range(20), 'high': range(20), 'low': range(20), 'close': range(20), 'volume': range(20)}).set_index('time'))

    # Make calculate_indicators return controlled indicators that trigger a buy
    indicators = {
        '5m': {'boll_mid': 100.0, 'boll_upper': 110.0, 'boll_lower': 90.0, 'rsi': 60.0, 'atr': 1.0},
        '1m': {'rsi': 20.0, 'close': 90.0, 'volume': 100}
    }
    monkeypatch.setattr(tiger1, 'calculate_indicators', lambda a, b: indicators)

    # force a grid interval that puts current price below grid_lower and choose a bullish trend
    def fake_adjust(trend, indicators):
        tiger1.grid_lower = 95.0
        tiger1.grid_upper = 200.0
    monkeypatch.setattr(tiger1, 'adjust_grid_interval', fake_adjust)
    monkeypatch.setattr(tiger1, 'judge_market_trend', lambda indicators: 'bull_trend')

    # capture calls to place_tiger_order
    calls = {}
    def fake_place(side, quantity, price, stop_loss_price=None):
        calls['called'] = True
        calls['side'] = side
        calls['quantity'] = quantity
        calls['price'] = price
        calls['stop'] = stop_loss_price
        return True

    monkeypatch.setattr(tiger1, 'place_tiger_order', fake_place)

    # Ensure risk control allows this trade
    monkeypatch.setattr(tiger1, 'check_risk_control', lambda price, side: True)

    tiger1.grid_trading_strategy()

    assert calls.get('called', False) is True
    assert calls['side'] == 'BUY'


def test_place_tiger_order_production_guard(monkeypatch):
    _require_tiger1()
    # force production
    monkeypatch.setattr(tiger1, 'RUN_ENV', 'production')
    # Ensure env var is not set
    monkeypatch.setenv('ALLOW_REAL_TRADING', '0')

    # monkeypatch a trade_client that would raise if called
    monkeypatch.setattr(tiger1, 'trade_client', SimpleNamespace(place_order=lambda req: (_ for _ in ()).throw(Exception('should not call'))))

    # call place_tiger_order should return False and not call trade_client
    result = tiger1.place_tiger_order('BUY', 1, 100.0, 99.0)
    assert result is False


def test_grid_trading_respects_risk_control(monkeypatch):
    _require_tiger1()
    # Prepare data and indicators to trigger buy attempt
    monkeypatch.setattr(tiger1, 'get_kline_data', lambda *a, **k: pd.DataFrame({'time': pd.date_range(end=pd.Timestamp.utcnow(), periods=20, freq='min'), 'open': range(20), 'high': range(20), 'low': range(20), 'close': range(20), 'volume': range(20)}).set_index('time'))
    indicators = {
        '5m': {'boll_mid': 100.0, 'boll_upper': 110.0, 'boll_lower': 90.0, 'rsi': 60.0, 'atr': 1.0},
        '1m': {'rsi': 40.0, 'close': 90.0, 'volume': 100}
    }
    monkeypatch.setattr(tiger1, 'calculate_indicators', lambda a, b: indicators)

    monkeypatch.setattr(tiger1, 'check_risk_control', lambda price, side: False)

    called = {'v': False}
    monkeypatch.setattr(tiger1, 'place_tiger_order', lambda *a, **k: called.update({'v': True}))

    tiger1.grid_trading_strategy()
    assert called['v'] is False


def test_boll1m_grid_buys_at_lower(monkeypatch):
    _require_tiger1()
    # Prepare dummy kline data so get_kline_data doesn't block
    monkeypatch.setattr(tiger1, 'get_kline_data', lambda *a, **k: pd.DataFrame({'time': pd.date_range(end=pd.Timestamp.utcnow(), periods=30, freq='min'), 'open': range(30), 'high': range(30), 'low': range(30), 'close': range(30), 'volume': range(30)}).set_index('time'))

    indicators = {
        '5m': {'boll_mid': 100.0, 'boll_upper': 110.0, 'boll_lower': 90.0, 'rsi': 60.0, 'atr': 1.0},
        '1m': {'rsi': 20.0, 'close': 89.0, 'volume': 100}
    }
    monkeypatch.setattr(tiger1, 'calculate_indicators', lambda a, b: indicators)

    calls = {}
    monkeypatch.setattr(tiger1, 'place_tiger_order', lambda *a, **k: calls.update({'called': True, 'side': a[0]}))
    monkeypatch.setattr(tiger1, 'check_risk_control', lambda price, side: True)

    tiger1.boll1m_grid_strategy()

    assert calls.get('called', False) is True
    assert calls['side'] == 'BUY'


def test_boll1m_grid_sells_at_mid(monkeypatch):
    _require_tiger1()
    # Prepare dummy kline data so get_kline_data doesn't block
    monkeypatch.setattr(tiger1, 'get_kline_data', lambda *a, **k: pd.DataFrame({'time': pd.date_range(end=pd.Timestamp.utcnow(), periods=30, freq='min'), 'open': range(30), 'high': range(30), 'low': range(30), 'close': range(30), 'volume': range(30)}).set_index('time'))

    indicators = {
        '5m': {'boll_mid': 100.0, 'boll_upper': 110.0, 'boll_lower': 90.0, 'rsi': 60.0, 'atr': 1.0},
        '1m': {'rsi': 70.0, 'close': 101.0, 'volume': 100}
    }
    monkeypatch.setattr(tiger1, 'calculate_indicators', lambda a, b: indicators)

    # start with a position
    monkeypatch.setattr(tiger1, 'place_tiger_order', lambda *a, **k: (_ for _ in ()).throw(SystemExit('SELL_CALLED')))
    tiger1.current_position = 1

    with pytest.raises(SystemExit):
        tiger1.boll1m_grid_strategy()

