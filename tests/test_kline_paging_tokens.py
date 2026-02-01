import pytest
import importlib
import types
import sys
from types import SimpleNamespace
import pandas as pd
from datetime import datetime, timedelta

# Ensure safe argv
sys.argv = ['pytest', 'd']

# prepare full tigeropen stubs so src.tiger1 can be imported
import types as _types
import sys as _sys
_sys.path.insert(0, '/home/cx/tigertrade')

def _inject_tigeropen_stubs():
    _consts = _types.SimpleNamespace(
        Language=None, Market=None,
        BarPeriod=_types.SimpleNamespace(ONE_MINUTE='ONE_MINUTE', THREE_MINUTES='THREE_MINUTES', FIVE_MINUTES='FIVE_MINUTES', TEN_MINUTES='TEN_MINUTES', FIFTEEN_MINUTES='FIFTEEN_MINUTES', HALF_HOUR='HALF_HOUR', FORTY_FIVE_MINUTES='FORTY_FIVE_MINUTES', ONE_HOUR='ONE_HOUR', TWO_HOURS='TWO_HOURS', THREE_HOURS='THREE_HOURS', FOUR_HOURS='FOUR_HOURS', SIX_HOURS='SIX_HOURS', DAY='DAY', WEEK='WEEK', MONTH='MONTH', YEAR='YEAR'),
        QuoteRight=None, Currency=_types.SimpleNamespace(USD='USD'),
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
    _sys.modules.setdefault('tigeropen.quote', _types.ModuleType('tigeropen.quote'))
    _sys.modules['tigeropen.quote.quote_client'] = _qmod
    _tmod = _types.ModuleType('tigeropen.trade.trade_client')
    setattr(_tmod, 'TradeClient', lambda cfg: _types.SimpleNamespace(place_order=lambda req: _types.SimpleNamespace(order_id='SIM')))
    _sys.modules.setdefault('tigeropen.trade', _types.ModuleType('tigeropen.trade'))
    _sys.modules['tigeropen.trade.trade_client'] = _tmod

try:
    tiger1 = importlib.reload(importlib.import_module('src.tiger1'))
    import_error = None
except Exception as e:
    _inject_tigeropen_stubs()
    try:
        tiger1 = importlib.reload(importlib.import_module('src.tiger1'))
        import_error = None
    except Exception as e2:
        tiger1 = None
        import_error = e2


def test_get_kline_data_next_page_token_column(monkeypatch):
    monkeypatch.setattr(tiger1.api_manager, 'is_mock_mode', False)
    now = pd.Timestamp.utcnow()
    times1 = pd.date_range(end=now, periods=3, freq='min')
    df1 = pd.DataFrame({'time': times1, 'open': [1,2,3], 'high':[1,2,3], 'low':[1,2,3], 'close':[1,2,3], 'volume':[10,10,10], 'next_page_token':['TOK', 'TOK', 'TOK']})
    times2 = pd.date_range(end=times1[0] - pd.Timedelta(minutes=1), periods=3, freq='min')
    df2 = pd.DataFrame({'time': times2, 'open': [4,5,6], 'high':[4,5,6], 'low':[4,5,6], 'close':[4,5,6], 'volume':[10,10,10]})

    calls = {'i': 0}
    def fake_by_page(*a, **k):
        if calls['i'] == 0:
            calls['i'] += 1
            return (df1, 'TOK')
        return (df2, None)

    if tiger1 is None:
        pytest.fail(f"can't import src.tiger1: {import_error}")
    monkeypatch.setattr(tiger1, 'quote_client', types.SimpleNamespace(get_future_bars_by_page=fake_by_page))

    df = tiger1.get_kline_data('SIL.COMEX.202603', '1min', count=4, start_time=datetime.utcnow() - timedelta(minutes=20), end_time=datetime.utcnow())
    assert not df.empty
    assert len(df) >= 3
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        assert str(df.index.tz) == 'Asia/Shanghai'


def test_get_kline_data_page_token_with_get_future_bars(monkeypatch):
    monkeypatch.setattr(tiger1.api_manager, 'is_mock_mode', False)
    now = pd.Timestamp.utcnow()
    times1 = pd.date_range(end=now, periods=3, freq='min')
    df1 = pd.DataFrame({'time': times1, 'open': [1,2,3], 'high':[1,2,3], 'low':[1,2,3], 'close':[1,2,3], 'volume':[10,10,10], 'next_page_token':['PTOK', 'PTOK', 'PTOK']})
    times2 = pd.date_range(end=times1[0] - pd.Timedelta(minutes=1), periods=3, freq='min')
    df2 = pd.DataFrame({'time': times2, 'open': [4,5,6], 'high':[4,5,6], 'low':[4,5,6], 'close':[4,5,6], 'volume':[10,10,10]})

    calls = {'i': 0}
    last_token_store = {}

    def fake_by_page(identifier, period, begin_time, end_time, total, page_size=None, time_interval=None, **kwargs):
        if calls['i'] == 0:
            calls['i'] += 1
            return (df1, None)
        return (df2, None)

    def fake_get_future_bars(symbols, period, a, b, count, page_token=None):
        last_token_store['token'] = page_token
        if page_token == 'PTOK':
            return df2
        return df1

    if tiger1 is None:
        pytest.fail(f"can't import src.tiger1: {import_error}")
    monkeypatch.setattr(tiger1, 'quote_client', types.SimpleNamespace(get_future_bars_by_page=fake_by_page, get_future_bars=fake_get_future_bars))

    df = tiger1.get_kline_data('SIL.COMEX.202603', '1min', count=4, start_time=datetime.utcnow() - timedelta(minutes=20), end_time=datetime.utcnow())
    assert not df.empty
    assert len(df) >= 3
    assert last_token_store.get('token') == 'PTOK'