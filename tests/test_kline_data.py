import pytest
import sys
from types import SimpleNamespace
import pandas as pd
from datetime import datetime, timedelta, timezone

# Ensure module reads a safe argv value when imported
sys.argv = ['pytest', 'd']

# Provide lightweight fake `tigeropen` modules so tests can import `tigertrade.tiger1` without the real SDK.
import types
import sys as _sys
# ensure project root is on sys.path so `importlib.import_module('tigertrade.tiger1')` works
_sys.path.insert(0, '/home/cx/tigertrade')

def _inject_tigeropen_stubs():
    """仅在 import 失败时注入，避免覆盖已加载的真实 tigeropen。"""
    _consts = types.SimpleNamespace(
        Language=None,
        Market=None,
        BarPeriod=types.SimpleNamespace(ONE_MINUTE='ONE_MINUTE', THREE_MINUTES='THREE_MINUTES', FIVE_MINUTES='FIVE_MINUTES', TEN_MINUTES='TEN_MINUTES', FIFTEEN_MINUTES='FIFTEEN_MINUTES', HALF_HOUR='HALF_HOUR', FORTY_FIVE_MINUTES='FORTY_FIVE_MINUTES', ONE_HOUR='ONE_HOUR', TWO_HOURS='TWO_HOURS', THREE_HOURS='THREE_HOURS', FOUR_HOURS='FOUR_HOURS', SIX_HOURS='SIX_HOURS', DAY='DAY', WEEK='WEEK', MONTH='MONTH', YEAR='YEAR'),
        QuoteRight=None,
        Currency=types.SimpleNamespace(USD='USD'),
        OrderStatus=types.SimpleNamespace(FILLED='FILLED'),
        OrderType=types.SimpleNamespace(MARKET='MARKET', LIMIT='LIMIT', LMT='LMT')
    )
    _mod = types.ModuleType('tigeropen.common.consts')
    for k, v in _consts.__dict__.items():
        setattr(_mod, k, v)
    _sys.modules.setdefault('tigeropen', types.ModuleType('tigeropen'))
    _sys.modules.setdefault('tigeropen.common', types.ModuleType('tigeropen.common'))
    _sys.modules['tigeropen.common.consts'] = _mod
    _util_mod = types.ModuleType('tigeropen.common.util')
    _sig_mod = types.ModuleType('tigeropen.common.util.signature_utils')
    setattr(_sig_mod, 'read_private_key', lambda path=None: 'FAKE_PRIVATE_KEY')
    _cu_mod = types.ModuleType('tigeropen.common.util.contract_utils')
    setattr(_cu_mod, 'stock_contract', lambda *a, **k: None)
    setattr(_cu_mod, 'future_contract', lambda *a, **k: None)
    _sys.modules['tigeropen.common.util'] = _util_mod
    _sys.modules['tigeropen.common.util.signature_utils'] = _sig_mod
    _sys.modules['tigeropen.common.util.contract_utils'] = _cu_mod
    setattr(_util_mod, 'contract_utils', _cu_mod)
    _conf_mod = types.ModuleType('tigeropen.tiger_open_config')
    setattr(_conf_mod, 'TigerOpenClientConfig', lambda props_path=None: types.SimpleNamespace(account='SIM', tiger_id='SIM'))
    _sys.modules['tigeropen.tiger_open_config'] = _conf_mod
    _qmod = types.ModuleType('tigeropen.quote.quote_client')
    setattr(_qmod, 'QuoteClient', lambda cfg: types.SimpleNamespace(get_future_bars=lambda *a, **k: []))
    _sys.modules['tigeropen.quote'] = types.ModuleType('tigeropen.quote')
    _sys.modules['tigeropen.quote.quote_client'] = _qmod
    _tmod = types.ModuleType('tigeropen.trade.trade_client')
    setattr(_tmod, 'TradeClient', lambda cfg: types.SimpleNamespace(place_order=lambda req: types.SimpleNamespace(order_id='SIM')))
    _sys.modules['tigeropen.trade'] = types.ModuleType('tigeropen.trade')
    _sys.modules['tigeropen.trade.trade_client'] = _tmod

import importlib
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



def make_bars(n, tz_naive=True):
    base = datetime.utcnow()
    bars = []
    for i in range(n):
        t = base - timedelta(minutes=(n - i))
        bar = SimpleNamespace(
            time=t if tz_naive else pd.Timestamp(t).tz_localize('UTC'),
            open=1.0 + i,
            high=1.1 + i,
            low=0.9 + i,
            close=1.05 + i,
            volume=100 + i
        )
        bars.append(bar)
    return bars


def test_get_kline_data_from_dataframe(monkeypatch):
    if tiger1 is None:
        pytest.fail(f"can't import src.tiger1: {import_error}")
    # 强制走 quote_client 路径，避免 is_mock_mode 时用 api_manager.quote_api 导致返回格式不一致
    monkeypatch.setattr(tiger1.api_manager, 'is_mock_mode', False)
    times = pd.date_range(end=pd.Timestamp.utcnow(), periods=tiger1.MIN_KLINES + 2, freq='min')
    df_in = pd.DataFrame({
        'time': times,
        'open': range(len(times)),
        'high': range(len(times)),
        'low': range(len(times)),
        'close': range(len(times)),
        'volume': range(len(times))
    })

    def fake_get_future_bars(symbols, period, a, b, count, c):
        return df_in

    monkeypatch.setattr(tiger1, 'quote_client', types.SimpleNamespace(get_future_bars=fake_get_future_bars))

    df = tiger1.get_kline_data('SIL.COMEX.202603', '1min', count=tiger1.MIN_KLINES)
    assert not df.empty
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        assert str(df.index.tz) == 'Asia/Shanghai'
    assert len(df) == tiger1.MIN_KLINES


def test_get_kline_data_from_list(monkeypatch):
    if tiger1 is None:
        pytest.fail(f"can't import src.tiger1: {import_error}")
    monkeypatch.setattr(tiger1.api_manager, 'is_mock_mode', False)
    bars = make_bars(tiger1.MIN_KLINES + 2, tz_naive=True)

    def fake_get_future_bars(symbols, period, a, b, count, c):
        return bars

    monkeypatch.setattr(tiger1, 'quote_client', types.SimpleNamespace(get_future_bars=fake_get_future_bars))

    df = tiger1.get_kline_data(['SIL.COMEX.202603'], '1min', count=tiger1.MIN_KLINES)
    assert not df.empty
    assert len(df) == tiger1.MIN_KLINES
    assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])


def test_get_kline_data_from_generator(monkeypatch):
    if tiger1 is None:
        pytest.fail(f"can't import src.tiger1: {import_error}")
    monkeypatch.setattr(tiger1.api_manager, 'is_mock_mode', False)
    bars = make_bars(tiger1.MIN_KLINES + 2, tz_naive=True)

    def gen():
        for b in bars:
            yield b

    def fake_get_future_bars(symbols, period, a, b, count, c):
        return gen()

    monkeypatch.setattr(tiger1, 'quote_client', types.SimpleNamespace(get_future_bars=fake_get_future_bars))

    df = tiger1.get_kline_data('SIL.COMEX.202603', '1min', count=tiger1.MIN_KLINES)
    assert not df.empty
    assert len(df) == tiger1.MIN_KLINES


def test_get_kline_data_insufficient(monkeypatch):
    if tiger1 is None:
        pytest.fail(f"can't import src.tiger1: {import_error}")
    monkeypatch.setattr(tiger1.api_manager, 'is_mock_mode', False)
    bars = make_bars(max(1, tiger1.MIN_KLINES - 1), tz_naive=True)

    def fake_get_future_bars(symbols, period, a, b, count, c):
        return bars

    monkeypatch.setattr(tiger1, 'quote_client', types.SimpleNamespace(get_future_bars=fake_get_future_bars))

    df = tiger1.get_kline_data('SIL.COMEX.202603', '1min', count=tiger1.MIN_KLINES)
    assert df.empty


def test_get_kline_data_with_paging(monkeypatch):
    if tiger1 is None:
        pytest.fail(f"can't import src.tiger1: {import_error}")
    monkeypatch.setattr(tiger1.api_manager, 'is_mock_mode', False)

    # two pages, each with 5 bars
    now = pd.Timestamp.utcnow()
    times1 = pd.date_range(end=now, periods=5, freq='min')
    df1 = pd.DataFrame({
        'time': times1,
        'open': range(5),
        'high': range(5),
        'low': range(5),
        'close': range(5),
        'volume': range(5)
    })
    times2 = pd.date_range(end=times1[0] - pd.Timedelta(minutes=1), periods=5, freq='min')
    df2 = pd.DataFrame({
        'time': times2,
        'open': range(5, 10),
        'high': range(5, 10),
        'low': range(5, 10),
        'close': range(5, 10),
        'volume': range(5, 10)
    })

    calls = {'i': 0}
    def fake_by_page(identifier, period, begin_time, end_time, total, page_size=None, time_interval=None):
        calls['begin_time'] = begin_time
        calls['end_time'] = end_time
        if calls['i'] == 0:
            calls['i'] += 1
            return df1, 'TOKEN'
        else:
            return df2, None

    monkeypatch.setattr(tiger1, 'quote_client', types.SimpleNamespace(get_future_bars_by_page=fake_by_page))

    # request 8 bars total — should concatenate and keep most recent 8
    start = datetime.utcnow() - timedelta(minutes=20)
    end = datetime.utcnow()
    df = tiger1.get_kline_data('SIL.COMEX.202603', '1min', count=8, start_time=start, end_time=end)

    assert not df.empty
    assert len(df) == 8
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        assert str(df.index.tz) == 'Asia/Shanghai'


def test_get_kline_data_time_range_ms_arg(monkeypatch):
    if tiger1 is None:
        pytest.fail(f"can't import src.tiger1: {import_error}")
    monkeypatch.setattr(tiger1.api_manager, 'is_mock_mode', False)

    now = datetime.utcnow()
    start_dt = now - timedelta(minutes=10)
    end_dt = now
    start_ms = int(start_dt.replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(end_dt.replace(tzinfo=timezone.utc).timestamp() * 1000)

    times = pd.date_range(end=pd.Timestamp.utcnow(), periods=5, freq='min')
    df_page = pd.DataFrame({
        'time': times,
        'open': range(5),
        'high': range(5),
        'low': range(5),
        'close': range(5),
        'volume': range(5)
    })

    captured = {}
    def fake_by_page(identifier, period, begin_time, end_time, total, page_size=None, time_interval=None):
        captured['begin_time'] = begin_time
        captured['end_time'] = end_time
        return df_page, None

    monkeypatch.setattr(tiger1, 'quote_client', types.SimpleNamespace(get_future_bars_by_page=fake_by_page))

    df = tiger1.get_kline_data('SIL.COMEX.202603', '1min', count=3, start_time=start_ms, end_time=end_ms)
    assert not df.empty
    assert 'begin_time' in captured and 'end_time' in captured
    assert isinstance(captured['begin_time'], (int, float)) or captured['begin_time'] is None
    assert isinstance(captured['end_time'], (int, float)) or captured['end_time'] is None
    if captured['begin_time'] is not None and captured['end_time'] is not None:
        assert captured['begin_time'] <= captured['end_time']
    assert len(df) == 3


def test_get_kline_data_symbol_conversion(monkeypatch):
    """Verify that dotted contract symbols are converted to compact identifiers
    before making by-page API calls (e.g., 'SIL.COMEX.202603' -> 'SIL2603')."""
    if tiger1 is None:
        pytest.fail(f"can't import src.tiger1: {import_error}")
    monkeypatch.setattr(tiger1.api_manager, 'is_mock_mode', False)

    times = pd.date_range(end=pd.Timestamp.utcnow(), periods=3, freq='min')
    df_page = pd.DataFrame({
        'time': times,
        'open': range(3),
        'high': range(3),
        'low': range(3),
        'close': range(3),
        'volume': range(3)
    })

    captured = {}
    def fake_by_page(identifier, period, begin_time, end_time, total, page_size=None, time_interval=None):
        captured['identifier'] = identifier
        return df_page, None

    monkeypatch.setattr(tiger1, 'quote_client', types.SimpleNamespace(get_future_bars_by_page=fake_by_page))

    df = tiger1.get_kline_data('SIL.COMEX.202603', '1min', count=2, start_time=datetime.utcnow() - timedelta(minutes=20), end_time=datetime.utcnow())
    assert not df.empty
    # expect converted identifier
    assert captured.get('identifier') == 'SIL2603'
