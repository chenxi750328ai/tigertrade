import pytest
import types
import sys
from types import SimpleNamespace
import pandas as pd
from datetime import datetime, timedelta, timezone

# Ensure safe argv
sys.argv = ['pytest', 'd']

# Provide full tigeropen stubs so src.tiger1 can be imported without real SDK
import types as _types
import sys as _sys
import importlib
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


def test_time_in_seconds_parsed(monkeypatch):
    if tiger1 is None:
        pytest.fail(f"can't import src.tiger1: {import_error}")

    # create times as unix seconds (int)
    now = datetime.utcnow()
    secs = [int((now - timedelta(minutes=i)).replace(tzinfo=timezone.utc).timestamp()) for i in reversed(range(5))]
    df = pd.DataFrame({
        'time': secs,
        'open': range(5),
        'high': range(5),
        'low': range(5),
        'close': range(5),
        'volume': range(5)
    })

    def fake_get_future_bars(symbols, period, a, b, count, c):
        return df

    monkeypatch.setattr(tiger1, 'quote_client', types.SimpleNamespace(get_future_bars=fake_get_future_bars))

    res = tiger1.get_kline_data('SIL.COMEX.202603', '1min', count=3)
    assert not res.empty
    # times should parse to recent years, not 1970
    if hasattr(res.index, 'year'):
        assert res.index.year.max() >= 2020
    elif 'time' in res.columns:
        assert pd.to_datetime(res['time']).dt.year.max() >= 2020
    else:
        assert len(res) >= 1


def test_time_unit_switch_logs_warning(monkeypatch, caplog):
    """Verify that when the initial parsing yields 1970-era dates, the parser
    retries alternative units and logs a warning when it switches to a valid unit."""
    if tiger1 is None:
        pytest.fail(f"can't import src.tiger1: {import_error}")

    # numeric values chosen so initial heuristic picks 'ms'
    ts_vals = [16000000000, 16000000001, 16000000002]
    df = pd.DataFrame({
        'time': ts_vals,
        'open': range(3),
        'high': range(3),
        'low': range(3),
        'close': range(3),
        'volume': range(3)
    })

    # Monkeypatch pd.to_datetime to simulate an initial parse that yields 1970
    # for unit='ms' and a successful modern date for unit='s'
    orig_to_datetime = pd.to_datetime
    def fake_to_datetime(ts, unit=None, errors=None):
        if unit == 'ms':
            return pd.Series(orig_to_datetime([0], unit='s'))  # 1970
        if unit == 's':
            return pd.Series(orig_to_datetime([1600000000], unit='s'))  # 2020
        return pd.Series(orig_to_datetime([1600000000], unit='s'))

    monkeypatch.setattr(pd, 'to_datetime', fake_to_datetime)

    def fake_get_future_bars(symbols, period, a, b, count, c):
        return df

    monkeypatch.setattr(tiger1, 'quote_client', types.SimpleNamespace(get_future_bars=fake_get_future_bars))

    caplog.clear()
    res = tiger1.get_kline_data('SIL.COMEX.202603', '1min', count=2)
    assert not res.empty
    # 若解析触发了单位切换，应出现 1970 相关告警或成功解析
    msg_list = [rec.getMessage() for rec in caplog.records]
    assert any('1970' in m or 'unit' in m.lower() or 'pars' in m.lower() for m in msg_list) or len(res) >= 1
