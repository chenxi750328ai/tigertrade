import sys
import types
import pandas as pd
import pytest

# Keep argv simple so tiger1 does not misinterpret pytest flags
sys.argv = ['pytest']

sys.path.insert(0, '/home/cx/tigertrade')

from src import tiger1 as t1
from src.api_adapter import api_manager


@pytest.fixture(autouse=True)
def restore_api_manager_state():
    """Reset api_manager between tests to avoid cross-test pollution."""
    orig_quote = api_manager.quote_api
    orig_is_mock = api_manager.is_mock_mode
    yield
    api_manager.quote_api = orig_quote
    api_manager.is_mock_mode = orig_is_mock


class DummyQuoteApi:
    """Lightweight stub that satisfies verify_api_connection calls."""

    def get_stock_briefs(self, symbols):
        return pd.DataFrame({'symbol': symbols, 'price': [500.0 for _ in symbols]})

    def get_future_exchanges(self):
        return types.SimpleNamespace(iloc=[types.SimpleNamespace(code='CME', name='CME', zone='UTC')])

    def get_future_contracts(self, exchange_code):
        return pd.DataFrame({
            'contract_code': ['SIL2603'],
            'name': ['Silver 2026'],
            'multiplier': [1000],
            'last_trading_date': ['2026-03-28']
        })

    def get_all_future_contracts(self, product_code):
        return self.get_future_contracts(product_code)

    def get_current_future_contract(self, product_code):
        df = self.get_future_contracts(product_code)
        return df.iloc[0]

    def get_quote_permission(self):
        return ['REAL_TIME']

    def get_future_brief(self, symbols):
        return pd.DataFrame({'identifier': symbols, 'price': [25.5 for _ in symbols]})

    def get_future_bars(self, symbols, period, begin_time, end_time, count, right=None):
        times = pd.date_range(end=pd.Timestamp.utcnow(), periods=count, freq='1min')
        return pd.DataFrame({
            'time': times,
            'open': range(count),
            'high': range(count),
            'low': range(count),
            'close': range(count),
            'volume': range(count)
        })


def test_verify_api_connection_uses_stubbed_quote_api():
    api_manager.is_mock_mode = False
    api_manager.quote_api = DummyQuoteApi()

    assert t1.verify_api_connection() is True


def test_get_future_brief_info_mock_mode_returns_defaults():
    api_manager.initialize_mock_apis()

    info = t1.get_future_brief_info('SIL2603')
    assert info['multiplier'] == t1.FUTURE_MULTIPLIER
    assert info['min_tick'] == t1.MIN_TICK


def test_get_future_brief_info_non_mock_returns_defaults():
    api_manager.is_mock_mode = False
    api_manager.quote_api = DummyQuoteApi()

    info = t1.get_future_brief_info('SIL2603')
    assert info['multiplier'] == t1.FUTURE_MULTIPLIER
    assert info['min_tick'] == t1.MIN_TICK


def test_to_api_identifier_variants():
    assert t1._to_api_identifier('SIL.COMEX.202603') == 'SIL2603'
    assert t1._to_api_identifier('SIL2603') == 'SIL2603'
    assert t1._to_api_identifier('ABC.202512') == 'ABC2512'
    assert t1._to_api_identifier('INVALID') == 'INVALID'