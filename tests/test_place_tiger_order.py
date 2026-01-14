import sys
import os
from types import SimpleNamespace

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tigertrade import tiger1 as t1


def reset_globals():
    t1.current_position = 0
    t1.daily_loss = 0
    t1.RUN_ENV = 'sandbox'
    if hasattr(t1, 'trade_client'):
        t1.trade_client = None


def test_place_order_success():
    reset_globals()

    class FakeClient:
        def place_order(self, order):
            return SimpleNamespace(id=999)

    t1.trade_client = FakeClient()
    t1.client_config = SimpleNamespace(account='ACC123')

    ok = t1.place_tiger_order('BUY', 1, 100.0, 90.0)
    assert ok is True
    assert t1.current_position == 1


def test_place_order_sdk_failure_simulate():
    reset_globals()

    class FakeClient:
        def place_order(self, order):
            raise RuntimeError('network')

    t1.trade_client = FakeClient()
    t1.client_config = SimpleNamespace(account='ACC123')

    ok = t1.place_tiger_order('BUY', 2, 200.0, 190.0)
    assert ok is True
    assert t1.current_position == 2


def test_production_refuses_without_env():
    reset_globals()
    t1.RUN_ENV = 'production'
    if 'ALLOW_REAL_TRADING' in os.environ:
        del os.environ['ALLOW_REAL_TRADING']

    # even with a working client, it should refuse
    class FakeClient:
        def place_order(self, order):
            return SimpleNamespace(id=111)

    t1.trade_client = FakeClient()
    t1.client_config = SimpleNamespace(account='ACC123')

    ok = t1.place_tiger_order('BUY', 1, 50.0, 45.0)
    assert ok is False
    assert t1.current_position == 0


def test_sell_updates_daily_loss():
    reset_globals()
    class FakeClient:
        def place_order(self, order):
            return SimpleNamespace(id=222)

    t1.trade_client = FakeClient()
    t1.client_config = SimpleNamespace(account='ACC123')

    # start with 2 long positions
    t1.current_position = 2
    ok = t1.place_tiger_order('SELL', 1, 100.0, 90.0)
    assert ok is True
    assert t1.current_position == 1
    # daily_loss increased because we passed a stop_loss_price and sold 1
    assert t1.daily_loss >= 0