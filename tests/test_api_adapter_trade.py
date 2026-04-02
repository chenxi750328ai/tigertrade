"""
RealTradeApiAdapter / MockTradeApiAdapter 关键路径单测（P1：下单正确性与边界）。
与项目目标对齐：减少 API 误用、account 为空、订单类型参数错误导致的实盘风险。
"""
import sys
import unittest
from unittest.mock import MagicMock, patch
from src import api_adapter
from src.api_adapter import MockTradeApiAdapter, RealTradeApiAdapter


class TestMockTradeApiAdapter(unittest.TestCase):
    def test_place_order_raises_without_account(self):
        api = MockTradeApiAdapter(account=None)
        with self.assertRaises(ValueError) as ctx:
            api.place_order("SIL2603", "BUY", "LMT", 1, "DAY", limit_price=30.0)
        self.assertIn("account", str(ctx.exception).lower())

    def test_place_order_stores_order_and_returns_with_id(self):
        api = MockTradeApiAdapter(account="TEST_ACC")
        r = api.place_order("SIL2603", "BUY", "LMT", 1, "DAY", limit_price=30.0, stop_price=None)
        self.assertTrue(hasattr(r, "order_id"))
        self.assertEqual(r.symbol, "SIL2603")
        self.assertEqual(r.time_in_force, "DAY")
        self.assertIn(r.order_id, api.orders)

    def test_place_order_with_order_object_requires_account_on_order(self):
        api = MockTradeApiAdapter(account="TEST_ACC")
        o = MagicMock()
        o.account = None
        o.contract = MagicMock(symbol="X")
        o.action = "BUY"
        o.order_type = "LMT"
        o.quantity = 1
        o.time_in_force = "DAY"
        o.limit_price = 1.0
        with self.assertRaises(ValueError):
            api.place_order(order=o)

    def test_get_orders_filters_by_symbol_variants(self):
        api = MockTradeApiAdapter(account="A")
        api.place_order("SIL.COMEX.202603", "BUY", "LMT", 1, "DAY", limit_price=30.0)
        stored = api.orders[next(iter(api.orders))]
        full = api.get_orders(account="A", symbol="SIL.COMEX.202603", limit=10)
        self.assertEqual(len(full), 1)
        self.assertEqual(full[0].symbol, stored.symbol)
        # 短格式 SIL2603 与 SIL.COMEX.202603 在变体上未必互查，单独测短码下单
        api2 = MockTradeApiAdapter(account="B")
        api2.place_order("SIL2603", "BUY", "LMT", 1, "DAY", limit_price=30.0)
        short_only = api2.get_orders(account="B", symbol="SIL2603", limit=10)
        self.assertEqual(len(short_only), 1)

    def test_get_order_by_string_id(self):
        api = MockTradeApiAdapter(account="A")
        r = api.place_order("S", "BUY", "MKT", 1, "DAY")
        found = api.get_order(id=r.order_id)
        self.assertIsNotNone(found)
        self.assertEqual(found.order_id, r.order_id)


class TestRealTradeApiAdapter(unittest.TestCase):
    def test_get_orders_returns_empty_without_account_or_symbol(self):
        client = MagicMock()
        ad = RealTradeApiAdapter(client, account="ACC")
        self.assertEqual(ad.get_orders(symbol=None), [])
        ad2 = RealTradeApiAdapter(client, account=None)
        self.assertEqual(ad2.get_orders(account=None, symbol="SIL2603"), [])

    def test_get_order_returns_none_without_id(self):
        ad = RealTradeApiAdapter(MagicMock(), account="ACC")
        self.assertIsNone(ad.get_order(id=None, order_id=None))

    def test_place_order_raises_when_account_missing(self):
        c = MagicMock()
        c.account = None
        c.config = MagicMock()
        c.config.account = None
        ad = RealTradeApiAdapter(c, account=None)
        bare_trade_api = type("Bare", (), {})()
        mock_mgr = MagicMock(_account=None, trade_api=bare_trade_api)
        with patch.object(api_adapter, "api_manager", mock_mgr):
            with self.assertRaises(Exception) as ctx:
                ad.place_order("SIL2603", "BUY", "LMT", 1, "DAY", limit_price=30.0)
        # ValueError 会被包装为 Exception，仍须包含 account 语义
        self.assertIn("account", str(ctx.exception).lower())

    @patch("tigeropen.common.util.order_utils.limit_order")
    @patch("tigeropen.common.util.contract_utils.future_contract")
    def test_place_order_lmt_calls_client_place_order(self, mock_fc, mock_limit):
        mock_fc.return_value = MagicMock(name="contract")
        mock_limit.return_value = MagicMock(name="order_obj")
        c = MagicMock()
        c.config = MagicMock()
        c.place_order.return_value = MagicMock(order_id=999)
        ad = RealTradeApiAdapter(c, account="MYACC")
        out = ad.place_order("SIL2603", "BUY", "LMT", 2, "DAY", limit_price=31.5)
        mock_limit.assert_called_once()
        c.place_order.assert_called_once()
        self.assertIsNotNone(out)

    @patch("tigeropen.common.util.order_utils.stop_order")
    @patch("tigeropen.common.util.contract_utils.future_contract")
    def test_place_order_stp_requires_stop_price(self, mock_fc, mock_stop):
        mock_fc.return_value = MagicMock()
        c = MagicMock()
        c.config = MagicMock()
        ad = RealTradeApiAdapter(c, account="MYACC")
        with self.assertRaises(Exception) as ctx:
            ad.place_order("SIL2603", "SELL", "STP", 1, "DAY", limit_price=None, stop_price=None)
        self.assertIn("stop_price", str(ctx.exception))

    @patch("tigeropen.common.util.order_utils.market_order")
    @patch("tigeropen.common.util.contract_utils.future_contract")
    def test_place_order_mkt_uses_market_order(self, mock_fc, mock_mkt):
        mock_fc.return_value = MagicMock()
        mock_mkt.return_value = MagicMock()
        c = MagicMock()
        c.config = MagicMock()
        c.place_order.return_value = MagicMock(order_id=1)
        ad = RealTradeApiAdapter(c, account="MYACC")
        ad.place_order("SIL2603", "BUY", "MKT", 1, "DAY")
        mock_mkt.assert_called_once()
        c.place_order.assert_called_once()


if __name__ == "__main__":
    unittest.main()
