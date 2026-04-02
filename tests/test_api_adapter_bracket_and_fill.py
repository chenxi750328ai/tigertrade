# -*- coding: utf-8 -*-
"""
组合单（BRACKETS）、成交判定、回退路径的回归测试 — 避免改下单逻辑时静默破坏止损/止盈。
"""
import sys
import unittest
from unittest.mock import MagicMock, patch
from src.api_adapter import RealTradeApiAdapter, _order_status_is_filled
from src.executor.order_executor import OrderExecutor


class TestOrderStatusIsFilled(unittest.TestCase):
    def test_none_not_filled(self):
        self.assertFalse(_order_status_is_filled(None))

    def test_enum_filled(self):
        class S:
            name = "FILLED"

        class O:
            status = S()

        self.assertTrue(_order_status_is_filled(O()))

    def test_cancelled_not_filled(self):
        class S:
            name = "CANCELLED"

        class O:
            status = S()

        self.assertFalse(_order_status_is_filled(O()))

    def test_string_order_status_held(self):
        class O:
            order_status = "HELD"

        self.assertFalse(_order_status_is_filled(O()))


class TestRealTradeApiAdapterPlaceLimitWithBracket(unittest.TestCase):
    """断言 place_limit_with_bracket 交给 Tiger 的 Order 含 LOSS+PROFIT 两腿。"""

    def test_order_has_two_legs_loss_and_profit(self):
        captured = {}

        def capture_place(order, lang=None):
            captured["order"] = order

            class R:
                order_id = 1234567890123

            return R()

        client = MagicMock()
        client.place_order.side_effect = capture_place
        adapter = RealTradeApiAdapter(client, account="1234567890123456789")

        adapter.place_limit_with_bracket(
            "SIL2603", "BUY", 1, 100.0, 97.0, 104.0, "DAY"
        )

        self.assertIn("order", captured)
        order = captured["order"]
        self.assertIsNotNone(getattr(order, "order_legs", None))
        self.assertEqual(len(order.order_legs), 2)
        leg_types = {leg.leg_type for leg in order.order_legs}
        self.assertEqual(leg_types, {"LOSS", "PROFIT"})
        prices = {leg.leg_type: leg.price for leg in order.order_legs}
        self.assertEqual(prices["LOSS"], 97.0)
        self.assertEqual(prices["PROFIT"], 104.0)
        self.assertEqual(order.order_type, "LMT")
        self.assertEqual(order.action, "BUY")
        client.place_order.assert_called_once()


class TestOrderExecutorBracketFallback(unittest.TestCase):
    """组合单失败须回退为限价 + wait FILLED + STP + TP。"""

    def setUp(self):
        from src import tiger1 as t1
        from src.api_adapter import api_manager

        self.t1 = t1
        t1.current_position = 0
        t1.daily_loss = 0
        self._orig_trade = api_manager.trade_api
        self._orig_mock = api_manager.is_mock_mode
        api_manager.trade_api = None
        api_manager.is_mock_mode = True
        self.executor = OrderExecutor(t1)

    def tearDown(self):
        from src.api_adapter import api_manager

        api_manager.trade_api = self._orig_trade
        api_manager.is_mock_mode = self._orig_mock
        self.t1.current_position = 0
        self.t1.daily_loss = 0

    def test_bracket_raises_then_three_place_orders(self):
        from src.api_adapter import api_manager

        mock_order_result = MagicMock()
        mock_order_result.order_id = "MAIN_BRACKET_FAIL"
        mock_trade_api = MagicMock(
            spec=[
                "place_order",
                "place_limit_with_bracket",
                "get_order",
                "get_orders",
                "wait_until_buy_filled",
                "account",
            ]
        )
        mock_trade_api.account = "MOCK_ACC"
        mock_trade_api.place_limit_with_bracket.side_effect = ValueError(
            "BRACKETS not supported"
        )
        mock_trade_api.place_order.return_value = mock_order_result
        mock_trade_api.get_order.return_value = mock_order_result
        mock_trade_api.get_orders.return_value = []
        mock_trade_api.wait_until_buy_filled.return_value = True
        api_manager.trade_api = mock_trade_api
        api_manager.is_mock_mode = False

        try:
            with patch.object(self.t1, "sync_positions_from_backend", return_value=None), patch.object(
                self.t1, "get_effective_position_for_buy", return_value=0
            ):
                ok, msg = self.executor.execute_buy(
                    100.0, 0.5, 98.0, 105.0, 0.6
                )
            self.assertTrue(ok, msg)
            self.assertEqual(mock_trade_api.place_limit_with_bracket.call_count, 1)
            self.assertEqual(
                mock_trade_api.place_order.call_count,
                3,
                "回退路径：主单 LMT + STP + 止盈 LMT",
            )
        finally:
            api_manager.trade_api = self._orig_trade
            api_manager.is_mock_mode = self._orig_mock

    def test_wait_fill_false_then_no_sl_tp_place_order(self):
        """限价未 FILLED：不得再下 STP/TP（仅 1 次主单 place_order）。"""
        from src.api_adapter import api_manager

        mock_order_result = MagicMock()
        mock_order_result.order_id = "MAIN_WAIT_FAIL"
        mock_trade_api = MagicMock(
            spec=[
                "place_order",
                "get_order",
                "get_orders",
                "wait_until_buy_filled",
                "account",
            ]
        )
        mock_trade_api.account = "MOCK_ACC"
        mock_trade_api.place_order.return_value = mock_order_result
        mock_trade_api.get_order.return_value = mock_order_result
        mock_trade_api.get_orders.return_value = []
        mock_trade_api.wait_until_buy_filled.return_value = False
        api_manager.trade_api = mock_trade_api
        api_manager.is_mock_mode = False

        try:
            with patch.object(self.t1, "sync_positions_from_backend", return_value=None), patch.object(
                self.t1, "get_effective_position_for_buy", return_value=0
            ):
                ok, msg = self.executor.execute_buy(
                    100.0, 0.5, 98.0, 105.0, 0.6
                )
            self.assertTrue(ok, msg)
            self.assertEqual(
                mock_trade_api.place_order.call_count,
                1,
                "未 FILLED 时不应提交 STP/TP",
            )
        finally:
            api_manager.trade_api = self._orig_trade
            api_manager.is_mock_mode = self._orig_mock


if __name__ == "__main__":
    unittest.main()
