"""
order_log 业务语义单测：实盘成功门禁、客服排障日志、order_type 归一化。
依赖 conftest 将 ORDER_LOG_FILE 指向临时目录，不污染 run/。
"""
import json
import sys
import unittest
from src import order_log


class TestOrderLogBehavior(unittest.TestCase):
    def test_real_success_with_mock_order_id_becomes_fail(self):
        """防回归：mock 风格 order_id 不得记为 real+success（与黑盒/对账一致）"""
        order_log.log_order(
            "BUY",
            1,
            30.0,
            "ORDER_12345",
            "success",
            mode="real",
            symbol="SIL2603",
            order_type="limit",
        )
        with open(order_log.ORDER_LOG_FILE, "r", encoding="utf-8") as f:
            line = f.readlines()[-1]
        rec = json.loads(line)
        self.assertEqual(rec["status"], "fail")
        self.assertIn("order_id", (rec.get("error") or "").lower())

    def test_real_success_with_numeric_long_id_kept(self):
        order_log.log_order(
            "BUY",
            1,
            30.0,
            "1234567890123",
            "success",
            mode="real",
            symbol="SIL2603",
        )
        with open(order_log.ORDER_LOG_FILE, "r", encoding="utf-8") as f:
            line = f.readlines()[-1]
        rec = json.loads(line)
        self.assertEqual(rec["status"], "success")
        self.assertEqual(rec["mode"], "real")

    def test_invalid_order_type_normalized_to_limit(self):
        order_log.log_order("SELL", 1, 29.0, "X1", "success", mode="mock", order_type="weird_type")
        with open(order_log.ORDER_LOG_FILE, "r", encoding="utf-8") as f:
            line = f.readlines()[-1]
        rec = json.loads(line)
        self.assertEqual(rec["order_type"], "limit")

    def test_log_api_failure_for_support_appends_jsonl(self):
        order_log.log_api_failure_for_support(
            side="BUY",
            quantity=1,
            price=30.0,
            symbol_submitted="SIL2603",
            order_type_api="LMT",
            time_in_force="DAY",
            limit_price=30.0,
            stop_price=None,
            error="test_error",
            source="auto",
            order_id="",
        )
        with open(order_log.API_FAILURE_FOR_SUPPORT_FILE, "r", encoding="utf-8") as f:
            line = f.readlines()[-1]
        rec = json.loads(line)
        self.assertEqual(rec["symbol_submitted"], "SIL2603")
        self.assertEqual(rec["error"], "test_error")
        self.assertEqual(rec["time_in_force"], "DAY")

    def test_log_take_profit_sets_order_type(self):
        order_log.log_take_profit("SELL", 1, 32.0, "OID", "success", mode="mock", symbol="SIL2603")
        with open(order_log.ORDER_LOG_FILE, "r", encoding="utf-8") as f:
            line = f.readlines()[-1]
        rec = json.loads(line)
        self.assertEqual(rec["order_type"], "take_profit")

    def test_log_stop_loss_sets_order_type(self):
        order_log.log_stop_loss("SELL", 1, 28.0, "OID", "fail", mode="mock", symbol="SIL2603", error="x")
        with open(order_log.ORDER_LOG_FILE, "r", encoding="utf-8") as f:
            line = f.readlines()[-1]
        rec = json.loads(line)
        self.assertEqual(rec["order_type"], "stop_loss")


if __name__ == "__main__":
    unittest.main()
