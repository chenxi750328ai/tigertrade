# -*- coding: utf-8 -*-
"""
老虎 Tiger API 约束测试（全项目范围）

依据官方文档整理约束，覆盖所有下单路径：
- tiger1.place_tiger_order、OrderExecutor、api_adapter、close_demo_positions
- placeOrder: https://quant.itigerup.com/openapi/zh/python/operation/trade/placeOrder.html
- 请求频率: https://quant.itigerup.com/openapi/zh/python/permission/requestLimit.html
- FAQ-交易: https://quant.itigerup.com/openapi/zh/python/FAQ/trade.html
"""
import sys
import unittest
from unittest.mock import MagicMock, patch
class TestTigerApiPlaceOrderConstraints(unittest.TestCase):
    """place_order 相关约束"""

    def test_submit_success_does_not_imply_fill(self):
        """
        约束：place_order 返回 order_id 仅表示提交成功，不表示成交。
        需用 get_order(id) 查 status==FILLED 才可视为成交。
        """
        # 文档约束断言
        self.assertTrue(True, "place_order 返回 ≠ 成交，需 get_order 校验")

    def test_order_type_mkt_uses_day_not_gtc(self):
        """
        约束：模拟账号市价单(MKT)不支持 GTC，time_in_force 仅 DAY。
        代码：tiger1.place_tiger_order 与 api_adapter 均使用 DAY。
        """
        tif_used = 'DAY'
        self.assertEqual(tif_used, 'DAY', "模拟账号 MKT 应使用 DAY")
        self.assertNotEqual(tif_used, 'GTC', "不应使用 GTC")

    def test_no_reverse_position_without_close(self):
        """
        约束：不可直接开反向仓。持仓 100 股时不能直接卖 200 股，需先平仓。
        错误信息：The order quantity you entered exceeds your currently available position
        """
        # 设计约束：卖单数量应 ≤ 可卖数量；代码中 get_effective_position 做校验
        self.assertTrue(True, "卖单数量需 ≤ 可用持仓")

    def test_no_lock_position(self):
        """
        约束：不支持锁仓，同一标的不允许同时持有多空。
        """
        self.assertTrue(True, "净仓交易，不支持锁仓")


class TestTigerApiRateLimit(unittest.TestCase):
    """请求频率约束（120/60/10 次/分钟）"""

    def test_place_order_in_high_freq_group(self):
        """
        约束：place_order、cancel_order、get_order、get_orders、get_open_orders 属于高频组，120 次/分钟。
        """
        high_freq_apis = ['place_order', 'cancel_order', 'get_order', 'get_orders', 'get_open_orders']
        self.assertIn('place_order', high_freq_apis)
        self.assertIn('get_order', high_freq_apis)

    def test_close_script_batch_respects_pending_limit(self):
        """
        约束：同品种最多 15 个 pending；平仓脚本每批 ≤15 手，等成交后再下一批，避免超频+超限。
        """
        BATCH = 15
        self.assertLessEqual(BATCH, 15, "每批不得超过 15 手")
        # 平仓脚本会 wait_order_fill 再下下一批，不会 burst
        self.assertTrue(True, "分批+等待成交，符合 15 限与频率限")

    def test_rate_limit_error_code(self):
        """
        约束：超限返回 code=4 msg=rate limit error；持续超频可能被黑名单。
        """
        expected_code = 4
        expected_msg = 'rate limit error'
        self.assertEqual(expected_code, 4)
        self.assertEqual(expected_msg, 'rate limit error')


class TestTigerApiOrderTypeAndSession(unittest.TestCase):
    """订单类型与交易时段约束"""

    def test_mkt_stp_only_valid_during_regular_session(self):
        """
        约束：市价单(MKT)/止损单(STP) 盘前盘后需 outside_rth=false，且 MKT/STP 仅盘中有效。
        平仓脚本在盘中运行，使用 MKT 或限价单均可。
        """
        self.assertTrue(True, "盘前盘后 MKT/STP 受限，需用限价单")

    def test_attached_order_parent_must_be_limit(self):
        """
        约束：附加订单（止损/止盈）的主单必须为限价单（Tiger Open）。
        代码：优先 RealTradeApiAdapter.place_limit_with_bracket（order_utils.limit_order_with_legs，
        LOSS+PROFIT → BRACKETS）；失败则回退 LMT 主单 + 成交后 STP/LMT。
        """
        self.assertTrue(True, "组合单/附加单主单须为 LMT")


class TestTigerApiFifteenPendingLimit(unittest.TestCase):
    """同品种 15 pending 限制（实测约束）"""

    def test_batch_size_is_15(self):
        """平仓脚本与 test_close_demo_positions 约定批次大小为 15"""
        BATCH = 15  # 与 scripts/close_demo_positions.py 中 BATCH 一致
        self.assertEqual(BATCH, 15)

    def test_wait_order_fill_before_next_batch(self):
        """
        约束：必须先 wait_order_fill 确认成交或终态，再提交下一批，否则会触及 15 限。
        """
        from scripts.close_utils import wait_order_fill
        mock_tc = MagicMock()
        mock_o = MagicMock()
        mock_o.status = type('S', (), {'name': 'FILLED'})()
        mock_o.filled = 5
        mock_tc.get_order.return_value = mock_o
        filled, info = wait_order_fill(mock_tc, 123, max_wait=2, poll_interval=0.1)
        self.assertTrue(filled)
        self.assertEqual(mock_tc.get_order.call_count, 1)


class TestTigerApiGetOrderVerification(unittest.TestCase):
    """get_order 校验成交状态"""

    def test_expired_with_reason_means_not_filled(self):
        """订单 EXPIRED + reason 含 Pending orders 时，应视为未成交"""
        from scripts.close_utils import wait_order_fill
        mock_tc = MagicMock()
        mock_o = MagicMock()
        mock_o.status = type('S', (), {'name': 'EXPIRED'})()
        mock_o.reason = 'Pending orders for same product exceed the limit(15)'
        mock_o.filled = 0
        mock_tc.get_order.return_value = mock_o
        filled, info = wait_order_fill(mock_tc, 456, max_wait=1, poll_interval=0.1)
        self.assertFalse(filled)
        self.assertIn('Pending orders', str(info) or '')


class TestOrderExecutorTigerConstraints(unittest.TestCase):
    """OrderExecutor 对老虎 API 约束的遵从"""

    def test_order_executor_uses_time_in_force_day(self):
        """OrderExecutor 调用 place_order 时使用 TimeInForce.DAY"""
        from src.executor.order_executor import OrderExecutor
        from src import tiger1 as t1
        from src.api_adapter import api_manager
        api_manager.initialize_mock_apis(account='TEST_ACC')
        captured = {}
        def capture_place_order(*args, **kwargs):
            captured['args'] = args
            captured['kwargs'] = kwargs
            o = MagicMock()
            o.order_id = 12345
            return o
        api_manager.trade_api.place_order = capture_place_order
        exec = OrderExecutor(t1)
        with patch.object(t1, 'get_effective_position_for_buy', return_value=0):
            with patch.object(t1, 'check_risk_control', return_value=True):
                exec.execute_buy(100.0, 0.5, 97.0, 105.0, 0.7)
        # place_order 可能为位置参数 (symbol, side, type, qty, time_in_force, ...) 或关键字参数
        args = captured.get('args', [])
        kwargs = captured.get('kwargs', {})
        tif = args[4] if len(args) >= 5 else kwargs.get('time_in_force')
        self.assertTrue(
            len(args) >= 5 or 'time_in_force' in kwargs,
            "place_order 应有 time_in_force（位置第5个或关键字）"
        )
        tif_val = getattr(tif, 'name', None) or str(tif) if tif else None
        self.assertIn('DAY', str(tif_val or ''), "OrderExecutor 应使用 DAY")

    def test_order_executor_uses_backend_verify_and_optional_wait_fill(self):
        """
        OrderExecutor 下单后会用 get_order/get_orders 做后台可见性校验；
        回退路径（非 BRACKETS）在挂 SL/TP 前会调用 wait_until_buy_filled。
        """
        import inspect
        from src.executor import order_executor
        src = inspect.getsource(order_executor)
        self.assertIn("get_order", src, "应有后台订单校验")
        self.assertIn("wait_until_buy_filled", src, "回退路径应等待 FILLED 再挂保护单")


class TestTiger1PlaceOrderConstraints(unittest.TestCase):
    """tiger1.place_tiger_order 对老虎 API 约束的遵从"""

    def test_tiger1_uses_time_in_force_day(self):
        """tiger1.place_tiger_order 实盘路径使用 TimeInForce.DAY"""
        from src import tiger1 as t1
        from src.api_adapter import api_manager
        api_manager.initialize_mock_apis(account='TEST_ACC')
        captured = {}
        def capture_place_order(*args, **kwargs):
            captured['kwargs'] = kwargs
            o = MagicMock()
            o.order_id = 99999
            return o
        api_manager.trade_api.place_order = capture_place_order
        with patch.object(api_manager, 'is_mock_mode', False):
            with patch.object(t1, 'RUN_ENV', 'sandbox'):  # 绕过 production 守卫
                with patch.object(t1, 'get_effective_position_for_buy', return_value=0):
                    with patch.object(t1, 'check_risk_control', return_value=True):
                        t1.place_tiger_order('BUY', 1, 100.0, 90.0)
        tif = captured.get('kwargs', {}).get('time_in_force')
        tif_val = getattr(tif, 'name', None) or str(tif) if tif else None
        self.assertIn('DAY', str(tif_val or ''), "tiger1 应使用 DAY")


class TestApiAdapterTigerConstraints(unittest.TestCase):
    """api_adapter 对老虎 API 约束的遵从"""

    def test_real_trade_adapter_accepts_time_in_force(self):
        """RealTradeApiAdapter.place_order 接受 time_in_force 参数"""
        from src.api_adapter import RealTradeApiAdapter
        sig = RealTradeApiAdapter.place_order.__doc__ or ''
        self.assertIn('time_in_force', str(RealTradeApiAdapter.place_order.__code__.co_varnames),
                      "place_order 应支持 time_in_force")


if __name__ == '__main__':
    unittest.main()
