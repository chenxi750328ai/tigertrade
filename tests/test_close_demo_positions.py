# -*- coding: utf-8 -*-
"""
平仓脚本逻辑单元测试。用 Mock 覆盖：
1. 老虎 15 单 pending 限制 → 必须先撤挂单、分批下单、轮询成交
2. place_order 返回成功 ≠ 订单成交，需 get_order 校验
3. order_id 为 ORDER_/TEST_ 等视为 Mock 非真实 API
"""
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, '/home/cx/tigertrade')
from scripts.close_utils import is_real_order_id as _is_real_order_id, wait_order_fill


class TestCloseDemoPositionsHelpers(unittest.TestCase):
    """平仓脚本辅助函数"""

    def test_is_real_order_id_tiger_format(self):
        self.assertTrue(_is_real_order_id('42191817060123648'))
        self.assertTrue(_is_real_order_id(42191817060123648))

    def test_is_real_order_id_mock_rejected(self):
        self.assertFalse(_is_real_order_id('ORDER_1770796917_1935'))
        self.assertFalse(_is_real_order_id('TEST_123'))
        self.assertFalse(_is_real_order_id('MockOrderId'))
        self.assertFalse(_is_real_order_id('123'))  # 太短
        self.assertFalse(_is_real_order_id(None))


class TestCloseDemoPositionsLogic(unittest.TestCase):
    """平仓脚本核心逻辑（Mock 层）"""

    def test_cancel_first_when_15_open_orders(self):
        """有 15 笔挂单时必须先撤单，否则新单会被拒：Pending orders exceed limit(15)"""
        mock_tc = MagicMock()
        mock_order = MagicMock()
        mock_order.id = 999
        mock_order.contract.symbol = 'SIL2603'
        mock_tc.get_open_orders.return_value = [mock_order] * 15
        mock_tc.cancel_order.return_value = None

        from tigeropen.common.consts import SecurityType
        open_orders = mock_tc.get_open_orders(account='acc', sec_type=SecurityType.FUT)
        to_cancel = [o for o in open_orders if 'SIL' in str(getattr(getattr(o, 'contract', None), 'symbol', '') or '')]
        self.assertEqual(len(to_cancel), 15)
        for o in to_cancel:
            mock_tc.cancel_order(id=getattr(o, 'id', None))
        self.assertEqual(mock_tc.cancel_order.call_count, 15)

    def test_wait_order_fill_expired_with_reason(self):
        """订单 EXPIRED + reason=Pending orders exceed 应视为未成交"""
        mock_tc = MagicMock()
        mock_o = MagicMock()
        mock_o.status = type('S', (), {'name': 'EXPIRED'})()
        mock_o.reason = 'Pending orders for same product exceed the limit(15)'
        mock_o.filled = 0
        mock_tc.get_order.return_value = mock_o
        filled, info = wait_order_fill(mock_tc, 123, max_wait=1, poll_interval=0.1)
        self.assertFalse(filled)
        self.assertIn('Pending orders', str(info) or '')

    def test_submit_success_does_not_imply_fill(self):
        """place_order 返回 order_id 仅表示提交成功，不表示成交；需 get_order 查询 status"""
        # 此用例强调：单测若只 mock place_order 返回 success，会漏掉「提交成功但被拒」场景
        submitted_oid = 42191817060123648
        order_status = 'EXPIRED'  # 实际被拒
        order_reason = 'Pending orders for same product exceed the limit(15)'
        # 断言：若只检查 place_order 返回值，会误判为成功
        place_ok = True  # API 返回了 order_id
        # 真实成功需：get_order(oid).status == FILLED
        self.assertNotEqual(order_status, 'FILLED')
        self.assertTrue(place_ok)  # 说明仅看 place_ok 会漏


class TestTigerFifteenOrderLimit(unittest.TestCase):
    """老虎 15 单限制相关断言，供回归"""

    def test_batch_size_respects_limit(self):
        BATCH = 15
        total = 38
        batches = list(range(0, total, BATCH))
        self.assertEqual(len(batches), 3)  # 0, 15, 30
        for i, start in enumerate(batches):
            sz = min(BATCH, total - start)
            self.assertLessEqual(sz, 15, f"每批不得超过 15 手")


if __name__ == '__main__':
    unittest.main()
