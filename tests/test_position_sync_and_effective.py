"""
持仓同步与有效仓位计算单测

覆盖此前未测到的逻辑（导致 52/62/74/81 手持仓超买）：
1. Tiger Position 结构：symbol 在 contract.symbol，非 p.symbol
2. get_positions 必须传 sec_type=FUT，默认 STK 会过滤掉期货
3. sync_positions_from_backend 与 get_effective_position_for_buy 的解析一致性
"""
import unittest
import sys
sys.path.insert(0, '/home/cx/tigertrade')

from unittest.mock import patch, MagicMock
from src import tiger1 as t1


def _make_tiger_position(symbol='SIL2603', quantity=81):
    """构造与 Tiger Position 结构一致的对象：symbol 在 contract 内"""
    contract = MagicMock()
    contract.symbol = symbol
    pos = MagicMock()
    pos.contract = contract
    pos.quantity = quantity
    pos.symbol = None  # 真实 Position 无此属性，模拟
    return pos


def _make_tiger_position_dict(symbol='SIL2603', quantity=81):
    """dict 格式的持仓（兼容解析逻辑）"""
    return {'symbol': symbol, 'quantity': quantity}


class TestPositionParsing(unittest.TestCase):
    """Position 解析逻辑单测，使用 Tiger 真实结构"""

    def setUp(self):
        self.orig_trade_client = t1.trade_client
        self.orig_client_config = t1.client_config
        self.orig_current_position = t1.current_position

    def tearDown(self):
        t1.trade_client = self.orig_trade_client
        t1.client_config = self.orig_client_config
        t1.current_position = self.orig_current_position

    def test_get_effective_position_parses_contract_symbol(self):
        """Tiger Position 的 symbol 在 contract.symbol，必须正确解析"""
        mock_tc = MagicMock()
        mock_tc.get_positions = MagicMock(return_value=[
            _make_tiger_position('SIL2603', 81),
        ])
        mock_tc.get_orders = MagicMock(return_value=[])
        mock_cfg = MagicMock()
        mock_cfg.account = 'TEST_ACC'

        t1.trade_client = mock_tc
        t1.client_config = mock_cfg

        try:
            from tigeropen.common.consts import SecurityType
            expected_sec_type = SecurityType.FUT
        except ImportError:
            expected_sec_type = 'FUT'

        with patch.object(t1, 'client_config', mock_cfg), \
             patch.object(t1, 'trade_client', mock_tc):
            result = t1.get_effective_position_for_buy()

        self.assertEqual(result, 81, "应正确解析 contract.symbol 持仓 81 手")
        # 验证调用了 sec_type=FUT
        mock_tc.get_positions.assert_called_once()
        call_kw = mock_tc.get_positions.call_args[1]
        self.assertIn('sec_type', call_kw, "get_positions 必须传 sec_type")
        self.assertEqual(call_kw['sec_type'], expected_sec_type,
                         "期货持仓必须 sec_type=FUT，否则过滤掉导致 pos=0")

    def test_get_effective_position_dict_format(self):
        """兼容 dict 格式持仓"""
        mock_tc = MagicMock()
        mock_tc.get_positions = MagicMock(return_value=[
            _make_tiger_position_dict('SIL2603', 5),
        ])
        mock_tc.get_orders = MagicMock(return_value=[])
        mock_cfg = MagicMock()
        mock_cfg.account = 'TEST_ACC'

        t1.trade_client = mock_tc
        t1.client_config = mock_cfg

        with patch.object(t1, 'client_config', mock_cfg), \
             patch.object(t1, 'trade_client', mock_tc):
            result = t1.get_effective_position_for_buy()

        self.assertEqual(result, 5, "应正确解析 dict 格式持仓")

    def test_sync_positions_parses_contract_symbol(self):
        """sync_positions_from_backend 同样必须正确解析 contract.symbol"""
        mock_tc = MagicMock()
        mock_tc.get_positions = MagicMock(return_value=[
            _make_tiger_position('SIL2603', 82),
        ])
        mock_cfg = MagicMock()
        mock_cfg.account = 'TEST_ACC'

        t1.trade_client = mock_tc
        t1.client_config = mock_cfg
        t1.current_position = 0

        with patch.object(t1, 'client_config', mock_cfg), \
             patch.object(t1, 'trade_client', mock_tc):
            t1.sync_positions_from_backend()

        self.assertEqual(t1.current_position, 82,
                         "sync 后 current_position 应为 82（从 contract.symbol 解析）")

    def test_get_effective_position_filters_non_sil(self):
        """只统计 SIL 相关持仓，其他品种不计入"""
        mock_tc = MagicMock()
        mock_tc.get_positions = MagicMock(return_value=[
            _make_tiger_position('SIL2603', 3),
            _make_tiger_position('CL2603', 10),  # 原油，应忽略
        ])
        mock_tc.get_orders = MagicMock(return_value=[])

        mock_cfg = MagicMock()
        mock_cfg.account = 'TEST_ACC'
        t1.trade_client = mock_tc
        t1.client_config = mock_cfg

        with patch.object(t1, 'client_config', mock_cfg), \
             patch.object(t1, 'trade_client', mock_tc):
            result = t1.get_effective_position_for_buy()

        self.assertEqual(result, 3, "只应计入 SIL2603 的 3 手")

    def test_get_effective_position_returns_999_on_exception(self):
        """异常时保守拒绝，返回 999"""
        mock_tc = MagicMock()
        mock_tc.get_positions = MagicMock(side_effect=Exception("API error"))
        mock_cfg = MagicMock()
        mock_cfg.account = 'TEST_ACC'

        t1.trade_client = mock_tc
        t1.client_config = mock_cfg

        with patch.object(t1, 'client_config', mock_cfg), \
             patch.object(t1, 'trade_client', mock_tc):
            result = t1.get_effective_position_for_buy()

        self.assertEqual(result, 999, "异常时应返回 999 保守拒绝")


if __name__ == '__main__':
    unittest.main()
