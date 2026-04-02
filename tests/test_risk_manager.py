"""
src/risk/risk_manager.py 单元测试
覆盖 RiskManager 的 check_signal、update_daily_loss、reset_daily，避免 0% 覆盖。
"""
import unittest
import sys
from src.risk import RiskManager


class TestRiskManager(unittest.TestCase):
    """RiskManager 基本行为"""

    def test_init_default_config(self):
        """默认 config 包含 max_loss_per_trade, max_daily_loss, max_position_size"""
        rm = RiskManager()
        self.assertIn('max_loss_per_trade', rm.config)
        self.assertIn('max_daily_loss', rm.config)
        self.assertIn('max_position_size', rm.config)
        self.assertEqual(rm.daily_loss, 0.0)

    def test_init_custom_config(self):
        """可传入自定义 config"""
        cfg = {'max_position_size': 0.5}
        rm = RiskManager(config=cfg)
        self.assertEqual(rm.config['max_position_size'], 0.5)

    def test_check_signal_hold_always_ok(self):
        """HOLD 信号始终通过"""
        rm = RiskManager()
        self.assertTrue(rm.check_signal({'action': 'HOLD'}))

    def test_check_signal_position_over_limit_rejected(self):
        """position_size 超过 max_position_size 时拒绝"""
        rm = RiskManager()
        rm.config['max_position_size'] = 0.2
        signal = {'action': 'BUY', 'position_size': 0.5}
        self.assertFalse(rm.check_signal(signal, account_value=10000))

    def test_check_signal_position_under_limit_ok(self):
        """position_size 未超限时通过"""
        rm = RiskManager()
        signal = {'action': 'BUY', 'position_size': 0.1}
        self.assertTrue(rm.check_signal(signal, account_value=10000))

    def test_check_signal_daily_loss_over_limit_rejected(self):
        """日内亏损达到 max_daily_loss * account_value 时拒绝"""
        rm = RiskManager()
        rm.config['max_daily_loss'] = 0.05
        rm.daily_loss = 600.0  # 超过 10000 * 0.05 = 500
        signal = {'action': 'BUY'}
        self.assertFalse(rm.check_signal(signal, account_value=10000))

    def test_check_signal_daily_loss_under_limit_ok(self):
        """日内亏损未达限时通过"""
        rm = RiskManager()
        rm.daily_loss = 100.0
        self.assertTrue(rm.check_signal({'action': 'SELL'}, account_value=10000))

    def test_update_daily_loss(self):
        """update_daily_loss 累加"""
        rm = RiskManager()
        rm.update_daily_loss(100.0)
        self.assertEqual(rm.daily_loss, 100.0)
        rm.update_daily_loss(50.0)
        self.assertEqual(rm.daily_loss, 150.0)

    def test_reset_daily(self):
        """reset_daily 将 daily_loss 置 0"""
        rm = RiskManager()
        rm.daily_loss = 200.0
        rm.reset_daily()
        self.assertEqual(rm.daily_loss, 0.0)
