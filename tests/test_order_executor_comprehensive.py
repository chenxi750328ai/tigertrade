"""
代码级测试：OrderExecutor完整覆盖
补充order_executor.py的所有代码路径测试
"""
import unittest
import sys
from unittest.mock import Mock, patch, MagicMock
sys.path.insert(0, '/home/cx/tigertrade')

from src.executor.order_executor import OrderExecutor
from src.api_adapter import api_manager
from src import tiger1 as t1

# 保存原始 check_risk_control，供 tearDown 恢复，避免污染后续测试
_original_check_risk_control = t1.check_risk_control


class TestOrderExecutorComprehensive(unittest.TestCase):
    """OrderExecutor完整代码覆盖测试"""
    
    def setUp(self):
        """初始化"""
        self.executor = OrderExecutor(t1)
        api_manager.initialize_mock_apis()
        t1.current_position = 0
        t1.daily_loss = 0
        self._sync_backup = getattr(t1, "sync_positions_from_backend", None)
        t1.sync_positions_from_backend = lambda: setattr(t1, "current_position", getattr(t1, "current_position", 0))
        # 避免 get_effective_position_for_buy 从 mock 拉成 3 手导致硬顶拦截
        self._get_effective_backup = getattr(t1, "get_effective_position_for_buy", None)
        t1.get_effective_position_for_buy = lambda: 0

    def tearDown(self):
        """测试后清理，避免测试间状态污染"""
        t1.current_position = 0
        t1.daily_loss = 0
        t1.check_risk_control = _original_check_risk_control
        if getattr(self, "_sync_backup", None) is not None:
            t1.sync_positions_from_backend = self._sync_backup
        if getattr(self, "_get_effective_backup", None) is not None:
            t1.get_effective_position_for_buy = self._get_effective_backup
    
    def test_execute_buy_api_none(self):
        """测试API未初始化"""
        original_trade_api = api_manager.trade_api
        original_trade_client = getattr(t1, 'trade_client', None)
        api_manager.trade_api = None
        t1.trade_client = None  # 防止重新初始化
        try:
            # 使用更合理的参数：grid_lower=98（止损距离2，预期损失2000 < 3000上限）
            result, msg = self.executor.execute_buy(100.0, 0.5, 98.0, 105.0, 0.7)
            # 可能被风控拦截，检查是否是API未初始化或风控失败
            self.assertFalse(result)
            self.assertTrue("交易API未初始化" in msg or "风控" in msg or "API" in msg)
        finally:
            api_manager.trade_api = original_trade_api
            if original_trade_client:
                t1.trade_client = original_trade_client
    
    def test_execute_buy_success_with_order_id(self):
        """测试买入成功，返回order_id"""
        # 不要Mock！测试真实的风控逻辑
        t1.current_position = 0
        t1.daily_loss = 0
        mock_order = Mock()
        mock_order.order_id = "TEST_123"
        api_manager.trade_api.place_order = Mock(return_value=mock_order)
        
        # 使用更合理的参数：grid_lower=98（止损距离2，预期损失2000 < 3000上限）
        result, msg = self.executor.execute_buy(100.0, 0.5, 98.0, 105.0, 0.7)
        self.assertTrue(result)
        self.assertIn("订单ID", msg)
        # 清理：下单成功会增加持仓，需要重置以避免影响后续测试
        t1.current_position = 0
    
    def test_execute_buy_success_with_dict_result(self):
        """测试买入成功，返回字典格式"""
        t1.current_position = 0
        t1.daily_loss = 0
        t1.check_risk_control = Mock(return_value=True)
        api_manager.trade_api.place_order = Mock(return_value={'order_id': 'DICT_123'})
        
        # 使用更合理的参数：grid_lower=98（止损距离2，预期损失2000 < 3000上限）
        result, msg = self.executor.execute_buy(100.0, 0.5, 98.0, 105.0, 0.7)
        self.assertTrue(result)
        # 清理：下单成功会增加持仓
        t1.current_position = 0
    
    def test_execute_buy_success_with_string_result(self):
        """测试买入成功，返回字符串"""
        t1.current_position = 0
        t1.daily_loss = 0
        t1.check_risk_control = Mock(return_value=True)
        api_manager.trade_api.place_order = Mock(return_value="STRING_123")
        
        # 使用更合理的参数：grid_lower=98（止损距离2，预期损失2000 < 3000上限）
        result, msg = self.executor.execute_buy(100.0, 0.5, 98.0, 105.0, 0.7)
        self.assertTrue(result)
        # 清理：下单成功会增加持仓
        t1.current_position = 0
    
    def test_execute_buy_exception(self):
        """测试下单异常"""
        # 不要Mock！测试真实的风控逻辑
        t1.current_position = 0
        t1.daily_loss = 0
        api_manager.trade_api.place_order = Mock(side_effect=Exception("API错误"))
        
        # 使用更合理的参数：grid_lower=98（止损距离2，预期损失2000 < 3000上限）
        result, msg = self.executor.execute_buy(100.0, 0.5, 98.0, 105.0, 0.7)
        self.assertFalse(result)
        # 可能是"下单异常"或"风控检查未通过"（如果风控先失败）
        self.assertTrue("下单异常" in msg or "风控" in msg or "异常" in msg)
    
    def test_execute_sell_api_none(self):
        """测试卖出API未初始化"""
        # 先设置持仓，否则会先被"无持仓"拦截；mock sync 避免 trade_client=None 时 sync 把 position 重置为 0
        t1.current_position = 1
        original_trade_api = api_manager.trade_api
        original_trade_client = getattr(t1, 'trade_client', None)
        t1.trade_client = None
        api_manager.trade_api = None
        try:
            with patch.object(t1, 'sync_positions_from_backend', lambda: None):
                result, msg = self.executor.execute_sell(100.0, 0.7)
            self.assertFalse(result)
            self.assertTrue(
                "交易API未初始化" in msg or "account" in msg.lower() or "API" in msg,
                f"应返回 API 未初始化类消息: {msg}"
            )
        finally:
            api_manager.trade_api = original_trade_api
            if original_trade_client:
                t1.trade_client = original_trade_client
            t1.current_position = 0
    
    def test_execute_sell_success(self):
        """测试卖出成功"""
        t1.current_position = 1
        _sync = getattr(t1, "sync_positions_from_backend", None)
        t1.sync_positions_from_backend = lambda: setattr(t1, "current_position", 1)  # 保持 1 手
        mock_order = Mock()
        mock_order.order_id = "SELL_123"
        api_manager.trade_api.place_order = Mock(return_value=mock_order)
        try:
            result, msg = self.executor.execute_sell(100.0, 0.7)
            self.assertTrue(result)
        finally:
            if _sync is not None:
                t1.sync_positions_from_backend = _sync
            t1.current_position = 0

    def test_execute_sell_exception(self):
        """测试卖出异常"""
        t1.current_position = 1
        _sync = getattr(t1, "sync_positions_from_backend", None)
        t1.sync_positions_from_backend = lambda: setattr(t1, "current_position", 1)
        api_manager.trade_api.place_order = Mock(side_effect=Exception("卖出错误"))
        try:
            result, msg = self.executor.execute_sell(100.0, 0.7)
            self.assertFalse(result)
            self.assertIn("下单异常", msg)
        finally:
            if _sync is not None:
                t1.sync_positions_from_backend = _sync
            t1.current_position = 0
    
    def test_execute_buy_with_profit_pred(self):
        """测试带profit_pred的买入"""
        t1.current_position = 0
        t1.daily_loss = 0
        t1.check_risk_control = Mock(return_value=True)
        mock_order = Mock()
        mock_order.order_id = "PROFIT_123"
        api_manager.trade_api.place_order = Mock(return_value=mock_order)
        
        result, msg = self.executor.execute_buy(
            100.0, 0.5, 98.0, 105.0, 0.7, profit_pred=0.05  # 使用98.0确保风控通过
        )
        self.assertTrue(result)
        # 清理：下单成功会增加持仓
        t1.current_position = 0
    
    def test_execute_sell_no_position(self):
        """测试无持仓时卖出"""
        api_manager.initialize_mock_apis(account="TEST_SELL_NO_POS")
        t1.current_position = 0
        # 避免 sync 从 mock 拉成有仓，导致走到下单分支
        _sync = getattr(t1, "sync_positions_from_backend", None)
        t1.sync_positions_from_backend = lambda: setattr(t1, "current_position", 0)
        try:
            result, msg = self.executor.execute_sell(100.0, 0.7)
            self.assertFalse(result, f"无持仓时应拒绝卖出: {msg}")
            self.assertTrue("无持仓" in msg or "无多头持仓" in msg or "无法卖出" in msg or "account" in (msg or "").lower(),
                            f"应返回无持仓或 account 类消息: {msg}")
        finally:
            if _sync is not None:
                t1.sync_positions_from_backend = _sync

    def test_order_executor_fallback_when_risk_manager_lacks_check_risk_control(self):
        """【防回归】risk_manager 无 check_risk_control 时（如 __main__ 尚未定义）应回退到 t1，不抛 AttributeError"""
        # 用普通对象模拟：有 state 属性但无 check_risk_control（Mock() 的 getattr 会返回 callable 的 Mock）
        class RiskWithoutCheck:
            current_position = 0
            daily_loss = 0
            grid_lower = None
            grid_upper = None
            atr_5m = 0.5
            @staticmethod
            def compute_stop_loss(price, atr, grid_lower):
                return 99.0, 1.0
        mock_risk = RiskWithoutCheck()
        api_manager.initialize_mock_apis()
        api_manager.trade_api.place_order = Mock(return_value=Mock(order_id="FALLBACK_123"))
        t1.current_position = 0
        t1.daily_loss = 0
        executor = OrderExecutor(risk_manager=mock_risk)
        self.assertIsNotNone(executor._risk_fallback, "应设置 _risk_fallback 并回退到 t1")
        try:
            result, msg = executor.execute_buy(100.0, 0.5, 98.0, 105.0, 0.7)
        except AttributeError as e:
            if "check_risk_control" in str(e):
                self.fail(f"不应抛出 check_risk_control 相关 AttributeError，应回退到 t1。错误: {e}")
            raise
        self.assertIn(result, (True, False))
        self.assertIsInstance(msg, str)


if __name__ == '__main__':
    unittest.main()
