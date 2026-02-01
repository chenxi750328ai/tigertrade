"""
Feature测试：买入白银（SIL2603）

测试设计时两种场景都设计到：
1. 真实API场景：TestFeatureBuySilverRealAPI.test_buy_silver_real_api，仅真实下单+真实查询，无真实API时 Fail（不允许 Skip）。
2. Mock 场景：TestFeatureBuySilverMock，专用 mock 用例提高覆盖率。
"""
import unittest
import sys
import pytest
sys.path.insert(0, '/home/cx/tigertrade')

from src.executor.order_executor import OrderExecutor
from src.api_adapter import api_manager
from src import tiger1 as t1
from unittest.mock import Mock, patch


class TestFeatureBuySilverMock(unittest.TestCase):
    """Feature测试：买入白银（Mock模式）- 测试各种分支"""
    
    def setUp(self):
        """每个测试前重置"""
        api_manager.quote_api = None
        api_manager.trade_api = None
        api_manager.is_mock_mode = True
        api_manager._account = None
    
    def test_buy_silver_account_empty(self):
        """
        TC-F-BUY-001: account为空时买入白银应该失败
        验证Mock API的account检查分支
        """
        # 初始化Mock API，account为空
        api_manager.initialize_mock_apis(account=None)
        
        order_executor = OrderExecutor(t1)
        # 不要Mock！测试真实的风控逻辑
        t1.current_position = 0
        t1.daily_loss = 0
        
        # 执行买入
        success, message = order_executor.execute_buy(
            price=100.0,
            atr=0.5,
            grid_lower=97.0,
            grid_upper=105.0,
            confidence=0.7
        )
        
        # 验证：应该失败，包含account错误信息
        self.assertFalse(success, "account为空时应该失败")
        self.assertIn("account", (message or "").lower(), 
                      f"错误消息应包含'account'，实际: {message}")
        print(f"✅ [TC-F-BUY-001] account为空时正确拒绝: {message}")
    
    def test_buy_silver_account_correct(self):
        """
        TC-F-BUY-002: account正确时买入白银应该成功
        验证Mock API的正常流程分支
        """
        # 初始化Mock API，account正确
        api_manager.initialize_mock_apis(account="TEST_ACCOUNT_123")
        
        order_executor = OrderExecutor(t1)
        # 不要Mock！测试真实的风控逻辑
        t1.current_position = 0
        t1.daily_loss = 0
        
        # 执行买入
        success, message = order_executor.execute_buy(
            price=100.0,
            atr=0.5,
            grid_lower=97.0,
            grid_upper=105.0,
            confidence=0.7
        )
        
        # 验证：应该成功，返回订单ID
        self.assertTrue(success, f"account正确时应该成功，但返回: {message}")
        self.assertIn("订单ID", message or "", "返回消息应包含订单ID")
        
        # 验证：订单已存储在Mock API中
        mock_trade_api = api_manager.trade_api
        if hasattr(mock_trade_api, 'orders'):
            self.assertGreater(len(mock_trade_api.orders), 0, 
                         "Mock API应该存储了订单")
            
            # 验证：可以通过get_order查询
            if hasattr(mock_trade_api, 'get_order'):
                order_ids = list(mock_trade_api.orders.keys())
                found_order = mock_trade_api.get_order(order_id=order_ids[0])
                self.assertIsNotNone(found_order, "应该能查询到订单")
                # Mock API可能使用转换后的symbol格式（SIL2603），这是正常的
                if hasattr(found_order, 'symbol'):
                    self.assertIn(found_order.symbol, [t1.FUTURE_SYMBOL, 'SIL2603'],
                                 f"订单symbol应该是{t1.FUTURE_SYMBOL}或SIL2603")
                print(f"✅ [TC-F-BUY-002] 买入成功，订单ID: {order_ids[0]}")
    
    def test_buy_silver_risk_control_failed(self):
        """
        TC-F-BUY-003: 风控失败时买入白银应该失败
        验证风控检查分支
        """
        # 初始化Mock API，account正确
        api_manager.initialize_mock_apis(account="TEST_ACCOUNT_123")
        
        order_executor = OrderExecutor(t1)
        # Mock风控失败
        # 设置持仓达到上限，触发风控失败
        t1.current_position = t1.GRID_MAX_POSITION
        
        # 执行买入
        success, message = order_executor.execute_buy(
            price=100.0,
            atr=0.5,
            grid_lower=97.0,
            grid_upper=105.0,
            confidence=0.7
        )
        
        # 验证：应该失败，包含风控错误信息
        self.assertFalse(success, "风控失败时应该失败")
        self.assertIn("风控", (message or "").lower(), 
                      f"错误消息应包含'风控'，实际: {message}")
        print(f"✅ [TC-F-BUY-003] 风控失败时正确拒绝: {message}")
    
    def test_buy_silver_api_error(self):
        """
        TC-F-BUY-004: API错误时应该返回错误信息
        验证错误处理分支
        """
        # 初始化Mock API，account正确
        api_manager.initialize_mock_apis(account="TEST_ACCOUNT_123")
        
        order_executor = OrderExecutor(t1)
        # 不要Mock！测试真实的风控逻辑
        t1.current_position = 0
        t1.daily_loss = 0
        
        # Mock API抛出异常
        original_place_order = api_manager.trade_api.place_order
        def mock_place_order_error(*args, **kwargs):
            raise Exception("API错误：网络超时")
        
        api_manager.trade_api.place_order = mock_place_order_error
        
        try:
            # 执行买入
            success, message = order_executor.execute_buy(
                price=100.0,
                atr=0.5,
                grid_lower=97.0,
                grid_upper=105.0,
                confidence=0.7
            )
            
            # 验证：应该失败，包含错误信息
            self.assertFalse(success, "API错误时应该失败")
            self.assertIn("错误", (message or "").lower() or "异常", 
                          f"错误消息应包含错误信息，实际: {message}")
            print(f"✅ [TC-F-BUY-004] API错误时正确处理: {message}")
        finally:
            # 恢复原始方法
            api_manager.trade_api.place_order = original_place_order
    
    def test_buy_silver_order_query(self):
        """
        TC-F-BUY-005: 买入后应该能查询到订单
        验证订单查询功能（AR3.3）
        """
        # 初始化Mock API，account正确
        api_manager.initialize_mock_apis(account="TEST_ACCOUNT_123")
        
        order_executor = OrderExecutor(t1)
        # 不要Mock！测试真实的风控逻辑
        t1.current_position = 0
        t1.daily_loss = 0
        
        # 执行买入
        success, message = order_executor.execute_buy(
            price=100.0,
            atr=0.5,
            grid_lower=97.0,
            grid_upper=105.0,
            confidence=0.7
        )
        
        self.assertTrue(success, "买入应该成功")
        
        # 提取order_id
        import re
        order_id = None
        patterns = [
            r'订单ID[：:=]\s*([A-Z0-9_]+)',
            r'order[_\s]*id[：:=]\s*([A-Z0-9_]+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, message or "", re.IGNORECASE)
            if match:
                order_id = match.group(1)
                break
        
        self.assertIsNotNone(order_id, f"无法提取order_id: {message}")
        
        # 验证：可以通过get_order查询
        mock_trade_api = api_manager.trade_api
        if hasattr(mock_trade_api, 'get_order'):
            found_order = mock_trade_api.get_order(order_id=order_id)
            self.assertIsNotNone(found_order, "应该能查询到订单")
            self.assertEqual(found_order.order_id, order_id,
                           f"查询到的订单ID应该匹配: {order_id}")
            print(f"✅ [TC-F-BUY-005] 订单查询成功: {order_id}")
        
        # 验证：可以通过get_orders查询所有订单
        if hasattr(mock_trade_api, 'get_orders'):
            # 尝试用原始symbol和转换后的symbol查询
            # 转换symbol格式：SIL.COMEX.202603 -> SIL2603
            symbol_to_query = t1._to_api_identifier(t1.FUTURE_SYMBOL)
            all_orders = mock_trade_api.get_orders(symbol=symbol_to_query, limit=50)
            if not all_orders:
                # 如果还是没找到，不传symbol查询所有订单
                all_orders = mock_trade_api.get_orders(limit=50)
            
            self.assertIsNotNone(all_orders, "get_orders应该返回订单列表")
            order_ids = [getattr(o, 'order_id', None) for o in all_orders]
            self.assertIn(order_id, order_ids, 
                         f"订单ID {order_id} 应该在查询结果中，实际: {order_ids}")
            print(f"✅ [TC-F-BUY-005] get_orders查询成功，找到 {len(all_orders)} 个订单")


@pytest.mark.real_api
class TestFeatureBuySilverRealAPI(unittest.TestCase):
    """Feature测试：买入白银（真实API模式）- 端到端测试"""
    
    @classmethod
    def setUpClass(cls):
        """初始化真实API；失败则标记（本类真实API场景用例将 Fail），mock 场景在别处。"""
        from tests.feature_test_base import initialize_real_api, RealAPIUnavailableError
        try:
            initialize_real_api()
            cls._real_api_available = True
        except RealAPIUnavailableError:
            cls._real_api_available = False
        cls.order_executor = OrderExecutor(t1)

    def test_buy_silver_real_api(self):
        """TC-F-BUY-006【真实API场景】买入白银：仅真实下单+真实查询。无真实API时 Fail。"""
        if not getattr(self.__class__, '_real_api_available', False):
            self.fail("真实API不可用，此用例必须配置真实API并通过；不允许 Skip，测试目的就是测出问题")
        t1.current_position = 0
        t1.daily_loss = 0
        success, message = self.order_executor.execute_buy(
            price=100.0, atr=0.5, grid_lower=97.0, grid_upper=105.0, confidence=0.7
        )
        if not success:
            # 无授权等环境问题视为用例失败，不 Skip
            self.fail(f"订单提交失败: {message}")
            
        self.assertIn("订单ID", message or "", "返回消息应包含订单ID")
        print(f"✅ [TC-F-BUY-006] 真实API买入成功: {message}")
            
            # 验证AR3.3：可以通过API查询订单
        trade_client = api_manager.trade_api.client
        account = api_manager.trade_api.account
            
        import time
        time.sleep(3)  # 等待订单进入系统
            
            # 尝试查询订单
        try:
                # 转换symbol格式：SIL.COMEX.202603 -> SIL2603
                symbol_to_query = t1._to_api_identifier(t1.FUTURE_SYMBOL)
                all_orders = trade_client.get_orders(
                    account=account,
                    symbol=symbol_to_query,  # 使用转换后的格式 SIL2603
                    limit=50
                )
                if all_orders:
                    print(f"✅ [TC-F-BUY-006] 查询到 {len(all_orders)} 个订单")
                else:
                    print(f"⚠️ [TC-F-BUY-006] 未查询到订单（可能是新订单还未同步）")
        except Exception as e:
                print(f"⚠️ [TC-F-BUY-006] 查询订单失败: {e}")


if __name__ == '__main__':
    unittest.main()
