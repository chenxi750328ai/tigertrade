"""
account传递端到端测试 - 必须能发现account为空的问题
"""
import os
import unittest
import sys
sys.path.insert(0, '/home/cx/tigertrade')

from unittest.mock import Mock, patch, MagicMock
from tigeropen.tiger_open_config import TigerOpenClientConfig
from tigeropen.quote.quote_client import QuoteClient
from tigeropen.trade.trade_client import TradeClient
from src.api_adapter import api_manager
from src.executor.order_executor import OrderExecutor
from src import tiger1 as t1


def _config_has_account():
    """当 openapicfg_dem 存在且配置了 account 时返回 True，否则跳过依赖配置的用例。"""
    try:
        path = os.path.join(os.path.dirname(__file__), '..', 'openapicfg_dem')
        if not os.path.exists(path):
            return False
        cfg = TigerOpenClientConfig(props_path=path)
        acc = getattr(cfg, 'account', None)
        return acc is not None and str(acc).strip() != ''
    except Exception:
        return False


class TestAccount传递端到端(unittest.TestCase):
    """account传递的端到端测试 - 必须能发现account为空的问题"""
    
    def setUp(self):
        """每个测试前重置api_manager"""
        api_manager.quote_api = None
        api_manager.trade_api = None
        api_manager.is_mock_mode = True
        api_manager._account = None
    
    @unittest.skipUnless(_config_has_account(), "需要 openapicfg_dem 且配置 account 才能运行（CI 无配置时跳过）")
    def test_account_从配置传递到下单(self):
        """测试1: account从配置文件传递到下单的完整流程"""
        # 1. 加载配置
        client_config = TigerOpenClientConfig(props_path='./openapicfg_dem')
        account_from_config = client_config.account
        
        # 验证配置中有account（skipUnless 已保证有值，此处双保险）
        self.assertIsNotNone(account_from_config, "配置文件必须包含account")
        self.assertNotEqual(account_from_config, "", "account不能为空字符串")
        print(f"✅ 配置中的account: {account_from_config}")
        
        # 2. 创建客户端（使用Mock避免真实网络调用）
        quote_client = Mock(spec=QuoteClient)
        trade_client = Mock(spec=TradeClient)
        trade_client.config = Mock()
        trade_client.config.account = account_from_config
        
        # 3. 初始化api_manager
        api_manager.initialize_real_apis(quote_client, trade_client, account=account_from_config)
        
        # 验证account已设置
        self.assertEqual(api_manager._account, account_from_config, 
                        f"api_manager._account应该是{account_from_config}，实际是{api_manager._account}")
        self.assertEqual(api_manager.trade_api.account, account_from_config,
                        f"trade_api.account应该是{account_from_config}，实际是{api_manager.trade_api.account}")
        print(f"✅ api_manager._account = {api_manager._account}")
        print(f"✅ trade_api.account = {api_manager.trade_api.account}")
        
        # 4. 创建OrderExecutor
        order_executor = OrderExecutor(t1)
        
        # 5. 不要直接Mock place_order，让MockTradeApiAdapter.place_order真正执行
        # 这样account检查才会生效，各种分支才能被覆盖
        # Mock风控通过
        # 不要Mock！测试真实的风控逻辑
        t1.current_position = 0
        t1.daily_loss = 0
        # 避免 execute_buy 内从后台同步持仓导致风控拒绝（测试仅验证 account 传递）
        with patch.object(t1, 'sync_positions_from_backend', return_value=None), \
             patch.object(t1, 'get_effective_position_for_buy', return_value=0):
            # 6. 执行下单（使用真实的MockTradeApiAdapter.place_order）
            # 参数需满足风控：单笔预期损失<=3000。price-grid_lower<=3 时止损距离<=3，预期损失<=3000
            success, message = order_executor.execute_buy(
                price=100.0,
                atr=0.5,
                grid_lower=97.0,
                grid_upper=105.0,
                confidence=0.7
            )
        
        # 7. 验证下单成功（如果account正确，应该成功）
        self.assertTrue(success, f"account正确时下单应该成功，但返回: {message}")
        
        # 8. 验证订单已创建并可通过Mock API查询（验证AR3.3）
        mock_trade_api = api_manager.trade_api
        if hasattr(mock_trade_api, 'orders') and mock_trade_api.orders:
            # 验证订单已存储在Mock API中
            order_ids = list(mock_trade_api.orders.keys())
            self.assertGreater(len(order_ids), 0, "Mock API应该存储了订单")
            print(f"✅ Mock API中存储了 {len(order_ids)} 个订单")
            
            # 验证可以通过get_order查询
            if hasattr(mock_trade_api, 'get_order'):
                found_order = mock_trade_api.get_order(order_id=order_ids[0])
                self.assertIsNotNone(found_order, "应该能通过get_order查询到订单")
                print(f"✅ 通过get_order查询到订单: {found_order.order_id}")
        
        print(f"✅ 端到端测试通过: account从配置传递到下单成功")
    
    def test_account为空时下单失败(self):
        """测试2: account为空时下单必须失败"""
        # 1. 使用Mock API，但account为空
        api_manager.initialize_mock_apis(account=None)
        
        # 验证account确实为空
        self.assertIsNone(api_manager._account, "account应该为空")
        self.assertIsNone(api_manager.trade_api.account, "trade_api.account应该为空")
        
        # 2. 创建OrderExecutor
        order_executor = OrderExecutor(t1)
        
        # Mock风控通过（只测试account传递，不测试风控）
        t1.current_position = 0
        t1.daily_loss = 0
        # 避免从后台同步持仓导致先被硬顶拒绝（需走到 account 检查）
        with patch.object(t1, 'sync_positions_from_backend', return_value=None), \
             patch.object(t1, 'get_effective_position_for_buy', return_value=0):
            # 3. 执行下单（grid_lower=97 使风控通过，才能走到 account 检查）
            success, message = order_executor.execute_buy(
                price=100.0,
                atr=0.5,
                grid_lower=97.0,
                grid_upper=105.0,
                confidence=0.7
            )
        
        # 4. 验证下单失败（account为空时必须失败）
        self.assertFalse(success, f"account为空时下单应该失败，但返回success={success}, message={message}")
        self.assertIn("account", (message or "").lower(), 
                      f"错误消息应包含'account'，实际: {message}")
        print(f"✅ account为空时正确拒绝下单: {message}")
    
    def test_account从api_manager获取(self):
        """测试3: place_order中从api_manager获取account的fallback逻辑"""
        # 1. 设置api_manager._account
        api_manager._account = "TEST_ACCOUNT_123"
        
        # 2. 创建trade_api但account为空
        quote_client = Mock(spec=QuoteClient)
        trade_client = Mock(spec=TradeClient)
        trade_client.config = Mock()
        trade_client.config.account = None
        
        api_manager.initialize_real_apis(quote_client, trade_client, account=None)
        
        # 3. 手动设置trade_api.account为空（模拟初始化时未设置）
        api_manager.trade_api.account = None
        
        # 4. 创建OrderExecutor并执行下单
        order_executor = OrderExecutor(t1)
        t1.current_position = 0
        t1.daily_loss = 0
        
        mock_order = Mock()
        mock_order.order_id = "TEST_ORDER_456"
        api_manager.trade_api.place_order = Mock(return_value=mock_order)
        
        with patch.object(t1, 'sync_positions_from_backend', return_value=None), \
             patch.object(t1, 'get_effective_position_for_buy', return_value=0):
            # grid_lower=97 使风控通过，才能走到 place_order
            success, message = order_executor.execute_buy(
                price=100.0,
                atr=0.5,
                grid_lower=97.0,
                grid_upper=105.0,
                confidence=0.7
            )
        
        # 5. 验证fallback逻辑工作（应该从api_manager._account获取）
        if success:
            print(f"✅ fallback逻辑工作: 从api_manager._account获取account成功")
        else:
            if "account" in (message or "").lower():
                self.fail(f"fallback逻辑未工作: {message}")
        
        # 验证place_order被调用
        api_manager.trade_api.place_order.assert_called_once()

    @unittest.skipIf(_config_has_account(), "openapicfg_dem has account so _account gets filled")
    def test_仅trade_client_config_account为None时account为空(self):
        """
        测试4（订单问题回归）: 仅用 trade_client.config.account 初始化时 account 为空。
        生产 bug：tiger1 模块加载时用 getattr(trade_client.config, 'account', None) 得到 None，
        导致 api_manager._account 为空、下单失败或后台看不到订单。
        """
        quote_client = Mock(spec=QuoteClient)
        quote_client.config = Mock()
        quote_client.config.account = None
        trade_client = Mock(spec=TradeClient)
        trade_client.config = Mock()
        trade_client.config.account = None  # 模拟 SDK 的 config 上无 account

        # 本机有 openapicfg_dem 且含 account 时，initialize_real_apis 会回填，_account 非 None，故跳过
        if _config_has_account():
            self.skipTest("本机有 openapicfg_dem 且含 account，无法验证「无回填时 _account 为 None」")
        with patch.dict('os.environ', {}, clear=False):
            with patch('os.getenv', return_value=None):
                api_manager.initialize_real_apis(quote_client, trade_client, account=None)
        self.assertIsNone(
            api_manager._account,
            "仅 trade_client.config.account 为 None 且不传 account 时，api_manager._account 应为 None"
        )
        print("✅ [回归] 仅 trade_client.config 时 account 为空，符合预期")

    @unittest.skipUnless(_config_has_account(), "需要 openapicfg_dem 且配置 account 才能运行（CI 无配置时跳过）")
    def test_必须用client_config_account初始化才能保证有值(self):
        """
        测试5（订单问题回归）: 必须显式传入 client_config.account，才能保证 api_manager 有 account。
        防止 tiger1 主入口只传 trade_client.config.account（可能为 None）导致订单问题。
        """
        client_config = TigerOpenClientConfig(props_path='./openapicfg_dem')
        expected_account = getattr(client_config, 'account', None)
        self.assertIsNotNone(expected_account, "openapicfg_dem 中应有 account")

        quote_client = Mock(spec=QuoteClient)
        trade_client = Mock(spec=TradeClient)
        trade_client.config = Mock()
        trade_client.config.account = None  # 模拟 SDK 上无 account

        # 与原始 tiger1 一致：显式传 client_config.account
        api_manager.initialize_real_apis(quote_client, trade_client, account=expected_account)
        self.assertEqual(
            api_manager._account,
            expected_account,
            "传入 client_config.account 后，api_manager._account 必须等于 client_config.account"
        )
        self.assertEqual(
            api_manager.trade_api.account,
            expected_account,
            "trade_api.account 必须与 api_manager._account 一致"
        )
        print("✅ [回归] 使用 client_config.account 初始化后 account 正确设置")


if __name__ == '__main__':
    unittest.main()
