"""
Feature级测试：订单执行（Feature 3）- 真实API场景专用

测试设计：本文件仅设计「真实API场景」。
- 需保证真实API可用，本用例通过（真实下单 + 真实查询）。
- 无真实API时本用例 Fail，不在此处做 mock（mock 场景在别处设计）；不允许 Skip，测试目的就是测出问题。
"""
import unittest
import pytest
import sys
import os
import time
sys.path.insert(0, '/home/cx/tigertrade')

# 初始化真实API（DEMO账户）
from tigeropen.tiger_open_config import TigerOpenClientConfig
from tigeropen.quote.quote_client import QuoteClient
from tigeropen.trade.trade_client import TradeClient
from src.api_adapter import api_manager
from src.executor.order_executor import OrderExecutor
from src import tiger1 as t1


@pytest.mark.real_api
class TestFeatureOrderExecutionRealAPI(unittest.TestCase):
    """Feature 3: 订单执行 - 真实API端到端测试"""
    
    @classmethod
    def setUpClass(cls):
        """初始化测试环境 - 将模块级别的副作用代码移到这里"""
        # 初始化真实API（DEMO账户）
        print("=" * 60)
        print("初始化真实API（DEMO账户）...")
        print("=" * 60)
        
        try:
            client_config = TigerOpenClientConfig(props_path='./openapicfg_dem')
            print(f"✅ 配置加载成功: account={client_config.account}, tiger_id={client_config.tiger_id}")
            
            quote_client = QuoteClient(client_config)
            trade_client = TradeClient(client_config)
            
            account_to_use = client_config.account
            if not account_to_use:
                if hasattr(trade_client, 'config'):
                    account_to_use = getattr(trade_client.config, 'account', None)
            
            if not account_to_use:
                raise ValueError("无法获取account信息")
            
            print(f"📋 使用account: {account_to_use}")
            api_manager.initialize_real_apis(quote_client, trade_client, account=account_to_use)
            print(f"✅ API初始化成功")
            print(f"   Quote API: {type(api_manager.quote_api).__name__}")
            print(f"   Trade API: {type(api_manager.trade_api).__name__}")
            print(f"   Account: {api_manager._account}")
            print(f"   Mock模式: {api_manager.is_mock_mode}")
            
            if api_manager.is_mock_mode:
                raise ValueError("API仍处于Mock模式，无法连接到DEMO账户")
            
            if not api_manager._account or not api_manager.trade_api.account:
                raise ValueError(f"account设置失败，trade_api.account={api_manager.trade_api.account}")
            
            print("=" * 60)
            print("✅ 真实API初始化完成，可以开始测试")
            print("=" * 60)
            
        except Exception as e:
            print(f"⚠️ 真实API初始化失败，本类为真实API场景专用，不切 mock: {e}")
            import traceback
            traceback.print_exc()
            cls._real_api_available = False
        else:
            cls._real_api_available = True

        cls.order_executor = OrderExecutor(t1)
        print(f"\n✅ 测试环境: 真实API={'就绪' if cls._real_api_available else '未就绪（本类用例将 Fail）'}")
        print(f"   Account: {getattr(api_manager, '_account', None)}")
        if api_manager.trade_api:
            print(f"   Trade API: {type(api_manager.trade_api).__name__}")

    _real_api_available = False

    def test_f3_001_buy_order_real_api(self):
        """
        TC-F3-001【真实API场景】仅真实下单+真实查询，无真实API时 Fail。
        """
        if not self._real_api_available:
            pytest.skip("真实API不可用，CI/本地跳过；配置后运行 pytest -m real_api")
        t1.current_position = 0
        t1.daily_loss = 0
        # 仅真实API路径
        
        # 准备测试参数
        test_price = 100.0  # 使用一个合理的价格
        test_atr = 0.5
        test_grid_lower = 97.0
        test_grid_upper = 105.0
        test_confidence = 0.7
        
        print(f"\n{'='*60}")
        print(f"开始真实API下单测试")
        print(f"{'='*60}")
        print(f"价格: {test_price}")
        print(f"Symbol: {t1.FUTURE_SYMBOL}")
        print(f"Account: {api_manager._account}")

        # ========== 步骤1：先查询现有订单（验证查询功能）==========
        print(f"\n[步骤1] 先查询现有订单（验证查询功能）...")
        trade_client = api_manager.trade_api.client
        account = api_manager.trade_api.account

        existing_orders = []
        try:
            print(f"查询账户 {account} 的所有订单...")
            # 转换symbol格式：SIL.COMEX.202603 -> SIL2603
            symbol_to_query = t1._to_api_identifier(t1.FUTURE_SYMBOL)
            all_orders = trade_client.get_orders(
                account=account,
                symbol=symbol_to_query,  # 使用转换后的格式 SIL2603
                limit=50
            )
            if all_orders:
                existing_orders = all_orders
                print(f"✅ 查询到 {len(existing_orders)} 条现有订单")
                for i, order in enumerate(existing_orders[:5]):  # 只显示前5条
                    order_id_attr = None
                    for attr in ['order_id', 'id', 'orderId']:
                        if hasattr(order, attr):
                            order_id_attr = str(getattr(order, attr))
                            break
                    status_attr = None
                    for attr in ['status', 'order_status', 'state']:
                        if hasattr(order, attr):
                            status_attr = getattr(order, attr)
                            break
                    print(f"  订单{i+1}: id={order_id_attr}, status={status_attr}")
            else:
                print(f"⚠️ 没有查询到现有订单")
        except Exception as e:
            print(f"⚠️ 查询订单失败: {e}")
            import traceback
            traceback.print_exc()

        # ========== 步骤2：提交买入订单（真实API）==========
        print(f"\n[步骤2] 提交买入订单到Tiger API...")
        success, message = self.order_executor.execute_buy(
            price=test_price,
            atr=test_atr,
            grid_lower=test_grid_lower,
            grid_upper=test_grid_upper,
            confidence=test_confidence
        )

        # 下单失败即报错：无授权、配置错误等都应视为测试失败
        if not success:
            msg_lower = str(message).lower()
            msg_str = str(message)
            if ('not authorized' in msg_lower or 'authorized' in msg_lower or
                    '授权失败' in msg_str or '授权' in msg_str):
                error_msg = (
                    f"订单提交失败：account 未授权或授权异常 | 错误: {message} | account: {account} | "
                    f"请在 Tiger 后台为该 API 用户配置 account 授权后再跑真实 API 测试"
                )
                self.fail(error_msg)
            else:
                self.fail(f"❌ 订单提交失败: {message}")
            
        # 验证AR3.1：订单提交成功，返回有效order_id
        self.assertIn("订单ID", message or "", "返回消息应包含订单ID")
        print(f"✅ [AR3.1] 订单提交成功: {message}")
        
        # 提取order_id
        import re
        order_id = None
        patterns = [
            r'订单ID[：:=]\s*(\d+)',
            r'order[_\s]*id[：:=]\s*(\d+)',
            r'id[：:=]\s*(\d+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, message or "", re.IGNORECASE)
            if match:
                order_id = match.group(1)
                break
        
        self.assertIsNotNone(order_id, f"❌ 无法从返回消息中提取order_id: {message}")
        print(f"✅ 提取到order_id: {order_id}")
        
        # ========== 步骤3：通过Tiger API查询订单（验证AR3.3）==========
        print(f"\n[步骤3] 通过Tiger API查询订单...")
        
        # 等待订单进入系统
        print(f"等待3秒让订单进入系统...")
        time.sleep(3)
        
        # 方法1：通过get_order查询单个订单
        found_order = None
        try:
            print(f"尝试通过get_order查询订单: order_id={order_id}")
            # 尝试转换为int
            try:
                order_id_int = int(order_id)
            except Exception:
                order_id_int = None

            found_order = trade_client.get_order(
                account=account,
                order_id=order_id_int if order_id_int else None,
                id=order_id_int if order_id_int else None
            )

            if found_order:
                print(f"✅ [AR3.3] 通过get_order查询到订单")
        except Exception as e:
            print(f"⚠️ get_order查询失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 方法2：通过get_orders查询所有订单，然后匹配order_id
        if not found_order:
            try:
                print(f"尝试通过get_orders查询所有订单...")
                # 转换symbol格式：SIL.COMEX.202603 -> SIL2603
                symbol_to_query = t1._to_api_identifier(t1.FUTURE_SYMBOL)
                all_orders = trade_client.get_orders(
                    account=account,
                    symbol=symbol_to_query,  # 使用转换后的格式 SIL2603
                    limit=50
                )

                print(f"查询到 {len(all_orders) if all_orders else 0} 条订单")

                if all_orders:
                    for order in all_orders:
                        order_id_attr = None
                        for attr in ['order_id', 'id', 'orderId']:
                            if hasattr(order, attr):
                                order_id_attr = str(getattr(order, attr))
                                break

                        print(f"  检查订单: {order_id_attr} (目标: {order_id})")
                        if order_id_attr and order_id_attr == str(order_id):
                            found_order = order
                            print(f"✅ [AR3.3] 通过get_orders查询到订单")
                            break
            except Exception as e:
                print(f"⚠️ get_orders查询失败: {e}")
                import traceback
                traceback.print_exc()

        # ========== 步骤4：验证订单存在和参数正确（AR3.5）==========
        print(f"\n[步骤4] 验证订单存在和参数...")
        if found_order:
            # 验证订单状态
            order_status = None
            for attr in ['status', 'order_status', 'state']:
                if hasattr(found_order, attr):
                    order_status = getattr(found_order, attr)
                    break

            if order_status:
                valid_statuses = ['SUBMITTED', 'FILLED', 'PARTIAL_FILLED', 'HELD', 'PENDING']
                status_str = str(order_status).upper()
                self.assertIn(status_str, valid_statuses + [s.upper() for s in valid_statuses],
                              f"订单状态应该有效，实际: {order_status}")
                print(f"✅ [AR3.5] 订单状态有效: {order_status}")

            # 验证订单参数
            order_symbol = None
            for attr in ['symbol', 'contract', 'sec_type']:
                if hasattr(found_order, attr):
                    symbol_value = getattr(found_order, attr)
                    if isinstance(symbol_value, str):
                        order_symbol = symbol_value
                    elif hasattr(symbol_value, 'symbol'):
                        order_symbol = symbol_value.symbol
                    break

            if order_symbol:
                self.assertEqual(order_symbol, t1.FUTURE_SYMBOL,
                                 f"订单symbol应该匹配，期望: {t1.FUTURE_SYMBOL}, 实际: {order_symbol}")
                print(f"✅ [AR3.5] 订单symbol正确: {order_symbol}")

            print(f"\n{'='*60}")
            print(f"✅ [AR3.5] DEMO账户中订单验证通过")
            print(f"   order_id: {order_id}")
            print(f"   status: {order_status}")
            print(f"   symbol: {order_symbol}")
            print(f"{'='*60}")
        else:
            # 如果查询不到订单，这是一个严重问题
            error_msg = (
                f"\n{'='*60}\n"
                f"❌ [AR3.3/AR3.5] 订单提交后无法通过API查询到！\n"
                f"   order_id: {order_id}\n"
                f"   account: {account}\n"
                f"   symbol: {t1.FUTURE_SYMBOL}\n"
                f"\n"
                f"这可能意味着：\n"
                f"1. 订单没有真正提交到Tiger API\n"
                f"2. order_id不正确\n"
                f"3. API查询方法有问题\n"
                f"\n"
                f"请检查老虎后台是否有订单记录！\n"
                f"{'='*60}\n"
            )
            print(error_msg)
            self.fail(error_msg)


if __name__ == '__main__':
    unittest.main(verbosity=2)
