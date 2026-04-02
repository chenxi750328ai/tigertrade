"""
Feature测试基类

测试设计时两种场景都要设计到（不是「先看API是否可用再决定怎么测」）：
1. 真实API场景：专用用例，假定真实API可用，真实下单+真实查询，必须通过；无真实API时该用例 Fail（不允许 Skip，测试目的就是测出问题）。
2. Mock 场景：专用用例，仅用 mock 测内部逻辑、提高覆盖率。
"""
import sys
import os
import unittest
from src.api_adapter import api_manager

# 全局标志：是否已初始化真实API
_REAL_API_INITIALIZED = False


class RealAPIUnavailableError(Exception):
    """真实API不可用时抛出，由调用方捕获并走 mock 路径"""
    pass


def initialize_real_api():
    """初始化真实API（DEMO账户）。失败时抛 RealAPIUnavailableError。"""
    global _REAL_API_INITIALIZED

    if _REAL_API_INITIALIZED:
        if api_manager.is_mock_mode:
            raise RealAPIUnavailableError("API已切换到Mock模式")
        return

    print("=" * 60)
    print("初始化真实API（DEMO账户）用于Feature测试...")
    print("=" * 60)

    try:
        from tigeropen.tiger_open_config import TigerOpenClientConfig
        from tigeropen.quote.quote_client import QuoteClient
        from tigeropen.trade.trade_client import TradeClient

        client_config = TigerOpenClientConfig(props_path='./openapicfg_dem')
        print(f"✅ 配置加载成功: account={client_config.account}, tiger_id={client_config.tiger_id}")

        quote_client = QuoteClient(client_config)
        trade_client = TradeClient(client_config)

        account_to_use = client_config.account
        if not account_to_use and hasattr(trade_client, 'config'):
            account_to_use = getattr(trade_client.config, 'account', None)

        if not account_to_use:
            raise RealAPIUnavailableError("无法获取account信息（无真实API配置）")

        print(f"📋 使用account: {account_to_use}")
        api_manager.initialize_real_apis(quote_client, trade_client, account=account_to_use)
        print(f"✅ API初始化成功")
        print(f"   Quote API: {type(api_manager.quote_api).__name__}")
        print(f"   Trade API: {type(api_manager.trade_api).__name__}")
        print(f"   Account: {api_manager._account}")
        print(f"   Mock模式: {api_manager.is_mock_mode}")

        if api_manager.is_mock_mode:
            raise RealAPIUnavailableError("API仍处于Mock模式")

        if not api_manager._account or not api_manager.trade_api.account:
            raise RealAPIUnavailableError("account设置失败")

        _REAL_API_INITIALIZED = True
        print("=" * 60)
        print("✅ 真实API初始化完成，Feature测试将使用真实环境")
        print("=" * 60)

    except RealAPIUnavailableError:
        raise
    except Exception as e:
        print(f"❌ 真实API初始化失败: {e}")
        import traceback
        traceback.print_exc()
        raise RealAPIUnavailableError(str(e))


class FeatureTestBase:
    """Feature测试基类 - 有真实API用真实，无则走 mock 路径；真实API场景用例无API时 Fail，不允许 Skip"""

    _real_api_available = False

    @classmethod
    def setUpClass(cls):
        """初始化真实API环境；失败则标记走 mock 路径，不跳过"""
        try:
            initialize_real_api()
            cls._real_api_available = True
        except RealAPIUnavailableError as e:
            print(f"⚠️ 真实API不可用，Feature 测试将走 mock 路径: {e}")
            cls._real_api_available = False
            if not api_manager.is_mock_mode:
                api_manager.initialize_mock_apis()
        except Exception as e:
            print(f"⚠️ 初始化异常，走 mock 路径: {e}")
            cls._real_api_available = False
            if not api_manager.is_mock_mode:
                api_manager.initialize_mock_apis()

        if cls._real_api_available:
            print(f"\n✅ Feature测试环境就绪（真实API）:")
            print(f"   Account: {api_manager._account}")
        else:
            print(f"\n✅ Feature测试环境就绪（Mock 路径）")
        if api_manager.trade_api:
            print(f"   Trade API: {type(api_manager.trade_api).__name__}")


# 注意：不再在模块级别自动初始化，避免pytest收集测试时的副作用
# 所有测试应该通过setUpClass或setUp方法调用initialize_real_api()
