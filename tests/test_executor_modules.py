#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
执行器模块单元测试
"""
import unittest
import sys
from unittest.mock import Mock, patch, MagicMock, PropertyMock
import pandas as pd
from src.executor import MarketDataProvider, OrderExecutor, TradingExecutor
from src.strategies.base_strategy import BaseTradingStrategy


class TestMarketDataProvider(unittest.TestCase):
    """MarketDataProvider测试"""
    
    def setUp(self):
        """测试前准备"""
        self.provider = MarketDataProvider('SIL.COMEX.202603')
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.provider.symbol, 'SIL.COMEX.202603')
        self.assertEqual(len(self.provider.historical_data_cache), 0)
    
    def test_clear_cache(self):
        """测试清空缓存"""
        self.provider.historical_data_cache = [{'test': 'data'}]
        self.provider.clear_cache()
        self.assertEqual(len(self.provider.historical_data_cache), 0)


class TestOrderExecutor(unittest.TestCase):
    """OrderExecutor测试"""
    
    def setUp(self):
        """测试前准备"""
        from src import tiger1 as t1
        from src.api_adapter import api_manager
        
        # 不要Mock risk_manager！使用真实的t1模块测试
        # 重置持仓和日亏损，确保风控能通过
        t1.current_position = 0
        t1.daily_loss = 0
        
        # 重置API状态，避免测试间污染
        api_manager.trade_api = None
        api_manager.is_mock_mode = True
        
        self.executor = OrderExecutor(t1)  # 使用真实的t1，不Mock
        self.t1 = t1  # 保存引用
    
    def tearDown(self):
        """测试后清理"""
        # 确保每次测试后重置状态，避免测试间污染
        self.t1.current_position = 0
        self.t1.daily_loss = 0
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.executor.risk_manager, self.t1)
    
    def test_execute_buy_success(self):
        """测试买入成功（OrderExecutor直接调用API）"""
        from src.api_adapter import api_manager
        
        # 保存原始状态
        original_trade_api = api_manager.trade_api
        original_mock_mode = api_manager.is_mock_mode
        
        try:
            # Mock API：使用 spec 避免 MagicMock 虚构 place_limit_with_bracket 导致误走组合单
            mock_order_result = MagicMock()
            mock_order_result.order_id = "TEST_ORDER_123"
            mock_trade_api = MagicMock(
                spec=["place_order", "place_limit_with_bracket", "get_order", "get_orders", "wait_until_buy_filled", "account"]
            )
            mock_trade_api.account = "MOCK_ACC"
            mock_trade_api.place_limit_with_bracket.return_value = mock_order_result
            mock_trade_api.get_order.return_value = mock_order_result
            mock_trade_api.get_orders.return_value = []
            api_manager.trade_api = mock_trade_api
            api_manager.is_mock_mode = False
            
            # 避免从后台同步持仓导致硬顶拒绝
            with patch.object(self.t1, 'sync_positions_from_backend', return_value=None), \
                 patch.object(self.t1, 'get_effective_position_for_buy', return_value=0):
                # 使用更合理的参数：grid_lower=98（止损距离2，预期损失2000 < 3000上限）
                result, message = self.executor.execute_buy(
                    price=100.0,
                    atr=0.5,
                    grid_lower=98.0,  # 修改为98，确保风控通过
                    grid_upper=105.0,
                    confidence=0.6
                )
            self.assertTrue(result)
            self.assertIn("订单提交成功", message)
            # 设计：优先老虎 BRACKETS（限价+LOSS+PROFIT 一次提交），不应再拆 3 次 place_order
            self.assertEqual(mock_trade_api.place_limit_with_bracket.call_count, 1, "应调用 1 次组合单")
            self.assertEqual(mock_trade_api.place_order.call_count, 0, "组合单成功时不应再调 place_order 挂 SL/TP")
        finally:
            # 恢复原始状态（包括持仓，因为下单成功会增加持仓）
            api_manager.trade_api = original_trade_api
            api_manager.is_mock_mode = original_mock_mode
            self.t1.current_position = 0  # 重置持仓，避免影响后续测试
            self.t1.daily_loss = 0  # 重置日亏损

    def test_execute_buy_submits_stop_loss_and_take_profit_orders(self):
        """【设计目标】主单成交后必须挂出止损单(STP)和止盈单(LMT)，本用例断言 SL/TP 的订单类型与价格。"""
        from src.api_adapter import api_manager

        original_trade_api = api_manager.trade_api
        original_mock_mode = api_manager.is_mock_mode
        try:
            mock_order_result = MagicMock()
            mock_order_result.order_id = "MAIN_ORDER_1"
            # 无 place_limit_with_bracket → 回退为限价 + 成交后 STP + LMT
            mock_trade_api = MagicMock(
                spec=["place_order", "get_order", "get_orders", "wait_until_buy_filled", "account"]
            )
            mock_trade_api.account = "MOCK_ACC"
            mock_trade_api.place_order.return_value = mock_order_result
            mock_trade_api.get_order.return_value = mock_order_result
            mock_trade_api.get_orders.return_value = []
            mock_trade_api.wait_until_buy_filled.return_value = True
            api_manager.trade_api = mock_trade_api
            api_manager.is_mock_mode = False

            # 固定止损=97、止盈=104，便于断言 place_order 的 stop_price/limit_price
            stop_loss_price = 97.0
            take_profit_price = 104.0
            with patch.object(self.t1, 'sync_positions_from_backend', return_value=None), \
                 patch.object(self.t1, 'get_effective_position_for_buy', return_value=0), \
                 patch.object(self.t1, 'compute_stop_loss', return_value=(stop_loss_price, 2.0)), \
                 patch.object(self.executor, '_calculate_take_profit', return_value=take_profit_price):
                result, message = self.executor.execute_buy(
                    price=100.0,
                    atr=0.5,
                    grid_lower=98.0,
                    grid_upper=105.0,
                    confidence=0.6,
                )
            self.assertTrue(result, message)
            self.assertIn("订单提交成功", message)

            calls = mock_trade_api.place_order.call_args_list
            self.assertEqual(len(calls), 3, "必须 3 次：主单 + 止损单 + 止盈单")

            # 第 1 次：主单 BUY LMT（ positional: symbol, side, order_type, quantity, time_in_force, limit_price, stop_price ）
            main_args, main_kw = calls[0]
            main_side = main_args[1] if len(main_args) > 1 else main_kw.get('side')
            main_otype = main_args[2] if len(main_args) > 2 else main_kw.get('order_type')
            self.assertEqual(getattr(main_side, 'value', main_side), 'BUY', "第 1 次应为 BUY")
            self.assertEqual(getattr(main_otype, 'value', main_otype), 'LMT', "第 1 次应为 LMT")
            main_stop = main_args[6] if len(main_args) > 6 else main_kw.get('stop_price')
            self.assertIsNone(main_stop, "主单 stop_price 应为 None")

            # 第 2 次：止损单 SELL STP，带 stop_price
            sl_kw = calls[1][1]
            self.assertEqual(getattr(sl_kw.get('side'), 'value', sl_kw.get('side')), 'SELL', "止损单应为 SELL")
            self.assertEqual(getattr(sl_kw.get('order_type'), 'value', sl_kw.get('order_type')), 'STP', "止损单应为 STP")
            self.assertEqual(sl_kw.get('stop_price'), stop_loss_price, "止损单 stop_price 应为 compute_stop_loss 返回值")
            self.assertIsNone(sl_kw.get('limit_price'), "止损单 limit_price 应为 None")

            # 第 3 次：止盈单 SELL LMT，带 limit_price
            tp_kw = calls[2][1]
            self.assertEqual(getattr(tp_kw.get('side'), 'value', tp_kw.get('side')), 'SELL', "止盈单应为 SELL")
            self.assertEqual(getattr(tp_kw.get('order_type'), 'value', tp_kw.get('order_type')), 'LMT', "止盈单应为 LMT")
            self.assertEqual(tp_kw.get('limit_price'), take_profit_price, "止盈单 limit_price 应为 _calculate_take_profit 返回值")
            self.assertIsNone(tp_kw.get('stop_price'), "止盈单 stop_price 应为 None")
        finally:
            api_manager.trade_api = original_trade_api
            api_manager.is_mock_mode = original_mock_mode
            self.t1.current_position = 0
            self.t1.daily_loss = 0

    def test_execute_buy_risk_control_failed(self):
        """测试买入风控失败"""
        from src.api_adapter import api_manager
        
        # 保存原始状态
        original_trade_api = api_manager.trade_api
        original_mock_mode = api_manager.is_mock_mode
        
        try:
            # 先Mock API，避免真实API调用
            mock_trade_api = MagicMock()
            api_manager.trade_api = mock_trade_api
            api_manager.is_mock_mode = False
            
            # 强制重置持仓状态（防止其他测试污染）
            self.t1.current_position = 0
            self.t1.daily_loss = 0
            
            # 重新创建executor，确保使用Mock的API
            executor = OrderExecutor(self.t1)
            
            # 设置持仓达到上限，触发风控失败（必须在executor创建后设置）
            self.t1.current_position = self.t1.GRID_MAX_POSITION
            # 让 execute_buy 内读到的有效持仓为上限（否则会从后台同步）
            with patch.object(self.t1, 'sync_positions_from_backend', return_value=None), \
                 patch.object(self.t1, 'get_effective_position_for_buy', return_value=self.t1.GRID_MAX_POSITION):
                result, message = executor.execute_buy(
                    price=100.0, atr=0.5, grid_lower=95.0, grid_upper=105.0, confidence=0.6
                )
            
            # 验证：风控/硬顶应该阻止下单（硬顶拒绝消息为「持仓已达硬顶」）
            self.assertFalse(result, 
                           f"持仓已达上限时应返回False，实际返回True，message={message}, 持仓={self.t1.current_position}, 上限={self.t1.GRID_MAX_POSITION}")
            self.assertTrue("风控" in message or "硬顶" in message or "拒绝" in message, f"消息应包含风控/硬顶/拒绝，实际: {message}")
            # 验证：API不应该被调用
            self.assertEqual(mock_trade_api.place_order.call_count, 0, 
                           "风控失败时不应调用place_order")
        finally:
            # 恢复原始状态
            api_manager.trade_api = original_trade_api
            api_manager.is_mock_mode = original_mock_mode
            self.t1.current_position = 0  # 强制重置
            self.t1.daily_loss = 0
    
    def test_execute_sell_no_position(self):
        """测试卖出无持仓（mock sync 避免后台同步覆盖为有仓）"""
        self.t1.current_position = 0
        with patch.object(self.t1, 'sync_positions_from_backend', return_value=None):
            result, message = self.executor.execute_sell(price=100.0, confidence=0.6)
        self.assertFalse(result)
        self.assertTrue("无持仓" in message or "无多头持仓" in message or "无法卖出" in message, f"应返回无持仓类消息: {message}")
    
    def test_execute_sell_success(self):
        """测试卖出成功（OrderExecutor直接调用API）"""
        from src.api_adapter import api_manager
        
        self.t1.current_position = 1
        
        # 保存原始状态
        original_trade_api = api_manager.trade_api
        original_mock_mode = api_manager.is_mock_mode
        
        try:
            # Mock API调用
            mock_trade_api = MagicMock()
            mock_order_result = MagicMock()
            mock_order_result.order_id = "TEST_ORDER_456"
            mock_trade_api.place_order.return_value = mock_order_result
            api_manager.trade_api = mock_trade_api
            api_manager.is_mock_mode = False
            
            result, message = self.executor.execute_sell(price=100.0, confidence=0.6)
            self.assertTrue(result)
            self.assertIn("订单提交成功", message)
            # OrderExecutor应该直接调用trade_api.place_order
            mock_trade_api.place_order.assert_called_once()
        finally:
            # 恢复原始状态
            api_manager.trade_api = original_trade_api
            api_manager.is_mock_mode = original_mock_mode


class TestTradingExecutor(unittest.TestCase):
    """TradingExecutor测试"""
    
    def setUp(self):
        """测试前准备"""
        # 创建Mock策略
        self.mock_strategy = Mock(spec=BaseTradingStrategy)
        self.mock_strategy.strategy_name = "Mock Strategy"
        self.mock_strategy.seq_length = 10
        self.mock_strategy.predict_action = Mock(return_value=(1, 0.6, 0.1))
        
        # 创建Mock数据提供者
        self.mock_data_provider = Mock()
        self.mock_data_provider.get_market_data = Mock(return_value={
            'current_data': {'tick_price': 100.0, 'grid_lower': 95.0, 'grid_upper': 105.0},
            'indicators': {'5m': {'atr': 0.5}},
            'historical_data': pd.DataFrame(),
            'tick_price': 100.0,
            'price_current': 100.0,
            'atr': 0.5,
            'grid_lower': 95.0,
            'grid_upper': 105.0
        })
        
        # 创建Mock订单执行器
        self.mock_order_executor = Mock()
        self.mock_order_executor.execute_buy = Mock(return_value=(True, "成功"))
        self.mock_order_executor.execute_sell = Mock(return_value=(True, "成功"))
        
        # 创建执行器
        self.executor = TradingExecutor(
            strategy=self.mock_strategy,
            data_provider=self.mock_data_provider,
            order_executor=self.mock_order_executor,
            config={'confidence_threshold': 0.4, 'loop_interval': 1}
        )
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.executor.strategy, self.mock_strategy)
        self.assertEqual(self.executor.data_provider, self.mock_data_provider)
        self.assertEqual(self.executor.order_executor, self.mock_order_executor)
        self.assertEqual(self.executor.stats['total_predictions'], 0)
    
    def test_parse_prediction(self):
        """测试解析预测结果"""
        # 3元组
        result = self.executor._parse_prediction((1, 0.6, 0.1))
        self.assertEqual(result, (1, 0.6, 0.1))
        
        # 2元组
        result = self.executor._parse_prediction((1, 0.6))
        self.assertEqual(result, (1, 0.6, None))
        
        # 其他
        result = self.executor._parse_prediction(1)
        self.assertEqual(result, (0, 0.0, None))
    
    def test_update_stats(self):
        """测试更新统计"""
        initial_count = self.executor.stats['total_predictions']
        self.executor._update_stats((1, 0.6, 0.1))
        self.assertEqual(self.executor.stats['total_predictions'], initial_count + 1)
        self.assertEqual(self.executor.stats['buy_signals'], 1)
    
    def test_execute_prediction_buy(self):
        """测试执行买入预测"""
        market_data = {
            'tick_price': 100.0,
            'atr': 0.5,
            'grid_lower': 95.0,
            'grid_upper': 105.0,
            'current_data': {'tick_price': 100.0}
        }
        
        self.executor._execute_prediction((1, 0.6, 0.1), market_data)
        self.mock_order_executor.execute_buy.assert_called_once()
    
    def test_execute_prediction_low_confidence(self):
        """测试低置信度不执行"""
        market_data = {
            'tick_price': 100.0,
            'atr': 0.5,
            'grid_lower': 95.0,
            'grid_upper': 105.0,
            'current_data': {'tick_price': 100.0}
        }
        
        # 置信度低于阈值
        self.executor._execute_prediction((1, 0.3, 0.1), market_data)
        self.mock_order_executor.execute_buy.assert_not_called()


def run_tests():
    """运行所有测试"""
    print("="*70)
    print("🧪 执行器模块单元测试")
    print("="*70)
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestMarketDataProvider))
    suite.addTests(loader.loadTestsFromTestCase(TestOrderExecutor))
    suite.addTests(loader.loadTestsFromTestCase(TestTradingExecutor))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*70)
    if result.wasSuccessful():
        print("✅ 所有测试通过！")
    else:
        print(f"❌ 测试失败: {len(result.failures)}个失败, {len(result.errors)}个错误")
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
