#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
执行器模块100%覆盖率测试
补充所有缺失的代码路径测试
"""
import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock, PropertyMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

sys.path.insert(0, '/home/cx/tigertrade')

from src.executor import MarketDataProvider, OrderExecutor, TradingExecutor
from src.strategies.base_strategy import BaseTradingStrategy
from src import tiger1 as t1


class TestMarketDataProvider100Coverage(unittest.TestCase):
    """MarketDataProvider 100%覆盖率测试"""
    
    def setUp(self):
        """测试前准备"""
        self.provider = MarketDataProvider('SIL.COMEX.202603')
    
    @patch('src.executor.data_provider.t1.get_kline_data')
    @patch('src.executor.data_provider.t1.get_tick_data')
    @patch('src.executor.data_provider.t1.calculate_indicators')
    @patch('src.executor.data_provider.t1.judge_market_trend')
    @patch('src.executor.data_provider.t1.adjust_grid_interval')
    def test_get_market_data_normal(self, mock_adjust, mock_judge, mock_calc, mock_tick, mock_kline):
        """测试正常获取市场数据"""
        # Mock K线数据
        df_5m = pd.DataFrame({
            'close': [100] * 50,
            'high': [101] * 50,
            'low': [99] * 50,
            'open': [100] * 50,
            'volume': [1000] * 50
        })
        df_1m = pd.DataFrame({
            'close': [100] * 50,
            'high': [101] * 50,
            'low': [99] * 50,
            'open': [100] * 50,
            'volume': [1000] * 50
        })
        mock_kline.return_value = df_5m
        mock_kline.side_effect = [df_5m, df_1m]
        
        # Mock Tick数据
        df_tick = pd.DataFrame({
            'price': [100.5] * 10
        })
        mock_tick.return_value = df_tick
        
        # Mock指标
        indicators = {
            '1m': {'close': 100, 'rsi': 45},
            '5m': {'close': 100, 'rsi': 50, 'atr': 0.5}
        }
        mock_calc.return_value = indicators
        mock_judge.return_value = 'osc_normal'
        
        # Mock网格参数
        t1.grid_upper = 105.0
        t1.grid_lower = 95.0
        
        result = self.provider.get_market_data(seq_length=10)
        
        self.assertIsNotNone(result)
        self.assertIn('current_data', result)
        self.assertIn('indicators', result)
        self.assertIn('historical_data', result)
    
    @patch('src.executor.data_provider.t1.get_kline_data')
    def test_get_market_data_empty_kline(self, mock_kline):
        """测试空K线数据"""
        mock_kline.return_value = pd.DataFrame()
        mock_kline.side_effect = [pd.DataFrame(), pd.DataFrame()]
        
        with self.assertRaises(ValueError):
            self.provider.get_market_data()
    
    @patch('src.executor.data_provider.t1.get_kline_data')
    @patch('src.executor.data_provider.t1.get_tick_data')
    @patch('src.executor.data_provider.t1.calculate_indicators')
    def test_get_market_data_no_indicators(self, mock_calc, mock_tick, mock_kline):
        """测试指标计算失败"""
        df_5m = pd.DataFrame({'close': [100] * 50})
        df_1m = pd.DataFrame({'close': [100] * 50})
        mock_kline.side_effect = [df_5m, df_1m]
        mock_tick.return_value = pd.DataFrame()
        mock_calc.return_value = {}
        
        with self.assertRaises(ValueError):
            self.provider.get_market_data()
    
    @patch('src.executor.data_provider.t1.get_kline_data')
    @patch('src.executor.data_provider.t1.get_tick_data')
    @patch('src.executor.data_provider.t1.calculate_indicators')
    @patch('src.executor.data_provider.t1.judge_market_trend')
    @patch('src.executor.data_provider.t1.adjust_grid_interval')
    def test_get_market_data_empty_tick(self, mock_adjust, mock_judge, mock_calc, mock_tick, mock_kline):
        """测试空Tick数据"""
        df_5m = pd.DataFrame({
            'close': [100] * 50, 'high': [101] * 50, 'low': [99] * 50,
            'open': [100] * 50, 'volume': [1000] * 50
        })
        df_1m = pd.DataFrame({
            'close': [100] * 50, 'high': [101] * 50, 'low': [99] * 50,
            'open': [100] * 50, 'volume': [1000] * 50
        })
        mock_kline.side_effect = [df_5m, df_1m]
        mock_tick.return_value = pd.DataFrame()
        
        indicators = {
            '1m': {'close': 100, 'rsi': 45},
            '5m': {'close': 100, 'rsi': 50, 'atr': 0.5}
        }
        mock_calc.return_value = indicators
        mock_judge.return_value = 'osc_normal'
        t1.grid_upper = 105.0
        t1.grid_lower = 95.0
        
        result = self.provider.get_market_data()
        self.assertIsNotNone(result)
        # 应该使用K线价格
        self.assertEqual(result['tick_price'], 100)
    
    @patch('src.executor.data_provider.t1.get_kline_data')
    @patch('src.executor.data_provider.t1.get_tick_data')
    @patch('src.executor.data_provider.t1.calculate_indicators')
    @patch('src.executor.data_provider.t1.judge_market_trend')
    @patch('src.executor.data_provider.t1.adjust_grid_interval')
    def test_get_market_data_tick_no_price_column(self, mock_adjust, mock_judge, mock_calc, mock_tick, mock_kline):
        """测试Tick数据没有price列"""
        df_5m = pd.DataFrame({
            'close': [100] * 50, 'high': [101] * 50, 'low': [99] * 50,
            'open': [100] * 50, 'volume': [1000] * 50
        })
        df_1m = pd.DataFrame({
            'close': [100] * 50, 'high': [101] * 50, 'low': [99] * 50,
            'open': [100] * 50, 'volume': [1000] * 50
        })
        mock_kline.side_effect = [df_5m, df_1m]
        mock_tick.return_value = pd.DataFrame({'other': [1] * 10})
        
        indicators = {
            '1m': {'close': 100, 'rsi': 45},
            '5m': {'close': 100, 'rsi': 50, 'atr': 0.5}
        }
        mock_calc.return_value = indicators
        mock_judge.return_value = 'osc_normal'
        t1.grid_upper = 105.0
        t1.grid_lower = 95.0
        
        result = self.provider.get_market_data()
        self.assertIsNotNone(result)
        # 应该使用K线价格
        self.assertEqual(result['tick_price'], 100)
    
    @patch('src.executor.data_provider.t1.get_kline_data')
    @patch('src.executor.data_provider.t1.get_tick_data')
    @patch('src.executor.data_provider.t1.calculate_indicators')
    @patch('src.executor.data_provider.t1.judge_market_trend')
    @patch('src.executor.data_provider.t1.adjust_grid_interval')
    def test_get_market_data_cache_overflow(self, mock_adjust, mock_judge, mock_calc, mock_tick, mock_kline):
        """测试缓存溢出"""
        df_5m = pd.DataFrame({
            'close': [100] * 50, 'high': [101] * 50, 'low': [99] * 50,
            'open': [100] * 50, 'volume': [1000] * 50
        })
        df_1m = pd.DataFrame({
            'close': [100] * 50, 'high': [101] * 50, 'low': [99] * 50,
            'open': [100] * 50, 'volume': [1000] * 50
        })
        # 使用return_value而不是side_effect，让每次调用都返回相同值
        def kline_side_effect(*args, **kwargs):
            if len(mock_kline.call_args_list) % 2 == 0:
                return df_5m
            else:
                return df_1m
        mock_kline.side_effect = kline_side_effect
        mock_tick.return_value = pd.DataFrame({'price': [100.5]})
        
        indicators = {
            '1m': {'close': 100, 'rsi': 45},
            '5m': {'close': 100, 'rsi': 50, 'atr': 0.5}
        }
        mock_calc.return_value = indicators
        mock_judge.return_value = 'osc_normal'
        t1.grid_upper = 105.0
        t1.grid_lower = 95.0
        
        # 填充缓存
        seq_length = 10
        for i in range(50):
            self.provider.get_market_data(seq_length=seq_length)
        
        # 检查缓存大小
        max_cache_size = seq_length + 20
        self.assertLessEqual(len(self.provider.historical_data_cache), max_cache_size)
    
    @patch('src.executor.data_provider.t1.get_kline_data')
    def test_get_kline_data(self, mock_kline):
        """测试get_kline_data方法"""
        df = pd.DataFrame({'close': [100] * 10})
        mock_kline.return_value = df
        
        result = self.provider.get_kline_data('5min', 10)
        self.assertIsNotNone(result)
        mock_kline.assert_called_once()
    
    @patch('src.executor.data_provider.t1.get_tick_data')
    def test_get_tick_data(self, mock_tick):
        """测试get_tick_data方法"""
        df = pd.DataFrame({'price': [100] * 10})
        mock_tick.return_value = df
        
        result = self.provider.get_tick_data(10)
        self.assertIsNotNone(result)
        mock_tick.assert_called_once()
    
    @patch('src.executor.data_provider.t1.calculate_indicators')
    def test_calculate_indicators(self, mock_calc):
        """测试calculate_indicators方法"""
        df_5m = pd.DataFrame({'close': [100] * 10})
        df_1m = pd.DataFrame({'close': [100] * 10})
        indicators = {'1m': {}, '5m': {}}
        mock_calc.return_value = indicators
        
        result = self.provider.calculate_indicators(df_5m, df_1m)
        self.assertEqual(result, indicators)
        mock_calc.assert_called_once_with(df_5m, df_1m)


class TestOrderExecutor100Coverage(unittest.TestCase):
    """OrderExecutor 100%覆盖率测试"""
    
    def setUp(self):
        """测试前准备"""
        # 不要Mock risk_manager！使用真实的t1模块测试
        t1.current_position = 0
        t1.daily_loss = 0
        self.executor = OrderExecutor(t1)  # 使用真实的t1，不Mock
    
    @patch('src.executor.order_executor.t1.get_effective_position_for_buy', return_value=0)
    @patch('src.executor.order_executor.t1.sync_positions_from_backend', return_value=None)
    @patch('src.executor.order_executor.api_manager')
    def test_execute_buy_api_none(self, mock_api, _sync, _pos):
        """测试API为None（mock 持仓避免先被硬顶拒绝）"""
        mock_api.trade_api = None
        
        result, message = self.executor.execute_buy(100.0, 0.5, 97.0, 105.0, 0.6)
        self.assertFalse(result)
        self.assertIn("交易API未初始化", message)
    
    @patch('src.executor.order_executor.t1.get_effective_position_for_buy', return_value=0)
    @patch('src.executor.order_executor.t1.sync_positions_from_backend', return_value=None)
    @patch('src.executor.order_executor.api_manager')
    def test_execute_buy_order_result_dict(self, mock_api, _sync, _pos):
        """测试订单结果为字典"""
        mock_trade_api = MagicMock()
        mock_order_result = {'order_id': 'TEST_123'}
        mock_trade_api.place_order.return_value = mock_order_result
        mock_api.trade_api = mock_trade_api
        mock_api.is_mock_mode = False
        
        result, message = self.executor.execute_buy(100.0, 0.5, 97.0, 105.0, 0.6)
        self.assertTrue(result)
        self.assertIn("订单提交成功", message)
    
    @patch('src.executor.order_executor.t1.get_effective_position_for_buy', return_value=0)
    @patch('src.executor.order_executor.t1.sync_positions_from_backend', return_value=None)
    @patch('src.executor.order_executor.api_manager')
    def test_execute_buy_order_result_other(self, mock_api, _sync, _pos):
        """测试订单结果为其他类型"""
        mock_trade_api = MagicMock()
        mock_order_result = "ORDER_123"
        mock_trade_api.place_order.return_value = mock_order_result
        mock_api.trade_api = mock_trade_api
        mock_api.is_mock_mode = False
        
        result, message = self.executor.execute_buy(100.0, 0.5, 97.0, 105.0, 0.6)
        self.assertTrue(result)
        self.assertIn("订单提交成功", message)
    
    @patch('src.executor.order_executor.t1.get_effective_position_for_buy', return_value=0)
    @patch('src.executor.order_executor.t1.sync_positions_from_backend', return_value=None)
    @patch('src.executor.order_executor.api_manager')
    def test_execute_buy_exception(self, mock_api, _sync, _pos):
        """测试下单异常"""
        mock_trade_api = MagicMock()
        mock_trade_api.place_order.side_effect = Exception("API错误")
        mock_api.trade_api = mock_trade_api
        
        result, message = self.executor.execute_buy(100.0, 0.5, 97.0, 105.0, 0.6)
        self.assertFalse(result)
        self.assertIn("下单异常", message)
    
    @patch('src.executor.order_executor.api_manager')
    def test_execute_sell_api_none(self, mock_api):
        """测试卖出时API为None"""
        mock_api.trade_api = None
        t1.current_position = 1
        
        result, message = self.executor.execute_sell(100.0, 0.6)
        self.assertFalse(result)
        self.assertIn("交易API未初始化", message)
    
    @patch('src.executor.order_executor.api_manager')
    def test_execute_sell_order_result_dict(self, mock_api):
        """测试卖出订单结果为字典"""
        mock_trade_api = MagicMock()
        mock_order_result = {'id': 'TEST_456'}
        mock_trade_api.place_order.return_value = mock_order_result
        mock_api.trade_api = mock_trade_api
        mock_api.is_mock_mode = False
        t1.current_position = 1
        
        result, message = self.executor.execute_sell(100.0, 0.6)
        self.assertTrue(result)
        self.assertIn("订单提交成功", message)
    
    @patch('src.executor.order_executor.api_manager')
    def test_execute_sell_exception(self, mock_api):
        """测试卖出异常"""
        mock_trade_api = MagicMock()
        mock_trade_api.place_order.side_effect = Exception("API错误")
        mock_api.trade_api = mock_trade_api
        t1.current_position = 1
        
        result, message = self.executor.execute_sell(100.0, 0.6)
        self.assertFalse(result)
        self.assertIn("下单异常", message)


class TestTradingExecutor100Coverage(unittest.TestCase):
    """TradingExecutor 100%覆盖率测试"""
    
    def setUp(self):
        """测试前准备"""
        self.mock_strategy = Mock(spec=BaseTradingStrategy)
        self.mock_strategy.strategy_name = "Test Strategy"
        self.mock_strategy.seq_length = 10
        self.mock_strategy.predict_action = Mock(return_value=(1, 0.6, 0.1))
        
        self.mock_data_provider = Mock()
        self.mock_data_provider.get_market_data = Mock(return_value={
            'current_data': {'tick_price': 100.0},
            'indicators': {},
            'historical_data': pd.DataFrame(),
            'tick_price': 100.0,
            'price_current': 100.0,
            'atr': 0.5,
            'grid_lower': 95.0,
            'grid_upper': 105.0
        })
        
        self.mock_order_executor = Mock()
        self.mock_order_executor.execute_buy = Mock(return_value=(True, "成功"))
        self.mock_order_executor.execute_sell = Mock(return_value=(True, "成功"))
        
        self.executor = TradingExecutor(
            strategy=self.mock_strategy,
            data_provider=self.mock_data_provider,
            order_executor=self.mock_order_executor,
            config={'confidence_threshold': 0.4, 'loop_interval': 1}
        )
    
    def test_parse_prediction_3_tuple(self):
        """测试解析3元组"""
        result = self.executor._parse_prediction((1, 0.6, 0.1))
        self.assertEqual(result, (1, 0.6, 0.1))
    
    def test_parse_prediction_2_tuple(self):
        """测试解析2元组"""
        result = self.executor._parse_prediction((1, 0.6))
        self.assertEqual(result, (1, 0.6, None))
    
    def test_parse_prediction_invalid(self):
        """测试解析无效结果"""
        result = self.executor._parse_prediction(1)
        self.assertEqual(result, (0, 0.0, None))
    
    def test_execute_prediction_hold(self):
        """测试执行持有操作"""
        market_data = {
            'tick_price': 100.0,
            'atr': 0.5,
            'grid_lower': 95.0,
            'grid_upper': 105.0,
            'current_data': {'tick_price': 100.0, 'threshold': 95.5}
        }
        
        self.executor._execute_prediction((0, 0.5, None), market_data)
        self.mock_order_executor.execute_buy.assert_not_called()
        self.mock_order_executor.execute_sell.assert_not_called()
    
    @patch('src.executor.trading_executor.datetime')
    def test_print_prediction(self, mock_datetime):
        """测试打印预测结果"""
        mock_datetime.now.return_value.strftime.return_value = "12:00:00"
        market_data = {
            'tick_price': 100.0,
            'price_current': 100.0,
            'atr': 0.5,
            'grid_lower': 95.0,
            'grid_upper': 105.0,
            'current_data': {'threshold': 95.5}
        }
        
        # 应该不抛出异常
        self.executor._print_prediction((1, 0.6, 0.1), market_data)
    
    def test_print_progress(self):
        """测试打印进度"""
        self.executor.stats['total_predictions'] = 10
        self.executor.stats['avg_confidence'] = 0.5
        
        # 应该不抛出异常
        self.executor._print_progress(timedelta(hours=1), timedelta(hours=19))
    
    def test_print_final_stats(self):
        """测试打印最终统计"""
        self.executor.stats['total_predictions'] = 10
        self.executor.stats['buy_signals'] = 5
        self.executor.stats['sell_signals'] = 3
        self.executor.stats['hold_signals'] = 2
        self.executor.stats['avg_confidence'] = 0.5
        self.executor.stats['successful_orders'] = 8
        self.executor.stats['failed_orders'] = 0
        self.executor.stats['errors'] = 0
        
        # 应该不抛出异常
        self.executor._print_final_stats()
    
    @patch('src.executor.trading_executor.datetime')
    @patch('src.executor.trading_executor.time')
    def test_run_loop_keyboard_interrupt(self, mock_time, mock_datetime):
        """测试键盘中断"""
        mock_datetime.now.return_value = datetime(2026, 1, 1, 10, 0, 0)
        mock_time.sleep.side_effect = KeyboardInterrupt()
        
        # 应该捕获KeyboardInterrupt
        try:
            self.executor.run_loop(duration_hours=20)
        except KeyboardInterrupt:
            pass
    
    @patch('src.executor.trading_executor.datetime')
    @patch('src.executor.trading_executor.time')
    def test_run_loop_exception(self, mock_time, mock_datetime):
        """测试运行循环异常"""
        start_time = datetime(2026, 1, 1, 10, 0, 0)
        end_time = datetime(2026, 1, 1, 10, 0, 1)
        
        # 设置datetime.now的返回值序列
        call_count = [0]
        def datetime_now():
            call_count[0] += 1
            if call_count[0] == 1:
                return start_time
            elif call_count[0] <= 3:
                return start_time  # 循环中
            else:
                return end_time  # 结束循环
        mock_datetime.now.side_effect = datetime_now
        
        self.mock_data_provider.get_market_data.side_effect = Exception("数据获取失败")
        mock_time.sleep.return_value = None
        
        self.executor.run_loop(duration_hours=0.0001)
        self.assertGreater(self.executor.stats['errors'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
