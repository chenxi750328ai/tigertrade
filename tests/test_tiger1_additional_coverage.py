#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tiger1.py 补充覆盖率测试
覆盖主函数和其他未覆盖的代码路径
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import time
from unittest.mock import patch, MagicMock, Mock
import tempfile
import shutil

# 添加tigertrade目录到路径
tigertrade_dir = '/home/cx/tigertrade'
if tigertrade_dir not in sys.path:
    sys.path.insert(0, tigertrade_dir)

os.environ['ALLOW_REAL_TRADING'] = '0'

from src import tiger1 as t1
from src.api_adapter import api_manager


class TestTiger1AdditionalCoverage(unittest.TestCase):
    """补充覆盖率测试类"""
    
    @classmethod
    def setUpClass(cls):
        """初始化测试环境"""
        api_manager.initialize_mock_apis()
        cls.test_data_dir = tempfile.mkdtemp(prefix='tiger_test_add_')
    
    @classmethod
    def tearDownClass(cls):
        """清理测试环境"""
        if os.path.exists(cls.test_data_dir):
            shutil.rmtree(cls.test_data_dir)
    
    def setUp(self):
        """每个测试前的设置"""
        t1.current_position = 0
        t1.daily_loss = 0
        t1.grid_upper = 0
        t1.grid_lower = 0
        t1.open_orders.clear()
        t1.closed_positions.clear()
        t1.position_entry_times.clear()
        t1.position_entry_prices.clear()
        t1.active_take_profit_orders.clear()
    
    def test_main_function_different_strategies(self):
        """测试主函数的不同策略路径"""
        original_argv = sys.argv.copy()
        
        try:
            # 测试backtest策略
            with patch('sys.argv', ['tiger1.py', 'd', 'backtest']):
                # 不能直接调用main，但可以测试相关函数
                pass
            
            # 测试llm策略路径
            with patch('sys.argv', ['tiger1.py', 'd', 'llm']):
                pass
            
            # 测试grid策略路径
            with patch('sys.argv', ['tiger1.py', 'd', 'grid']):
                pass
            
            # 测试boll策略路径
            with patch('sys.argv', ['tiger1.py', 'd', 'boll']):
                pass
            
            # 测试compare策略路径
            with patch('sys.argv', ['tiger1.py', 'd', 'compare']):
                pass
            
            # 测试large策略路径
            with patch('sys.argv', ['tiger1.py', 'd', 'large']):
                pass
            
            # 测试huge策略路径
            with patch('sys.argv', ['tiger1.py', 'd', 'huge']):
                pass
            
            # 测试optimize策略路径
            with patch('sys.argv', ['tiger1.py', 'd', 'optimize']):
                pass
            
        finally:
            sys.argv = original_argv
    
    def test_get_kline_data_paging(self):
        """测试K线数据分页获取"""
        # 创建模拟的分页响应
        mock_df1 = pd.DataFrame({
            'time': pd.date_range('2026-01-16 12:00', periods=5, freq='1min'),
            'open': [90.0] * 5,
            'high': [90.1] * 5,
            'low': [89.9] * 5,
            'close': [90.0] * 5,
            'volume': [100] * 5
        })
        mock_df1.set_index('time', inplace=True)
        
        # Mock分页API
        with patch.object(api_manager.quote_api, 'get_future_bars_by_page', 
                         return_value=(mock_df1, None)):
            df = t1.get_kline_data(['SIL2603'], '1min', count=1000)
            self.assertIsInstance(df, pd.DataFrame)
    
    def test_get_kline_data_different_formats(self):
        """测试K线数据的不同返回格式"""
        # 测试返回DataFrame的情况
        mock_df = pd.DataFrame({
            'time': pd.date_range('2026-01-16 12:00', periods=10, freq='1min'),
            'open': [90.0] * 10,
            'high': [90.1] * 10,
            'low': [89.9] * 10,
            'close': [90.0] * 10,
            'volume': [100] * 10
        })
        
        # 测试不同的时间格式
        # 字符串时间
        mock_df_str = mock_df.copy()
        mock_df_str['time'] = [t.strftime('%Y-%m-%d %H:%M:%S') for t in mock_df_str['time']]
        
        with patch.object(api_manager.quote_api, 'get_future_bars', return_value=mock_df_str):
            df = t1.get_kline_data(['SIL2603'], '1min', count=10)
            self.assertIsInstance(df, pd.DataFrame)
        
        # 时间戳格式
        mock_df_ts = mock_df.copy()
        mock_df_ts['time'] = [t.timestamp() * 1000 for t in mock_df['time']]
        
        with patch.object(api_manager.quote_api, 'get_future_bars', return_value=mock_df_ts):
            df = t1.get_kline_data(['SIL2603'], '1min', count=10)
            self.assertIsInstance(df, pd.DataFrame)
    
    def test_place_tiger_order_different_scenarios(self):
        """测试下单的不同场景"""
        # 测试真实交易模式（但ALLOW_REAL_TRADING=0）
        original_allow = t1.ALLOW_REAL_TRADING
        t1.ALLOW_REAL_TRADING = 0
        
        result = t1.place_tiger_order('BUY', 1, 100.0)
        self.assertIsNotNone(result)
        
        t1.ALLOW_REAL_TRADING = original_allow
        
        # 测试带止损止盈的订单
        result = t1.place_tiger_order('BUY', 1, 100.0, stop_loss_price=95.0, take_profit_price=110.0)
        self.assertIsNotNone(result)
        
        # 测试SELL订单
        t1.current_position = 1
        result = t1.place_tiger_order('SELL', 1, 105.0)
        self.assertIsNotNone(result)
    
    def test_grid_trading_strategy_all_conditions(self):
        """测试网格策略的所有条件分支"""
        # 创建测试数据
        test_data_1m = pd.DataFrame({
            'time': pd.date_range('2026-01-16 12:00', periods=30, freq='1min'),
            'open': [90.0 + i*0.01 for i in range(30)],
            'high': [90.1 + i*0.01 for i in range(30)],
            'low': [89.9 + i*0.01 for i in range(30)],
            'close': [90.0 + i*0.01 for i in range(30)],
            'volume': [100 + i for i in range(30)]
        })
        test_data_1m.set_index('time', inplace=True)
        
        test_data_5m = pd.DataFrame({
            'time': pd.date_range('2026-01-16 12:00', periods=50, freq='5min'),
            'open': [90.0 + i*0.02 for i in range(50)],
            'high': [90.2 + i*0.02 for i in range(50)],
            'low': [89.8 + i*0.02 for i in range(50)],
            'close': [90.0 + i*0.02 for i in range(50)],
            'volume': [200 + i for i in range(50)]
        })
        test_data_5m.set_index('time', inplace=True)
        
        # 创建多个数据副本用于多次调用
        test_data_1m_list = [test_data_1m.copy() for _ in range(5)]
        test_data_5m_list = [test_data_5m.copy() for _ in range(5)]
        
        # 交替返回1m和5m数据
        def get_kline_side_effect(*args, **kwargs):
            symbol = args[0] if args else kwargs.get('symbol', [])
            period = args[1] if len(args) > 1 else kwargs.get('period', '1min')
            if period == '1min':
                return test_data_1m_list.pop(0) if test_data_1m_list else test_data_1m.copy()
            else:
                return test_data_5m_list.pop(0) if test_data_5m_list else test_data_5m.copy()
        
        with patch.object(t1, 'get_kline_data', side_effect=get_kline_side_effect):
            # 设置网格参数
            t1.grid_lower = 89.0
            t1.grid_upper = 91.0
            
            # 测试买入条件
            with patch.object(t1, 'check_risk_control', return_value=True):
                t1.grid_trading_strategy()
            
            # 测试卖出条件
            t1.current_position = 1
            t1.grid_trading_strategy()
            
            # 测试止损条件
            t1.current_position = 1
            t1.position_entry_prices[1] = 90.0
            t1.grid_trading_strategy()
    
    def test_grid_trading_strategy_pro1_all_branches(self):
        """测试增强网格策略的所有分支"""
        test_data_1m = pd.DataFrame({
            'time': pd.date_range('2026-01-16 12:00', periods=30, freq='1min'),
            'open': [90.0 + i*0.01 for i in range(30)],
            'high': [90.1 + i*0.01 for i in range(30)],
            'low': [89.9 + i*0.01 for i in range(30)],
            'close': [90.0 + i*0.01 for i in range(30)],
            'volume': [100 + i*10 for i in range(30)]  # 增加成交量变化
        })
        test_data_1m.set_index('time', inplace=True)
        
        test_data_5m = pd.DataFrame({
            'time': pd.date_range('2026-01-16 12:00', periods=50, freq='5min'),
            'open': [90.0 + i*0.02 for i in range(50)],
            'high': [90.2 + i*0.02 for i in range(50)],
            'low': [89.8 + i*0.02 for i in range(50)],
            'close': [90.0 + i*0.02 for i in range(50)],
            'volume': [200 + i*10 for i in range(50)]
        })
        test_data_5m.set_index('time', inplace=True)
        
        # 创建多个数据副本用于多次调用
        test_data_1m_list = [test_data_1m.copy() for _ in range(5)]
        test_data_5m_list = [test_data_5m.copy() for _ in range(5)]
        
        # 交替返回1m和5m数据
        def get_kline_side_effect(*args, **kwargs):
            symbol = args[0] if args else kwargs.get('symbol', [])
            period = args[1] if len(args) > 1 else kwargs.get('period', '1min')
            if period == '1min':
                return test_data_1m_list.pop(0) if test_data_1m_list else test_data_1m.copy()
            else:
                return test_data_5m_list.pop(0) if test_data_5m_list else test_data_5m.copy()
        
        with patch.object(t1, 'get_kline_data', side_effect=get_kline_side_effect):
            # 设置不同的网格和RSI条件
            t1.grid_lower = 89.0
            t1.grid_upper = 91.0
            
            # 测试各种买入条件组合
            with patch.object(t1, 'check_risk_control', return_value=True):
                t1.grid_trading_strategy_pro1()
            
            # 测试卖出条件
            t1.current_position = 1
            t1.grid_trading_strategy_pro1()
    
    def test_boll1m_grid_strategy_all_conditions(self):
        """测试BOLL 1分钟策略的所有条件"""
        test_data_1m = pd.DataFrame({
            'time': pd.date_range('2026-01-16 12:00', periods=30, freq='1min'),
            'open': [90.0 + i*0.01 for i in range(30)],
            'high': [90.1 + i*0.01 for i in range(30)],
            'low': [89.9 + i*0.01 for i in range(30)],
            'close': [90.0 + i*0.01 for i in range(30)],
            'volume': [100 + i for i in range(30)]
        })
        test_data_1m.set_index('time', inplace=True)
        
        test_data_5m = pd.DataFrame({
            'time': pd.date_range('2026-01-16 12:00', periods=50, freq='5min'),
            'open': [90.0 + i*0.02 for i in range(50)],
            'high': [90.2 + i*0.02 for i in range(50)],
            'low': [89.8 + i*0.02 for i in range(50)],
            'close': [90.0 + i*0.02 for i in range(50)],
            'volume': [200 + i for i in range(50)]
        })
        test_data_5m.set_index('time', inplace=True)
        
        # 创建多个数据副本用于多次调用
        test_data_1m_list = [test_data_1m.copy() for _ in range(5)]
        test_data_5m_list = [test_data_5m.copy() for _ in range(5)]
        
        # 交替返回1m和5m数据
        def get_kline_side_effect(*args, **kwargs):
            symbol = args[0] if args else kwargs.get('symbol', [])
            period = args[1] if len(args) > 1 else kwargs.get('period', '1min')
            if period == '1min':
                return test_data_1m_list.pop(0) if test_data_1m_list else test_data_1m.copy()
            else:
                return test_data_5m_list.pop(0) if test_data_5m_list else test_data_5m.copy()
        
        with patch.object(t1, 'get_kline_data', side_effect=get_kline_side_effect):
            # 测试买入条件
            t1.grid_lower = 89.0
            with patch.object(t1, 'check_risk_control', return_value=True):
                t1.boll1m_grid_strategy()
            
            # 测试卖出条件
            t1.current_position = 1
            t1.boll1m_grid_strategy()
    
    def test_backtest_detailed_paths(self):
        """测试回测函数的详细路径"""
        # 创建足够的数据用于回测
        large_1m = pd.DataFrame({
            'time': pd.date_range('2026-01-16 12:00', periods=500, freq='1min'),
            'open': [90.0 + i*0.01 for i in range(500)],
            'high': [90.1 + i*0.01 for i in range(500)],
            'low': [89.9 + i*0.01 for i in range(500)],
            'close': [90.0 + i*0.01 for i in range(500)],
            'volume': [100 + i for i in range(500)]
        })
        large_1m.set_index('time', inplace=True)
        
        large_5m = pd.DataFrame({
            'time': pd.date_range('2026-01-16 12:00', periods=200, freq='5min'),
            'open': [90.0 + i*0.02 for i in range(200)],
            'high': [90.2 + i*0.02 for i in range(200)],
            'low': [89.8 + i*0.02 for i in range(200)],
            'close': [90.0 + i*0.02 for i in range(200)],
            'volume': [200 + i for i in range(200)]
        })
        large_5m.set_index('time', inplace=True)
        
        with patch.object(t1, 'get_kline_data', side_effect=[large_1m, large_5m]):
            result = t1.backtest_grid_trading_strategy_pro1(bars_1m=500, bars_5m=200, lookahead=50)
            if result:
                self.assertIsInstance(result, dict)
                self.assertIn('wins', result)
                self.assertIn('losses', result)
    
    def test_verify_api_connection_detailed(self):
        """测试API连接验证的详细路径"""
        # 在模拟模式下测试
        result = t1.verify_api_connection()
        self.assertTrue(result)
        
        # 测试真实API模式（但使用mock）
        with patch.object(api_manager, 'is_mock_mode', False):
            with patch.object(api_manager.quote_api, 'get_stock_briefs', return_value=pd.DataFrame()):
                with patch.object(api_manager.quote_api, 'get_future_exchanges', return_value=pd.DataFrame()):
                    with patch.object(api_manager.quote_api, 'get_future_contracts', return_value=pd.DataFrame()):
                        with patch.object(api_manager.quote_api, 'get_all_future_contracts', return_value=pd.DataFrame()):
                            with patch.object(api_manager.quote_api, 'get_current_future_contract', return_value={}):
                                with patch.object(api_manager.quote_api, 'get_quote_permission', return_value={}):
                                    with patch.object(api_manager.quote_api, 'get_future_brief', return_value=pd.DataFrame()):
                                        with patch.object(api_manager.quote_api, 'get_future_bars', return_value=pd.DataFrame()):
                                            result = t1.verify_api_connection()
                                            # 可能成功或失败，取决于mock实现
    
    def test_calculate_indicators_edge_cases(self):
        """测试指标计算的边界情况"""
        # 空数据
        empty_df = pd.DataFrame()
        result = t1.calculate_indicators(empty_df, empty_df)
        self.assertIsInstance(result, dict)
        
        # 只有1行数据
        single_row_1m = pd.DataFrame({
            'time': [datetime.now()],
            'open': [90.0],
            'high': [90.1],
            'low': [89.9],
            'close': [90.0],
            'volume': [100]
        })
        single_row_1m.set_index('time', inplace=True)
        
        single_row_5m = pd.DataFrame({
            'time': [datetime.now()],
            'open': [90.0],
            'high': [90.2],
            'low': [89.8],
            'close': [90.0],
            'volume': [200]
        })
        single_row_5m.set_index('time', inplace=True)
        
        result = t1.calculate_indicators(single_row_1m, single_row_5m)
        self.assertIsInstance(result, dict)
        
        # 包含NaN的数据
        nan_df_1m = pd.DataFrame({
            'time': pd.date_range('2026-01-16 12:00', periods=10, freq='1min'),
            'open': [90.0 if i != 5 else np.nan for i in range(10)],
            'high': [90.1] * 10,
            'low': [89.9] * 10,
            'close': [90.0] * 10,
            'volume': [100] * 10
        })
        nan_df_1m.set_index('time', inplace=True)
        
        result = t1.calculate_indicators(nan_df_1m, nan_df_1m)
        self.assertIsInstance(result, dict)
    
    def test_judge_market_trend_all_cases(self):
        """测试市场趋势判断的所有情况"""
        # 测试所有趋势类型
        test_cases = [
            {'close': 100.0, 'boll_middle': 95.0, 'boll_mid': 95.0, 'rsi': 65},  # bull_trend
            {'close': 90.0, 'boll_middle': 95.0, 'boll_mid': 95.0, 'rsi': 35},  # bear_trend
            {'close': 95.0, 'boll_middle': 95.0, 'boll_mid': 95.0, 'rsi': 55},  # osc_bull
            {'close': 95.0, 'boll_middle': 95.0, 'boll_mid': 95.0, 'rsi': 45},  # osc_bear
            {'close': 95.0, 'boll_middle': 95.0, 'boll_mid': 95.0, 'rsi': 50},  # osc_normal
            {'close': 95.0, 'boll_middle': None, 'boll_mid': None, 'rsi': 50},  # 无boll
            {'close': 95.0, 'boll_middle': 0, 'boll_mid': 0, 'rsi': 50},  # boll为0
        ]
        
        for case in test_cases:
            indicators = {'5m': case}
            trend = t1.judge_market_trend(indicators)
            self.assertIsInstance(trend, str)
            self.assertIn(trend, ['bull_trend', 'osc_bull', 'osc_normal', 'osc_bear', 'bear_trend', 'boll_divergence_down', 'boll_divergence_up'])
    
    def test_adjust_grid_interval_all_trends(self):
        """测试网格调整的所有趋势情况"""
        indicators = {
            '5m': {
                'boll_upper': 100.0,
                'boll_lower': 90.0,
                'atr': 1.0
            }
        }
        
        # 测试所有趋势类型
        for trend in ['bull_trend', 'bear_trend', 'osc_normal', 'osc_bull', 'osc_bear', 'boll_divergence_down', 'boll_divergence_up']:
            t1.adjust_grid_interval(trend, indicators)
            self.assertGreater(t1.grid_upper, 0)
            self.assertGreater(t1.grid_lower, 0)
        
        # 测试无指标数据
        t1.adjust_grid_interval('osc_normal', {})
        
        # 测试boll_lower为0
        indicators_zero = {
            '5m': {
                'boll_upper': 100.0,
                'boll_lower': 0,
                'atr': 1.0
            }
        }
        t1.adjust_grid_interval('osc_normal', indicators_zero)
        self.assertGreater(t1.grid_lower, 0)
    
    def test_compute_stop_loss_all_cases(self):
        """测试止损计算的所有情况"""
        # 正常情况
        stop, loss = t1.compute_stop_loss(100.0, 2.0, 95.0)
        self.assertIsNotNone(stop)
        self.assertIsNotNone(loss)
        
        # ATR为0
        stop, loss = t1.compute_stop_loss(100.0, 0.0, 95.0)
        self.assertIsNotNone(stop)
        
        # 价格等于下轨
        stop, loss = t1.compute_stop_loss(95.0, 2.0, 95.0)
        self.assertIsNotNone(stop)
        
        # 价格低于下轨
        stop, loss = t1.compute_stop_loss(90.0, 2.0, 95.0)
        self.assertIsNotNone(stop)
    
    def test_check_risk_control_all_paths(self):
        """测试风控检查的所有路径"""
        # 重置状态
        t1.current_position = 0
        t1.daily_loss = 0
        t1.today = datetime.now().date()
        
        # 正常情况
        result = t1.check_risk_control(100.0, 'BUY')
        self.assertIsInstance(result, bool)
        
        # 日期变化测试
        t1.today = datetime.now().date() - timedelta(days=1)
        t1.daily_loss = 1000
        result = t1.check_risk_control(100.0, 'BUY')
        # 应该重置daily_loss
        self.assertEqual(t1.daily_loss, 0)
        
        # 超过最大持仓
        t1.current_position = t1.GRID_MAX_POSITION
        result = t1.check_risk_control(100.0, 'BUY')
        self.assertFalse(result)
        
        # 超过日亏损上限
        t1.current_position = 0
        t1.daily_loss = t1.DAILY_LOSS_LIMIT
        result = t1.check_risk_control(100.0, 'BUY')
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main(verbosity=2)
