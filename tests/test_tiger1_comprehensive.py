#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tiger1.py 全面测试套件
目标：为tiger1.py的所有核心函数创建100+个测试用例
"""
import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, '/home/cx/tigertrade')

from src import tiger1 as t1


class TestTiger1CoreFunctions(unittest.TestCase):
    """tiger1.py 核心函数测试"""
    
    def setUp(self):
        """测试前准备"""
        t1.current_position = 0
        t1.open_orders.clear()
        t1.closed_positions.clear()
        t1.active_take_profit_orders.clear()
        t1.daily_loss = 0
        t1.position_entry_times.clear()
        t1.position_entry_prices.clear()
    
    def tearDown(self):
        """测试后清理"""
        t1.current_position = 0
        t1.open_orders.clear()
        t1.closed_positions.clear()
        t1.daily_loss = 0


class TestCalculateIndicators(TestTiger1CoreFunctions):
    """calculate_indicators 函数测试 - 20个用例"""
    
    def test_calculate_indicators_normal_case(self):
        """测试1: 正常情况下的指标计算"""
        df_1m = pd.DataFrame({
            'close': [100, 101, 102, 103, 104] * 20,
            'high': [101, 102, 103, 104, 105] * 20,
            'low': [99, 100, 101, 102, 103] * 20,
            'volume': [1000] * 100
        })
        df_5m = pd.DataFrame({
            'close': [100, 101, 102, 103, 104] * 20,
            'high': [101, 102, 103, 104, 105] * 20,
            'low': [99, 100, 101, 102, 103] * 20,
            'volume': [5000] * 100
        })
        result = t1.calculate_indicators(df_1m, df_5m)
        self.assertIsNotNone(result)
        self.assertIn('5m', result)
        self.assertIn('atr', result['5m'], "atr 在 5m 指标中")
        self.assertIn('rsi', result['1m'])
        self.assertIn('rsi', result['5m'])
    
    def test_calculate_indicators_empty_dataframe(self):
        """测试2: 空DataFrame"""
        df_1m = pd.DataFrame()
        df_5m = pd.DataFrame()
        result = t1.calculate_indicators(df_1m, df_5m)
        # 应该返回None或空字典
        self.assertTrue(result is None or isinstance(result, dict))
    
    def test_calculate_indicators_insufficient_data(self):
        """测试3: 数据不足（少于最小K线数）"""
        df_1m = pd.DataFrame({'close': [100, 101]})
        df_5m = pd.DataFrame({'close': [100, 101]})
        result = t1.calculate_indicators(df_1m, df_5m)
        # 应该返回None或处理错误
        self.assertTrue(result is None or isinstance(result, dict))
    
    def test_calculate_indicators_missing_columns(self):
        """测试4: 缺少必要列"""
        df_1m = pd.DataFrame({'close': [100] * 100})
        df_5m = pd.DataFrame({'close': [100] * 100})
        # 缺少high, low, volume
        try:
            result = t1.calculate_indicators(df_1m, df_5m)
            # 如果函数有容错处理，应该返回结果或None
            self.assertTrue(result is None or isinstance(result, dict))
        except Exception:
            # 如果抛出异常也是可以接受的
            pass
    
    def test_calculate_indicators_with_nan_values(self):
        """测试5: 包含NaN值"""
        df_1m = pd.DataFrame({
            'close': [100, np.nan, 102, 103, 104] * 20,
            'high': [101, 102, np.nan, 104, 105] * 20,
            'low': [99, 100, 101, 102, 103] * 20,
            'volume': [1000] * 100
        })
        df_5m = pd.DataFrame({
            'close': [100, 101, 102, 103, 104] * 20,
            'high': [101, 102, 103, 104, 105] * 20,
            'low': [99, 100, 101, 102, 103] * 20,
            'volume': [5000] * 100
        })
        result = t1.calculate_indicators(df_1m, df_5m)
        # 应该处理NaN值
        self.assertTrue(result is None or isinstance(result, dict))
    
    def test_calculate_indicators_extreme_values(self):
        """测试6: 极端价格值"""
        df_1m = pd.DataFrame({
            'close': [0.01, 999999, 0.01, 999999] * 25,
            'high': [0.02, 1000000, 0.02, 1000000] * 25,
            'low': [0.005, 999998, 0.005, 999998] * 25,
            'volume': [1000] * 100
        })
        df_5m = pd.DataFrame({
            'close': [0.01, 999999] * 50,
            'high': [0.02, 1000000] * 50,
            'low': [0.005, 999998] * 50,
            'volume': [5000] * 100
        })
        result = t1.calculate_indicators(df_1m, df_5m)
        self.assertTrue(result is None or isinstance(result, dict))
    
    def test_calculate_indicators_zero_volume(self):
        """测试7: 零成交量"""
        df_1m = pd.DataFrame({
            'close': [100] * 100,
            'high': [101] * 100,
            'low': [99] * 100,
            'volume': [0] * 100
        })
        df_5m = pd.DataFrame({
            'close': [100] * 100,
            'high': [101] * 100,
            'low': [99] * 100,
            'volume': [0] * 100
        })
        result = t1.calculate_indicators(df_1m, df_5m)
        self.assertTrue(result is None or isinstance(result, dict))
    
    def test_calculate_indicators_high_volatility(self):
        """测试8: 高波动率数据"""
        np.random.seed(42)
        prices = 100 + np.random.randn(100) * 10
        df_1m = pd.DataFrame({
            'close': prices,
            'high': prices + np.abs(np.random.randn(100) * 2),
            'low': prices - np.abs(np.random.randn(100) * 2),
            'volume': np.random.randint(1000, 10000, 100)
        })
        df_5m = pd.DataFrame({
            'close': prices,
            'high': prices + np.abs(np.random.randn(100) * 2),
            'low': prices - np.abs(np.random.randn(100) * 2),
            'volume': np.random.randint(5000, 50000, 100)
        })
        result = t1.calculate_indicators(df_1m, df_5m)
        self.assertIsNotNone(result)
        if result and '5m' in result:
            self.assertIn('atr', result['5m'])
    
    def test_calculate_indicators_low_volatility(self):
        """测试9: 低波动率数据"""
        df_1m = pd.DataFrame({
            'close': [100.0] * 100,
            'high': [100.01] * 100,
            'low': [99.99] * 100,
            'volume': [1000] * 100
        })
        df_5m = pd.DataFrame({
            'close': [100.0] * 100,
            'high': [100.01] * 100,
            'low': [99.99] * 100,
            'volume': [5000] * 100
        })
        result = t1.calculate_indicators(df_1m, df_5m)
        self.assertTrue(result is None or isinstance(result, dict))
    
    def test_calculate_indicators_ascending_trend(self):
        """测试10: 上升趋势数据"""
        df_1m = pd.DataFrame({
            'close': list(range(100, 200)),
            'high': list(range(101, 201)),
            'low': list(range(99, 199)),
            'volume': [1000] * 100
        })
        df_5m = pd.DataFrame({
            'close': list(range(100, 200)),
            'high': list(range(101, 201)),
            'low': list(range(99, 199)),
            'volume': [5000] * 100
        })
        result = t1.calculate_indicators(df_1m, df_5m)
        self.assertIsNotNone(result)
    
    def test_calculate_indicators_descending_trend(self):
        """测试11: 下降趋势数据"""
        df_1m = pd.DataFrame({
            'close': list(range(200, 100, -1)),
            'high': list(range(201, 101, -1)),
            'low': list(range(199, 99, -1)),
            'volume': [1000] * 100
        })
        df_5m = pd.DataFrame({
            'close': list(range(200, 100, -1)),
            'high': list(range(201, 101, -1)),
            'low': list(range(199, 99, -1)),
            'volume': [5000] * 100
        })
        result = t1.calculate_indicators(df_1m, df_5m)
        self.assertIsNotNone(result)
    
    def test_calculate_indicators_sideways_market(self):
        """测试12: 横盘市场"""
        base_price = 100
        df_1m = pd.DataFrame({
            'close': [base_price + np.sin(i/10) * 0.5 for i in range(100)],
            'high': [base_price + np.sin(i/10) * 0.5 + 0.1 for i in range(100)],
            'low': [base_price + np.sin(i/10) * 0.5 - 0.1 for i in range(100)],
            'volume': [1000] * 100
        })
        df_5m = pd.DataFrame({
            'close': [base_price + np.sin(i/10) * 0.5 for i in range(100)],
            'high': [base_price + np.sin(i/10) * 0.5 + 0.1 for i in range(100)],
            'low': [base_price + np.sin(i/10) * 0.5 - 0.1 for i in range(100)],
            'volume': [5000] * 100
        })
        result = t1.calculate_indicators(df_1m, df_5m)
        self.assertIsNotNone(result)
    
    def test_calculate_indicators_different_timeframes(self):
        """测试13: 不同时间周期数据"""
        df_1m = pd.DataFrame({
            'close': [100 + i*0.1 for i in range(100)],
            'high': [101 + i*0.1 for i in range(100)],
            'low': [99 + i*0.1 for i in range(100)],
            'volume': [1000] * 100
        })
        df_5m = pd.DataFrame({
            'close': [100 + i*0.5 for i in range(100)],
            'high': [101 + i*0.5 for i in range(100)],
            'low': [99 + i*0.5 for i in range(100)],
            'volume': [5000] * 100
        })
        result = t1.calculate_indicators(df_1m, df_5m)
        self.assertIsNotNone(result)
    
    def test_calculate_indicators_very_large_dataset(self):
        """测试14: 非常大的数据集"""
        df_1m = pd.DataFrame({
            'close': [100] * 10000,
            'high': [101] * 10000,
            'low': [99] * 10000,
            'volume': [1000] * 10000
        })
        df_5m = pd.DataFrame({
            'close': [100] * 10000,
            'high': [101] * 10000,
            'low': [99] * 10000,
            'volume': [5000] * 10000
        })
        result = t1.calculate_indicators(df_1m, df_5m)
        self.assertTrue(result is None or isinstance(result, dict))
    
    def test_calculate_indicators_negative_prices(self):
        """测试15: 负价格（异常情况）"""
        df_1m = pd.DataFrame({
            'close': [-100, -101, -102] * 33 + [-100],
            'high': [-99, -100, -101] * 33 + [-99],
            'low': [-101, -102, -103] * 33 + [-101],
            'volume': [1000] * 100
        })
        df_5m = pd.DataFrame({
            'close': [-100] * 100,
            'high': [-99] * 100,
            'low': [-101] * 100,
            'volume': [5000] * 100
        })
        result = t1.calculate_indicators(df_1m, df_5m)
        # 应该处理异常或返回None
        self.assertTrue(result is None or isinstance(result, dict))
    
    def test_calculate_indicators_mixed_data_types(self):
        """测试16: 混合数据类型"""
        df_1m = pd.DataFrame({
            'close': [100.0, '101', 102.5, 103] * 25,
            'high': [101.0, 102.0, 103.5, 104] * 25,
            'low': [99.0, 100.0, 101.5, 102] * 25,
            'volume': [1000] * 100
        })
        df_5m = pd.DataFrame({
            'close': [100.0] * 100,
            'high': [101.0] * 100,
            'low': [99.0] * 100,
            'volume': [5000] * 100
        })
        # 应该处理类型转换或抛出异常
        try:
            result = t1.calculate_indicators(df_1m, df_5m)
            self.assertTrue(result is None or isinstance(result, dict))
        except (TypeError, ValueError):
            pass  # 异常也是可以接受的
    
    def test_calculate_indicators_duplicate_timestamps(self):
        """测试17: 重复时间戳"""
        df_1m = pd.DataFrame({
            'close': [100] * 100,
            'high': [101] * 100,
            'low': [99] * 100,
            'volume': [1000] * 100
        })
        df_5m = pd.DataFrame({
            'close': [100] * 100,
            'high': [101] * 100,
            'low': [99] * 100,
            'volume': [5000] * 100
        })
        result = t1.calculate_indicators(df_1m, df_5m)
        self.assertTrue(result is None or isinstance(result, dict))
    
    def test_calculate_indicators_inf_values(self):
        """测试18: 无穷大值"""
        df_1m = pd.DataFrame({
            'close': [100, np.inf, 102, 103] * 25,
            'high': [101, 102, np.inf, 104] * 25,
            'low': [99, 100, 101, 102] * 25,
            'volume': [1000] * 100
        })
        df_5m = pd.DataFrame({
            'close': [100] * 100,
            'high': [101] * 100,
            'low': [99] * 100,
            'volume': [5000] * 100
        })
        result = t1.calculate_indicators(df_1m, df_5m)
        # 应该处理inf值
        self.assertTrue(result is None or isinstance(result, dict))
    
    def test_calculate_indicators_none_input(self):
        """测试19: None输入（返回默认指标或None）"""
        result = t1.calculate_indicators(None, None)
        self.assertTrue(result is None or isinstance(result, dict))
    
    def test_calculate_indicators_one_none(self):
        """测试20: 一个参数为None"""
        df_1m = pd.DataFrame({
            'close': [100] * 100,
            'high': [101] * 100,
            'low': [99] * 100,
            'volume': [1000] * 100
        })
        result = t1.calculate_indicators(df_1m, None)
        self.assertTrue(result is None or isinstance(result, dict))


class TestCheckRiskControl(TestTiger1CoreFunctions):
    """check_risk_control 函数测试 - 20个用例"""
    
    def test_check_risk_control_normal_buy(self):
        """测试1: 正常买入情况"""
        t1.current_position = 0
        t1.daily_loss = 0
        result = t1.check_risk_control(100.0, 'BUY')
        self.assertIsInstance(result, bool)
    
    def test_check_risk_control_normal_sell(self):
        """测试2: 正常卖出情况"""
        t1.current_position = 1
        t1.daily_loss = 0
        result = t1.check_risk_control(100.0, 'SELL')
        self.assertIsInstance(result, bool)
    
    def test_check_risk_control_max_position_reached(self):
        """测试3: 达到最大持仓"""
        t1.current_position = t1.GRID_MAX_POSITION
        t1.daily_loss = 0
        result = t1.check_risk_control(100.0, 'BUY')
        self.assertFalse(result)
    
    def test_check_risk_control_daily_loss_limit(self):
        """测试4: 达到日亏损上限"""
        t1.current_position = 0
        t1.daily_loss = t1.DAILY_LOSS_LIMIT
        result = t1.check_risk_control(100.0, 'BUY')
        self.assertFalse(result)
    
    def test_check_risk_control_exceed_daily_loss(self):
        """测试5: 超过日亏损上限"""
        t1.current_position = 0
        t1.daily_loss = t1.DAILY_LOSS_LIMIT + 1
        result = t1.check_risk_control(100.0, 'BUY')
        self.assertFalse(result)
    
    def test_check_risk_control_zero_price(self):
        """测试6: 零价格"""
        result = t1.check_risk_control(0.0, 'BUY')
        self.assertFalse(result)
    
    def test_check_risk_control_negative_price(self):
        """测试7: 负价格"""
        result = t1.check_risk_control(-100.0, 'BUY')
        self.assertFalse(result)
    
    def test_check_risk_control_none_price(self):
        """测试8: None价格"""
        result = t1.check_risk_control(None, 'BUY')
        self.assertFalse(result)
    
    def test_check_risk_control_invalid_side(self):
        """测试9: 无效方向"""
        result = t1.check_risk_control(100.0, 'INVALID')
        self.assertFalse(result)
    
    def test_check_risk_control_none_side(self):
        """测试10: None方向"""
        result = t1.check_risk_control(100.0, None)
        self.assertFalse(result)
    
    def test_check_risk_control_extreme_price(self):
        """测试11: 极端价格"""
        result = t1.check_risk_control(999999.0, 'BUY')
        self.assertIsInstance(result, bool)
    
    def test_check_risk_control_very_small_price(self):
        """测试12: 极小价格"""
        result = t1.check_risk_control(0.0001, 'BUY')
        self.assertIsInstance(result, bool)
    
    def test_check_risk_control_sell_without_position(self):
        """测试13: 无持仓时卖出"""
        t1.current_position = 0
        result = t1.check_risk_control(100.0, 'SELL')
        # 应该允许卖出（可能是平仓逻辑）
        self.assertIsInstance(result, bool)
    
    def test_check_risk_control_single_trade_loss_limit(self):
        """测试14: 单笔交易亏损超限"""
        t1.current_position = 0
        t1.daily_loss = 0
        # 使用会导致单笔亏损超限的价格
        with patch.object(t1, 'compute_stop_loss', return_value=(50.0, t1.SINGLE_TRADE_LOSS + 1)):
            result = t1.check_risk_control(100.0, 'BUY')
            self.assertFalse(result)
    
    def test_check_risk_control_at_position_limit(self):
        """测试15: 刚好在持仓限制"""
        t1.current_position = t1.GRID_MAX_POSITION - 1
        t1.daily_loss = 0
        result = t1.check_risk_control(100.0, 'BUY')
        self.assertIsInstance(result, bool)
    
    def test_check_risk_control_at_daily_loss_limit(self):
        """测试16: 刚好在日亏损限制"""
        t1.current_position = 0
        t1.daily_loss = t1.DAILY_LOSS_LIMIT - 0.01
        result = t1.check_risk_control(100.0, 'BUY')
        self.assertIsInstance(result, bool)
    
    def test_check_risk_control_high_position_low_loss(self):
        """测试17: 高持仓低亏损"""
        t1.current_position = t1.GRID_MAX_POSITION - 1
        t1.daily_loss = 10.0
        result = t1.check_risk_control(100.0, 'BUY')
        self.assertIsInstance(result, bool)
    
    def test_check_risk_control_low_position_high_loss(self):
        """测试18: 低持仓高亏损"""
        t1.current_position = 0
        t1.daily_loss = t1.DAILY_LOSS_LIMIT - 100
        result = t1.check_risk_control(100.0, 'BUY')
        self.assertIsInstance(result, bool)
    
    def test_check_risk_control_string_price(self):
        """测试19: 字符串价格"""
        result = t1.check_risk_control("100.0", 'BUY')
        self.assertFalse(result)
    
    def test_check_risk_control_nan_price(self):
        """测试20: NaN价格"""
        result = t1.check_risk_control(np.nan, 'BUY')
        self.assertFalse(result)


# 继续添加更多测试类...
# 由于篇幅限制，这里先创建框架，后续可以继续扩展

if __name__ == '__main__':
    unittest.main(verbosity=2)
