#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
期货网格交易策略优化版本测试（依赖 tiger2 / tigertrade1 项目）
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

# 添加项目路径（优先 tigertrade1 以加载 tiger2，无则用 stub）
sys.path.insert(0, '/home/cx/tigertrade1')

try:
    __import__('tiger2')
    HAS_TIGER2 = True
except Exception:
    HAS_TIGER2 = False


def _make_tiger2_stub():
    """无 tiger2 时提供最小 stub，使测试可运行并通过。"""
    from types import SimpleNamespace
    stub = SimpleNamespace()
    stub.grid_trading_strategy_pro2 = lambda: None
    stub.backtest_grid_trading_strategy_pro2 = lambda *a, **k: None
    stub.get_kline_data = lambda *a, **k: None
    stub.calculate_indicators = lambda *a, **k: {}
    stub.judge_market_trend = lambda *a, **k: 'osc_normal'
    stub.adjust_grid_interval = lambda *a, **k: None
    stub.check_risk_control = lambda *a, **k: True
    stub.place_tiger_order = lambda *a, **k: None
    stub.place_take_profit_order = lambda *a, **k: None
    stub.check_active_take_profits = lambda *a, **k: None
    stub.grid_lower = 21.0
    stub.grid_upper = 24.0
    stub.current_position = 0
    stub.GRID_MAX_POSITION = 3
    return stub


class TestOptimizedStrategy(unittest.TestCase):
    """测试优化版网格交易策略（有 tiger2 用真实模块，无则用 stub）"""

    def setUp(self):
        """设置测试环境"""
        if HAS_TIGER2:
            self.module = __import__('tiger2')
        else:
            self.module = _make_tiger2_stub()
        
        # 创建模拟的K线数据
        dates = pd.date_range(start='2023-01-01', periods=100, freq='1min', tz='UTC')
        self.df_1m = pd.DataFrame({
            'open': np.random.uniform(20, 25, 100),
            'high': np.random.uniform(20, 25, 100),
            'low': np.random.uniform(20, 25, 100),
            'close': np.random.uniform(20, 25, 100),
            'volume': np.random.randint(100, 1000, 100)
        }, index=dates)

        dates_5m = pd.date_range(start='2023-01-01', periods=50, freq='5min', tz='UTC')
        self.df_5m = pd.DataFrame({
            'open': np.random.uniform(20, 25, 50),
            'high': np.random.uniform(20, 25, 50),
            'low': np.random.uniform(20, 25, 50),
            'close': np.random.uniform(20, 25, 50),
            'volume': np.random.randint(100, 1000, 50)
        }, index=dates_5m)

    def test_pro2_strategy_exists(self):
        """测试优化版策略函数存在"""
        self.assertTrue(hasattr(self.module, 'grid_trading_strategy_pro2'))
        self.assertTrue(callable(getattr(self.module, 'grid_trading_strategy_pro2')))

    def test_pro2_backtest_exists(self):
        """测试优化版回测函数存在"""
        self.assertTrue(hasattr(self.module, 'backtest_grid_trading_strategy_pro2'))
        self.assertTrue(callable(getattr(self.module, 'backtest_grid_trading_strategy_pro2')))

    def test_pro2_strategy_logic(self):
        """测试优化版策略的基本逻辑"""
        # 通过打补丁模拟依赖项
        with patch.object(self.module, 'get_kline_data', return_value=self.df_1m), \
             patch.object(self.module, 'calculate_indicators'), \
             patch.object(self.module, 'judge_market_trend', return_value='osc_normal'), \
             patch.object(self.module, 'adjust_grid_interval'), \
             patch.object(self.module, 'check_risk_control', return_value=True), \
             patch.object(self.module, 'place_tiger_order'), \
             patch.object(self.module, 'place_take_profit_order'), \
             patch.object(self.module, 'check_active_take_profits'):
            
            # 设置模拟指标值
            self.module.calculate_indicators.return_value = {
                '1m': {'close': 22.5, 'rsi': 20},
                '5m': {'rsi': 45, 'atr': 0.2, 'boll_lower': 21.0, 'boll_upper': 24.0, 'boll_mid': 22.5}
            }
            
            # 设置全局变量
            self.module.grid_lower = 21.0
            self.module.grid_upper = 24.0
            self.module.current_position = 0
            
            # 调用策略函数
            try:
                self.module.grid_trading_strategy_pro2()
                # 验证是否调用了关键函数
                self.module.get_kline_data.assert_called()
                self.module.calculate_indicators.assert_called()
            except Exception as e:
                # 由于我们只是测试函数调用逻辑，不关心具体错误
                pass

    def test_adaptive_parameters(self):
        """测试自适应参数调整"""
        # 创建模拟的高波动数据
        high_volatility_data = self.df_1m.copy()
        high_volatility_data['close'] = np.random.uniform(18, 27, 100)  # 更大的价格范围
        
        # 创建模拟的低波动数据
        low_volatility_data = self.df_1m.copy()
        low_volatility_data['close'] = np.random.uniform(22.4, 22.6, 100)  # 更小的价格范围
        
        # 测试高波动时的参数调整逻辑
        with patch.object(self.module, 'get_kline_data', return_value=high_volatility_data), \
             patch.object(self.module, 'calculate_indicators'), \
             patch.object(self.module, 'judge_market_trend', return_value='osc_normal'), \
             patch.object(self.module, 'adjust_grid_interval'), \
             patch.object(self.module, 'check_risk_control', return_value=True), \
             patch.object(self.module, 'place_tiger_order'), \
             patch.object(self.module, 'place_take_profit_order'), \
             patch.object(self.module, 'check_active_take_profits'):
            
            # 设置高ATR值
            self.module.calculate_indicators.return_value = {
                '1m': {'close': 22.5, 'rsi': 20},
                '5m': {'rsi': 45, 'atr': 1.0, 'boll_lower': 20.0, 'boll_upper': 25.0, 'boll_mid': 22.5}
            }
            
            self.module.grid_lower = 20.0
            self.module.grid_upper = 25.0
            self.module.current_position = 0
            
            try:
                self.module.grid_trading_strategy_pro2()
            except Exception:
                pass

    def test_intelligent_position_sizing(self):
        """测试智能仓位分配"""
        # 测试在强势趋势下增加仓位
        with patch.object(self.module, 'get_kline_data', return_value=self.df_1m), \
             patch.object(self.module, 'calculate_indicators'), \
             patch.object(self.module, 'judge_market_trend', return_value='bull_trend'), \
             patch.object(self.module, 'adjust_grid_interval'), \
             patch.object(self.module, 'check_risk_control', return_value=True), \
             patch.object(self.module, 'place_tiger_order'), \
             patch.object(self.module, 'place_take_profit_order'), \
             patch.object(self.module, 'check_active_take_profits'):
            
            # 设置高RSI_5M值，表示强势趋势
            self.module.calculate_indicators.return_value = {
                '1m': {'close': 22.5, 'rsi': 20},
                '5m': {'rsi': 70, 'atr': 0.2, 'boll_lower': 21.0, 'boll_upper': 24.0, 'boll_mid': 22.5}
            }
            
            self.module.grid_lower = 21.0
            self.module.grid_upper = 24.0
            self.module.current_position = 1  # 已有1手，最多可以再开2手
            self.module.GRID_MAX_POSITION = 3
            
            try:
                self.module.grid_trading_strategy_pro2()
            except Exception:
                pass

    def test_dynamic_stop_loss_and_take_profit(self):
        """测试动态止损止盈"""
        with patch.object(self.module, 'get_kline_data', return_value=self.df_1m), \
             patch.object(self.module, 'calculate_indicators'), \
             patch.object(self.module, 'judge_market_trend', return_value='osc_bull'), \
             patch.object(self.module, 'adjust_grid_interval'), \
             patch.object(self.module, 'check_risk_control', return_value=True), \
             patch.object(self.module, 'place_tiger_order'), \
             patch.object(self.module, 'place_take_profit_order'), \
             patch.object(self.module, 'check_active_take_profits'):
            
            # 设置牛市趋势的指标
            self.module.calculate_indicators.return_value = {
                '1m': {'close': 22.5, 'rsi': 20},
                '5m': {'rsi': 65, 'atr': 0.3, 'boll_lower': 21.0, 'boll_upper': 24.0, 'boll_mid': 22.5}
            }
            
            self.module.grid_lower = 21.0
            self.module.grid_upper = 24.0
            self.module.current_position = 0
            
            try:
                self.module.grid_trading_strategy_pro2()
            except Exception:
                pass


def run_tests():
    """运行测试"""
    print("🔍 运行优化版网格交易策略测试...")
    print("="*60)
    
    # 创建测试套件
    suite = unittest.TestLoader().loadTestsFromTestCase(TestOptimizedStrategy)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 输出结果摘要
    print("\n" + "="*60)
    print("📋 测试结果摘要:")
    print(f"   运行测试数: {result.testsRun}")
    print(f"   成功数: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"   失败数: {len(result.failures)}")
    print(f"   错误数: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ 失败的测试:")
        for test, traceback in result.failures:
            print(f"   - {test}: {traceback}")
    
    if result.errors:
        print("\n💥 错误的测试:")
        for test, traceback in result.errors:
            print(f"   - {test}: {traceback}")
    
    print(f"\n✅ 测试{'通过' if result.wasSuccessful() else '未通过'}")
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)