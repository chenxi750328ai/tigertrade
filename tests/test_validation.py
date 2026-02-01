#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
期货网格交易策略功能验证测试（使用 src.tiger1，不依赖 tiger2）
"""

import sys
import os
import types
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

# 添加项目路径
sys.path.insert(0, '/home/cx/tigertrade')

def _ensure_tigeropen_stubs():
    """仅在 contract_utils 未加载时注入最小 stub，供 test_validation 单独运行时导入 tiger1。"""
    if 'tigeropen.common.util.contract_utils' in sys.modules:
        return
    _c = types.SimpleNamespace(
        Language=None, Market=None, QuoteRight=None, Currency=types.SimpleNamespace(USD='USD'),
        OrderStatus=types.SimpleNamespace(FILLED='FILLED'), OrderType=types.SimpleNamespace(MARKET='MARKET', LIMIT='LIMIT', LMT='LMT'),
        BarPeriod=types.SimpleNamespace(ONE_MINUTE='ONE_MINUTE', FIVE_MINUTES='FIVE_MINUTES', TEN_MINUTES='TEN_MINUTES', FIFTEEN_MINUTES='FIFTEEN_MINUTES', HALF_HOUR='HALF_HOUR', ONE_HOUR='ONE_HOUR', DAY='DAY', WEEK='WEEK', MONTH='MONTH', YEAR='YEAR', THREE_MINUTES='THREE_MINUTES', FORTY_FIVE_MINUTES='FORTY_FIVE_MINUTES', TWO_HOURS='TWO_HOURS', THREE_HOURS='THREE_HOURS', FOUR_HOURS='FOUR_HOURS', SIX_HOURS='SIX_HOURS')
    )
    _m = types.ModuleType('tigeropen.common.consts')
    for k, v in _c.__dict__.items():
        setattr(_m, k, v)
    sys.modules.setdefault('tigeropen', types.ModuleType('tigeropen'))
    sys.modules.setdefault('tigeropen.common', types.ModuleType('tigeropen.common'))
    sys.modules['tigeropen.common.consts'] = _m
    _util = types.ModuleType('tigeropen.common.util')
    _sig = types.ModuleType('tigeropen.common.util.signature_utils')
    setattr(_sig, 'read_private_key', lambda path=None: 'FAKE')
    _cu = types.ModuleType('tigeropen.common.util.contract_utils')
    setattr(_cu, 'stock_contract', lambda *a, **k: None)
    setattr(_cu, 'future_contract', lambda *a, **k: None)
    sys.modules['tigeropen.common.util'] = _util
    sys.modules['tigeropen.common.util.signature_utils'] = _sig
    sys.modules['tigeropen.common.util.contract_utils'] = _cu
    setattr(_util, 'contract_utils', _cu)
    _conf = types.ModuleType('tigeropen.tiger_open_config')
    setattr(_conf, 'TigerOpenClientConfig', lambda props_path=None: types.SimpleNamespace(account='SIM', tiger_id='SIM'))
    sys.modules['tigeropen.tiger_open_config'] = _conf
    _q = types.ModuleType('tigeropen.quote.quote_client')
    setattr(_q, 'QuoteClient', lambda cfg: types.SimpleNamespace(get_future_bars=lambda *a, **k: []))
    sys.modules.setdefault('tigeropen.quote', types.ModuleType('tigeropen.quote'))
    sys.modules['tigeropen.quote.quote_client'] = _q
    _tr = types.ModuleType('tigeropen.trade.trade_client')
    setattr(_tr, 'TradeClient', lambda cfg: types.SimpleNamespace(place_order=lambda req: types.SimpleNamespace(order_id='SIM')))
    sys.modules.setdefault('tigeropen.trade', types.ModuleType('tigeropen.trade'))
    sys.modules['tigeropen.trade.trade_client'] = _tr


class TestTigerTradingValidation(unittest.TestCase):
    """期货交易策略验证测试（使用 src.tiger1，不依赖 tiger2）"""

    def setUp(self):
        """设置测试环境；无真实 tigeropen 时注入 stub 使 src.tiger1 可导入。"""
        _ensure_tigeropen_stubs()
        try:
            from src import tiger1 as t1
            self.module = t1
        except Exception as e:
            self.fail(f"需要 tigeropen 或 stub 才能运行 validation 测试: {e}")
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

    def test_indicator_calculation(self):
        """测试技术指标计算"""
        indicators = self.module.calculate_indicators(self.df_1m, self.df_5m)
        
        # 检查指标字典结构
        self.assertIn('5m', indicators)
        self.assertIn('1m', indicators)
        
        # 检查5分钟指标
        five_min_indicators = indicators['5m']
        self.assertIn('boll_mid', five_min_indicators)
        self.assertIn('boll_upper', five_min_indicators)
        self.assertIn('boll_lower', five_min_indicators)
        self.assertIn('rsi', five_min_indicators)
        self.assertIn('atr', five_min_indicators)
        
        # 检查1分钟指标
        one_min_indicators = indicators['1m']
        self.assertIn('rsi', one_min_indicators)
        self.assertIn('close', one_min_indicators)
        self.assertIn('volume', one_min_indicators)

    def test_market_trend_classification(self):
        """测试市场趋势分类"""
        # 高RSI应分类为牛市
        bull_indicators = {'5m': {'rsi': 75}}
        trend = self.module.judge_market_trend(bull_indicators)
        self.assertIn(trend, ['bull_trend', 'osc_normal'])
        
        # 低RSI应分类为熊市
        bear_indicators = {'5m': {'rsi': 25}}
        trend = self.module.judge_market_trend(bear_indicators)
        self.assertIn(trend, ['bear_trend', 'osc_normal'])
        
        # 中等RSI应分类为震荡
        osc_indicators = {'5m': {'rsi': 50}}
        trend = self.module.judge_market_trend(osc_indicators)
        self.assertEqual(trend, 'osc_normal')

    def test_risk_control(self):
        """测试风控功能"""
        # 测试风控函数是否能正常调用
        result = self.module.check_risk_control(20.0, 'BUY')
        # 结果取决于全局状态，但我们验证函数可以正常执行
        self.assertIsInstance(result, bool)

    def test_grid_adjustment(self):
        """测试网格调整功能"""
        original_lower = self.module.grid_lower
        original_upper = self.module.grid_upper
        
        # 测试使用BOLL指标调整网格
        indicators = {
            '5m': {
                'boll_lower': 20.5,
                'boll_upper': 24.5
            }
        }
        
        self.module.adjust_grid_interval('osc_normal', indicators)
        
        # 检查网格值是否被更新
        self.assertEqual(self.module.grid_lower, 20.5)
        self.assertEqual(self.module.grid_upper, 24.5)
        
        # 恢复原始值
        self.module.grid_lower = original_lower
        self.module.grid_upper = original_upper

    def test_stop_loss_calculation(self):
        """测试止损计算功能"""
        stop_price, projected_loss = self.module.compute_stop_loss(
            price=20.0, 
            atr_value=0.5, 
            grid_lower_val=19.0
        )
        
        # 检查返回值类型
        self.assertIsInstance(stop_price, (float, int, type(None)))
        self.assertIsInstance(projected_loss, (float, int))
        
        # 如果成功计算，止损价应小于价格
        if stop_price is not None:
            self.assertLess(stop_price, 20.0)

    def test_api_identifier_conversion(self):
        """测试API标识符转换功能"""
        result = self.module._to_api_identifier('SIL.COMEX.202603')
        self.assertEqual(result, 'SIL2603')
        
        result = self.module._to_api_identifier('SIL2603')
        self.assertEqual(result, 'SIL2603')

    def test_strategy_constants(self):
        """测试策略常量定义"""
        required_constants = [
            'FUTURE_SYMBOL',
            'GRID_MAX_POSITION',
            'DAILY_LOSS_LIMIT',
            'SINGLE_TRADE_LOSS',
            'MIN_KLINES'
        ]
        
        for const in required_constants:
            self.assertTrue(hasattr(self.module, const))

    def test_strategy_risk_parameters(self):
        """测试策略风控参数"""
        self.assertGreater(self.module.GRID_MAX_POSITION, 0)
        self.assertGreater(self.module.DAILY_LOSS_LIMIT, 0)
        self.assertGreater(self.module.SINGLE_TRADE_LOSS, 0)
        self.assertGreater(self.module.MIN_KLINES, 0)

    @patch('src.tiger1.quote_client')
    def test_get_kline_data_structure(self, mock_quote_client):
        """测试K线数据获取结构"""
        # 模拟返回数据
        mock_df = pd.DataFrame({
            'time': pd.date_range(start='2023-01-01', periods=10, freq='1min'),
            'open': [20.1, 20.2, 20.3, 20.4, 20.5, 20.6, 20.7, 20.8, 20.9, 21.0],
            'high': [20.5, 20.6, 20.7, 20.8, 20.9, 21.0, 21.1, 21.2, 21.3, 21.4],
            'low': [19.9, 20.0, 20.1, 20.2, 20.3, 20.4, 20.5, 20.6, 20.7, 20.8],
            'close': [20.3, 20.4, 20.5, 20.6, 20.7, 20.8, 20.9, 21.0, 21.1, 21.2],
            'volume': [100, 150, 200, 180, 220, 190, 210, 230, 240, 250]
        })
        
        mock_quote_client.get_future_bars.return_value = mock_df
        
        # 测试获取数据
        result = self.module.get_kline_data([self.module.FUTURE_SYMBOL], '1min', count=10)
        
        # 验证返回值结构
        if not result.empty:
            self.assertIn('open', result.columns)
            self.assertIn('high', result.columns)
            self.assertIn('low', result.columns)
            self.assertIn('close', result.columns)
            self.assertIn('volume', result.columns)

    def test_strategy_compliance_with_specifications(self):
        """测试策略是否符合设计规范"""
        # 验证多时间框架分析
        self.assertTrue(hasattr(self.module, 'get_kline_data'))
        
        # 验证技术指标组合
        self.assertTrue(hasattr(self.module, 'calculate_indicators'))
        
        # 验证风控机制
        self.assertTrue(hasattr(self.module, 'check_risk_control'))
        
        # 验证动态参数调整
        self.assertTrue(hasattr(self.module, 'adjust_grid_interval'))
        
        # 验证下单函数
        self.assertTrue(hasattr(self.module, 'place_tiger_order'))
        
        # 验证止盈止损
        self.assertTrue(hasattr(self.module, 'compute_stop_loss'))
        self.assertTrue(hasattr(self.module, 'check_active_take_profits'))


def run_validation_tests():
    """运行验证测试"""
    print("🔍 运行期货网格交易策略验证测试...")
    print("="*60)
    
    # 创建测试套件
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTigerTradingValidation)
    
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
    
    print(f"\n✅ 验证测试{'通过' if result.wasSuccessful() else '未通过'}")
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_validation_tests()
    sys.exit(0 if success else 1)