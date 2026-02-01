"""
代码级测试：MarketDataProvider完整覆盖
补充data_provider.py的所有代码路径测试
"""
import unittest
import sys
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
sys.path.insert(0, '/home/cx/tigertrade')

from src.executor.data_provider import MarketDataProvider
from src import tiger1 as t1
from src.api_adapter import api_manager


class TestMarketDataProviderComprehensive(unittest.TestCase):
    """MarketDataProvider完整代码覆盖测试"""
    
    def setUp(self):
        """初始化（Provider 接受 symbol，不传则用 t1.FUTURE_SYMBOL）"""
        self.provider = MarketDataProvider()
        api_manager.initialize_mock_apis()
    
    @patch('src.executor.data_provider.t1.get_kline_data')
    @patch('src.executor.data_provider.t1.get_tick_data')
    @patch('src.executor.data_provider.t1.calculate_indicators')
    @patch('src.executor.data_provider.t1.judge_market_trend')
    @patch('src.executor.data_provider.t1.adjust_grid_interval')
    def test_get_market_data_normal(self, mock_adjust, mock_judge, mock_calc, mock_tick, mock_kline):
        """测试正常数据获取"""
        # Mock返回值 - 需要返回包含5分钟和1分钟K线的字典
        df_5m = pd.DataFrame({
            'time': pd.date_range('2026-01-28', periods=10, freq='5min'),
            'open': [100.0] * 10,
            'high': [101.0] * 10,
            'low': [99.0] * 10,
            'close': [100.5] * 10,
            'volume': [100] * 10
        })
        df_1m = pd.DataFrame({
            'time': pd.date_range('2026-01-28', periods=10, freq='1min'),
            'open': [100.0] * 10,
            'high': [101.0] * 10,
            'low': [99.0] * 10,
            'close': [100.5] * 10,
            'volume': [100] * 10
        })
        # get_kline_data 被调用两次：5min、1min，需按顺序返回
        mock_kline.side_effect = [df_5m, df_1m]
        
        mock_tick.return_value = pd.DataFrame({
            'price': [100.5],
            'volume': [50]
        })
        mock_calc.return_value = {
            '5m': {'close': 100.5, 'atr': 0.5, 'rsi': 50, 'high': 101.0, 'low': 99.0, 'open': 100.0, 'volume': 100},
            '1m': {'close': 100.5, 'high': 101.0, 'low': 99.0, 'open': 100.0, 'volume': 100, 'rsi': 50}
        }
        mock_judge.return_value = '震荡'
        mock_adjust.return_value = None
        
        result = self.provider.get_market_data()
        
        self.assertIsNotNone(result)
        if result:
            self.assertIsInstance(result, dict)
    
    @patch('src.executor.data_provider.t1.get_kline_data')
    def test_get_market_data_empty_kline(self, mock_kline):
        """测试空K线数据应抛出 K线数据不足"""
        mock_kline.return_value = pd.DataFrame()
        
        with self.assertRaises(ValueError) as ctx:
            self.provider.get_market_data()
        self.assertIn("K线数据不足", str(ctx.exception))
    
    @patch('src.executor.data_provider.t1.get_kline_data')
    @patch('src.executor.data_provider.t1.get_tick_data')
    @patch('src.executor.data_provider.t1.calculate_indicators')
    @patch('src.executor.data_provider.t1.judge_market_trend')
    @patch('src.executor.data_provider.t1.adjust_grid_interval')
    def test_get_market_data_no_tick(self, mock_adjust, mock_judge, mock_calc, mock_tick, mock_kline):
        """测试无Tick数据时使用K线价格作为 fallback"""
        df = pd.DataFrame({
            'time': pd.date_range('2026-01-28', periods=10, freq='1min'),
            'open': [100.0] * 10,
            'high': [101.0] * 10,
            'low': [99.0] * 10,
            'close': [100.5] * 10,
            'volume': [100] * 10
        })
        mock_kline.side_effect = [df, df]
        mock_tick.return_value = pd.DataFrame()
        mock_calc.return_value = {
            '5m': {'close': 100.5, 'atr': 0.5, 'rsi': 50, 'high': 101.0, 'low': 99.0, 'open': 100.0, 'volume': 100},
            '1m': {'close': 100.5, 'high': 101.0, 'low': 99.0, 'open': 100.0, 'volume': 100, 'rsi': 50}
        }
        mock_judge.return_value = '震荡'
        mock_adjust.return_value = None
        
        result = self.provider.get_market_data()
        
        self.assertIsNotNone(result)
        self.assertEqual(result['tick_price'], 100.5)
    
    @patch('src.executor.data_provider.t1.get_kline_data')
    @patch('src.executor.data_provider.t1.get_tick_data')
    @patch('src.executor.data_provider.t1.calculate_indicators')
    def test_get_market_data_no_indicators(self, mock_calc, mock_tick, mock_kline):
        """测试无指标数据应抛出 指标计算失败"""
        df = pd.DataFrame({'close': [100.0] * 10, 'high': [101.0] * 10, 'low': [99.0] * 10, 'open': [100.0] * 10, 'volume': [100] * 10})
        mock_kline.side_effect = [df, df]
        mock_tick.return_value = pd.DataFrame()
        mock_calc.return_value = None
        
        with self.assertRaises(ValueError) as ctx:
            self.provider.get_market_data()
        self.assertIn("指标计算失败", str(ctx.exception))
    
    @patch('src.executor.data_provider.t1.adjust_grid_interval')
    @patch('src.executor.data_provider.t1.judge_market_trend')
    @patch('src.executor.data_provider.t1.calculate_indicators')
    @patch('src.executor.data_provider.t1.get_tick_data')
    @patch('src.executor.data_provider.t1.get_kline_data')
    def test_get_market_data_cache(self, mock_kline, mock_tick, mock_calc, mock_judge, mock_adjust):
        """测试缓存功能：多次调用会追加 historical_data_cache"""
        df = pd.DataFrame({'close': [100.0] * 10, 'high': [101.0] * 10, 'low': [99.0] * 10, 'open': [100.0] * 10, 'volume': [100] * 10})
        # get_market_data 每次调用会请求 5min、1min 各一次；调用两次共 4 次
        mock_kline.side_effect = [df, df, df, df]
        mock_tick.return_value = pd.DataFrame()
        mock_calc.return_value = {
            '5m': {'close': 100.0, 'atr': 0.5, 'rsi': 50, 'high': 101.0, 'low': 99.0, 'open': 100.0, 'volume': 100},
            '1m': {'close': 100.0, 'high': 101.0, 'low': 99.0, 'open': 100.0, 'volume': 100, 'rsi': 50}
        }
        mock_judge.return_value = '震荡'
        mock_adjust.return_value = None
        
        result1 = self.provider.get_market_data()
        result2 = self.provider.get_market_data()
        
        self.assertIsNotNone(result1)
        self.assertIsNotNone(result2)
        self.assertGreaterEqual(len(self.provider.historical_data_cache), 1)
    
    @patch('src.executor.data_provider.t1.get_kline_data')
    def test_get_market_data_exception(self, mock_kline):
        """测试 get_kline_data 抛异常时向上抛出"""
        mock_kline.side_effect = Exception("API错误")
        
        with self.assertRaises(Exception) as ctx:
            self.provider.get_market_data()
        self.assertIn("API错误", str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
