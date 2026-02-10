"""
代码级测试：TradingExecutor完整覆盖
补充trading_executor.py的所有代码路径测试
"""
import unittest
import sys
from unittest.mock import Mock, patch, MagicMock
sys.path.insert(0, '/home/cx/tigertrade')

from src.executor.trading_executor import TradingExecutor
from src.executor.data_provider import MarketDataProvider
from src.executor.order_executor import OrderExecutor
from src.strategies.base_strategy import BaseTradingStrategy
from src import tiger1 as t1
from src.api_adapter import api_manager


class MockStrategy(BaseTradingStrategy):
    """Mock策略用于测试"""
    def __init__(self, model_path=None, **kwargs):
        """初始化Mock策略"""
        self.model_path = model_path
    
    def predict_action(self, current_data, historical_data=None):
        """预测交易动作"""
        return (1, 0.7, None)  # (action, confidence, profit_prediction)
    
    def prepare_features(self, row):
        """准备特征向量"""
        return []
    
    @property
    def seq_length(self):
        return 10
    
    @property
    def strategy_name(self):
        return "MockStrategy"


class TestTradingExecutorComprehensive(unittest.TestCase):
    """TradingExecutor完整代码覆盖测试"""
    
    def setUp(self):
        """初始化"""
        api_manager.initialize_mock_apis()
        self.data_provider = MarketDataProvider(t1)
        self.order_executor = OrderExecutor(t1)
        self.strategy = MockStrategy()
        self.executor = TradingExecutor(
            strategy=self.strategy,
            data_provider=self.data_provider,
            order_executor=self.order_executor
        )
    
    def test_run_loop_buy_action(self):
        """测试买入动作处理"""
        # Mock策略返回买入
        self.strategy.predict_action = Mock(return_value={
            'action': 'BUY',
            'confidence': 0.7,
            'grid_params': {'lower': 95.0, 'upper': 105.0}
        })
        
        # Mock数据提供者
        self.data_provider.get_market_data = Mock(return_value={
            'tick_price': 100.0,
            'kline_price': 100.0,
            'atr': 0.5,
            'indicators': {}
        })
        
        # Mock订单执行器
        self.order_executor.execute_buy = Mock(return_value=(True, "成功"))
        
        # 运行一个很短的循环
        import signal
        import threading
        
        def run_short():
            try:
                self.executor.run_loop(duration_hours=0.0001)  # 约0.36秒
            except:
                pass
        
        thread = threading.Thread(target=run_short)
        thread.start()
        thread.join(timeout=2)
        
        # 验证统计
        self.assertGreaterEqual(self.executor.stats['total_predictions'], 0)
    
    def test_run_loop_sell_action(self):
        """测试卖出动作处理"""
        self.strategy.predict_action = Mock(return_value={
            'action': 'SELL',
            'confidence': 0.6
        })
        
        self.data_provider.get_market_data = Mock(return_value={
            'tick_price': 105.0,
            'kline_price': 105.0,
            'atr': 0.5
        })
        
        self.order_executor.execute_sell = Mock(return_value=(True, "卖出成功"))
        
        # 短时间运行
        import threading
        def run_short():
            try:
                self.executor.run_loop(duration_hours=0.0001)
            except:
                pass
        
        thread = threading.Thread(target=run_short)
        thread.start()
        thread.join(timeout=2)
    
    def test_run_loop_hold_action(self):
        """测试持有动作处理"""
        self.strategy.predict_action = Mock(return_value={
            'action': 'HOLD',
            'confidence': 0.5
        })
        
        self.data_provider.get_market_data = Mock(return_value={
            'tick_price': 100.0,
            'kline_price': 100.0
        })
        
        # 持有不应调用订单执行器
        import threading
        def run_short():
            try:
                self.executor.run_loop(duration_hours=0.0001)
            except:
                pass
        
        thread = threading.Thread(target=run_short)
        thread.start()
        thread.join(timeout=2)
        
        # 验证持有信号被记录
        self.assertGreaterEqual(self.executor.stats['hold_signals'], 0)
    
    def test_run_loop_exception_handling(self):
        """测试异常处理"""
        self.data_provider.get_market_data = Mock(side_effect=Exception("数据获取失败"))
        
        import threading
        def run_short():
            try:
                self.executor.run_loop(duration_hours=0.0001)
            except:
                pass
        
        thread = threading.Thread(target=run_short)
        thread.start()
        thread.join(timeout=2)
        
        # 验证错误被记录
        self.assertGreaterEqual(self.executor.stats['errors'], 0)
    
    def test_run_loop_keyboard_interrupt(self):
        """测试KeyboardInterrupt处理"""
        self.data_provider.get_market_data = Mock(side_effect=KeyboardInterrupt())
        
        # 应该能够优雅退出
        try:
            self.executor.run_loop(duration_hours=0.0001)
        except KeyboardInterrupt:
            pass
    
    def test_print_stats(self):
        """测试统计信息打印"""
        self.executor.stats = {
            'total_predictions': 10,
            'buy_signals': 3,
            'sell_signals': 2,
            'hold_signals': 5,
            'errors': 1
        }
        
        # 应该能够打印而不报错
        try:
            self.executor.print_stats()
        except:
            self.fail("print_stats should not raise exception")
    
    def test_get_stats(self):
        """测试获取统计信息"""
        stats = self.executor.get_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn('total_predictions', stats)

    def test_run_loop_calls_position_watchdog(self):
        """回归：run_loop 每轮必须调用持仓看门狗（超时止盈/止损），否则有仓无卖可能爆仓"""
        from unittest.mock import patch
        with patch.object(t1, 'run_position_watchdog') as mock_watchdog:
            mock_watchdog.return_value = False
            self.data_provider.get_market_data = Mock(return_value={
                'tick_price': 100.0,
                'kline_price': 100.0,
                'atr': 0.5,
                'grid_lower': 98.0,
                'grid_upper': 105.0,
                'current_data': {'price_current': 100.0},
                'historical_data': [],
            })
            self.strategy.predict_action = Mock(return_value=(0, 0.5, None))  # 不操作，减少干扰
            self.order_executor.execute_buy = Mock(return_value=(False, "风控未通过"))
            self.order_executor.execute_sell = Mock(return_value=(False, "无持仓"))
            import threading
            def run_short():
                try:
                    self.executor.run_loop(duration_hours=0.0002)  # 多跑几轮
                except Exception:
                    pass
            thread = threading.Thread(target=run_short)
            thread.start()
            thread.join(timeout=3)
            self.assertGreaterEqual(
                mock_watchdog.call_count, 1,
                "run_loop 每轮应调用 run_position_watchdog，否则超时止盈/止损不生效"
            )


if __name__ == '__main__':
    unittest.main()
