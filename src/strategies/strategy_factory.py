"""
策略工厂 - 根据配置创建策略实例
支持通过配置切换模型，无需修改代码
"""
import os
from typing import Optional, Dict, Any
from src.strategies.base_strategy import BaseTradingStrategy


class StrategyFactory:
    """策略工厂类"""
    
    # 策略注册表：策略名称 -> 策略类
    _strategies = {}
    
    @classmethod
    def register(cls, name: str, strategy_class: type):
        """注册策略类"""
        if not issubclass(strategy_class, BaseTradingStrategy):
            raise TypeError(f"策略类必须继承自BaseTradingStrategy: {strategy_class}")
        cls._strategies[name] = strategy_class
        print(f"✅ 注册策略: {name} -> {strategy_class.__name__}")
    
    @classmethod
    def create(cls, strategy_name: str, model_path: Optional[str] = None, 
               data_dir: str = '/home/cx/trading_data', **kwargs) -> BaseTradingStrategy:
        """
        创建策略实例
        
        Args:
            strategy_name: 策略名称（如 'moe_transformer', 'lstm', 'transformer'等）
            model_path: 模型文件路径（可选，默认使用策略的默认路径）
            data_dir: 数据目录
            **kwargs: 策略特定参数
        
        Returns:
            策略实例
        """
        if strategy_name not in cls._strategies:
            available = ', '.join(cls._strategies.keys())
            raise ValueError(f"未知的策略名称: {strategy_name}。可用策略: {available}")
        
        strategy_class = cls._strategies[strategy_name]
        
        # 如果没有提供model_path，使用默认路径
        if model_path is None:
            default_model_paths = {
                'moe_transformer': os.path.join(data_dir, 'best_moe_transformer.pth'),
                'enhanced_transformer_peft': os.path.join(data_dir, 'best_enhanced_transformer_peft.pth'),
                'lstm': os.path.join(data_dir, 'best_lstm_improved.pth'),
                'transformer': os.path.join(data_dir, 'best_transformer_with_profit.pth'),
                'gru': os.path.join(data_dir, 'best_gru_with_profit.pth'),
            }
            model_path = default_model_paths.get(strategy_name)
        
        print(f"🏭 创建策略: {strategy_name}")
        print(f"   模型路径: {model_path}")
        
        # 创建策略实例
        try:
            strategy = strategy_class(model_path=model_path, data_dir=data_dir, **kwargs)
            print(f"✅ 策略创建成功: {strategy.strategy_name}")
            return strategy
        except Exception as e:
            print(f"❌ 策略创建失败: {e}")
            raise
    
    @classmethod
    def list_strategies(cls):
        """列出所有已注册的策略"""
        return list(cls._strategies.keys())
    
    @classmethod
    def get_strategy_info(cls, strategy_name: str) -> Dict[str, Any]:
        """获取策略信息"""
        if strategy_name not in cls._strategies:
            return {}
        
        strategy_class = cls._strategies[strategy_name]
        return {
            'name': strategy_name,
            'class': strategy_class.__name__,
            'module': strategy_class.__module__,
        }


# 自动注册所有策略
def register_all_strategies():
    """注册所有可用的策略"""
    try:
        from src.strategies.moe_strategy import MoETradingStrategy
        StrategyFactory.register('moe_transformer', MoETradingStrategy)
    except ImportError as e:
        print(f"⚠️ 无法导入MoE策略: {e}")
    
    try:
        from src.strategies.llm_strategy import LLMTradingStrategy
        # LSTM策略需要mode参数，创建一个包装类
        class LSTMStrategyWrapper(LLMTradingStrategy):
            def __init__(self, model_path=None, data_dir='/home/cx/trading_data', **kwargs):
                mode = kwargs.get('mode', 'hybrid')
                predict_profit = kwargs.get('predict_profit', True)
                super().__init__(data_dir=data_dir, model_path=model_path, mode=mode, predict_profit=predict_profit)
        
        StrategyFactory.register('lstm', LSTMStrategyWrapper)
    except ImportError as e:
        print(f"⚠️ 无法导入LSTM策略: {e}")
    
    # 可以继续注册其他策略...
    print(f"📋 已注册策略: {StrategyFactory.list_strategies()}")


# 初始化时自动注册
register_all_strategies()
