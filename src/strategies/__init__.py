"""
Module 5: 策略引擎
各种交易策略实现
"""

from .base import TradingStrategy
from .grid_strategy import GridStrategy
from .transformer_strategy import TransformerStrategy

__all__ = ['TradingStrategy', 'GridStrategy', 'TransformerStrategy']

def get_strategy(name='grid'):
    """
    获取策略实例
    
    Args:
        name: 策略名称 ('grid', 'transformer', 'boll')
    
    Returns:
        TradingStrategy实例
    """
    strategies = {
        'grid': GridStrategy,
        'transformer': TransformerStrategy,
    }
    
    strategy_class = strategies.get(name, GridStrategy)
    return strategy_class()
