"""
策略基类
所有策略必须继承此类
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd


class TradingStrategy(ABC):
    """
    交易策略基类
    
    所有策略必须实现 generate_signal 方法
    """
    
    def __init__(self, name='BaseStrategy'):
        self.name = name
        self.position = 0  # 当前持仓
        self.last_signal = None
    
    @abstractmethod
    def generate_signal(self, data: Dict[str, pd.DataFrame], **kwargs) -> Dict[str, Any]:
        """
        生成交易信号
        
        Args:
            data: 多周期数据 {'1m': df, '5m': df, ...}
            **kwargs: 其他参数
        
        Returns:
            {
                'action': 'BUY' | 'SELL' | 'HOLD',
                'confidence': 0.0-1.0,
                'position_size': 0.0-1.0,
                'stop_loss': float | None,
                'take_profit': float | None,
                'reason': str
            }
        """
        pass
    
    def update_position(self, position: float):
        """更新持仓"""
        self.position = position
    
    def reset(self):
        """重置策略状态"""
        self.position = 0
        self.last_signal = None
