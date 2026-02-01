"""
网格交易策略
从tiger1.py的grid_trading_strategy提取
"""

from .base import TradingStrategy
import pandas as pd
import numpy as np


class GridStrategy(TradingStrategy):
    """
    网格交易策略
    
    基于价格网格的自动化交易
    """
    
    def __init__(self, grid_size=0.5, grid_count=10):
        super().__init__(name='GridStrategy')
        self.grid_size = grid_size  # 网格间距（%）
        self.grid_count = grid_count  # 网格数量
        self.grids = []  # 网格价位
    
    def generate_signal(self, data, **kwargs):
        """生成交易信号"""
        df_1m = data.get('1m')
        df_5m = data.get('5m')
        
        if df_1m is None or df_1m.empty:
            return self._hold_signal('数据为空')
        
        current_price = df_1m['close'].iloc[-1]
        
        # 初始化网格
        if not self.grids:
            self._init_grids(current_price)
        
        # 判断信号
        if self.position == 0:
            # 寻找买入机会
            for grid_price in self.grids:
                if current_price <= grid_price * 0.995:  # 价格触及网格下方
                    return {
                        'action': 'BUY',
                        'confidence': 0.7,
                        'position_size': 0.2,
                        'stop_loss': grid_price * 0.98,
                        'take_profit': grid_price * 1.02,
                        'reason': f'价格触及网格{grid_price:.2f}'
                    }
        
        else:
            # 持仓中，判断止盈止损
            if self.position > 0:
                # 做多持仓
                entry_price = kwargs.get('entry_price', current_price)
                profit_pct = (current_price - entry_price) / entry_price
                
                if profit_pct >= 0.02:  # 盈利2%
                    return {
                        'action': 'SELL',
                        'confidence': 0.8,
                        'position_size': 1.0,
                        'stop_loss': None,
                        'take_profit': None,
                        'reason': f'止盈 ({profit_pct*100:.1f}%)'
                    }
                elif profit_pct <= -0.01:  # 亏损1%
                    return {
                        'action': 'SELL',
                        'confidence': 0.9,
                        'position_size': 1.0,
                        'stop_loss': None,
                        'take_profit': None,
                        'reason': f'止损 ({profit_pct*100:.1f}%)'
                    }
        
        return self._hold_signal('等待信号')
    
    def _init_grids(self, base_price):
        """初始化网格价位"""
        self.grids = []
        for i in range(-self.grid_count//2, self.grid_count//2 + 1):
            grid_price = base_price * (1 + i * self.grid_size / 100)
            self.grids.append(grid_price)
        self.grids.sort()
    
    def _hold_signal(self, reason):
        """持有信号"""
        return {
            'action': 'HOLD',
            'confidence': 0.0,
            'position_size': 0.0,
            'stop_loss': None,
            'take_profit': None,
            'reason': reason
        }
