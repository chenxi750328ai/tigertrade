"""
市场数据提供者
统一的数据获取和指标计算模块
"""
import pandas as pd
from typing import Dict, Any, Optional
import sys
sys.path.insert(0, '/home/cx/tigertrade')

from src import tiger1 as t1


class MarketDataProvider:
    """市场数据提供者 - 统一的数据获取和指标计算"""
    
    def __init__(self, symbol: str = None):
        """
        初始化数据提供者
        
        Args:
            symbol: 交易标的符号（默认使用tiger1中的FUTURE_SYMBOL）
        """
        self.symbol = symbol or t1.FUTURE_SYMBOL
        self.historical_data_cache = []
    
    def get_market_data(self, seq_length: int = 10) -> Dict[str, Any]:
        """
        获取完整的市场数据
        
        Args:
            seq_length: 序列长度（用于历史数据缓存）
        
        Returns:
            包含current_data、indicators、historical_data的字典
        """
        # 获取K线数据
        df_5m = t1.get_kline_data([self.symbol], '5min', count=t1.GRID_PERIOD + 5)
        df_1m = t1.get_kline_data([self.symbol], '1min', count=max(t1.GRID_PERIOD + 5, seq_length + 10))
        
        # 获取Tick数据
        df_tick = t1.get_tick_data([self.symbol], count=100)
        
        if df_5m.empty or df_1m.empty:
            raise ValueError("K线数据不足")
        
        # 计算技术指标
        indicators = t1.calculate_indicators(df_5m, df_1m)
        if indicators is None or '5m' not in indicators or '1m' not in indicators:
            raise ValueError("指标计算失败")
        
        # 获取关键指标
        price_current = indicators['1m']['close']
        atr = indicators['5m']['atr']
        rsi_1m = indicators['1m']['rsi']
        rsi_5m = indicators['5m']['rsi']
        
        # 获取Tick价格
        tick_price = price_current
        if not df_tick.empty:
            latest_tick = df_tick.iloc[-1]
            tick_price = latest_tick['price'] if 'price' in latest_tick else price_current
        
        # 使用时段自适应网格参数
        trend = t1.judge_market_trend(indicators)
        t1.adjust_grid_interval(trend, indicators)
        
        grid_upper_val = t1.grid_upper
        grid_lower_val = t1.grid_lower
        
        # 计算缓冲区
        buffer = max(atr * 0.3, 0.0025)
        threshold = grid_lower_val + buffer
        
        # 准备当前数据
        current_data = {
            'price_current': tick_price,
            'grid_lower': grid_lower_val,
            'grid_upper': grid_upper_val,
            'atr': atr,
            'rsi_1m': rsi_1m,
            'rsi_5m': rsi_5m,
            'buffer': buffer,
            'threshold': threshold,
            'near_lower': tick_price <= threshold,
            'rsi_ok': rsi_1m < 30 or (rsi_5m > 45 and rsi_5m < 55),
            'tick_price': tick_price,
            'kline_price': price_current
        }
        
        # 更新历史数据缓存
        self.historical_data_cache.append(current_data)
        max_cache_size = seq_length + 20
        if len(self.historical_data_cache) > max_cache_size:
            self.historical_data_cache = self.historical_data_cache[-max_cache_size:]
        
        # 准备历史数据DataFrame
        historical_data = None
        if len(self.historical_data_cache) >= seq_length:
            historical_data = pd.DataFrame(self.historical_data_cache)
        
        return {
            'current_data': current_data,
            'indicators': indicators,
            'historical_data': historical_data,
            'tick_price': tick_price,
            'price_current': price_current,
            'atr': atr,
            'grid_lower': grid_lower_val,
            'grid_upper': grid_upper_val
        }
    
    def get_kline_data(self, period: str, count: int = 100) -> pd.DataFrame:
        """获取K线数据"""
        return t1.get_kline_data([self.symbol], period, count=count)
    
    def get_tick_data(self, count: int = 100) -> pd.DataFrame:
        """获取Tick数据"""
        return t1.get_tick_data([self.symbol], count=count)
    
    def calculate_indicators(self, df_5m: pd.DataFrame, df_1m: pd.DataFrame) -> Dict[str, Any]:
        """计算技术指标"""
        return t1.calculate_indicators(df_5m, df_1m)
    
    def clear_cache(self):
        """清空历史数据缓存"""
        self.historical_data_cache = []
