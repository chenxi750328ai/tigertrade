"""
基础策略接口 - 所有交易策略的统一接口
确保模型切换不需要修改主程序代码
"""
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
import pandas as pd


class BaseTradingStrategy(ABC):
    """交易策略基类 - 所有策略必须实现的接口"""
    
    @abstractmethod
    def __init__(self, model_path: Optional[str] = None, **kwargs):
        """
        初始化策略
        
        Args:
            model_path: 模型文件路径
            **kwargs: 其他策略特定参数
        """
        pass
    
    @abstractmethod
    def predict_action(self, current_data: Dict[str, Any], historical_data: Optional[pd.DataFrame] = None) -> Tuple[int, float, Optional[float]]:
        """
        预测交易动作
        
        Args:
            current_data: 当前市场数据字典，包含：
                - price_current: 当前价格
                - grid_lower: 网格下轨
                - grid_upper: 网格上轨
                - atr: ATR值
                - rsi_1m: 1分钟RSI
                - rsi_5m: 5分钟RSI
                - buffer: 缓冲区
                - threshold: 阈值
                - near_lower: 是否接近下轨
                - rsi_ok: RSI是否OK
                - tick_price: Tick价格
                - kline_price: K线价格
            historical_data: 历史数据DataFrame（可选，用于序列模型）
        
        Returns:
            (action, confidence, profit_prediction)
            - action: 0=不操作, 1=买入, 2=卖出
            - confidence: 置信度 [0, 1]
            - profit_prediction: 预测收益率（可选，如果模型支持）
        """
        pass
    
    @abstractmethod
    def prepare_features(self, row: Dict[str, Any]) -> list:
        """
        准备特征向量
        
        Args:
            row: 数据行字典
        
        Returns:
            特征向量列表
        """
        pass
    
    @property
    @abstractmethod
    def seq_length(self) -> int:
        """返回策略需要的序列长度"""
        pass
    
    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """返回策略名称"""
        pass
