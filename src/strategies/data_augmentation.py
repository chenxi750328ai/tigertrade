"""
数据增强 - 针对交易数据的增强方法
"""
import numpy as np
import pandas as pd
from typing import List, Tuple


class TradingDataAugmentation:
    """交易数据增强"""
    
    @staticmethod
    def mixup(x1: np.ndarray, y1: np.ndarray, y_profit1: np.ndarray,
              x2: np.ndarray, y2: np.ndarray, y_profit2: np.ndarray,
              alpha: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Mixup数据增强
        Args:
            alpha: Beta分布的参数，控制混合强度
        """
        lam = np.random.beta(alpha, alpha)
        mixed_x = lam * x1 + (1 - lam) * x2
        mixed_y = y1  # 保持原始标签（或使用加权标签）
        mixed_y_profit = lam * y_profit1 + (1 - lam) * y_profit2
        return mixed_x, mixed_y, mixed_y_profit
    
    @staticmethod
    def add_noise(x: np.ndarray, noise_std: float = 0.01) -> np.ndarray:
        """添加高斯噪声"""
        noise = np.random.normal(0, noise_std, x.shape)
        return x + noise
    
    @staticmethod
    def time_shift(x: np.ndarray, max_shift: int = 2) -> np.ndarray:
        """时间序列平移"""
        shift = np.random.randint(-max_shift, max_shift + 1)
        if shift == 0:
            return x
        if shift > 0:
            return np.pad(x[shift:], ((0, shift), (0, 0)), mode='edge')
        else:
            return np.pad(x[:shift], ((-shift, 0), (0, 0)), mode='edge')
    
    @staticmethod
    def scale_features(x: np.ndarray, scale_range: Tuple[float, float] = (0.95, 1.05)) -> np.ndarray:
        """特征缩放"""
        scale = np.random.uniform(scale_range[0], scale_range[1])
        return x * scale
