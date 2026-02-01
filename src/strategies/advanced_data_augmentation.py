"""
高级数据增强 - 时间序列特定的增强方法
"""
import numpy as np
import torch
from typing import Tuple


class AdvancedDataAugmentation:
    """高级数据增强"""
    
    @staticmethod
    def time_warp(x: np.ndarray, sigma: float = 0.2) -> np.ndarray:
        """
        时间扭曲：随机拉伸或压缩时间序列
        Args:
            x: (seq_len, features)
            sigma: 扭曲强度
        """
        seq_len = x.shape[0]
        # 生成扭曲路径
        warp_path = np.cumsum(np.random.normal(1.0, sigma, seq_len))
        warp_path = (warp_path - warp_path[0]) / (warp_path[-1] - warp_path[0]) * (seq_len - 1)
        
        # 插值
        from scipy.interpolate import interp1d
        warped_x = np.zeros_like(x)
        for i in range(x.shape[1]):
            f = interp1d(np.arange(seq_len), x[:, i], kind='linear', fill_value='extrapolate')
            warped_x[:, i] = f(warp_path)
        
        return warped_x
    
    @staticmethod
    def window_slice(x: np.ndarray, reduce_ratio: float = 0.9) -> np.ndarray:
        """
        窗口切片：随机选择子序列
        Args:
            x: (seq_len, features)
            reduce_ratio: 保留的比例
        """
        seq_len = x.shape[0]
        new_len = int(seq_len * reduce_ratio)
        start = np.random.randint(0, seq_len - new_len + 1)
        return x[start:start+new_len]
    
    @staticmethod
    def add_gaussian_noise(x: np.ndarray, noise_std: float = 0.01) -> np.ndarray:
        """添加高斯噪声"""
        noise = np.random.normal(0, noise_std, x.shape)
        return x + noise
    
    @staticmethod
    def price_scaling(x: np.ndarray, scale_range: Tuple[float, float] = (0.95, 1.05)) -> np.ndarray:
        """
        价格缩放：对价格相关特征进行缩放
        Args:
            x: (seq_len, features)
            scale_range: 缩放范围
        """
        scaled_x = x.copy()
        # 假设前几个特征是价格相关
        price_features = [0, 1, 2, 3]  # 根据实际特征调整
        scale = np.random.uniform(scale_range[0], scale_range[1])
        
        for idx in price_features:
            if idx < x.shape[1]:
                scaled_x[:, idx] *= scale
        
        return scaled_x
    
    @staticmethod
    def mixup(x1: np.ndarray, y1: np.ndarray, y_profit1: np.ndarray,
              x2: np.ndarray, y2: np.ndarray, y_profit2: np.ndarray,
              alpha: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Mixup数据增强"""
        lam = np.random.beta(alpha, alpha)
        mixed_x = lam * x1 + (1 - lam) * x2
        mixed_y = y1  # 保持原始标签
        mixed_y_profit = lam * y_profit1 + (1 - lam) * y_profit2
        return mixed_x, mixed_y, mixed_y_profit
    
    @staticmethod
    def apply_augmentation(x: np.ndarray, augmentation_type: str = 'random') -> np.ndarray:
        """
        应用数据增强
        Args:
            x: (seq_len, features)
            augmentation_type: 增强类型
                - 'random': 随机选择一种增强
                - 'time_warp': 时间扭曲
                - 'window_slice': 窗口切片
                - 'noise': 添加噪声
                - 'scaling': 价格缩放
        """
        if augmentation_type == 'random':
            aug_types = ['time_warp', 'window_slice', 'noise', 'scaling']
            augmentation_type = np.random.choice(aug_types)
        
        if augmentation_type == 'time_warp' and np.random.random() < 0.3:
            return AdvancedDataAugmentation.time_warp(x)
        elif augmentation_type == 'window_slice' and np.random.random() < 0.3:
            return AdvancedDataAugmentation.window_slice(x)
        elif augmentation_type == 'noise' and np.random.random() < 0.5:
            return AdvancedDataAugmentation.add_gaussian_noise(x)
        elif augmentation_type == 'scaling' and np.random.random() < 0.3:
            return AdvancedDataAugmentation.price_scaling(x)
        else:
            return x
