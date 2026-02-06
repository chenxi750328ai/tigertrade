"""
数据标准化器
Z-score标准化或Min-Max归一化
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path


class DataNormalizer:
    """
    数据标准化器
    
    支持：
    - Z-score标准化: (x - mean) / std
    - Min-Max归一化: (x - min) / (max - min)
    """
    
    def __init__(self, method='zscore'):
        """
        Args:
            method: 'zscore' 或 'minmax'
        """
        self.method = method
        self.scalers = {}  # 保存每列的scale参数

    def fit_transform_rolling(self, df, feature_cols=None, window=60):
        """
        滚动窗口归一化（避免未来信息泄露，适配金融时序）。
        每列用过去 window 步的滚动均值和标准差做 Z-score。
        """
        if feature_cols is None:
            feature_cols = [c for c in df.columns if df[c].dtype in ('float64', 'int64')]
        df_norm = df.copy()
        for col in feature_cols:
            if col not in df.columns:
                continue
            r = df[col].rolling(window=window, min_periods=1)
            mean_ = r.mean()
            std_ = r.std()
            df_norm[col] = (df[col] - mean_) / (std_ + 1e-8)
        return df_norm

    def fit_transform(self, df, feature_cols=None):
        """
        拟合并转换
        
        Args:
            df: 数据
            feature_cols: 要标准化的列（默认OHLCV）
        
        Returns:
            pd.DataFrame: 标准化后的数据
        """
        if feature_cols is None:
            feature_cols = ['open', 'high', 'low', 'close', 'volume']
        
        df_norm = df.copy()
        
        for col in feature_cols:
            if col not in df.columns:
                continue
            
            if self.method == 'zscore':
                mean = df[col].mean()
                std = df[col].std()
                df_norm[col] = (df[col] - mean) / (std + 1e-8)  # 避免除零
                self.scalers[col] = {'mean': float(mean), 'std': float(std)}
            
            elif self.method == 'minmax':
                min_val = df[col].min()
                max_val = df[col].max()
                df_norm[col] = (df[col] - min_val) / (max_val - min_val + 1e-8)
                self.scalers[col] = {'min': float(min_val), 'max': float(max_val)}
        
        print(f"  标准化方法: {self.method}")
        print(f"  标准化列: {feature_cols}")
        
        return df_norm
    
    def transform(self, df, feature_cols=None):
        """
        使用已拟合的参数转换新数据
        
        Args:
            df: 新数据
            feature_cols: 要标准化的列
        
        Returns:
            pd.DataFrame: 标准化后的数据
        """
        if not self.scalers:
            raise ValueError("请先调用fit_transform()拟合参数")
        
        if feature_cols is None:
            feature_cols = list(self.scalers.keys())
        
        df_norm = df.copy()
        
        for col in feature_cols:
            if col not in self.scalers:
                continue
            
            params = self.scalers[col]
            
            if self.method == 'zscore':
                df_norm[col] = (df[col] - params['mean']) / (params['std'] + 1e-8)
            elif self.method == 'minmax':
                df_norm[col] = (df[col] - params['min']) / (params['max'] - params['min'] + 1e-8)
        
        return df_norm
    
    def save_scalers(self, filepath):
        """保存scale参数"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump({
                'method': self.method,
                'scalers': self.scalers
            }, f, indent=2)
        
        print(f"  ✅ Scale参数已保存: {filepath}")
    
    def load_scalers(self, filepath):
        """加载scale参数"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.method = data['method']
        self.scalers = data['scalers']
        
        print(f"  ✅ Scale参数已加载: {filepath}")
