"""
数据清洗器
处理异常值、缺失值、重复数据
"""

import pandas as pd
import numpy as np


class DataCleaner:
    """
    数据清洗器
    
    功能：
    - 删除重复行
    - 填充缺失值
    - 移除异常值（价格跳变>10%）
    - 时间戳排序
    """
    
    def __init__(self, outlier_threshold=0.10):
        """
        Args:
            outlier_threshold: 价格跳变阈值（默认10%）
        """
        self.outlier_threshold = outlier_threshold
        self.stats = {}
    
    def clean(self, df):
        """
        清洗数据
        
        Args:
            df: 原始DataFrame
        
        Returns:
            pd.DataFrame: 清洗后的数据
        """
        print(f"  原始数据: {len(df)}条")
        
        # 统计
        self.stats['original_count'] = len(df)
        self.stats['duplicates'] = 0
        self.stats['outliers'] = 0
        self.stats['missing'] = 0
        
        # 1. 删除重复
        before_dup = len(df)
        df = df.drop_duplicates(subset=['datetime'] if 'datetime' in df.columns else None)
        self.stats['duplicates'] = before_dup - len(df)
        if self.stats['duplicates'] > 0:
            print(f"  删除重复: {self.stats['duplicates']}条")
        
        # 2. 排序
        if 'datetime' in df.columns:
            df = df.sort_values('datetime').reset_index(drop=True)
        
        # 3. 填充缺失值
        missing_before = df.isnull().sum().sum()
        if missing_before > 0:
            df = df.fillna(method='ffill').fillna(method='bfill')
            self.stats['missing'] = missing_before
            print(f"  填充缺失值: {missing_before}个")
        
        # 4. 移除异常值
        before_outlier = len(df)
        df = self._remove_outliers(df)
        self.stats['outliers'] = before_outlier - len(df)
        if self.stats['outliers'] > 0:
            print(f"  移除异常值: {self.stats['outliers']}条")
        
        self.stats['final_count'] = len(df)
        print(f"  清洗后: {len(df)}条")
        
        return df
    
    def _remove_outliers(self, df, threshold=None):
        """移除价格异常跳变"""
        if threshold is None:
            threshold = self.outlier_threshold
        
        if 'close' not in df.columns:
            return df
        
        # 计算价格变化率
        price_change = df['close'].pct_change().abs()
        
        # 保留正常数据
        mask = (price_change < threshold) | price_change.isna()
        df = df[mask].reset_index(drop=True)
        
        return df
    
    def get_stats(self):
        """获取清洗统计"""
        return self.stats
