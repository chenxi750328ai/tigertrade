"""
数据集划分器
按时间序列划分训练/验证/测试集
"""

import pandas as pd


class DataSplitter:
    """
    数据集划分器
    
    时间序列划分：
    - 训练集：最早的数据
    - 验证集：中间的数据
    - 测试集：最新的数据
    """
    
    def __init__(self, train_ratio=0.7, val_ratio=0.15):
        """
        Args:
            train_ratio: 训练集比例
            val_ratio: 验证集比例（测试集 = 1 - train - val）
        """
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1 - train_ratio - val_ratio
    
    def split(self, df):
        """
        划分数据集
        
        Args:
            df: 完整数据
        
        Returns:
            (df_train, df_val, df_test)
        """
        n = len(df)
        
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))
        
        df_train = df[:train_end].copy()
        df_val = df[train_end:val_end].copy()
        df_test = df[val_end:].copy()
        
        print(f"  训练集: {len(df_train)}条 ({len(df_train)/n*100:.1f}%)")
        print(f"  验证集: {len(df_val)}条 ({len(df_val)/n*100:.1f}%)")
        print(f"  测试集: {len(df_test)}条 ({len(df_test)/n*100:.1f}%)")
        
        if 'datetime' in df.columns:
            print(f"  训练集时间: {df_train['datetime'].iloc[0]} ~ {df_train['datetime'].iloc[-1]}")
            print(f"  验证集时间: {df_val['datetime'].iloc[0]} ~ {df_val['datetime'].iloc[-1]}")
            print(f"  测试集时间: {df_test['datetime'].iloc[0]} ~ {df_test['datetime'].iloc[-1]}")
        
        return df_train, df_val, df_test
