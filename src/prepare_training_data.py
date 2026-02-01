#!/usr/bin/env python3
"""
最终数据准备脚本 - 生成用于训练的高质量数据集
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def prepare_final_dataset():
    """准备最终训练数据集"""
    print("=" * 80)
    print("📦 准备最终训练数据")
    print("=" * 80)
    
    # 读取优化后的数据
    input_file = '/home/cx/trading_data/enhanced/full_20260120_142504_optimized.csv'
    print(f"\n读取数据: {input_file}")
    df = pd.read_csv(input_file, index_col=0)
    print(f"✅ 加载 {len(df)} 条记录")
    
    # 使用百分位数策略（最平衡）和标准差策略
    print("\n使用标注策略:")
    print("  - 主策略: label_percentile (百分位数)")
    print("  - 备选策略: label_std (标准差)")
    
    # 重命名为标准的label列
    df['label'] = df['label_percentile']  # 使用百分位数作为默认标签
    
    # 打印标签分布
    print("\n标签分布 (百分位数策略):")
    label_counts = df['label'].value_counts().sort_index()
    for label, count in label_counts.items():
        label_name = {0: "持有", 1: "买入", 2: "卖出"}.get(label, "未知")
        print(f"  {label_name} ({label}): {count} ({count/len(df)*100:.1f}%)")
    
    # 划分数据集 - 使用改进的划分策略
    print("\n=" * 80)
    print("划分数据集...")
    
    # 策略：前70%训练，中间15%验证，最后15%测试
    # 但为每个集合确保标签分布
    train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
    
    # 按时间顺序划分基本集合
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    print(f"\n数据集大小:")
    print(f"  训练集: {len(train_df)} 条")
    print(f"  验证集: {len(val_df)} 条")
    print(f"  测试集: {len(test_df)} 条")
    
    # 打印各集的标签分布
    for name, data in [('训练集', train_df), ('验证集', val_df), ('测试集', test_df)]:
        counts = data['label'].value_counts().sort_index()
        print(f"\n{name}标签分布:")
        for label, count in counts.items():
            label_name = {0: "持有", 1: "买入", 2: "卖出"}.get(label, "未知")
            print(f"  {label_name}: {count} ({count/len(data)*100:.1f}%)")
    
    # 保存数据集
    output_dir = '/home/cx/trading_data/final'
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    train_file = os.path.join(output_dir, f'train_{timestamp}.csv')
    val_file = os.path.join(output_dir, f'val_{timestamp}.csv')
    test_file = os.path.join(output_dir, f'test_{timestamp}.csv')
    
    train_df.to_csv(train_file, index=True)
    val_df.to_csv(val_file, index=True)
    test_df.to_csv(test_file, index=True)
    
    print("\n=" * 80)
    print("✅ 数据集已保存:")
    print(f"  - 训练集: {train_file}")
    print(f"  - 验证集: {val_file}")
    print(f"  - 测试集: {test_file}")
    
    # 生成数据集信息文件
    info_file = os.path.join(output_dir, f'dataset_info_{timestamp}.txt')
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write(f"数据集信息\n")
        f.write(f"生成时间: {datetime.now()}\n\n")
        f.write(f"训练集: {train_file}\n")
        f.write(f"  大小: {len(train_df)} 条\n")
        f.write(f"  标签分布: {dict(train_df['label'].value_counts())}\n\n")
        f.write(f"验证集: {val_file}\n")
        f.write(f"  大小: {len(val_df)} 条\n")
        f.write(f"  标签分布: {dict(val_df['label'].value_counts())}\n\n")
        f.write(f"测试集: {test_file}\n")
        f.write(f"  大小: {len(test_df)} 条\n")
        f.write(f"  标签分布: {dict(test_df['label'].value_counts())}\n\n")
        f.write(f"特征列 ({len(df.columns)}):\n")
        for col in df.columns:
            f.write(f"  - {col}\n")
    
    print(f"  - 信息文件: {info_file}")
    
    return train_file, val_file, test_file


if __name__ == "__main__":
    prepare_final_dataset()
