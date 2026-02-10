#!/usr/bin/env python3
"""
数据准备主脚本 - Agent 1核心任务
整合所有数据处理步骤，生成train/val/test.csv
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from pathlib import Path
from datetime import datetime

from src.data_processor.cleaner import DataCleaner
from src.data_processor.normalizer import DataNormalizer
from src.data_processor.splitter import DataSplitter


def main():
    print("="*80)
    print("📊 Agent 1: 数据准备Pipeline")
    print("="*80)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 配置
    data_dir = Path('/home/cx/trading_data')
    output_dir = data_dir / 'processed'
    output_dir.mkdir(exist_ok=True)
    
    # 1. 加载所有数据
    print(f"{'='*80}")
    print(f"步骤1: 加载所有原始数据")
    print(f"{'='*80}\n")
    
    data_files = [
        data_dir / 'ticks/SIL2603_ticks_20260121.csv',
        data_dir / 'SIL2603_1min_combined.csv',
        data_dir / 'SIL2603_5min_7days.csv',
        data_dir / 'SIL2603_1h_30days.csv',
        data_dir / 'SIL2603_daily_90days.csv'
    ]
    
    all_data = []
    total_size = 0
    
    for file in data_files:
        if file.exists():
            df = pd.read_csv(file)
            size_mb = file.stat().st_size / 1024 / 1024
            total_size += size_mb
            
            # 标准化时间列
            for time_col in ['time', 'datetime']:
                if time_col in df.columns:
                    if time_col == 'time':
                        df['datetime'] = pd.to_datetime(df['time'], unit='ms')
                    else:
                        df['datetime'] = pd.to_datetime(df['datetime'])
                    break
            
            # 标准化列名（Tick数据用price作为close）
            if 'price' in df.columns and 'close' not in df.columns:
                df['close'] = df['price']
            if 'open' not in df.columns:
                df['open'] = df['close']
            if 'high' not in df.columns:
                df['high'] = df['close']
            if 'low' not in df.columns:
                df['low'] = df['close']
            if 'volume' not in df.columns:
                df['volume'] = 0
            
            # 保留核心列
            core_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
            df = df[[col for col in core_cols if col in df.columns]]
            
            all_data.append(df)
            print(f"  ✅ {file.name:<40} {len(df):>8}条  {size_mb:>6.2f}MB")
    
    print(f"\n  总文件大小: {total_size:.2f}MB")
    
    # 合并所有数据
    print(f"\n  合并所有数据...")
    df_all = pd.concat(all_data, ignore_index=True)
    df_all = df_all.sort_values('datetime').reset_index(drop=True)
    
    print(f"  合并后总计: {len(df_all):,}条")
    print(f"  时间范围: {df_all['datetime'].iloc[0]} ~ {df_all['datetime'].iloc[-1]}")
    print(f"  时间跨度: {(df_all['datetime'].iloc[-1] - df_all['datetime'].iloc[0]).days}天")
    
    # 2. 数据清洗
    print(f"\n{'='*80}")
    print(f"步骤2: 数据清洗")
    print(f"{'='*80}\n")
    
    cleaner = DataCleaner(outlier_threshold=0.10)
    df_clean = cleaner.clean(df_all)
    
    # 显示清洗统计
    stats = cleaner.get_stats()
    print(f"\n  清洗统计:")
    print(f"    原始: {stats['original_count']:,}条")
    print(f"    重复: {stats['duplicates']:,}条")
    print(f"    异常: {stats['outliers']:,}条")
    print(f"    缺失: {stats['missing']:,}个值")
    print(f"    最终: {stats['final_count']:,}条")
    
    # 3. 计算基础特征
    print(f"\n{'='*80}")
    print(f"步骤3: 计算基础特征")
    print(f"{'='*80}\n")
    
    # 价格变化
    df_clean['price_change'] = df_clean['close'].diff()
    df_clean['price_change_pct'] = df_clean['close'].pct_change()
    
    # 时间间隔
    df_clean['time_delta'] = df_clean['datetime'].diff().dt.total_seconds()
    
    # 价格范围
    df_clean['price_range'] = df_clean['high'] - df_clean['low']
    df_clean['price_range_pct'] = df_clean['price_range'] / df_clean['close']
    
    # 成交量变化
    df_clean['volume_change'] = df_clean['volume'].diff()
    df_clean['volume_change_pct'] = df_clean['volume'].pct_change()
    
    # 填充第一行的NaN
    df_clean = df_clean.fillna(0)
    
    print(f"  ✅ 特征数量: {len(df_clean.columns)}列")
    print(f"  特征列:")
    for col in df_clean.columns:
        if col not in ['datetime']:
            print(f"    - {col}")
    
    # 4. 数据标准化
    print(f"\n{'='*80}")
    print(f"步骤4: 数据标准化")
    print(f"{'='*80}\n")
    
    normalizer = DataNormalizer(method='zscore')
    
    feature_cols = ['open', 'high', 'low', 'close', 'volume',
                    'price_change', 'price_change_pct', 'time_delta',
                    'price_range', 'price_range_pct',
                    'volume_change', 'volume_change_pct']
    
    df_norm = normalizer.fit_transform(df_clean, feature_cols)
    
    # 保存标准化参数
    normalizer.save_scalers(output_dir / 'scaler_params.json')
    
    # 5. 划分数据集
    print(f"\n{'='*80}")
    print(f"步骤5: 划分数据集")
    print(f"{'='*80}\n")
    
    splitter = DataSplitter(train_ratio=0.7, val_ratio=0.15)
    df_train, df_val, df_test = splitter.split(df_norm)
    
    # 6. 保存数据
    print(f"\n{'='*80}")
    print(f"步骤6: 保存数据")
    print(f"{'='*80}\n")
    
    df_train.to_csv(output_dir / 'train.csv', index=False)
    df_val.to_csv(output_dir / 'val.csv', index=False)
    df_test.to_csv(output_dir / 'test.csv', index=False)
    
    print(f"  ✅ {output_dir / 'train.csv'}")
    print(f"  ✅ {output_dir / 'val.csv'}")
    print(f"  ✅ {output_dir / 'test.csv'}")
    
    # 7. 生成数据报告
    print(f"\n{'='*80}")
    print(f"📋 数据质量报告")
    print(f"{'='*80}\n")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_records': len(df_norm),
        'train_records': len(df_train),
        'val_records': len(df_val),
        'test_records': len(df_test),
        'time_range': {
            'start': str(df_norm['datetime'].iloc[0]),
            'end': str(df_norm['datetime'].iloc[-1]),
            'days': (df_norm['datetime'].iloc[-1] - df_norm['datetime'].iloc[0]).days
        },
        'price_range': {
            'min': float(df_clean['close'].min()),
            'max': float(df_clean['close'].max()),
            'change_pct': float((df_clean['close'].iloc[-1] / df_clean['close'].iloc[0] - 1) * 100)
        },
        'cleaning_stats': stats,
        'feature_count': len(feature_cols),
        'normalization_method': 'zscore'
    }
    
    import json
    report_file = output_dir / 'data_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"总记录数: {report['total_records']:,}条")
    print(f"时间跨度: {report['time_range']['days']}天")
    print(f"价格变化: {report['price_range']['change_pct']:+.2f}%")
    print(f"特征数量: {report['feature_count']}个")
    print(f"\n报告已保存: {report_file}")
    
    # 完成
    print(f"\n{'='*80}")
    print(f"✅ Agent 1 任务完成！")
    print(f"{'='*80}")
    print(f"\n📁 输出文件:")
    print(f"   - train.csv: {len(df_train):,}条")
    print(f"   - val.csv: {len(df_val):,}条")
    print(f"   - test.csv: {len(df_test):,}条")
    print(f"   - scaler_params.json: 标准化参数")
    print(f"   - data_report.json: 数据质量报告")
    print(f"\n🎯 下一步: Agent 2 使用这些数据训练模型")
    print(f"{'='*80}\n")

    return report


def prepare_training_data():
    """供每日例行脚本调用，与 main() 一致，返回数据报告摘要。"""
    return main()


if __name__ == '__main__':
    main()
