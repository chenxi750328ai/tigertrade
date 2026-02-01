"""
合并多个训练数据文件
将新获取的数据与历史数据合并，增加训练数据量
"""
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import glob

sys.path.insert(0, '/home/cx/tigertrade')


def merge_training_data_files(data_dir='/home/cx/trading_data', output_file=None):
    """
    合并多个训练数据文件
    
    Args:
        data_dir: 数据目录
        output_file: 输出文件路径（可选）
    """
    print("="*70)
    print("合并训练数据文件")
    print("="*70)
    
    # 查找所有训练数据文件
    data_files = glob.glob(os.path.join(data_dir, 'training_data_multitimeframe_*.csv'))
    
    if not data_files:
        print("❌ 未找到训练数据文件")
        return None
    
    print(f"\n📁 找到 {len(data_files)} 个训练数据文件:")
    for f in sorted(data_files):
        size = os.path.getsize(f)
        print(f"  - {os.path.basename(f)} ({size:,} bytes)")
    
    # 加载所有数据文件
    print(f"\n📊 加载数据文件...")
    all_dataframes = []
    
    for file_path in sorted(data_files):
        try:
            df = pd.read_csv(file_path)
            print(f"  ✅ {os.path.basename(file_path)}: {len(df)} 条记录")
            all_dataframes.append(df)
        except Exception as e:
            print(f"  ❌ {os.path.basename(file_path)}: 加载失败 - {e}")
            continue
    
    if not all_dataframes:
        print("❌ 没有成功加载任何数据文件")
        return None
    
    # 合并所有数据
    print(f"\n🔄 合并数据...")
    merged_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"  合并前总记录数: {len(merged_df)}")
    
    # 去重（基于时间戳，如果有的话）
    if 'timestamp' in merged_df.columns:
        print(f"\n🔍 去重（基于timestamp）...")
        before_dedup = len(merged_df)
        merged_df = merged_df.drop_duplicates(subset=['timestamp'], keep='last')
        after_dedup = len(merged_df)
        print(f"  去重前: {before_dedup} 条")
        print(f"  去重后: {after_dedup} 条")
        print(f"  删除重复: {before_dedup - after_dedup} 条")
        
        # 按时间排序（使用更灵活的时间解析）
        try:
            merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'], errors='coerce', format='mixed')
            # 删除无法解析的时间戳
            invalid_timestamps = merged_df['timestamp'].isna().sum()
            if invalid_timestamps > 0:
                print(f"  ⚠️ 发现 {invalid_timestamps} 条无效时间戳，将被删除")
                merged_df = merged_df.dropna(subset=['timestamp'])
            merged_df = merged_df.sort_values('timestamp').reset_index(drop=True)
            print(f"  ✅ 已按时间排序")
        except Exception as e:
            print(f"  ⚠️ 时间排序失败: {e}，跳过排序")
    else:
        # 如果没有timestamp列，尝试使用第一列作为时间
        print(f"\n⚠️ 未找到timestamp列，尝试使用第一列...")
        if len(merged_df.columns) > 0:
            first_col = merged_df.columns[0]
            if 'time' in first_col.lower() or 'date' in first_col.lower():
                merged_df = merged_df.drop_duplicates(subset=[first_col], keep='last')
                merged_df = merged_df.sort_values(first_col).reset_index(drop=True)
                print(f"  ✅ 已基于 {first_col} 去重和排序")
    
    # 检查数据质量
    print(f"\n📊 数据质量检查:")
    print(f"  总记录数: {len(merged_df)}")
    print(f"  特征数量: {len(merged_df.columns)}")
    print(f"  缺失值统计:")
    missing_counts = merged_df.isnull().sum()
    if missing_counts.sum() > 0:
        for col, count in missing_counts[missing_counts > 0].items():
            print(f"    {col}: {count} ({count/len(merged_df)*100:.2f}%)")
    else:
        print(f"    ✅ 无缺失值")
    
    # 保存合并后的数据
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(data_dir, f'training_data_multitimeframe_merged_{timestamp}.csv')
    
    print(f"\n💾 保存合并后的数据...")
    merged_df.to_csv(output_file, index=False)
    print(f"  ✅ 已保存到: {output_file}")
    print(f"  文件大小: {os.path.getsize(output_file):,} bytes")
    
    # 显示时间范围（如果有timestamp列）
    if 'timestamp' in merged_df.columns:
        try:
            # 统一时区处理
            if merged_df['timestamp'].dtype == 'object':
                # 如果还是object类型，尝试转换
                merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'], errors='coerce', utc=True)
            
            # 转换为naive datetime（移除时区信息）以便比较
            if merged_df['timestamp'].dt.tz is not None:
                merged_df['timestamp'] = merged_df['timestamp'].dt.tz_localize(None)
            
            print(f"\n📅 数据时间范围:")
            print(f"  开始时间: {merged_df['timestamp'].min()}")
            print(f"  结束时间: {merged_df['timestamp'].max()}")
            time_span = (merged_df['timestamp'].max() - merged_df['timestamp'].min())
            print(f"  时间跨度: {time_span}")
        except Exception as e:
            print(f"\n⚠️ 无法显示时间范围: {e}")
    
    return merged_df


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='合并训练数据文件')
    parser.add_argument('--data-dir', type=str, default='/home/cx/trading_data', 
                       help='数据目录（默认: /home/cx/trading_data）')
    parser.add_argument('--output', type=str, default=None, 
                       help='输出文件路径（可选）')
    
    args = parser.parse_args()
    
    # 合并数据
    merged_df = merge_training_data_files(data_dir=args.data_dir, output_file=args.output)
    
    if merged_df is not None:
        print(f"\n✅ 合并完成！")
        print(f"  总记录数: {len(merged_df)}")
        print(f"  输出文件: {args.output if args.output else '自动生成'}")
    else:
        print("\n❌ 合并失败")


if __name__ == "__main__":
    main()
