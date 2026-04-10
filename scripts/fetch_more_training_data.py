"""
获取更多训练数据
通过API获取更多的K线和Tick数据，扩大数据时间范围
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src import tiger1 as t1
from scripts.analysis.generate_multitimeframe_training_data_direct import generate_training_data_direct


def fetch_more_data(days_back=60, output_file=None):
    """
    获取更多历史数据
    
    Args:
        days_back: 回溯天数
        output_file: 输出文件路径
    """
    print("="*70)
    print("获取更多训练数据")
    print("="*70)
    
    print(f"\n📅 将获取过去 {days_back} 天的数据")
    
    # 计算需要的数据量
    # 1分钟: 每天约390条（交易时段）
    # 5分钟: 每天约78条
    # 1小时: 每天约6.5条
    # 日线: 每天1条
    # 周线: 每周1条
    # 月线: 每月1条
    
    count_1m = days_back * 400  # 多留余量
    count_5m = days_back * 100
    count_1h = days_back * 10
    count_1d = days_back * 2
    count_1w = int(days_back / 7) * 2
    count_1M = int(days_back / 30) * 2
    
    print(f"\n📊 请求数据量:")
    print(f"  1分钟: {count_1m}条")
    print(f"  5分钟: {count_5m}条")
    print(f"  1小时: {count_1h}条")
    print(f"  日线: {count_1d}条")
    print(f"  周线: {count_1w}条")
    print(f"  月线: {count_1M}条")
    
    # 使用现有的数据生成函数
    print(f"\n🔄 开始获取数据...")
    try:
        result_df = generate_training_data_direct(
            count_1m=count_1m,
            count_5m=count_5m,
            count_1h=count_1h,
            count_1d=count_1d,
            count_1w=count_1w,
            count_1M=count_1M,
            output_file=output_file
        )
        
        if result_df is not None:
            print(f"\n✅ 数据获取成功！")
            print(f"  数据形状: {result_df.shape}")
            print(f"  时间范围: {result_df.index[0] if hasattr(result_df, 'index') else 'N/A'} 到 {result_df.index[-1] if hasattr(result_df, 'index') else 'N/A'}")
            
            if output_file:
                print(f"  数据已保存到: {output_file}")
            
            return result_df
        else:
            print("❌ 数据获取失败")
            return None
            
    except Exception as e:
        print(f"❌ 获取数据时出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='获取更多训练数据')
    parser.add_argument('--days', type=int, default=60, help='回溯天数（默认60天）')
    parser.add_argument('--output', type=str, default=None, help='输出文件路径（可选）')
    
    args = parser.parse_args()
    
    # 如果没有指定输出文件，使用默认命名
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f'/home/cx/trading_data/training_data_multitimeframe_extended_{timestamp}.csv'
    
    # 获取数据
    df = fetch_more_data(days_back=args.days, output_file=args.output)
    
    if df is not None:
        print(f"\n✅ 完成！数据已保存到: {args.output}")
    else:
        print("\n❌ 数据获取失败")


if __name__ == "__main__":
    main()
