"""
获取更多训练数据（改进版）
目标：50K+样本，覆盖不同市场状态
"""
import sys
import os
sys.path.insert(0, '/home/cx/tigertrade')

from scripts.analysis.generate_multitimeframe_training_data_direct import generate_training_data_direct
from datetime import datetime, timedelta
import time


def fetch_more_data_comprehensive(days_back=90):
    """
    获取更多历史数据
    Args:
        days_back: 回溯天数
    """
    print("="*70)
    print(f"获取更多训练数据（回溯{days_back}天）")
    print("="*70)
    
    # 计算需要获取的数据量
    # 1分钟: 一天大约390-400条（交易时段）
    # 5分钟: 一天大约78-80条
    # 1小时: 一天大约6-7条
    # 日线: 1天1条
    # 周线: 1周1条
    # 月线: 1月1条
    
    count_1m = days_back * 400  # 1分钟数据
    count_5m = days_back * 100  # 5分钟数据
    count_1h = days_back * 10   # 1小时数据
    count_1d = days_back + 50   # 日线数据（多留一些）
    count_1w = (days_back // 7) + 10  # 周线数据
    count_1M = (days_back // 30) + 5  # 月线数据
    
    print(f"\n📊 数据量估算:")
    print(f"  1分钟: {count_1m:,} 条")
    print(f"  5分钟: {count_5m:,} 条")
    print(f"  1小时: {count_1h:,} 条")
    print(f"  日线: {count_1d:,} 条")
    print(f"  周线: {count_1w:,} 条")
    print(f"  月线: {count_1M:,} 条")
    
    total_samples = 0
    
    try:
        # 生成训练数据（一次性获取所有数据）
        print(f"\n🔄 开始获取数据...")
        df = generate_training_data_direct(
            count_1m=count_1m,
            count_5m=count_5m,
            count_1h=count_1h,
            count_1d=count_1d,
            count_1w=count_1w,
            count_1M=count_1M
        )
        
        if df is not None and len(df) > 0:
            total_samples = len(df)
            print(f"  ✅ 获取了 {total_samples:,} 个样本")
        else:
            print(f"  ⚠️ 未获取到数据")
    
    except Exception as e:
        print(f"  ❌ 数据获取失败: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n" + "="*70)
    print(f"数据获取完成")
    print(f"  总样本数: {total_samples:,}")
    print(f"  目标: 50,000+ 样本")
    if total_samples >= 50000:
        print(f"  ✅ 已达到目标")
    else:
        print(f"  ⚠️ 还需获取 {50000 - total_samples:,} 个样本")
    print("="*70)
    
    return total_samples


if __name__ == '__main__':
    # 获取90天的历史数据
    fetch_more_data_comprehensive(days_back=90)
