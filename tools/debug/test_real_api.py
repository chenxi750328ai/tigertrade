#!/usr/bin/env python3
"""测试真实API的数据获取和格式"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# 设置环境变量
os.environ['USE_REAL_API'] = '1'

from collect_large_dataset import LargeDatasetCollector

def main():
    print("=" * 80)
    print("测试真实API数据采集")
    print("=" * 80)
    
    # 创建收集器（使用真实API）
    collector = LargeDatasetCollector(
        use_real_api=True,
        days=5,  # 少量天数测试
        max_records=2000
    )
    
    print("\n测试获取少量数据...")
    
    # 测试获取1分钟数据
    try:
        df_1m = collector.fetch_kline_data_with_retry('1min', 100)
        print(f"\n1分钟数据:")
        print(f"  类型: {type(df_1m)}")
        print(f"  形状: {df_1m.shape}")
        print(f"  列名: {df_1m.columns.tolist()}")
        print(f"  索引类型: {type(df_1m.index)}")
        print(f"\n前5行:")
        print(df_1m.head())
        
        # 测试切片
        print(f"\n测试切片...")
        window = df_1m.iloc[-10:]
        print(f"  切片形状: {window.shape}")
        print(f"  切片列名: {window.columns.tolist()}")
        print(f"  可访问close列: {window['close'].iloc[-1]}")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试获取5分钟数据
    try:
        df_5m = collector.fetch_kline_data_with_retry('5min', 100)
        print(f"\n5分钟数据:")
        print(f"  类型: {type(df_5m)}")
        print(f"  形状: {df_5m.shape}")
        print(f"  列名: {df_5m.columns.tolist()}")
        print(f"  索引类型: {type(df_5m.index)}")
        print(f"\n前5行:")
        print(df_5m.head())
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
