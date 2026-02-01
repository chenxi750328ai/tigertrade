#!/usr/bin/env python3
"""调试完整的特征计算流程"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from collect_large_dataset import LargeDatasetCollector
from tiger1 import calculate_indicators
from config import DataConfig

def main():
    print("=" * 80)
    print("调试完整特征计算流程")
    print("=" * 80)
    
    # 创建收集器
    collector = LargeDatasetCollector(
        use_real_api=True,
        days=90,
        max_records=50000
    )
    
    # 获取数据
    print("\n获取测试数据...")
    df_5m = collector.fetch_kline_data_with_retry('5min', 1000)
    df_1m = collector.fetch_kline_data_with_retry('1min', 5000)
    
    print(f"\n5分钟数据: {len(df_5m)} 条")
    print(f"1分钟数据: {len(df_1m)} 条")
    
    # 模拟循环
    min_len = DataConfig.MIN_REQUIRED_BARS
    window_size = DataConfig.WINDOW_SIZE
    
    print(f"\n测试循环（索引990-1010）...")
    for i in [999, 1000, 1001]:
        if i >= len(df_5m):
            break
        
        print(f"\n=== 索引 {i} ===")
        try:
            window_5m = df_5m.iloc[max(0, i-window_size):i+1]
            timestamp_5m = df_5m.index[i]
            df_1m_slice = df_1m[df_1m.index <= timestamp_5m]
            
            print(f"  window_5m shape: {window_5m.shape}")
            print(f"  window_5m columns: {window_5m.columns.tolist()}")
            print(f"  timestamp_5m: {timestamp_5m}")
            print(f"  df_1m_slice len: {len(df_1m_slice)}")
            
            if len(df_1m_slice) < min_len:
                print(f"  ⚠️ df_1m_slice数据不足，跳过")
                continue
            
            window_1m = df_1m_slice.iloc[-window_size:]
            print(f"  window_1m shape: {window_1m.shape}")
            print(f"  window_1m columns: {window_1m.columns.tolist()}")
            
            # 调用 calculate_indicators
            print(f"  调用 calculate_indicators...")
            inds = calculate_indicators(window_1m, window_5m)
            
            print(f"  ✅ 成功: inds['5m']['close'] = {inds['5m']['close']}")
            
        except Exception as e:
            print(f"  ❌ 错误: {e}")
            
            # 打印详细信息
            print(f"\n  详细调试:")
            print(f"    window_5m type: {type(window_5m)}")
            print(f"    window_5m shape: {window_5m.shape}")
            print(f"    window_5m.iloc[-1] type: {type(window_5m.iloc[-1])}")
            print(f"    window_5m.iloc[-1]:\n{window_5m.iloc[-1]}")
            
            import traceback
            print(f"\n  详细错误:\n{traceback.format_exc()}")

if __name__ == "__main__":
    main()
