#!/usr/bin/env python3
"""调试真实API返回的数据格式"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from tiger1 import get_kline_data, FUTURE_SYMBOL

def main():
    print("=" * 80)
    print("调试真实API数据格式")
    print("=" * 80)
    
    # 获取少量数据进行测试
    print(f"\n标的: {FUTURE_SYMBOL}")
    print("获取10条1分钟数据...")
    
    df_1m = get_kline_data([FUTURE_SYMBOL], '1min', count=10)
    
    print(f"\n1分钟数据:")
    print(f"  数据类型: {type(df_1m)}")
    print(f"  数据形状: {df_1m.shape}")
    print(f"  列名: {df_1m.columns.tolist()}")
    print(f"  索引类型: {type(df_1m.index)}")
    print(f"  索引名: {df_1m.index.name}")
    print(f"\n前5行数据:")
    print(df_1m.head())
    
    print(f"\n获取10条5分钟数据...")
    df_5m = get_kline_data([FUTURE_SYMBOL], '5min', count=10)
    
    print(f"\n5分钟数据:")
    print(f"  数据类型: {type(df_5m)}")
    print(f"  数据形状: {df_5m.shape}")
    print(f"  列名: {df_5m.columns.tolist()}")
    print(f"  索引类型: {type(df_5m.index)}")
    print(f"  索引名: {df_5m.index.name}")
    print(f"\n前5行数据:")
    print(df_5m.head())
    
    # 测试访问列
    print(f"\n测试访问列...")
    try:
        print(f"  df_1m['close'].iloc[-1] = {df_1m['close'].iloc[-1]}")
        print(f"  ✅ 可以访问 close 列")
    except Exception as e:
        print(f"  ❌ 无法访问 close 列: {e}")
    
    try:
        print(f"  df_5m['close'].iloc[-1] = {df_5m['close'].iloc[-1]}")
        print(f"  ✅ 可以访问 close 列")
    except Exception as e:
        print(f"  ❌ 无法访问 close 列: {e}")

if __name__ == "__main__":
    main()
