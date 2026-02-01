#!/usr/bin/env python3
"""检查Tiger API返回的数据列名"""

import sys
sys.path.insert(0, '/home/cx/tigertrade/src')

from tiger1 import get_kline_data, FUTURE_SYMBOL

print("=" * 80)
print("测试获取K线数据的列名")
print("=" * 80)

# 获取5分钟数据
print("\n获取5分钟K线数据...")
df_5m = get_kline_data(symbol=FUTURE_SYMBOL, period='5min', count=10)
print(f"5分钟数据形状: {df_5m.shape}")
print(f"5分钟数据列名: {list(df_5m.columns)}")
print(f"5分钟数据前3行:")
print(df_5m.head(3))

# 获取1分钟数据
print("\n获取1分钟K线数据...")
df_1m = get_kline_data(symbol=FUTURE_SYMBOL, period='1min', count=10)
print(f"1分钟数据形状: {df_1m.shape}")
print(f"1分钟数据列名: {list(df_1m.columns)}")
print(f"1分钟数据前3行:")
print(df_1m.head(3))

print("\n" + "=" * 80)
