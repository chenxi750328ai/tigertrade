#!/usr/bin/env python3
"""
修复calculate_indicators函数，添加完整的列名检查
"""

import sys

# 读取文件
with open('/home/cx/tigertrade/src/tiger1.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 找到calculate_indicators函数
import re

# 在函数开始添加列名检查
new_func_start = '''def calculate_indicators(df_1m, df_5m):
    """
    计算技术指标
    
    参数:
        df_1m: 1分钟K线数据
        df_5m: 5分钟K线数据
    """
    indicators = {'1m': {}, '5m': {}}
    
    # 检查DataFrame是否有必需的列
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    
    # 检查1分钟数据
    if len(df_1m) == 0 or not all(col in df_1m.columns for col in required_cols):
        print(f"⚠️ 1分钟数据无效: len={len(df_1m)}, cols={list(df_1m.columns) if len(df_1m) > 0 else '[]'}")
        # 返回默认值
        for key in required_cols:
            indicators['1m'][key] = 0
        indicators['1m']['rsi'] = 50
        indicators['1m']['atr'] = 0
    else:'''

# 使用正则替换
pattern = r'def calculate_indicators\(df_1m, df_5m\):.*?indicators = \{\'1m\': \{\}, \'5m\': \{\}\}'
replacement = new_func_start

content_new = re.sub(pattern, replacement, content, flags=re.DOTALL, count=1)

# 保存
with open('/home/cx/tigertrade/src/tiger1.py', 'w', encoding='utf-8') as f:
    f.write(content_new)

print("✅ 已添加列名检查逻辑")
