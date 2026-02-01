#!/usr/bin/env python3
"""
完整修复tiger1.py的calculate_indicators函数
"""

with open('/home/cx/tigertrade/src/tiger1.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 找到calculate_indicators函数并修复
in_function = False
indent_count = 0
new_lines = []
i = 0

while i < len(lines):
    line = lines[i]
    
    # 在函数开始处添加列检查
    if 'def calculate_indicators(df_1m, df_5m):' in line:
        in_function = True
        new_lines.append(line)
        # 跳过注释和docstring
        i += 1
        while i < len(lines) and (lines[i].strip().startswith('"""') or lines[i].strip().startswith(':') or lines[i].strip() == '"""'):
            new_lines.append(lines[i])
            i += 1
        
        # 添加完整的列检查逻辑
        new_lines.append('    # 完整检查：确保DataFrame有所需列\n')
        new_lines.append('    required_cols = ["open", "high", "low", "close", "volume"]\n')
        new_lines.append('    \n')
        new_lines.append('    # 检查并修复1分钟数据\n')
        new_lines.append('    if len(df_1m) == 0 or not all(col in df_1m.columns for col in required_cols):\n')
        new_lines.append('        # 数据无效，返回默认值\n')
        new_lines.append('        return {\n')
        new_lines.append('            "1m": {"close": 0, "high": 0, "low": 0, "open": 0, "volume": 0, "rsi": 50, "atr": 0},\n')
        new_lines.append('            "5m": {"close": 0, "high": 0, "low": 0, "open": 0, "volume": 0, "rsi": 50, "atr": 0,\n')
        new_lines.append('                   "boll_upper": 0, "boll_lower": 0, "boll_middle": 0, "boll_mid": 0}\n')
        new_lines.append('        }\n')
        new_lines.append('    \n')
        new_lines.append('    # 检查并修复5分钟数据\n')
        new_lines.append('    if len(df_5m) == 0 or not all(col in df_5m.columns for col in required_cols):\n')
        new_lines.append('        # 5分钟数据无效，使用1分钟数据替代\n')
        new_lines.append('        latest_1m = df_1m.iloc[-1]\n')
        new_lines.append('        return {\n')
        new_lines.append('            "1m": {"close": latest_1m["close"], "high": latest_1m["high"], "low": latest_1m["low"],\n')
        new_lines.append('                   "open": latest_1m["open"], "volume": latest_1m["volume"], "rsi": 50, "atr": 0},\n')
        new_lines.append('            "5m": {"close": latest_1m["close"], "high": latest_1m["high"], "low": latest_1m["low"],\n')
        new_lines.append('                   "open": latest_1m["open"], "volume": latest_1m["volume"], "rsi": 50, "atr": 0,\n')
        new_lines.append('                   "boll_upper": latest_1m["close"] * 1.02, "boll_lower": latest_1m["close"] * 0.98,\n')
        new_lines.append('                   "boll_middle": latest_1m["close"], "boll_mid": latest_1m["close"]}\n')
        new_lines.append('        }\n')
        new_lines.append('    \n')
        continue
    
    new_lines.append(line)
    i += 1

# 写回文件
with open('/home/cx/tigertrade/src/tiger1.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("✅ 完成修复！添加了完整的DataFrame列检查")
