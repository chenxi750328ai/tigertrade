#!/usr/bin/env python3
"""
修复所有策略的prepare_features方法
统一使用12个正确的特征
"""

import sys
import os

# 正确的12个特征
CORRECT_FEATURES = """
        try:
            features = [
                row['price_current'],
                row['atr'],
                row['rsi_1m'] if pd.notna(row['rsi_1m']) else 50,
                row['rsi_5m'] if pd.notna(row['rsi_5m']) else 50,
                row.get('boll_upper', 0),
                row.get('boll_mid', 0),
                row.get('boll_lower', 0),
                row.get('boll_position', 0.5),
                row.get('price_change_1', 0),
                row.get('price_change_5', 0),
                row.get('volatility', 0),
                row.get('volume_1m', 0)
            ]
            # 归一化特征
            features_np = np.array(features)
            mean_val = np.mean(features_np)
            std_val = np.std(features_np) + 1e-8
            normalized_features = (features_np - mean_val) / std_val
            return normalized_features.tolist()
        except Exception as e:
            print(f"prepare_features错误: {e}")
            # 返回默认特征值
            return [0.0] * 12
"""

# 需要修复的文件列表
FILES = [
    'src/strategies/huge_transformer_strategy.py',
    'src/strategies/enhanced_transformer_strategy.py',
    'src/strategies/large_transformer_strategy.py',
    'src/strategies/llm_strategy.py',
    'src/strategies/large_model_strategy.py'
]

def fix_file(filepath):
    """修复单个文件的prepare_features方法"""
    print(f"修复: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找prepare_features方法
    if 'def prepare_features(self, row):' not in content:
        print(f"  跳过：未找到prepare_features方法")
        return False
    
    # 找到方法的开始和结束
    import re
    
    # 匹配整个prepare_features方法
    pattern = r'(    def prepare_features\(self, row\):.*?"""从数据行中准备特征向量""")(.*?)(return \[0\.0\] \* 10)'
    
    def replacer(match):
        return match.group(1) + CORRECT_FEATURES.rstrip() + '\n    \n    '
    
    new_content = re.sub(pattern, replacer, content, flags=re.DOTALL)
    
    if new_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"  ✅ 已修复")
        return True
    else:
        print(f"  ⚠️ 未找到匹配的模式")
        return False

if __name__ == "__main__":
    os.chdir('/home/cx/tigertrade')
    
    print("=" * 80)
    print("修复所有策略的prepare_features方法")
    print("=" * 80)
    
    fixed_count = 0
    for file in FILES:
        if os.path.exists(file):
            if fix_file(file):
                fixed_count += 1
        else:
            print(f"文件不存在: {file}")
    
    print(f"\n修复完成！共修复 {fixed_count} 个文件")
