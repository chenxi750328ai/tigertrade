import pandas as pd
import os

def fix_csv_format(file_path):
    # 读取原始CSV文件作为文本
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 检查表头
    header_line = lines[0].strip()
    headers = header_line.split(',')
    
    # 修复数据行 - 删除开头多余的逗号
    fixed_lines = [header_line + '\n']  # 添加表头
    
    for line in lines[1:]:
        line = line.strip()
        if line.startswith(','):
            line = line[1:]  # 移除开头的逗号
        fixed_lines.append(line + '\n')
    
    # 写回文件
    fixed_file_path = file_path.replace('.csv', '_fixed.csv')
    with open(fixed_file_path, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print(f"已修复CSV文件，保存到: {fixed_file_path}")
    
    # 验证修复结果
    try:
        df = pd.read_csv(fixed_file_path)
        print(f"✅ CSV文件读取成功，形状: {df.shape}")
        print(f"✅ 第一行数据: {df.iloc[0].values}")
        return fixed_file_path
    except Exception as e:
        print(f"❌ 修复后仍无法读取文件: {e}")
        return None

if __name__ == "__main__":
    file_path = "/home/cx/trading_data/trading_data_2026-01-19.csv"
    fixed_file_path = fix_csv_format(file_path)
    if fixed_file_path:
        # 替换原文件
        import shutil
        shutil.move(fixed_file_path, file_path)
        print(f"✅ 已将修复后的文件替换原文件: {file_path}")