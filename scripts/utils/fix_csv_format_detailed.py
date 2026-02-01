import pandas as pd
import os

def fix_csv_file(file_path):
    # 读取原始文件
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"原始文件有 {len(lines)} 行")
    
    # 定义正确的表头
    correct_headers = [
        'timestamp', 'price_current', 'grid_lower', 'grid_upper', 'atr', 
        'rsi_1m', 'rsi_5m', 'buffer', 'threshold', 'near_lower', 
        'rsi_ok', 'trend_check', 'rebound', 'vol_ok', 'final_decision',
        'take_profit_price', 'stop_loss_price', 'position_size', 'side',
        'deviation_percent', 'atr_multiplier', 'min_buffer_val', 'market_regime',
        'boll_upper', 'boll_mid', 'boll_lower'
    ]
    
    # 修复每一行
    fixed_lines = [','.join(correct_headers) + '\n']  # 添加正确表头
    
    for i, line in enumerate(lines[1:], 1):  # 跳过原表头
        line = line.rstrip()  # 去掉末尾空白字符
        if line:
            # 如果行以逗号开始（意味着时间戳字段为空），去掉开头的逗号
            if line.startswith(','):
                line = line[1:]
            
            # 确保字段数量正确
            fields = line.split(',')
            if len(fields) != len(correct_headers):
                print(f"警告: 第{i}行有{len(fields)}个字段，期望{len(correct_headers)}个字段")
                print(f"内容: {fields[:10]}...")  # 只打印前10个字段
                
                # 如果字段数少于期望值，用空值填充
                if len(fields) < len(correct_headers):
                    fields.extend([''] * (len(correct_headers) - len(fields)))
                
                # 如果字段数多于期望值，需要合并多余的字段（可能是数据中的逗号导致的分割）
                elif len(fields) > len(correct_headers):
                    # 尝试合并超出的字段
                    merged_fields = fields[:len(correct_headers)-1]  # 保留前面的字段
                    extra_field = ','.join(fields[len(correct_headers)-1:])  # 合并超出的字段
                    merged_fields.append(extra_field)
                    fields = merged_fields
            
            # 重新构建行
            fixed_line = ','.join(fields) + '\n'
            fixed_lines.append(fixed_line)
    
    # 写入修复后的文件
    fixed_file_path = file_path.replace('.csv', '_fixed.csv')
    with open(fixed_file_path, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print(f"修复后的文件已保存到: {fixed_file_path}")
    
    # 验证修复结果
    try:
        df = pd.read_csv(fixed_file_path)
        print(f"✅ CSV文件读取成功，形状: {df.shape}")
        print(f"✅ 列名: {list(df.columns)}")
        print(f"✅ 第一行数据: {df.iloc[0].values[:10]}...")  # 只显示前10个值
        return fixed_file_path
    except Exception as e:
        print(f"❌ 修复后仍无法读取文件: {e}")
        return None

if __name__ == "__main__":
    file_path = "/home/cx/trading_data/trading_data_2026-01-19.csv"
    fixed_file_path = fix_csv_file(file_path)  # 修复函数名
    if fixed_file_path:
        # 替换原文件
        import shutil
        shutil.move(fixed_file_path, file_path)
        print(f"✅ 已将修复后的文件替换原文件: {file_path}")