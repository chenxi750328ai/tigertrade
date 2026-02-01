import pandas as pd
from PyPDF2 import PdfReader
import re


def extract_pdf_content_to_excel(pdf_path, excel_path):
    """
    从PDF中提取内容并整理成Excel格式
    """
    try:
        reader = PdfReader(pdf_path)
        print(f"PDF总页数: {len(reader.pages)}")
        
        # 创建一个列表来存储所有页面的数据
        all_data = []
        
        for i, page in enumerate(reader.pages):
            print(f"正在处理第 {i+1} 页...")
            text = page.extract_text()
            
            # 分析每一页的内容，提取表格数据
            lines = text.split('\n')
            
            # 尝试识别表格数据
            page_data = parse_pdf_page(lines)
            all_data.extend(page_data)
        
        # 创建DataFrame
        if all_data:
            df = pd.DataFrame(all_data)
            # 保存到Excel
            df.to_excel(excel_path, index=False, engine='openpyxl')
            print(f"成功将PDF内容保存到Excel: {excel_path}")
            print(f"共提取 {len(df)} 行数据")
            return df
        else:
            print("未能从PDF中提取到数据")
            return None
            
    except Exception as e:
        print(f"处理PDF时出错: {str(e)}")
        return None


def parse_pdf_page(lines):
    """
    解析PDF页面的行内容，尝试识别表格数据
    """
    data_rows = []
    
    for line in lines:
        # 清理行内容
        cleaned_line = line.strip()
        
        # 如果行不为空，尝试解析为表格数据
        if cleaned_line:
            # 尝试识别可能的表格行 - 基于常见模式
            # 比如包含日期、金额、编号等元素的行
            row_data = extract_table_like_data(cleaned_line)
            if row_data:
                data_rows.append(row_data)
    
    return data_rows


def extract_table_like_data(text_line):
    """
    从文本行中提取表格样式的数据
    """
    # 尝试识别可能包含交易记录的数据行
    # 这里可以根据实际PDF的格式调整正则表达式
    
    # 尝试匹配包含日期、金额等信息的行
    # 常见的日期格式: YYYY-MM-DD, DD/MM/YYYY, MM/DD/YYYY 等
    date_pattern = r'\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4}'
    amount_pattern = r'[+-]?\$?[\d,]+\.?\d*'
    number_pattern = r'\d+'
    
    # 提取所有匹配的元素
    dates = re.findall(date_pattern, text_line)
    amounts = re.findall(amount_pattern, text_line)
    numbers = re.findall(number_pattern, text_line)
    
    # 如果这一行包含日期和金额，可能是一条交易记录
    if dates or amounts:
        # 创建一个字典表示一行数据
        row_dict = {
            '原文本': text_line,
            '日期': ', '.join(dates) if dates else '',
            '金额': ', '.join(amounts) if amounts else '',
            '数字': ', '.join(numbers) if numbers else ''
        }
        return row_dict
    
    # 如果没找到日期和金额，返回原始文本
    return {'原始文本': text_line}


if __name__ == "__main__":
    pdf_path = "/home/cx/tigertrade/docs/今年以来账单报表 (2).pdf"
    excel_path = "/home/cx/账单报表整理.xlsx"
    
    print("开始处理PDF文件...")
    df = extract_pdf_content_to_excel(pdf_path, excel_path)
    
    if df is not None:
        print("\n前几行数据预览:")
        print(df.head(10))
    else:
        print("PDF处理失败")