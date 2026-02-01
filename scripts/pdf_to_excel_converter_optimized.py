import pandas as pd
from PyPDF2 import PdfReader
import re


def extract_transaction_data_from_pdf(pdf_path, excel_path):
    """
    专门针对账单报表PDF提取交易数据
    """
    try:
        reader = PdfReader(pdf_path)
        print(f"PDF总页数: {len(reader.pages)}")
        
        # 存储交易数据
        transactions = []
        
        for i, page in enumerate(reader.pages):
            print(f"正在处理第 {i+1} 页...")
            text = page.extract_text()
            
            # 按行分割文本
            lines = text.split('\n')
            
            # 解析每一行，查找可能的交易记录
            for line in lines:
                transaction = parse_transaction_line(line)
                if transaction:
                    transactions.append(transaction)
        
        # 创建DataFrame并保存到Excel
        if transactions:
            df = pd.DataFrame(transactions)
            
            # 重新排列列的顺序，让最重要的信息靠前
            cols = ['Date', 'Description', 'Symbol', 'Quantity', 'Price', 'Amount', 'Balance', 'Transaction_Type', 'Original_Line']
            existing_cols = [col for col in cols if col in df.columns]
            remaining_cols = [col for col in df.columns if col not in cols]
            ordered_cols = existing_cols + remaining_cols
            
            df = df[ordered_cols]
            
            # 保存到Excel
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='账单明细', index=False)
                
                # 创建汇总表
                summary_df = create_summary(transactions)
                summary_df.to_excel(writer, sheet_name='汇总', index=False)
            
            print(f"成功将PDF内容保存到Excel: {excel_path}")
            print(f"共提取 {len(df)} 条交易记录")
            return df
        else:
            print("未能从PDF中提取到交易数据")
            return None
            
    except Exception as e:
        print(f"处理PDF时出错: {str(e)}")
        return None


def parse_transaction_line(line):
    """
    解析可能包含交易信息的行
    """
    # 清理行内容
    cleaned_line = line.strip()
    if not cleaned_line or cleaned_line.isspace():
        return None
    
    # 尝试匹配常见的交易记录格式
    # 包含日期、股票代码、交易类型、数量、价格、金额等信息
    transaction = {}
    
    # 匹配日期格式 (YYYY-MM-DD 或 MM/DD/YYYY 或 DD/MM/YYYY)
    date_match = re.search(r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})|(\d{1,2}[-/]\d{1,2}[-/]\d{4})|(\d{1,2}[./]\d{1,2}[./]\d{4})', cleaned_line)
    if date_match:
        transaction['Date'] = date_match.group(0)
    
    # 匹配股票代码 (通常是字母和数字组合)
    # 假设是4-6个字符的字母，后面可能跟有数字
    symbol_match = re.search(r'\b([A-Z]{3,6}\d?)\b', cleaned_line)
    if symbol_match:
        symbol = symbol_match.group(1)
        # 排除一些非股票代码的词
        if symbol not in ['USD', 'CNY', 'HKD', 'AUD', 'CAD', 'GBP', 'EUR', 'JPY', 'NZD']:
            transaction['Symbol'] = symbol
    
    # 匹配数量 (整数或带小数的数量)
    quantity_match = re.search(r'([-+]?\d+\.?\d*)\s*(?:@|$)', cleaned_line)
    if quantity_match:
        try:
            qty = float(quantity_match.group(1))
            if abs(qty) > 0.01:  # 排除非常小的数值
                transaction['Quantity'] = qty
        except ValueError:
            pass
    
    # 匹配价格 (通常以@符号开头)
    price_match = re.search(r'@\s*([+-]?\$?[\d,]+\.?\d*)', cleaned_line)
    if price_match:
        price_str = price_match.group(1).replace(',', '').replace('$', '')
        try:
            price = float(price_str)
            transaction['Price'] = price
        except ValueError:
            pass
    
    # 匹配金额 ($符号后的数字)
    amount_match = re.search(r'([+-]?\$?[\d,]+\.?\d*)', cleaned_line.replace('@', ''))
    if amount_match:
        amount_str = amount_match.group(1).replace(',', '').replace('$', '')
        try:
            # 检查是否是合理的金额值
            amount = float(amount_str)
            # 如果绝对值大于1，则认为是金额（排除小数手续费等）
            if abs(amount) >= 1:
                transaction['Amount'] = amount
        except ValueError:
            pass
    
    # 匹配余额
    balance_match = re.search(r'Balance[:\s]+([+-]?\$?[\d,]+\.?\d*)', cleaned_line, re.IGNORECASE)
    if balance_match:
        balance_str = balance_match.group(1).replace(',', '').replace('$', '')
        try:
            balance = float(balance_str)
            transaction['Balance'] = balance
        except ValueError:
            pass
    
    # 检测交易类型
    if 'Buy' in cleaned_line or 'BUY' in cleaned_line:
        transaction['Transaction_Type'] = 'Buy'
    elif 'Sell' in cleaned_line or 'SELL' in cleaned_line:
        transaction['Transaction_Type'] = 'Sell'
    elif 'Dividend' in cleaned_line or 'DIVIDEND' in cleaned_line:
        transaction['Transaction_Type'] = 'Dividend'
    elif 'Interest' in cleaned_line or 'INTEREST' in cleaned_line:
        transaction['Transaction_Type'] = 'Interest'
    
    # 添加原始行内容
    transaction['Original_Line'] = cleaned_line
    
    # 添加描述信息（去除日期、金额、股票代码等已提取的信息）
    desc_parts = []
    parts = cleaned_line.split()
    for part in parts:
        # 跳过已识别的部分
        if (
            not re.match(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}', part) and
            not re.match(r'\d{1,2}[-/]\d{1,2}[-/]\d{4}', part) and
            not re.match(r'[A-Z]{3,6}\d?', part) and
            not re.match(r'[+-]?\$?[\d,]+\.?\d*', part) and
            '@' not in part
        ):
            desc_parts.append(part)
    
    if desc_parts:
        transaction['Description'] = ' '.join(desc_parts)
    
    # 只有当至少有一个关键字段被识别时才返回
    key_fields = ['Date', 'Symbol', 'Quantity', 'Amount', 'Transaction_Type']
    if any(field in transaction for field in key_fields):
        return transaction
    
    return None


def create_summary(transactions):
    """
    创建交易摘要
    """
    if not transactions:
        return pd.DataFrame({'Summary': ['No transactions to summarize']})
    
    # 计算各类交易的统计信息
    buy_count = sell_count = dividend_count = interest_count = 0
    buy_amount = sell_amount = dividend_amount = interest_amount = 0
    
    for trans in transactions:
        ttype = trans.get('Transaction_Type', '')
        amount = trans.get('Amount', 0)
        
        if ttype == 'Buy':
            buy_count += 1
            buy_amount += amount
        elif ttype == 'Sell':
            sell_count += 1
            sell_amount += amount
        elif ttype == 'Dividend':
            dividend_count += 1
            dividend_amount += amount
        elif ttype == 'Interest':
            interest_count += 1
            interest_amount += amount
    
    summary_data = {
        '类别': ['买入交易', '卖出交易', '股息收入', '利息收入', '总计'],
        '次数': [buy_count, sell_count, dividend_count, interest_count, 
                 buy_count + sell_count + dividend_count + interest_count],
        '总金额': [buy_amount, sell_amount, dividend_amount, interest_amount,
                  buy_amount + sell_amount + dividend_amount + interest_amount]
    }
    
    return pd.DataFrame(summary_data)


if __name__ == "__main__":
    pdf_path = "/home/cx/tigertrade/docs/今年以来账单报表 (2).pdf"
    excel_path = "/home/cx/账单报表整理_优化版.xlsx"
    
    print("开始处理PDF文件...")
    df = extract_transaction_data_from_pdf(pdf_path, excel_path)
    
    if df is not None:
        print("\n前几行数据预览:")
        print(df.head(10))
        
        print(f"\n数据形状: {df.shape}")
        print(f"列名: {list(df.columns)}")
    else:
        print("PDF处理失败")