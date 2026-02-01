#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
检查tiger1.py中是否有未定义的函数
"""

import ast
import re

def find_undefined_functions(file_path):
    """查找文件中调用但未定义的函数"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    tree = ast.parse(content)
    
    # 获取所有函数定义
    defined_functions = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            defined_functions.add(node.name)
    
    # 获取所有函数调用
    called_functions = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called_functions.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                # 处理 obj.method() 形式的调用
                if isinstance(node.func.value, ast.Name):
                    # 这种情况是外部模块的方法调用，不需要在当前文件中定义
                    continue
                else:
                    # 可能是 self.method() 或其他对象方法
                    called_functions.add(node.func.attr)
    
    # 获取导入的模块和别名
    imported_items = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_items.add(alias.name)
                if alias.asname:
                    imported_items.add(alias.asname)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported_items.add(alias.name)
                if alias.asname:
                    imported_items.add(alias.asname)
    
    # 查找可能的外部库函数调用
    # 通过正则表达式查找可能的函数调用模式
    # 这里我们特别关注直接函数调用，而非对象方法调用
    external_calls = set()
    pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
    matches = re.findall(pattern, content)
    
    for match in matches:
        # 跳过Python关键字和内置函数
        if match not in ['if', 'for', 'while', 'def', 'class', 'import', 'from', 'as', 'with', 'try', 'except', 'finally', 'lambda', 'and', 'or', 'not', 'in', 'is', 'return', 'yield', 'break', 'continue', 'pass', 'raise', 'assert', 'del', 'global', 'nonlocal']:
            if match not in ['print', 'len', 'range', 'list', 'dict', 'tuple', 'set', 'str', 'int', 'float', 'bool', 'max', 'min', 'sum', 'abs', 'round', 'isinstance', 'hasattr', 'getattr', 'setattr', 'enumerate', 'zip', 'map', 'filter', 'open', 'input', 'type', 'id', 'dir', 'vars', 'locals', 'globals', 'all', 'any', 'sorted', 'reversed', 'callable', 'hash', 'format', 'ord', 'chr', 'hex', 'oct', 'bin', 'pow', 'divmod']:
                external_calls.add(match)
    
    # 分析哪些函数是未定义的
    undefined = set()
    
    # 检查直接调用的函数中哪些不在定义列表中，也不是导入项
    for func in called_functions:
        if func not in defined_functions and func not in imported_items:
            undefined.add(func)
    
    # 检查正则表达式找到的调用
    for func in external_calls:
        if func not in defined_functions and func not in imported_items:
            # 检查是否是对象方法调用（如obj.method）或模块函数调用（如module.function）
            # 这些通常不是当前模块需要定义的
            undefined.add(func)
    
    return sorted(list(undefined)), sorted(list(defined_functions))

def main():
    print("🔍 检查tiger1.py中的未定义函数...")
    
    undefined_funcs, defined_funcs = find_undefined_functions('/home/cx/tigertrade/tiger1.py')
    
    print(f"✅ 已定义函数数量: {len(defined_funcs)}")
    print(f"🔍 已定义的部分函数: {defined_funcs[:10]}...")  # 显示前10个
    
    if undefined_funcs:
        print(f"\n❌ 发现 {len(undefined_funcs)} 个可能未定义的函数:")
        for func in undefined_funcs:
            print(f"  - {func}")
        
        print("\n⚠️  注意: 这些标记为'未定义'的函数可能是:")
        print("  - 外部库函数 (如pandas, numpy等)")
        print("  - 模块级变量或对象的方法")
        print("  - Python内置函数")
        print("  - 通过from import导入的函数")
        
        # 特别检查一些常见的可能问题
        common_issues = ['compute_stop_loss', 'place_take_profit_order']
        for issue in common_issues:
            if issue in undefined_funcs:
                print(f"\n🚨 严重警告: {issue} 在可能未定义的函数列表中!")
    else:
        print("\n✅ 未发现未定义的函数!")
    
    # 额外检查特定函数是否存在
    print(f"\n🔍 验证特定函数的存在性:")
    essential_funcs = [
        'check_risk_control',  # 之前有问题的函数
        'compute_stop_loss',
        'calculate_indicators', 
        'get_kline_data',
        'place_tiger_order',
        'judge_market_trend',
        'adjust_grid_interval',
        'place_take_profit_order'
    ]
    
    for func_name in essential_funcs:
        if func_name in defined_funcs:
            print(f"  ✅ {func_name}: DEFINED")
        else:
            print(f"  ❌ {func_name}: MISSING")

if __name__ == "__main__":
    main()