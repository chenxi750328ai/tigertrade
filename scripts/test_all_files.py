#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试所有Python文件是否可以正常导入和运行
"""

import os
import sys
import importlib.util
from pathlib import Path

# 添加tigertrade目录到路径
sys.path.insert(0, '/home/cx/tigertrade')

def test_python_file(file_path):
    """测试单个Python文件"""
    try:
        # 读取文件检查语法
        with open(file_path, 'r', encoding='utf-8') as f:
            compile(f.read(), file_path, 'exec')
        return True, "语法正确"
    except SyntaxError as e:
        return False, f"语法错误: {e}"
    except Exception as e:
        return False, f"其他错误: {e}"

def scan_directory(directory, base_path):
    """扫描目录中的所有Python文件"""
    results = {
        'total': 0,
        'passed': 0,
        'failed': 0,
        'errors': []
    }
    
    for root, dirs, files in os.walk(directory):
        # 跳过__pycache__和.git目录
        dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', 'htmlcov', '.pytest_cache']]
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, base_path)
                
                results['total'] += 1
                success, message = test_python_file(file_path)
                
                if success:
                    results['passed'] += 1
                    print(f"✅ {rel_path}")
                else:
                    results['failed'] += 1
                    results['errors'].append((rel_path, message))
                    print(f"❌ {rel_path}: {message}")
    
    return results

def main():
    """主函数"""
    print("🚀 开始测试所有Python文件...\n")
    
    base_path = '/home/cx/tigertrade'
    
    # 测试各个目录
    directories = {
        'src': os.path.join(base_path, 'src'),
        'tests': os.path.join(base_path, 'tests'),
        'scripts': os.path.join(base_path, 'scripts'),
    }
    
    all_results = {
        'total': 0,
        'passed': 0,
        'failed': 0,
        'errors': []
    }
    
    for name, directory in directories.items():
        print(f"\n{'='*60}")
        print(f"测试 {name}/ 目录")
        print('='*60)
        
        if os.path.exists(directory):
            results = scan_directory(directory, base_path)
            all_results['total'] += results['total']
            all_results['passed'] += results['passed']
            all_results['failed'] += results['failed']
            all_results['errors'].extend(results['errors'])
            
            print(f"\n{name}/ 小结: {results['passed']}/{results['total']} 通过")
        else:
            print(f"⚠️  目录不存在: {directory}")
    
    # 打印总结
    print(f"\n{'='*60}")
    print("总体测试结果")
    print('='*60)
    print(f"总文件数: {all_results['total']}")
    print(f"通过: {all_results['passed']} ✅")
    print(f"失败: {all_results['failed']} ❌")
    print(f"通过率: {all_results['passed']/all_results['total']*100:.1f}%")
    
    if all_results['errors']:
        print(f"\n错误详情:")
        for file, error in all_results['errors']:
            print(f"  ❌ {file}")
            print(f"     {error}")
    
    print(f"\n{'='*60}")
    if all_results['failed'] == 0:
        print("🎉 所有文件测试通过！")
        return 0
    else:
        print(f"⚠️  有 {all_results['failed']} 个文件存在问题")
        return 1

if __name__ == "__main__":
    sys.exit(main())
