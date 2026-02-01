#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试用例统计脚本
统计项目中的所有测试用例
"""
import os
import re
import sys

def count_test_cases_in_file(file_path):
    """统计单个文件中的测试用例数量"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # 匹配 def test_ 开头的函数
            test_pattern = r'def\s+test_\w+\s*\('
            matches = re.findall(test_pattern, content)
            return len(matches)
    except Exception as e:
        print(f"⚠️ 读取文件失败 {file_path}: {e}")
        return 0

def analyze_tests():
    """分析所有测试文件"""
    tests_dir = '/home/cx/tigertrade/tests'
    
    if not os.path.exists(tests_dir):
        print(f"❌ 测试目录不存在: {tests_dir}")
        return
    
    test_files = []
    total_cases = 0
    
    # 查找所有测试文件
    for root, dirs, files in os.walk(tests_dir):
        for file in files:
            if file.startswith('test_') and file.endswith('.py'):
                file_path = os.path.join(root, file)
                case_count = count_test_cases_in_file(file_path)
                test_files.append((file_path, case_count))
                total_cases += case_count
    
    # 按用例数量排序
    test_files.sort(key=lambda x: x[1], reverse=True)
    
    # 输出统计结果
    print("="*70)
    print("📊 测试用例统计报告")
    print("="*70)
    print(f"\n📁 测试目录: {tests_dir}")
    print(f"📄 测试文件数: {len(test_files)}")
    print(f"🧪 总测试用例数: {total_cases}")
    print(f"\n✅ 已超过260个用例的目标！" if total_cases >= 260 else f"\n⚠️ 未达到260个用例的目标，还差{260-total_cases}个")
    
    print("\n" + "="*70)
    print("📋 测试文件详情（按用例数量排序）")
    print("="*70)
    
    for i, (file_path, count) in enumerate(test_files[:30], 1):  # 显示前30个
        rel_path = os.path.relpath(file_path, tests_dir)
        print(f"{i:2d}. {rel_path:50s} : {count:3d} 个用例")
    
    if len(test_files) > 30:
        print(f"\n... 还有 {len(test_files) - 30} 个文件")
    
    # 分类统计
    print("\n" + "="*70)
    print("📊 分类统计")
    print("="*70)
    
    categories = {
        'tiger1核心': 0,
        '策略模块': 0,
        '执行器': 0,
        '集成测试': 0,
        '其他': 0
    }
    
    for file_path, count in test_files:
        filename = os.path.basename(file_path)
        if 'tiger1' in filename.lower():
            categories['tiger1核心'] += count
        elif 'strategy' in filename.lower():
            categories['策略模块'] += count
        elif 'executor' in filename.lower():
            categories['执行器'] += count
        elif 'integration' in filename.lower() or 'demo' in filename.lower():
            categories['集成测试'] += count
        else:
            categories['其他'] += count
    
    for category, count in categories.items():
        percentage = (count / total_cases * 100) if total_cases > 0 else 0
        print(f"{category:15s}: {count:4d} 个用例 ({percentage:5.1f}%)")
    
    print("\n" + "="*70)
    print("✅ 统计完成")
    print("="*70)

if __name__ == '__main__':
    analyze_tests()
