#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
识别可能重复或不再需要的文件
"""

import os
from collections import defaultdict
from pathlib import Path

def analyze_files():
    """分析文件并识别可能的重复"""
    base_path = Path('/home/cx/tigertrade/tests')
    
    # 按功能分类文件
    categories = defaultdict(list)
    
    for file in base_path.glob('*.py'):
        name = file.name
        
        # 跳过__init__.py
        if name == '__init__.py':
            continue
        
        # 分类
        if 'coverage' in name:
            categories['覆盖率测试'].append(name)
        elif 'comprehensive' in name:
            categories['综合测试'].append(name)
        elif 'final' in name:
            categories['最终测试'].append(name)
        elif 'debug' in name or 'exact' in name:
            categories['调试测试'].append(name)
        elif 'mock' in name:
            categories['模拟测试'].append(name)
        elif 'validation' in name or 'verification' in name:
            categories['验证测试'].append(name)
        elif name.startswith('test_tiger1'):
            categories['tiger1测试'].append(name)
        elif name.startswith('test_'):
            categories['功能测试'].append(name)
        else:
            categories['其他'].append(name)
    
    print("=" * 60)
    print("测试文件分类统计")
    print("=" * 60)
    
    for category, files in sorted(categories.items()):
        print(f"\n【{category}】({len(files)}个)")
        for f in sorted(files):
            print(f"  - {f}")
    
    # 识别可能重复的文件
    print("\n" + "=" * 60)
    print("可能需要清理的文件（建议）")
    print("=" * 60)
    
    # 覆盖率测试 - 可能有很多重复
    if len(categories['覆盖率测试']) > 5:
        print(f"\n⚠️  覆盖率测试文件过多({len(categories['覆盖率测试'])}个)")
        print("   建议：只保留最新的几个覆盖率测试文件")
        print("   可以考虑删除旧的测试版本")
    
    # 综合测试 - 可能有重复
    if len(categories['综合测试']) > 3:
        print(f"\n⚠️  综合测试文件过多({len(categories['综合测试'])}个)")
        print("   建议：只保留1-2个综合测试文件")
    
    # 最终测试 - 只需要保留一个
    if len(categories['最终测试']) > 2:
        print(f"\n⚠️  最终测试文件过多({len(categories['最终测试'])}个)")
        print("   建议：只保留最新的最终测试文件")
    
    # 调试测试 - 可能是临时文件
    if len(categories['调试测试']) > 0:
        print(f"\n⚠️  调试测试文件({len(categories['调试测试'])}个)")
        print("   建议：调试完成后可以删除这些临时文件")
    
    # 总结
    total_files = sum(len(files) for files in categories.values())
    print(f"\n" + "=" * 60)
    print(f"总计：{total_files}个测试文件")
    print("=" * 60)
    
    # 具体建议删除的文件
    print("\n" + "=" * 60)
    print("具体清理建议（可安全删除的文件）")
    print("=" * 60)
    
    old_coverage_files = [
        'additional_coverage_test.py',
        'remaining_coverage_test.py',
        'full_coverage_test.py',
        'ultimate_coverage_test.py',
    ]
    
    old_comprehensive_files = [
        'comprehensive_calculation_test.py',
        'comprehensive_path_test.py',
        'comprehensive_test.py',
        'comprehensive_final_test.py',
    ]
    
    debug_files = [
        'exact_debug_test.py',
    ]
    
    old_final_files = [
        'final_100_percent_test.py',
        'final_complete_coverage_test.py',
        'final_coverage_test.py',
    ]
    
    mock_files = [
        'mock_real_api_test.py',
        'fixed_mock_real_api_test.py',
    ]
    
    other_old_files = [
        'test2.py',
        'test_fix.py',
        'end_to_end_test.py',
        'actual_test_verification.py',
    ]
    
    print("\n【旧版覆盖率测试】可删除（已有新版）")
    for f in old_coverage_files:
        if f in categories['覆盖率测试']:
            print(f"  - {f}")
    
    print("\n【旧版综合测试】可删除（已有test_tiger1_comprehensive.py）")
    for f in old_comprehensive_files:
        if f in categories['综合测试']:
            print(f"  - {f}")
    
    print("\n【调试临时文件】可删除（调试已完成）")
    for f in debug_files:
        if f in categories['调试测试']:
            print(f"  - {f}")
    
    print("\n【旧版最终测试】可删除（已有final_tiger1_coverage_test.py）")
    for f in old_final_files:
        if f in categories['最终测试']:
            print(f"  - {f}")
    
    print("\n【旧版模拟测试】可删除（功能已整合）")
    for f in mock_files:
        if f in categories['模拟测试']:
            print(f"  - {f}")
    
    print("\n【其他旧文件】可删除（已过时）")
    for f in other_old_files:
        for cat_files in categories.values():
            if f in cat_files:
                print(f"  - {f}")
                break
    
    # 建议保留的核心文件
    print("\n" + "=" * 60)
    print("建议保留的核心测试文件")
    print("=" * 60)
    
    keep_files = [
        'test_tiger1_strategies.py',
        'test_tiger1_comprehensive.py',
        'final_tiger1_coverage_test.py',
        'tiger1_full_coverage_test.py',
        'test_tiger1_complete_coverage.py',
        'test_tiger1_phase4_coverage.py',
        'test_boll1m_grid.py',
        'test_place_tiger_order.py',
        'test_grid_trading_strategy_pro1.py',
        'test_api_connection.py',
        'test_calculate_indicators.py',
        'test_kline_data.py',
        'final_validation.py',
        'final_verification.py',
    ]
    
    print("\n【核心测试文件】应保留")
    for f in keep_files:
        print(f"  ✅ {f}")

if __name__ == "__main__":
    analyze_files()
