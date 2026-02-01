#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证所有tiger1相关文件是否正常工作
"""

import sys
import os
import subprocess

# 添加tigertrade目录到路径
sys.path.insert(0, '/home/cx/tigertrade')

def print_section(title):
    """打印分节标题"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def validate_imports():
    """验证所有模块是否可以正常导入"""
    print_section("1. 验证模块导入")
    
    try:
        from src import tiger1 as t1
        print("✅ tiger1模块导入成功")
    except Exception as e:
        print(f"❌ tiger1模块导入失败: {e}")
        return False
    
    try:
        from src.api_adapter import api_manager
        print("✅ api_adapter模块导入成功")
    except Exception as e:
        print(f"❌ api_adapter模块导入失败: {e}")
        return False
    
    try:
        from src import api_agent
        print("✅ api_agent模块导入成功")
    except Exception as e:
        print(f"⚠️  api_agent模块导入警告: {e}")
    
    try:
        from src import data_fetcher
        print("✅ data_fetcher模块导入成功")
    except Exception as e:
        print(f"⚠️  data_fetcher模块导入警告: {e}")
    
    # 验证策略模块
    strategies = [
        'llm_strategy',
        'rl_trading_strategy',
        'model_comparison_strategy',
        'large_model_strategy',
        'huge_transformer_strategy',
        'data_driven_optimization'
    ]
    
    for strategy in strategies:
        try:
            exec(f"from src.strategies import {strategy}")
            print(f"✅ {strategy}模块导入成功")
        except Exception as e:
            print(f"⚠️  {strategy}模块导入警告: {e}")
    
    return True

def validate_key_functions():
    """验证关键函数是否存在"""
    print_section("2. 验证关键函数")
    
    from src import tiger1 as t1
    
    functions = [
        'check_risk_control',
        'compute_stop_loss',
        'calculate_indicators',
        'get_kline_data',
        'place_tiger_order',
        'judge_market_trend',
        'adjust_grid_interval',
        'grid_trading_strategy',
        'grid_trading_strategy_pro1',
        'boll1m_grid_strategy'
    ]
    
    all_exist = True
    for func_name in functions:
        if hasattr(t1, func_name):
            print(f"✅ {func_name}函数存在")
        else:
            print(f"❌ {func_name}函数不存在")
            all_exist = False
    
    return all_exist

def run_tests():
    """运行测试套件"""
    print_section("3. 运行测试套件")
    
    pytest_test_files = [
        'tests/test_tiger1_strategies.py',
        'tests/test_boll1m_grid.py',
        'tests/test_place_tiger_order.py'
    ]
    
    python_test_files = [
        'tests/test_tiger1_comprehensive.py'
    ]
    
    all_passed = True
    
    # 运行pytest测试
    for test_file in pytest_test_files:
        print(f"\n📝 运行pytest测试: {test_file}")
        result = subprocess.run(
            ['python', '-m', 'pytest', test_file, '-v', '--tb=short'],
            cwd='/home/cx/tigertrade',
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            # 统计通过的测试数
            output = result.stdout
            if 'passed' in output:
                print(f"✅ {test_file} - 测试通过")
            else:
                print(f"⚠️  {test_file} - 测试完成但无明确结果")
        else:
            print(f"❌ {test_file} - 测试失败")
            all_passed = False
    
    # 运行python测试
    for test_file in python_test_files:
        print(f"\n📝 运行python测试: {test_file}")
        result = subprocess.run(
            ['python', test_file],
            cwd='/home/cx/tigertrade',
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"✅ {test_file} - 测试通过")
        else:
            print(f"❌ {test_file} - 测试失败")
            all_passed = False
    
    return all_passed

def validate_directory_structure():
    """验证目录结构"""
    print_section("4. 验证目录结构")
    
    base_dir = '/home/cx/tigertrade'
    expected_dirs = [
        'src',
        'src/strategies',
        'tests',
        'scripts',
        'config',
        'config/openapicfg_com',
        'config/openapicfg_dem',
        'data',
        'docs'
    ]
    
    all_exist = True
    for dir_path in expected_dirs:
        full_path = os.path.join(base_dir, dir_path)
        if os.path.exists(full_path):
            print(f"✅ {dir_path}/ 目录存在")
        else:
            print(f"❌ {dir_path}/ 目录不存在")
            all_exist = False
    
    return all_exist

def main():
    """主函数"""
    print("\n" + "🚀" * 30)
    print("Tigertrade 项目完整性验证")
    print("🚀" * 30)
    
    results = {
        '模块导入': validate_imports(),
        '关键函数': validate_key_functions(),
        '目录结构': validate_directory_structure(),
        '测试套件': run_tests()
    }
    
    print_section("验证总结")
    
    for check_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{check_name}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 所有验证通过！Tigertrade项目已成功整理并可以正常运行。")
        return 0
    else:
        print("\n⚠️  部分验证未通过，请检查上述错误信息。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
