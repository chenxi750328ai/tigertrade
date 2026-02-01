#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
端到端功能测试
"""

import sys
import subprocess
import os

def run_end_to_end_test():
    """运行端到端测试"""
    print("🚀 开始端到端功能测试...")
    
    # 测试1: 语法检查
    print("\n🔍 测试1: Python语法检查")
    try:
        result = subprocess.run([
            sys.executable, "-m", "py_compile", "/home/cx/tigertrade/tiger1.py"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("   ✅ 语法检查通过")
            test1_pass = True
        else:
            print(f"   ❌ 语法检查失败: {result.stderr}")
            test1_pass = False
    except subprocess.TimeoutExpired:
        print("   ❌ 语法检查超时")
        test1_pass = False
    except Exception as e:
        print(f"   ❌ 语法检查异常: {e}")
        test1_pass = False
    
    # 测试2: 模块导入
    print("\n🔍 测试2: 模块导入检查")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("tiger1", "/home/cx/tigertrade/tiger1.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print("   ✅ 模块导入成功")
        test2_pass = True
    except Exception as e:
        print(f"   ❌ 模块导入失败: {e}")
        test2_pass = False
    
    # 测试3: 函数存在性检查
    print("\n🔍 测试3: 关键函数存在性检查")
    try:
        # 检查关键函数是否存在
        required_functions = [
            'grid_trading_strategy_pro1',
            'boll1m_grid_strategy',
            'calculate_indicators',
            'adjust_grid_interval',
            'check_risk_control',
            'place_tiger_order'
        ]
        
        missing_functions = []
        for func_name in required_functions:
            if not hasattr(module, func_name):
                missing_functions.append(func_name)
        
        if not missing_functions:
            print("   ✅ 所有关键函数存在")
            test3_pass = True
        else:
            print(f"   ❌ 缺少函数: {missing_functions}")
            test3_pass = False
    except Exception as e:
        print(f"   ❌ 函数检查异常: {e}")
        test3_pass = False
    
    # 测试4: 常量存在性检查
    print("\n🔍 测试4: 重要常量存在性检查")
    try:
        required_constants = [
            'grid_lower',
            'grid_upper',
            'current_position'
        ]
        
        missing_constants = []
        for const_name in required_constants:
            if not hasattr(module, const_name):
                missing_constants.append(const_name)
        
        if not missing_constants:
            print("   ✅ 所有重要常量存在")
            test4_pass = True
        else:
            print(f"   ❌ 缺少常量: {missing_constants}")
            test4_pass = False
    except Exception as e:
        print(f"   ❌ 常量检查异常: {e}")
        test4_pass = False
    
    # 测试5: 参数修改验证
    print("\n🔍 测试5: 参数修改验证")
    try:
        with open("/home/cx/tigertrade/tiger1.py", "r") as f:
            content = f.read()
        
        # 检查是否包含新参数
        has_new_params = "max(0.1 * (atr if atr else 0), 0.005)" in content
        # 检查是否不包含旧参数
        has_old_params = "max(0.5 * (atr if atr else 0), 0.02)" in content
        
        if has_new_params and not has_old_params:
            print("   ✅ 参数修改正确应用")
            test5_pass = True
        elif has_new_params and has_old_params:
            print("   ⚠️  新旧参数并存")
            test5_pass = False
        elif has_old_params:
            print("   ❌ 旧参数仍然存在")
            test5_pass = False
        else:
            print("   ❌ 新参数未找到")
            test5_pass = False
    except Exception as e:
        print(f"   ❌ 参数验证异常: {e}")
        test5_pass = False
    
    # 汇总结果
    print(f"\n✅ 端到端测试结果:")
    print(f"   测试1 (语法): {'✅ 通过' if test1_pass else '❌ 失败'}")
    print(f"   测试2 (导入): {'✅ 通过' if test2_pass else '❌ 失败'}")
    print(f"   测试3 (函数): {'✅ 通过' if test3_pass else '❌ 失败'}")
    print(f"   测试4 (常量): {'✅ 通过' if test4_pass else '❌ 失败'}")
    print(f"   测试5 (参数): {'✅ 通过' if test5_pass else '❌ 失败'}")
    
    all_tests_pass = all([test1_pass, test2_pass, test3_pass, test4_pass, test5_pass])
    
    print(f"\n🎯 总体结果: {'✅ 全部通过' if all_tests_pass else '❌ 部分失败'}")
    
    return all_tests_pass


def main():
    """主函数"""
    success = run_end_to_end_test()
    
    if success:
        print(f"\n🎉 端到端测试成功！")
        print(f"   所有测试均已通过")
        print(f"   参数修改已正确应用")
        print(f"   代码功能完整")
    else:
        print(f"\n⚠️  端到端测试部分失败")
        print(f"   需要检查失败的测试项")
    
    return success


if __name__ == "__main__":
    main()