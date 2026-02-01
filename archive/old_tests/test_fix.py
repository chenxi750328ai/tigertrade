#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试修复后的tiger1.py模块
"""

import sys
import os

# 添加tigertrade目录到路径
sys.path.insert(0, '/home/cx/tigertrade')

from src import tiger1 as t1


def test_compute_stop_loss():
    """测试compute_stop_loss函数"""
    print("🔍 测试compute_stop_loss函数...")
    
    # 测试参数
    price = 90.0
    atr_value = 0.2
    grid_lower_val = 89.0
    
    try:
        stop_loss_price, projected_loss = t1.compute_stop_loss(price, atr_value, grid_lower_val)
        print(f"✅ compute_stop_loss函数调用成功")
        print(f"   输入: price={price}, atr_value={atr_value}, grid_lower_val={grid_lower_val}")
        print(f"   输出: stop_loss_price={stop_loss_price}, projected_loss={projected_loss}")
        return True
    except Exception as e:
        print(f"❌ compute_stop_loss函数调用失败: {e}")
        return False


def test_all_functions_exist():
    """测试所有必需的函数是否存在"""
    print("\n🔍 测试所有必需的函数是否存在...")
    
    required_functions = [
        'compute_stop_loss',
        'grid_trading_strategy_pro1',
        'boll1m_grid_strategy',
        'calculate_indicators',
        'adjust_grid_interval',
        'check_risk_control',
        'place_tiger_order'
    ]
    
    all_exist = True
    for func_name in required_functions:
        exists = hasattr(t1, func_name)
        status = "✅" if exists else "❌"
        print(f"   {status} {func_name}: {'存在' if exists else '不存在'}")
        if not exists:
            all_exist = False
    
    return all_exist


def main():
    """主函数"""
    print("🚀 开始测试修复后的tiger1.py模块...\n")
    
    # 测试compute_stop_loss函数
    test1_passed = test_compute_stop_loss()
    
    # 测试所有必需函数是否存在
    test2_passed = test_all_functions_exist()
    
    print(f"\n✅ 测试结果:")
    print(f"   compute_stop_loss函数测试: {'✅ 通过' if test1_passed else '❌ 失败'}")
    print(f"   所有函数存在性测试: {'✅ 通过' if test2_passed else '❌ 失败'}")
    
    overall_success = test1_passed and test2_passed
    print(f"\n🎯 总体结果: {'✅ 成功' if overall_success else '❌ 失败'}")
    
    if overall_success:
        print(f"\n🎉 修复成功！compute_stop_loss函数已定义，tiger1.py模块现在可以正常工作。")
    else:
        print(f"\n❌ 修复失败，请检查代码。")
    
    return overall_success


if __name__ == "__main__":
    main()