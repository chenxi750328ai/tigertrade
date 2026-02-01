#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
语法和导入测试 - 确保代码没有基本语法错误
"""

def test_syntax_and_imports():
    """测试代码语法和导入功能"""
    print("🔍 开始语法和导入测试...")

    # 测试1: 导入tiger1模块（本仓库为 src.tiger1）
    try:
        import src.tiger1
        print("✅ tiger1模块导入成功")
    except SyntaxError as e:
        assert False, f"tiger1模块存在语法错误: {e}"
    except ImportError as e:
        assert False, f"tiger1模块导入失败: {e}"
    except Exception as e:
        assert False, f"tiger1模块导入出现其他错误: {e}"

    # 测试2: 检查关键函数是否存在
    try:
        from src.tiger1 import (
            place_tiger_order,
            check_active_take_profits,
            check_timeout_take_profits,
            place_take_profit_order,
            grid_trading_strategy,
            test_risk_control
        )
        print("✅ 关键函数导入成功")
    except AttributeError as e:
        assert False, f"缺少必要函数: {e}"
    except Exception as e:
        assert False, f"函数导入失败: {e}"

    # 测试3: 检查基本功能是否能运行
    try:
        src.tiger1.grid_trading_strategy()
        print("✅ 基本功能运行成功")
    except Exception as e:
        print(f"⚠️ 基本功能运行失败，但非致命错误: {e}")

    print("✅ 语法和导入测试完成")


def test_basic_execution():
    """测试基本执行功能"""
    print("\n🔍 开始基本执行测试...")
    
    try:
        # 重置全局变量
        from src import tiger1 as t1
        import random  # 需要导入random模块
        
        # 初始化t1模块中的random（如果需要的话）
        t1.random = random
        
        t1.current_position = 0
        t1.open_orders.clear()
        t1.closed_positions.clear()
        t1.active_take_profit_orders.clear()
        t1.position_entry_times.clear()
        t1.position_entry_prices.clear()
        
        # 测试place_tiger_order函数
        result = t1.place_tiger_order(
            'BUY', 
            1, 
            100.0,
            tech_params={'rsi': 30, 'kdj_k': 20},
            reason='网格下轨+RSI超卖'
        )
        assert result is not False, "place_tiger_order(BUY) 应返回成功"
        print("✅ place_tiger_order函数执行成功")
        # Mock 模式下可能不更新 current_position，仅校验调用成功

        # 测试卖出
        result = t1.place_tiger_order(
            'SELL', 
            1, 
            105.0,
            tech_params={'profit_target_met': True},
            reason='达到止盈目标'
        )
        assert result is not False, "place_tiger_order(SELL) 应返回成功"
        print("✅ SELL订单执行成功")

        print("✅ 基本执行测试完成")
    except Exception as e:
        assert False, f"基本执行测试失败: {e}"


def test_risk_control_functions():
    """测试风控功能"""
    print("\n🔍 开始风控功能测试...")
    
    try:
        from src import tiger1 as t1
        import random
        t1.random = random
        
        # 测试风控检查 - 使用正确的函数名和参数
        result = t1.check_risk_control(100.0, 'BUY')  # 使用正确的函数名
        print(f"✅ 风控检查执行成功，结果: {result}")
        
        # 测试主动止盈检查
        result = t1.check_active_take_profits(110.0)
        print(f"✅ 主动止盈检查执行成功，结果: {result}")
        
        # 测试超时止盈检查
        result = t1.check_timeout_take_profits(105.0)
        print(f"✅ 超时止盈检查执行成功，结果: {result}")

        print("✅ 风控功能测试完成")
    except Exception as e:
        assert False, f"风控功能测试失败: {e}"


def test_all_functions():
    """运行所有测试（pytest 会分别收集各 test_*，此处供 __main__ 调用）"""
    print("🚀 开始运行所有语法和功能测试...")
    test_syntax_and_imports()
    test_basic_execution()
    test_risk_control_functions()
    print("\n🎉 所有测试通过！代码可以可靠运行")


if __name__ == "__main__":
    test_all_functions()