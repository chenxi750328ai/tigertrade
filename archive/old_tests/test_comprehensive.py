#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
期货网格交易策略综合功能验证测试
"""

import sys
import os
import inspect

# 添加项目路径
sys.path.insert(0, '/home/cx/tigertrade1')

def test_module_import():
    """测试模块导入"""
    try:
        import tiger2
        print("✅ tiger2模块导入成功")
        return tiger2
    except ImportError as e:
        print(f"❌ Tiger2模块导入失败: {e}")
        return None


def test_function_docstrings(module):
    """测试函数文档字符串"""
    print("\n=== 函数文档字符串检查 ===")
    
    functions_to_check = [
        'get_kline_data',
        'calculate_indicators', 
        'judge_market_trend',
        'adjust_grid_interval',
        'check_active_take_profits',
        'compute_stop_loss',
        'check_risk_control',
        'place_tiger_order',
        'place_take_profit_order',
        'grid_trading_strategy',
        'grid_trading_strategy_pro1',
        'boll1m_grid_strategy',
        'backtest_grid_trading_strategy_pro1'
    ]
    
    for func_name in functions_to_check:
        if hasattr(module, func_name):
            func = getattr(module, func_name)
            if callable(func) and func.__doc__:
                print(f"✅ {func_name}: 已找到文档字符串")
            else:
                print(f"⚠️ {func_name}: 文档字符串缺失")
        else:
            print(f"❌ {func_name}: 函数未定义")


def test_constants(module):
    """测试常量定义"""
    print("\n=== 常量定义检查 ===")
    
    constants_to_check = [
        'FUTURE_SYMBOL',
        'GRID_MAX_POSITION',
        'GRID_ATR_PERIOD',
        'GRID_BOLL_PERIOD',
        'DAILY_LOSS_LIMIT',
        'SINGLE_TRADE_LOSS',
        'MIN_KLINES',
        'TAKE_PROFIT_TIMEOUT'
    ]
    
    for const in constants_to_check:
        if hasattr(module, const):
            value = getattr(module, const)
            print(f"✅ {const}: {value}")
        else:
            print(f"❌ {const}: 未定义")


def test_strategy_components(module):
    """测试策略组件"""
    print("\n=== 策略组件检查 ===")
    
    # 全局变量
    global_vars = [
        'current_position',
        'daily_loss',
        'grid_upper',
        'grid_lower',
        'atr_5m',
        'position_entry_times',
        'position_entry_prices',
        'active_take_profit_orders'
    ]
    
    for var in global_vars:
        if hasattr(module, var):
            print(f"✅ 全局变量 {var}: 存在")
        else:
            print(f"❌ 全局变量 {var}: 不存在")


def test_code_quality(module):
    """测试代码质量指标"""
    print("\n=== 代码质量检查 ===")
    
    # 检查函数长度和复杂度
    functions_to_analyze = [
        'grid_trading_strategy',
        'grid_trading_strategy_pro1',
        'boll1m_grid_strategy',
        'calculate_indicators',
        'get_kline_data'
    ]
    
    for func_name in functions_to_analyze:
        if hasattr(module, func_name):
            func = getattr(module, func_name)
            source_lines = inspect.getsource(func).split('\n')
            line_count = len(source_lines)
            
            # 检查是否有足够的注释
            comment_lines = sum(1 for line in source_lines if '#' in line and not line.strip().startswith('#'))
            has_good_comments = comment_lines / max(line_count, 1) > 0.1  # 至少10%的行是注释
            
            print(f"✅ {func_name}: {line_count}行代码, 注释比例{'良好' if has_good_comments else '待改进'}")


def test_implementation_details(module):
    """测试实现细节"""
    print("\n=== 实现细节检查 ===")
    
    # 检查关键算法是否存在
    source_code = inspect.getsource(module)
    
    algorithms = [
        'talib.MA',
        'talib.BBANDS', 
        'talib.ATR',
        'talib.RSI',
        'STOP_LOSS_MULTIPLIER',
        'TAKE_PROFIT_ATR_OFFSET'
    ]
    
    for algo in algorithms:
        if algo in source_code:
            print(f"✅ {algo}: 已实现")
        else:
            print(f"⚠️ {algo}: 未找到")
    
    # 风控检查
    risk_controls = [
        'check_risk_control',
        'SINGLE_TRADE_LOSS',
        'DAILY_LOSS_LIMIT',
        'GRID_MAX_POSITION'
    ]
    
    for rc in risk_controls:
        if rc in source_code:
            print(f"✅ 风控组件 {rc}: 已实现")
        else:
            print(f"⚠️ 风控组件 {rc}: 未找到")


def main():
    """主函数"""
    print("🔍 期货网格交易策略综合功能验证")
    print("="*50)
    
    # 测试模块导入
    module = test_module_import()
    if not module:
        return
    
    # 测试各项功能
    test_function_docstrings(module)
    test_constants(module)
    test_strategy_components(module)
    test_code_quality(module)
    test_implementation_details(module)
    
    print("\n" + "="*50)
    print("✅ 综合测试完成")


if __name__ == "__main__":
    main()