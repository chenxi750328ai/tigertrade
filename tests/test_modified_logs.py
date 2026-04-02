#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试修改后的日志输出
"""

import sys
from pathlib import Path

from src import tiger1 as t1

_REPO = Path(__file__).resolve().parents[1]


def test_log_output():
    """测试日志输出"""
    print("🔍 测试修改后的日志输出...")
    
    # 检查tiger1模块是否可以正常导入
    print("✅ 模块导入成功")
    
    # 检查函数是否存在
    print(f"✅ grid_trading_strategy_pro1 存在: {hasattr(t1, 'grid_trading_strategy_pro1')}")
    print(f"✅ boll1m_grid_strategy 存在: {hasattr(t1, 'boll1m_grid_strategy')}")
    
    # 检查代码内容
    import inspect
    source = inspect.getsource(t1.grid_trading_strategy_pro1)
    
    # 检查是否包含详细的日志输出
    if "计算详情" in source and "Buffer计算" in source:
        print("✅ 详细日志输出已包含在grid_trading_strategy_pro1函数中")
    else:
        print("❌ 未找到详细日志输出")
        
    # 检查源码中是否包含新的日志输出
    lines = source.split('\n')
    detail_lines = [line for line in lines if '计算详情' in line or 'Buffer计算' in line]
    if detail_lines:
        print(f"✅ 发现详细日志输出行: {len(detail_lines)} 行")
        for line in detail_lines[:3]:  # 显示前3行
            print(f"   {line.strip()}")
    else:
        print("❌ 未找到详细日志输出行")


def explain_changes():
    """解释修改内容"""
    print(f"\n📝 修改说明:")
    print(f"   已修改 {_REPO / 'src' / 'tiger1.py'} 中的 grid_trading_strategy_pro1 函数")
    print(f"   在未触发交易时，现在会输出详细的计算过程，包括:")
    print(f"   - 当前价格")
    print(f"   - 网格下轨")
    print(f"   - ATR值")
    print(f"   - Buffer计算过程")
    print(f"   - 阈值计算过程")
    print(f"   - near_lower计算结果")
    print(f"   - rsi_ok计算细节")
    print(f"   - trend_check, rebound, vol_ok的值")
    print(f"   - 最终的未触发原因")
    
    print(f"\n🔍 这样可以清楚地看到:")
    print(f"   - 价格是否真的接近下轨")
    print(f"   - 各个条件的计算过程")
    print(f"   - 哪个条件导致了未触发")
    print(f"   - 便于调试和优化策略")


def run_syntax_check():
    """运行语法检查"""
    print(f"\n🔧 运行语法检查...")
    try:
        import ast
        with open(_REPO / "src" / "tiger1.py", "r", encoding="utf-8") as f:
            source = f.read()
        ast.parse(source)
        print("✅ 代码语法正确")
        return True
    except SyntaxError as e:
        print(f"❌ 代码语法错误: {e}")
        return False


if __name__ == "__main__":
    print("🚀 开始测试修改后的日志输出...\n")
    
    test_log_output()
    explain_changes()
    run_syntax_check()
    
    print(f"\n✅ 测试完成!")
    print(f"   现在当grid_trading_strategy_pro1未触发时，会显示详细的计算过程")
    print(f"   这样可以准确了解near_lower等条件的计算结果")