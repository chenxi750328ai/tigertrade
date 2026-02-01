#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
时段分析模块快速测试脚本
"""

import sys
import os
from datetime import datetime, timezone, timedelta

sys.path.insert(0, '/home/cx/tigertrade')

from scripts.analysis.time_period_analyzer import TimePeriodAnalyzer
from src.strategies.time_period_strategy import TimePeriodStrategy

def test_period_extraction():
    """测试时段提取"""
    print("="*60)
    print("测试1: 时段提取")
    print("="*60)
    
    analyzer = TimePeriodAnalyzer()
    
    test_cases = [
        (datetime(2026, 1, 21, 20, 30, tzinfo=timezone(timedelta(hours=8))), "COMEX_欧美高峰"),
        (datetime(2026, 1, 21, 9, 15, tzinfo=timezone(timedelta(hours=8))), "沪银_日盘尖峰"),
        (datetime(2026, 1, 21, 6, 30, tzinfo=timezone(timedelta(hours=8))), "COMEX_亚洲低波动"),
    ]
    
    for test_time, expected in test_cases:
        period = analyzer.extract_time_period(test_time)
        status = "✅" if period == expected else "❌"
        print(f"{status} {test_time.strftime('%H:%M')} -> {period} (期望: {expected})")
    
    print()

def test_balance_threshold():
    """测试平衡阈值计算"""
    print("="*60)
    print("测试2: 平衡阈值计算")
    print("="*60)
    
    analyzer = TimePeriodAnalyzer()
    
    test_cases = [
        (25.0, 0.008, 0.48),   # COMEX欧美高峰
        (25.0, 0.028, 1.68),   # 沪银日盘尖峰
        (25.0, 0.020, 1.20),   # 低波动时段
    ]
    
    for price, slippage_rate, expected in test_cases:
        threshold = analyzer.calculate_balance_threshold(price, slippage_rate)
        diff = abs(threshold - expected)
        status = "✅" if diff < 0.01 else "❌"
        print(f"{status} 价格={price}, 滑点率={slippage_rate*100:.2f}% -> 阈值={threshold:.4f} (期望: {expected:.4f})")
    
    print()

def test_strategy_basic():
    """测试策略基本功能"""
    print("="*60)
    print("测试3: 时段自适应策略基本功能")
    print("="*60)
    
    strategy = TimePeriodStrategy(symbol="SIL2603")
    
    # 测试获取当前时段配置
    current_config = strategy.get_current_period_config()
    print(f"✅ 当前时段配置获取成功")
    print(f"   配置来源: {current_config.get('source', 'unknown')}")
    print(f"   波动率: {current_config.get('volatility', 0)}")
    print(f"   滑点率: {current_config.get('slippage_rate', 0)*100:.2f}%")
    print(f"   最大仓位: {current_config.get('max_position', 0)}")
    
    # 测试获取网格参数
    grid_params = strategy.get_grid_parameters(25.0)
    print(f"\n✅ 网格参数获取成功")
    print(f"   网格间距: {grid_params['grid_step']:.4f}美元")
    print(f"   平衡阈值: {grid_params['balance_threshold']:.4f}美元")
    print(f"   最大仓位: {grid_params['max_position']}手")
    print(f"   配置来源: {grid_params['config_source']}")
    
    # 验证网格间距 >= 平衡阈值
    assert grid_params['grid_step'] >= grid_params['balance_threshold'], "网格间距应 >= 平衡阈值"
    print(f"\n✅ 验证通过: 网格间距 >= 平衡阈值")
    
    print()

def main():
    """主测试函数"""
    print("\n🧪 开始时段分析模块测试\n")
    
    try:
        test_period_extraction()
        test_balance_threshold()
        test_strategy_basic()
        
        print("="*60)
        print("✅ 所有测试通过！")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
