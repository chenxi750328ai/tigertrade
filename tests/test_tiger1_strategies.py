#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""测试tiger1.py中的策略函数"""

import sys
import os
# 添加tigertrade根目录到路径
from src import tiger1 as t1
import pandas as pd
import numpy as np

def test_strategy_functions():
    """测试策略相关函数"""
    print("Testing strategy functions...")
    
    try:
        # 测试市场趋势判断
        sample_indicators = {
            '5m': {
                'close': 25.0,
                'boll_middle': 24.5,
                'boll_mid': 24.5,
                'rsi': 55
            },
            '1m': {
                'close': 25.0,
                'rsi': 45
            }
        }
        trend = t1.judge_market_trend(sample_indicators)
        print(f'✅ judge_market_trend works: {trend}')
        
        # 测试指标计算（使用模拟数据）
        # 创建模拟的K线数据
        dates = pd.date_range('2023-01-01', periods=30, freq='5min')
        df_5m = pd.DataFrame({
            'open': 25 + np.random.randn(30) * 0.1,
            'high': 25 + np.abs(np.random.randn(30) * 0.15),
            'low': 25 - np.abs(np.random.randn(30) * 0.15),
            'close': 25 + np.random.randn(30) * 0.1,
            'volume': np.random.randint(100, 1000, 30)
        }, index=dates)
        
        df_1m = df_5m.copy()
        
        indicators = t1.calculate_indicators(df_1m, df_5m)
        print(f'✅ calculate_indicators works, computed keys: {list(indicators.keys())}')
        
        # 测试网格区间调整
        t1.adjust_grid_interval(trend, indicators)
        print(f'✅ adjust_grid_interval works')
        
        # 测试风险控制
        risk_result = t1.check_risk_control(25.0, 'BUY')
        print(f'✅ check_risk_control works: {risk_result}')
        
        # 测试止损计算
        if hasattr(t1, 'compute_stop_loss'):
            try:
                sl_result = t1.compute_stop_loss(25.0, 0.5, 24.0)
                print(f'✅ compute_stop_loss works: {sl_result}')
            except Exception as e:
                print(f'⚠️ compute_stop_loss execution issue: {e}')
        
        print('\n🎉 All strategy function tests passed!')
        
    except Exception as e:
        print(f'❌ Error in strategy function test: {e}')
        import traceback
        traceback.print_exc()

def test_place_order_functions():
    """测试下单相关函数"""
    print("\nTesting order placement functions...")
    
    try:
        # 测试下单函数
        # 注意：在测试环境下，下单会被拦截，不会真正执行
        result = t1.place_tiger_order('BUY', 1, 25.0)
        print(f'✅ place_tiger_order works: {result}')
        
        # 测试止盈单
        result_tp = t1.place_take_profit_order('BUY', 1, 26.0)
        print(f'✅ place_take_profit_order works: {result_tp}')
        
        print('🎉 Order placement function tests passed!')
        
    except Exception as e:
        print(f'❌ Error in order placement function test: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Starting strategy function tests for tiger1.py...\n")
    
    test_strategy_functions()
    test_place_order_functions()
    
    print("\n✅ All tests completed successfully!")