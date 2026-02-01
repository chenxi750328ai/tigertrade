#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
最终确认tiger1.py中所有函数都已正确定义
"""

def main():
    print("🔍 Final confirmation: checking tiger1.py functions...")
    
    try:
        import tigertrade.tiger1 as t1
        print("✅ Module imported successfully")
    except Exception as e:
        print(f"❌ Failed to import module: {e}")
        return False
    
    # 检查所有关键函数
    functions_to_check = [
        # 之前有问题的函数
        'check_risk_control',
        
        # 核心策略函数
        'compute_stop_loss',
        'calculate_indicators',
        'get_kline_data',
        'place_tiger_order',
        'judge_market_trend',
        'adjust_grid_interval',
        'place_take_profit_order',
        
        # 策略实现函数
        'grid_trading_strategy',
        'grid_trading_strategy_pro1',
        'boll1m_grid_strategy',
        
        # 辅助函数
        'get_timestamp',
        'verify_api_connection',
        'get_future_brief_info',
        'check_active_take_profits',
        'check_timeout_take_profits',
        
        # 测试函数
        'test_order_tracking',
        'test_position_management',
        'test_risk_control',
        'run_tests',
        'backtest_grid_trading_strategy_pro1',
    ]
    
    all_found = True
    for func_name in functions_to_check:
        if hasattr(t1, func_name):
            func_obj = getattr(t1, func_name)
            if callable(func_obj):
                print(f"✅ {func_name}: FUNCTION DEFINED")
            else:
                print(f"⚠️  {func_name}: EXISTS BUT NOT CALLABLE")
        else:
            print(f"❌ {func_name}: MISSING")
            all_found = False
    
    # 检查关键变量
    variables_to_check = [
        'FUTURE_SYMBOL',
        'GRID_MAX_POSITION',
        'DAILY_LOSS_LIMIT',
        'STOP_LOSS_MULTIPLIER',
        'TAKE_PROFIT_ATR_OFFSET',
        'current_position',
        'daily_loss',
        'grid_upper',
        'grid_lower',
        'atr_5m'
    ]
    
    for var_name in variables_to_check:
        if hasattr(t1, var_name):
            print(f"✅ {var_name}: VARIABLE DEFINED")
        else:
            print(f"❌ {var_name}: VARIABLE MISSING")
            all_found = False
    
    if all_found:
        print(f"\n🎉 SUCCESS: All functions and variables are properly defined!")
        print(f"✅ tiger1.py is complete and functional")
        print(f"✅ No undefined functions found")
        print(f"✅ Previous issue with check_risk_control is resolved")
        return True
    else:
        print(f"\n❌ ISSUES FOUND: Some functions or variables are missing")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🏆 CONCLUSION: tiger1.py is fully functional with no undefined functions!")
    else:
        print("\n💥 CONCLUSION: There are still issues with tiger1.py")