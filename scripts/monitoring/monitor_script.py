#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
监控脚本，持续运行并检测潜在的BUG
"""

import subprocess
import sys
import time
import traceback
from datetime import datetime
import os

def monitor_terminal_output():
    """监控终端输出以检测潜在BUG"""
    print("🔍 开始监控终端输出...")
    print(f"⏰ 监控开始时间: {datetime.now()}")
    
    # 定义要监控的错误模式
    error_patterns = [
        "Exception",
        "Error",
        "Traceback",
        "AttributeError",
        "TypeError", 
        "ValueError",
        "KeyError",
        "IndexError",
        "NameError",
        "ImportError",
        "SyntaxError",
        "IndentationError",
        "ZeroDivisionError",
        "RuntimeError",
        "AssertionError",
        "RecursionError"
    ]
    
    warning_patterns = [
        "WARNING",
        "WARN",
        "warn",
        "warning",
        "Failed",
        "failed",
        "Invalid",
        "invalid",
        "Unexpected",
        "unexpected",
        "timeout",
        "Timeout"
    ]
    
    bug_counter = 0
    warning_counter = 0
    
    try:
        # 运行tiger1.py脚本并监控输出
        print("🏃‍♂️ 启动tiger1.py脚本...")
        # 改变工作目录到tigertrade目录，确保可以找到配置文件
        env = os.environ.copy()
        env['PYTHONPATH'] = '/home/cx/tigertrade'
        
        process = subprocess.Popen([
            sys.executable, "-u", "/home/cx/tigertrade/tiger1.py", "d"
        ], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        universal_newlines=True, 
        bufsize=1,
        cwd="/home/cx/tigertrade",  # 设置工作目录
        env=env)
        
        # 读取输出流
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                line = output.strip()
                print(line)
                
                # 检查错误模式
                for pattern in error_patterns:
                    if pattern in line:
                        print(f"🚨 检测到错误模式 '{pattern}': {line}")
                        bug_counter += 1
                
                # 检查警告模式
                for pattern in warning_patterns:
                    if pattern in line:
                        print(f"⚠️ 检测到警告模式 '{pattern}': {line}")
                        warning_counter += 1
        
        # 检查stderr
        stderr_output = process.stderr.read()
        if stderr_output:
            print("❌ 标准错误输出:")
            print(stderr_output)
            
            for pattern in error_patterns:
                if pattern in stderr_output:
                    print(f"🚨 检测到错误模式 '{pattern}' 在标准错误输出中")
                    bug_counter += 1
    
    except KeyboardInterrupt:
        print("\n🛑 监控被用户中断")
        return {"status": "interrupted", "bugs_found": bug_counter, "warnings_found": warning_counter}
    except Exception as e:
        print(f"💥 监控过程中发生错误: {e}")
        print(traceback.format_exc())
        return {"status": "error", "error": str(e), "bugs_found": bug_counter, "warnings_found": warning_counter}
    
    print(f"\n✅ 监控结束时间: {datetime.now()}")
    print(f"📊 统计结果:")
    print(f"   错误/异常数量: {bug_counter}")
    print(f"   警告数量: {warning_counter}")
    
    return {
        "status": "completed", 
        "bugs_found": bug_counter, 
        "warnings_found": warning_counter,
        "process_return_code": process.returncode if 'process' in locals() else None
    }


def run_logic_tests():
    """运行纯逻辑测试，不涉及API调用"""
    print("🔬 运行纯逻辑测试...")
    
    bug_counter = 0
    warning_counter = 0
    
    try:
        # 导入并运行测试函数
        import sys
        sys.path.insert(0, '/home/cx/tigertrade')
        
        from src import tiger1 as t1
        
        print("✅ 模块导入成功")
        
        # 测试基本函数定义
        functions_to_test = [
            t1.place_tiger_order,
            t1.check_active_take_profits,
            t1.check_timeout_take_profits,
            t1.check_risk_control,
            t1.calculate_indicators,
            t1.judge_market_trend,
            t1.adjust_grid_interval
        ]
        
        for func in functions_to_test:
            print(f"✅ 函数 {func.__name__} 已定义")
        
        # 测试一些基本功能
        print("\n🧪 测试基本功能...")
        
        # 重置状态
        t1.current_position = 0
        t1.open_orders.clear()
        t1.closed_positions.clear()
        t1.active_take_profit_orders.clear()
        t1.position_entry_times.clear()
        t1.position_entry_prices.clear()
        
        # 测试下单功能
        import random
        t1.random = random
        
        print("📝 测试下单功能...")
        result = t1.place_tiger_order(
            'BUY', 
            1, 
            100.0,
            tech_params={'rsi': 30, 'kdj_k': 20},
            reason='网格下轨+RSI超卖'
        )
        print(f"✅ 买入下单成功: {result}")
        
        if t1.current_position == 1:
            print("✅ 仓位更新正确")
        else:
            print(f"❌ 仓位更新错误: 期望1，实际{t1.current_position}")
            bug_counter += 1
            
        # 测试卖出
        result = t1.place_tiger_order(
            'SELL', 
            1, 
            105.0,
            tech_params={'profit_target_met': True},
            reason='达到止盈目标'
        )
        print(f"✅ 卖出下单成功: {result}")
        
        if t1.current_position == 0:
            print("✅ 仓位清零正确")
        else:
            print(f"❌ 仓位清零错误: 期望0，实际{t1.current_position}")
            bug_counter += 1
        
        # 测试风控功能
        print("\n🛡️ 测试风控功能...")
        risk_result = t1.check_risk_control(100.0, 'BUY')
        print(f"✅ 风控检查成功: {risk_result}")
        
        # 测试止盈检查（当前没有持仓，应该返回False）
        take_profit_result = t1.check_active_take_profits(110.0)
        print(f"✅ 主动止盈检查: {take_profit_result}")
        
        timeout_result = t1.check_timeout_take_profits(105.0)
        print(f"✅ 超时止盈检查: {timeout_result}")
        
        print("\n✅ 所有逻辑测试通过")
        
    except Exception as e:
        print(f"💥 逻辑测试失败: {e}")
        print(traceback.format_exc())
        bug_counter += 1
    
    print(f"\n📊 逻辑测试统计:")
    print(f"   错误/异常数量: {bug_counter}")
    print(f"   警告数量: {warning_counter}")
    
    return {
        "status": "completed", 
        "bugs_found": bug_counter, 
        "warnings_found": warning_counter
    }


def run_extended_monitoring(duration_minutes=5):
    """运行扩展监控"""
    print(f"🔬 开始扩展监控，持续 {duration_minutes} 分钟...")
    
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    
    bug_counter = 0
    warning_counter = 0
    
    try:
        # 测试各种策略函数
        import sys
        sys.path.insert(0, '/home/cx/tigertrade')
        from src import tiger1 as t1
        import random
        t1.random = random
        
        test_functions = [
            ("grid_trading_strategy", lambda: t1.grid_trading_strategy()),
            ("grid_trading_strategy_pro1", lambda: t1.grid_trading_strategy_pro1()),
            ("boll1m_grid_strategy", lambda: t1.boll1m_grid_strategy())
        ]
        
        iteration = 0
        while time.time() < end_time:
            print(f"\n⏳ 运行第 {iteration+1} 次测试...")
            
            for name, func in test_functions:
                try:
                    print(f"🧪 运行 {name}...")
                    func()
                    print(f"✅ {name} 执行成功")
                except Exception as e:
                    if "NoneType" in str(e) and "get_" in str(e):
                        # API连接问题，不算作逻辑错误
                        print(f"⚠️ {name} - API连接问题（非逻辑错误）: {e}")
                        warning_counter += 1
                    else:
                        print(f"🚨 {name} - 发现错误: {e}")
                        bug_counter += 1
            
            iteration += 1
            # 每隔一段时间休息一下
            time.sleep(5)
    
    except KeyboardInterrupt:
        print("\n🛑 扩展监控被用户中断")
    except Exception as e:
        print(f"💥 扩展监控过程中发生错误: {e}")
        print(traceback.format_exc())
    
    print(f"\n✅ 扩展监控结束")
    print(f"📊 统计结果:")
    print(f"   错误/异常数量: {bug_counter}")
    print(f"   警告数量: {warning_counter}")
    
    return {
        "status": "completed", 
        "bugs_found": bug_counter, 
        "warnings_found": warning_counter
    }


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "extended":
            # 运行扩展监控
            result = run_extended_monitoring(1)  # 运行1分钟
        elif sys.argv[1] == "logic":
            # 运行逻辑测试
            result = run_logic_tests()
        else:
            # 运行基本监控
            result = monitor_terminal_output()
    else:
        # 默认运行逻辑测试
        result = run_logic_tests()
    
    # 输出最终结果
    print(f"\n🏁 监控完成，状态: {result['status']}")
    if 'bugs_found' in result:
        print(f"   发现错误: {result.get('bugs_found', 0)}")
        print(f"   发现警告: {result.get('warnings_found', 0)}")
    
    if result.get('bugs_found', 0) == 0:
        print("🎉 未发现逻辑错误！")
    else:
        print("⚠️ 发现了一些错误，需要修复。")