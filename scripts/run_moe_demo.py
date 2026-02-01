#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行MOE策略的包装脚本
统一使用tiger1.py作为总入口，避免重复实现初始化逻辑

用法：
    python scripts/run_moe_demo.py [策略名称] [运行时长]
    
示例：
    python scripts/run_moe_demo.py moe_transformer 20
"""
import sys
import os
import subprocess

# 统一使用tiger1.py作为总入口
# tiger1.py支持命令行参数：python src/tiger1.py [账户类型] [策略类型]
# 账户类型：'d' = DEMO账户, 'c' = 综合账户
# 策略类型：'moe', 'llm', 'grid', 'boll', 'all' 等

def main():
    """主函数 - 调用tiger1.py"""
    # 获取命令行参数
    strategy_name = sys.argv[1] if len(sys.argv) > 1 else 'moe'
    duration_hours = sys.argv[2] if len(sys.argv) > 2 else '20'
    
    print("="*70)
    print(f"🚀 启动MOE策略（通过tiger1.py统一入口）")
    print("="*70)
    print(f"📋 策略: {strategy_name}")
    print(f"⏱️  时长: {duration_hours} 小时")
    print(f"💡 提示: 统一使用 src/tiger1.py 作为总入口")
    print("="*70)
    
    # 设置环境变量（如果需要）
    os.environ['TRADING_STRATEGY'] = strategy_name
    os.environ['RUN_DURATION_HOURS'] = str(duration_hours)
    
    # 调用tiger1.py
    # 参数：'d' = DEMO账户, strategy_name = 策略类型
    cmd = [
        sys.executable,
        'src/tiger1.py',
        'd',  # DEMO账户
        strategy_name  # 策略类型
    ]
    
    print(f"\n执行命令: {' '.join(cmd)}\n")
    
    try:
        # 运行tiger1.py
        result = subprocess.run(cmd, cwd='/home/cx/tigertrade')
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n🛑 用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
