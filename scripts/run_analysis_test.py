#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试新添加的详细交易分析功能
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'tigertrade'))

def run_analysis_test():
    """运行详细分析功能测试"""
    print("="*60)
    print("🔍 开始测试详细交易分析功能")
    print("="*60)
    
    try:
        # 导入测试文件并运行测试
        import test_order_tracking
        test_order_tracking.run_tests()
        
        print("\n✅ 详细分析功能测试完成！")
        
    except ImportError as e:
        print(f"❌ 导入测试模块失败: {e}")
        print("💡 请确保依赖库已安装，或者在正确的环境中运行测试")
        
    except Exception as e:
        print(f"❌ 测试执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_analysis_test()