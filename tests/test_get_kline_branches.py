#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试get_kline_data的各种分支 - 通过不同的API返回值触发不同代码路径
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import time

# 添加tigertrade目录到路径
from src import tiger1 as t1
from src.api_adapter import api_manager


class TestGetKLineBranches(unittest.TestCase):
    """测试get_kline_data的各种分支"""
    
    @classmethod
    def setUpClass(cls):
        """初始化测试环境"""
        print("🔧 初始化get_kline_data分支测试环境...")
        api_manager.initialize_mock_apis()
        print("✅ 模拟API已初始化")
    
    def test_get_kline_various_branches(self):
        """测试get_kline_data的各种分支"""
        # 循环调用get_kline_data以触发MockQuoteApiAdapter中的各种返回情况
        for i in range(20):  # 多次调用以触发不同的返回值
            # 测试不同参数组合
            result1 = t1.get_kline_data(['SIL2603'], '1min', count=10)
            result2 = t1.get_kline_data(['SIL2603'], '5min', count=50)
            result3 = t1.get_kline_data(['SIL2603'], '1h', count=100)
            result4 = t1.get_kline_data(['SIL2603'], '5min', count=1001)  # 触发分页逻辑
            result5 = t1.get_kline_data('SIL2603', '1min', count=10)  # 字符串符号而非列表
            result6 = t1.get_kline_data(['SIL2603', 'SIL2604'], '1min', count=10)  # 多个符号
            result7 = t1.get_kline_data(['SIL2603'], 'invalid_period', count=10)  # 无效周期

        # 特别测试大数据量以触发分页逻辑
        result_large = t1.get_kline_data(['SIL2603'], '5min', count=2000)
        
        # 测试带时间范围的请求
        start_time = datetime.now() - timedelta(days=1)
        end_time = datetime.now()
        result_with_time = t1.get_kline_data(['SIL2603'], '1min', count=100, start_time=start_time, end_time=end_time)
        
        print("✅ test_get_kline_various_branches passed")


def run_test_get_kline_branches():
    """运行get_kline分支测试"""
    print("🚀 开始运行get_kline分支测试...")
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestGetKLineBranches)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 统计结果
    total_tests = result.testsRun
    failed_tests = len(result.failures)
    error_tests = len(result.errors)
    passed_tests = total_tests - failed_tests - error_tests
    
    print(f"\n📊 get_kline分支测试结果汇总:")
    print(f"   总测试数: {total_tests}")
    print(f"   通过测试: {passed_tests}")
    print(f"   失败测试: {failed_tests}")
    print(f"   错误测试: {error_tests}")
    print(f"   通过率: {passed_tests/total_tests*100:.2f}%")
    
    return result


if __name__ == '__main__':
    result = run_test_get_kline_branches()
    
    if result.wasSuccessful():
        print("\n🎉 get_kline分支测试全部通过！")
    else:
        print("\n❌ 部分get_kline分支测试失败")