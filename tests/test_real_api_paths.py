#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试真实API路径 - 通过模拟quote_client来触发真实API代码路径
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import time

# 添加tigertrade目录到路径
sys.path.insert(0, '/home/cx/tigertrade')

from src import tiger1 as t1
from src.api_adapter import api_manager


class TestRealApiPaths(unittest.TestCase):
    """测试真实API路径"""
    
    def test_real_api_paths_by_monkey_patch(self):
        """通过猴子补丁测试真实API路径"""
        # 保存原始对象
        original_quote_client = getattr(t1, 'quote_client', None)
        
        try:
            # 创建一个模拟的quote_client
            mock_client = type('MockClient', (), {})()
            
            # 添加必要的方法
            def mock_get_future_bars_by_page(identifier, period, begin_time, end_time, total, page_size, time_interval, page_token=None):
                # 返回一个包含DataFrame和token的元组来触发分页逻辑
                df = pd.DataFrame({
                    'time': [datetime.now(), datetime.now() + timedelta(minutes=1)],
                    'open': [90.0, 90.1],
                    'high': [91.0, 91.1],
                    'low': [89.0, 89.1],
                    'close': [90.5, 90.6],
                    'volume': [100, 101],
                    'next_page_token': [None, 'token123']
                })
                return df, 'next_token'
            
            def mock_get_future_bars(symbols, period, begin, end, count, page_token):
                # 返回一个DataFrame来触发非分页逻辑
                return pd.DataFrame({
                    'time': [datetime.now(), datetime.now() + timedelta(minutes=1)],
                    'open': [90.0, 90.1],
                    'high': [91.0, 91.1],
                    'low': [89.0, 89.1],
                    'close': [90.5, 90.6],
                    'volume': [100, 101]
                })
            
            # 将方法附加到模拟客户端
            mock_client.get_future_bars_by_page = mock_get_future_bars_by_page
            mock_client.get_future_bars = mock_get_future_bars
            
            # 通过setattr动态设置t1模块的quote_client
            setattr(t1, 'quote_client', mock_client)
            
            # 现在调用get_kline_data，这将执行真实API路径
            result = t1.get_kline_data(['SIL2603'], t1.BarPeriod.FIVE_MINUTES, count=1001)  # 触发分页逻辑
            result2 = t1.get_kline_data(['SIL2603'], t1.BarPeriod.ONE_MINUTE, count=10)   # 触发非分页逻辑
            
            # 测试其他可能的参数组合
            result3 = t1.get_kline_data(['SIL2603'], 'invalid_period', count=10)  # 触发错误周期路径
            result4 = t1.get_kline_data('SIL2603', t1.BarPeriod.ONE_MINUTE, count=10)  # 字符串符号
            
            # 为了触发异常路径，创建一个会抛出异常的客户端
            error_client = type('ErrorClient', (), {})()
            def error_get_future_bars(*args, **kwargs):
                raise Exception("Simulated API Error")
            error_client.get_future_bars = error_get_future_bars
            error_client.get_future_bars_by_page = error_get_future_bars
            
            setattr(t1, 'quote_client', error_client)
            result_error = t1.get_kline_data(['SIL2603'], t1.BarPeriod.ONE_MINUTE, count=10)
            
            # 测试返回可迭代对象的情况
            iterable_client = type('IterableClient', (), {})()
            
            class MockBar:
                def __init__(self, time, open, high, low, close, volume):
                    self.time = time
                    self.open = open
                    self.high = high
                    self.low = low
                    self.close = close
                    self.volume = volume
            
            def iterable_get_future_bars(*args, **kwargs):
                return [
                    MockBar(datetime.now(), 90.0, 91.0, 89.0, 90.5, 100),
                    MockBar(datetime.now() + timedelta(minutes=1), 90.1, 91.1, 89.1, 90.6, 101)
                ]
            
            iterable_client.get_future_bars = iterable_get_future_bars
            iterable_client.get_future_bars_by_page = iterable_get_future_bars
            
            setattr(t1, 'quote_client', iterable_client)
            result_iterable = t1.get_kline_data(['SIL2603'], t1.BarPeriod.ONE_MINUTE, count=10)
            
        finally:
            # 恢复原始对象
            if original_quote_client is not None:
                setattr(t1, 'quote_client', original_quote_client)
            elif hasattr(t1, 'quote_client'):
                delattr(t1, 'quote_client')
        
        print("✅ test_real_api_paths_by_monkey_patch passed")


def run_test_real_api_paths():
    """运行真实API路径测试"""
    print("🚀 开始运行真实API路径测试...")
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestRealApiPaths)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 统计结果
    total_tests = result.testsRun
    failed_tests = len(result.failures)
    error_tests = len(result.errors)
    passed_tests = total_tests - failed_tests - error_tests
    
    print(f"\n📊 真实API路径测试结果汇总:")
    print(f"   总测试数: {total_tests}")
    print(f"   通过测试: {passed_tests}")
    print(f"   失败测试: {failed_tests}")
    print(f"   错误测试: {error_tests}")
    print(f"   通过率: {passed_tests/total_tests*100:.2f}%")
    
    return result


if __name__ == '__main__':
    result = run_test_real_api_paths()
    
    if result.wasSuccessful():
        print("\n🎉 真实API路径测试全部通过！")
    else:
        print("\n❌ 部分真实API路径测试失败")