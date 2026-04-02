#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""使用Monkey Patch模拟quote_client来测试真实API路径"""

import unittest
import sys
import os
import pandas as pd
from unittest.mock import Mock, MagicMock
import datetime
from zoneinfo import ZoneInfo

# 添加当前目录到路径
from src import tiger1 as t1
from src.api_adapter import MockQuoteApiAdapter, ApiAdapterManager


class TestWithQuoteClient(unittest.TestCase):
    """使用Monkey Patch模拟quote_client的测试"""
    
    def setUp(self):
        """设置测试环境"""
        # 确保在模拟模式下
        self.original_is_mock_mode = t1.api_manager.is_mock_mode
        t1.api_manager.is_mock_mode = False  # 临时切换到非模拟模式
    
    def tearDown(self):
        """恢复原始设置"""
        t1.api_manager.is_mock_mode = self.original_is_mock_mode
    
    def test_with_mock_quote_client(self):
        """使用Mock quote_client测试真实API路径"""
        # 创建模拟的quote_client
        mock_quote_client = Mock()
        
        # 模拟get_future_bars方法返回模拟数据
        mock_klines = pd.DataFrame({
            'time': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
            'open': [90.0, 91.0, 92.0],
            'high': [91.0, 92.0, 93.0],
            'low': [89.0, 90.0, 91.0],
            'close': [90.5, 91.5, 92.5],
            'volume': [100, 150, 200]
        })
        
        mock_quote_client.get_future_bars.return_value = mock_klines
        mock_quote_client.get_future_bars_by_page = None  # 确保不使用分页
        
        # 临时替换tiger1中的quote_client
        original_quote_client = getattr(t1, 'quote_client', None)
        t1.quote_client = mock_quote_client
        
        try:
            # 测试获取K线数据，这将执行真实API路径
            result = t1.get_kline_data("SIL2603", "1min", count=10)
            self.assertIsInstance(result, pd.DataFrame)
            print("✅ test_with_mock_quote_client passed")
        finally:
            # 恢复原始值
            if original_quote_client is not None:
                t1.quote_client = original_quote_client
            else:
                if hasattr(t1, 'quote_client'):
                    delattr(t1, 'quote_client')
    
    def test_with_paging_client(self):
        """测试分页API路径"""
        # 创建支持分页的模拟客户端
        mock_quote_client = Mock()
        
        # 模拟分页API
        mock_df = pd.DataFrame({
            'time': pd.to_datetime(['2023-01-01', '2023-01-02']),
            'open': [90.0, 91.0],
            'high': [91.0, 92.0],
            'low': [89.0, 90.0],
            'close': [90.5, 91.5],
            'volume': [100, 150]
        })
        
        # 模拟分页返回元组 (DataFrame, next_token)
        mock_quote_client.get_future_bars_by_page.return_value = (mock_df, None)
        mock_quote_client.get_future_bars.return_value = mock_df
        mock_quote_client.get_future_bars_by_page = Mock()  # 表示支持分页
        mock_quote_client.get_future_bars_by_page.__bool__ = Mock(return_value=True)
        
        # 临时替换tiger1中的quote_client
        original_quote_client = getattr(t1, 'quote_client', None)
        t1.quote_client = mock_quote_client
        
        try:
            # 测试带时间范围的大批量数据获取，触发分页逻辑
            now = datetime.datetime.now()
            start_time = now - datetime.timedelta(days=10)
            result = t1.get_kline_data("SIL2603", "1min", count=2000, start_time=start_time, end_time=now)
            self.assertIsInstance(result, pd.DataFrame)
            print("✅ test_with_paging_client passed")
        finally:
            # 恢复原始值
            if original_quote_client is not None:
                t1.quote_client = original_quote_client
            else:
                if hasattr(t1, 'quote_client'):
                    delattr(t1, 'quote_client')
    
    def test_with_iterable_bars(self):
        """测试处理可迭代bar对象的情况"""
        # 创建模拟的bar对象
        class MockBar:
            def __init__(self, time, open, high, low, close, volume):
                self.time = time
                self.open = open
                self.high = high
                self.low = low
                self.close = close
                self.volume = volume
        
        mock_bars = [
            MockBar(pd.Timestamp('2023-01-01'), 90.0, 91.0, 89.0, 90.5, 100),
            MockBar(pd.Timestamp('2023-01-02'), 90.5, 91.5, 89.5, 91.0, 150),
            MockBar(pd.Timestamp('2023-01-03'), 91.0, 92.0, 90.0, 91.5, 200)
        ]
        
        mock_quote_client = Mock()
        mock_quote_client.get_future_bars.return_value = mock_bars
        mock_quote_client.get_future_bars_by_page = None
        
        # 临时替换tiger1中的quote_client
        original_quote_client = getattr(t1, 'quote_client', None)
        t1.quote_client = mock_quote_client
        
        try:
            result = t1.get_kline_data("SIL2603", "1min", count=10)
            self.assertIsInstance(result, pd.DataFrame)
            print("✅ test_with_iterable_bars passed")
        finally:
            # 恢复原始值
            if original_quote_client is not None:
                t1.quote_client = original_quote_client
            else:
                if hasattr(t1, 'quote_client'):
                    delattr(t1, 'quote_client')


if __name__ == '__main__':
    print("🚀 开始运行使用quote_client的测试...")
    
    # 运行测试
    unittest.main(verbosity=2)