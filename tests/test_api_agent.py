#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
专门测试API代理的测试用例
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np
from datetime import datetime

# 添加tigertrade目录到路径
sys.path.insert(0, '/home/cx/tigertrade')

from src.api_agent import APIAgent, MockQuoteClient, MockTradeClient


class TestAPIAgent(unittest.TestCase):
    """测试API代理"""
    
    def setUp(self):
        """初始化测试环境"""
        self.agent = APIAgent(use_mock=True)
    
    def test_mock_quote_client(self):
        """测试模拟行情客户端"""
        client = MockQuoteClient()
        
        # 测试获取股票行情
        stock_data = client.get_stock_briefs(['00700'])
        self.assertIsInstance(stock_data, pd.DataFrame)
        self.assertIn('symbol', stock_data.columns)
        
        # 测试获取期货交易所
        exchanges = client.get_future_exchanges()
        self.assertIsInstance(exchanges, pd.DataFrame)
        self.assertGreaterEqual(len(exchanges), 1)
        
        # 测试获取期货合约
        contracts = client.get_future_contracts('CME')
        self.assertIsInstance(contracts, pd.DataFrame)
        self.assertGreaterEqual(len(contracts), 1)
        
        # 测试获取所有期货合约
        all_contracts = client.get_all_future_contracts('SIL')
        self.assertIsInstance(all_contracts, pd.DataFrame)
        self.assertGreaterEqual(len(all_contracts), 1)
        
        # 测试获取当前期货合约
        current_contract = client.get_current_future_contract('SIL')
        self.assertIsInstance(current_contract, dict)
        self.assertIn('contract_code', current_contract)
        
        # 测试获取行情权限
        permissions = client.get_quote_permission()
        self.assertIsInstance(permissions, dict)
        self.assertIn('real_time', permissions)
        
        # 测试获取期货简要信息
        brief = client.get_future_brief(['SIL2603'])
        self.assertIsInstance(brief, pd.DataFrame)
        
        # 测试获取期货K线数据
        bars = client.get_future_bars(['SIL2603'], '1min', None, None, 10, None)
        self.assertIsInstance(bars, pd.DataFrame)
        
        # 测试获取股票K线数据
        stock_bars = client.get_bars(['AAPL'], '1min', None, None, 10, None)
        self.assertIsInstance(stock_bars, pd.DataFrame)
        
        print("✅ test_mock_quote_client passed")
    
    def test_mock_trade_client(self):
        """测试模拟交易客户端"""
        client = MockTradeClient()
        
        # 测试下单
        order = client.place_order('SIL2603', 'BUY', 'MKT', 1)
        self.assertIsNotNone(order.order_id)
        self.assertEqual(order.side, 'BUY')
        
        # 测试获取订单
        orders = client.get_orders()
        self.assertGreaterEqual(len(orders), 1)
        
        # 测试修改订单
        updated_order = client.modify_order(order.order_id, quantity=2)
        self.assertEqual(updated_order.quantity, 2)
        
        # 测试取消订单
        result = client.cancel_order(order.order_id)
        self.assertTrue(result)
        
        # 测试获取账户信息
        account_info = client.get_account_info()
        self.assertIsInstance(account_info, dict)
        self.assertIn('account_id', account_info)
        
        # 测试获取持仓
        positions = client.get_positions()
        self.assertIsInstance(positions, list)
        
        print("✅ test_mock_trade_client passed")
    
    def test_api_agent(self):
        """测试API代理"""
        # 测试获取K线数据
        kline_data = self.agent.get_kline_data(['SIL2603'], '1min', 10)
        self.assertIsInstance(kline_data, pd.DataFrame)
        self.assertGreaterEqual(len(kline_data), 1)
        
        # 测试获取账户信息
        account_info = self.agent.get_account_info()
        self.assertIsInstance(account_info, dict)
        
        # 测试下单
        order = self.agent.place_order('SIL2603', 'BUY', 'MKT', 1)
        self.assertIsNotNone(order.order_id)
        
        print("✅ test_api_agent passed")
    
    def test_edge_cases(self):
        """测试边缘情况"""
        client = MockQuoteClient()
        
        # 测试获取K线数据时异常处理
        bars = client.get_future_bars([], '1min', None, None, 0, None)
        self.assertIsInstance(bars, pd.DataFrame)
        
        print("✅ test_edge_cases passed")


def run_api_agent_tests():
    """运行API代理测试"""
    print("🚀 开始运行API代理测试...")
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestAPIAgent)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 统计结果
    total_tests = result.testsRun
    failed_tests = len(result.failures)
    error_tests = len(result.errors)
    passed_tests = total_tests - failed_tests - error_tests
    
    print(f"\n📊 API代理测试结果汇总:")
    print(f"   总测试数: {total_tests}")
    print(f"   通过测试: {passed_tests}")
    print(f"   失败测试: {failed_tests}")
    print(f"   错误测试: {error_tests}")
    print(f"   通过率: {passed_tests/total_tests*100:.2f}%")
    
    return result


if __name__ == '__main__':
    result = run_api_agent_tests()
    if result.wasSuccessful():
        print("\n🎉 API代理测试全部通过！")
    else:
        print("\n❌ 部分API代理测试失败")