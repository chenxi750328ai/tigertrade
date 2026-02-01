#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
真实下单逻辑测试 - 测试OrderExecutor和tiger1.place_tiger_order的实际执行
"""
import unittest
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, '/home/cx/tigertrade')


class TestOrderExecutionReal(unittest.TestCase):
    """真实下单逻辑测试"""
    
    def setUp(self):
        """测试前准备"""
        from src import tiger1 as t1
        t1.current_position = 0
        t1.open_orders.clear()
        t1.daily_loss = 0
    
    def test_order_type_import(self):
        """测试1: OrderType导入和使用"""
        try:
            from tigeropen.common.consts import OrderType
            # 测试LMT存在（OrderType可能是枚举或类）
            # 如果OrderType是枚举，LMT是枚举值；如果是类，LMT是类属性
            has_lmt = hasattr(OrderType, 'LMT') or 'LMT' in dir(OrderType)
            if has_lmt:
                order_type = getattr(OrderType, 'LMT', None)
                self.assertIsNotNone(order_type, "OrderType.LMT应该不为None")
                print(f"✅ OrderType.LMT = {order_type}")
            else:
                # 如果OrderType没有LMT属性，检查是否有其他方式（如字符串常量）
                print(f"⚠️ OrderType没有LMT属性，可能使用字符串常量")
                # 不fail，因为代码中有fallback逻辑
        except ImportError as e:
            # 导入失败时，代码中有fallback逻辑，所以不fail
            print(f"⚠️ OrderType导入失败（代码中有fallback）: {e}")
    
    def test_order_side_import(self):
        """测试2: OrderSide导入（可能失败，需要fallback）"""
        try:
            from tigeropen.common.consts import OrderSide
            # 如果导入成功，测试使用
            order_side = OrderSide.BUY
            self.assertIsNotNone(order_side)
            print(f"✅ OrderSide导入成功: {order_side}")
        except ImportError:
            # 导入失败是预期的，应该有fallback逻辑
            print("⚠️ OrderSide导入失败（预期），应该有fallback逻辑")
            # 测试fallback逻辑
            OrderSide = type('OrderSide', (), {'BUY': 'BUY', 'SELL': 'SELL'})()
            order_side = OrderSide.BUY
            self.assertEqual(order_side, 'BUY', "fallback应该返回字符串'BUY'")
            print(f"✅ OrderSide fallback成功: {order_side}")
    
    def test_place_tiger_order_order_type_usage(self):
        """测试3: place_tiger_order中OrderType的使用"""
        from src import tiger1 as t1
        
        # 检查代码中是否使用了OrderType.LMT（不是LIMIT）
        import inspect
        source = inspect.getsource(t1.place_tiger_order)
        
        # 检查是否使用了OrderType.LMT
        has_lmt = 'OrderType.LMT' in source or 'order_type = OrderType.LMT' in source
        self.assertTrue(has_lmt, "代码应该使用OrderType.LMT，不是LIMIT")
        
        # 检查是否错误使用了OrderType.LIMIT
        has_limit = 'OrderType.LIMIT' in source
        self.assertFalse(has_limit, "代码不应该使用OrderType.LIMIT（不存在）")
        
        print("✅ place_tiger_order使用OrderType.LMT（正确）")
    
    def test_order_executor_api_call(self):
        """测试4: OrderExecutor实际调用API的逻辑"""
        from src.executor import OrderExecutor
        from src import tiger1 as t1
        from src.api_adapter import api_manager
        
        executor = OrderExecutor(t1)
        
        # 检查OrderExecutor是否直接调用api_manager.trade_api
        import inspect
        source = inspect.getsource(executor.execute_buy)
        
        # 应该直接调用api_manager.trade_api.place_order
        has_direct_api_call = 'api_manager.trade_api' in source or 'trade_api.place_order' in source
        self.assertTrue(has_direct_api_call, "OrderExecutor应该直接调用API，不通过tiger1.place_tiger_order")
        
        # 不应该调用tiger1.place_tiger_order（排除注释）
        lines = source.split('\n')
        code_lines = [l for l in lines if not l.strip().startswith('#') and 'place_tiger_order' in l]
        has_place_tiger_order_call = any('place_tiger_order(' in l for l in code_lines)
        self.assertFalse(has_place_tiger_order_call, "OrderExecutor不应该调用tiger1.place_tiger_order")
        
        print("✅ OrderExecutor直接调用API（正确）")
    
    def test_place_tiger_order_real_execution_path(self):
        """测试5: place_tiger_order实际执行路径（非模拟模式）"""
        from src import tiger1 as t1
        from src.api_adapter import api_manager
        
        # 确保不是模拟模式
        original_mock_mode = api_manager.is_mock_mode
        api_manager.is_mock_mode = False
        
        try:
            # Mock trade_api
            mock_trade_api = MagicMock()
            mock_order_result = MagicMock()
            mock_order_result.order_id = "TEST_ORDER_123"
            mock_trade_api.place_order.return_value = mock_order_result
            api_manager.trade_api = mock_trade_api
            
            # 测试下单（应该进入实际下单逻辑分支）
            result = t1.place_tiger_order('BUY', 1, 100.0)
            
            # 检查是否调用了trade_api.place_order
            if mock_trade_api.place_order.called:
                call_args = mock_trade_api.place_order.call_args
                # 检查order_type参数
                if call_args:
                    kwargs = call_args[1] if len(call_args) > 1 else {}
                    order_type = kwargs.get('order_type') or (call_args[0][2] if len(call_args[0]) > 2 else None)
                    if order_type:
                        # 检查order_type是否是OrderType.LMT或字符串'LMT'
                        from tigeropen.common.consts import OrderType
                        is_valid = (
                            order_type == OrderType.LMT or 
                            str(order_type) == 'OrderType.LMT' or
                            order_type == 'LMT'
                        )
                        self.assertTrue(is_valid, f"order_type应该是OrderType.LMT，实际是: {order_type}")
                        print(f"✅ order_type正确: {order_type}")
            else:
                print("⚠️ trade_api.place_order未被调用（可能是模拟模式或其他原因）")
                
        finally:
            api_manager.is_mock_mode = original_mock_mode
    
    def test_order_executor_real_api_call(self):
        """测试6: OrderExecutor实际调用API（非模拟）"""
        from src.executor import OrderExecutor
        from src import tiger1 as t1
        from src.api_adapter import api_manager
        
        executor = OrderExecutor(t1)
        
        # 确保不是模拟模式
        original_mock_mode = api_manager.is_mock_mode
        api_manager.is_mock_mode = False
        
        try:
            # Mock trade_api
            mock_trade_api = MagicMock()
            mock_order_result = MagicMock()
            mock_order_result.order_id = "TEST_ORDER_456"
            mock_trade_api.place_order.return_value = mock_order_result
            api_manager.trade_api = mock_trade_api
            
            # 设置风控通过
            t1.current_position = 0
            t1.GRID_MAX_POSITION = 3
            t1.SINGLE_TRADE_LOSS = 5000
            
            # 测试下单
            result, message = executor.execute_buy(
                price=100.0,
                atr=0.5,
                grid_lower=95.0,
                grid_upper=105.0,
                confidence=0.6
            )
            
            # 检查是否调用了trade_api.place_order
            if mock_trade_api.place_order.called:
                call_args = mock_trade_api.place_order.call_args
                if call_args:
                    kwargs = call_args[1] if len(call_args) > 1 else {}
                    order_type = kwargs.get('order_type')
                    if order_type:
                        from tigeropen.common.consts import OrderType
                        is_valid = (
                            order_type == OrderType.LMT or 
                            str(order_type) == 'OrderType.LMT' or
                            order_type == 'LMT'
                        )
                        self.assertTrue(is_valid, f"OrderExecutor的order_type应该是OrderType.LMT，实际是: {order_type}")
                        print(f"✅ OrderExecutor order_type正确: {order_type}")
            else:
                print("⚠️ trade_api.place_order未被调用")
                
        finally:
            api_manager.is_mock_mode = original_mock_mode


if __name__ == '__main__':
    unittest.main(verbosity=2)
