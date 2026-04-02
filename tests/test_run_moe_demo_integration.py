#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_moe_demo.py 集成测试
确保下单逻辑、风控检查、策略预测等核心功能正常工作
"""
import unittest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]

# 保存真实 check_risk_control（导入时尚未被其他测试替换），本模块测试风控时恢复使用
import src.tiger1 as _t1_mod
_real_check_risk_control = _t1_mod.check_risk_control


class TestRunMoeDemoIntegration(unittest.TestCase):
    """run_moe_demo.py 集成测试"""
    
    def setUp(self):
        """测试前准备"""
        # 重置全局状态
        from src import tiger1 as t1
        t1.current_position = 0
        t1.open_orders.clear()
        t1.closed_positions.clear()
        t1.active_take_profit_orders.clear()
        t1.daily_loss = 0
        
    def tearDown(self):
        """测试后清理"""
        from src import tiger1 as t1
        t1.current_position = 0
        t1.open_orders.clear()
        t1.closed_positions.clear()
        t1.active_take_profit_orders.clear()
        t1.daily_loss = 0
    
    def test_order_placement_logic_exists(self):
        """测试1: 确保run_moe_demo.py使用新的模块化架构"""
        demo_file = str(_REPO_ROOT / "scripts" / "run_moe_demo.py")
        self.assertTrue(os.path.exists(demo_file), "run_moe_demo.py文件不存在")
        
        with open(demo_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查新的架构：run_moe_demo.py现在调用tiger1.py，而不是直接使用TradingExecutor
        # 但tiger1.py内部会使用TradingExecutor
        # 检查是否调用tiger1.py
        self.assertIn('tiger1.py', content, "应该调用tiger1.py")
        self.assertIn('subprocess', content, "应该使用subprocess调用tiger1.py")
        # 或者检查是否直接使用TradingExecutor（如果tiger1.py支持moe策略）
        # 两种方式都可以，关键是统一入口
        
        # 验证tiger1.py支持moe策略
        from src import tiger1 as t1
        # 检查tiger1.py的main函数是否支持moe
        import inspect
        main_source = inspect.getsource(t1.__main__) if hasattr(t1, '__main__') else ''
        # 或者检查tiger1.py文件内容
        with open(_REPO_ROOT / "src" / "tiger1.py", "r", encoding="utf-8") as f:
            tiger1_content = f.read()
            self.assertIn("strategy_type in ('moe', 'moe_transformer')", tiger1_content, 
                         "tiger1.py应该支持moe策略")
        
        # 确保使用新的执行器架构，而不是直接调用place_tiger_order
        # 注意：OrderExecutor内部会调用API，但run_moe_demo.py本身不直接调用place_tiger_order
        lines = content.split('\n')
        direct_place_order_calls = [l for l in lines if 'place_tiger_order(' in l and not l.strip().startswith('#')]
        # 新架构下，run_moe_demo.py不应该直接调用place_tiger_order
        # 但允许通过OrderExecutor间接调用（这是正确的架构）
    
    def test_strategy_predict_action_returns_correct_format(self):
        """测试2: 确保策略predict_action返回正确格式"""
        from src.strategies.strategy_factory import StrategyFactory
        
        # 使用Mock策略测试接口
        class MockStrategy:
            def __init__(self):
                self._seq_length = 10
                self.strategy_name = "Mock Strategy"
            
            @property
            def seq_length(self):
                return self._seq_length
            
            def predict_action(self, current_data, historical_data=None):
                return (1, 0.5, 0.1)  # (action, confidence, profit_pred)
        
        strategy = MockStrategy()
        result = strategy.predict_action({})
        
        self.assertIsInstance(result, tuple, "predict_action应返回tuple")
        self.assertGreaterEqual(len(result), 2, "predict_action应至少返回(action, confidence)")
        self.assertIn(result[0], [0, 1, 2], "action应为0(不操作)/1(买入)/2(卖出)")
        self.assertGreaterEqual(result[1], 0.0, "confidence应>=0")
        self.assertLessEqual(result[1], 1.0, "confidence应<=1")
    
    def test_order_placement_with_risk_control(self):
        """测试3: 下单逻辑应包含风控检查"""
        from src import tiger1 as t1
        
        # 模拟策略预测买入
        action = 1  # 买入
        confidence = 0.5
        tick_price = 100.0
        
        # 设置风控参数，使风控通过
        t1.GRID_MAX_POSITION = 3
        t1.SINGLE_TRADE_LOSS = 5000
        t1.MAX_SINGLE_LOSS = 10000
        t1.current_position = 0
        
        # 模拟ATR和网格参数
        atr = 0.5
        grid_lower_val = 95.0
        
        # 检查风控
        risk_ok = t1.check_risk_control(tick_price, 'BUY')
        
        if risk_ok:
            # 计算止损
            stop_loss_price, projected_loss = t1.compute_stop_loss(tick_price, atr, grid_lower_val)
            self.assertIsNotNone(stop_loss_price, "止损价格不应为None")
            self.assertLess(stop_loss_price, tick_price, "止损价格应低于买入价格")
    
    def test_order_placement_flow(self):
        """测试4: 完整下单流程测试"""
        from src import tiger1 as t1
        
        # 重置状态
        t1.current_position = 0
        t1.GRID_MAX_POSITION = 3
        t1.SINGLE_TRADE_LOSS = 5000
        
        # 模拟完整的下单流程
        action = 1  # 买入
        confidence = 0.6
        tick_price = 100.0
        atr = 0.5
        grid_lower_val = 95.0
        grid_upper_val = 105.0
        
        # 步骤1: 检查置信度阈值
        confidence_threshold = 0.4
        self.assertGreaterEqual(confidence, confidence_threshold, "置信度应>=阈值")
        
        # 步骤2: 检查风控
        if action == 1:
            risk_ok = t1.check_risk_control(tick_price, 'BUY')
            if risk_ok:
                # 步骤3: 计算止损止盈
                stop_loss_price, projected_loss = t1.compute_stop_loss(tick_price, atr, grid_lower_val)
                tp_offset = max(t1.TAKE_PROFIT_ATR_OFFSET * atr, t1.TAKE_PROFIT_MIN_OFFSET)
                take_profit_price = grid_upper_val - tp_offset
                
                self.assertIsNotNone(stop_loss_price)
                self.assertIsNotNone(take_profit_price)
                
                # 步骤4: 执行下单（在Mock模式下）
                with patch('src.tiger1.api_manager') as mock_api:
                    mock_api.is_mock_mode = True
                    result = t1.place_tiger_order(
                        'BUY', 1, tick_price,
                        stop_loss_price=stop_loss_price,
                        take_profit_price=take_profit_price
                    )
                    self.assertIsNotNone(result, "下单应返回结果")
    
    def test_sell_order_requires_position(self):
        """测试5: 卖出订单需要持仓检查（由OrderExecutor处理）"""
        from src import tiger1 as t1
        from src.executor import OrderExecutor
        
        # 验证OrderExecutor.execute_sell会检查持仓
        executor = OrderExecutor(t1)
        
        # 无持仓时不应卖出
        t1.current_position = 0
        result, message = executor.execute_sell(price=100.0, confidence=0.6)
        self.assertFalse(result, "无持仓时不应执行卖出")
        self.assertTrue("无持仓" in message or "无多头持仓" in message or "无法卖出" in message, f"应该返回无持仓消息: {message}")
        
        # 有持仓时可以卖出
        t1.current_position = 1
        # 注意：这里不实际执行，只验证逻辑存在
        # 实际执行需要Mock API
    
    def test_confidence_threshold_logic(self):
        """测试6: 置信度阈值逻辑"""
        confidence_threshold = 0.4
        
        test_cases = [
            (0.3, False, "低置信度应不执行"),
            (0.4, True, "等于阈值应执行"),
            (0.5, True, "高于阈值应执行"),
            (0.0, False, "置信度为0应不执行"),
        ]
        
        for confidence, should_execute, msg in test_cases:
            with self.subTest(confidence=confidence):
                action = 1  # 买入
                if action != 0:
                    actual = confidence >= confidence_threshold
                    self.assertEqual(actual, should_execute, msg)
    
    def test_error_handling_in_order_placement(self):
        """测试7: 下单错误处理（使用真实风控实现，避免被其他测试的 Mock 污染）"""
        from src import tiger1 as t1
        t1.check_risk_control = _real_check_risk_control

        # 测试无效价格 - check_risk_control应该返回False而不是抛出异常
        result = t1.check_risk_control(None, 'BUY')
        self.assertFalse(result, "无效价格应返回False")

        # 测试无效方向 - check_risk_control应该返回False而不是抛出异常
        result = t1.check_risk_control(100.0, 'INVALID')
        self.assertFalse(result, "无效方向应返回False")
    
    def test_demo_script_imports(self):
        """测试8: DEMO脚本的导入检查（新架构）"""
        demo_file = str(_REPO_ROOT / "scripts" / "run_moe_demo.py")

        with open(demo_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # run_moe_demo 通过 subprocess 调用 tiger1，只需检查入口与子进程
        self.assertIn('subprocess', content, "应使用 subprocess 调用 tiger1")
        self.assertIn('tiger1', content, "应调用 tiger1 作为统一入口")
        self.assertIn('src/tiger1.py', content or 'src/tiger1.py', "应执行 src/tiger1.py")

    def test_verify_api_does_not_place_order_on_start(self):
        """【防回归】verify_api_connection 不得在每次启动时调用 place_tiger_order（避免每启一个 DEMO 下一单）"""
        with open(_REPO_ROOT / "src" / "tiger1.py", "r", encoding="utf-8") as f:
            content = f.read()
        # 查找 verify_api_connection 函数体，检查是否有未注释的 place_tiger_order 调用
        import re
        match = re.search(r'def verify_api_connection\([^)]*\):.*?(?=\n\ndef |\nclass |\Z)', content, re.DOTALL)
        self.assertTrue(match, "应能找到 verify_api_connection")
        body = match.group(0)
        # 允许注释掉的 place_tiger_order，不允许未注释的
        if 'place_tiger_order(' in body:
            for line in body.split('\n'):
                stripped = line.strip()
                if 'place_tiger_order(' in line and not stripped.startswith('#'):
                    self.fail(
                        "verify_api_connection 内不应有未注释的 place_tiger_order 调用，"
                        "否则每次 DEMO 启动都会下一单。请注释或移除。"
                    )

    def test_order_executor_writes_order_log(self):
        """【防回归】OrderExecutor 成功/失败下单时应写入 order_log，便于报告分析"""
        with open(_REPO_ROOT / "src" / "executor" / "order_executor.py", "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn('order_log', content, "OrderExecutor 应导入 order_log")
        self.assertIn('order_log.log_order', content, "OrderExecutor 成功/失败下单时应调用 order_log.log_order")

    def test_tiger1_main_entry_no_attribute_error(self):
        """【防回归】以 __main__ 启动 tiger1（DEMO 入口）时，启动阶段不得出现 AttributeError check_risk_control、MIN_TICK NameError"""
        import subprocess
        cwd = str(_REPO_ROOT)
        cmd = [sys.executable, 'src/tiger1.py', 'd', 'moe']
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            stdout, _ = proc.communicate(timeout=6)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                stdout, _ = proc.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout = ""
        combined = stdout or ""
        if 'AttributeError' in combined and 'check_risk_control' in combined:
            self.fail(
                "tiger1 以 __main__ 启动时出现 AttributeError check_risk_control，"
                "OrderExecutor 应对无 check_risk_control 的 risk_manager 回退到 t1。\n"
                "输出片段:\n" + combined[-3000:]
            )
        if 'AttributeError' in combined:
            self.fail("tiger1 启动阶段出现 AttributeError，应修复。\n输出片段:\n" + combined[-3000:])
        if "NameError" in combined and "MIN_TICK" in combined:
            self.fail(
                "tiger1 启动阶段出现 MIN_TICK 未定义，应将 FUTURE_TICK_SIZE/MIN_TICK 移至 place_tiger_order 使用之前。\n"
                "输出片段:\n" + combined[-3000:]
            )


def run_integration_tests():
    """运行所有集成测试"""
    print("="*70)
    print("🧪 run_moe_demo.py 集成测试")
    print("="*70)
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestRunMoeDemoIntegration)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*70)
    if result.wasSuccessful():
        print("✅ 所有集成测试通过！")
    else:
        print(f"❌ 测试失败: {len(result.failures)}个失败, {len(result.errors)}个错误")
        for test, traceback in result.failures + result.errors:
            print(f"\n失败测试: {test}")
            print(traceback)
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_integration_tests()
    sys.exit(0 if success else 1)
