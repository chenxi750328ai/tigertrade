#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
清晰的测试运行脚本，输出简洁可见
"""

import sys
import os
import unittest
import io
from contextlib import redirect_stdout, redirect_stderr

# 添加tigertrade目录到路径
tigertrade_dir = '/home/cx/tigertrade'
if tigertrade_dir not in sys.path:
    sys.path.insert(0, tigertrade_dir)

os.environ['ALLOW_REAL_TRADING'] = '0'

# 创建一个输出过滤器
class QuietOutput:
    """安静的输出，只显示测试结果"""
    def __init__(self, original_stdout):
        self.original = original_stdout
        self.buffer = []
    
    def write(self, s):
        # 过滤掉tiger1.py的调试输出
        if any(keyword in s for keyword in [
            '📈 网格参数已更新',
            '📊 数据点已记录',
            '🔸 grid_trading_strategy_pro1',
            '🔸 boll1m_grid_strategy',
            '🧭 [模拟]',
            '✅ [模拟单]',
            '🛡️ [模拟单]',
            '⚠️ 数据不足',
            '⚠️ 指标计算失败',
            '⚠️ backtest_pro1',
            'pro1 回测:',
            'Using device:',
            'prepare_features错误',
            '训练过程错误',
            '标签分布:',
            '类别权重:'
        ]):
            return  # 不输出这些调试信息
        
        # 保留重要的测试输出
        self.original.write(s)
    
    def flush(self):
        self.original.flush()


def run_tests():
    """运行测试并显示清晰的结果"""
    print("=" * 80)
    print("🚀 开始运行 tiger1.py 测试")
    print("=" * 80)
    print()
    
    # 导入测试模块
    test_modules = []
    try:
        import test_tiger1_full_coverage
        test_modules.append(test_tiger1_full_coverage)
    except ImportError as e:
        print(f"⚠️  无法导入test_tiger1_full_coverage: {e}")
    
    try:
        import test_tiger1_additional_coverage
        test_modules.append(test_tiger1_additional_coverage)
    except ImportError as e:
        print(f"⚠️  无法导入test_tiger1_additional_coverage: {e}")
    
    try:
        import test_tiger1_100_coverage
        test_modules.append(test_tiger1_100_coverage)
    except ImportError as e:
        print(f"⚠️  无法导入test_tiger1_100_coverage: {e}")
    
    try:
        import test_tiger1_complete_coverage
        test_modules.append(test_tiger1_complete_coverage)
    except ImportError as e:
        print(f"⚠️  无法导入test_tiger1_complete_coverage: {e}")
    
    if not test_modules:
        print(f"❌ 没有可用的测试模块")
        return False
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试
    print("📦 加载测试用例...")
    for module in test_modules:
        suite.addTests(loader.loadTestsFromModule(module))
    
    total_tests = suite.countTestCases()
    print(f"✅ 已加载 {total_tests} 个测试用例")
    print()
    
    # 替换stdout以过滤输出
    original_stdout = sys.stdout
    quiet_stdout = QuietOutput(original_stdout)
    sys.stdout = quiet_stdout
    
    print("=" * 80)
    print("🧪 开始执行测试...")
    print("=" * 80)
    print()
    
    # 创建一个自定义的TestResult来显示进度
    class ProgressTestResult(unittest.TextTestResult):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.test_count = 0
        
        def startTest(self, test):
            super().startTest(test)
            self.test_count += 1
            test_name = test._testMethodName.replace('test_', '').replace('_', ' ')
            original_stdout.write(f"[{self.test_count:2d}/{total_tests}] {test_name:50s} ... ")
            original_stdout.flush()
        
        def addSuccess(self, test):
            super().addSuccess(test)
            original_stdout.write("✅ 通过\n")
            original_stdout.flush()
        
        def addError(self, test, err):
            super().addError(test, err)
            original_stdout.write("❌ 错误\n")
            original_stdout.flush()
            error_type = err[0].__name__
            original_stdout.write(f"      └─ 错误类型: {error_type}\n")
            original_stdout.flush()
        
        def addFailure(self, test, err):
            super().addFailure(test, err)
            original_stdout.write("⚠️  失败\n")
            original_stdout.flush()
            error_msg = str(err[1]).split('\n')[0]
            original_stdout.write(f"      └─ {error_msg[:80]}\n")
            original_stdout.flush()
    
    # 运行测试
    runner = unittest.TextTestRunner(
        stream=io.StringIO(),  # 输出到StringIO，不显示
        verbosity=0,
        resultclass=ProgressTestResult,
        buffer=True
    )
    
    try:
        result = runner.run(suite)
    finally:
        # 恢复stdout
        sys.stdout = original_stdout
    
    print()
    print("=" * 80)
    print("📊 测试结果汇总")
    print("=" * 80)
    print()
    
    # 统计结果
    total = result.testsRun
    passed = total - len(result.failures) - len(result.errors)
    failed = len(result.failures)
    errors = len(result.errors)
    
    print(f"总测试数:     {total}")
    print(f"✅ 通过:      {passed}")
    print(f"⚠️  失败:      {failed}")
    print(f"❌ 错误:      {errors}")
    
    if total > 0:
        pass_rate = (passed / total) * 100
        print(f"通过率:       {pass_rate:.2f}%")
    
    print()
    
    # 显示失败的测试
    if result.failures:
        print("=" * 80)
        print("⚠️  失败的测试:")
        print("=" * 80)
        for test, traceback in result.failures:
            print(f"\n❌ {test._testMethodName}")
            lines = traceback.split('\n')[:3]
            for line in lines:
                if line.strip():
                    print(f"   {line}")
    
    # 显示错误的测试
    if result.errors:
        print()
        print("=" * 80)
        print("❌ 错误的测试:")
        print("=" * 80)
        for test, traceback in result.errors:
            print(f"\n❌ {test._testMethodName}")
            lines = traceback.split('\n')[:3]
            for line in lines:
                if line.strip():
                    print(f"   {line}")
    
    print()
    print("=" * 80)
    
    # 运行覆盖率测试
    try:
        import coverage
        print()
        print("📈 生成代码覆盖率报告...")
        print()
        
        # 运行覆盖率测试
        cov = coverage.Coverage(source=[tigertrade_dir])
        cov.start()
        
        # 重新运行测试（安静模式）
        sys.stdout = quiet_stdout
        runner2 = unittest.TextTestRunner(verbosity=0, stream=io.StringIO())
        result2 = runner2.run(suite)
        sys.stdout = original_stdout
        
        cov.stop()
        cov.save()
        
        # 显示覆盖率报告
        print("=" * 80)
        print("📊 代码覆盖率报告 (tiger1.py):")
        print("=" * 80)
        
        tiger1_path = os.path.join(tigertrade_dir, 'tiger1.py')
        if os.path.exists(tiger1_path):
            try:
                # 生成报告到StringIO
                report_output = io.StringIO()
                cov.report(file=report_output, include=[tiger1_path])
                report_text = report_output.getvalue()
                
                # 只显示摘要行
                lines = report_text.split('\n')
                for line in lines:
                    if 'tiger1.py' in line or 'TOTAL' in line or 'Name' in line or '---' in line:
                        print(line)
                
                # 生成HTML报告
                cov.html_report(directory='htmlcov', include=[tiger1_path])
                print()
                print(f"✅ HTML覆盖率报告已生成: htmlcov/index.html")
                
            except Exception as e:
                print(f"⚠️  生成覆盖率报告时出错: {e}")
        else:
            print(f"⚠️  未找到文件: {tiger1_path}")
        
    except ImportError:
        print()
        print("⚠️  coverage模块未安装，跳过覆盖率报告")
        print("   安装命令: pip install coverage")
    except Exception as e:
        print()
        print(f"⚠️  生成覆盖率报告时出错: {e}")
    
    print()
    print("=" * 80)
    print("✅ 测试完成！")
    print("=" * 80)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
