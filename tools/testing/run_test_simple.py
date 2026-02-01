#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化的测试运行脚本，输出清晰可见
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

# 抑制tiger1.py中的print输出
class SuppressOutput:
    """抑制标准输出"""
    def __init__(self):
        self.stdout = sys.stdout
        self.stderr = sys.stderr
    
    def __enter__(self):
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        return self
    
    def __exit__(self, *args):
        sys.stdout = self.stdout
        sys.stderr = self.stderr

def run_tests():
    """运行测试并显示清晰的结果"""
    print("=" * 80)
    print("🚀 开始运行 tiger1.py 测试")
    print("=" * 80)
    print()
    
    # 导入测试模块
    try:
        import test_tiger1_full_coverage
        import test_tiger1_additional_coverage
    except ImportError as e:
        print(f"❌ 导入测试模块失败: {e}")
        return False
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试
    print("📦 加载测试用例...")
    suite.addTests(loader.loadTestsFromModule(test_tiger1_full_coverage))
    suite.addTests(loader.loadTestsFromModule(test_tiger1_additional_coverage))
    
    total_tests = suite.countTestCases()
    print(f"✅ 已加载 {total_tests} 个测试用例")
    print()
    
    # 运行测试 - 使用StringIO捕获输出，然后过滤
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
            # 使用sys.stdout直接输出，避免被抑制
            sys.stdout.write(f"[{self.test_count:2d}/{total_tests}] {test_name:50s} ... ")
            sys.stdout.flush()
        
        def addSuccess(self, test):
            super().addSuccess(test)
            sys.stdout.write("✅ 通过\n")
            sys.stdout.flush()
        
        def addError(self, test, err):
            super().addError(test, err)
            sys.stdout.write("❌ 错误\n")
            sys.stdout.flush()
            # 只显示错误类型
            error_type = err[0].__name__
            sys.stdout.write(f"      └─ 错误类型: {error_type}\n")
            sys.stdout.flush()
        
        def addFailure(self, test, err):
            super().addFailure(test, err)
            sys.stdout.write("⚠️  失败\n")
            sys.stdout.flush()
            # 只显示失败原因的第一行
            error_msg = str(err[1]).split('\n')[0]
            sys.stdout.write(f"      └─ {error_msg[:80]}\n")
            sys.stdout.flush()
    
    # 运行测试 - 抑制tiger1.py的输出
    stream = io.StringIO()
    
    # 创建一个自定义的runner来抑制输出
    class QuietTestRunner(unittest.TextTestRunner):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.buffer = True
    
    runner = QuietTestRunner(
        stream=stream,
        verbosity=0,
        resultclass=ProgressTestResult,
        buffer=True  # 缓冲输出
    )
    
    # 使用StringIO捕获输出，然后过滤
    import io
    from contextlib import redirect_stdout, redirect_stderr
    
    # 创建一个过滤器来抑制tiger1.py的调试输出
    class OutputFilter(io.TextIOWrapper):
        def __init__(self, original):
            self.original = original
            self.buffer = io.StringIO()
        
        def write(self, s):
            # 过滤掉不需要的输出
            if any(keyword in s for keyword in ['📈 网格参数', '📊 数据点', '🔸', '🧭', '✅ [模拟单]', '🛡️', '⚠️ 数据不足', '⚠️ 指标计算']):
                return len(s)  # 假装写入了，但不实际输出
            # 保留测试相关的输出
            if any(keyword in s for keyword in ['测试', '通过', '失败', '错误', '✅', '❌', '⚠️', '📊', '📈', '🚀']):
                return self.original.write(s)
            return len(s)
        
        def flush(self):
            self.original.flush()
    
    # 运行测试
    result = runner.run(suite)
    
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
            # 只显示前几行错误信息
            lines = traceback.split('\n')[:5]
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
            # 只显示前几行错误信息
            lines = traceback.split('\n')[:5]
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
        
        # 重新运行测试
        runner2 = unittest.TextTestRunner(verbosity=0)
        result2 = runner2.run(suite)
        
        cov.stop()
        cov.save()
        
        # 显示覆盖率报告
        print("=" * 80)
        print("📊 代码覆盖率报告 (tiger1.py):")
        print("=" * 80)
        
        tiger1_path = os.path.join(tigertrade_dir, 'tiger1.py')
        if os.path.exists(tiger1_path):
            try:
                # 生成报告
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
                # 尝试生成所有文件的报告
                try:
                    cov.report()
                except:
                    pass
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
