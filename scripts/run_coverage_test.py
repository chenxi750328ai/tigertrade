#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行覆盖率测试的脚本
"""

import sys
import os
import unittest

# 添加tigertrade目录到路径
sys.path.insert(0, '/home/cx/tigertrade')

# 导入测试类
from comprehensive_test_suite import TestTiger1Functions

def run_coverage_test():
    """运行覆盖率测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestTiger1Functions)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result

if __name__ == '__main__':
    run_coverage_test()