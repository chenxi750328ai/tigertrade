#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行全面测试的脚本
"""

import sys
import os

# 添加tigertrade目录到路径
sys.path.insert(0, '/home/cx/tigertrade')

# 导入测试类
from comprehensive_coverage_test import run_comprehensive_tests

if __name__ == '__main__':
    run_comprehensive_tests()