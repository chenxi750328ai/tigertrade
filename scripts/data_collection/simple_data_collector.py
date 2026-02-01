#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单数据收集脚本，直接运行增强版策略函数来收集数据
"""

import sys
import os
from datetime import datetime

# 添加tigertrade目录到路径
sys.path.insert(0, '/home/cx/tigertrade')

def collect_data_once():
    """运行一次数据收集"""
    print(f"🚀 启动数据收集 - {datetime.now()}")
    
    # 导入tiger1模块
    from src import tiger1 as t1
    
    # 导入并应用数据收集增强
    from data_collector_analyzer import enhance_strategy_with_logging
    enhance_strategy_with_logging()
    
    print("✅ 策略函数已增强，开始执行...")
    
    # 执行增强版策略函数
    t1.grid_trading_strategy_pro1()
    
    print(f"✅ 数据收集完成 - {datetime.now()}")


if __name__ == "__main__":
    collect_data_once()