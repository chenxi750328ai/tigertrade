#!/usr/bin/env python3
"""
带验证的数据采集脚本

强制要求：
1. 使用真实API
2. 拒绝Mock数据
3. 检查所有异常
4. 验证数据质量
"""

import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(__file__))

from data_collection_validator import DataCollectionValidator, force_real_api_initialization
from collect_large_dataset import LargeDatasetCollector
from config import FUTURE_SYMBOL


def collect_data_with_validation(symbol: str = None):
    """
    带完整验证的数据采集
    
    流程：
    1. 强制初始化真实API
    2. 验证API
    3. 采集数据
    4. 验证K线数据
    5. 计算特征
    6. 验证特征
    7. 检查异常
    """
    
    print("=" * 80)
    print("📥 带验证的数据采集")
    print("=" * 80)
    print()
    
    symbol = symbol or FUTURE_SYMBOL
    
    # 创建验证器（严格模式）
    validator = DataCollectionValidator(strict_mode=True)
    
    try:
        # 步骤1: 强制初始化真实API
        print("阶段1: API初始化")
        print("-" * 80)
        api_manager = force_real_api_initialization()
        
        # 步骤2: 验证API
        if not validator.validate_api_initialization(api_manager):
            print("❌ API验证失败，终止")
            return None
        
        # 步骤3: 创建采集器（强制使用真实API）
        print("阶段2: 创建数据采集器")
        print("-" * 80)
        collector = LargeDatasetCollector(symbol)
        collector.use_real_api = True  # 强制真实API
        
        # 验证采集器配置
        if not collector.use_real_api:
            raise RuntimeError("❌ 采集器未配置为使用真实API")
        
        print(f"✅ 采集器配置:")
        print(f"   标的: {symbol}")
        print(f"   使用真实API: {collector.use_real_api}")
        print()
        
        # 步骤4: 获取K线数据
        print("阶段3: 获取K线数据")
        print("-" * 80)
        
        # 捕获输出以检查异常
        import io
        from contextlib import redirect_stdout, redirect_stderr
        
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            df_1m = collector.fetch_kline_data_with_retry('1min', 500)
            df_5m = collector.fetch_kline_data_with_retry('5min', 100)
        
        stdout_text = stdout_capture.getvalue()
        stderr_text = stderr_capture.getvalue()
        
        # 检查异常
        if stdout_text:
            print("标准输出:")
            print(stdout_text)
            validator.check_for_exceptions_in_output(stdout_text)
        
        if stderr_text:
            print("标准错误:")
            print(stderr_text)
            validator.check_for_exceptions_in_output(stderr_text)
        
        # 步骤5: 验证K线数据
        validator.validate_kline_data(df_1m, '1min', expected_min_rows=500)
        validator.validate_kline_data(df_5m, '5min', expected_min_rows=100)
        
        # 步骤6: 计算特征
        print("阶段4: 计算特征")
        print("-" * 80)
        
        df_features = collector.calculate_features_optimized(df_5m, df_1m)
        
        # 步骤7: 验证特征
        validator.validate_features(df_features)
        
        # 步骤8: 生成标签和分割数据
        print("阶段5: 生成标签和分割数据")
        print("-" * 80)
        
        df_labeled = collector.generate_labels(df_features)
        train, val, test = collector.split_dataset(df_labeled)
        
        print(f"✅ 数据分割完成:")
        print(f"   训练集: {len(train)} 条")
        print(f"   验证集: {len(val)} 条")
        print(f"   测试集: {len(test)} 条")
        print()
        
        # 步骤9: 保存数据
        print("阶段6: 保存数据")
        print("-" * 80)
        
        files = collector.save_datasets(train, val, test, df_labeled)
        
        print(f"✅ 数据已保存:")
        for key, path in files.items():
            print(f"   {key}: {path}")
        print()
        
        # 打印验证摘要
        validator.print_summary()
        
        return files
        
    except Exception as e:
        print()
        print("=" * 80)
        print("❌ 数据采集失败")
        print("=" * 80)
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        
        validator.print_summary()
        
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='带验证的数据采集')
    parser.add_argument('--symbol', type=str, help='期货标的符号')
    
    args = parser.parse_args()
    
    result = collect_data_with_validation(args.symbol)
    
    if result:
        print()
        print("✅ 数据采集成功！")
        sys.exit(0)
    else:
        print()
        print("❌ 数据采集失败！")
        sys.exit(1)
