#!/usr/bin/env python3
"""
灵活的数据采集脚本

特点：
1. 让用户选择使用Mock还是真实API
2. 明确标识数据来源
3. 检查所有异常
4. 验证数据质量
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from data_collection_validator import DataCollectionValidator
from collect_large_dataset import LargeDatasetCollector

# 默认标的
FUTURE_SYMBOL = 'SIL2603'


def collect_data_with_options(
    symbol: str = None,
    use_mock: bool = False,
    strict: bool = True
):
    """
    灵活的数据采集
    
    参数：
        symbol: 期货标的
        use_mock: 是否允许使用Mock数据
        strict: 严格模式（发现错误立即终止）
    """
    
    print("=" * 80)
    print("📥 数据采集")
    print("=" * 80)
    print(f"配置:")
    print(f"  标的: {symbol or FUTURE_SYMBOL}")
    print(f"  允许Mock: {use_mock}")
    print(f"  严格模式: {strict}")
    print("=" * 80)
    print()
    
    symbol = symbol or FUTURE_SYMBOL
    
    # 创建验证器
    validator = DataCollectionValidator(strict_mode=strict)
    
    try:
        # 检查API状态
        from api_adapter import api_manager
        
        # 验证API（根据use_mock决定是否允许Mock）
        validator.validate_api_initialization(
            api_manager,
            allow_mock=use_mock,
            warn_on_mock=True
        )
        
        # 创建采集器
        collector = LargeDatasetCollector(symbol)
        
        # 如果不允许Mock，设置use_real_api=True
        if not use_mock:
            collector.use_real_api = True
        
        print(f"采集器配置:")
        print(f"  使用真实API: {collector.use_real_api}")
        print()
        
        # 获取数据
        print("开始获取数据...")
        print("-" * 80)
        
        df_1m = collector.fetch_kline_data_with_retry('1min', 500)
        df_5m = collector.fetch_kline_data_with_retry('5min', 100)
        
        # 验证数据
        validator.validate_kline_data(df_1m, '1min', expected_min_rows=100)
        validator.validate_kline_data(df_5m, '5min', expected_min_rows=50)
        
        # 计算特征
        print("计算特征...")
        print("-" * 80)
        
        df_features = collector.calculate_features_optimized(df_5m, df_1m)
        
        # 验证特征
        validator.validate_features(df_features)
        
        # 生成标签和分割
        df_labeled = collector.generate_labels(df_features)
        train, val, test = collector.split_dataset(df_labeled)
        
        # 保存
        files = collector.save_datasets(train, val, test, df_labeled)
        
        # 打印摘要
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
    
    parser = argparse.ArgumentParser(description='数据采集脚本')
    parser.add_argument('--symbol', type=str, help='期货标的符号')
    parser.add_argument('--use-mock', action='store_true', 
                       help='允许使用Mock数据（开发测试用）')
    parser.add_argument('--no-strict', action='store_true',
                       help='非严格模式（遇到错误继续执行）')
    
    args = parser.parse_args()
    
    result = collect_data_with_options(
        symbol=args.symbol,
        use_mock=args.use_mock,
        strict=not args.no_strict
    )
    
    if result:
        print()
        print("✅ 数据采集成功！")
        sys.exit(0)
    else:
        print()
        print("❌ 数据采集失败！")
        sys.exit(1)
