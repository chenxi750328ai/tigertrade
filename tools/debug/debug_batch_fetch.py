#!/usr/bin/env python3
"""调试分批获取数据的问题"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from collect_large_dataset import LargeDatasetCollector

def main():
    print("=" * 80)
    print("调试分批获取数据")
    print("=" * 80)
    
    # 创建收集器
    collector = LargeDatasetCollector(
        use_real_api=True,
        days=90,
        max_records=50000
    )
    
    print("\n测试分批获取大量数据...")
    
    # 测试获取大量数据（会触发分批逻辑）
    try:
        print("\n获取10000条1分钟数据（会分批）...")
        df = collector._fetch_in_batches('1min', 10000, batch_size=5000)
        
        print(f"\n返回结果:")
        print(f"  类型: {type(df)}")
        print(f"  形状: {df.shape}")
        print(f"  列名: {df.columns.tolist()}")
        print(f"  索引类型: {type(df.index)}")
        print(f"  索引名: {df.index.name}")
        
        print(f"\n前5行:")
        print(df.head())
        
        print(f"\n后5行:")
        print(df.tail())
        
        # 测试iloc[-1]
        print(f"\n测试 iloc[-1]...")
        latest = df.iloc[-1]
        print(f"  类型: {type(latest)}")
        print(f"  索引: {latest.index.tolist()}")
        print(f"  值:\n{latest}")
        
        # 测试访问close列
        print(f"\n测试访问close...")
        try:
            close_val = latest['close']
            print(f"  ✅ close = {close_val}")
        except Exception as e:
            print(f"  ❌ 错误: {e}")
            
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
