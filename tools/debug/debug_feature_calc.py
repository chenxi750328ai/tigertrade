#!/usr/bin/env python3
"""调试特征计算流程"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from tiger1 import get_kline_data, calculate_indicators, FUTURE_SYMBOL
import pandas as pd

def main():
    print("=" * 80)
    print("调试特征计算流程")
    print("=" * 80)
    
    #获取测试数据
    print(f"\n标的: {FUTURE_SYMBOL}")
    print("获取100条数据进行测试...")
    
    df_1m = get_kline_data([FUTURE_SYMBOL], '1min', count=100)
    df_5m = get_kline_data([FUTURE_SYMBOL], '5min', count=100)
    
    print(f"\n数据获取完成:")
    print(f"  1分钟: {len(df_1m)} 条")
    print(f"  5分钟: {len(df_5m)} 条")
    
    # 测试 calculate_indicators
    print(f"\n测试 calculate_indicators...")
    print(f"  参数顺序: (df_1m, df_5m)")
    
    try:
        # 取最后50条测试
        window_1m = df_1m.iloc[-50:]
        window_5m = df_5m.iloc[-50:]
        
        print(f"\n测试数据:")
        print(f"  window_1m shape: {window_1m.shape}")
        print(f"  window_1m columns: {window_1m.columns.tolist()}")
        print(f"  window_1m index type: {type(window_1m.index)}")
        print(f"  window_5m shape: {window_5m.shape}")
        print(f"  window_5m columns: {window_5m.columns.tolist()}")
        print(f"  window_5m index type: {type(window_5m.index)}")
        
        # 打印前几行数据
        print(f"\n  window_1m 前3行:")
        print(window_1m.head(3))
        print(f"\n  window_5m 前3行:")
        print(window_5m.head(3))
        
        # 调用 calculate_indicators
        print(f"\n调用 calculate_indicators(window_1m, window_5m)...")
        inds = calculate_indicators(window_1m, window_5m)
        
        print(f"\n返回结果:")
        print(f"  类型: {type(inds)}")
        print(f"  键: {list(inds.keys())}")
        
        for key in inds:
            print(f"\n  inds['{key}']:")
            print(f"    类型: {type(inds[key])}")
            print(f"    键: {list(inds[key].keys())}")
            print(f"    内容: {inds[key]}")
        
        # 测试访问具体字段
        print(f"\n测试访问字段...")
        try:
            close_1m = inds['1m']['close']
            print(f"  ✅ inds['1m']['close'] = {close_1m}")
        except Exception as e:
            print(f"  ❌ 无法访问 inds['1m']['close']: {e}")
        
        try:
            close_5m = inds['5m']['close']
            print(f"  ✅ inds['5m']['close'] = {close_5m}")
        except Exception as e:
            print(f"  ❌ 无法访问 inds['5m']['close']: {e}")
            
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
