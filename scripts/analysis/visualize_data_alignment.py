#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化训练数据的组合和对齐过程
展示K线数据、Tick数据、技术指标如何对齐成训练输入
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import glob

sys.path.insert(0, '/home/cx/tigertrade')

try:
    from src import tiger1 as t1
    from scripts.analysis.generate_training_data_from_klines import calculate_technical_indicators
except ImportError as e:
    print(f"⚠️ 导入错误: {e}")
    sys.exit(1)


def visualize_data_alignment(days=1, seq_length=10, show_example=True):
    """
    可视化数据对齐过程
    
    Args:
        days: 获取最近N天的数据
        seq_length: 序列长度
        show_example: 是否显示一个具体的对齐示例
    """
    print("=" * 80)
    print("📊 训练数据对齐过程可视化")
    print("=" * 80)
    
    # 1. 获取K线数据
    print(f"\n【步骤1】获取K线数据（最近{days}天）...")
    df_1m = t1.get_kline_data('SIL2603', '1min', count=days * 1440)
    if df_1m.empty:
        print("❌ 无法获取K线数据")
        return
    
    print(f"✅ 获取到 {len(df_1m)} 条1分钟K线数据")
    print(f"   时间范围: {df_1m.index[0]} 到 {df_1m.index[-1]}")
    print(f"   列: {list(df_1m.columns)}")
    
    # 2. 加载Tick数据
    print(f"\n【步骤2】加载真实Tick数据（从DEMO账户采集）...")
    tick_data = None
    tick_dir = '/home/cx/trading_data/ticks'
    if os.path.exists(tick_dir):
        tick_files = glob.glob(os.path.join(tick_dir, 'SIL2603_ticks_*.csv'))
        if tick_files:
            all_ticks = []
            for tick_file in sorted(tick_files):
                try:
                    df_ticks = pd.read_csv(tick_file)
                    if 'time' in df_ticks.columns:
                        df_ticks['datetime'] = pd.to_datetime(df_ticks['time'], unit='ms')
                    elif 'datetime' in df_ticks.columns:
                        df_ticks['datetime'] = pd.to_datetime(df_ticks['datetime'])
                    all_ticks.append(df_ticks)
                except Exception as e:
                    print(f"   ⚠️ 加载 {os.path.basename(tick_file)} 失败: {e}")
            
            if all_ticks:
                tick_data = pd.concat(all_ticks, ignore_index=True)
                tick_data = tick_data.sort_values('datetime').reset_index(drop=True)
                print(f"✅ 加载 {len(tick_data)} 条真实Tick数据（来自 {len(tick_files)} 个文件）")
                print(f"   Tick时间范围: {tick_data['datetime'].min()} 到 {tick_data['datetime'].max()}")
        else:
            print("⚠️ 未找到Tick数据文件")
    else:
        print("⚠️ Tick数据目录不存在")
    
    # 3. 计算技术指标
    print(f"\n【步骤3】计算技术指标...")
    indicators = calculate_technical_indicators(df_1m)
    print(f"✅ 计算完成")
    print(f"   指标: RSI, ATR, 布林带(上/中/下)")
    
    # 4. 展示对齐过程
    print(f"\n【步骤4】数据对齐过程（序列长度={seq_length}）...")
    print("-" * 80)
    
    # 选择一个示例时间点
    example_idx = 30  # 从第30个K线开始（确保有足够历史数据）
    if example_idx >= len(df_1m):
        example_idx = len(df_1m) - 1
    
    row = df_1m.iloc[example_idx]
    kline_time = row.name if hasattr(row, 'name') else pd.Timestamp(datetime.now())
    if isinstance(kline_time, str):
        kline_time = pd.to_datetime(kline_time)
    
    print(f"\n📌 示例时间点: {kline_time}")
    print(f"   K线价格: {row['close']:.4f}")
    
    # 4.1 Tick数据对齐
    if tick_data is not None and 'datetime' in tick_data.columns and 'price' in tick_data.columns:
        # 简化时区处理：统一转换为无时区的datetime
        try:
            # 移除K线时间的时区信息（如果有）
            if hasattr(kline_time, 'tz') and kline_time.tz is not None:
                kline_time_naive = kline_time.tz_localize(None) if hasattr(kline_time, 'tz_localize') else kline_time.replace(tzinfo=None)
            else:
                kline_time_naive = kline_time
            
            # 确保Tick数据也是无时区的
            if hasattr(tick_data['datetime'].dtype, 'tz') and tick_data['datetime'].dtype.tz is not None:
                tick_data_local = tick_data.copy()
                tick_data_local['datetime'] = tick_data_local['datetime'].dt.tz_localize(None)
            else:
                tick_data_local = tick_data
            
            time_window_start = kline_time_naive - pd.Timedelta(seconds=30)
            time_window_end = kline_time_naive + pd.Timedelta(seconds=30)
            
            mask = (tick_data_local['datetime'] >= time_window_start) & (tick_data_local['datetime'] <= time_window_end)
            ticks_in_window = tick_data_local[mask].copy()
        except Exception as e:
            print(f"   ⚠️ Tick数据对齐失败: {e}，使用K线价格")
            ticks_in_window = pd.DataFrame()
        
        if not ticks_in_window.empty:
            tick_price = ticks_in_window['price'].iloc[-1]
            tick_count = len(ticks_in_window)
            tick_volume = ticks_in_window['volume'].sum() if 'volume' in ticks_in_window.columns else tick_count
            
            print(f"\n   【Tick数据对齐】")
            print(f"   时间窗口: {time_window_start} 到 {time_window_end} (±30秒)")
            print(f"   找到 {tick_count} 条Tick数据")
            print(f"   最新Tick价格: {tick_price:.4f} (K线价格: {row['close']:.4f})")
            print(f"   Tick成交量: {tick_volume}")
            print(f"   价格差异: {(tick_price - row['close']) / row['close'] * 100:.4f}%")
            
            if len(ticks_in_window) > 1:
                tick_volatility = ticks_in_window['price'].std() / row['close'] if row['close'] > 0 else 0.0
                print(f"   Tick波动率: {tick_volatility:.6f}")
        else:
            print(f"\n   【Tick数据对齐】")
            print(f"   ⚠️ 该时间窗口内未找到Tick数据，使用K线价格")
    else:
        print(f"\n   【Tick数据对齐】")
        print(f"   ⚠️ Tick数据不可用，使用K线价格")
    
    # 4.2 技术指标对齐
    print(f"\n   【技术指标对齐】")
    rsi_1m = indicators['rsi'][example_idx] if not np.isnan(indicators['rsi'][example_idx]) else 50.0
    atr = indicators['atr'][example_idx] if not np.isnan(indicators['atr'][example_idx]) else 0.2
    boll_upper = indicators['boll_upper'][example_idx] if not np.isnan(indicators['boll_upper'][example_idx]) else row['close'] * 1.01
    boll_mid = indicators['boll_mid'][example_idx] if not np.isnan(indicators['boll_mid'][example_idx]) else row['close']
    boll_lower = indicators['boll_lower'][example_idx] if not np.isnan(indicators['boll_lower'][example_idx]) else row['close'] * 0.99
    
    print(f"   RSI(1m): {rsi_1m:.2f}")
    print(f"   ATR: {atr:.4f}")
    print(f"   布林带上轨: {boll_upper:.4f}")
    print(f"   布林带中轨: {boll_mid:.4f}")
    print(f"   布林带下轨: {boll_lower:.4f}")
    
    # 4.3 序列构建
    print(f"\n   【序列构建（序列长度={seq_length}）】")
    print(f"   当前时间点索引: {example_idx}")
    print(f"   序列起始索引: {max(0, example_idx - seq_length + 1)}")
    print(f"   序列结束索引: {example_idx}")
    print(f"   序列包含的时间点:")
    
    start_idx = max(0, example_idx - seq_length + 1)
    sequence_indices = list(range(start_idx, example_idx + 1))
    
    for i, idx in enumerate(sequence_indices):
        seq_row = df_1m.iloc[idx]
        seq_time = seq_row.name if hasattr(seq_row, 'name') else pd.Timestamp(datetime.now())
        if isinstance(seq_time, str):
            seq_time = pd.to_datetime(seq_time)
        
        # 获取该时间点的Tick数据
        tick_info = ""
        if tick_data is not None:
            time_window_start = seq_time - pd.Timedelta(seconds=30)
            time_window_end = seq_time + pd.Timedelta(seconds=30)
            mask = (tick_data['datetime'] >= time_window_start) & (tick_data['datetime'] <= time_window_end)
            ticks_in_window = tick_data[mask]
            if not ticks_in_window.empty:
                tick_price = ticks_in_window['price'].iloc[-1]
                tick_info = f" | Tick: {tick_price:.4f}"
        
        print(f"      [{i+1}/{len(sequence_indices)}] 索引{idx}: {seq_time} | K线: {seq_row['close']:.4f}{tick_info}")
    
    # 5. 特征向量构建
    print(f"\n【步骤5】特征向量构建（18维，包含真实Tick数据）...")
    print("-" * 80)
    
    feature_names = [
        'price_current',      # 0: K线价格
        'tick_price',         # 1: 真实Tick价格
        'tick_price_change', # 2: Tick价格变化
        'tick_volatility',   # 3: Tick波动率
        'tick_volume',       # 4: Tick成交量
        'tick_count',        # 5: Tick数量
        'tick_buy_sell_ratio', # 6: Tick买卖比例
        'atr',               # 7: 平均真实波幅
        'rsi_1m',           # 8: 1分钟RSI
        'rsi_5m',           # 9: 5分钟RSI
        'grid_lower',       # 10: 网格下轨
        'grid_upper',       # 11: 网格上轨
        'boll_upper',       # 12: 布林带上轨
        'boll_mid',         # 13: 布林带中轨
        'boll_lower',       # 14: 布林带下轨
        'boll_position',    # 15: 布林带位置
        'volatility',       # 16: 波动率
        'volume_1m'         # 17: 1分钟成交量
    ]
    
    # 计算示例时间点的特征
    example_features = []
    
    # K线价格
    example_features.append(row['close'])
    
    # Tick价格（如果可用）
    if tick_data is not None:
        try:
            # 简化时区处理
            if hasattr(kline_time, 'tz') and kline_time.tz is not None:
                kline_time_naive = kline_time.tz_localize(None) if hasattr(kline_time, 'tz_localize') else kline_time.replace(tzinfo=None)
            else:
                kline_time_naive = kline_time
            
            if hasattr(tick_data['datetime'].dtype, 'tz') and tick_data['datetime'].dtype.tz is not None:
                tick_data_local = tick_data.copy()
                tick_data_local['datetime'] = tick_data_local['datetime'].dt.tz_localize(None)
            else:
                tick_data_local = tick_data
            
            time_window_start = kline_time_naive - pd.Timedelta(seconds=30)
            time_window_end = kline_time_naive + pd.Timedelta(seconds=30)
            mask = (tick_data_local['datetime'] >= time_window_start) & (tick_data_local['datetime'] <= time_window_end)
            ticks_in_window = tick_data_local[mask]
        except Exception:
            ticks_in_window = pd.DataFrame()
        
        if not ticks_in_window.empty:
            tick_price = ticks_in_window['price'].iloc[-1]
            tick_price_change = (tick_price - row['close']) / row['close'] if row['close'] > 0 else 0.0
            tick_volatility = ticks_in_window['price'].std() / row['close'] if len(ticks_in_window) > 1 and row['close'] > 0 else 0.0
            tick_volume = ticks_in_window['volume'].sum() if 'volume' in ticks_in_window.columns else len(ticks_in_window)
            tick_count = len(ticks_in_window)
            tick_buy_sell_ratio = 0.5  # 简化处理
        else:
            tick_price = row['close']
            tick_price_change = 0.0
            tick_volatility = 0.0
            tick_volume = 0
            tick_count = 0
            tick_buy_sell_ratio = 0.5
    else:
        tick_price = row['close']
        tick_price_change = 0.0
        tick_volatility = 0.0
        tick_volume = 0
        tick_count = 0
        tick_buy_sell_ratio = 0.5
    
    example_features.extend([
        tick_price,
        tick_price_change,
        tick_volatility,
        tick_volume,
        tick_count,
        tick_buy_sell_ratio
    ])
    
    # 技术指标
    example_features.extend([
        atr,
        rsi_1m,
        rsi_1m,  # rsi_5m简化处理
        row['close'] * 0.99,  # grid_lower
        row['close'] * 1.01,  # grid_upper
        boll_upper,
        boll_mid,
        boll_lower,
        (row['close'] - boll_lower) / (boll_upper - boll_lower) if boll_upper != boll_lower else 0.5,
        0.01,  # volatility简化
        row['volume'] if 'volume' in row else 0
    ])
    
    print(f"\n   当前时间点（索引{example_idx}）的特征向量:")
    for i, (name, value) in enumerate(zip(feature_names, example_features)):
        print(f"      [{i:2d}] {name:20s}: {value:12.6f}")
    
    # 6. 序列数据形状
    print(f"\n【步骤6】最终训练数据形状...")
    print("-" * 80)
    print(f"   单个样本形状: ({seq_length}, {len(feature_names)})")
    print(f"   含义: {seq_length}个时间步 × {len(feature_names)}个特征")
    print(f"   总特征数: {seq_length * len(feature_names)}")
    
    # 7. 数据对齐总结
    print(f"\n【总结】数据对齐流程:")
    print("-" * 80)
    print("   1. K线数据（1分钟）: 基础时间序列")
    print("   2. Tick数据对齐: 每个K线时间点 ±30秒窗口内的Tick数据")
    print("   3. 技术指标计算: 基于K线数据计算RSI、ATR、布林带等")
    print("   4. 特征提取: 18维特征向量（包含真实Tick数据）")
    print("   5. 序列构建: 取最近seq_length个时间步的特征")
    print("   6. 标签生成: 基于未来10步的价格变化计算动作标签和收益率")
    
    print("\n" + "=" * 80)
    print("✅ 数据对齐可视化完成")
    print("=" * 80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='可视化训练数据对齐过程')
    parser.add_argument('--days', type=int, default=1, help='获取最近N天的数据')
    parser.add_argument('--seq-length', type=int, default=10, help='序列长度')
    
    args = parser.parse_args()
    
    visualize_data_alignment(days=args.days, seq_length=args.seq_length)
