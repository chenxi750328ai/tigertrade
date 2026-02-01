#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
直接通过API获取多时间尺度K线数据并生成训练数据
使用所有获取到的数据，不限制数量
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


def calculate_multitimeframe_indicators(df_1m, df_5m, df_1h, df_1d, df_1w=None, df_1M=None):
    """计算多时间尺度的技术指标"""
    print("📊 计算多时间尺度技术指标...")
    
    indicators = {}
    
    # 1分钟指标
    indicators['1m'] = calculate_technical_indicators(df_1m)
    print(f"  ✅ 1分钟指标计算完成")
    
    # 5分钟指标
    if not df_5m.empty:
        indicators['5m'] = calculate_technical_indicators(df_5m)
        print(f"  ✅ 5分钟指标计算完成")
    
    # 1小时指标
    if not df_1h.empty:
        indicators['1h'] = calculate_technical_indicators(df_1h)
        print(f"  ✅ 1小时指标计算完成")
    
    # 日线指标
    if not df_1d.empty:
        indicators['1d'] = calculate_technical_indicators(df_1d)
        print(f"  ✅ 日线指标计算完成")
    
    # 周线指标
    if df_1w is not None and not df_1w.empty:
        indicators['1w'] = calculate_technical_indicators(df_1w)
        print(f"  ✅ 周线指标计算完成")
    
    # 月线指标
    if df_1M is not None and not df_1M.empty:
        indicators['1M'] = calculate_technical_indicators(df_1M)
        print(f"  ✅ 月线指标计算完成")
    
    return indicators


def generate_training_data_direct(count_1m=10000, count_5m=2000, count_1h=500, count_1d=100, count_1w=50, count_1M=12, output_file=None):
    """
    直接通过API获取数据并生成训练数据
    
    Args:
        count_1m: 1分钟数据数量
        count_5m: 5分钟数据数量
        count_1h: 1小时数据数量
        count_1d: 日线数据数量
        output_file: 输出文件路径
    """
    print(f"🔄 通过API获取多时间尺度K线数据并生成训练数据...")
    print("=" * 80)
    
    try:
        # 1. 获取多时间尺度K线数据
        print(f"\n【步骤1】获取多时间尺度K线数据...")
        print(f"  获取1分钟数据（请求{count_1m}条）...")
        df_1m = t1.get_kline_data('SIL2603', '1min', count=count_1m)
        print(f"  获取5分钟数据（请求{count_5m}条）...")
        df_5m = t1.get_kline_data('SIL2603', '5min', count=count_5m)
        print(f"  获取1小时数据（请求{count_1h}条）...")
        df_1h = t1.get_kline_data('SIL2603', '1h', count=count_1h)
        print(f"  获取日线数据（请求{count_1d}条）...")
        df_1d = t1.get_kline_data('SIL2603', '1d', count=count_1d)
        print(f"  获取周线数据（请求{count_1w}条）...")
        df_1w = t1.get_kline_data('SIL2603', '1w', count=count_1w)
        print(f"  获取月线数据（请求{count_1M}条）...")
        df_1M = t1.get_kline_data('SIL2603', '1M', count=count_1M)
        
        if df_1m.empty:
            print("❌ 无法获取1分钟K线数据")
            return None
        
        print(f"✅ 获取到:")
        print(f"  1分钟: {len(df_1m)}条")
        print(f"  5分钟: {len(df_5m)}条")
        print(f"  1小时: {len(df_1h)}条")
        print(f"  日线: {len(df_1d)}条")
        print(f"  周线: {len(df_1w)}条")
        print(f"  月线: {len(df_1M)}条")
        
        if len(df_1m) < 100:
            print("⚠️ 1分钟数据量不足（少于100条），可能影响训练效果")
        
        # 2. 加载真实Tick数据
        print(f"\n【步骤2】加载真实Tick数据（从DEMO账户采集）...")
        tick_data = None
        tick_dir = '/home/cx/trading_data/ticks'
        if os.path.exists(tick_dir):
            tick_files = glob.glob(os.path.join(tick_dir, 'SIL2603_ticks_*.csv'))
            if tick_files:
                print(f"📁 找到 {len(tick_files)} 个Tick数据文件")
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
                        print(f"  ⚠️ 加载 {os.path.basename(tick_file)} 失败: {e}")
                
                if all_ticks:
                    tick_data = pd.concat(all_ticks, ignore_index=True)
                    tick_data = tick_data.sort_values('datetime').reset_index(drop=True)
                    tick_data = tick_data.drop_duplicates(subset=['datetime', 'price'], keep='last')
                    print(f"✅ 加载 {len(tick_data)} 条真实Tick数据")
                    if len(tick_data) > 0:
                        print(f"   Tick时间范围: {tick_data['datetime'].min()} 到 {tick_data['datetime'].max()}")
        
        # 3. 计算多时间尺度技术指标
        print(f"\n【步骤3】计算多时间尺度技术指标...")
        indicators = calculate_multitimeframe_indicators(df_1m, df_5m, df_1h, df_1d, df_1w, df_1M)
        
        # 4. 构建训练数据（使用所有可用数据）
        print(f"\n【步骤4】构建训练数据（包含多时间尺度特征）...")
        training_data = []
        
        # 需要确保有足够的数据计算指标
        min_required = max(20, len(df_1d) * 2 if len(df_1d) > 0 else 20)
        # 使用所有可用数据（除了最后10个用于look_ahead）
        max_usable = len(df_1m) - 10
        
        print(f"   可用数据范围: 索引{min_required} 到 {max_usable}")
        print(f"   将生成 {max_usable - min_required} 条训练数据")
        
        if max_usable <= min_required:
            print(f"❌ 数据不足，无法生成训练数据（需要至少{min_required + 10}条，实际{len(df_1m)}条）")
            return None
        
        for i in range(min_required, max_usable):
            row_1m = df_1m.iloc[i]
            kline_time = row_1m.name if hasattr(row_1m, 'name') else pd.Timestamp(datetime.now())
            if isinstance(kline_time, str):
                kline_time = pd.to_datetime(kline_time)
            
            # 获取Tick数据
            # 改进：使用NaN表示无效值，而不是0
            tick_price = row_1m['close']  # 默认使用K线收盘价
            tick_volume = np.nan
            tick_count = np.nan
            tick_price_change = np.nan
            tick_volatility = np.nan
            tick_buy_volume = np.nan
            tick_sell_volume = np.nan
            tick_data_valid = False  # 标记Tick数据是否有效
            
            if tick_data is not None and 'datetime' in tick_data.columns and 'price' in tick_data.columns:
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
                    
                    # 改进：扩大时间窗口，并尝试最近邻匹配
                    time_window_start = kline_time_naive - pd.Timedelta(seconds=120)  # 扩大到120秒
                    time_window_end = kline_time_naive + pd.Timedelta(seconds=120)   # 扩大到120秒
                    mask = (tick_data_local['datetime'] >= time_window_start) & (tick_data_local['datetime'] <= time_window_end)
                    ticks_in_window = tick_data_local[mask].copy()
                    
                    # 如果窗口内没有Tick，尝试找最近的Tick（在5分钟内）
                    if ticks_in_window.empty:
                        time_window_large_start = kline_time_naive - pd.Timedelta(minutes=5)
                        time_window_large_end = kline_time_naive + pd.Timedelta(minutes=5)
                        mask_large = (tick_data_local['datetime'] >= time_window_large_start) & (tick_data_local['datetime'] <= time_window_large_end)
                        ticks_nearby = tick_data_local[mask_large].copy()
                        if not ticks_nearby.empty:
                            # 找最近的Tick
                            nearest_idx = (ticks_nearby['datetime'] - kline_time_naive).abs().idxmin()
                            ticks_in_window = ticks_nearby.loc[[nearest_idx]]
                    
                    if not ticks_in_window.empty:
                        tick_data_valid = True
                        tick_price = ticks_in_window['price'].iloc[-1]
                        if 'volume' in ticks_in_window.columns:
                            tick_volume = ticks_in_window['volume'].sum()
                            if 'side' in ticks_in_window.columns:
                                buy_ticks = ticks_in_window[ticks_in_window['side'] == 'BUY']
                                sell_ticks = ticks_in_window[ticks_in_window['side'] == 'SELL']
                                tick_buy_volume = buy_ticks['volume'].sum() if not buy_ticks.empty else 0
                                tick_sell_volume = sell_ticks['volume'].sum() if not sell_ticks.empty else 0
                            else:
                                tick_volume = len(ticks_in_window)
                        else:
                            tick_volume = len(ticks_in_window)
                        tick_count = len(ticks_in_window)
                        tick_price_change = (tick_price - row_1m['close']) / row_1m['close'] if row_1m['close'] > 0 else np.nan
                        if len(ticks_in_window) > 1:
                            tick_volatility = ticks_in_window['price'].std() / row_1m['close'] if row_1m['close'] > 0 else np.nan
                        else:
                            # 单条Tick，使用价格变化作为波动率估计
                            tick_volatility = abs(tick_price_change) if not np.isnan(tick_price_change) else np.nan
                except Exception as e:
                    # 改进：记录Tick对齐失败的情况，而不是静默处理
                    if i % 1000 == 0:  # 每1000条记录一次，避免日志过多
                        print(f"⚠️ Tick数据对齐失败 (索引{i}): {e}")
                    # 保持NaN值，但记录失败次数
                    if not hasattr(generate_training_data_direct, '_tick_alignment_failures'):
                        generate_training_data_direct._tick_alignment_failures = 0
                    generate_training_data_direct._tick_alignment_failures += 1
            
            # 如果没有匹配到Tick数据，所有Tick特征保持为NaN（无效值）
            if not tick_data_valid:
                # tick_price使用K线收盘价（这是合理的默认值）
                # 其他Tick特征保持为NaN
                pass
            
            # 获取多时间尺度的指标值（与之前相同的逻辑）
            # ... (这里省略详细代码，使用与generate_multitimeframe_training_data相同的逻辑)
            # 为了简化，这里直接调用原函数的核心逻辑
            
            # 1分钟指标
            rsi_1m = indicators['1m']['rsi'][i] if i < len(indicators['1m']['rsi']) and not np.isnan(indicators['1m']['rsi'][i]) else 50.0
            atr_1m = indicators['1m']['atr'][i] if i < len(indicators['1m']['atr']) and not np.isnan(indicators['1m']['atr'][i]) else 0.2
            boll_upper_1m = indicators['1m']['boll_upper'][i] if i < len(indicators['1m']['boll_upper']) and not np.isnan(indicators['1m']['boll_upper'][i]) else row_1m['close'] * 1.01
            boll_mid_1m = indicators['1m']['boll_mid'][i] if i < len(indicators['1m']['boll_mid']) and not np.isnan(indicators['1m']['boll_mid'][i]) else row_1m['close']
            boll_lower_1m = indicators['1m']['boll_lower'][i] if i < len(indicators['1m']['boll_lower']) and not np.isnan(indicators['1m']['boll_lower'][i]) else row_1m['close'] * 0.99
            
            # 5分钟指标
            time_5m = kline_time.floor('5min')
            idx_5m = df_5m.index.get_indexer([time_5m], method='nearest')[0] if len(df_5m) > 0 else -1
            price_5m = row_1m['close']
            rsi_5m = 50.0
            atr_5m = atr_1m
            boll_upper_5m = boll_upper_1m
            boll_mid_5m = boll_mid_1m
            boll_lower_5m = boll_lower_1m
            volume_5m = 0
            
            if idx_5m >= 0 and idx_5m < len(df_5m) and '5m' in indicators:
                row_5m = df_5m.iloc[idx_5m]
                price_5m = row_5m['close']
                volume_5m = row_5m.get('volume', 0)
                if idx_5m < len(indicators['5m']['rsi']):
                    rsi_5m = indicators['5m']['rsi'][idx_5m] if not np.isnan(indicators['5m']['rsi'][idx_5m]) else 50.0
                    atr_5m = indicators['5m']['atr'][idx_5m] if not np.isnan(indicators['5m']['atr'][idx_5m]) else atr_1m
                    boll_upper_5m = indicators['5m']['boll_upper'][idx_5m] if not np.isnan(indicators['5m']['boll_upper'][idx_5m]) else price_5m * 1.01
                    boll_mid_5m = indicators['5m']['boll_mid'][idx_5m] if not np.isnan(indicators['5m']['boll_mid'][idx_5m]) else price_5m
                    boll_lower_5m = indicators['5m']['boll_lower'][idx_5m] if not np.isnan(indicators['5m']['boll_lower'][idx_5m]) else price_5m * 0.99
            
            # 1小时指标
            time_1h = kline_time.floor('H')
            idx_1h = df_1h.index.get_indexer([time_1h], method='nearest')[0] if len(df_1h) > 0 else -1
            price_1h = row_1m['close']
            rsi_1h = 50.0
            atr_1h = atr_1m
            boll_upper_1h = boll_upper_1m
            boll_mid_1h = boll_mid_1m
            boll_lower_1h = boll_lower_1m
            volume_1h = 0
            trend_1h = 0.5
            
            if idx_1h >= 0 and idx_1h < len(df_1h) and '1h' in indicators:
                row_1h = df_1h.iloc[idx_1h]
                price_1h = row_1h['close']
                volume_1h = row_1h.get('volume', 0)
                if idx_1h < len(indicators['1h']['rsi']):
                    rsi_1h = indicators['1h']['rsi'][idx_1h] if not np.isnan(indicators['1h']['rsi'][idx_1h]) else 50.0
                    atr_1h = indicators['1h']['atr'][idx_1h] if not np.isnan(indicators['1h']['atr'][idx_1h]) else atr_1m
                    boll_upper_1h = indicators['1h']['boll_upper'][idx_1h] if not np.isnan(indicators['1h']['boll_upper'][idx_1h]) else price_1h * 1.01
                    boll_mid_1h = indicators['1h']['boll_mid'][idx_1h] if not np.isnan(indicators['1h']['boll_mid'][idx_1h]) else price_1h
                    boll_lower_1h = indicators['1h']['boll_lower'][idx_1h] if not np.isnan(indicators['1h']['boll_lower'][idx_1h]) else price_1h * 0.99
                    if idx_1h > 0:
                        prev_price_1h = df_1h.iloc[idx_1h-1]['close']
                        trend_1h = 1.0 if price_1h > prev_price_1h * 1.001 else (0.0 if price_1h < prev_price_1h * 0.999 else 0.5)
            
            # 日线指标
            time_1d = kline_time.floor('D')
            idx_1d = df_1d.index.get_indexer([time_1d], method='nearest')[0] if len(df_1d) > 0 else -1
            price_1d = row_1m['close']
            rsi_1d = 50.0
            atr_1d = atr_1m
            boll_upper_1d = boll_upper_1m
            boll_mid_1d = boll_mid_1m
            boll_lower_1d = boll_lower_1m
            volume_1d = 0
            trend_1d = 0.5
            ma_5d = price_1d
            ma_10d = price_1d
            ma_20d = price_1d
            
            if idx_1d >= 0 and idx_1d < len(df_1d) and '1d' in indicators:
                row_1d = df_1d.iloc[idx_1d]
                price_1d = row_1d['close']
                volume_1d = row_1d.get('volume', 0)
                if idx_1d < len(indicators['1d']['rsi']):
                    rsi_1d = indicators['1d']['rsi'][idx_1d] if not np.isnan(indicators['1d']['rsi'][idx_1d]) else 50.0
                    atr_1d = indicators['1d']['atr'][idx_1d] if not np.isnan(indicators['1d']['atr'][idx_1d]) else atr_1m
                    boll_upper_1d = indicators['1d']['boll_upper'][idx_1d] if not np.isnan(indicators['1d']['boll_upper'][idx_1d]) else price_1d * 1.01
                    boll_mid_1d = indicators['1d']['boll_mid'][idx_1d] if not np.isnan(indicators['1d']['boll_mid'][idx_1d]) else price_1d
                    boll_lower_1d = indicators['1d']['boll_lower'][idx_1d] if not np.isnan(indicators['1d']['boll_lower'][idx_1d]) else price_1d * 0.99
                    if idx_1d > 0:
                        prev_price_1d = df_1d.iloc[idx_1d-1]['close']
                        trend_1d = 1.0 if price_1d > prev_price_1d * 1.001 else (0.0 if price_1d < prev_price_1d * 0.999 else 0.5)
                    if idx_1d >= 4:
                        ma_5d = df_1d.iloc[idx_1d-4:idx_1d+1]['close'].mean()
                    if idx_1d >= 9:
                        ma_10d = df_1d.iloc[idx_1d-9:idx_1d+1]['close'].mean()
                    if idx_1d >= 19:
                        ma_20d = df_1d.iloc[idx_1d-19:idx_1d+1]['close'].mean()
            
            # 计算布林带位置
            boll_position_1m = (row_1m['close'] - boll_lower_1m) / (boll_upper_1m - boll_lower_1m) if boll_upper_1m != boll_lower_1m else 0.5
            boll_position_5m = (price_5m - boll_lower_5m) / (boll_upper_5m - boll_lower_5m) if boll_upper_5m != boll_lower_5m else 0.5
            boll_position_1h = (price_1h - boll_lower_1h) / (boll_upper_1h - boll_lower_1h) if boll_upper_1h != boll_lower_1h else 0.5
            boll_position_1d = (price_1d - boll_lower_1d) / (boll_upper_1d - boll_lower_1d) if boll_upper_1d != boll_lower_1d else 0.5
            
            # 计算波动率
            volatility_1m = 0.0
            if i >= 20:
                recent_returns = df_1m.iloc[i-19:i+1]['close'].pct_change().dropna()
                volatility_1m = recent_returns.std() if len(recent_returns) > 0 else 0.0
            
            # 计算Tick买卖比例
            tick_buy_sell_ratio = np.nan  # 改进：使用NaN表示无效值，而不是0.5
            if tick_buy_volume + tick_sell_volume > 0:
                tick_buy_sell_ratio = tick_buy_volume / (tick_buy_volume + tick_sell_volume)
            
            # 构建数据点（46维特征）
            data_point = {
                'timestamp': kline_time,
                'price_current': row_1m['close'],
                # Tick特征
                'tick_price': tick_price,
                'tick_price_change': tick_price_change,
                'tick_volatility': tick_volatility,
                'tick_volume': tick_volume,
                'tick_count': tick_count,
                'tick_buy_sell_ratio': tick_buy_sell_ratio,
                # 1分钟特征
                'atr_1m': atr_1m,
                'rsi_1m': rsi_1m,
                'boll_upper_1m': boll_upper_1m,
                'boll_mid_1m': boll_mid_1m,
                'boll_lower_1m': boll_lower_1m,
                'boll_position_1m': boll_position_1m,
                'volatility_1m': volatility_1m,
                'volume_1m': row_1m.get('volume', 0),
                # 5分钟特征
                'price_5m': price_5m,
                'rsi_5m': rsi_5m,
                'atr_5m': atr_5m,
                'boll_upper_5m': boll_upper_5m,
                'boll_mid_5m': boll_mid_5m,
                'boll_lower_5m': boll_lower_5m,
                'boll_position_5m': boll_position_5m,
                'volume_5m': volume_5m,
                # 1小时特征
                'price_1h': price_1h,
                'rsi_1h': rsi_1h,
                'atr_1h': atr_1h,
                'boll_upper_1h': boll_upper_1h,
                'boll_mid_1h': boll_mid_1h,
                'boll_lower_1h': boll_lower_1h,
                'boll_position_1h': boll_position_1h,
                'volume_1h': volume_1h,
                'trend_1h': trend_1h,
                # 日线特征
                'price_1d': price_1d,
                'rsi_1d': rsi_1d,
                'atr_1d': atr_1d,
                'boll_upper_1d': boll_upper_1d,
                'boll_mid_1d': boll_mid_1d,
                'boll_lower_1d': boll_lower_1d,
                'boll_position_1d': boll_position_1d,
                'volume_1d': volume_1d,
                'trend_1d': trend_1d,
                'ma_5d': ma_5d,
                'ma_10d': ma_10d,
                'ma_20d': ma_20d,
                # 网格参数
                'grid_lower': boll_lower_1m,
                'grid_upper': boll_upper_1m,
            }
            
            training_data.append(data_point)
        
        # 5. 保存训练数据
        if training_data:
            df_training = pd.DataFrame(training_data)
            
            if output_file is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_file = f'/home/cx/trading_data/training_data_multitimeframe_{timestamp}.csv'
            
            df_training.to_csv(output_file, index=False)
            print(f"\n✅ 训练数据已保存: {output_file}")
            print(f"   总数据量: {len(df_training)}条")
            print(f"   特征维度: {len(df_training.columns)}维（包含多时间尺度特征）")
            
            return df_training
        else:
            print("❌ 未生成训练数据")
            return None
            
    except Exception as e:
        print(f"❌ 生成训练数据失败: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='直接通过API获取多时间尺度K线数据并生成训练数据')
    parser.add_argument('--count-1m', type=int, default=10000, help='1分钟数据数量')
    parser.add_argument('--count-5m', type=int, default=2000, help='5分钟数据数量')
    parser.add_argument('--count-1h', type=int, default=500, help='1小时数据数量')
    parser.add_argument('--count-1d', type=int, default=100, help='日线数据数量')
    parser.add_argument('--count-1w', type=int, default=50, help='周线数据数量')
    parser.add_argument('--count-1M', type=int, default=12, help='月线数据数量')
    parser.add_argument('--output', type=str, default=None, help='输出文件路径')
    
    args = parser.parse_args()
    
    generate_training_data_direct(
        count_1m=args.count_1m,
        count_5m=args.count_5m,
        count_1h=args.count_1h,
        count_1d=args.count_1d,
        count_1w=args.count_1w,
        count_1M=args.count_1M,
        output_file=args.output
    )
