#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从多时间尺度K线数据生成训练数据
包含1分钟、5分钟、1小时、日线数据
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


def align_multitimeframe_data(df_1m, df_5m, df_1h, df_1d):
    """
    对齐多时间尺度的K线数据
    
    Args:
        df_1m: 1分钟K线数据
        df_5m: 5分钟K线数据
        df_1h: 1小时K线数据
        df_1d: 日线K线数据
    
    Returns:
        aligned_data: 对齐后的数据框
    """
    print("📊 对齐多时间尺度数据...")
    
    # 以1分钟数据为基准
    aligned_data = []
    
    for i, (time_1m, row_1m) in enumerate(df_1m.iterrows()):
        # 确保time_1m是Timestamp
        if isinstance(time_1m, str):
            time_1m = pd.to_datetime(time_1m)
        
        # 对齐5分钟数据（向下取整到5分钟）
        time_5m = time_1m.floor('5min')
        row_5m = df_5m.loc[df_5m.index <= time_5m].iloc[-1] if len(df_5m.loc[df_5m.index <= time_5m]) > 0 else None
        
        # 对齐1小时数据（向下取整到小时）
        time_1h = time_1m.floor('H')
        row_1h = df_1h.loc[df_1h.index <= time_1h].iloc[-1] if len(df_1h.loc[df_1h.index <= time_1h]) > 0 else None
        
        # 对齐日线数据（向下取整到日）
        time_1d = time_1m.floor('D')
        row_1d = df_1d.loc[df_1d.index <= time_1d].iloc[-1] if len(df_1d.loc[df_1d.index <= time_1d]) > 0 else None
        
        # 构建对齐后的数据点
        data_point = {
            'timestamp': time_1m,
            'price_1m': row_1m['close'],
            'volume_1m': row_1m.get('volume', 0),
        }
        
        # 添加5分钟数据
        if row_5m is not None:
            data_point.update({
                'price_5m': row_5m['close'],
                'volume_5m': row_5m.get('volume', 0),
            })
        else:
            data_point.update({
                'price_5m': row_1m['close'],
                'volume_5m': 0,
            })
        
        # 添加1小时数据
        if row_1h is not None:
            data_point.update({
                'price_1h': row_1h['close'],
                'volume_1h': row_1h.get('volume', 0),
            })
        else:
            data_point.update({
                'price_1h': row_1m['close'],
                'volume_1h': 0,
            })
        
        # 添加日线数据
        if row_1d is not None:
            data_point.update({
                'price_1d': row_1d['close'],
                'volume_1d': row_1d.get('volume', 0),
            })
        else:
            data_point.update({
                'price_1d': row_1m['close'],
                'volume_1d': 0,
            })
        
        aligned_data.append(data_point)
    
    return pd.DataFrame(aligned_data).set_index('timestamp')


def calculate_multitimeframe_indicators(df_1m, df_5m, df_1h, df_1d):
    """
    计算多时间尺度的技术指标
    
    Returns:
        dict: 各时间尺度的指标
    """
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
    
    return indicators


def generate_multitimeframe_training_data(days=30, output_file=None):
    """
    从多时间尺度K线数据生成训练数据
    
    Args:
        days: 获取最近N天的数据
        output_file: 输出文件路径
    """
    print(f"🔄 开始从多时间尺度K线数据生成训练数据（最近{days}天）...")
    print("=" * 80)
    
    try:
        # 1. 获取多时间尺度K线数据
        print(f"\n【步骤1】获取多时间尺度K线数据...")
        # 请求更多数据以确保有足够的数据用于训练
        request_count_1m = max(days * 1440, 5000)  # 至少5000条
        request_count_5m = max(days * 288, 2000)  # 至少2000条
        request_count_1h = max(days * 24, 500)     # 至少500条
        request_count_1d = max(days, 100)          # 至少100条
        
        print(f"  获取1分钟数据（请求{request_count_1m}条）...")
        df_1m = t1.get_kline_data('SIL2603', '1min', count=request_count_1m)
        print(f"  获取5分钟数据（请求{request_count_5m}条）...")
        df_5m = t1.get_kline_data('SIL2603', '5min', count=request_count_5m)
        print(f"  获取1小时数据（请求{request_count_1h}条）...")
        df_1h = t1.get_kline_data('SIL2603', '1h', count=request_count_1h)
        print(f"  获取日线数据（请求{request_count_1d}条）...")
        df_1d = t1.get_kline_data('SIL2603', '1d', count=request_count_1d)
        
        if df_1m.empty:
            print("❌ 无法获取1分钟K线数据")
            return None
        
        print(f"✅ 获取到:")
        print(f"  1分钟: {len(df_1m)}条")
        print(f"  5分钟: {len(df_5m)}条")
        print(f"  1小时: {len(df_1h)}条")
        print(f"  日线: {len(df_1d)}条")
        
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
                    print(f"   Tick时间范围: {tick_data['datetime'].min()} 到 {tick_data['datetime'].max()}")
        
        # 3. 计算多时间尺度技术指标
        print(f"\n【步骤3】计算多时间尺度技术指标...")
        indicators = calculate_multitimeframe_indicators(df_1m, df_5m, df_1h, df_1d)
        
        # 4. 对齐多时间尺度数据
        print(f"\n【步骤4】对齐多时间尺度数据...")
        aligned_df = align_multitimeframe_data(df_1m, df_5m, df_1h, df_1d)
        print(f"✅ 对齐完成，共 {len(aligned_df)} 条数据")
        
        # 5. 构建训练数据（包含多时间尺度特征）
        print(f"\n【步骤5】构建训练数据（包含多时间尺度特征）...")
        training_data = []
        
        # 需要确保有足够的数据计算指标
        # 至少需要20个数据点来计算技术指标，但如果有更多数据就使用更多
        min_required = max(20, len(df_1d) * 2 if len(df_1d) > 0 else 20)
        
        # 如果数据量足够，使用所有可用数据（除了最后look_ahead个，用于生成标签）
        max_usable = len(df_1m) - 10  # 留出10个用于look_ahead
        
        print(f"   可用数据范围: 索引{min_required} 到 {max_usable}")
        print(f"   将生成 {max_usable - min_required} 条训练数据")
        
        for i in range(min_required, max_usable):
            row_1m = df_1m.iloc[i]
            kline_time = row_1m.name if hasattr(row_1m, 'name') else pd.Timestamp(datetime.now())
            if isinstance(kline_time, str):
                kline_time = pd.to_datetime(kline_time)
            
            # 获取Tick数据（与之前相同）
            tick_price = row_1m['close']
            tick_volume = 0
            tick_count = 0
            tick_price_change = 0.0
            tick_volatility = 0.0
            tick_buy_volume = 0
            tick_sell_volume = 0
            
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
                    
                    time_window_start = kline_time_naive - pd.Timedelta(seconds=30)
                    time_window_end = kline_time_naive + pd.Timedelta(seconds=30)
                    mask = (tick_data_local['datetime'] >= time_window_start) & (tick_data_local['datetime'] <= time_window_end)
                    ticks_in_window = tick_data_local[mask].copy()
                    
                    if not ticks_in_window.empty:
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
                        tick_count = len(ticks_in_window)
                        tick_price_change = (tick_price - row_1m['close']) / row_1m['close'] if row_1m['close'] > 0 else 0.0
                        if len(ticks_in_window) > 1:
                            tick_volatility = ticks_in_window['price'].std() / row_1m['close'] if row_1m['close'] > 0 else 0.0
                except Exception as e:
                    pass  # 如果Tick对齐失败，使用默认值
            
            # 获取多时间尺度的指标值
            # 1分钟指标
            rsi_1m = indicators['1m']['rsi'][i] if not np.isnan(indicators['1m']['rsi'][i]) else 50.0
            atr_1m = indicators['1m']['atr'][i] if not np.isnan(indicators['1m']['atr'][i]) else 0.2
            boll_upper_1m = indicators['1m']['boll_upper'][i] if not np.isnan(indicators['1m']['boll_upper'][i]) else row_1m['close'] * 1.01
            boll_mid_1m = indicators['1m']['boll_mid'][i] if not np.isnan(indicators['1m']['boll_mid'][i]) else row_1m['close']
            boll_lower_1m = indicators['1m']['boll_lower'][i] if not np.isnan(indicators['1m']['boll_lower'][i]) else row_1m['close'] * 0.99
            
            # 5分钟指标（需要找到对应的5分钟索引）
            rsi_5m = 50.0
            atr_5m = 0.2
            boll_upper_5m = row_1m['close'] * 1.01
            boll_mid_5m = row_1m['close']
            boll_lower_5m = row_1m['close'] * 0.99
            price_5m = row_1m['close']
            volume_5m = 0
            
            if '5m' in indicators and not df_5m.empty:
                time_5m = kline_time.floor('5min')
                idx_5m = df_5m.index.get_indexer([time_5m], method='nearest')[0]
                if idx_5m >= 0 and idx_5m < len(df_5m):
                    row_5m = df_5m.iloc[idx_5m]
                    price_5m = row_5m['close']
                    volume_5m = row_5m.get('volume', 0)
                    if idx_5m < len(indicators['5m']['rsi']):
                        rsi_5m = indicators['5m']['rsi'][idx_5m] if not np.isnan(indicators['5m']['rsi'][idx_5m]) else 50.0
                        atr_5m = indicators['5m']['atr'][idx_5m] if not np.isnan(indicators['5m']['atr'][idx_5m]) else 0.2
                        boll_upper_5m = indicators['5m']['boll_upper'][idx_5m] if not np.isnan(indicators['5m']['boll_upper'][idx_5m]) else price_5m * 1.01
                        boll_mid_5m = indicators['5m']['boll_mid'][idx_5m] if not np.isnan(indicators['5m']['boll_mid'][idx_5m]) else price_5m
                        boll_lower_5m = indicators['5m']['boll_lower'][idx_5m] if not np.isnan(indicators['5m']['boll_lower'][idx_5m]) else price_5m * 0.99
            
            # 1小时指标
            rsi_1h = 50.0
            atr_1h = 0.2
            boll_upper_1h = row_1m['close'] * 1.01
            boll_mid_1h = row_1m['close']
            boll_lower_1h = row_1m['close'] * 0.99
            price_1h = row_1m['close']
            volume_1h = 0
            trend_1h = 0.5  # 0=下跌, 0.5=横盘, 1=上涨
            
            if '1h' in indicators and not df_1h.empty:
                time_1h = kline_time.floor('H')
                idx_1h = df_1h.index.get_indexer([time_1h], method='nearest')[0]
                if idx_1h >= 0 and idx_1h < len(df_1h):
                    row_1h = df_1h.iloc[idx_1h]
                    price_1h = row_1h['close']
                    volume_1h = row_1h.get('volume', 0)
                    if idx_1h < len(indicators['1h']['rsi']):
                        rsi_1h = indicators['1h']['rsi'][idx_1h] if not np.isnan(indicators['1h']['rsi'][idx_1h]) else 50.0
                        atr_1h = indicators['1h']['atr'][idx_1h] if not np.isnan(indicators['1h']['atr'][idx_1h]) else 0.2
                        boll_upper_1h = indicators['1h']['boll_upper'][idx_1h] if not np.isnan(indicators['1h']['boll_upper'][idx_1h]) else price_1h * 1.01
                        boll_mid_1h = indicators['1h']['boll_mid'][idx_1h] if not np.isnan(indicators['1h']['boll_mid'][idx_1h]) else price_1h
                        boll_lower_1h = indicators['1h']['boll_lower'][idx_1h] if not np.isnan(indicators['1h']['boll_lower'][idx_1h]) else price_1h * 0.99
                        # 计算趋势（基于价格变化）
                        if idx_1h > 0:
                            prev_price_1h = df_1h.iloc[idx_1h-1]['close']
                            trend_1h = 1.0 if price_1h > prev_price_1h * 1.001 else (0.0 if price_1h < prev_price_1h * 0.999 else 0.5)
            
            # 日线指标
            rsi_1d = 50.0
            atr_1d = 0.2
            boll_upper_1d = row_1m['close'] * 1.01
            boll_mid_1d = row_1m['close']
            boll_lower_1d = row_1m['close'] * 0.99
            price_1d = row_1m['close']
            volume_1d = 0
            trend_1d = 0.5
            ma_5d = row_1m['close']
            ma_10d = row_1m['close']
            ma_20d = row_1m['close']
            
            if '1d' in indicators and not df_1d.empty:
                time_1d = kline_time.floor('D')
                idx_1d = df_1d.index.get_indexer([time_1d], method='nearest')[0]
                if idx_1d >= 0 and idx_1d < len(df_1d):
                    row_1d = df_1d.iloc[idx_1d]
                    price_1d = row_1d['close']
                    volume_1d = row_1d.get('volume', 0)
                    if idx_1d < len(indicators['1d']['rsi']):
                        rsi_1d = indicators['1d']['rsi'][idx_1d] if not np.isnan(indicators['1d']['rsi'][idx_1d]) else 50.0
                        atr_1d = indicators['1d']['atr'][idx_1d] if not np.isnan(indicators['1d']['atr'][idx_1d]) else 0.2
                        boll_upper_1d = indicators['1d']['boll_upper'][idx_1d] if not np.isnan(indicators['1d']['boll_upper'][idx_1d]) else price_1d * 1.01
                        boll_mid_1d = indicators['1d']['boll_mid'][idx_1d] if not np.isnan(indicators['1d']['boll_mid'][idx_1d]) else price_1d
                        boll_lower_1d = indicators['1d']['boll_lower'][idx_1d] if not np.isnan(indicators['1d']['boll_lower'][idx_1d]) else price_1d * 0.99
                        # 计算趋势
                        if idx_1d > 0:
                            prev_price_1d = df_1d.iloc[idx_1d-1]['close']
                            trend_1d = 1.0 if price_1d > prev_price_1d * 1.001 else (0.0 if price_1d < prev_price_1d * 0.999 else 0.5)
                        # 计算均线（需要历史数据）
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
            tick_buy_sell_ratio = 0.5
            if tick_buy_volume + tick_sell_volume > 0:
                tick_buy_sell_ratio = tick_buy_volume / (tick_buy_volume + tick_sell_volume)
            
            # 构建数据点（包含多时间尺度特征）
            data_point = {
                'timestamp': kline_time,
                'price_current': row_1m['close'],  # 1分钟价格（作为基准）
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
                # 网格参数（简化处理，使用1分钟布林带）
                'grid_lower': boll_lower_1m,
                'grid_upper': boll_upper_1m,
            }
            
            training_data.append(data_point)
        
        # 6. 保存训练数据
        if training_data:
            df_training = pd.DataFrame(training_data)
            
            if output_file is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_file = f'/home/cx/trading_data/training_data_multitimeframe_{timestamp}.csv'
            
            df_training.to_csv(output_file, index=False)
            print(f"\n✅ 训练数据已保存: {output_file}")
            print(f"   总数据量: {len(df_training)}条")
            print(f"   特征维度: {len(df_training.columns)}维（包含多时间尺度特征）")
            print(f"   特征列表: {list(df_training.columns)}")
            
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
    
    parser = argparse.ArgumentParser(description='从多时间尺度K线数据生成训练数据')
    parser.add_argument('--days', type=int, default=30, help='获取最近N天的数据')
    parser.add_argument('--output', type=str, default=None, help='输出文件路径')
    
    args = parser.parse_args()
    
    generate_multitimeframe_training_data(days=args.days, output_file=args.output)
