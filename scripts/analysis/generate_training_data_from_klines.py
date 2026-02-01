#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从历史K线数据生成训练数据
用于序列长度测试
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
except ImportError:
    print("⚠️ 无法导入tiger1模块")


def calculate_technical_indicators(df):
    """计算技术指标"""
    import talib
    
    # 计算RSI
    rsi = talib.RSI(df['close'].values, timeperiod=14)
    
    # 计算ATR
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    atr = talib.ATR(high, low, close, timeperiod=14)
    
    # 计算布林带
    upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    
    return {
        'rsi': rsi,
        'atr': atr,
        'boll_upper': upper,
        'boll_mid': middle,
        'boll_lower': lower
    }


def generate_training_data_from_klines(days=30, output_file=None):
    """
    从历史K线数据生成训练数据（包含Tick数据）
    
    Args:
        days: 获取最近N天的数据
        output_file: 输出文件路径
    """
    print(f"🔄 开始从历史K线数据生成训练数据（最近{days}天，包含Tick数据）...")
    
    try:
        # 获取1分钟K线数据
        df_1m = t1.get_kline_data('SIL2603', '1min', count=days * 1440)
        
        if df_1m.empty:
            print("❌ 无法获取K线数据")
            return None
        
        print(f"✅ 获取到{len(df_1m)}条1分钟K线数据")
        
        # 加载真实的Tick数据（从DEMO账户采集器保存的文件）
        # 重要：Tick数据是从DEMO账户通过tick_data_collector.py真实获取的，不是伪造的！
        tick_data = None
        tick_files = []
        tick_dir = '/home/cx/trading_data/ticks'
        if os.path.exists(tick_dir):
            # 查找所有Tick数据文件（由tick_data_collector.py从DEMO账户采集保存）
            tick_files = glob.glob(os.path.join(tick_dir, 'SIL2603_ticks_*.csv'))
            if tick_files:
                print(f"📁 找到 {len(tick_files)} 个Tick数据文件（从DEMO账户真实采集）")
                # 合并所有Tick文件（按时间排序）
                all_ticks = []
                for tick_file in sorted(tick_files):  # 按文件名排序
                    try:
                        df_ticks = pd.read_csv(tick_file)
                        print(f"   📄 加载 {os.path.basename(tick_file)}: {len(df_ticks)}条")
                        
                        # 处理时间列
                        if 'time' in df_ticks.columns or 'datetime' in df_ticks.columns:
                            time_col = 'time' if 'time' in df_ticks.columns else 'datetime'
                            if time_col == 'time' and df_ticks[time_col].dtype in [np.int64, np.float64]:
                                # 如果是时间戳（毫秒），转换为datetime
                                df_ticks['datetime'] = pd.to_datetime(df_ticks[time_col], unit='ms')
                            else:
                                df_ticks['datetime'] = pd.to_datetime(df_ticks[time_col])
                            
                            # 确保有price列（Tick数据的关键列，从DEMO账户真实获取）
                            if 'price' in df_ticks.columns:
                                all_ticks.append(df_ticks)
                            else:
                                print(f"   ⚠️ {os.path.basename(tick_file)} 缺少price列，跳过")
                    except Exception as e:
                        print(f"   ❌ 加载Tick文件 {os.path.basename(tick_file)} 失败: {e}")
                
                if all_ticks:
                    tick_data = pd.concat(all_ticks, ignore_index=True)
                    tick_data = tick_data.sort_values('datetime').reset_index(drop=True)
                    # 去重（避免重复数据）
                    tick_data = tick_data.drop_duplicates(subset=['datetime', 'price'], keep='last')
                    print(f"✅ 加载真实Tick数据（从DEMO账户采集）: 总计 {len(tick_data)}条")
                    print(f"   Tick时间范围: {tick_data['datetime'].min()} 到 {tick_data['datetime'].max()}")
                    print(f"   价格范围: {tick_data['price'].min():.2f} 到 {tick_data['price'].max():.2f}")
                else:
                    print("⚠️ 所有Tick文件都无法加载，将使用K线价格作为Tick价格")
            else:
                print("⚠️ 未找到Tick数据文件（需要先运行tick_data_collector.py从DEMO账户采集）")
                print("   将使用K线价格作为Tick价格（这不是真实Tick数据！）")
        else:
            print("⚠️ Tick数据目录不存在，将使用K线价格作为Tick价格（这不是真实Tick数据！）")
        
        # 计算技术指标
        print("📊 计算技术指标...")
        indicators = calculate_technical_indicators(df_1m)
        
        # 构建训练数据
        training_data = []
        
        for i in range(20, len(df_1m)):  # 从第20条开始（确保有足够数据计算指标）
            row = df_1m.iloc[i]
            
            # 获取K线时间（用于匹配Tick数据）
            kline_time = row.name if hasattr(row, 'name') else pd.Timestamp(datetime.now())
            if isinstance(kline_time, str):
                kline_time = pd.to_datetime(kline_time)
            elif hasattr(kline_time, 'to_pydatetime'):
                kline_time = kline_time.to_pydatetime()
            
            # 获取真实的Tick价格和特征（如果可用）
            tick_price = row['close']  # 默认使用K线收盘价
            tick_volume = 0
            tick_count = 0
            tick_price_change = 0.0  # Tick价格相对于K线价格的变化
            tick_volatility = 0.0  # Tick价格在该K线周期内的波动率
            tick_buy_volume = 0  # 买入成交量
            tick_sell_volume = 0  # 卖出成交量
            
            if tick_data is not None and 'datetime' in tick_data.columns and 'price' in tick_data.columns:
                # 找到该K线时间范围内的Tick数据
                # 1分钟K线通常包含该分钟内的所有Tick
                time_window_start = kline_time - pd.Timedelta(seconds=30)  # K线时间前30秒
                time_window_end = kline_time + pd.Timedelta(seconds=30)   # K线时间后30秒
                
                # 筛选该时间窗口内的Tick数据
                mask = (tick_data['datetime'] >= time_window_start) & (tick_data['datetime'] <= time_window_end)
                ticks_in_window = tick_data[mask].copy()
                
                if not ticks_in_window.empty:
                    # 使用最新的Tick价格（最接近K线时间的Tick）
                    tick_price = ticks_in_window['price'].iloc[-1]
                    
                    # Tick成交量统计
                    if 'volume' in ticks_in_window.columns:
                        tick_volume = ticks_in_window['volume'].sum()
                        # 如果有side列，分别统计买卖成交量
                        if 'side' in ticks_in_window.columns:
                            buy_ticks = ticks_in_window[ticks_in_window['side'] == 'BUY']
                            sell_ticks = ticks_in_window[ticks_in_window['side'] == 'SELL']
                            tick_buy_volume = buy_ticks['volume'].sum() if not buy_ticks.empty else 0
                            tick_sell_volume = sell_ticks['volume'].sum() if not sell_ticks.empty else 0
                    else:
                        tick_volume = len(ticks_in_window)  # 如果没有volume列，用Tick数量代替
                    
                    tick_count = len(ticks_in_window)
                    
                    # 计算Tick价格变化（相对于K线价格）
                    tick_price_change = (tick_price - row['close']) / row['close'] if row['close'] > 0 else 0.0
                    
                    # 计算Tick波动率（该窗口内Tick价格的标准差，归一化）
                    if len(ticks_in_window) > 1:
                        tick_volatility = ticks_in_window['price'].std() / row['close'] if row['close'] > 0 else 0.0
            
            # 获取指标值
            rsi_1m = indicators['rsi'][i] if not np.isnan(indicators['rsi'][i]) else 50.0
            atr = indicators['atr'][i] if not np.isnan(indicators['atr'][i]) else 0.2
            boll_upper = indicators['boll_upper'][i] if not np.isnan(indicators['boll_upper'][i]) else row['close'] * 1.01
            boll_mid = indicators['boll_mid'][i] if not np.isnan(indicators['boll_mid'][i]) else row['close']
            boll_lower = indicators['boll_lower'][i] if not np.isnan(indicators['boll_lower'][i]) else row['close'] * 0.99
            
            # 计算价格变化
            if i > 0:
                price_change_1 = (row['close'] - df_1m.iloc[i-1]['close']) / df_1m.iloc[i-1]['close']
            else:
                price_change_1 = 0.0
            
            if i >= 5:
                price_change_5 = (row['close'] - df_1m.iloc[i-5]['close']) / df_1m.iloc[i-5]['close']
            else:
                price_change_5 = 0.0
            
            # 计算波动率（最近20个周期的标准差）
            if i >= 20:
                recent_returns = df_1m.iloc[i-19:i+1]['close'].pct_change().dropna()
                volatility = recent_returns.std() if len(recent_returns) > 0 else 0.0
            else:
                volatility = 0.0
            
            # 计算布林带位置
            if boll_upper != boll_lower:
                boll_position = (row['close'] - boll_lower) / (boll_upper - boll_lower)
            else:
                boll_position = 0.5
            
            # 构建数据点（包含真实的Tick数据）
            data_point = {
                'timestamp': row.name if hasattr(row, 'name') else datetime.now(),
                'price_current': row['close'],  # K线价格
                'tick_price': tick_price,  # 真实Tick价格（重要！）
                'tick_price_change': tick_price_change,  # Tick价格相对于K线价格的变化
                'tick_volatility': tick_volatility,  # Tick价格波动率
                'tick_volume': tick_volume,  # Tick成交量
                'tick_count': tick_count,  # Tick数量
                'tick_buy_volume': tick_buy_volume,  # 买入Tick成交量
                'tick_sell_volume': tick_sell_volume,  # 卖出Tick成交量
                'grid_lower': boll_lower,
                'grid_upper': boll_upper,
                'atr': atr,
                'rsi_1m': rsi_1m,
                'rsi_5m': rsi_1m,  # 简化处理，使用1分钟RSI
                'boll_upper': boll_upper,
                'boll_mid': boll_mid,
                'boll_lower': boll_lower,
                'boll_position': boll_position,
                'price_change_1': price_change_1,
                'price_change_5': price_change_5,
                'volatility': volatility,
                'volume_1m': row['volume'] if 'volume' in row else 0
            }
            
            training_data.append(data_point)
        
        # 转换为DataFrame
        df_training = pd.DataFrame(training_data)
        
        print(f"✅ 生成训练数据: {len(df_training)}条")
        
        # 保存文件
        if output_file is None:
            output_file = f'/home/cx/trading_data/training_data_from_klines_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        
        df_training.to_csv(output_file, index=False)
        print(f"💾 训练数据已保存到: {output_file}")
        
        return df_training, output_file
        
    except Exception as e:
        print(f"❌ 生成训练数据失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def main():
    """主函数"""
    # 生成训练数据（最近30天）
    df, output_file = generate_training_data_from_klines(days=30)
    
    if df is not None:
        print(f"\n✅ 训练数据生成成功！")
        print(f"   文件: {output_file}")
        print(f"   数据量: {len(df)}条")
        print(f"   时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
        print(f"\n💡 现在可以使用此文件进行序列长度测试:")
        print(f"   python scripts/analysis/sequence_length_tester.py")


if __name__ == "__main__":
    main()
