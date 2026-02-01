#!/usr/bin/env python3
"""
通过API获取更多的历史数据用于模型训练
"""

import sys
import os
import time
import pandas as pd
from datetime import datetime, timedelta

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入tiger1模块的必要函数
from tiger1 import (
    get_kline_data, calculate_indicators, 
    FUTURE_SYMBOL, data_collector
)


def fetch_historical_data(days=30, periods=['1min', '5min']):
    """
    获取历史K线数据
    
    Args:
        days: 获取的天数
        periods: 周期列表
    
    Returns:
        dict: 各周期的数据
    """
    print(f"🔄 开始获取 {days} 天的历史数据...")
    
    # 计算需要的K线数量
    # 1分钟: 一天大约有390条(交易时段)
    # 5分钟: 一天大约有78条
    counts = {
        '1min': days * 400,  # 多留一些余量
        '5min': days * 100,
    }
    
    historical_data = {}
    
    for period in periods:
        try:
            print(f"  正在获取 {period} 数据...")
            count = counts.get(period, 1000)
            df = get_kline_data([FUTURE_SYMBOL], period, count=count)
            
            if not df.empty:
                historical_data[period] = df
                print(f"  ✅ {period} 数据获取成功: {len(df)} 条记录")
                print(f"     时间范围: {df.index[0]} 到 {df.index[-1]}")
            else:
                print(f"  ⚠️ {period} 数据为空")
                
        except Exception as e:
            print(f"  ❌ 获取 {period} 数据失败: {e}")
            import traceback
            traceback.print_exc()
    
    return historical_data


def calculate_features_batch(df_5m, df_1m):
    """
    批量计算特征
    
    Args:
        df_5m: 5分钟数据
        df_1m: 1分钟数据
    
    Returns:
        DataFrame: 特征数据
    """
    features_list = []
    
    # 至少需要50条数据来计算指标
    min_len = 50
    
    if len(df_5m) < min_len or len(df_1m) < min_len:
        print(f"⚠️ 数据不足，需要至少 {min_len} 条")
        return pd.DataFrame()
    
    # 滑动窗口计算特征
    window_size = 20  # 使用20个周期的窗口
    
    print(f"📊 开始批量计算特征，数据量: 5分钟={len(df_5m)}, 1分钟={len(df_1m)}")
    
    for i in range(min_len, len(df_5m)):
        try:
            # 获取窗口数据
            window_5m = df_5m.iloc[max(0, i-window_size):i+1]
            
            # 找到对应的1分钟数据窗口
            timestamp_5m = df_5m.index[i]
            df_1m_slice = df_1m[df_1m.index <= timestamp_5m]
            
            if len(df_1m_slice) < min_len:
                continue
            
            window_1m = df_1m_slice.iloc[-window_size:]
            
            # 计算指标
            inds = calculate_indicators(window_5m, window_1m)
            
            if '5m' not in inds or '1m' not in inds:
                continue
            
            price_current = inds['1m']['close']
            atr = inds['5m']['atr']
            rsi_1m = inds['1m']['rsi']
            rsi_5m = inds['5m']['rsi']
            
            # 使用硬编码的网格值
            grid_upper = price_current * 1.01
            grid_lower = price_current * 0.99
            
            buffer = max(atr * 0.3, 0.0025)
            threshold = grid_lower + buffer
            
            # 获取布林带数据
            boll_upper = inds['5m'].get('boll_upper', 0)
            boll_mid = inds['5m'].get('boll_mid', 0)
            boll_lower = inds['5m'].get('boll_lower', 0)
            
            # 构建特征
            features = {
                'timestamp': timestamp_5m,
                'price_current': price_current,
                'grid_lower': grid_lower,
                'grid_upper': grid_upper,
                'atr': atr,
                'rsi_1m': rsi_1m,
                'rsi_5m': rsi_5m,
                'buffer': buffer,
                'threshold': threshold,
                'near_lower': price_current <= threshold,
                'rsi_ok': rsi_1m < 30 or (rsi_5m > 45 and rsi_5m < 55),
                'boll_upper': boll_upper,
                'boll_mid': boll_mid,
                'boll_lower': boll_lower,
                # 可以添加更多特征
                'price_change_pct': (price_current - window_5m['close'].iloc[-2]) / window_5m['close'].iloc[-2] * 100 if len(window_5m) > 1 else 0,
            }
            
            features_list.append(features)
            
        except Exception as e:
            # 忽略单个计算错误，继续处理
            if i % 100 == 0:  # 每100条打印一次错误
                print(f"  ⚠️ 计算特征时出错 (索引 {i}): {e}")
            continue
    
    df_features = pd.DataFrame(features_list)
    print(f"✅ 特征计算完成: {len(df_features)} 条记录")
    
    return df_features


def save_data(df, filename):
    """
    保存数据到文件
    
    Args:
        df: DataFrame
        filename: 文件名
    """
    try:
        # 创建数据目录
        data_dir = '/home/cx/trading_data/historical'
        os.makedirs(data_dir, exist_ok=True)
        
        # 生成完整路径
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(data_dir, f'{filename}_{timestamp}.csv')
        
        # 保存为CSV
        df.to_csv(filepath, index=True, encoding='utf-8')
        print(f"✅ 数据已保存到: {filepath}")
        
        # 同时保存为Parquet格式（更高效）
        parquet_path = filepath.replace('.csv', '.parquet')
        df.to_parquet(parquet_path, index=True)
        print(f"✅ Parquet格式已保存到: {parquet_path}")
        
        return filepath
        
    except Exception as e:
        print(f"❌ 保存数据失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_training_labels(df):
    """
    为数据生成训练标签
    
    策略:
    - 如果未来价格上涨超过1%，标签为1 (买入)
    - 如果未来价格下跌超过1%，标签为2 (卖出)
    - 否则标签为0 (持有)
    
    Args:
        df: 特征DataFrame
    
    Returns:
        DataFrame: 添加了标签的数据
    """
    print("🏷️ 开始生成训练标签...")
    
    df = df.copy()
    df['label'] = 0  # 默认为持有
    
    # 计算未来价格变化（向前看N个周期）
    look_ahead = 5  # 向前看5个周期
    
    for i in range(len(df) - look_ahead):
        current_price = df.iloc[i]['price_current']
        future_price = df.iloc[i + look_ahead]['price_current']
        
        price_change_pct = (future_price - current_price) / current_price * 100
        
        # 设置阈值
        buy_threshold = 0.5   # 上涨超过0.5%标记为买入
        sell_threshold = -0.5  # 下跌超过0.5%标记为卖出
        
        if price_change_pct > buy_threshold:
            df.iloc[i, df.columns.get_loc('label')] = 1  # 买入
        elif price_change_pct < sell_threshold:
            df.iloc[i, df.columns.get_loc('label')] = 2  # 卖出
        else:
            df.iloc[i, df.columns.get_loc('label')] = 0  # 持有
    
    # 最后几条数据没有未来数据，设为持有
    df.iloc[-look_ahead:, df.columns.get_loc('label')] = 0
    
    # 打印标签分布
    label_counts = df['label'].value_counts()
    print(f"  标签分布:")
    print(f"    持有 (0): {label_counts.get(0, 0)} ({label_counts.get(0, 0)/len(df)*100:.1f}%)")
    print(f"    买入 (1): {label_counts.get(1, 0)} ({label_counts.get(1, 0)/len(df)*100:.1f}%)")
    print(f"    卖出 (2): {label_counts.get(2, 0)} ({label_counts.get(2, 0)/len(df)*100:.1f}%)")
    
    return df


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("📥 数据采集工具 - 通过API获取历史数据")
    print("=" * 80)
    
    # 解析命令行参数
    days = 30  # 默认获取30天数据
    if len(sys.argv) > 1:
        try:
            days = int(sys.argv[1])
        except ValueError:
            print("⚠️ 无效的天数，使用默认值30天")
    
    print(f"\n📅 将获取过去 {days} 天的数据\n")
    
    # 1. 获取历史数据
    historical_data = fetch_historical_data(days=days)
    
    if not historical_data:
        print("❌ 没有获取到任何数据")
        return
    
    # 2. 保存原始K线数据
    for period, df in historical_data.items():
        save_data(df, f'kline_{period}')
    
    # 3. 计算特征
    if '5min' in historical_data and '1min' in historical_data:
        print("\n" + "=" * 80)
        print("📊 开始计算技术特征")
        print("=" * 80)
        
        df_features = calculate_features_batch(
            historical_data['5min'], 
            historical_data['1min']
        )
        
        if not df_features.empty:
            # 4. 生成训练标签
            df_with_labels = generate_training_labels(df_features)
            
            # 5. 保存特征数据
            filepath = save_data(df_with_labels, 'training_data')
            
            # 6. 显示数据统计
            print("\n" + "=" * 80)
            print("📈 数据统计信息")
            print("=" * 80)
            print(f"\n总记录数: {len(df_with_labels)}")
            print(f"\n特征列:")
            for col in df_with_labels.columns:
                print(f"  - {col}")
            
            print(f"\n数值统计:")
            print(df_with_labels.describe())
            
            print("\n" + "=" * 80)
            print("✅ 数据采集完成！")
            print("=" * 80)
            print(f"\n可以使用这些数据来训练和改进模型:")
            print(f"  - 原始K线数据保存在: /home/cx/trading_data/historical/kline_*.csv")
            print(f"  - 训练数据保存在: {filepath}")
            
    else:
        print("⚠️ 缺少必要的数据周期，无法计算特征")


if __name__ == "__main__":
    main()
