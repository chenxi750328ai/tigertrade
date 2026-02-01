import subprocess
import time
import threading
from datetime import datetime
import pandas as pd
import numpy as np
import os
import glob
from collections import Counter

def run_tiger1():
    """运行tiger1策略"""
    try:
        result = subprocess.run(
            ["python", "/home/cx/tigertrade/tiger1.py", "d"],
            cwd="/home/cx/tigertrade",
            capture_output=True,
            text=True,
            timeout=60
        )
        print(f"Strategy execution completed with return code: {result.returncode}")
        if result.stdout:
            print("STDOUT:", result.stdout[-2000:])  # 只打印最后2000个字符
        if result.stderr:
            print("STDERR:", result.stderr)
    except subprocess.TimeoutExpired:
        print("Strategy execution timed out after 60 seconds")

def analyze_today_data():
    """分析今天的交易数据并提供优化建议"""
    # 获取今天的数据文件
    today = datetime.now().strftime('%Y-%m-%d')
    data_dir = f'/home/cx/trading_data/{today}'
    data_file_pattern = os.path.join(data_dir, f'trading_data_{today}.csv')
    
    # 查找今天的数据文件
    files = glob.glob(data_file_pattern)
    
    if not files:
        print("❌ 未找到今天的交易数据文件")
        return

    data_file = files[0]
    print(f"🔍 分析数据文件: {data_file}")
    
    try:
        df = pd.read_csv(data_file)
    except Exception as e:
        print(f"❌ 读取数据文件失败: {e}")
        return

    if df.empty:
        print("❌ 数据文件为空")
        return

    print(f"📊 今日数据记录数: {len(df)}")
    
    # 处理时间戳列
    timestamp_mask = df['timestamp'].notna() & (df['timestamp'] != '')
    df.loc[timestamp_mask, 'timestamp'] = pd.to_datetime(df.loc[timestamp_mask, 'timestamp'], errors='coerce')
    
    # 数据清理和预处理
    df = df.dropna(subset=['price_current', 'grid_lower', 'grid_upper', 'atr'])  # 移除关键数据缺失的行
    
    if df.empty:
        print("❌ 清理后数据为空")
        return

    print(f"📊 清理后数据记录数: {len(df)}")
    
    # 分析各条件触发频率
    condition_cols = ['near_lower', 'rsi_ok', 'trend_check', 'rebound', 'vol_ok', 'final_decision']
    condition_freq = {}
    
    for col in condition_cols:
        if col in df.columns:
            if df[col].dtype == 'object':
                # 尝试转换布尔值
                df[col] = df[col].astype(str).str.lower().isin(['true', '1', 'yes', 'y'])
            true_count = df[col].sum()
            total_count = len(df)
            freq = true_count / total_count if total_count > 0 else 0
            condition_freq[col] = freq
            print(f"📈 {col} 触发频率: {freq:.2%} ({true_count}/{total_count})")
    
    # 计算价格统计
    if 'price_current' in df.columns:
        price_stats = {
            'min': df['price_current'].min(),
            'max': df['price_current'].max(),
            'mean': df['price_current'].mean(),
            'std': df['price_current'].std()
        }
        print(f"💰 价格统计 - 最低: {price_stats['min']:.3f}, 最高: {price_stats['max']:.3f}, 平均: {price_stats['mean']:.3f}, 标准差: {price_stats['std']:.3f}")
    
    # 分析网格参数
    if 'grid_lower' in df.columns and 'grid_upper' in df.columns:
        grid_stats = {
            'lower_min': df['grid_lower'].min(),
            'lower_max': df['grid_lower'].max(),
            'upper_min': df['grid_upper'].min(),
            'upper_max': df['grid_upper'].max(),
            'width_mean': (df['grid_upper'] - df['grid_lower']).mean()
        }
        print(f"📊 网格统计 - 下轨范围: [{grid_stats['lower_min']:.3f}, {grid_stats['lower_max']:.3f}], 上轨范围: [{grid_stats['upper_min']:.3f}, {grid_stats['upper_max']:.3f}], 平均宽度: {grid_stats['width_mean']:.3f}")
    
    # 分析ATR
    if 'atr' in df.columns:
        atr_stats = {
            'min': df['atr'].min(),
            'max': df['atr'].max(),
            'mean': df['atr'].mean(),
            'std': df['atr'].std()
        }
        print(f"📈 ATR统计 - 范围: [{atr_stats['min']:.3f}, {atr_stats['max']:.3f}], 平均值: {atr_stats['mean']:.3f}, 标准差: {atr_stats['std']:.3f}")
    
    # 分析near_lower为True的情况
    if 'near_lower' in df.columns:
        near_lower_true_df = df[df['near_lower'] == True]
        if len(near_lower_true_df) > 0:
            print(f"\n🔍 当near_lower为True时:")
            print(f"   - 平均价格: {near_lower_true_df['price_current'].mean():.3f}")
            print(f"   - 平均下轨: {near_lower_true_df['grid_lower'].mean():.3f}")
            print(f"   - 平均ATR: {near_lower_true_df['atr'].mean():.3f}")
            print(f"   - 触发买入次数: {near_lower_true_df['final_decision'].sum()} / {len(near_lower_true_df)} ({near_lower_true_df['final_decision'].mean():.2%})")
        else:
            print("\n🔍 未发现near_lower为True的情况")
    
    # 分析交易决策
    if 'final_decision' in df.columns:
        buy_decisions = df[df['final_decision'] == True]
        if len(buy_decisions) > 0:
            print(f"\n🎯 总共 {len(buy_decisions)} 次买入决策")
            if 'price_current' in df.columns:
                print(f"   - 平均买入价格: {buy_decisions['price_current'].mean():.3f}")
        else:
            print("\n🎯 今日无买入决策")
    
    # 参数优化建议
    print(f"\n💡 参数优化建议:")
    if 'atr' in df.columns:
        avg_atr = df['atr'].mean()
        print(f"   - 当前ATR平均值: {avg_atr:.3f}")
        print(f"   - 建议STOP_LOSS_MULTIPLIER: 当前值1.2，可根据ATR波动调整")
        print(f"   - 建议网格间距: 基于ATR调整，当前平均网格宽度{grid_stats['width_mean']:.3f}")
    
    if 'near_lower' in df.columns:
        near_lower_rate = condition_freq.get('near_lower', 0)
        if near_lower_rate < 0.1:  # 如果near_lower触发频率低于10%
            print(f"   - near_lower触发率较低({near_lower_rate:.2%})，建议:")
            print(f"     • 调整buffer计算公式，增加缓冲区大小")
            print(f"     • 或者降低网格下轨，增加触发概率")
        elif near_lower_rate > 0.5:  # 如果near_lower触发频率过高
            print(f"   - near_lower触发率较高({near_lower_rate:.2%})，建议:")
            print(f"     • 缩小buffer，减少误触发")
            print(f"     • 提高网格下轨，更严格控制买入时机")
    
    if 'rsi_ok' in df.columns:
        rsi_ok_rate = condition_freq.get('rsi_ok', 0)
        print(f"   - RSI条件触发率: {rsi_ok_rate:.2%}，可根据此值调整RSI阈值")
    
    # 分析最新数据
    if 'timestamp' in df.columns and timestamp_mask.any():
        latest_data = df[df['timestamp'].notna()].tail(5)  # 最近5条有时间戳的数据
        if len(latest_data) > 0:
            print(f"\n🆕 最新数据快照:")
            for idx, row in latest_data.iterrows():
                if pd.notna(row['price_current']) and pd.notna(row['grid_lower']) and pd.notna(row['atr']):
                    price = row['price_current']
                    lower = row['grid_lower']
                    atr = row['atr']
                    buffer = max(0.3 * atr, 0.0025) if pd.notna(atr) else 0.0025
                    threshold = lower + buffer
                    near_lower_calc = price <= threshold
                    
                    print(f"   - 时间: {row['timestamp']}")
                    print(f"     价格: {price:.3f}, 下轨: {lower:.3f}, ATR: {atr:.3f}")
                    print(f"     Buffer: {buffer:.4f}, 阈值: {threshold:.4f}")
                    print(f"     near_lower计算: {price:.3f} <= {threshold:.4f} = {near_lower_calc}")
                    print(f"     实际near_lower: {row['near_lower']}")
                    print(f"     决策: {'买入' if row['final_decision'] else '不买入'}")
                    print()

def periodic_run_and_analysis():
    """定时运行策略并分析数据"""
    print("🚀 启动定时运行和分析程序...")
    
    while True:
        # 运行策略
        print(f"\n🕒 [{datetime.now().strftime('%H:%M:%S')}] 开始运行tiger1策略...")
        run_tiger1()
        
        # 等待一段时间让数据积累
        time.sleep(30)
        
        # 分析数据
        print(f"\n📊 [{datetime.now().strftime('%H:%M:%S')}] 开始分析数据...")
        analyze_today_data()
        
        # 等待下一个运行周期
        print(f"\n⏸️  等待下一轮运行...")
        time.sleep(120)  # 每2分钟运行一次

if __name__ == "__main__":
    periodic_run_and_analysis()