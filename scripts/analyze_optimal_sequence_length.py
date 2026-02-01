"""
分析最优序列长度
根据数据量和历史成交情况，动态确定序列长度
"""
import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime


def analyze_data_and_sequence_length(data_dir='/home/cx/trading_data'):
    """分析数据量和最优序列长度"""
    print("="*70)
    print("分析数据量和最优序列长度")
    print("="*70)
    
    # 查找最新的合并数据文件
    data_files = glob.glob(os.path.join(data_dir, 'training_data_multitimeframe_merged_*.csv'))
    if not data_files:
        print("❌ 未找到训练数据文件")
        return None
    
    # 使用最新的文件
    latest_file = max(data_files, key=os.path.getmtime)
    print(f"\n📊 使用数据文件: {os.path.basename(latest_file)}")
    
    # 加载数据
    try:
        df = pd.read_csv(latest_file)
        print(f"✅ 数据加载成功")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return None
    
    # 分析数据
    total_samples = len(df)
    print(f"\n数据统计:")
    print(f"  总样本数: {total_samples:,}")
    
    # 检查时间戳
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp']).sort_values('timestamp')
        
        time_span = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600
        print(f"  时间跨度: {time_span:.1f} 小时")
        
        # 估算时间间隔（假设数据是1分钟K线）
        if len(df) > 1:
            time_diffs = df['timestamp'].diff().dropna()
            avg_interval_minutes = time_diffs.median().total_seconds() / 60
            print(f"  平均时间间隔: {avg_interval_minutes:.1f} 分钟")
    
    # 分析价格历史成交情况
    if 'price_current' in df.columns:
        prices = df['price_current'].values
        
        # 计算价格变化的相关性
        print(f"\n价格分析:")
        print(f"  价格范围: {prices.min():.2f} - {prices.max():.2f}")
        print(f"  价格标准差: {prices.std():.4f}")
        
        # 计算自相关（价格与历史价格的相关性）
        max_lag = min(500, len(prices) // 2)  # 最多看500步
        autocorrelations = []
        
        for lag in range(1, min(100, max_lag), 10):  # 每10步采样一次
            if lag < len(prices):
                corr = np.corrcoef(prices[:-lag], prices[lag:])[0, 1]
                if not np.isnan(corr):
                    autocorrelations.append((lag, corr))
        
        if autocorrelations:
            print(f"\n价格自相关分析（滞后步数 vs 相关系数）:")
            for lag, corr in autocorrelations[:10]:  # 显示前10个
                print(f"  滞后 {lag:3d} 步: {corr:.4f}")
            
            # 找到相关性仍然较高的最大滞后
            significant_lags = [lag for lag, corr in autocorrelations if abs(corr) > 0.1]
            if significant_lags:
                max_significant_lag = max(significant_lags)
                print(f"\n  仍有显著相关性（|r|>0.1）的最大滞后: {max_significant_lag} 步")
            else:
                max_significant_lag = 50
    
    # 推荐序列长度
    print(f"\n" + "="*70)
    print("序列长度推荐")
    print("="*70)
    
    # 基于数据量的推荐
    if total_samples < 1000:
        recommended_seq_short = min(50, total_samples // 10)
        recommended_seq_long = min(100, total_samples // 5)
    elif total_samples < 10000:
        recommended_seq_short = 100
        recommended_seq_long = 200
    else:
        recommended_seq_short = 200
        recommended_seq_long = 500
    
    print(f"\n基于数据量的推荐:")
    print(f"  保守推荐: {recommended_seq_short} 步")
    print(f"  激进推荐: {recommended_seq_long} 步")
    print(f"  数据量/序列长度比例: {total_samples/recommended_seq_long:.1f}:1")
    
    # 基于价格相关性的推荐
    if 'price_current' in df.columns and autocorrelations:
        # 找到相关性降到0.1以下的滞后
        low_corr_lags = [lag for lag, corr in autocorrelations if abs(corr) < 0.1]
        if low_corr_lags:
            correlation_based_seq = min(low_corr_lags)
        else:
            correlation_based_seq = max_significant_lag if 'max_significant_lag' in locals() else 200
        
        print(f"\n基于价格相关性的推荐:")
        print(f"  推荐序列长度: {correlation_based_seq} 步")
    
    # 理论分析：覆盖所有历史成交情况
    print(f"\n理论分析（覆盖所有历史成交情况）:")
    print(f"  如果要覆盖所有历史成交情况，序列长度应该 = 总样本数")
    print(f"  但这会导致:")
    print(f"    - 每个样本都需要 {total_samples} 步历史")
    print(f"    - 实际可用样本数: {total_samples - total_samples} = 0")
    print(f"    - 这是不可行的")
    
    print(f"\n  更合理的方案:")
    print(f"    - 使用尽可能长的序列，但不超过数据量的80%")
    print(f"    - 推荐序列长度: {int(total_samples * 0.8)} 步")
    print(f"    - 实际可用样本数: {int(total_samples * 0.2)} 个")
    
    # 动态序列长度策略
    print(f"\n" + "="*70)
    print("动态序列长度策略")
    print("="*70)
    
    print(f"\n方案1: 固定长序列（推荐用于大模型）")
    print(f"  序列长度: {recommended_seq_long} 步")
    print(f"  优势: 简单，覆盖更多历史信息")
    print(f"  劣势: 计算量大，可能过拟合")
    
    print(f"\n方案2: 自适应序列长度")
    print(f"  根据数据量动态调整:")
    print(f"    - 数据量 < 5K: 序列长度 = 数据量 / 10")
    print(f"    - 数据量 5K-20K: 序列长度 = 200-500")
    print(f"    - 数据量 > 20K: 序列长度 = 500-1000")
    
    print(f"\n方案3: 分层序列长度（多时间尺度）")
    print(f"  短期序列: 50-100 步（1-2小时）")
    print(f"  中期序列: 200-500 步（4-10小时）")
    print(f"  长期序列: 1000+ 步（20+小时）")
    print(f"  模型同时使用多个序列长度")
    
    return {
        'total_samples': total_samples,
        'recommended_seq_short': recommended_seq_short,
        'recommended_seq_long': recommended_seq_long,
        'max_sequence': int(total_samples * 0.8)
    }


if __name__ == '__main__':
    result = analyze_data_and_sequence_length()
    if result:
        print(f"\n✅ 分析完成")
        print(f"\n建议使用的序列长度: {result['recommended_seq_long']} 步")
