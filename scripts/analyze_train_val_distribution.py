"""
分析训练集和验证集的分布差异
判断是过拟合还是分布偏移
"""
import pandas as pd
import numpy as np
import os
import glob
import sys
sys.path.insert(0, '/home/cx/tigertrade')

from src.strategies import llm_strategy


def analyze_distribution_difference(data_dir='/home/cx/trading_data'):
    """分析训练集和验证集的分布差异"""
    print("="*70)
    print("分析训练集和验证集的分布差异")
    print("="*70)
    
    # 加载数据
    data_files = glob.glob(os.path.join(data_dir, 'training_data_multitimeframe_merged_*.csv'))
    if not data_files:
        print("❌ 未找到训练数据文件")
        return
    
    latest_file = max(data_files, key=os.path.getmtime)
    print(f"\n📊 使用数据文件: {os.path.basename(latest_file)}")
    
    df = pd.read_csv(latest_file)
    print(f"数据形状: {df.shape}")
    
    # 准备数据（与训练时一致）
    strategy = llm_strategy.LLMTradingStrategy(mode='hybrid', predict_profit=True)
    seq_length = 500  # 使用当前训练的序列长度
    strategy._seq_length = seq_length
    
    X, y, y_profit = [], [], []
    
    for i in range(seq_length, len(df)):
        try:
            historical_data = df.iloc[i-seq_length:i+1]
            sequence = strategy.prepare_sequence_features(historical_data, len(historical_data)-1, seq_length)
            
            # 标签生成（与训练时一致）
            current_price = df.iloc[i]['price_current']
            if i + 120 < len(df):
                future_prices = df.iloc[i+1:i+121]['price_current'].values
                buy_profit = (np.max(future_prices) - current_price) / current_price
                sell_profit = (current_price - np.min(future_prices)) / current_price
                
                profit_threshold = 0.003
                min_diff = 0.002
                current_position = int(df.iloc[i].get('current_position', 0))
                
                if current_position > 0:
                    if sell_profit > profit_threshold:
                        label = 2
                    elif buy_profit > profit_threshold:
                        label = 1
                    else:
                        label = 0
                else:
                    if abs(buy_profit - sell_profit) >= min_diff:
                        if buy_profit > sell_profit and buy_profit > profit_threshold:
                            label = 1
                        elif sell_profit > buy_profit and sell_profit > profit_threshold:
                            label = 2
                        else:
                            label = 0
                    else:
                        label = 0
                
                if label == 1:
                    actual_profit = buy_profit
                elif label == 2:
                    actual_profit = sell_profit
                else:
                    actual_profit = 0.0
                
                X.append(sequence)
                y.append(label)
                y_profit.append(actual_profit)
        except Exception as e:
            continue
    
    X = np.array(X)
    y = np.array(y)
    y_profit = np.array(y_profit)
    
    # 划分训练集和验证集（与训练时一致）
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    y_profit_train, y_profit_val = y_profit[:split_idx], y_profit[split_idx:]
    
    print(f"\n数据划分:")
    print(f"  训练集: {len(X_train):,} 个样本")
    print(f"  验证集: {len(X_val):,} 个样本")
    
    # 分析类别分布
    print(f"\n" + "="*70)
    print("类别分布分析")
    print("="*70)
    
    train_classes, train_counts = np.unique(y_train, return_counts=True)
    val_classes, val_counts = np.unique(y_val, return_counts=True)
    
    print(f"\n训练集类别分布:")
    for cls, count in zip(train_classes, train_counts):
        pct = count / len(y_train) * 100
        print(f"  类别 {cls}: {count:,} ({pct:.2f}%)")
    
    print(f"\n验证集类别分布:")
    for cls, count in zip(val_classes, val_counts):
        pct = count / len(y_val) * 100
        print(f"  类别 {cls}: {count:,} ({pct:.2f}%)")
    
    # 计算分布差异（KL散度）
    from scipy.stats import entropy
    
    # 对齐类别
    all_classes = np.unique(np.concatenate([train_classes, val_classes]))
    train_dist = np.zeros(len(all_classes))
    val_dist = np.zeros(len(all_classes))
    
    for i, cls in enumerate(all_classes):
        train_idx = np.where(train_classes == cls)[0]
        val_idx = np.where(val_classes == cls)[0]
        
        if len(train_idx) > 0:
            train_dist[i] = train_counts[train_idx[0]] / len(y_train)
        if len(val_idx) > 0:
            val_dist[i] = val_counts[val_idx[0]] / len(y_val)
    
    kl_div = entropy(train_dist + 1e-10, val_dist + 1e-10)
    print(f"\n分布差异（KL散度）: {kl_div:.6f}")
    if kl_div < 0.1:
        print("  ✅ 训练集和验证集分布相似（可能是过拟合）")
    else:
        print("  ⚠️ 训练集和验证集分布差异较大（可能是分布偏移）")
    
    # 分析收益率分布
    print(f"\n" + "="*70)
    print("收益率分布分析")
    print("="*70)
    
    print(f"\n训练集收益率:")
    print(f"  均值: {np.mean(y_profit_train):.6f}")
    print(f"  标准差: {np.std(y_profit_train):.6f}")
    print(f"  最小值: {np.min(y_profit_train):.6f}")
    print(f"  最大值: {np.max(y_profit_train):.6f}")
    print(f"  中位数: {np.median(y_profit_train):.6f}")
    
    print(f"\n验证集收益率:")
    print(f"  均值: {np.mean(y_profit_val):.6f}")
    print(f"  标准差: {np.std(y_profit_val):.6f}")
    print(f"  最小值: {np.min(y_profit_val):.6f}")
    print(f"  最大值: {np.max(y_profit_val):.6f}")
    print(f"  中位数: {np.median(y_profit_val):.6f}")
    
    # 分析价格特征分布
    print(f"\n" + "="*70)
    print("价格特征分布分析")
    print("="*70)
    
    # 提取价格相关特征
    train_prices = []
    val_prices = []
    
    for i in range(len(X_train)):
        # 假设价格特征在序列的某个位置
        if len(X_train[i]) > 0:
            train_prices.append(X_train[i][-1][0])  # 假设第一个特征是价格
    
    for i in range(len(X_val)):
        if len(X_val[i]) > 0:
            val_prices.append(X_val[i][-1][0])
    
    if train_prices and val_prices:
        train_prices = np.array(train_prices)
        val_prices = np.array(val_prices)
        
        print(f"\n训练集价格特征:")
        print(f"  均值: {np.mean(train_prices):.4f}")
        print(f"  标准差: {np.std(train_prices):.4f}")
        
        print(f"\n验证集价格特征:")
        print(f"  均值: {np.mean(val_prices):.4f}")
        print(f"  标准差: {np.std(val_prices):.4f}")
        
        # 统计检验
        from scipy.stats import ks_2samp
        ks_stat, ks_pvalue = ks_2samp(train_prices, val_prices)
        print(f"\nKolmogorov-Smirnov检验:")
        print(f"  统计量: {ks_stat:.6f}")
        print(f"  p值: {ks_pvalue:.6f}")
        if ks_pvalue < 0.05:
            print("  ⚠️ 训练集和验证集价格分布显著不同（p<0.05）")
        else:
            print("  ✅ 训练集和验证集价格分布相似（p>=0.05）")
    
    # 结论和建议
    print(f"\n" + "="*70)
    print("结论和建议")
    print("="*70)
    
    if kl_div < 0.1 and (not train_prices or ks_pvalue >= 0.05):
        print("\n✅ 训练集和验证集分布相似")
        print("   可能原因: 过拟合（模型学习了训练集特有的模式）")
        print("\n建议:")
        print("  1. 增加数据量（根本解决）")
        print("  2. 使用更强的正则化（PEFT、MoE、数据增强）")
        print("  3. 减少模型容量（如果数据量不足）")
    else:
        print("\n⚠️ 训练集和验证集分布差异较大")
        print("   可能原因: 分布偏移（训练集和验证集的市场状态不同）")
        print("\n建议:")
        print("  1. 使用领域适配（Domain Adaptation）")
        print("  2. 增加数据多样性（不同市场状态的数据）")
        print("  3. 使用更通用的特征（减少对特定市场状态的依赖）")


if __name__ == '__main__':
    analyze_distribution_difference()
