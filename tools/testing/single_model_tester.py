import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
from datetime import datetime
import glob
import argparse
import warnings
import gc

warnings.filterwarnings("ignore")

from data_fetcher import aggregate_data_for_training, prepare_features_from_raw_data


def prepare_features_and_labels(df, look_ahead=10):
    """准备特征和标签"""
    print("Preparing features and labels from extended dataset...")
    
    # 准备特征
    X = []
    for i in range(len(df)):
        row = df.iloc[i]
        
        # 准备特征
        features = [
            row.get('price_current', 0),
            row.get('grid_lower', 0),
            row.get('grid_upper', 0),
            row.get('atr', 0),
            row.get('rsi_1m', 50),
            row.get('rsi_5m', 50),
            row.get('buffer', 0),
            row.get('threshold', 0),
            1 if row.get('near_lower', False) else 0,
            1 if row.get('rsi_ok', False) else 0
        ]
        
        # 归一化特征
        features_np = np.array(features)
        mean_val = np.mean(features_np)
        std_val = np.std(features_np) + 1e-8
        normalized_features = (features_np - mean_val) / std_val
        X.append(normalized_features.tolist())
    
    # 生成标签
    y = []
    for i in range(len(df) - look_ahead):
        current_price = df.iloc[i]['price_current']
        future_prices = df.iloc[i+1:i+look_ahead+1]['price_current'].values
        
        if len(future_prices) == 0:
            # 如果未来价格不可用，使用默认标签
            label = 0
        else:
            # 计算最大盈利和最大亏损
            max_future_price = max(future_prices)
            min_future_price = min(future_prices)
            
            buy_profit = (max_future_price - current_price) / current_price
            sell_profit = (current_price - min_future_price) / current_price
            
            # 创建标签: 0=不操作, 1=买入, 2=卖出
            # 只有当预期盈利超过阈值时才建议操作
            profit_threshold = 0.005  # 提高阈值到0.5%，减少交易频率但提高质量
            min_diff = 0.003  # 最小差值，确保买卖之间有足够差距
            
            # 只有当买卖盈利差值超过最小差值且超过阈值时才交易
            if abs(buy_profit - sell_profit) >= min_diff:
                if buy_profit > sell_profit and buy_profit > profit_threshold:
                    label = 1  # 买入
                elif sell_profit > buy_profit and sell_profit > profit_threshold:
                    label = 2  # 卖出
                else:
                    label = 0  # 不操作
            else:
                label = 0  # 不操作 - 差值太小，不确定性高
        
        y.append(label)
    
    # 对于最后look_ahead个数据点，复制最后一个标签
    for _ in range(min(look_ahead, len(X) - len(y))):
        y.append(y[-1] if y else 0)
    
    # 确保X和y长度一致
    X = X[:len(y)]
    
    return np.array(X), np.array(y)


def test_specific_model(model_name):
    """测试特定的模型"""
    print(f"🔍 Loading extended historical data for {model_name}...")
    
    # 加载扩展的历史数据
    df = aggregate_data_for_training()
    if df is None or len(df) < 1000:  # 要求至少1000个数据点
        print("❌ Insufficient extended data for training")
        return
    
    print(f"📊 Loaded {len(df)} data points from extended dataset")
    
    # 准备特征
    print("🔍 Preparing features from raw data...")
    features_df = prepare_features_from_raw_data(df)
    if features_df is None or len(features_df) < 1000:
        print("❌ Insufficient features for training")
        return
    
    # 生成标签
    X, y = prepare_features_and_labels(features_df)
    
    if len(X) < 1000:
        print("❌ Insufficient prepared data for training")
        return
    
    print(f"📊 Prepared data: Feature matrix {X.shape}, Label vector {y.shape}")
    
    # 分割数据集
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"📈 Training set size: {len(X_train)}, Test set size: {len(X_test)}")
    
    # 根据模型名称导入并测试相应的模型
    if model_name == "llm":
        from llm_strategy import LLMTradingStrategy
        print("🚀 Testing LLM Trading Strategy...")
        strategy = LLMTradingStrategy()
        model_params = sum(p.numel() for p in strategy.model.parameters() if p.requires_grad)
        print(f"LLM Trading Parameters: {model_params:,}")
    elif model_name == "large_model":
        from large_model_strategy import LargeModelStrategy
        print("🚀 Testing Large Model Strategy...")
        strategy = LargeModelStrategy()
        model_params = sum(p.numel() for p in strategy.model.parameters() if p.requires_grad)
        print(f"Large Model Parameters: {model_params:,}")
    elif model_name == "large_transformer":
        from large_transformer_strategy import LargeTransformerStrategy
        print("🚀 Testing Large Transformer Strategy...")
        strategy = LargeTransformerStrategy()
        model_params = sum(p.numel() for p in strategy.model.parameters() if p.requires_grad)
        print(f"Large Transformer Parameters: {model_params:,}")
    elif model_name == "enhanced_transformer":
        from enhanced_transformer_strategy import EnhancedTransformerStrategy
        print("🚀 Testing Enhanced Transformer Strategy...")
        strategy = EnhancedTransformerStrategy()
        model_params = sum(p.numel() for p in strategy.model.parameters() if p.requires_grad)
        print(f"Enhanced Transformer Parameters: {model_params:,}")
    elif model_name == "huge_transformer":
        from huge_transformer_strategy import HugeTransformerStrategy
        print("🚀 Testing Huge Transformer Strategy...")
        strategy = HugeTransformerStrategy()
        model_params = sum(p.numel() for p in strategy.model.parameters() if p.requires_grad)
        print(f"Huge Transformer Parameters: {model_params:,}")
    else:
        print(f"❌ Unknown model name: {model_name}")
        return
    
    # 创建临时DataFrame用于训练
    temp_df = pd.DataFrame({
        'price_current': X_train[:, 0],
        'grid_lower': X_train[:, 1],
        'grid_upper': X_train[:, 2],
        'atr': X_train[:, 3],
        'rsi_1m': X_train[:, 4] * 100,
        'rsi_5m': X_train[:, 5] * 100,
        'buffer': X_train[:, 6],
        'threshold': X_train[:, 7],
        'near_lower': X_train[:, 8] > 0.5,
        'rsi_ok': X_train[:, 9] > 0.5
    })
    
    # 训练模型
    if hasattr(strategy, 'train_model'):
        print(f"   📊 Starting training for {model_name}...")
        strategy.train_model(temp_df)
    
    # 评估模型
    correct = 0
    total = 0
    
    # 准备测试数据
    temp_test_df = pd.DataFrame({
        'price_current': X_test[:, 0],
        'grid_lower': X_test[:, 1],
        'grid_upper': X_test[:, 2],
        'atr': X_test[:, 3],
        'rsi_1m': X_test[:, 4] * 100,
        'rsi_5m': X_test[:, 5] * 100,
        'buffer': X_test[:, 6],
        'threshold': X_test[:, 7],
        'near_lower': X_test[:, 8] > 0.5,
        'rsi_ok': X_test[:, 9] > 0.5
    })
    
    # 对每个测试样本进行预测
    for idx in range(len(temp_test_df)):
        row = temp_test_df.iloc[idx]
        true_label = y_test[idx]
        
        # 准备当前数据用于模型预测
        current_data = {
            'price_current': row['price_current'],
            'grid_lower': row['grid_lower'],
            'grid_upper': row['grid_upper'],
            'atr': row['atr'],
            'rsi_1m': row['rsi_1m'],
            'rsi_5m': row['rsi_5m'],
            'buffer': row['buffer'],
            'threshold': row['threshold'],
            'near_lower': row['near_lower'],
            'rsi_ok': row['rsi_ok']
        }
        
        # 使用策略进行预测
        if hasattr(strategy, 'predict_action'):
            try:
                pred_action, confidence = strategy.predict_action(current_data)
                if pred_action == true_label:
                    correct += 1
                total += 1
            except Exception as e:
                print(f"    ⚠️ Prediction error: {e}")
                continue
        else:
            continue
    
    accuracy = correct / total if total > 0 else 0
    
    print(f"   ✅ {model_name} Test Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"   📊 Model Parameters: {model_params:,}")
    
    # 清理内存
    del strategy, temp_df, temp_test_df
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description='Test specific trading model with extended data')
    parser.add_argument('--model', type=str, required=True, 
                        choices=['llm', 'large_model', 'large_transformer', 'enhanced_transformer', 'huge_transformer'],
                        help='Name of the model to test')
    
    args = parser.parse_args()
    
    test_specific_model(args.model)


if __name__ == "__main__":
    main()