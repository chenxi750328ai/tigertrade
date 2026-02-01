#!/usr/bin/env python3
"""
测试收益率回归训练效果
对比分类方法和回归方法的性能
"""

import sys
import os
sys.path.insert(0, '/home/cx/tigertrade')

import pandas as pd
import numpy as np
from datetime import datetime
from src.strategies.llm_strategy import LLMTradingStrategy

def load_training_data():
    """加载训练数据"""
    data_dirs = [
        '/home/cx/trading_data',
        '/home/cx/tigertrade/trading_data'
    ]
    
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            import glob
            files = glob.glob(os.path.join(data_dir, '**/training_data_from_klines_*.csv'), recursive=True)
            if files:
                latest_file = max(files, key=os.path.getmtime)
                df = pd.read_csv(latest_file)
                print(f"✅ 从 {latest_file} 加载数据: {len(df)} 条")
                return df
    
    raise FileNotFoundError("未找到训练数据文件")

def calculate_profit_based_accuracy(predictions, labels, prices, look_ahead=10):
    """计算基于收益的准确率"""
    total_profit = 0.0
    max_possible_profit = 0.0
    
    for i in range(len(predictions)):
        if i + look_ahead >= len(prices):
            break
        
        current_price = prices[i]
        future_prices = prices[i+1:i+look_ahead+1]
        
        # 计算所有动作的收益
        profits = {
            0: 0.0,  # 不操作收益为0
            1: (max(future_prices) - current_price) / current_price,  # 买入收益
            2: (current_price - min(future_prices)) / current_price   # 卖出收益
        }
        
        # 最优动作的收益
        best_action = max(profits, key=profits.get)
        max_possible_profit += profits[best_action]
        
        # 预测动作的收益
        predicted_action = predictions[i]
        total_profit += profits[predicted_action]
    
    # 收益加权准确率
    if max_possible_profit > 0:
        profit_accuracy = total_profit / max_possible_profit
    else:
        profit_accuracy = 0.0
    
    return profit_accuracy

def evaluate_model(strategy, df, seq_length=10):
    """评估模型性能"""
    look_ahead = 10
    min_required = seq_length + look_ahead
    
    predictions = []
    labels = []
    prices = []
    predicted_profits = []
    
    for i in range(min_required, len(df) - look_ahead):
        # 准备序列特征
        try:
            sequence = strategy.prepare_sequence_features(df, i, seq_length)
            sequence_tensor = torch.tensor([sequence], dtype=torch.float32).to(strategy.device)
            
            # 预测
            with torch.no_grad():
                strategy.model.eval()
                model_output = strategy.model(sequence_tensor)
                
                # 解析输出
                if isinstance(model_output, tuple):
                    if len(model_output) == 3:
                        action_logits, profit, grid_adjustment = model_output
                        predicted_profit = profit.item()
                    else:
                        action_logits = model_output[0]
                        predicted_profit = 0.0
                else:
                    action_logits = model_output
                    predicted_profit = 0.0
                
                probs = torch.softmax(action_logits, dim=1).cpu().numpy()[0]
                action = np.argmax(probs)
                
                predictions.append(action)
                predicted_profits.append(predicted_profit)
                
                # 计算标签
                current_price = df.iloc[i]['price_current']
                future_prices = df.iloc[i+1:i+look_ahead+1]['price_current'].values
                
                buy_profit = (max(future_prices) - current_price) / current_price
                sell_profit = (current_price - min(future_prices)) / current_price
                
                profit_threshold = 0.005
                min_diff = 0.003
                
                if abs(buy_profit - sell_profit) >= min_diff:
                    if buy_profit > sell_profit and buy_profit > profit_threshold:
                        label = 1
                    elif sell_profit > buy_profit and sell_profit > profit_threshold:
                        label = 2
                    else:
                        label = 0
                else:
                    label = 0
                
                labels.append(label)
                prices.append(current_price)
                
        except Exception as e:
            continue
    
    # 计算准确率
    accuracy = np.mean(np.array(predictions) == np.array(labels))
    
    # 计算收益加权准确率
    profit_accuracy = calculate_profit_based_accuracy(predictions, labels, prices, look_ahead)
    
    return {
        'accuracy': accuracy,
        'profit_accuracy': profit_accuracy,
        'predictions': predictions,
        'labels': labels,
        'predicted_profits': predicted_profits
    }

def main():
    print("=" * 80)
    print("收益率回归训练效果测试")
    print("=" * 80)
    
    # 加载数据
    df = load_training_data()
    
    # 分割数据（80%训练，20%测试）
    split_idx = int(len(df) * 0.8)
    df_train = df[:split_idx].copy()
    df_test = df[split_idx:].copy()
    
    print(f"\n📊 数据分割: 训练集 {len(df_train)} 条, 测试集 {len(df_test)} 条")
    
    # 测试1: 分类方法（不预测收益率）
    print("\n" + "=" * 80)
    print("测试1: 分类方法（不预测收益率）")
    print("=" * 80)
    
    strategy_classification = LLMTradingStrategy(mode='hybrid', predict_profit=False)
    
    print("\n📊 开始训练分类模型...")
    start_time = datetime.now()
    strategy_classification.train_model(
        df_train,
        seq_length=10,
        max_epochs=20,  # 快速测试
        patience=5,
        train_grid_adjustment=True
    )
    training_time_classification = (datetime.now() - start_time).total_seconds()
    
    print("\n📊 评估分类模型...")
    import torch
    results_classification = evaluate_model(strategy_classification, df_test, seq_length=10)
    
    print(f"\n✅ 分类方法结果:")
    print(f"   训练时间: {training_time_classification:.2f} 秒")
    print(f"   传统准确率: {results_classification['accuracy']:.4f}")
    print(f"   收益加权准确率: {results_classification['profit_accuracy']:.4f}")
    
    # 测试2: 收益率回归方法
    print("\n" + "=" * 80)
    print("测试2: 收益率回归方法（预测收益率）")
    print("=" * 80)
    
    strategy_profit = LLMTradingStrategy(mode='hybrid', predict_profit=True)
    
    print("\n📊 开始训练收益率回归模型...")
    start_time = datetime.now()
    strategy_profit.train_model(
        df_train,
        seq_length=10,
        max_epochs=20,  # 快速测试
        patience=5,
        train_grid_adjustment=True
    )
    training_time_profit = (datetime.now() - start_time).total_seconds()
    
    print("\n📊 评估收益率回归模型...")
    results_profit = evaluate_model(strategy_profit, df_test, seq_length=10)
    
    print(f"\n✅ 收益率回归方法结果:")
    print(f"   训练时间: {training_time_profit:.2f} 秒")
    print(f"   传统准确率: {results_profit['accuracy']:.4f}")
    print(f"   收益加权准确率: {results_profit['profit_accuracy']:.4f}")
    
    # 对比结果
    print("\n" + "=" * 80)
    print("对比结果")
    print("=" * 80)
    
    print(f"\n📊 性能对比:")
    print(f"   训练时间:")
    print(f"     分类方法: {training_time_classification:.2f} 秒")
    print(f"     收益率回归: {training_time_profit:.2f} 秒")
    print(f"     差异: {training_time_profit - training_time_classification:.2f} 秒")
    
    print(f"\n   传统准确率:")
    print(f"     分类方法: {results_classification['accuracy']:.4f}")
    print(f"     收益率回归: {results_profit['accuracy']:.4f}")
    print(f"     差异: {results_profit['accuracy'] - results_classification['accuracy']:.4f}")
    
    print(f"\n   收益加权准确率（关键指标）:")
    print(f"     分类方法: {results_classification['profit_accuracy']:.4f}")
    print(f"     收益率回归: {results_profit['profit_accuracy']:.4f}")
    improvement = results_profit['profit_accuracy'] - results_classification['profit_accuracy']
    print(f"     改进: {improvement:+.4f} ({improvement/results_classification['profit_accuracy']*100:+.2f}%)")
    
    # 保存结果
    results_file = f"/home/cx/tigertrade/docs/profit_regression_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("# 收益率回归训练效果对比\n\n")
        f.write(f"**测试时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## 测试结果\n\n")
        f.write("### 分类方法（不预测收益率）\n\n")
        f.write(f"- 训练时间: {training_time_classification:.2f} 秒\n")
        f.write(f"- 传统准确率: {results_classification['accuracy']:.4f}\n")
        f.write(f"- 收益加权准确率: {results_classification['profit_accuracy']:.4f}\n\n")
        f.write("### 收益率回归方法（预测收益率）\n\n")
        f.write(f"- 训练时间: {training_time_profit:.2f} 秒\n")
        f.write(f"- 传统准确率: {results_profit['accuracy']:.4f}\n")
        f.write(f"- 收益加权准确率: {results_profit['profit_accuracy']:.4f}\n\n")
        f.write("## 对比分析\n\n")
        f.write(f"- 收益加权准确率改进: {improvement:+.4f} ({improvement/results_classification['profit_accuracy']*100:+.2f}%)\n")
    
    print(f"\n✅ 结果已保存到: {results_file}")

if __name__ == '__main__':
    import torch
    main()
