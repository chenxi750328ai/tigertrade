#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
公平的模型对比测试
在相同数据、相同训练配置下对比LSTM和Transformer
"""

import sys
import os
sys.path.insert(0, '/home/cx/tigertrade')

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datetime import datetime
import json
from pathlib import Path

from src.strategies.llm_strategy import LLMTradingStrategy
from src.strategies.large_transformer_strategy import LargeTransformerStrategy


class FairModelComparison:
    """公平的模型对比测试"""
    
    def __init__(self, data_file=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_file = data_file
        self.results = {}
        
    def load_data(self):
        """加载训练数据"""
        if self.data_file and os.path.exists(self.data_file):
            df = pd.read_csv(self.data_file)
            print(f"✅ 从 {self.data_file} 加载数据: {len(df)} 条")
            return df
        
        # 尝试从默认位置加载
        data_dirs = [
            '/home/cx/trading_data',
            '/home/cx/tigertrade/trading_data'
        ]
        
        for data_dir in data_dirs:
            if os.path.exists(data_dir):
                # 查找最新的训练数据文件
                import glob
                files = glob.glob(os.path.join(data_dir, '**/training_data_from_klines_*.csv'), recursive=True)
                if files:
                    latest_file = max(files, key=os.path.getmtime)
                    df = pd.read_csv(latest_file)
                    print(f"✅ 从 {latest_file} 加载数据: {len(df)} 条")
                    return df
        
        raise FileNotFoundError("未找到训练数据文件")
    
    def calculate_profit_based_accuracy(self, predictions, labels, prices, grid_params, look_ahead=10):
        """
        计算基于收益的准确率
        
        Args:
            predictions: 预测动作
            labels: 真实标签
            prices: 价格序列
            grid_params: 网格参数
            look_ahead: 向前看的步数
        
        Returns:
            profit_accuracy: 收益加权准确率
        """
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
    
    def train_and_evaluate(self, model_type='lstm', seq_length=10, epochs=50, hidden_size=64):
        """
        训练和评估模型
        
        Args:
            model_type: 'lstm' 或 'transformer'
            seq_length: 序列长度
            epochs: 训练轮次
            hidden_size: 隐藏层大小
        """
        print(f"\n{'='*80}")
        print(f"训练 {model_type.upper()} 模型")
        print(f"序列长度: {seq_length}, 训练轮次: {epochs}, 隐藏层: {hidden_size}")
        print(f"{'='*80}")
        
        # 加载数据
        df = self.load_data()
        
        # 初始化模型
        if model_type == 'lstm':
            strategy = LLMTradingStrategy(mode='hybrid')
            strategy._seq_length = seq_length
        elif model_type == 'transformer':
            strategy = LargeTransformerStrategy()
            # Transformer策略的序列长度在prepare_features中处理
        else:
            raise ValueError(f"未知的模型类型: {model_type}")
        
        # 训练模型
        print(f"\n📊 开始训练...")
        start_time = datetime.now()
        
        if model_type == 'lstm':
            strategy.train_model(df, seq_length=seq_length, max_epochs=epochs, 
                                patience=10, train_grid_adjustment=True)
        else:
            # Transformer策略使用相同的训练配置
            # 注意：Transformer的train_model方法需要修改以支持序列长度
            strategy.train_model(df)  # Transformer内部会使用序列长度10
        
        training_time = (datetime.now() - start_time).total_seconds()
        print(f"✅ 训练完成，耗时: {training_time:.2f}秒")
        
        # 评估模型
        print(f"\n📊 开始评估...")
        
        # 准备测试数据
        look_ahead = 10
        min_required = seq_length + look_ahead
        
        X_test = []
        y_test = []
        prices_test = []
        
        for i in range(min_required, len(df)):
            # 检查数据是否足够
            if i + look_ahead >= len(df):
                break
            
            # 两种模型都使用相同的特征准备方式
            try:
                if hasattr(strategy, 'prepare_sequence_features'):
                    sequence = strategy.prepare_sequence_features(df, i, seq_length)
                elif model_type == 'transformer':
                    # Transformer需要序列特征，手动构建
                    sequence_features = []
                    for j in range(max(0, i - seq_length + 1), i + 1):
                        row = df.iloc[j]
                        features = strategy.prepare_features(row)
                        sequence_features.append(features)
                    # 如果序列不足，用第一个值填充
                    while len(sequence_features) < seq_length:
                        if sequence_features:
                            sequence_features.insert(0, sequence_features[0])
                        else:
                            sequence_features.insert(0, [0.0] * 12)
                    sequence = np.array(sequence_features[-seq_length:], dtype=np.float32)
                else:
                    # 如果没有prepare_sequence_features，使用prepare_features
                    row = df.iloc[i]
                    features = strategy.prepare_features(row)
                    # 构建序列（重复当前特征）
                    sequence = np.array([features] * seq_length, dtype=np.float32)
                
                # 计算标签
                current_price = df.iloc[i]['price_current']
                future_prices = df.iloc[i+1:i+look_ahead+1]['price_current'].values
                
                if len(future_prices) == 0:
                    continue
                
                max_future_price = max(future_prices)
                min_future_price = min(future_prices)
                
                buy_profit = (max_future_price - current_price) / current_price
                sell_profit = (current_price - min_future_price) / current_price
                
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
                
                # 只有所有数据都准备好后才添加
                X_test.append(sequence)
                y_test.append(label)
                prices_test.append(current_price)
            except Exception as e:
                print(f"⚠️ 处理第 {i} 条数据时出错: {e}，跳过")
                continue
        
        # 预测
        predictions = []
        strategy.model.eval()
        
        with torch.no_grad():
            for x in X_test:
                try:
                    x_tensor = torch.tensor([x], dtype=torch.float32).to(strategy.device)
                    # 确保维度正确: (batch, seq, features)
                    if len(x_tensor.shape) == 2:
                        # (seq, features) -> (1, seq, features)
                        x_tensor = x_tensor.unsqueeze(0)
                    elif len(x_tensor.shape) == 3:
                        # 已经是正确的形状 (batch, seq, features)
                        pass
                    else:
                        # 需要调整维度
                        x_tensor = x_tensor.view(1, -1, x_tensor.shape[-1])
                    
                    output = strategy.model(x_tensor)
                    
                    # 处理模型输出
                    if isinstance(output, tuple):
                        action_logits, _ = output
                    else:
                        action_logits = output
                    
                    # 处理softmax输出（Transformer可能已经应用softmax）
                    if len(action_logits.shape) == 2 and action_logits.shape[1] == 3:
                        # 如果是logits，需要argmax
                        action = torch.argmax(action_logits, dim=1).item()
                    elif len(action_logits.shape) == 1 and len(action_logits) == 3:
                        # 如果已经是概率分布
                        action = torch.argmax(action_logits).item()
                    else:
                        # 默认处理
                        action = torch.argmax(action_logits, dim=-1).item()
                    
                    predictions.append(action)
                except Exception as e:
                    print(f"⚠️ 预测错误: {e}, 使用默认动作0")
                    predictions.append(0)
        
        # 确保predictions和y_test长度一致
        min_len = min(len(predictions), len(y_test), len(prices_test))
        if len(predictions) != min_len or len(y_test) != min_len:
            print(f"⚠️ 警告: predictions({len(predictions)})和y_test({len(y_test)})长度不一致，调整到最小长度 {min_len}")
            predictions = predictions[:min_len]
            y_test = y_test[:min_len]
            prices_test = prices_test[:min_len]
        
        # 计算指标
        predictions = np.array(predictions)
        y_test = np.array(y_test)
        prices_test = np.array(prices_test)
        
        # 传统准确率
        accuracy = (predictions == y_test).mean() if len(predictions) > 0 else 0.0
        
        # 收益加权准确率
        profit_accuracy = self.calculate_profit_based_accuracy(
            predictions, y_test, prices_test, None, look_ahead
        )
        
        # 保存结果
        result = {
            'model_type': model_type,
            'seq_length': seq_length,
            'epochs': epochs,
            'hidden_size': hidden_size,
            'training_time': training_time,
            'accuracy': float(accuracy),
            'profit_accuracy': float(profit_accuracy),
            'num_params': sum(p.numel() for p in strategy.model.parameters()),
            'predictions': predictions.tolist(),
            'labels': y_test.tolist()
        }
        
        self.results[f"{model_type}_{seq_length}"] = result
        
        print(f"\n📊 评估结果:")
        print(f"  传统准确率: {accuracy:.4f}")
        print(f"  收益加权准确率: {profit_accuracy:.4f}")
        print(f"  参数量: {result['num_params']:,}")
        print(f"  训练时间: {training_time:.2f}秒")
        
        return result
    
    def run_comparison(self, seq_lengths=[10, 50, 100], epochs=50):
        """
        运行完整的对比测试
        
        Args:
            seq_lengths: 要测试的序列长度列表
            epochs: 训练轮次
        """
        print(f"\n{'='*80}")
        print(f"开始公平模型对比测试")
        print(f"序列长度: {seq_lengths}")
        print(f"训练轮次: {epochs}")
        print(f"{'='*80}")
        
        all_results = []
        
        for seq_length in seq_lengths:
            # 训练LSTM
            lstm_result = self.train_and_evaluate(
                model_type='lstm',
                seq_length=seq_length,
                epochs=epochs
            )
            all_results.append(lstm_result)
            
            # 训练Transformer
            transformer_result = self.train_and_evaluate(
                model_type='transformer',
                seq_length=seq_length,
                epochs=epochs
            )
            all_results.append(transformer_result)
        
        # 保存结果
        output_file = f"/home/cx/tigertrade/docs/fair_comparison_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 结果已保存到: {output_file}")
        
        # 打印对比表格
        self.print_comparison_table(all_results)
        
        return all_results
    
    def print_comparison_table(self, results):
        """打印对比表格"""
        print(f"\n{'='*80}")
        print(f"模型对比结果")
        print(f"{'='*80}")
        print(f"{'模型':<15} {'序列长度':<10} {'准确率':<10} {'收益准确率':<12} {'参数量':<15} {'训练时间':<10}")
        print(f"{'-'*80}")
        
        for result in results:
            print(f"{result['model_type']:<15} {result['seq_length']:<10} "
                  f"{result['accuracy']:<10.4f} {result['profit_accuracy']:<12.4f} "
                  f"{result['num_params']:<15,} {result['training_time']:<10.2f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='公平的模型对比测试')
    parser.add_argument('--data-file', type=str, help='训练数据文件路径')
    parser.add_argument('--seq-lengths', type=int, nargs='+', default=[10, 50, 100],
                        help='要测试的序列长度列表')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮次')
    
    args = parser.parse_args()
    
    comparator = FairModelComparison(data_file=args.data_file)
    results = comparator.run_comparison(seq_lengths=args.seq_lengths, epochs=args.epochs)
    
    print("\n✅ 对比测试完成")
