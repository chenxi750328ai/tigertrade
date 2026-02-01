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
warnings.filterwarnings("ignore")

# 导入各种策略
from llm_strategy import LLMTradingStrategy
from enhanced_transformer_strategy import EnhancedTransformerStrategy
from large_transformer_strategy import LargeTransformerStrategy
from model_comparison_strategy import ModelComparisonStrategy
from rl_trading_strategy import RLTradingStrategy
from large_model_strategy import LargeModelStrategy
from huge_transformer_strategy import HugeTransformerStrategy
from data_fetcher import aggregate_data_for_training, prepare_features_from_raw_data


class ExtendedTrainingValidator:
    """扩展训练验证器 - 使用更多数据训练和评估各种交易策略的性能"""
    
    def __init__(self, data_dir="/home/cx/trading_data"):
        self.data_dir = data_dir
        self.results = {}
    
    def prepare_features_and_labels(self, df, look_ahead=10):
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
                profit_threshold = 0.002  # 0.2%的阈值
                
                if buy_profit > profit_threshold and buy_profit > sell_profit:
                    label = 1  # 买入
                elif sell_profit > profit_threshold and sell_profit > buy_profit:
                    label = 2  # 卖出
                else:
                    label = 0  # 不操作
            
            y.append(label)
        
        # 对于最后look_ahead个数据点，复制最后一个标签
        for _ in range(min(look_ahead, len(X) - len(y))):
            y.append(y[-1] if y else 0)
        
        # 确保X和y长度一致
        X = X[:len(y)]
        
        return np.array(X), np.array(y)
    
    def train_and_evaluate_strategy(self, strategy_class, strategy_name, X, y):
        """训练并评估策略"""
        print(f"\n🚀 Training and evaluating {strategy_name} strategy...")
        
        try:
            # 初始化策略
            strategy = strategy_class()
            
            # 分割数据集
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            print(f"📈 Training set size: {len(X_train)}, Test set size: {len(X_test)}")
            
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
                print(f"   📊 Starting training for {strategy_name}...")
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
                        print(f"    ⚠️ Prediction error for {strategy_name}: {e}")
                        continue
                else:
                    # 对于ModelComparisonStrategy等特殊策略
                    if strategy_name == "Model Comparison":
                        try:
                            predictions = strategy.predict_both_models(current_data)
                            # 使用LSTM的预测结果进行评估
                            pred_action = predictions['lstm']['action']
                            if pred_action == true_label:
                                correct += 1
                            total += 1
                        except:
                            continue
                    else:
                        continue
            
            accuracy = correct / total if total > 0 else 0
            
            print(f"   ✅ {strategy_name} Test Accuracy: {accuracy:.4f} ({correct}/{total})")
            
            # 记录结果
            self.results[strategy_name] = {
                'accuracy': accuracy,
                'total_samples': total,
                'correct_predictions': correct
            }
            
        except Exception as e:
            print(f"❌ Error evaluating {strategy_name}: {e}")
            import traceback
            traceback.print_exc()
    
    def run_extended_validation(self):
        """运行扩展验证"""
        print("🔍 Loading extended historical data...")
        
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
        X, y = self.prepare_features_and_labels(features_df)
        
        if len(X) < 1000:
            print("❌ Insufficient prepared data for training")
            return
        
        print(f"📊 Prepared data: Feature matrix {X.shape}, Label vector {y.shape}")
        
        # 评估各种策略
        strategies_to_test = [
            (LLMTradingStrategy, "LLM Trading"),
            (LargeModelStrategy, "Large Model"),
            (LargeTransformerStrategy, "Large Transformer"),
            (EnhancedTransformerStrategy, "Enhanced Transformer"),
            (HugeTransformerStrategy, "Huge Transformer")
        ]
        
        # 评估RL策略
        try:
            self.train_and_evaluate_rl_strategy(X, y)
        except Exception as e:
            print(f"❌ Error evaluating RL strategy: {e}")
        
        # 评估Model Comparison策略
        try:
            self.train_and_evaluate_strategy(ModelComparisonStrategy, "Model Comparison", X, y)
        except Exception as e:
            print(f"❌ Error evaluating Model Comparison strategy: {e}")
        
        # 评估其他策略
        for strategy_class, strategy_name in strategies_to_test:
            try:
                self.train_and_evaluate_strategy(strategy_class, strategy_name, X, y)
            except Exception as e:
                print(f"❌ Error evaluating {strategy_name}: {e}")
        
        # 输出汇总结果
        self.print_results_summary()
    
    def train_and_evaluate_rl_strategy(self, X, y):
        """训练并评估RL策略"""
        print(f"\n🚀 Training and evaluating RL Trading strategy...")
        
        try:
            # 初始化RL策略
            rl_strategy = RLTradingStrategy()
            
            # 分割数据集
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            print(f"📈 Training set size: {len(X_train)}, Test set size: {len(X_test)}")
            
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
            
            # 训练RL模型
            rl_strategy.train_model(temp_df)
            
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
                
                try:
                    pred_action, confidence = rl_strategy.predict_action(current_data)
                    if pred_action == true_label:
                        correct += 1
                    total += 1
                except Exception as e:
                    print(f"    ⚠️ Prediction error for RL: {e}")
                    continue
            
            accuracy = correct / total if total > 0 else 0
            
            print(f"   ✅ RL Trading Test Accuracy: {accuracy:.4f} ({correct}/{total})")
            
            # 记录结果
            self.results["RL Trading"] = {
                'accuracy': accuracy,
                'total_samples': total,
                'correct_predictions': correct
            }
            
        except Exception as e:
            print(f"❌ Error evaluating RL Trading strategy: {e}")
            import traceback
            traceback.print_exc()
    
    def print_results_summary(self):
        """打印结果汇总"""
        print("\n" + "="*70)
        print("📊 Extended Dataset Strategy Performance Summary")
        print("="*70)
        
        if not self.results:
            print("❌ No results available")
            return
        
        # 按准确率排序
        sorted_results = sorted(self.results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        print(f"{'Strategy Name':<25} {'Accuracy':<10} {'Samples':<10} {'Correct':<10}")
        print("-"*60)
        
        for name, metrics in sorted_results:
            print(f"{name:<25} {metrics['accuracy']:<10.4f} {metrics['total_samples']:<10} {metrics['correct_predictions']:<10}")
        
        print(f"\n📈 Total strategies evaluated: {len(self.results)}")


def main():
    parser = argparse.ArgumentParser(description='Extended training and validation using more data')
    parser.add_argument('--data_dir', type=str, default='/home/cx/trading_data',
                        help='Data directory path')
    parser.add_argument('--days', type=int, default=30,
                        help='Number of days of historical data to use')
    
    args = parser.parse_args()
    
    validator = ExtendedTrainingValidator(data_dir=args.data_dir)
    validator.run_extended_validation()


if __name__ == "__main__":
    main()