import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
from datetime import datetime, timedelta
import glob
import argparse
import sys
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


class PerformanceValidator:
    """性能验证器 - 用于训练和评估各种交易策略的性能"""
    
    def __init__(self, data_dir="/home/cx/trading_data"):
        self.data_dir = data_dir
        self.results = {}
    
    def load_historical_data(self, days=30):
        """加载历史数据"""
        all_data_files = []
        
        # 获取最近几天的数据目录
        date_dirs = glob.glob(os.path.join(self.data_dir, '202*-*-*'))
        if not date_dirs:
            print("❌ 没有找到历史数据")
            return None
            
        # 按日期排序，获取最近days天的数据
        sorted_dirs = sorted(date_dirs, reverse=True)[:days]
        
        for data_dir in sorted_dirs:
            data_files = glob.glob(os.path.join(data_dir, 'trading_data_*.csv'))
            all_data_files.extend(data_files)
        
        if not all_data_files:
            print("❌ 没有找到交易数据文件")
            return None
        
        # 按修改时间排序，获取最新的文件
        all_data_files = sorted(all_data_files, key=os.path.getmtime, reverse=True)
        
        # 合并所有数据文件
        all_data = []
        for file_path in all_data_files:
            try:
                df = pd.read_csv(file_path)
                all_data.append(df)
            except Exception as e:
                print(f"❌ 读取文件 {file_path} 失败: {e}")
        
        if not all_data:
            print("❌ 没有成功读取任何数据文件")
            return None
        
        # 合并所有数据
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"📊 加载了 {len(combined_df)} 条历史数据")
        
        # 清理数据
        combined_df = combined_df.dropna(subset=['price_current', 'grid_lower', 'grid_upper', 'atr', 'rsi_1m', 'rsi_5m'])
        print(f"📊 清理后剩余 {len(combined_df)} 条有效数据")
        
        return combined_df
    
    def prepare_features_and_labels(self, df, look_ahead=10):
        """准备特征和标签"""
        X, y = [], []
        
        for i in range(len(df) - look_ahead):
            row = df.iloc[i]
            
            # 准备特征
            features = [
                row['price_current'],
                row['grid_lower'],
                row['grid_upper'],
                row['atr'],
                row['rsi_1m'] if pd.notna(row['rsi_1m']) else 50,
                row['rsi_5m'] if pd.notna(row['rsi_5m']) else 50,
                row['buffer'],
                row['threshold'],
                1 if row['near_lower'] else 0,
                1 if row['rsi_ok'] else 0
            ]
            
            # 归一化特征
            features_np = np.array(features)
            mean_val = np.mean(features_np)
            std_val = np.std(features_np) + 1e-8
            normalized_features = (features_np - mean_val) / std_val
            X.append(normalized_features.tolist())
            
            # 计算未来look_ahead步的盈利
            current_price = row['price_current']
            future_prices = df.iloc[i+1:i+look_ahead+1]['price_current'].values
            
            if len(future_prices) == 0:
                continue
                
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
        
        return np.array(X), np.array(y)
    
    def evaluate_strategy(self, strategy_class, strategy_name, X, y):
        """评估策略性能"""
        print(f"\n🚀 开始评估 {strategy_name} 策略...")
        
        try:
            # 初始化策略
            strategy = strategy_class()
            
            # 分割数据集
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            print(f"📈 训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
            
            # 创建一个临时DataFrame用于训练
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
            
            # 训练模型（如果策略支持）
            if hasattr(strategy, 'train_model'):
                try:
                    strategy.train_model(temp_df)
                except Exception as e:
                    print(f"⚠️ 训练 {strategy_name} 时出现问题: {e}")
            
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
                    except:
                        # 如果预测失败，跳过这个样本
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
            
            print(f"   ✅ {strategy_name} 测试准确率: {accuracy:.4f} ({correct}/{total})")
            
            # 记录结果
            self.results[strategy_name] = {
                'accuracy': accuracy,
                'total_samples': total,
                'correct_predictions': correct
            }
            
        except Exception as e:
            print(f"❌ 评估 {strategy_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
    def run_validation(self):
        """运行完整验证"""
        print("🔍 开始加载历史数据...")
        
        # 加载历史数据
        df = self.load_historical_data(days=30)
        if df is None or len(df) < 50:
            print("❌ 历史数据不足，无法进行性能验证")
            return
        
        print("🔍 准备特征和标签...")
        X, y = self.prepare_features_and_labels(df)
        
        if len(X) < 50:
            print("❌ 准备的数据不足，无法进行性能验证")
            return
        
        print(f"📊 准备好的数据: 特征矩阵 {X.shape}, 标签向量 {y.shape}")
        
        # 评估各种策略
        strategies_to_test = [
            (LLMTradingStrategy, "LLM Trading"),
            (EnhancedTransformerStrategy, "Enhanced Transformer"),
            (LargeTransformerStrategy, "Large Transformer"),
            (ModelComparisonStrategy, "Model Comparison"),
            (LargeModelStrategy, "Large Model"),
            (HugeTransformerStrategy, "Huge Transformer")
        ]
        
        # 尝试评估强化学习策略
        try:
            # RL策略需要特殊处理
            self.evaluate_rl_strategy(X, y)
        except Exception as e:
            print(f"❌ 评估 RL 策略时出错: {e}")
        
        # 评估其他策略
        for strategy_class, strategy_name in strategies_to_test:
            try:
                self.evaluate_strategy(strategy_class, strategy_name, X, y)
            except Exception as e:
                print(f"❌ 评估 {strategy_name} 时出错: {e}")
        
        # 输出汇总结果
        self.print_results_summary()
    
    def evaluate_rl_strategy(self, X, y):
        """评估RL策略"""
        print(f"\n🚀 开始评估 RL Trading 策略...")
        
        try:
            # 初始化RL策略
            rl_strategy = RLTradingStrategy()
            
            # 分割数据集
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            print(f"📈 训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
            
            # 创建一个临时DataFrame用于训练
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
                except:
                    # 如果预测失败，跳过这个样本
                    continue
            
            accuracy = correct / total if total > 0 else 0
            
            print(f"   ✅ RL Trading 测试准确率: {accuracy:.4f} ({correct}/{total})")
            
            # 记录结果
            self.results["RL Trading"] = {
                'accuracy': accuracy,
                'total_samples': total,
                'correct_predictions': correct
            }
            
        except Exception as e:
            print(f"❌ 评估 RL Trading 策略时出错: {e}")
            import traceback
            traceback.print_exc()
    
    def print_results_summary(self):
        """打印结果汇总"""
        print("\n" + "="*60)
        print("📊 策略性能汇总")
        print("="*60)
        
        if not self.results:
            print("❌ 没有可用的结果")
            return
        
        # 按准确率排序
        sorted_results = sorted(self.results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        print(f"{'策略名称':<25} {'准确率':<10} {'样本数':<10}")
        print("-"*50)
        
        for name, metrics in sorted_results:
            print(f"{name:<25} {metrics['accuracy']:<10.4f} {metrics['total_samples']:<10}")


def main():
    parser = argparse.ArgumentParser(description='验证各种交易策略的性能')
    parser.add_argument('--data_dir', type=str, default='/home/cx/trading_data',
                        help='数据目录路径')
    parser.add_argument('--days', type=int, default=30,
                        help='加载历史数据的天数')
    
    args = parser.parse_args()
    
    validator = PerformanceValidator(data_dir=args.data_dir)
    validator.run_validation()


if __name__ == "__main__":
    main()