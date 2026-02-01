#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
动态序列长度测试器
自动测试不同序列长度，找到最优序列长度
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import matplotlib.pyplot as plt
import glob

sys.path.insert(0, '/home/cx/tigertrade')

try:
    from src.strategies.llm_strategy import TradingLSTM, LLMTradingStrategy
except ImportError:
    print("⚠️ 无法导入策略模块")


class SequenceLengthTester:
    """动态序列长度测试器"""
    
    def __init__(self, data_dir='/home/cx/trading_data', 
                 min_length=10, max_length=500, step=50,  # 增大步长以加快测试
                 convergence_window=3, convergence_threshold=0.02):  # 放宽收敛条件
        """
        初始化测试器
        
        Args:
            data_dir: 数据目录
            min_length: 最小序列长度
            max_length: 最大序列长度
            step: 测试步长
            convergence_window: 收敛检测窗口大小
            convergence_threshold: 收敛阈值（相对变化率）
        """
        self.data_dir = data_dir
        self.min_length = min_length
        self.max_length = max_length
        self.step = step
        self.convergence_window = convergence_window
        self.convergence_threshold = convergence_threshold
        self.results = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def prepare_features(self, row):
        """准备单个时间点的特征"""
        try:
            features = [
                row['price_current'],
                row['atr'],
                row['rsi_1m'] if pd.notna(row['rsi_1m']) else 50,
                row['rsi_5m'] if pd.notna(row['rsi_5m']) else 50,
                row.get('boll_upper', 0),
                row.get('boll_mid', 0),
                row.get('boll_lower', 0),
                row.get('boll_position', 0.5),
                row.get('price_change_1', 0),
                row.get('price_change_5', 0),
                row.get('volatility', 0),
                row.get('volume_1m', 0)
            ]
            # 归一化
            features_np = np.array(features)
            mean_val = np.mean(features_np)
            std_val = np.std(features_np) + 1e-8
            normalized_features = (features_np - mean_val) / std_val
            return normalized_features.tolist()
        except Exception as e:
            return [0.0] * 12
    
    def prepare_sequence_features(self, df, current_idx, seq_length):
        """
        准备历史序列特征
        
        Args:
            df: 数据框
            current_idx: 当前索引
            seq_length: 序列长度
        
        Returns:
            sequence: (seq_length, 12) 的数组
        """
        start_idx = max(0, current_idx - seq_length + 1)
        sequence_df = df.iloc[start_idx:current_idx+1]
        
        sequences = []
        for _, row in sequence_df.iterrows():
            features = self.prepare_features(row)
            sequences.append(features)
        
        # 如果序列不足seq_length，用第一个值填充
        while len(sequences) < seq_length:
            if sequences:
                sequences.insert(0, sequences[0])
            else:
                sequences.insert(0, [0.0] * 12)
        
        return np.array(sequences, dtype=np.float32)
    
    def load_training_data(self):
        """加载训练数据（优先使用包含price_current的文件）"""
        data_files = []
        
        # 1. 优先查找从K线生成的数据文件（数据量大）
        kline_data_files = glob.glob(os.path.join(self.data_dir, 'training_data_from_klines_*.csv'))
        for csv_file in kline_data_files:
            try:
                df_test = pd.read_csv(csv_file, nrows=1)
                if 'price_current' in df_test.columns:
                    total_rows = len(pd.read_csv(csv_file))
                    data_files.append((csv_file, total_rows))
            except:
                pass
        
        # 2. 查找包含price_current的数据文件（最新的trading_data文件）
        date_dirs = [d for d in os.listdir(self.data_dir) 
                     if os.path.isdir(os.path.join(self.data_dir, d)) and d.startswith('2026')]
        date_dirs.sort(reverse=True)
        
        for date_dir in date_dirs[:7]:  # 最近7天
            date_path = os.path.join(self.data_dir, date_dir)
            csv_files = glob.glob(os.path.join(date_path, 'trading_data_*.csv'))
            for csv_file in csv_files:
                try:
                    df_test = pd.read_csv(csv_file, nrows=1)
                    if 'price_current' in df_test.columns:
                        total_rows = len(pd.read_csv(csv_file))
                        data_files.append((csv_file, total_rows))
                except:
                    pass
        
        # 3. 查找其他训练数据文件（production等）
        if not data_files:
            for root, dirs, files in os.walk(self.data_dir):
                for file in files:
                    if file.endswith('.csv') and ('train' in file.lower() or 'trading_data' in file.lower()):
                        file_path = os.path.join(root, file)
                        try:
                            df_test = pd.read_csv(file_path, nrows=1)
                            if 'price_current' in df_test.columns or 'close' in df_test.columns:
                                data_files.append(file_path)
                        except:
                            pass
        
        if not data_files:
            print("⚠️ 未找到训练数据文件")
            return None
        
        # 使用数据量最大的文件（优先）或最新的文件
        if isinstance(data_files[0], tuple):
            # 如果data_files是(文件路径, 数据量)的列表
            data_files.sort(key=lambda x: x[1], reverse=True)  # 按数据量排序
            latest_file = data_files[0][0]
            print(f"📊 选择数据量最大的文件: {data_files[0][1]}条数据")
        else:
            # 如果data_files是文件路径列表
            latest_file = max(data_files, key=os.path.getmtime)
        print(f"📂 加载训练数据: {latest_file}")
        
        try:
            df = pd.read_csv(latest_file)
            print(f"✅ 加载成功，共{len(df)}条数据")
            
            # 如果数据使用'close'而不是'price_current'，进行转换
            if 'close' in df.columns and 'price_current' not in df.columns:
                df['price_current'] = df['close']
                print("📝 已将'close'列映射为'price_current'")
            
            # 检查必需的列
            required_cols = ['price_current', 'atr', 'rsi_1m', 'rsi_5m']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"⚠️ 缺少列: {missing_cols}，将使用默认值")
                for col in missing_cols:
                    if 'rsi' in col:
                        df[col] = 50.0
                    else:
                        df[col] = 0.0
            
            return df
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def prepare_data_with_sequence(self, df, seq_length, look_ahead=10):
        """
        准备带序列的训练数据
        
        Args:
            df: 原始数据框
            seq_length: 序列长度
            look_ahead: 向前看的步数（用于生成标签）
        
        Returns:
            X: 序列特征 (n_samples, seq_length, 12)
            y: 标签 (n_samples,)
        """
        X, y = [], []
        
        # 需要至少seq_length + look_ahead个数据点
        min_required = seq_length + look_ahead
        
        for i in range(min_required, len(df)):
            # 生成标签（基于未来look_ahead步的盈利）
            current_price = df.iloc[i]['price_current']
            future_prices = df.iloc[i+1:i+look_ahead+1]['price_current'].values
            
            if len(future_prices) < look_ahead:
                continue  # 如果未来数据不足，跳过这个样本
            
            # 准备序列特征（在确认有未来数据后再准备）
            sequence = self.prepare_sequence_features(df, i, seq_length)
            X.append(sequence)
            
            max_future_price = max(future_prices)
            min_future_price = min(future_prices)
            
            buy_profit = (max_future_price - current_price) / current_price
            sell_profit = (current_price - min_future_price) / current_price
            
            profit_threshold = 0.005
            min_diff = 0.003
            
            if abs(buy_profit - sell_profit) >= min_diff:
                if buy_profit > sell_profit and buy_profit > profit_threshold:
                    label = 1  # 买入
                elif sell_profit > buy_profit and sell_profit > profit_threshold:
                    label = 2  # 卖出
                else:
                    label = 0  # 持有
            else:
                label = 0  # 持有
            
            y.append(label)
        
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)
    
    def train_and_evaluate(self, seq_length, X_train, y_train, X_val, y_val):
        """
        训练并评估模型
        
        Args:
            seq_length: 序列长度
            X_train: 训练特征 (n_samples, seq_length, 12)
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
        
        Returns:
            results: 评估结果字典
        """
        # 初始化模型
        model = TradingLSTM(
            input_size=12,
            hidden_size=64,
            num_layers=2,
            output_size=3
        ).to(self.device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        
        # 转换为张量
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(self.device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(self.device)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        # 训练模型（快速测试模式：减少epochs）
        best_val_acc = 0.0
        best_val_loss = float('inf')
        no_improvement = 0
        max_epochs = 15  # 增加到15个epoch以获得更可靠的结果
        patience = 5  # 增加patience
        
        for epoch in range(max_epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # 验证阶段
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            predictions = []
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
                    
                    predictions.extend(predicted.cpu().numpy())
            
            val_acc = correct / total if total > 0 else 0.0
            avg_val_loss = val_loss / len(val_loader)
            
            # 早停
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = avg_val_loss
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement >= patience:
                    break
        
        # 计算预测稳定性（方差）
        prediction_variance = np.var(predictions) if len(predictions) > 0 else 0.0
        
        # 计算综合评分
        composite_score = self.calculate_composite_score({
            'accuracy': best_val_acc,
            'loss': best_val_loss,
            'prediction_variance': prediction_variance
        })
        
        return {
            'accuracy': best_val_acc,
            'loss': best_val_loss,
            'prediction_variance': prediction_variance,
            'composite_score': composite_score
        }
    
    def calculate_composite_score(self, results):
        """计算综合评分"""
        accuracy_score = results['accuracy'] * 0.4
        loss_score = (1 - min(results['loss'], 1.0)) * 0.3
        stability_score = (1 - min(results['prediction_variance'] / 2.0, 1.0)) * 0.3
        
        composite_score = accuracy_score + loss_score + stability_score
        return composite_score
    
    def check_convergence(self):
        """检查是否收敛"""
        if len(self.results) < self.convergence_window * 2:
            return False, None
        
        # 使用综合评分判断收敛
        recent_scores = [r['composite_score'] for r in self.results[-self.convergence_window:]]
        prev_scores = [r['composite_score'] for r in self.results[-self.convergence_window*2:-self.convergence_window]]
        
        recent_avg = np.mean(recent_scores)
        prev_avg = np.mean(prev_scores)
        
        relative_change = abs(recent_avg - prev_avg) / (abs(prev_avg) + 1e-8)
        
        is_converged = relative_change < self.convergence_threshold
        
        if is_converged:
            # 找到性能最高的序列长度
            best_result = max(self.results, key=lambda x: x['composite_score'])
            optimal_length = best_result['seq_length']
        else:
            optimal_length = None
        
        return is_converged, optimal_length
    
    def test_sequence_lengths(self):
        """测试不同序列长度"""
        print("🚀 开始动态序列长度测试")
        print(f"测试范围: {self.min_length} - {self.max_length}, 步长: {self.step}")
        print(f"收敛窗口: {self.convergence_window}, 收敛阈值: {self.convergence_threshold}")
        
        # 加载数据
        df = self.load_training_data()
        if df is None:
            return None, None
        
        # 检查数据量是否足够
        min_required = self.max_length + 20  # 至少需要max_length + 20个数据点
        if len(df) < min_required:
            print(f"⚠️ 数据量不足: 需要至少{min_required}条，实际{len(df)}条")
            print("💡 建议: 使用历史K线数据生成训练数据，或收集更多数据")
            return None, None
        
        # 分割训练集和验证集
        split_idx = int(len(df) * 0.8)
        df_train = df.iloc[:split_idx].reset_index(drop=True)
        df_val = df.iloc[split_idx:].reset_index(drop=True)
        
        print(f"📊 数据分割: 训练集{len(df_train)}条, 验证集{len(df_val)}条")
        
        seq_lengths = range(self.min_length, self.max_length + 1, self.step)
        
        for seq_len in seq_lengths:
            print(f"\n{'='*60}")
            print(f"测试序列长度: {seq_len}")
            print(f"{'='*60}")
            
            try:
                # 准备数据
                print("📊 准备序列数据...")
                X_train, y_train = self.prepare_data_with_sequence(df_train, seq_len)
                X_val, y_val = self.prepare_data_with_sequence(df_val, seq_len)
                
                print(f"训练集: {len(X_train)} 样本, 验证集: {len(X_val)} 样本")
                
                # 训练并评估
                print("🔬 训练和评估模型...")
                results = self.train_and_evaluate(seq_length=seq_len,
                                                  X_train=X_train, y_train=y_train,
                                                  X_val=X_val, y_val=y_val)
                
                # 记录结果
                result_record = {
                    'seq_length': seq_len,
                    'accuracy': results['accuracy'],
                    'loss': results['loss'],
                    'prediction_variance': results['prediction_variance'],
                    'composite_score': results['composite_score']
                }
                self.results.append(result_record)
                
                print(f"✅ 结果: 准确率={results['accuracy']:.4f}, "
                      f"损失={results['loss']:.4f}, "
                      f"综合评分={results['composite_score']:.4f}")
                
                # 检查收敛
                is_converged, optimal_length = self.check_convergence()
                if is_converged and optimal_length:
                    print(f"\n🎯 序列长度收敛于: {optimal_length}")
                    print(f"   最佳综合评分: {max(r['composite_score'] for r in self.results):.4f}")
                    break
                
            except Exception as e:
                print(f"❌ 测试序列长度 {seq_len} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 找到最优长度
        if self.results:
            best_result = max(self.results, key=lambda x: x['composite_score'])
            optimal_length = best_result['seq_length']
            print(f"\n🏆 最优序列长度: {optimal_length}")
            print(f"   最佳综合评分: {best_result['composite_score']:.4f}")
            print(f"   准确率: {best_result['accuracy']:.4f}")
        else:
            optimal_length = None
        
        return self.results, optimal_length
    
    def plot_results(self, output_file=None):
        """绘制测试结果"""
        if not self.results:
            print("⚠️ 没有结果可绘制")
            return
        
        seq_lengths = [r['seq_length'] for r in self.results]
        accuracies = [r['accuracy'] for r in self.results]
        composite_scores = [r['composite_score'] for r in self.results]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 准确率图
        ax1.plot(seq_lengths, accuracies, 'b-o', label='准确率')
        ax1.set_xlabel('序列长度')
        ax1.set_ylabel('准确率')
        ax1.set_title('序列长度 vs 准确率')
        ax1.grid(True)
        ax1.legend()
        
        # 综合评分图
        ax2.plot(seq_lengths, composite_scores, 'r-o', label='综合评分')
        ax2.set_xlabel('序列长度')
        ax2.set_ylabel('综合评分')
        ax2.set_title('序列长度 vs 综合评分')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file)
            print(f"📊 图表已保存到: {output_file}")
        else:
            plt.savefig(f'/home/cx/trading_data/sequence_length_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        
        plt.close()
    
    def save_results(self, output_file=None):
        """保存测试结果"""
        if not self.results:
            return
        
        if output_file is None:
            output_file = f'/home/cx/trading_data/sequence_length_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 结果已保存到: {output_file}")


def main():
    """主函数"""
    tester = SequenceLengthTester(
        data_dir='/home/cx/trading_data',
        min_length=10,
        max_length=500,
        step=20,
        convergence_window=5,
        convergence_threshold=0.01
    )
    
    results, optimal_length = tester.test_sequence_lengths()
    
    if results:
        tester.plot_results()
        tester.save_results()
        
        print("\n" + "="*60)
        print("📊 测试总结")
        print("="*60)
        print(f"测试了 {len(results)} 个不同的序列长度")
        if optimal_length:
            print(f"最优序列长度: {optimal_length}")
            best_result = max(results, key=lambda x: x['composite_score'])
            print(f"最佳综合评分: {best_result['composite_score']:.4f}")
            print(f"最佳准确率: {best_result['accuracy']:.4f}")


if __name__ == "__main__":
    main()
