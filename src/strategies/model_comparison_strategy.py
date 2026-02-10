import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
from datetime import datetime
import threading
import time
import glob
import argparse
from typing import Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

class TradingLSTM(nn.Module):
    """用于交易决策的LSTM模型"""
    def __init__(self, input_size=12, hidden_size=64, num_layers=2, output_size=3):
        super(TradingLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        out = self.softmax(out)
        return out


class TradingTransformer(nn.Module):
    """用于交易决策的Transformer模型"""
    def __init__(self, input_size=12, nhead=2, num_layers=2, output_size=3, d_model=64):
        super(TradingTransformer, self).__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoding = nn.Parameter(torch.zeros(1, 1, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(d_model, output_size)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        x = x + self.pos_encoding  # Add positional encoding
        
        out = self.transformer(x)  # (batch_size, seq_len, d_model)
        out = self.dropout(out[:, -1, :])  # Take the last sequence element
        out = self.fc(out)  # (batch_size, output_size)
        out = self.softmax(out)
        return out


class ModelComparisonStrategy:
    """模型比较策略 - 同时使用LSTM和Transformer模型"""
    def __init__(self, data_dir='/home/cx/trading_data', model_path=None):
        # 强制使用GPU
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            raise RuntimeError("CUDA不可用，此策略需要GPU运行")
        
        self.data_dir = data_dir
        
        # 初始化两个模型
        self.lstm_model = TradingLSTM(input_size=12).to(self.device)
        self.transformer_model = TradingTransformer(input_size=12).to(self.device)
        
        # 优化器和损失函数
        self.lstm_optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=0.001)
        self.transformer_optimizer = torch.optim.Adam(self.transformer_model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
        # 控制训练和推理的标志
        self.should_train = True
        self.model_lock = threading.Lock()
        
        # 记录模型性能
        self.performance_log = {
            'lstm_correct': 0,
            'lstm_total': 0,
            'transformer_correct': 0,
            'transformer_total': 0
        }
        
        # 如果提供了模型路径，则加载模型
        if model_path and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                if 'lstm_state_dict' in checkpoint:
                    self.lstm_model.load_state_dict(checkpoint['lstm_state_dict'])
                if 'transformer_state_dict' in checkpoint:
                    self.transformer_model.load_state_dict(checkpoint['transformer_state_dict'])
                print(f"✅ 从 {model_path} 加载模型成功")
            except Exception as e:
                print(f"❌ 加载模型失败: {e}，使用初始模型")
        
        # 启动训练线程
        self.training_thread = threading.Thread(target=self.train_continuously, daemon=True)
        self.training_thread.start()
    
    def prepare_features(self, row):
        """从数据行中准备特征向量"""
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
            # 归一化特征
            features_np = np.array(features)
            mean_val = np.mean(features_np)
            std_val = np.std(features_np) + 1e-8
            normalized_features = (features_np - mean_val) / std_val
            return normalized_features.tolist()
        except Exception as e:
            print(f"prepare_features错误: {e}")
            # 返回默认特征值
            return [0.0] * 12
    
    def predict_both_models(self, current_data):
        """使用两个模型预测交易动作"""
        with self.model_lock:
            try:
                # 准备输入数据
                features = self.prepare_features(current_data)
                input_tensor = torch.tensor([features], dtype=torch.float32).unsqueeze(1).to(self.device)
                
                # LSTM预测
                with torch.no_grad():
                    self.lstm_model.eval()
                    lstm_prediction = self.lstm_model(input_tensor)
                    lstm_probabilities = lstm_prediction.cpu().numpy()
                    lstm_action = np.argmax(lstm_probabilities[0])
                    lstm_confidence = lstm_probabilities[0][lstm_action]
                
                # Transformer预测 (需要序列长度为1)
                transformer_input = input_tensor  # Already shaped as (1, 1, 10)
                with torch.no_grad():
                    self.transformer_model.eval()
                    transformer_prediction = self.transformer_model(transformer_input)
                    transformer_probabilities = transformer_prediction.cpu().numpy()
                    transformer_action = np.argmax(transformer_probabilities[0])
                    transformer_confidence = transformer_probabilities[0][transformer_action]
                
                return {
                    'lstm': {'action': int(lstm_action), 'confidence': float(lstm_confidence)},
                    'transformer': {'action': int(transformer_action), 'confidence': float(transformer_confidence)}
                }
            except Exception as e:
                print(f"预测错误: {e}")
                import traceback
                traceback.print_exc()
                return {
                    'lstm': {'action': 0, 'confidence': 0.0},
                    'transformer': {'action': 0, 'confidence': 0.0}
                }
    
    def load_training_data(self):
        """加载训练数据"""
        # 获取最新的数据文件
        all_data_files = []
        
        # 获取所有数据目录
        all_data_dirs = glob.glob(os.path.join(self.data_dir, '202*-*-*'))
        if all_data_dirs:
            # 按日期排序，获取最新的几个文件
            sorted_dirs = sorted(all_data_dirs, reverse=True)
            for data_dir in sorted_dirs[:7]:  # 使用最近7天的数据
                # 包含原始数据和扩展数据
                data_files = glob.glob(os.path.join(data_dir, 'trading_data_*.csv'))
                data_files.extend(glob.glob(os.path.join(data_dir, 'extended_trading_data_*.csv')))
                data_files.extend(glob.glob(os.path.join(data_dir, 'prepared_features_*.csv')))  # 也包含准备好的特征数据
                all_data_files.extend(data_files)
        
        if not all_data_files:
            return None
        
        # 按修改时间排序，获取最新的文件
        data_file = sorted(all_data_files, key=os.path.getmtime, reverse=True)[0]
        
        try:
            df = pd.read_csv(data_file)
            # 清理数据
            df = df.dropna(subset=['price_current', 'grid_lower', 'grid_upper', 'atr', 'rsi_1m', 'rsi_5m'])
            
            if len(df) < 10:  # 需要至少10个样本进行训练
                return None
                
            return df
        except Exception as e:
            print(f"加载训练数据错误: {e}")
            return None
    
    def train_lstm(self, df):
        """训练LSTM模型"""
        try:
            # 准备训练数据，使用基于盈利的标签
            X, y = [], []
            look_ahead = 10  # 向前看10个时间步长来计算盈利
            
            for i in range(len(df) - look_ahead):
                row = df.iloc[i]
                features = self.prepare_features(row)
                X.append(features)
                
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
            
            if len(X) < 10:  # 需要至少10个样本进行训练
                return
            
            # 分割训练集和验证集 (80% 训练, 20% 验证)
            split_idx = int(len(X) * 0.8)
            
            X_train = X[:split_idx]
            y_train = y[:split_idx]
            X_val = X[split_idx:]
            y_val = y[split_idx:]
            
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            X_val = np.array(X_val)
            y_val = np.array(y_val)
            
            # 转换为张量
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(self.device)
            y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(self.device)
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1).to(self.device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(self.device)
            
            # 创建数据集
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
            
            # 训练LSTM模型
            self.lstm_model.train()  # 设置为训练模式
            best_val_acc = 0.0
            
            for epoch in range(5):  # 增加到5轮训练
                # 训练阶段
                total_loss = 0
                num_batches = 0
                correct_predictions = 0
                total_predictions = 0
                
                for batch_x, batch_y in train_loader:
                    self.lstm_optimizer.zero_grad()
                    outputs = self.lstm_model(batch_x)
                    loss = self.criterion(outputs, batch_y)
                    loss.backward()
                    self.lstm_optimizer.step()
                    
                    # 计算准确率
                    predictions = torch.argmax(outputs, dim=1)
                    correct_predictions += (predictions == batch_y).sum().item()
                    total_predictions += batch_y.size(0)
                    
                    total_loss += loss.item()
                    num_batches += 1
                
                train_avg_loss = total_loss / num_batches
                train_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
                
                # 验证阶段
                self.lstm_model.eval()  # 设置为评估模式
                val_correct = 0
                val_total = 0
                val_loss = 0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        outputs = self.lstm_model(batch_x)
                        loss = self.criterion(outputs, batch_y)
                        val_loss += loss.item()
                        
                        predictions = torch.argmax(outputs, dim=1)
                        val_correct += (predictions == batch_y).sum().item()
                        val_total += batch_y.size(0)
                
                val_avg_loss = val_loss / len(val_loader)
                val_accuracy = val_correct / val_total if val_total > 0 else 0
                
                print(f"LSTM训练轮次 {epoch+1}/{5}")
                print(f"  训练 - 损失: {train_avg_loss:.4f}, 准确率: {train_accuracy:.3f}")
                print(f"  验证 - 损失: {val_avg_loss:.4f}, 准确率: {val_accuracy:.3f}")
                
                # 保存最佳模型
                if val_accuracy > best_val_acc:
                    best_val_acc = val_accuracy
                    print(f"  🏆 LSTM新最佳验证准确率: {best_val_acc:.3f}")
                
                self.lstm_model.train()  # 重新设置为训练模式以进行下一轮训练
        
        except Exception as e:
            print(f"LSTM训练过程错误: {e}")
            import traceback
            traceback.print_exc()

    def train_transformer(self, df):
        """训练Transformer模型"""
        try:
            # 准备训练数据，使用基于盈利的标签
            X, y = [], []
            look_ahead = 10  # 向前看10个时间步长来计算盈利
            
            for i in range(len(df) - look_ahead):
                row = df.iloc[i]
                features = self.prepare_features(row)
                X.append(features)
                
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
            
            if len(X) < 10:  # 需要至少10个样本进行训练
                return
            
            # 分割训练集和验证集 (80% 训练, 20% 验证)
            split_idx = int(len(X) * 0.8)
            
            X_train = X[:split_idx]
            y_train = y[:split_idx]
            X_val = X[split_idx:]
            y_val = y[split_idx:]
            
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            X_val = np.array(X_val)
            y_val = np.array(y_val)
            
            # 转换为张量
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(self.device)
            y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(self.device)
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1).to(self.device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(self.device)
            
            # 创建数据集
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
            
            # 训练Transformer模型
            self.transformer_model.train()  # 设置为训练模式
            best_val_acc = 0.0
            
            for epoch in range(5):  # 增加到5轮训练
                # 训练阶段
                total_loss = 0
                num_batches = 0
                correct_predictions = 0
                total_predictions = 0
                
                for batch_x, batch_y in train_loader:
                    self.transformer_optimizer.zero_grad()
                    outputs = self.transformer_model(batch_x)
                    loss = self.criterion(outputs, batch_y)
                    loss.backward()
                    self.transformer_optimizer.step()
                    
                    # 计算准确率
                    predictions = torch.argmax(outputs, dim=1)
                    correct_predictions += (predictions == batch_y).sum().item()
                    total_predictions += batch_y.size(0)
                    
                    total_loss += loss.item()
                    num_batches += 1
                
                train_avg_loss = total_loss / num_batches
                train_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
                
                # 验证阶段
                self.transformer_model.eval()  # 设置为评估模式
                val_correct = 0
                val_total = 0
                val_loss = 0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        outputs = self.transformer_model(batch_x)
                        loss = self.criterion(outputs, batch_y)
                        val_loss += loss.item()
                        
                        predictions = torch.argmax(outputs, dim=1)
                        val_correct += (predictions == batch_y).sum().item()
                        val_total += batch_y.size(0)
                
                val_avg_loss = val_loss / len(val_loader)
                val_accuracy = val_correct / val_total if val_total > 0 else 0
                
                print(f"Transformer训练轮次 {epoch+1}/{5}")
                print(f"  训练 - 损失: {train_avg_loss:.4f}, 准确率: {train_accuracy:.3f}")
                print(f"  验证 - 损失: {val_avg_loss:.4f}, 准确率: {val_accuracy:.3f}")
                
                # 保存最佳模型
                if val_accuracy > best_val_acc:
                    best_val_acc = val_accuracy
                    print(f"  🏆 Transformer新最佳验证准确率: {best_val_acc:.3f}")
                
                self.transformer_model.train()  # 重新设置为训练模式以进行下一轮训练
        
        except Exception as e:
            print(f"Transformer训练过程错误: {e}")
            import traceback
            traceback.print_exc()
    
    def train_continuously(self):
        """连续训练两个模型的后台线程"""
        while self.should_train:
            try:
                # 加载数据
                df = self.load_training_data()
                if df is not None and len(df) > 0:
                    print(f"开始训练两个模型，数据量: {len(df)}")
                    with self.model_lock:
                        self.train_lstm(df)
                        self.train_transformer(df)
                    print("两个模型训练完成")
                
                # 训练较慢，每30分钟训练一次
                time.sleep(1800)
                
            except Exception as e:
                print(f"训练线程错误: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(60)  # 出错后等待1分钟后继续
    
    def save_model(self, path):
        """保存两个模型"""
        with self.model_lock:
            torch.save({
                'lstm_state_dict': self.lstm_model.state_dict(),
                'lstm_optimizer_state_dict': self.lstm_optimizer.state_dict(),
                'transformer_state_dict': self.transformer_model.state_dict(),
                'transformer_optimizer_state_dict': self.transformer_optimizer.state_dict(),
            }, path)
            print(f"模型已保存到 {path}")
    
    def log_performance(self, actual_action, lstm_pred, transformer_pred):
        """记录模型性能"""
        with self.model_lock:
            if actual_action == lstm_pred['action']:
                self.performance_log['lstm_correct'] += 1
            self.performance_log['lstm_total'] += 1
            
            if actual_action == transformer_pred['action']:
                self.performance_log['transformer_correct'] += 1
            self.performance_log['transformer_total'] += 1
            
            # 打印性能摘要
            if self.performance_log['lstm_total'] > 0:
                lstm_acc = self.performance_log['lstm_correct'] / self.performance_log['lstm_total']
                trans_acc = self.performance_log['transformer_correct'] / self.performance_log['transformer_total']
                print(f"📊 模型性能 - LSTM准确率: {lstm_acc:.3f}, Transformer准确率: {trans_acc:.3f}")


def main():
    parser = argparse.ArgumentParser(description='Model Comparison Trading Strategy')
    parser.add_argument('--mode', choices=['train', 'predict', 'compare'], default='compare',
                        help='运行模式: train(仅训练), predict(仅预测), compare(比较)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='模型保存或加载路径')
    args = parser.parse_args()
    
    strategy = ModelComparisonStrategy(model_path=args.model_path)
    
    if args.mode == 'train':
        print("仅运行训练模式...")
        # 训练模式，不退出
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            print("训练已停止")
    elif args.mode == 'predict' or args.mode == 'compare':
        print("运行比较模式...")
        # 演示如何使用
        sample_data = {
            'price_current': 93.5,
            'grid_lower': 93.0,
            'grid_upper': 94.0,
            'atr': 0.2,
            'rsi_1m': 30.0,
            'rsi_5m': 40.0,
            'buffer': 0.05,
            'threshold': 93.05,
            'near_lower': True,
            'rsi_ok': True
        }
        
        predictions = strategy.predict_both_models(sample_data)
        action_map = {0: "不操作", 1: "买入", 2: "卖出"}
        
        print(f"🧠 LSTM预测: {action_map[predictions['lstm']['action']]}, 置信度: {predictions['lstm']['confidence']:.3f}")
        print(f"🧠 Transformer预测: {action_map[predictions['transformer']['action']]}, 置信度: {predictions['transformer']['confidence']:.3f}")
        
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            print("程序已停止")
    
    # 如果提供了保存路径，保存模型
    if args.model_path:
        strategy.save_model(args.model_path)


if __name__ == "__main__":
    main()