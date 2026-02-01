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


class LargeTradingNetwork(nn.Module):
    """大型交易网络 - 使用更多参数和更深的网络结构"""
    def __init__(self, input_size=12, hidden_size=256, num_layers=4, output_size=3, dropout_rate=0.1):
        super(LargeTradingNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 多层LSTM，更宽更深
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # 多层全连接层，增加非线性，明确设置inplace=False避免inplace操作问题
        self.fc_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=False),  # 明确设置inplace=False
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=False),  # 明确设置inplace=False
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=False),  # 明确设置inplace=False
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, output_size)
        ])
        
        # 移除softmax层，因为CrossEntropyLoss内部处理
        # self.softmax = nn.Softmax(dim=1)
        
        # 参数初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x):
        # 确保输入维度正确
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # 添加序列维度 (batch, seq, features)
        
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # 取最后一个时间步
        
        # 通过多层全连接
        for layer in self.fc_layers:
            out = layer(out)
        
        return out


class LargeModelStrategy:
    """大模型交易策略"""
    def __init__(self, data_dir='/home/cx/trading_data', model_path=None):
        # 强制使用GPU
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name()}")
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU Memory: {gpu_memory:.2f} GB")
        else:
            raise RuntimeError("CUDA不可用，此策略需要GPU运行")
        
        self.data_dir = data_dir
        
        # 初始化大型模型
        self.model = LargeTradingNetwork(
            input_size=12,
            hidden_size=256,  # 更宽的隐藏层
            num_layers=4,     # 更深的层数
            dropout_rate=0.1
        ).to(self.device)
        
        # 计算参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Large Model参数数量: {total_params:,}")
        print(f"可训练参数数量: {trainable_params:,}")
        
        # 优化器和损失函数将在训练时动态设置
        self.optimizer = None
        self.criterion = None
        
        # 控制训练和推理的标志
        self.should_train = True
        self.model_lock = threading.Lock()
        
        # 如果提供了模型路径，则加载模型
        if model_path and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                # 只有当优化器存在时才加载优化器状态
                if 'optimizer_state_dict' in checkpoint and self.optimizer is not None:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"✅ 从 {model_path} 加载模型成功")
            except Exception as e:
                print(f"❌ 加载模型失败: {e}，使用初始模型")
        
        # 启动训练线程
        self.training_thread = threading.Thread(target=self.train_continuously, daemon=True)
        self.training_thread.start()
    
    def calculate_class_weights(self, y):
        """计算类别权重以处理不平衡数据"""
        import numpy as np
        
        # 计算每个类别的样本数
        classes, counts = np.unique(y, return_counts=True)
        total_samples = len(y)
        
        # 计算每个类别的权重 (总样本数 / (类别数 * 每个类别的样本数))
        weights = []
        for count in counts:
            weight = total_samples / (len(classes) * count)
            weights.append(weight)
        
        # 转换为tensor
        class_weights = torch.FloatTensor(weights).to(self.device)
        return class_weights

    def train_model(self, df):
        """训练大型模型"""
        try:
            # 初始化优化器（如果尚未初始化）
            if self.optimizer is None:
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-5)
            
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
            
            if len(X) < 10:  # 需要至少10个样本进行训练
                return
            
            # 分析标签分布
            unique, counts = np.unique(y, return_counts=True)
            label_distribution = dict(zip(unique, counts))
            print(f"标签分布: {label_distribution}")
            
            # 如果某个类别占比过高，可能存在数据不平衡问题
            total_samples = len(y)
            for label, count in label_distribution.items():
                percentage = count / total_samples * 100
                print(f"标签 {label} 占比: {percentage:.2f}%")
            
            # 计算类别权重
            class_weights = self.calculate_class_weights(y)
            print(f"类别权重: {class_weights}")
            
            # 更新损失函数
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            
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
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
            y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(self.device)
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(self.device)
            
            # 创建数据集
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            
            # 训练模型
            self.model.train()  # 设置为训练模式
            best_val_acc = 0.0
            
            for epoch in range(20):  # 增加到20轮训练
                # 训练阶段
                total_loss = 0
                num_batches = 0
                correct_predictions = 0
                total_predictions = 0
                
                for batch_x, batch_y in train_loader:
                    # 创建新的张量副本以避免版本冲突
                    batch_x = batch_x.clone().detach().to(self.device)
                    batch_y = batch_y.clone().detach().to(self.device)
                    
                    # 确保每次处理批次时都添加维度以匹配模型输入
                    batch_x = batch_x.unsqueeze(1).contiguous()  # 添加序列维度 (batch, seq, features)
                    
                    self.optimizer.zero_grad()
                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs, batch_y)
                    loss.backward()  # 这里可能出现None张量错误
                    self.optimizer.step()
                    
                    # 计算准确率
                    predictions = torch.argmax(outputs, dim=1)
                    correct_predictions += (predictions == batch_y).sum().item()
                    total_predictions += batch_y.size(0)
                    
                    total_loss += loss.item()
                    num_batches += 1
                
                train_avg_loss = total_loss / num_batches
                train_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
                
                # 验证阶段
                self.model.eval()  # 设置为评估模式
                val_correct = 0
                val_total = 0
                val_loss = 0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        # 创建新的张量副本以避免版本冲突
                        batch_x = batch_x.clone().detach().to(self.device)
                        batch_y = batch_y.clone().detach().to(self.device)
                        
                        # 确保验证批次也添加维度
                        batch_x = batch_x.unsqueeze(1).contiguous()  # 添加序列维度 (batch, seq, features)
                        
                        outputs = self.model(batch_x)
                        loss = self.criterion(outputs, batch_y)
                        val_loss += loss.item()
                        
                        predictions = torch.argmax(outputs, dim=1)
                        val_correct += (predictions == batch_y).sum().item()
                        val_total += batch_y.size(0)
                
                val_avg_loss = val_loss / len(val_loader)
                val_accuracy = val_correct / val_total if val_total > 0 else 0
                
                print(f"大模型训练轮次 {epoch+1}/20")
                print(f"  训练 - 损失: {train_avg_loss:.4f}, 准确率: {train_accuracy:.3f}")
                print(f"  验证 - 损失: {val_avg_loss:.4f}, 准确率: {val_accuracy:.3f}")
                
                # 保存最佳模型
                if val_accuracy > best_val_acc:
                    best_val_acc = val_accuracy
                    print(f"  🏆 大模型新最佳验证准确率: {best_val_acc:.3f}")
                
                self.model.train()  # 重新设置为训练模式以进行下一轮训练
        
        except Exception as e:
            print(f"大模型训练过程错误: {e}")
            import traceback
            traceback.print_exc()
    
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
    
    
    
    def predict_action(self, current_data):
        """预测交易动作"""
        with self.model_lock:
            try:
                # 准备输入数据
                features = self.prepare_features(current_data)
                input_tensor = torch.tensor([features], dtype=torch.float32).unsqueeze(1).to(self.device)
                
                # 预测
                with torch.no_grad():
                    self.model.eval()
                    prediction = self.model(input_tensor)
                    probabilities = prediction.cpu().numpy()[0]
                    
                    # 返回最可能的动作: 0=不操作, 1=买入, 2=卖出
                    action = np.argmax(probabilities)
                    confidence = probabilities[action]
                    
                    return int(action), float(confidence)
            except Exception as e:
                print(f"预测错误: {e}")
                import traceback
                traceback.print_exc()
                return 0, 0.0  # 默认返回不操作，置信度0
    
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
    
    def train_continuously(self):
        """连续训练大模型的后台线程"""
        while self.should_train:
            try:
                # 加载数据
                df = self.load_training_data()
                if df is not None and len(df) > 0:
                    print(f"开始训练大模型，数据量: {len(df)}")
                    with self.model_lock:
                        self.train_model(df)
                    print("大模型训练完成")
                
                # 训练较慢，每30分钟训练一次
                time.sleep(1800)
                
            except Exception as e:
                print(f"大模型训练线程错误: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(60)  # 出错后等待1分钟后继续
    
    def save_model(self, path):
        """保存模型"""
        with self.model_lock:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, path)
            print(f"大模型已保存到 {path}")


def main():
    parser = argparse.ArgumentParser(description='Large Model Trading Strategy')
    parser.add_argument('--mode', choices=['train', 'predict'], default='predict',
                        help='运行模式: train(仅训练), predict(仅预测)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='模型保存或加载路径')
    args = parser.parse_args()
    
    strategy = LargeModelStrategy(model_path=args.model_path)
    
    if args.mode == 'train':
        print("仅运行大模型训练模式...")
        # 训练模式，不退出
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            print("大模型训练已停止")
    elif args.mode == 'predict':
        print("运行大模型预测模式...")
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
        
        action, confidence = strategy.predict_action(sample_data)
        action_map = {0: "不操作", 1: "买入", 2: "卖出"}
        
        print(f"🧠 大模型预测: {action_map[action]}, 置信度: {confidence:.3f}")
        
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            print("大模型预测已停止")
    
    # 如果提供了保存路径，保存模型
    if args.model_path:
        strategy.save_model(args.model_path)


if __name__ == "__main__":
    main()