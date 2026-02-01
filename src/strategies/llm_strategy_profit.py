"""
收益率回归版本的交易策略

直接预测收益率，而不是预测动作
损失函数直接优化收益目标
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import os


class TradingLSTMProfit(nn.Module):
    """预测收益率的LSTM模型"""
    def __init__(self, input_size=12, hidden_size=64, num_layers=2, predict_grid_adjustment=True):
        super(TradingLSTMProfit, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.predict_grid_adjustment = predict_grid_adjustment
        
        # LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        
        # 收益率预测头（回归任务）
        self.profit_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 1)  # 预测收益率
        )
        
        # 网格调整系数头（可选）
        if predict_grid_adjustment:
            self.grid_adjustment_head = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x shape: (batch, seq, features)
        lstm_out, _ = self.lstm(x)
        # 使用最后一个时间步的输出
        out = lstm_out[:, -1, :]  # (batch, hidden_size)
        
        # 预测收益率
        profit = self.profit_head(out)  # (batch, 1)
        
        # 预测网格调整系数（如果启用）
        if self.predict_grid_adjustment and self.grid_adjustment_head is not None:
            grid_adjustment_raw = self.grid_adjustment_head(out)
            grid_adjustment = torch.sigmoid(grid_adjustment_raw) * 0.4 + 0.8  # [0.8, 1.2]
            return profit, grid_adjustment
        else:
            return profit


class LLMTradingStrategyProfit:
    """基于收益率回归的交易策略"""
    def __init__(self, mode='hybrid', data_dir='/home/cx/trading_data'):
        self.mode = mode
        self.data_dir = data_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 根据模式确定输入维度
        if mode == 'hybrid':
            input_size = 12
        elif mode == 'pure_ml':
            input_size = 10
        else:
            input_size = 12
        
        # 初始化模型
        self.model = TradingLSTMProfit(
            input_size=input_size,
            hidden_size=64,
            num_layers=2,
            predict_grid_adjustment=True
        ).to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        
        # 损失函数（使用Huber损失，对异常值更鲁棒）
        self.criterion = nn.HuberLoss(delta=0.01)
        self.grid_criterion = nn.MSELoss()
        
        self._seq_length = 10
    
    def prepare_features(self, row):
        """准备特征（与原始版本相同）"""
        def get_value(key, default=0):
            if isinstance(row, pd.Series):
                return row.get(key, default)
            elif isinstance(row, dict):
                return row.get(key, default)
            else:
                return getattr(row, key, default)
        
        def get_value_safe(key, default=0, check_na=False):
            val = get_value(key, default)
            if check_na and (val is None or (isinstance(val, float) and np.isnan(val))):
                return default
            return val
        
        if self.mode == 'hybrid':
            features = [
                get_value_safe('price_current', 0),
                get_value_safe('atr', 0),
                get_value_safe('rsi_1m', 50, check_na=True),
                get_value_safe('rsi_5m', 50, check_na=True),
                get_value('grid_lower', 0),
                get_value('grid_upper', 0),
                get_value('boll_upper', 0),
                get_value('boll_mid', 0),
                get_value('boll_lower', 0),
                get_value('boll_position', 0.5),
                get_value('volatility', 0),
                get_value('volume_1m', 0)
            ]
        elif self.mode == 'pure_ml':
            features = [
                get_value_safe('open_1m', 0),
                get_value_safe('high_1m', 0),
                get_value_safe('low_1m', 0),
                get_value_safe('close_1m', get_value_safe('price_current', 0)),
                get_value_safe('volume_1m', 0),
                get_value_safe('open_5m', 0),
                get_value_safe('high_5m', 0),
                get_value('low_5m', 0),
                get_value('close_5m', get_value_safe('price_current', 0)),
                get_value('volume_5m', 0)
            ]
        else:
            features = [0.0] * 12
        
        # 归一化
        features = np.array(features, dtype=np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return features.tolist()
    
    def prepare_sequence_features(self, df, idx, seq_length):
        """准备序列特征"""
        sequence = []
        for i in range(max(0, idx - seq_length + 1), idx + 1):
            row = df.iloc[i]
            features = self.prepare_features(row)
            sequence.append(features)
        
        # 如果序列不足，用第一个值填充
        while len(sequence) < seq_length:
            if sequence:
                sequence.insert(0, sequence[0])
            else:
                sequence.insert(0, [0.0] * (12 if self.mode == 'hybrid' else 10))
        
        return np.array(sequence[-seq_length:], dtype=np.float32)
    
    def calculate_actual_profit(self, current_price, future_prices):
        """计算实际收益率"""
        if len(future_prices) == 0:
            return 0.0
        
        # 买入收益：未来最高价 - 当前价格
        buy_profit = (max(future_prices) - current_price) / current_price
        
        # 卖出收益：当前价格 - 未来最低价
        sell_profit = (current_price - min(future_prices)) / current_price
        
        # 返回最大收益
        return max(buy_profit, sell_profit)
    
    def profit_to_action(self, predicted_profit, buy_profit, sell_profit, threshold=0.005):
        """将预测收益率转换为动作"""
        if predicted_profit > threshold:
            if buy_profit > sell_profit:
                return 1  # 买入
            else:
                return 2  # 卖出
        else:
            return 0  # 不操作
    
    def train_model(self, df, seq_length=10, max_epochs=50, patience=10, train_grid_adjustment=True):
        """训练模型（收益率回归版本）"""
        try:
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            
            # 准备训练数据
            X, y_profit, y_grid = [], [], []
            look_ahead = 10
            min_required = seq_length + look_ahead
            
            print(f"📊 使用序列长度: {seq_length}, 需要至少 {min_required} 个数据点")
            
            for i in range(min_required, len(df)):
                if i + look_ahead >= len(df):
                    break
                
                # 准备序列特征
                sequence = self.prepare_sequence_features(df, i, seq_length)
                X.append(sequence)
                
                # 计算实际收益率
                current_price = df.iloc[i]['price_current']
                future_prices = df.iloc[i+1:i+look_ahead+1]['price_current'].values
                
                if len(future_prices) == 0:
                    X.pop()
                    continue
                
                actual_profit = self.calculate_actual_profit(current_price, future_prices)
                y_profit.append(actual_profit)
                
                # 计算网格调整系数（如果启用）
                if train_grid_adjustment:
                    grid_lower = df.iloc[i].get('grid_lower', current_price * 0.99)
                    grid_upper = df.iloc[i].get('grid_upper', current_price * 1.01)
                    grid_base = max(grid_upper - grid_lower, 0.01)
                    
                    # 简化：基于价格波动计算调整系数
                    price_range = max(future_prices) - min(future_prices)
                    optimal_spacing = price_range / 3.0
                    adjustment = max(0.8, min(1.2, optimal_spacing / grid_base if grid_base > 0 else 1.0))
                    y_grid.append(adjustment)
                else:
                    y_grid.append(1.0)
            
            # 确保长度一致
            min_len = min(len(X), len(y_profit), len(y_grid))
            X = X[:min_len]
            y_profit = y_profit[:min_len]
            y_grid = y_grid[:min_len]
            
            if len(X) < 10:
                print("数据不足，跳过训练")
                return
            
            print(f"✅ 数据准备完成: {len(X)} 个样本")
            print(f"📊 收益率统计: min={min(y_profit):.4f}, max={max(y_profit):.4f}, mean={np.mean(y_profit):.4f}")
            
            # 分割训练集和验证集
            split_idx = int(len(X) * 0.8)
            X_train = np.array(X[:split_idx])
            y_profit_train = np.array(y_profit[:split_idx], dtype=np.float32)
            y_grid_train = np.array(y_grid[:split_idx], dtype=np.float32)
            X_val = np.array(X[split_idx:])
            y_profit_val = np.array(y_profit[split_idx:], dtype=np.float32)
            y_grid_val = np.array(y_grid[split_idx:], dtype=np.float32)
            
            # 转换为张量
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
            y_profit_train_tensor = torch.tensor(y_profit_train, dtype=torch.float32).to(self.device)
            y_grid_train_tensor = torch.tensor(y_grid_train, dtype=torch.float32).to(self.device)
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            y_profit_val_tensor = torch.tensor(y_profit_val, dtype=torch.float32).to(self.device)
            y_grid_val_tensor = torch.tensor(y_grid_val, dtype=torch.float32).to(self.device)
            
            # 创建数据集
            train_dataset = TensorDataset(X_train_tensor, y_profit_train_tensor, y_grid_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_profit_val_tensor, y_grid_val_tensor)
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
            
            # 学习率调度器
            scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
            
            # 训练模型
            self.model.train()
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(max_epochs):
                # 训练阶段
                total_loss = 0
                num_batches = 0
                
                for batch_x, batch_y_profit, batch_y_grid in train_loader:
                    self.optimizer.zero_grad()
                    
                    # 模型输出
                    model_output = self.model(batch_x)
                    
                    if isinstance(model_output, tuple):
                        predicted_profit, grid_adjustment = model_output
                        # 收益率损失
                        profit_loss = self.criterion(predicted_profit.squeeze(), batch_y_profit)
                        # 网格调整损失
                        grid_loss = self.grid_criterion(grid_adjustment.squeeze(), batch_y_grid)
                        # 组合损失
                        loss = profit_loss + 0.1 * grid_loss
                    else:
                        predicted_profit = model_output
                        profit_loss = self.criterion(predicted_profit.squeeze(), batch_y_profit)
                        loss = profit_loss
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                
                train_avg_loss = total_loss / num_batches
                
                # 验证阶段
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_x, batch_y_profit, batch_y_grid in val_loader:
                        model_output = self.model(batch_x)
                        
                        if isinstance(model_output, tuple):
                            predicted_profit, grid_adjustment = model_output
                            profit_loss = self.criterion(predicted_profit.squeeze(), batch_y_profit)
                            grid_loss = self.grid_criterion(grid_adjustment.squeeze(), batch_y_grid)
                            loss = profit_loss + 0.1 * grid_loss
                        else:
                            predicted_profit = model_output
                            loss = self.criterion(predicted_profit.squeeze(), batch_y_profit)
                        
                        val_loss += loss.item()
                
                val_avg_loss = val_loss / len(val_loader)
                scheduler.step(val_avg_loss)
                
                print(f"训练轮次 {epoch+1}/{max_epochs}")
                print(f"  训练 - 损失: {train_avg_loss:.6f}")
                print(f"  验证 - 损失: {val_avg_loss:.6f}")
                print(f"  学习率: {self.optimizer.param_groups[0]['lr']:.6f}")
                
                # 早停
                if val_avg_loss < best_val_loss:
                    best_val_loss = val_avg_loss
                    patience_counter = 0
                    print(f"  🏆 新的最佳验证损失: {best_val_loss:.6f}")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"  ⏹️ 早停于第 {epoch+1} 轮")
                        break
                
                self.model.train()
            
            print("✅ 训练完成")
            
        except Exception as e:
            print(f"训练过程错误: {e}")
            import traceback
            traceback.print_exc()
    
    def predict_profit(self, current_data):
        """预测收益率"""
        self.model.eval()
        with torch.no_grad():
            # 准备序列特征（需要历史数据）
            # 这里简化处理，实际使用时需要传入历史序列
            features = self.prepare_features(current_data)
            sequence = np.array([features] * self._seq_length, dtype=np.float32)
            sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            model_output = self.model(sequence_tensor)
            
            if isinstance(model_output, tuple):
                predicted_profit, grid_adjustment = model_output
                return predicted_profit.item(), grid_adjustment.item()
            else:
                return model_output.item(), 1.0
