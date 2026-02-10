"""
BERT MASK式的无监督预训练
用于收益率预测的预训练，不需要动作标签
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, '/home/cx/tigertrade')

from src.strategies import llm_strategy


class UnsupervisedPretrainer:
    """无监督预训练器（BERT MASK方式）"""
    
    def __init__(self, data_dir='/home/cx/trading_data'):
        self.data_dir = data_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def load_training_data(self):
        """加载训练数据"""
        print("\n" + "="*70)
        print("加载训练数据（用于无监督预训练）")
        print("="*70)
        
        data_files = [f for f in os.listdir(self.data_dir) if f.startswith('training_data_multitimeframe_') and f.endswith('.csv')]
        if not data_files:
            raise FileNotFoundError("未找到训练数据文件")
        
        # 优先选择合并后的数据文件（merged），如果没有则选择数据量最大的文件
        merged_files = [f for f in data_files if 'merged' in f]
        if merged_files:
            # 选择最新的合并文件
            latest_file = max(merged_files, key=lambda f: os.path.getmtime(os.path.join(self.data_dir, f)))
        else:
            # 优先选择数据量大的文件（排除extended文件）
            regular_files = [f for f in data_files if 'extended' not in f]
            if regular_files:
                file_sizes = {f: os.path.getsize(os.path.join(self.data_dir, f)) for f in regular_files}
                latest_file = max(file_sizes, key=file_sizes.get)
            else:
                file_sizes = {f: os.path.getsize(os.path.join(self.data_dir, f)) for f in data_files}
                latest_file = max(file_sizes, key=file_sizes.get)
        data_path = os.path.join(self.data_dir, latest_file)
        
        print(f"加载数据文件: {latest_file}")
        df = pd.read_csv(data_path)
        
        if 'current_position' not in df.columns:
            df['current_position'] = 0
        
        print(f"数据形状: {df.shape}")
        return df
    
    def prepare_unsupervised_data(self, df, seq_length=50, mask_ratio=0.15):
        """
        准备无监督预训练数据
        使用BERT MASK方式：随机mask未来收益率，让模型预测
        """
        print(f"\n准备无监督预训练数据（序列长度={seq_length}, mask比例={mask_ratio}）...")
        
        strategy = llm_strategy.LLMTradingStrategy(mode='hybrid', predict_profit=True)
        strategy._seq_length = seq_length
        
        X, y_return = [], []
        
        for i in range(seq_length, len(df) - 120):  # 保留120个未来点用于计算收益率（2小时）
            try:
                historical_data = df.iloc[i-seq_length:i+1]
                sequence = strategy.prepare_sequence_features(historical_data, len(historical_data)-1, seq_length)
                
                # 计算未来收益率（作为预测目标）
                current_price = df.iloc[i]['price_current']
                if i + 120 < len(df):
                    # 计算未来1小时（120分钟）的最大收益率
                    future_prices = df.iloc[i+1:i+121]['price_current'].values
                    future_returns = (future_prices - current_price) / current_price
                    max_return = np.max(future_returns)  # 最大正向收益率
                    min_return = np.min(future_returns)  # 最大负向收益率（绝对值）
                    
                    # 使用最大绝对收益率作为目标
                    target_return = max(abs(max_return), abs(min_return))
                else:
                    continue
                
                X.append(sequence)
                y_return.append(target_return)
            except Exception as e:
                continue
        
        X = np.array(X)
        y_return = np.array(y_return)
        
        if len(X) == 0:
            raise ValueError(f"数据量不足！需要至少 {seq_length + 120} 条数据，但只有 {len(df)} 条")
        
        print(f"准备的数据形状: X={X.shape}, y_return={y_return.shape}")
        print(f"收益率范围: [{np.min(y_return):.6f}, {np.max(y_return):.6f}], 均值={np.mean(y_return):.6f}")
        
        # 划分训练集和验证集
        split_idx = int(len(X) * 0.9)  # 90%用于训练，10%用于验证
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y_return[:split_idx], y_return[split_idx:]
        
        return (X_train, y_train), (X_val, y_val)
    
    def create_pretraining_model(self, input_size=46, hidden_size=128, num_layers=3):
        """创建预训练模型（只预测收益率）"""
        print("\n创建预训练模型...")
        
        # 使用LSTM作为编码器
        class PretrainingLSTM(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers):
                super(PretrainingLSTM, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
                self.dropout = nn.Dropout(0.3)
                
                # 收益率预测头（回归）
                self.return_head = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.LayerNorm(hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.LayerNorm(hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_size // 2, 1)
                )
                
                self._initialize_weights()
            
            def _initialize_weights(self):
                """初始化权重"""
                for name, param in self.named_parameters():
                    if 'weight' in name and len(param.shape) >= 2:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
            
            def forward(self, x):
                # LSTM处理
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
                out, _ = self.lstm(x, (h0, c0))
                out = out[:, -1, :]  # 取最后一个时间步
                out = self.dropout(out)
                
                # 收益率预测
                return_pred = self.return_head(out)
                return return_pred
        
        model = PretrainingLSTM(input_size, hidden_size, num_layers).to(self.device)
        
        # 计算参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"模型参数数量: {total_params:,} (可训练: {trainable_params:,})")
        
        return model
    
    def pretrain(self, train_data, val_data, max_epochs=100):
        """执行无监督预训练"""
        print("\n" + "="*70)
        print("开始无监督预训练（收益率预测）")
        print("="*70)
        
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        # 创建模型
        input_size = X_train.shape[2]
        model = self.create_pretraining_model(input_size=input_size)
        
        # 优化器和损失函数
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        # 准备数据
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        # 训练
        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0
        
        for epoch in range(max_epochs):
            # 训练阶段
            model.train()
            train_loss = 0
            train_mae = 0
            num_batches = 0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                pred = model(batch_x).squeeze()
                loss = criterion(pred, batch_y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                train_mae += torch.mean(torch.abs(pred - batch_y)).item()
                num_batches += 1
            
            avg_train_loss = train_loss / num_batches
            avg_train_mae = train_mae / num_batches
            
            # 验证阶段
            model.eval()
            val_loss = 0
            val_mae = 0
            num_batches = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    pred = model(batch_x).squeeze()
                    loss = criterion(pred, batch_y)
                    
                    val_loss += loss.item()
                    val_mae += torch.mean(torch.abs(pred - batch_y)).item()
                    num_batches += 1
            
            avg_val_loss = val_loss / num_batches
            avg_val_mae = val_mae / num_batches
            
            scheduler.step(avg_val_loss)
            
            print(f"Epoch {epoch+1}/{max_epochs}: "
                  f"Train Loss={avg_train_loss:.6f}, Train MAE={avg_train_mae:.6f}, "
                  f"Val Loss={avg_val_loss:.6f}, Val MAE={avg_val_mae:.6f}")
            
            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                model_path = os.path.join(self.data_dir, 'pretrained_return_model.pth')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': avg_val_loss,
                    'val_mae': avg_val_mae
                }, model_path)
                print(f"  ✅ 保存最佳预训练模型: Val Loss={avg_val_loss:.6f}, Val MAE={avg_val_mae:.6f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  早停触发（patience={patience}）")
                    break
        
        print(f"\n✅ 无监督预训练完成！最佳验证损失: {best_val_loss:.6f}")
        return model
    
    def run(self):
        """运行完整的无监督预训练流程"""
        print("="*70)
        print("BERT MASK式无监督预训练（收益率预测）")
        print("="*70)
        
        # 1. 加载数据
        df = self.load_training_data()
        
        # 2. 准备无监督数据
        train_data, val_data = self.prepare_unsupervised_data(df, seq_length=50)
        
        # 3. 执行预训练
        model = self.pretrain(train_data, val_data, max_epochs=100)
        
        print("\n✅ 无监督预训练完成！")
        print("预训练模型已保存，可用于后续的有监督微调")


if __name__ == "__main__":
    trainer = UnsupervisedPretrainer()
    trainer.run()
