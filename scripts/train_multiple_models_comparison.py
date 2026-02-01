"""
多模型对比训练脚本
训练至少3个模型（LSTM、Transformer、Enhanced Transformer）并对比结果
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# 添加项目路径
sys.path.insert(0, '/home/cx/tigertrade')

from src.strategies import llm_strategy
from src.strategies.large_transformer_strategy import LargeTradingTransformer, LargeTransformerStrategy
from src.strategies.enhanced_transformer_strategy import EnhancedTradingTransformer, EnhancedTransformerStrategy
from src.strategies.transformer_with_profit import TradingTransformerWithProfit
from src.strategies.gru_with_profit import TradingGRUWithProfit
from src.strategies.enhanced_transformer_with_profit import EnhancedTradingTransformerWithProfit
from src.strategies.enhanced_transformer_improved import ImprovedEnhancedTradingTransformerWithProfit
from src.strategies.enhanced_transformer_regularized import RegularizedEnhancedTradingTransformerWithProfit, LabelSmoothingCrossEntropy
from src.strategies.focal_loss import FocalLoss
from src.strategies.enhanced_transformer_with_profit import EnhancedTradingTransformerWithProfit
from src.strategies.data_augmentation import TradingDataAugmentation
from src.strategies.advanced_data_augmentation import AdvancedDataAugmentation
from src.strategies.moe_transformer import MoETradingTransformerWithProfit

class ModelComparisonTrainer:
    """多模型对比训练器"""
    
    def __init__(self, data_dir='/home/cx/trading_data'):
        self.data_dir = data_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 模型结果存储
        self.results = {}
        
    def load_training_data(self):
        """加载训练数据"""
        print("\n" + "="*70)
        print("加载训练数据")
        print("="*70)
        
        # 查找最新的训练数据文件（优先选择数据量大的）
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
        print(f"数据列: {df.columns.tolist()[:10]}...")
        
        return df
    
    def prepare_data_for_model(self, df, model_type='lstm', seq_length=50):
        """为不同模型准备数据"""
        print(f"\n为{model_type}模型准备数据（序列长度={seq_length}）...")
        
        # 使用LSTM策略的数据准备方法
        strategy = llm_strategy.LLMTradingStrategy(mode='hybrid', predict_profit=True)
        strategy._seq_length = seq_length
        
        X, y, y_profit = [], [], []
        
        for i in range(seq_length, len(df)):
            try:
                historical_data = df.iloc[i-seq_length:i+1]
                sequence = strategy.prepare_sequence_features(historical_data, len(historical_data)-1, seq_length)
                
                # 标签生成（与训练时一致）
                current_price = df.iloc[i]['price_current']
                if i + 120 < len(df):
                    future_prices = df.iloc[i+1:i+121]['price_current'].values
                    buy_profit = (np.max(future_prices) - current_price) / current_price
                    sell_profit = (current_price - np.min(future_prices)) / current_price
                    
                    profit_threshold = 0.003
                    min_diff = 0.002
                    current_position = int(df.iloc[i].get('current_position', 0))
                    
                    if current_position > 0:
                        if sell_profit > profit_threshold:
                            label = 2  # 卖出
                        elif buy_profit > profit_threshold:
                            label = 1  # 买入
                        else:
                            label = 0  # 不操作
                    else:
                        if abs(buy_profit - sell_profit) >= min_diff:
                            if buy_profit > sell_profit and buy_profit > profit_threshold:
                                label = 1  # 买入
                            elif sell_profit > buy_profit and sell_profit > profit_threshold:
                                label = 2  # 卖出
                            else:
                                label = 0  # 不操作
                        else:
                            label = 0  # 不操作
                    
                    # 收益率标签
                    if label == 1:
                        actual_profit = buy_profit
                    elif label == 2:
                        actual_profit = sell_profit
                    else:
                        actual_profit = 0.0
                    
                    X.append(sequence)
                    y.append(label)
                    y_profit.append(actual_profit)
            except Exception as e:
                continue
        
        X = np.array(X)
        y = np.array(y)
        y_profit = np.array(y_profit)
        
        print(f"准备的数据形状: X={X.shape}, y={y.shape}, y_profit={y_profit.shape}")
        
        # 划分训练集和验证集
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        y_profit_train, y_profit_val = y_profit[:split_idx], y_profit[split_idx:]
        
        return (X_train, y_train, y_profit_train), (X_val, y_val, y_profit_val)
    
    def train_lstm_model(self, train_data, val_data, seq_length=50):
        """训练LSTM模型（改进版）"""
        print("\n" + "="*70)
        print("训练LSTM模型（改进版）")
        print("="*70)
        
        X_train, y_train, y_profit_train = train_data
        X_val, y_val, y_profit_val = val_data
        
        # 创建改进的LSTM策略（使用MSELoss）
        strategy = llm_strategy.LLMTradingStrategy(mode='hybrid', predict_profit=True)
        strategy._seq_length = seq_length
        
        # 设置损失函数（动作分类和收益率预测）
        strategy.criterion = nn.CrossEntropyLoss()  # 动作分类损失
        strategy.profit_criterion = nn.MSELoss()  # 收益率损失（使用MSELoss）
        
        # 准备数据
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
            torch.tensor(y_profit_train, dtype=torch.float32)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long),
            torch.tensor(y_profit_val, dtype=torch.float32)
        )
        
        # 根据序列长度动态调整批次大小（长序列需要更小的批次）
        if seq_length >= 500:
            batch_size = 8  # 长序列使用小批次
        elif seq_length >= 300:
            batch_size = 16
        else:
            batch_size = 32
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 训练模型
        best_val_acc = 0
        best_val_profit_mae = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(50):
            # 训练阶段
            strategy.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            train_profit_errors = []
            
            for batch_x, batch_y, batch_y_profit in train_loader:
                batch_x = batch_x.to(strategy.device)
                batch_y = batch_y.to(strategy.device)
                batch_y_profit = batch_y_profit.to(strategy.device)
                
                strategy.optimizer.zero_grad()
                model_output = strategy.model(batch_x)
                
                if isinstance(model_output, tuple) and len(model_output) >= 2:
                    action_logits = model_output[0]
                    profit = model_output[1]
                    
                    # 动作分类损失
                    action_loss = strategy.criterion(action_logits, batch_y)
                    
                    # 收益率损失（使用MSELoss）
                    profit_loss = strategy.profit_criterion(profit.squeeze(), batch_y_profit)
                    
                    # 组合损失
                    loss = action_loss + 1.0 * profit_loss
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(strategy.model.parameters(), max_norm=1.0)
                    strategy.optimizer.step()
                    
                    train_loss += loss.item()
                    predictions = torch.argmax(action_logits, dim=1)
                    train_correct += (predictions == batch_y).sum().item()
                    train_total += batch_y.size(0)
                    
                    # 计算收益率误差
                    profit_errors = torch.abs(profit.squeeze() - batch_y_profit)
                    train_profit_errors.extend(profit_errors.detach().cpu().numpy().tolist())
            
            train_acc = train_correct / train_total if train_total > 0 else 0
            train_profit_mae = np.mean(train_profit_errors) if train_profit_errors else None
            
            # 验证阶段
            strategy.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            val_profit_errors = []
            
            with torch.no_grad():
                for batch_x, batch_y, batch_y_profit in val_loader:
                    batch_x = batch_x.to(strategy.device)
                    batch_y = batch_y.to(strategy.device)
                    batch_y_profit = batch_y_profit.to(strategy.device)
                    
                    model_output = strategy.model(batch_x)
                    
                    if isinstance(model_output, tuple) and len(model_output) >= 2:
                        action_logits = model_output[0]
                        profit = model_output[1]
                        
                        action_loss = strategy.criterion(action_logits, batch_y)
                        profit_loss = strategy.profit_criterion(profit.squeeze(), batch_y_profit)
                        loss = action_loss + 1.0 * profit_loss
                        
                        val_loss += loss.item()
                        predictions = torch.argmax(action_logits, dim=1)
                        val_correct += (predictions == batch_y).sum().item()
                        val_total += batch_y.size(0)
                        
                        profit_errors = torch.abs(profit.squeeze() - batch_y_profit)
                        val_profit_errors.extend(profit_errors.detach().cpu().numpy().tolist())
            
            val_acc = val_correct / val_total if val_total > 0 else 0
            val_profit_mae = np.mean(val_profit_errors) if val_profit_errors else None
            
            train_profit_str = f"{train_profit_mae:.6f}" if train_profit_mae is not None else "N/A"
            val_profit_str = f"{val_profit_mae:.6f}" if val_profit_mae is not None else "N/A"
            print(f"Epoch {epoch+1}/50: Train Acc={train_acc:.4f}, Train Profit MAE={train_profit_str}, "
                  f"Val Acc={val_acc:.4f}, Val Profit MAE={val_profit_str}")
            
            # 保存最佳模型
            if val_acc > best_val_acc or (val_acc == best_val_acc and val_profit_mae < best_val_profit_mae):
                best_val_acc = val_acc
                best_val_profit_mae = val_profit_mae if val_profit_mae else float('inf')
                patience_counter = 0
                
                model_path = os.path.join(self.data_dir, 'best_lstm_improved.pth')
                torch.save({
                    'model_state_dict': strategy.model.state_dict(),
                    'optimizer_state_dict': strategy.optimizer.state_dict(),
                    'epoch': epoch,
                    'val_acc': val_acc,
                    'val_profit_mae': val_profit_mae
                }, model_path, _use_new_zipfile_serialization=False)
                print(f"  ✅ 保存最佳模型: Val Acc={val_acc:.4f}, Val Profit MAE={val_profit_mae:.6f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  早停触发（patience={patience}）")
                    break
        
        self.results['lstm'] = {
            'best_val_acc': best_val_acc,
            'best_val_profit_mae': best_val_profit_mae,
            'model_path': os.path.join(self.data_dir, 'best_lstm_improved.pth')
        }
        
        return strategy
    
    def train_transformer_model(self, train_data, val_data, seq_length=50):
        """训练Transformer模型（支持收益率预测）"""
        print("\n" + "="*70)
        print("训练Transformer模型（支持收益率预测）")
        print("="*70)
        
        X_train, y_train, y_profit_train = train_data
        X_val, y_val, y_profit_val = val_data
        
        # 创建支持收益率预测的Transformer模型
        input_size = X_train.shape[2]  # 46维
        model = TradingTransformerWithProfit(
            input_size=input_size,
            nhead=8,
            num_layers=6,
            output_size=3,
            d_model=256,
            predict_profit=True
        ).to(self.device)
        
        # 优化器和损失函数
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
        action_criterion = nn.CrossEntropyLoss()
        profit_criterion = nn.MSELoss()
        
        # 准备数据
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
            torch.tensor(y_profit_train, dtype=torch.float32)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long),
            torch.tensor(y_profit_val, dtype=torch.float32)
        )
        
        # 根据序列长度动态调整批次大小（长序列需要更小的批次）
        if seq_length >= 500:
            batch_size = 8  # 长序列使用小批次
        elif seq_length >= 300:
            batch_size = 16
        else:
            batch_size = 32
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 训练模型
        best_val_acc = 0
        best_val_profit_mae = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(50):
            # 训练阶段
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            train_profit_errors = []
            
            for batch_x, batch_y, batch_y_profit in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_y_profit = batch_y_profit.to(self.device)
                
                optimizer.zero_grad()
                model_output = model(batch_x)
                
                if isinstance(model_output, tuple) and len(model_output) >= 2:
                    action_logits = model_output[0]
                    profit = model_output[1]
                    
                    # 动作分类损失
                    action_loss = action_criterion(action_logits, batch_y)
                    
                    # 收益率损失
                    profit_loss = profit_criterion(profit.squeeze(), batch_y_profit)
                    
                    # 组合损失
                    loss = action_loss + 1.0 * profit_loss
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    train_loss += loss.item()
                    predictions = torch.argmax(action_logits, dim=1)
                    train_correct += (predictions == batch_y).sum().item()
                    train_total += batch_y.size(0)
                    
                    # 计算收益率误差
                    profit_errors = torch.abs(profit.squeeze() - batch_y_profit)
                    train_profit_errors.extend(profit_errors.detach().cpu().numpy().tolist())
            
            train_acc = train_correct / train_total if train_total > 0 else 0
            train_profit_mae = np.mean(train_profit_errors) if train_profit_errors else None
            
            # 验证阶段
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            val_profit_errors = []
            
            with torch.no_grad():
                for batch_x, batch_y, batch_y_profit in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    batch_y_profit = batch_y_profit.to(self.device)
                    
                    model_output = model(batch_x)
                    
                    if isinstance(model_output, tuple) and len(model_output) >= 2:
                        action_logits = model_output[0]
                        profit = model_output[1]
                        
                        action_loss = action_criterion(action_logits, batch_y)
                        profit_loss = profit_criterion(profit.squeeze(), batch_y_profit)
                        loss = action_loss + 1.0 * profit_loss
                        
                        val_loss += loss.item()
                        predictions = torch.argmax(action_logits, dim=1)
                        val_correct += (predictions == batch_y).sum().item()
                        val_total += batch_y.size(0)
                        
                        profit_errors = torch.abs(profit.squeeze() - batch_y_profit)
                        val_profit_errors.extend(profit_errors.detach().cpu().numpy().tolist())
            
            val_acc = val_correct / val_total if val_total > 0 else 0
            val_profit_mae = np.mean(val_profit_errors) if val_profit_errors else None
            
            train_profit_str = f"{train_profit_mae:.6f}" if train_profit_mae is not None else "N/A"
            val_profit_str = f"{val_profit_mae:.6f}" if val_profit_mae is not None else "N/A"
            print(f"Epoch {epoch+1}/50: Train Acc={train_acc:.4f}, Train Profit MAE={train_profit_str}, "
                  f"Val Acc={val_acc:.4f}, Val Profit MAE={val_profit_str}")
            
            # 保存最佳模型
            if val_acc > best_val_acc or (val_acc == best_val_acc and val_profit_mae < best_val_profit_mae):
                best_val_acc = val_acc
                best_val_profit_mae = val_profit_mae if val_profit_mae else float('inf')
                patience_counter = 0
                
                model_path = os.path.join(self.data_dir, 'best_transformer_with_profit.pth')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_acc': val_acc,
                    'val_profit_mae': val_profit_mae
                }, model_path, _use_new_zipfile_serialization=False)
                print(f"  ✅ 保存最佳模型: Val Acc={val_acc:.4f}, Val Profit MAE={val_profit_mae:.6f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  早停触发（patience={patience}）")
                    break
        
        self.results['transformer'] = {
            'best_val_acc': best_val_acc,
            'best_val_profit_mae': best_val_profit_mae,
            'model_path': os.path.join(self.data_dir, 'best_transformer_with_profit.pth')
        }
        
        return model
    
    def train_gru_model(self, train_data, val_data, seq_length=50):
        """训练GRU模型（支持收益率预测）"""
        print("\n" + "="*70)
        print("训练GRU模型（支持收益率预测）")
        print("="*70)
        
        X_train, y_train, y_profit_train = train_data
        X_val, y_val, y_profit_val = val_data
        
        # 创建支持收益率预测的GRU模型
        input_size = X_train.shape[2]  # 46维
        model = TradingGRUWithProfit(
            input_size=input_size,
            hidden_size=128,
            num_layers=3,
            output_size=3,
            predict_profit=True
        ).to(self.device)
        
        # 优化器和损失函数
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        action_criterion = nn.CrossEntropyLoss()
        profit_criterion = nn.MSELoss()
        
        # 准备数据
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
            torch.tensor(y_profit_train, dtype=torch.float32)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long),
            torch.tensor(y_profit_val, dtype=torch.float32)
        )
        
        # 根据序列长度动态调整批次大小（长序列需要更小的批次）
        if seq_length >= 500:
            batch_size = 8  # 长序列使用小批次
        elif seq_length >= 300:
            batch_size = 16
        else:
            batch_size = 32
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 训练模型
        best_val_acc = 0
        best_val_profit_mae = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(50):
            # 训练阶段
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            train_profit_errors = []
            
            for batch_x, batch_y, batch_y_profit in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_y_profit = batch_y_profit.to(self.device)
                
                optimizer.zero_grad()
                model_output = model(batch_x)
                
                if isinstance(model_output, tuple) and len(model_output) >= 2:
                    action_logits = model_output[0]
                    profit = model_output[1]
                    
                    # 动作分类损失
                    action_loss = action_criterion(action_logits, batch_y)
                    
                    # 收益率损失
                    profit_loss = profit_criterion(profit.squeeze(), batch_y_profit)
                    
                    # 组合损失
                    loss = action_loss + 1.0 * profit_loss
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    train_loss += loss.item()
                    predictions = torch.argmax(action_logits, dim=1)
                    train_correct += (predictions == batch_y).sum().item()
                    train_total += batch_y.size(0)
                    
                    # 计算收益率误差
                    profit_errors = torch.abs(profit.squeeze() - batch_y_profit)
                    train_profit_errors.extend(profit_errors.detach().cpu().numpy().tolist())
            
            train_acc = train_correct / train_total if train_total > 0 else 0
            train_profit_mae = np.mean(train_profit_errors) if train_profit_errors else None
            
            # 验证阶段
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            val_profit_errors = []
            
            with torch.no_grad():
                for batch_x, batch_y, batch_y_profit in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    batch_y_profit = batch_y_profit.to(self.device)
                    
                    model_output = model(batch_x)
                    
                    if isinstance(model_output, tuple) and len(model_output) >= 2:
                        action_logits = model_output[0]
                        profit = model_output[1]
                        
                        action_loss = action_criterion(action_logits, batch_y)
                        profit_loss = profit_criterion(profit.squeeze(), batch_y_profit)
                        loss = action_loss + 1.0 * profit_loss
                        
                        val_loss += loss.item()
                        predictions = torch.argmax(action_logits, dim=1)
                        val_correct += (predictions == batch_y).sum().item()
                        val_total += batch_y.size(0)
                        
                        profit_errors = torch.abs(profit.squeeze() - batch_y_profit)
                        val_profit_errors.extend(profit_errors.detach().cpu().numpy().tolist())
            
            val_acc = val_correct / val_total if val_total > 0 else 0
            val_profit_mae = np.mean(val_profit_errors) if val_profit_errors else None
            
            train_profit_str = f"{train_profit_mae:.6f}" if train_profit_mae is not None else "N/A"
            val_profit_str = f"{val_profit_mae:.6f}" if val_profit_mae is not None else "N/A"
            print(f"Epoch {epoch+1}/50: Train Acc={train_acc:.4f}, Train Profit MAE={train_profit_str}, "
                  f"Val Acc={val_acc:.4f}, Val Profit MAE={val_profit_str}")
            
            # 保存最佳模型
            if val_acc > best_val_acc or (val_acc == best_val_acc and val_profit_mae < best_val_profit_mae):
                best_val_acc = val_acc
                best_val_profit_mae = val_profit_mae if val_profit_mae else float('inf')
                patience_counter = 0
                
                model_path = os.path.join(self.data_dir, 'best_gru_with_profit.pth')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_acc': val_acc,
                    'val_profit_mae': val_profit_mae
                }, model_path, _use_new_zipfile_serialization=False)
                print(f"  ✅ 保存最佳模型: Val Acc={val_acc:.4f}, Val Profit MAE={val_profit_mae:.6f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  早停触发（patience={patience}）")
                    break
        
        self.results['gru'] = {
            'best_val_acc': best_val_acc,
            'best_val_profit_mae': best_val_profit_mae,
            'model_path': os.path.join(self.data_dir, 'best_gru_with_profit.pth')
        }
        
        return model
    
    def calculate_class_weights(self, y):
        """计算类别权重以处理不平衡数据"""
        import numpy as np
        classes, counts = np.unique(y, return_counts=True)
        total_samples = len(y)
        weights = []
        for count in counts:
            weight = total_samples / (len(classes) * count)
            weights.append(weight)
        class_weights = torch.FloatTensor(weights).to(self.device)
        return class_weights
    
    def train_enhanced_transformer_model(self, train_data, val_data, seq_length=50):
        """训练Enhanced Transformer模型（保持大容量，使用大模型防过拟合方案）"""
        print("\n" + "="*70)
        print("训练Enhanced Transformer模型（保持大容量，使用大模型防过拟合方案）")
        print("="*70)
        
        X_train, y_train, y_profit_train = train_data
        X_val, y_val, y_profit_val = val_data
        
        # 计算类别权重（处理类别不平衡）
        class_weights = self.calculate_class_weights(y_train)
        print(f"\n类别权重: {class_weights.cpu().numpy()}")
        
        # 1. 使用保持大容量的Enhanced Transformer（d_model=512, num_layers=8）
        input_size = X_train.shape[2]  # 46维
        base_model = EnhancedTradingTransformerWithProfit(
            input_size=input_size,
            nhead=8,
            num_layers=8,  # 保持8层
            output_size=3,
            d_model=512,  # 保持512
            predict_profit=True
        ).to(self.device)
        
        # 打印基础模型参数量
        total_params = sum(p.numel() for p in base_model.parameters())
        print(f"\n基础模型参数量: {total_params:,} ({total_params * 4 / (1024*1024):.2f} MB)")
        
        # 2. 尝试加载预训练模型（如果有）
        pretrained_path = os.path.join(self.data_dir, 'pretrained_return_model.pth')
        if os.path.exists(pretrained_path):
            try:
                print(f"\n尝试加载预训练模型: {pretrained_path}")
                pretrained = torch.load(pretrained_path, map_location=self.device, weights_only=False)
                model_dict = base_model.state_dict()
                pretrained_dict = {k: v for k, v in pretrained.get('model_state_dict', pretrained).items() 
                                 if k in model_dict and model_dict[k].shape == v.shape}
                model_dict.update(pretrained_dict)
                base_model.load_state_dict(model_dict, strict=False)
                print(f"✅ 加载了 {len(pretrained_dict)} 个预训练层")
            except Exception as e:
                print(f"⚠️ 预训练模型加载失败: {e}，使用随机初始化")
        
        # 3. 参数高效微调（PEFT）- 冻结Transformer层，只训练分类头和收益率头
        print(f"\n应用参数高效微调（PEFT）")
        print(f"策略: 冻结Transformer所有层，只训练分类头和收益率头")
        
        # 冻结Transformer的所有层（input_projection, transformer, attention_pool）
        for name, param in base_model.named_parameters():
            if 'input_projection' in name or 'transformer' in name or 'attention_pool' in name:
                param.requires_grad = False
        
        # 保持分类头和收益率头可训练
        for name, param in base_model.named_parameters():
            if 'action_head' in name or 'profit_head' in name:
                param.requires_grad = True
        
        model = base_model
        
        # 打印PEFT参数统计
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        print(f"\nPEFT参数统计:")
        print(f"  总参数量: {total_params:,}")
        print(f"  冻结参数: {frozen_params:,} ({frozen_params/total_params*100:.2f}%)")
        print(f"  可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        print(f"  数据/可训练参数比例: {len(X_train)}/{trainable_params} = {len(X_train)/trainable_params:.2f}:1")
        
        # 4. 优化器和损失函数（增强正则化）
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], 
            lr=2e-4,
            weight_decay=5e-3  # 增加权重衰减（从1e-3增加到5e-3）
        )
        # 使用Warmup + CosineAnnealingLR
        from torch.optim.lr_scheduler import LambdaLR
        def lr_lambda(epoch):
            warmup_epochs = 5
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            else:
                return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (50 - warmup_epochs)))
        scheduler = LambdaLR(optimizer, lr_lambda)
        
        # 5. 使用标签平滑 + 类别权重 + Focal Loss
        print(f"\n使用标签平滑 (smoothing=0.2) + 类别权重 + Focal Loss")
        action_criterion_smooth = LabelSmoothingCrossEntropy(smoothing=0.2, weight=class_weights)
        action_criterion_focal = FocalLoss(alpha=class_weights, gamma=2.0).to(self.device)
        # 组合使用：70% Label Smoothing + 30% Focal Loss
        action_criterion = lambda pred, target: 0.7 * action_criterion_smooth(pred, target) + 0.3 * action_criterion_focal(pred, target)
        profit_criterion = nn.MSELoss()
        
        # 准备数据
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
            torch.tensor(y_profit_train, dtype=torch.float32)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long),
            torch.tensor(y_profit_val, dtype=torch.float32)
        )
        
        # 6. 数据增强（Mixup）
        use_mixup = True
        mixup_alpha = 0.2
        
        # 根据序列长度动态调整批次大小（长序列需要更小的批次）
        if seq_length >= 500:
            batch_size = 8  # 长序列使用小批次
        elif seq_length >= 300:
            batch_size = 16
        else:
            batch_size = 32
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 训练模型
        best_val_acc = 0
        best_val_profit_mae = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(50):
            # 训练阶段
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            train_profit_errors = []
            
            for batch_x, batch_y, batch_y_profit in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_y_profit = batch_y_profit.to(self.device)
                
                optimizer.zero_grad()
                model_output = model(batch_x)
                
                if isinstance(model_output, tuple) and len(model_output) >= 2:
                    action_logits = model_output[0]
                    profit = model_output[1]
                    
                    # 动作分类损失
                    action_loss = action_criterion(action_logits, batch_y)
                    
                    # 收益率损失
                    profit_loss = profit_criterion(profit.squeeze(), batch_y_profit)
                    
                    # 组合损失
                    loss = action_loss + 1.0 * profit_loss
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    train_loss += loss.item()
                    predictions = torch.argmax(action_logits, dim=1)
                    train_correct += (predictions == batch_y).sum().item()
                    train_total += batch_y.size(0)
                    
                    # 计算收益率误差
                    profit_errors = torch.abs(profit.squeeze() - batch_y_profit)
                    train_profit_errors.extend(profit_errors.detach().cpu().numpy().tolist())
            
            train_acc = train_correct / train_total if train_total > 0 else 0
            train_profit_mae = np.mean(train_profit_errors) if train_profit_errors else None
            
            # 验证阶段
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            val_profit_errors = []
            
            with torch.no_grad():
                for batch_x, batch_y, batch_y_profit in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    batch_y_profit = batch_y_profit.to(self.device)
                    
                    model_output = model(batch_x)
                    
                    if isinstance(model_output, tuple) and len(model_output) >= 2:
                        action_logits = model_output[0]
                        profit = model_output[1]
                        
                        action_loss = action_criterion(action_logits, batch_y)
                        profit_loss = profit_criterion(profit.squeeze(), batch_y_profit)
                        loss = action_loss + 1.0 * profit_loss
                        
                        val_loss += loss.item()
                        predictions = torch.argmax(action_logits, dim=1)
                        val_correct += (predictions == batch_y).sum().item()
                        val_total += batch_y.size(0)
                        
                        profit_errors = torch.abs(profit.squeeze() - batch_y_profit)
                        val_profit_errors.extend(profit_errors.detach().cpu().numpy().tolist())
            
            val_acc = val_correct / val_total if val_total > 0 else 0
            val_profit_mae = np.mean(val_profit_errors) if val_profit_errors else None
            
            # 更新学习率
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            train_profit_str = f"{train_profit_mae:.6f}" if train_profit_mae is not None else "N/A"
            val_profit_str = f"{val_profit_mae:.6f}" if val_profit_mae is not None else "N/A"
            print(f"Epoch {epoch+1}/50: Train Acc={train_acc:.4f}, Train Profit MAE={train_profit_str}, "
                  f"Val Acc={val_acc:.4f}, Val Profit MAE={val_profit_str}, LR={current_lr:.6f}")
            
            # 保存最佳模型
            if val_acc > best_val_acc or (val_acc == best_val_acc and val_profit_mae < best_val_profit_mae):
                best_val_acc = val_acc
                best_val_profit_mae = val_profit_mae if val_profit_mae else float('inf')
                patience_counter = 0
                
                model_path = os.path.join(self.data_dir, 'best_enhanced_transformer_lora.pth')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                    'val_acc': val_acc,
                    'val_profit_mae': val_profit_mae
                }, model_path, _use_new_zipfile_serialization=False)
                print(f"  ✅ 保存最佳模型: Val Acc={val_acc:.4f}, Val Profit MAE={val_profit_mae:.6f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  早停触发（patience={patience}）")
                    break
        
        self.results['enhanced_transformer_peft'] = {
            'best_val_acc': best_val_acc,
            'best_val_profit_mae': best_val_profit_mae,
            'model_path': os.path.join(self.data_dir, 'best_enhanced_transformer_peft.pth')
        }
    
    def train_moe_transformer_model(self, train_data, val_data, seq_length=50):
        """训练MoE Transformer模型（使用MoE和稀疏注意力）"""
        print("\n" + "="*70)
        print("训练MoE Transformer模型（使用MoE和稀疏注意力）")
        print("="*70)
        
        X_train, y_train, y_profit_train = train_data
        X_val, y_val, y_profit_val = val_data
        
        # 计算类别权重
        class_weights = self.calculate_class_weights(y_train)
        print(f"\n类别权重: {class_weights.cpu().numpy()}")
        
        # 创建MoE Transformer模型（减少参数量以节省内存）
        input_size = X_train.shape[2]
        model = MoETradingTransformerWithProfit(
            input_size=input_size,
            nhead=8,
            num_layers=6,  # 从8层减少到6层（减少内存）
            output_size=3,
            d_model=256,  # 从512减少到256（减少内存）
            predict_profit=True,
            num_experts=4,  # 从8个专家减少到4个（减少内存）
            top_k=2,  # 每次激活2个专家（稀疏激活率50%）
            window_size=20,  # 局部窗口注意力（每个位置只关注前后20个位置）
            attention_dropout_rate=0.1  # 10%的注意力头dropout
        ).to(self.device)
        
        # 打印模型参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n模型参数量:")
        print(f"  总参数量: {total_params:,} ({total_params * 4 / (1024*1024):.2f} MB)")
        print(f"  可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        print(f"  数据/可训练参数比例: {len(X_train)}/{trainable_params} = {len(X_train)/trainable_params:.2f}:1")
        print(f"\nMoE配置:")
        print(f"  专家数量: 4")
        print(f"  每次激活专家数: 2 (稀疏激活率: 50%)")
        print(f"  稀疏注意力窗口: 20 (局部窗口)")
        print(f"  注意力头dropout: 10%")
        print(f"  层数: 6 (从8减少)")
        print(f"  d_model: 256 (从512减少)")
        
        # 优化器和损失函数
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
        
        action_criterion = LabelSmoothingCrossEntropy(smoothing=0.2, weight=class_weights)
        profit_criterion = nn.MSELoss()
        
        # MoE负载均衡损失的权重
        moe_aux_loss_weight = 0.01
        
        # 准备数据
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
            torch.tensor(y_profit_train, dtype=torch.float32)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long),
            torch.tensor(y_profit_val, dtype=torch.float32)
        )
        
        # 根据序列长度动态调整批次大小（长序列需要更小的批次）
        if seq_length >= 500:
            batch_size = 8  # 长序列使用小批次
        elif seq_length >= 300:
            batch_size = 16
        else:
            batch_size = 32
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 训练模型
        best_val_acc = 0
        best_val_profit_mae = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(50):
            # 训练阶段
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            train_profit_errors = []
            total_moe_aux_loss = 0
            
            for batch_x, batch_y, batch_y_profit in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_y_profit = batch_y_profit.to(self.device)
                
                # 数据增强（Mixup + 高级增强）
                apply_aug = np.random.random()
                if apply_aug < 0.3:
                    # Mixup
                    indices = torch.randperm(batch_x.size(0))
                    batch_x2 = batch_x[indices]
                    batch_y2 = batch_y[indices]
                    batch_y_profit2 = batch_y_profit[indices]
                    
                    lam = np.random.beta(0.2, 0.2)
                    batch_x = lam * batch_x + (1 - lam) * batch_x2
                    batch_y_profit = lam * batch_y_profit + (1 - lam) * batch_y_profit2
                elif apply_aug < 0.6:
                    # 高级增强（时间序列特定的增强）
                    batch_x_np = batch_x.cpu().numpy()
                    augmented_batch = []
                    for i in range(batch_x_np.shape[0]):
                        aug_x = AdvancedDataAugmentation.apply_augmentation(batch_x_np[i], 'random')
                        augmented_batch.append(aug_x)
                    batch_x = torch.tensor(np.array(augmented_batch), dtype=torch.float32).to(self.device)
                
                optimizer.zero_grad()
                model_output = model(batch_x)
                
                # MoE模型返回((outputs), aux_loss)
                moe_aux_loss = 0
                if isinstance(model_output, tuple) and len(model_output) == 2:
                    outputs, moe_aux_loss = model_output
                    if isinstance(outputs, tuple) and len(outputs) >= 2:
                        action_logits = outputs[0]
                        profit = outputs[1]
                    else:
                        action_logits = outputs
                        profit = None
                elif isinstance(model_output, tuple) and len(model_output) >= 2:
                    # 标准模型返回(action_logits, profit)
                    action_logits = model_output[0]
                    profit = model_output[1]
                else:
                    action_logits = model_output
                    profit = None
                
                # 动作分类损失
                action_loss = action_criterion(action_logits, batch_y)
                
                # 收益率损失
                if profit is not None:
                    profit_loss = profit_criterion(profit.squeeze(), batch_y_profit)
                    loss = action_loss + 1.0 * profit_loss + moe_aux_loss_weight * moe_aux_loss
                else:
                    loss = action_loss + moe_aux_loss_weight * moe_aux_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                total_moe_aux_loss += moe_aux_loss.item() if isinstance(moe_aux_loss, torch.Tensor) else moe_aux_loss
                
                _, predicted = torch.max(action_logits.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
                
                if profit is not None:
                    train_profit_errors.append(torch.abs(profit.squeeze() - batch_y_profit).mean().item())
            
            # 验证阶段
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            val_profit_errors = []
            val_moe_aux_loss = 0
            
            with torch.no_grad():
                for batch_x, batch_y, batch_y_profit in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    batch_y_profit = batch_y_profit.to(self.device)
                    
                    model_output = model(batch_x)
                    
                    # MoE模型返回((outputs), aux_loss)
                    moe_aux_loss = 0
                    if isinstance(model_output, tuple) and len(model_output) == 2:
                        outputs, moe_aux_loss = model_output
                        if isinstance(outputs, tuple) and len(outputs) >= 2:
                            action_logits = outputs[0]
                            profit = outputs[1]
                        else:
                            action_logits = outputs
                            profit = None
                    elif isinstance(model_output, tuple) and len(model_output) >= 2:
                        # 标准模型返回(action_logits, profit)
                        action_logits = model_output[0]
                        profit = model_output[1]
                    else:
                        action_logits = model_output
                        profit = None
                    
                    action_loss = action_criterion(action_logits, batch_y)
                    
                    if profit is not None:
                        profit_loss = profit_criterion(profit.squeeze(), batch_y_profit)
                        loss = action_loss + 1.0 * profit_loss + moe_aux_loss_weight * moe_aux_loss
                    else:
                        loss = action_loss + moe_aux_loss_weight * moe_aux_loss
                    
                    val_loss += loss.item()
                    val_moe_aux_loss += moe_aux_loss.item() if isinstance(moe_aux_loss, torch.Tensor) else moe_aux_loss
                    
                    _, predicted = torch.max(action_logits.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
                    
                    if profit is not None:
                        val_profit_errors.append(torch.abs(profit.squeeze() - batch_y_profit).mean().item())
            
            train_acc = train_correct / train_total if train_total > 0 else 0
            val_acc = val_correct / val_total if val_total > 0 else 0
            train_profit_mae = np.mean(train_profit_errors) if train_profit_errors else None
            val_profit_mae = np.mean(val_profit_errors) if val_profit_errors else None
            
            # 更新学习率
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            train_profit_str = f"{train_profit_mae:.6f}" if train_profit_mae is not None else "N/A"
            val_profit_str = f"{val_profit_mae:.6f}" if val_profit_mae is not None else "N/A"
            print(f"Epoch {epoch+1}/50: Train Acc={train_acc:.4f}, Train Profit MAE={train_profit_str}, "
                  f"Val Acc={val_acc:.4f}, Val Profit MAE={val_profit_str}, "
                  f"MoE Aux Loss={total_moe_aux_loss/len(train_loader):.6f}, LR={current_lr:.6f}")
            
            # 保存最佳模型
            if val_acc > best_val_acc or (val_acc == best_val_acc and val_profit_mae is not None and val_profit_mae < best_val_profit_mae):
                best_val_acc = val_acc
                if val_profit_mae is not None:
                    best_val_profit_mae = val_profit_mae
                
                model_path = os.path.join(self.data_dir, 'best_moe_transformer.pth')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'val_acc': best_val_acc,
                    'val_profit_mae': best_val_profit_mae,
                    'epoch': epoch + 1
                }, model_path, _use_new_zipfile_serialization=False)
                print(f"  ✅ 保存最佳模型: Val Acc={best_val_acc:.4f}, Val Profit MAE={val_profit_str}")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  早停触发（patience={patience}）")
                    break
        
        self.results['moe_transformer'] = {
            'best_val_acc': best_val_acc,
            'best_val_profit_mae': best_val_profit_mae,
            'model_path': os.path.join(self.data_dir, 'best_moe_transformer.pth')
        }
        
        return model
    
    def compare_models(self):
        """对比模型结果"""
        print("\n" + "="*70)
        print("模型对比结果")
        print("="*70)
        
        print("\n模型性能对比:")
        print("-" * 70)
        print(f"{'模型':<20} {'验证准确率':<15} {'收益率MAE':<15} {'备注':<20}")
        print("-" * 70)
        
        for model_name, result in self.results.items():
            val_acc = result.get('best_val_acc', 0.0)
            profit_mae = result.get('best_val_profit_mae', None)
            note = result.get('note', '')
            
            profit_str = f"{profit_mae:.6f}" if profit_mae else "N/A"
            print(f"{model_name:<20} {val_acc:<15.4f} {profit_str:<15} {note:<20}")
        
        # 保存对比结果
        comparison_path = os.path.join(self.data_dir, 'model_comparison_results.txt')
        with open(comparison_path, 'w', encoding='utf-8') as f:
            f.write("模型对比结果\n")
            f.write("="*70 + "\n")
            f.write(f"{'模型':<20} {'验证准确率':<15} {'收益率MAE':<15} {'备注':<20}\n")
            f.write("-" * 70 + "\n")
            for model_name, result in self.results.items():
                val_acc = result.get('best_val_acc', 0.0)
                profit_mae = result.get('best_val_profit_mae', None)
                note = result.get('note', '')
                profit_str = f"{profit_mae:.6f}" if profit_mae else "N/A"
                f.write(f"{model_name:<20} {val_acc:<15.4f} {profit_str:<15} {note:<20}\n")
        
        print(f"\n对比结果已保存到: {comparison_path}")
    
    def run(self):
        """运行完整的训练和对比流程"""
        print("="*70)
        print("多模型对比训练")
        print("="*70)
        
        # 1. 加载数据
        df = self.load_training_data()
        
        # 2. 分析数据并确定最优序列长度
        total_samples = len(df)
        print(f"\n数据量: {total_samples:,} 个样本")
        
        # 动态确定序列长度：覆盖尽可能多的历史成交情况
        # 策略：平衡历史覆盖和GPU内存限制
        # 注意：序列长度会影响GPU内存，需要根据GPU容量调整
        if total_samples < 2000:
            seq_length = min(100, int(total_samples * 0.3))  # 小数据集：使用30%的数据作为序列
        elif total_samples < 10000:
            seq_length = min(300, int(total_samples * 0.15))  # 中等数据集：使用15%的数据作为序列，最多300步（减少内存占用）
        else:
            seq_length = min(500, int(total_samples * 0.1))  # 大数据集：使用10%的数据作为序列，最多500步（减少内存占用）
        
        # 确保序列长度至少为50
        seq_length = max(50, seq_length)
        
        print(f"📊 动态序列长度: {seq_length} 步")
        print(f"   覆盖历史: {seq_length} 步 ≈ {seq_length/60:.1f} 小时（假设1分钟K线）")
        print(f"   可用样本数: {total_samples - seq_length:,} 个")
        print(f"   数据利用率: {(total_samples - seq_length)/total_samples*100:.1f}%")
        
        # 3. 准备数据
        train_data, val_data = self.prepare_data_for_model(df, model_type='lstm', seq_length=seq_length)
        
        # 4. 训练LSTM模型（改进版）
        self.train_lstm_model(train_data, val_data, seq_length)
        
        # 5. 训练Transformer模型（支持收益率预测）
        self.train_transformer_model(train_data, val_data, seq_length)
        
        # 6. 训练GRU模型（支持收益率预测）
        self.train_gru_model(train_data, val_data, seq_length)
        
        # 7. 训练Enhanced Transformer模型（保持大容量，使用大模型防过拟合方案：PEFT + 数据增强）
        self.train_enhanced_transformer_model(train_data, val_data, seq_length)
        
        # 8. 训练MoE Transformer模型（使用MoE和稀疏注意力）
        self.train_moe_transformer_model(train_data, val_data, seq_length)
        
        # 6. 对比模型
        self.compare_models()
        
        print("\n✅ 训练和对比完成！")

if __name__ == "__main__":
    trainer = ModelComparisonTrainer()
    trainer.run()
