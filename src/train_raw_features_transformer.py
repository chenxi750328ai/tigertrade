#!/usr/bin/env python3
"""
基于原生特征的Transformer交易模型
不使用人为设计的指标（RSI/BOLL/MACD），让模型自己从原始数据中学习
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime
import json

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)


class TradingSequenceDataset(Dataset):
    """交易序列数据集"""
    
    def __init__(self, csv_file, sequence_length=128, predict_horizon=10):
        """
        Args:
            csv_file: 数据文件路径
            sequence_length: 输入序列长度（过去N个时间点）
            predict_horizon: 预测未来N个时间点后的收益
        """
        self.df = pd.read_csv(csv_file)
        self.sequence_length = sequence_length
        self.predict_horizon = predict_horizon
        
        # 特征列（原生特征）
        self.feature_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'price_change', 'price_change_pct',
            'time_delta', 'price_range', 'price_range_pct',
            'volume_change', 'volume_change_pct'
        ]
        
        # 标准化（使用训练集的统计量）
        self.features = self.df[self.feature_cols].values.astype(np.float32)
        
        # 计算未来收益（标签）
        self.df['future_return'] = self.df['close'].shift(-predict_horizon) / self.df['close'] - 1
        
        # 三分类：买入(+1), 持有(0), 卖出(-1)
        self.df['action'] = 0
        self.df.loc[self.df['future_return'] > 0.002, 'action'] = 1   # 上涨>0.2% → 买入
        self.df.loc[self.df['future_return'] < -0.002, 'action'] = -1  # 下跌>0.2% → 卖出
        
        self.labels = self.df['action'].values
        
        # 有效的样本索引（需要足够的历史和未来数据）
        self.valid_indices = list(range(
            sequence_length, 
            len(self.df) - predict_horizon
        ))
        
        print(f"{'='*60}")
        print(f"数据集: {Path(csv_file).name}")
        print(f"{'='*60}")
        print(f"总数据量: {len(self.df):,}条")
        print(f"序列长度: {sequence_length}")
        print(f"预测间隔: {predict_horizon}个时间点后")
        print(f"有效样本: {len(self.valid_indices):,}个")
        
        # 标签分布
        action_counts = pd.Series(self.labels[self.valid_indices]).value_counts()
        print(f"\n标签分布:")
        print(f"  买入(+1):  {action_counts.get(1, 0):>6}个 ({action_counts.get(1, 0)/len(self.valid_indices)*100:>5.1f}%)")
        print(f"  持有(0):   {action_counts.get(0, 0):>6}个 ({action_counts.get(0, 0)/len(self.valid_indices)*100:>5.1f}%)")
        print(f"  卖出(-1):  {action_counts.get(-1, 0):>6}个 ({action_counts.get(-1, 0)/len(self.valid_indices)*100:>5.1f}%)")
        print(f"{'='*60}\n")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        
        # 输入序列：过去sequence_length个时间点的特征
        seq_start = real_idx - self.sequence_length
        seq_end = real_idx
        
        sequence = self.features[seq_start:seq_end]  # (sequence_length, num_features)
        label = self.labels[real_idx]
        
        # 转为tensor
        sequence = torch.from_numpy(sequence)
        label = torch.tensor(label, dtype=torch.long) + 1  # 转为 0, 1, 2（方便交叉熵）
        
        return sequence, label


class TransformerTradingModel(nn.Module):
    """基于Transformer的交易模型"""
    
    def __init__(self, num_features, d_model=128, nhead=8, num_layers=4, dropout=0.1):
        """
        Args:
            num_features: 输入特征数量
            d_model: Transformer隐藏维度
            nhead: 多头注意力的头数
            num_layers: Transformer层数
            dropout: Dropout比例
        """
        super().__init__()
        
        # 输入投影
        self.input_projection = nn.Linear(num_features, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 3)  # 三分类：卖出/持有/买入
        )
        
        print(f"{'='*60}")
        print(f"🤖 模型架构")
        print(f"{'='*60}")
        print(f"输入特征数: {num_features}")
        print(f"Transformer维度: {d_model}")
        print(f"注意力头数: {nhead}")
        print(f"Transformer层数: {num_layers}")
        print(f"Dropout: {dropout}")
        print(f"输出: 3类（卖出/持有/买入）")
        print(f"{'='*60}\n")
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, sequence_length, num_features)
        Returns:
            (batch_size, 3) - 三个类别的logits
        """
        # 输入投影
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # 位置编码
        x = self.pos_encoding(x)
        
        # Transformer编码
        x = self.transformer(x)  # (batch, seq_len, d_model)
        
        # 取最后一个时间步的表示
        x = x[:, -1, :]  # (batch, d_model)
        
        # 分类
        x = self.fc(x)  # (batch, 3)
        
        return x


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 计算位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (sequences, labels) in enumerate(dataloader):
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # 每50批次打印一次
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx+1}/{len(dataloader)}: Loss={loss.item():.4f}, Acc={correct/total*100:.2f}%")
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    # 计算每个类别的准确率
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    class_acc = {}
    for cls in [0, 1, 2]:
        mask = all_labels == cls
        if mask.sum() > 0:
            class_acc[cls] = (all_preds[mask] == all_labels[mask]).mean()
    
    return avg_loss, accuracy, class_acc


def main():
    print("\n" + "="*80)
    print("🚀 基于原生特征的Transformer交易模型训练")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"核心理念: 不使用RSI/BOLL/MACD等人为指标，让模型自己学习")
    print("="*80 + "\n")
    
    # 配置
    config = {
        'train_file': '/home/cx/trading_data/train_raw_features.csv',
        'val_file': '/home/cx/trading_data/val_raw_features.csv',
        'sequence_length': 128,      # 用过去128个时间点
        'predict_horizon': 10,       # 预测10个时间点后的走势
        'd_model': 128,              # Transformer维度
        'nhead': 8,                  # 注意力头数
        'num_layers': 4,             # Transformer层数
        'dropout': 0.2,              # Dropout
        'batch_size': 64,
        'learning_rate': 0.0001,
        'num_epochs': 50,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"配置:")
    for key, value in config.items():
        if not key.endswith('_file'):
            print(f"  {key}: {value}")
    print()
    
    # 加载数据
    train_dataset = TradingSequenceDataset(
        config['train_file'],
        sequence_length=config['sequence_length'],
        predict_horizon=config['predict_horizon']
    )
    
    val_dataset = TradingSequenceDataset(
        config['val_file'],
        sequence_length=config['sequence_length'],
        predict_horizon=config['predict_horizon']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2
    )
    
    # 创建模型
    num_features = len(train_dataset.feature_cols)
    model = TransformerTradingModel(
        num_features=num_features,
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )
    
    model = model.to(config['device'])
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # 训练
    print(f"\n{'='*80}")
    print(f"开始训练")
    print(f"{'='*80}\n")
    
    best_val_acc = 0
    best_epoch = 0
    training_history = []
    
    for epoch in range(config['num_epochs']):
        print(f"Epoch {epoch+1}/{config['num_epochs']}")
        print("-" * 60)
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, config['device']
        )
        
        # 验证
        val_loss, val_acc, class_acc = validate(
            model, val_loader, criterion, config['device']
        )
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 打印结果
        print(f"\n结果:")
        print(f"  训练 - Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}%")
        print(f"  验证 - Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%")
        print(f"  各类别准确率:")
        print(f"    卖出(-1): {class_acc.get(0, 0)*100:.2f}%")
        print(f"    持有(0):  {class_acc.get(1, 0)*100:.2f}%")
        print(f"    买入(+1): {class_acc.get(2, 0)*100:.2f}%")
        print()
        
        # 记录历史
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'class_acc': class_acc
        })
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            
            save_dir = Path('/home/cx/tigertrade/models')
            save_dir.mkdir(exist_ok=True)
            
            model_path = save_dir / 'transformer_raw_features_best.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, model_path)
            
            print(f"✅ 保存最佳模型: {model_path} (验证准确率: {val_acc*100:.2f}%)")
        
        print("="*80 + "\n")
    
    # 训练完成
    print(f"\n{'='*80}")
    print(f"✅ 训练完成！")
    print(f"{'='*80}")
    print(f"最佳验证准确率: {best_val_acc*100:.2f}% (Epoch {best_epoch})")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 保存训练历史
    history_file = '/home/cx/tigertrade/results/transformer_raw_features_history.json'
    Path(history_file).parent.mkdir(exist_ok=True)
    
    with open(history_file, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"\n训练历史已保存: {history_file}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
