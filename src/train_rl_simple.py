#!/usr/bin/env python3
"""
简化的强化学习策略训练 - 使用标准的分类网络
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_with_detailed_logging import TradingDataset
from config import TrainingConfig

def create_simple_logger(log_file):
    """创建简单日志记录器"""
    class SimpleLogger:
        def __init__(self, log_file):
            self.log_file = log_file
            
        def log(self, message):
            timestamp = datetime.now().strftime('[%Y-%m-%d %H:%M:%S.%f')[:-3] + ']'
            log_msg = f"{timestamp} [INFO] {message}"
            print(log_msg)
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_msg + '\n')
    
    return SimpleLogger(log_file)


class SimpleRLNetwork(nn.Module):
    """简化的强化学习网络 - 用于监督学习"""
    def __init__(self, input_size=12, num_classes=3):
        super(SimpleRLNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)


def train_simple_rl(data_file, output_dir, logger):
    """简化的RL训练"""
    
    logger.log("=" * 80)
    logger.log("🚀 强化学习策略 - 简化监督学习模式")
    logger.log("=" * 80)
    
    # 加载数据（使用已经分层分割好的数据）
    data_dir = os.path.dirname(data_file)
    # 从 full_20260120_192018.csv 提取 20260120_192018
    basename = os.path.basename(data_file)
    timestamp = '_'.join(basename.split('_')[1:]).replace('.csv', '')
    
    train_file = os.path.join(data_dir, f'train_{timestamp}.csv')
    val_file = os.path.join(data_dir, f'val_{timestamp}.csv')
    test_file = os.path.join(data_dir, f'test_{timestamp}.csv')
    
    logger.log(f"📊 加载分层数据:")
    logger.log(f"  训练集: {train_file}")
    logger.log(f"  验证集: {val_file}")
    logger.log(f"  测试集: {test_file}")
    
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)
    
    # 特征列
    feature_cols = [
        'rsi_1m', 'rsi_5m', 'atr', 'boll_position',
        'boll_upper', 'boll_lower', 'boll_mid',
        'price_change_1', 'price_change_5',
        'volatility', 'volume_1m', 'price_current'
    ]
    
    logger.log(f"训练集: {len(train_df)} 条")
    logger.log(f"验证集: {len(val_df)} 条")
    logger.log(f"测试集: {len(test_df)} 条")
    
    # 标签分布
    logger.log(f"训练集标签分布: {dict(train_df['label'].value_counts().sort_index())}")
    logger.log(f"验证集标签分布: {dict(val_df['label'].value_counts().sort_index())}")
    
    # 创建数据集
    train_dataset = TradingDataset(train_df, feature_cols)
    val_dataset = TradingDataset(val_df, feature_cols)
    test_dataset = TradingDataset(test_df, feature_cols)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.log(f"🔧 使用设备: {device}")
    
    model = SimpleRLNetwork(input_size=len(feature_cols)).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"总参数数量: {total_params:,}")
    logger.log(f"可训练参数数量: {trainable_params:,}")
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # 训练
    best_val_acc = 0
    patience_counter = 0
    max_patience = 10
    
    logger.log("\n开始训练...")
    logger.log("=" * 80)
    
    for epoch in range(1, 51):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = train_correct / train_total if train_total > 0 else 0
        train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device)
                
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = val_correct / val_total if val_total > 0 else 0
        val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        
        # 学习率调度
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        logger.log(f"Epoch {epoch:2d}/50 | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | LR: {current_lr:.6f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = os.path.join(output_dir, '强化学习策略_best.pth')
            torch.save(model.state_dict(), model_path)
            logger.log(f"  🏆 新的最佳准确率: {best_val_acc:.4f}, 模型已保存")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 早停
        if patience_counter >= max_patience:
            logger.log(f"⏹️ 早停触发，停止训练")
            break
    
    # 测试阶段
    logger.log("\n" + "=" * 80)
    logger.log("📊 测试阶段")
    logger.log("=" * 80)
    
    model_path = os.path.join(output_dir, '强化学习策略_best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    test_loss = 0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    test_acc = test_correct / test_total if test_total > 0 else 0
    test_loss = test_loss / len(test_loader) if len(test_loader) > 0 else 0
    
    logger.log(f"\n✅ 强化学习策略训练完成!")
    logger.log(f"最佳验证准确率: {best_val_acc:.4f}")
    logger.log(f"测试准确率: {test_acc:.4f}")
    logger.log(f"测试损失: {test_loss:.4f}")
    
    return {
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_loss': test_loss
    }

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='简化强化学习策略训练')
    parser.add_argument('--data-file', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, 'rl_simple_training.log')
    logger = create_simple_logger(log_file)
    
    result = train_simple_rl(args.data_file, args.output_dir, logger)
    
    print("\n" + "=" * 80)
    print("🎉 训练完成！")
    print(f"最佳验证准确率: {result['best_val_acc']:.4f}")
    print(f"测试准确率: {result['test_acc']:.4f}")
    print("=" * 80)
