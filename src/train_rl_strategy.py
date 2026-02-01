#!/usr/bin/env python3
"""
强化学习策略专用训练脚本
将强化学习问题转换为监督学习问题
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

from strategies.rl_trading_strategy import RLTradingNetwork
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

def train_rl_as_supervised(data_file, output_dir, logger):
    """将RL网络用于监督学习训练"""
    
    logger.log("=" * 80)
    logger.log("🚀 强化学习策略 - 监督学习训练模式")
    logger.log("=" * 80)
    
    # 加载数据
    logger.log(f"📊 加载数据: {data_file}")
    df = pd.read_csv(data_file)
    logger.log(f"总数据量: {len(df)} 条")
    
    # 特征列（与数据采集时的特征名称匹配）
    feature_cols = [
        'rsi_1m', 'rsi_5m', 'atr', 'boll_position',
        'boll_upper', 'boll_lower', 'boll_mid',
        'price_change_1', 'price_change_5',
        'volatility', 'volume_1m', 'price_current'
    ]
    
    # 数据分割
    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.15)
    
    train_df = df[:train_size]
    val_df = df[train_size:train_size + val_size]
    test_df = df[train_size + val_size:]
    
    logger.log(f"训练集: {len(train_df)} 条")
    logger.log(f"验证集: {len(val_df)} 条")
    logger.log(f"测试集: {len(test_df)} 条")
    
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
    
    model = RLTradingNetwork().to(device)
    
    # 统计参数
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
        
        for batch_idx, (features, labels) in enumerate(train_loader):
            features = features.to(device)
            labels = labels.to(device)
            
            # RLTradingNetwork需要3D输入: (batch, seq_len, features)
            if len(features.shape) == 2:
                features = features.unsqueeze(1)  # (batch, features) -> (batch, 1, features)
            
            optimizer.zero_grad()
            action_probs, _ = model(features)  # RLTradingNetwork返回(action_probs, q_values)
            loss = criterion(action_probs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(action_probs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = train_correct / train_total if train_total > 0 else 0
        train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_predictions = []
        
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device)
                
                # RLTradingNetwork需要3D输入
                if len(features.shape) == 2:
                    features = features.unsqueeze(1)
                
                action_probs, _ = model(features)
                loss = criterion(action_probs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(action_probs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                all_predictions.extend(predicted.cpu().tolist())
        
        # 调试信息：检查预测分布
        if epoch == 1:
            from collections import Counter
            pred_dist = Counter(all_predictions)
            logger.log(f"  验证集预测分布: {dict(pred_dist)}")
        
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
    if not os.path.exists(model_path):
        logger.log(f"⚠️ 没有保存最佳模型（验证准确率为0），使用最后训练的模型进行测试")
    else:
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.eval()
    
    test_loss = 0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            # RLTradingNetwork需要3D输入
            if len(features.shape) == 2:
                features = features.unsqueeze(1)
            
            action_probs, _ = model(features)
            loss = criterion(action_probs, labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(action_probs.data, 1)
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
    
    parser = argparse.ArgumentParser(description='强化学习策略训练')
    parser.add_argument('--data-file', type=str, required=True, help='训练数据CSV文件')
    parser.add_argument('--output-dir', type=str, required=True, help='模型输出目录')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, 'rl_training.log')
    logger = create_simple_logger(log_file)
    
    result = train_rl_as_supervised(args.data_file, args.output_dir, logger)
    
    print("\n" + "=" * 80)
    print("🎉 训练完成！")
    print(f"最佳验证准确率: {result['best_val_acc']:.4f}")
    print(f"测试准确率: {result['test_acc']:.4f}")
    print("=" * 80)
