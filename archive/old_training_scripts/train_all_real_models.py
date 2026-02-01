#!/usr/bin/env python3
"""
训练所有真实模型（LSTM、Transformer、双对比）
"""

import sys
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from datetime import datetime
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import TrainingConfig, FeatureConfig

# 导入所有模型类（不是策略类）
from strategies.llm_strategy import TradingLSTM
from strategies.large_model_strategy import LargeTradingNetwork  
from strategies.huge_transformer_strategy import HugeTransformer
from strategies.enhanced_transformer_strategy import EnhancedTradingTransformer
from strategies.rl_trading_strategy import RLTradingNetwork
from strategies.large_transformer_strategy import LargeTradingTransformer
from strategies.model_comparison_strategy import TradingLSTM as ComparisonLSTM, TradingTransformer as ComparisonTransformer


class TradingDataset(Dataset):
    """交易数据集"""
    
    def __init__(self, dataframe, feature_cols, label_col='label'):
        self.features = dataframe[feature_cols].values.astype(np.float32)
        self.labels = dataframe[label_col].values.astype(np.int64)
        
        # 标准化
        self.mean = self.features.mean(axis=0)
        self.std = self.features.std(axis=0) + 1e-8
        self.features = (self.features - self.mean) / self.std
        
        # 检查NaN
        if np.isnan(self.features).any():
            print("⚠️ 警告：特征中存在NaN值，已替换为0")
            self.features = np.nan_to_num(self.features)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx].copy(), dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_data, batch_labels in dataloader:
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)
        
        # 某些模型需要3D输入（batch, seq, features）
        if len(batch_data.shape) == 2:
            batch_data = batch_data.unsqueeze(1)  # 添加序列维度
        
        optimizer.zero_grad()
        outputs = model(batch_data)
        
        # 某些模型可能返回多个值
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        # 确保输出维度正确
        if len(outputs.shape) == 3:
            outputs = outputs.squeeze(1)
        
        loss = criterion(outputs, batch_labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += batch_labels.size(0)
        correct += (predicted == batch_labels).sum().item()
    
    return total_loss / len(dataloader), correct / total


def validate(model, dataloader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_data, batch_labels in dataloader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            # 某些模型需要3D输入
            if len(batch_data.shape) == 2:
                batch_data = batch_data.unsqueeze(1)
            
            outputs = model(batch_data)
            
            # 某些模型可能返回多个值
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            # 确保输出维度正确
            if len(outputs.shape) == 3:
                outputs = outputs.squeeze(1)
            
            loss = criterion(outputs, batch_labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
    
    return total_loss / len(dataloader), correct / total


def get_all_models(input_dim):
    """获取所有模型配置"""
    models = {
        'LSTM模型 (LLM策略)': lambda: TradingLSTM(input_size=input_dim, hidden_size=64, num_layers=2, output_size=3),
        '大型LSTM模型': lambda: LargeTradingNetwork(input_size=input_dim, hidden_size=256, num_layers=4, output_size=3),
        '强化学习网络 (LSTM)': lambda: RLTradingNetwork(input_size=input_dim, action_size=3, hidden_size=512, num_layers=4),
        '大型Transformer (256维-6层)': lambda: LargeTradingTransformer(input_size=input_dim, nhead=8, num_layers=6, output_size=3, d_model=256),
        '超大Transformer (512维-8层)': lambda: HugeTransformer(input_size=input_dim, d_model=512, nhead=8, num_layers=8, output_size=3),
        '增强型Transformer (512维-8层+注意力池化)': lambda: EnhancedTradingTransformer(input_size=input_dim, nhead=8, num_layers=8, output_size=3, d_model=512),
        '对比模型-LSTM': lambda: ComparisonLSTM(input_size=input_dim, hidden_size=64, num_layers=2, output_size=3),
        '对比模型-Transformer': lambda: ComparisonTransformer(input_size=input_dim, nhead=2, num_layers=2, output_size=3, d_model=64),
    }
    return models


def train_single_model(model_name, model_fn, train_loader, val_loader, device, output_dir):
    """训练单个模型"""
    print(f"\n{'='*80}")
    print(f"🚀 开始训练: {model_name}")
    print(f"{'='*80}")
    
    try:
        model = model_fn().to(device)
        
        # 统计参数
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型参数数量: {total_params:,}")
        
        # 优化器和损失
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # 计算类别权重
        label_counts = np.bincount([label for _, label in train_loader.dataset])
        class_weights = torch.FloatTensor([1.0 / count if count > 0 else 1.0 for count in label_counts])
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        
        # 训练
        best_val_acc = 0
        patience_counter = 0
        results = {
            'model_name': model_name,
            'total_params': total_params,
            'epochs': [],
            'best_val_acc': 0,
            'best_epoch': 0,
            'total_time': 0
        }
        
        start_time = time.time()
        
        for epoch in range(1, 31):  # 最多30轮
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            
            scheduler.step(val_acc)
            current_lr = optimizer.param_groups[0]['lr']
            
            results['epochs'].append({
                'epoch': epoch,
                'train_loss': float(train_loss),
                'train_acc': float(train_acc),
                'val_loss': float(val_loss),
                'val_acc': float(val_acc),
                'lr': float(current_lr)
            })
            
            print(f"Epoch {epoch:2d} - Train: Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
                  f"Val: Loss={val_loss:.4f}, Acc={val_acc:.4f} | LR={current_lr:.6f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                results['best_val_acc'] = float(best_val_acc)
                results['best_epoch'] = epoch
                
                # 保存模型
                safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '-').replace('+', '_')
                model_path = os.path.join(output_dir, f'{safe_name}_best.pth')
                torch.save(model.state_dict(), model_path)
                print(f"  🏆 新的最佳准确率: {best_val_acc:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= 10:
                print(f"  ⏹️ 早停触发")
                break
        
        total_time = time.time() - start_time
        results['total_time'] = float(total_time)
        
        print(f"\n✅ {model_name} 训练完成!")
        print(f"  最佳验证准确率: {best_val_acc:.4f} (Epoch {results['best_epoch']})")
        print(f"  训练耗时: {total_time:.1f}秒")
        
        return results
        
    except Exception as e:
        print(f"❌ {model_name} 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='训练所有真实模型')
    parser.add_argument('--train-file', type=str, required=True)
    parser.add_argument('--val-file', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='/home/cx/trading_data/all_real_models')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("🚀 开始训练所有真实模型 (LSTM + Transformer + 双对比)")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    print(f"\n加载数据...")
    train_df = pd.read_csv(args.train_file, index_col=0)
    val_df = pd.read_csv(args.val_file, index_col=0)
    print(f"训练集: {len(train_df)}, 验证集: {len(val_df)}")
    
    # 准备数据
    feature_cols = FeatureConfig.get_all_features()
    input_dim = len(feature_cols)
    print(f"特征数量: {input_dim}")
    
    train_dataset = TradingDataset(train_df, feature_cols)
    val_dataset = TradingDataset(val_df, feature_cols)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 获取所有模型
    all_models = get_all_models(input_dim)
    print(f"\n找到 {len(all_models)} 个模型:")
    for i, name in enumerate(all_models.keys(), 1):
        print(f"  {i}. {name}")
    
    # 训练所有模型
    all_results = []
    
    for model_name, model_fn in all_models.items():
        result = train_single_model(model_name, model_fn, train_loader, val_loader, device, args.output_dir)
        if result:
            all_results.append(result)
    
    # 保存结果
    results_file = os.path.join(args.output_dir, 'all_models_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print("📊 所有模型训练完成！")
    print(f"{'='*80}")
    
    # 排序显示
    all_results.sort(key=lambda x: x['best_val_acc'], reverse=True)
    
    print(f"\n排名结果:")
    print("-" * 80)
    for i, result in enumerate(all_results, 1):
        print(f"{i}. {result['model_name']}: {result['best_val_acc']:.4f} "
              f"(Epoch {result['best_epoch']}, {result['total_params']:,} 参数, {result['total_time']:.1f}s)")
    
    print(f"\n结果已保存到: {results_file}")


if __name__ == "__main__":
    main()
