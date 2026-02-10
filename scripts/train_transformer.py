#!/usr/bin/env python3
"""
TigerTrade - Transformer模型训练
目标：预测未来收益率，实现月盈利20%
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from datetime import datetime
import sys

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TimeSeriesDataset(Dataset):
    """时间序列数据集"""
    
    def __init__(self, data_path, sequence_length=20):
        self.df = pd.read_csv(data_path)
        self.sequence_length = sequence_length
        
        # 准备特征和目标
        self.feature_cols = [c for c in self.df.columns 
                            if c not in ['timestamp', 'Unnamed: 0'] 
                            and not c.startswith('target_')]
        
        self.target_col = 'target_return_5'  # 预测5期后的收益
        
        # 归一化特征
        self.features = self.df[self.feature_cols].fillna(0).astype(np.float32).values
        self.targets = self.df[self.target_col].fillna(0).astype(np.float32).values
        
        # 标准化
        self.feature_mean = np.mean(self.features, axis=0)
        self.feature_std = np.std(self.features, axis=0) + 1e-8
        self.features = (self.features - self.feature_mean) / self.feature_std
        
        print(f"数据加载: {len(self.df)} 条记录")
        print(f"特征数量: {len(self.feature_cols)}")
        print(f"序列长度: {sequence_length}")
    
    def __len__(self):
        return len(self.df) - self.sequence_length
    
    def __getitem__(self, idx):
        # 获取序列
        x = self.features[idx:idx+self.sequence_length]
        y = self.targets[idx+self.sequence_length]
        
        return torch.FloatTensor(x), torch.FloatTensor([y])


class TransformerModel(nn.Module):
    """Transformer预测模型"""
    
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=3, dropout=0.1):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//2, 1)
        )
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x[:, -1, :]  # 取最后一个时间步
        x = self.fc(x)
        return x


def train_epoch(model, dataloader, optimizer, criterion):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion):
    """评估模型"""
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item()
            predictions.extend(outputs.cpu().numpy())
            targets.extend(batch_y.cpu().numpy())
    
    predictions = np.array(predictions).flatten()
    targets = np.array(targets).flatten()
    
    # 计算方向准确率（涨跌预测准确率）
    direction_acc = ((predictions > 0) == (targets > 0)).mean()
    
    return total_loss / len(dataloader), direction_acc, predictions, targets


def main():
    """主函数"""
    
    print("="*70)
    print("🤖 TigerTrade Transformer模型训练")
    print("="*70)
    print(f"目标: 预测未来收益率，实现月盈利20%")
    print("="*70)
    
    # 配置
    DATA_DIR = Path("/home/cx/tigertrade/data/processed")
    OUTPUT_DIR = Path("/home/cx/tigertrade/models")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    SEQUENCE_LENGTH = 20
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    
    print(f"\n📁 数据目录: {DATA_DIR}")
    print(f"💾 模型输出: {OUTPUT_DIR}")
    print(f"\n超参数:")
    print(f"  序列长度: {SEQUENCE_LENGTH}")
    print(f"  批次大小: {BATCH_SIZE}")
    print(f"  训练轮数: {EPOCHS}")
    print(f"  学习率: {LEARNING_RATE}")
    
    # 加载数据
    print(f"\n📂 加载数据...")
    train_dataset = TimeSeriesDataset(
        DATA_DIR / "train.csv",
        sequence_length=SEQUENCE_LENGTH
    )
    val_dataset = TimeSeriesDataset(
        DATA_DIR / "val.csv",
        sequence_length=SEQUENCE_LENGTH
    )
    test_dataset = TimeSeriesDataset(
        DATA_DIR / "test.csv",
        sequence_length=SEQUENCE_LENGTH
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"✅ 训练集: {len(train_dataset)} 样本")
    print(f"✅ 验证集: {len(val_dataset)} 样本")
    print(f"✅ 测试集: {len(test_dataset)} 样本")
    
    # 创建模型
    print(f"\n🔧 创建Transformer模型...")
    input_dim = len(train_dataset.feature_cols)
    model = TransformerModel(
        input_dim=input_dim,
        d_model=128,
        nhead=4,
        num_layers=3,
        dropout=0.1
    ).to(device)
    
    print(f"✅ 模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 优化器和损失
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # 训练
    print(f"\n🚀 开始训练...")
    print("="*70)
    
    best_val_loss = float('inf')
    patience_counter = 0
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'val_direction_acc': []
    }
    
    for epoch in range(EPOCHS):
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        
        # 验证
        val_loss, val_dir_acc, _, _ = evaluate(model, val_loader, criterion)
        
        # 学习率调整
        scheduler.step(val_loss)
        
        # 记录
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['val_direction_acc'].append(val_dir_acc)
        
        # 打印进度
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
              f"训练损失: {train_loss:.6f} | "
              f"验证损失: {val_loss:.6f} | "
              f"方向准确率: {val_dir_acc:.2%}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            model_path = OUTPUT_DIR / "transformer_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_direction_acc': val_dir_acc,
                'feature_cols': train_dataset.feature_cols,
                'feature_mean': train_dataset.feature_mean,
                'feature_std': train_dataset.feature_std
            }, model_path)
            
            print(f"  💾 保存最佳模型 (验证损失: {val_loss:.6f})")
        else:
            patience_counter += 1
        
        # 早停
        if patience_counter >= 10:
            print(f"\n⚠️  早停：验证损失10轮未改善")
            break
    
    # 测试集评估
    print(f"\n📊 在测试集上评估...")
    test_loss, test_dir_acc, test_preds, test_targets = evaluate(
        model, test_loader, criterion
    )
    
    print(f"✅ 测试损失: {test_loss:.6f}")
    print(f"✅ 方向准确率: {test_dir_acc:.2%}")
    
    # 计算盈利指标
    print(f"\n💰 盈利能力评估...")
    
    # 简单策略：预测为正就做多
    strategy_returns = []
    for pred, actual in zip(test_preds, test_targets):
        if pred > 0:  # 预测上涨，做多
            strategy_returns.append(actual)
        else:  # 预测下跌，不交易或做空
            strategy_returns.append(-actual)
    
    strategy_returns = np.array(strategy_returns)
    
    # 计算收益
    total_return = strategy_returns.sum()
    mean_return = strategy_returns.mean()
    sharpe = strategy_returns.mean() / (strategy_returns.std() + 1e-8) * np.sqrt(252)
    max_drawdown = (np.maximum.accumulate(np.cumsum(strategy_returns)) - 
                    np.cumsum(strategy_returns)).max()
    
    win_rate = (strategy_returns > 0).mean()
    
    print(f"  总收益率: {total_return:.2%}")
    print(f"  平均收益率: {mean_return:.4%}")
    print(f"  夏普比率: {sharpe:.2f}")
    print(f"  最大回撤: {max_drawdown:.2%}")
    print(f"  胜率: {win_rate:.2%}")
    
    # 月度收益估算
    periods_per_month = 252 / 12 / 5  # 假设5期=1天
    monthly_return = mean_return * periods_per_month
    print(f"\n  📈 预估月收益率: {monthly_return:.2%}")
    
    if monthly_return >= 0.20:
        print(f"  ✅ 达到目标：月盈利率 >= 20%！")
    else:
        print(f"  ⚠️  未达标：需要优化")
        print(f"     差距: {(0.20 - monthly_return):.2%}")
    
    # 保存结果
    results = {
        'timestamp': datetime.now().isoformat(),
        'model': 'Transformer',
        'test_loss': float(test_loss),
        'direction_accuracy': float(test_dir_acc),
        'total_return': float(total_return),
        'mean_return': float(mean_return),
        'sharpe_ratio': float(sharpe),
        'max_drawdown': float(max_drawdown),
        'win_rate': float(win_rate),
        'estimated_monthly_return': float(monthly_return),
        'target_achieved': bool(monthly_return >= 0.20),
        'training_history': {
            k: [float(v) for v in vals] 
            for k, vals in training_history.items()
        }
    }
    
    result_path = OUTPUT_DIR / "training_results.json"
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 结果已保存: {result_path}")
    
    print("\n" + "="*70)
    print("🎉 训练完成！")
    print("="*70)
    print(f"\n下一步：策略回测和风险管理")
    
    return results


if __name__ == "__main__":
    try:
        results = main()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
