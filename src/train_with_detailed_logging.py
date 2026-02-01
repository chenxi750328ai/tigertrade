#!/usr/bin/env python3
"""
带详细迭代日志和错误检测的训练脚本
解决inplace操作错误和其他PyTorch问题
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
import traceback as tb

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import TrainingConfig, FeatureConfig


# 设置PyTorch以检测异常
torch.autograd.set_detect_anomaly(True)


class TradingDataset(Dataset):
    """交易数据集"""
    
    def __init__(self, dataframe, feature_cols, label_col='label'):
        self.features = dataframe[feature_cols].values.astype(np.float32)
        self.labels = dataframe[label_col].values.astype(np.int64)
        
        # 标准化 - 避免inplace操作
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
        # 返回新张量，避免inplace操作
        return (
            torch.tensor(self.features[idx].copy()),
            torch.tensor(self.labels[idx])
        )


class ImprovedTransformer(nn.Module):
    """改进的Transformer模型 - 避免inplace操作"""
    
    def __init__(self, input_dim, hidden_dim=128, num_heads=4, num_layers=2, 
                 num_classes=3, dropout=0.1):
        super(ImprovedTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 输入嵌入
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.embedding_norm = nn.LayerNorm(hidden_dim)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # 使用Pre-LN，更稳定
        )
        # 禁用nested tensor避免警告
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers,
            enable_nested_tensor=False
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        前向传播 - 避免所有inplace操作
        
        Args:
            x: (batch, features)
        
        Returns:
            (batch, num_classes)
        """
        # 嵌入 - 创建新张量
        x = self.embedding(x)
        x = self.embedding_norm(x)
        
        # 添加序列维度
        x = x.unsqueeze(1)  # (batch, 1, hidden)
        
        # Transformer编码
        x = self.transformer(x)  # (batch, 1, hidden)
        
        # 移除序列维度
        x = x.squeeze(1)  # (batch, hidden)
        
        # 分类
        x = self.classifier(x)  # (batch, num_classes)
        
        return x


class DetailedLogger:
    """详细日志记录器"""
    
    def __init__(self, log_dir=None):
        """初始化日志记录器"""
        if log_dir is None:
            log_dir = TrainingConfig.LOG_DIR
        
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(log_dir, f'training_{timestamp}.log')
        self.metrics_file = os.path.join(log_dir, f'metrics_{timestamp}.csv')
        self.error_file = os.path.join(log_dir, f'errors_{timestamp}.log')
        
        # 初始化metrics文件
        with open(self.metrics_file, 'w') as f:
            f.write('epoch,batch,phase,loss,accuracy,lr,grad_norm,time\n')
        
        self.log("=" * 80)
        self.log("训练日志初始化")
        self.log("=" * 80)
        self.log(f"日志文件: {self.log_file}")
        self.log(f"指标文件: {self.metrics_file}")
        self.log(f"错误文件: {self.error_file}")
        self.log("=" * 80)
    
    def log(self, message, level='INFO'):
        """记录日志"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        log_message = f"[{timestamp}] [{level}] {message}"
        print(log_message)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
    
    def log_error(self, error_message, exception=None):
        """记录错误"""
        self.log(f"❌ ERROR: {error_message}", level='ERROR')
        
        error_details = f"\n{'='*80}\n"
        error_details += f"[{datetime.now()}] ERROR\n"
        error_details += f"{'-'*80}\n"
        error_details += f"{error_message}\n"
        
        if exception:
            error_details += f"\n{tb.format_exc()}\n"
        
        error_details += f"{'='*80}\n"
        
        with open(self.error_file, 'a', encoding='utf-8') as f:
            f.write(error_details)
    
    def log_batch(self, epoch, batch, phase, loss, acc, lr, grad_norm, elapsed):
        """记录批次指标"""
        with open(self.metrics_file, 'a') as f:
            f.write(f'{epoch},{batch},{phase},{loss:.6f},{acc:.4f},'
                   f'{lr:.8f},{grad_norm:.6f},{elapsed:.4f}\n')
    
    def log_iteration_details(self, epoch, batch, total_batches, loss, acc, lr, 
                             grad_norm, elapsed):
        """记录详细的迭代信息"""
        message = (f"Epoch {epoch:3d} | Batch {batch:4d}/{total_batches:4d} | "
                  f"Loss: {loss:.6f} | Acc: {acc:.4f} | "
                  f"LR: {lr:.8f} | GradNorm: {grad_norm:.6f} | "
                  f"Time: {elapsed*1000:.2f}ms")
        self.log(message)
    
    def log_gradient_stats(self, model):
        """记录梯度统计信息"""
        total_norm = 0.0
        max_grad = 0.0
        min_grad = float('inf')
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                max_grad = max(max_grad, param.grad.abs().max().item())
                min_grad = min(min_grad, param.grad.abs().min().item())
        
        total_norm = total_norm ** 0.5
        
        self.log(f"  梯度统计: Norm={total_norm:.6f}, Max={max_grad:.6f}, Min={min_grad:.6f}")
        
        return total_norm
    
    def log_model_info(self, model):
        """记录模型信息"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.log("\n" + "=" * 80)
        self.log("模型信息")
        self.log("=" * 80)
        self.log(f"总参数量: {total_params:,}")
        self.log(f"可训练参数: {trainable_params:,}")
        self.log(f"模型架构:\n{model}")
        self.log("=" * 80 + "\n")
    
    def check_tensors(self, batch_data, batch_labels, logger_prefix=""):
        """检查张量是否有问题"""
        issues = []
        
        # 检查NaN
        if torch.isnan(batch_data).any():
            issues.append(f"{logger_prefix}输入数据包含NaN")
        if torch.isnan(batch_labels).any():
            issues.append(f"{logger_prefix}标签数据包含NaN")
        
        # 检查Inf
        if torch.isinf(batch_data).any():
            issues.append(f"{logger_prefix}输入数据包含Inf")
        
        if issues:
            for issue in issues:
                self.log_error(issue)
            return False
        
        return True


def train_epoch(model, dataloader, criterion, optimizer, device, logger, epoch):
    """训练一个epoch - 带详细日志"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (features, labels) in enumerate(dataloader):
        batch_start_time = time.time()
        
        try:
            # 数据移到设备
            features = features.to(device)
            labels = labels.to(device)
            
            # 检查数据
            if TrainingConfig.DEBUG_MODE:
                if not logger.check_tensors(features, labels, f"Batch {batch_idx}: "):
                    logger.log_error(f"Batch {batch_idx} 数据检查失败，跳过")
                    continue
            
            # 前向传播
            optimizer.zero_grad()
            
            try:
                # 为Transformer模型添加seq_len维度
                if len(features.shape) == 2:
                    features = features.unsqueeze(1)  # (batch, features) -> (batch, 1, features)
                outputs = model(features)
            except RuntimeError as e:
                logger.log_error(f"前向传播错误 (Batch {batch_idx}): {str(e)}", e)
                continue
            
            # 计算损失
            try:
                loss = criterion(outputs, labels)
            except RuntimeError as e:
                logger.log_error(f"损失计算错误 (Batch {batch_idx}): {str(e)}", e)
                continue
            
            # 检查损失
            if torch.isnan(loss) or torch.isinf(loss):
                logger.log_error(f"Batch {batch_idx}: 损失为NaN或Inf: {loss.item()}")
                continue
            
            # 反向传播
            try:
                loss.backward()
            except RuntimeError as e:
                logger.log_error(f"反向传播错误 (Batch {batch_idx}): {str(e)}", e)
                logger.log_error("这通常是由inplace操作引起的", e)
                continue
            
            # 梯度裁剪
            if TrainingConfig.GRAD_CLIP > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    TrainingConfig.GRAD_CLIP
                )
            else:
                grad_norm = 0.0
            
            # 检查梯度
            if TrainingConfig.CHECK_GRADIENTS and batch_idx % 50 == 0:
                grad_norm_detailed = logger.log_gradient_stats(model)
            
            # 优化步骤
            try:
                optimizer.step()
            except RuntimeError as e:
                logger.log_error(f"优化器步骤错误 (Batch {batch_idx}): {str(e)}", e)
                continue
            
            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            # 批次耗时
            batch_time = time.time() - batch_start_time
            
            # 记录批次指标
            batch_acc = 100. * predicted.eq(labels).sum().item() / labels.size(0)
            current_lr = optimizer.param_groups[0]['lr']
            logger.log_batch(epoch, batch_idx, 'train', loss.item(), 
                           batch_acc/100, current_lr, grad_norm, batch_time)
            
            # 定期打印详细信息
            if (batch_idx + 1) % TrainingConfig.LOG_INTERVAL == 0:
                logger.log_iteration_details(
                    epoch, batch_idx + 1, len(dataloader),
                    loss.item(), batch_acc/100, current_lr,
                    grad_norm, batch_time
                )
        
        except Exception as e:
            logger.log_error(f"Batch {batch_idx} 处理异常", e)
            continue
    
    if total > 0:
        return total_loss / len(dataloader), correct / total
    else:
        return 0.0, 0.0


def validate(model, dataloader, criterion, device, logger, epoch):
    """验证模型 - 带详细日志"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (features, labels) in enumerate(dataloader):
            batch_start_time = time.time()
            
            try:
                features, labels = features.to(device), labels.to(device)
                
                # 为Transformer模型添加seq_len维度
                if len(features.shape) == 2:
                    features = features.unsqueeze(1)  # (batch, features) -> (batch, 1, features)
                    
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
                
                batch_time = time.time() - batch_start_time
                batch_acc = 100. * predicted.eq(labels).sum().item() / labels.size(0)
                
                logger.log_batch(epoch, batch_idx, 'val', loss.item(),
                               batch_acc/100, 0.0, 0.0, batch_time)
            
            except Exception as e:
                logger.log_error(f"验证Batch {batch_idx} 处理异常", e)
                continue
    
    if total > 0:
        return total_loss / len(dataloader), correct / total
    else:
        return 0.0, 0.0


def main():
    """主训练流程"""
    parser = argparse.ArgumentParser(description='带详细日志的模型训练')
    parser.add_argument('--train-file', type=str, required=True, help='训练数据文件')
    parser.add_argument('--val-file', type=str, required=True, help='验证数据文件')
    parser.add_argument('--config-file', type=str, help='配置文件（可选）')
    
    args = parser.parse_args()
    
    # 初始化日志
    logger = DetailedLogger()
    logger.log("=" * 80)
    logger.log("🚀 开始训练流程（详细日志模式）")
    logger.log("=" * 80)
    
    # 打印配置
    TrainingConfig.print_config()
    FeatureConfig.print_config()
    
    try:
        # 1. 加载数据
        logger.log("\n步骤 1: 加载数据")
        logger.log(f"  训练数据: {args.train_file}")
        logger.log(f"  验证数据: {args.val_file}")
        
        train_df = pd.read_csv(args.train_file, index_col=0)
        val_df = pd.read_csv(args.val_file, index_col=0)
        
        logger.log(f"  训练集大小: {len(train_df)}")
        logger.log(f"  验证集大小: {len(val_df)}")
        
        # 2. 准备特征
        logger.log("\n步骤 2: 准备特征")
        feature_cols = FeatureConfig.get_selected_features()
        logger.log(f"  使用 {len(feature_cols)} 个特征")
        
        # 检查特征是否存在
        missing_features = [f for f in feature_cols if f not in train_df.columns]
        if missing_features:
            logger.log_error(f"缺少特征: {missing_features}")
            return
        
        # 3. 创建数据集
        logger.log("\n步骤 3: 创建数据加载器")
        train_dataset = TradingDataset(train_df, feature_cols, 'label')
        val_dataset = TradingDataset(val_df, feature_cols, 'label')
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=TrainingConfig.BATCH_SIZE,
            shuffle=True,
            num_workers=0,  # 避免多进程问题
            pin_memory=True if TrainingConfig.DEVICE == 'cuda' else False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=TrainingConfig.BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=True if TrainingConfig.DEVICE == 'cuda' else False
        )
        
        logger.log(f"  训练批次数: {len(train_loader)}")
        logger.log(f"  验证批次数: {len(val_loader)}")
        
        # 4. 创建模型
        logger.log("\n步骤 4: 创建模型")
        device = torch.device(TrainingConfig.DEVICE if torch.cuda.is_available() else 'cpu')
        logger.log(f"  使用设备: {device}")
        
        if device.type == 'cuda':
            logger.log(f"  GPU: {torch.cuda.get_device_name(0)}")
            logger.log(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        model = ImprovedTransformer(
            input_dim=len(feature_cols),
            hidden_dim=TrainingConfig.HIDDEN_DIM,
            num_heads=TrainingConfig.NUM_HEADS,
            num_layers=TrainingConfig.NUM_LAYERS,
            num_classes=3,
            dropout=TrainingConfig.DROPOUT
        ).to(device)
        
        logger.log_model_info(model)
        
        # 5. 设置训练参数
        logger.log("步骤 5: 设置训练参数")
        
        # 计算类别权重
        label_counts = train_df['label'].value_counts().sort_index()
        total = len(train_df)
        weights = torch.tensor(
            [total / (len(label_counts) * count) for count in label_counts],
            dtype=torch.float32
        ).to(device)
        logger.log(f"  类别权重: {weights.cpu().numpy()}")
        
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = optim.Adam(model.parameters(), lr=TrainingConfig.LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',
            patience=TrainingConfig.LR_PATIENCE,
            factor=TrainingConfig.LR_FACTOR,
            verbose=True
        )
        
        best_val_acc = 0
        best_model_path = os.path.join(TrainingConfig.MODEL_DIR, 'best_model.pth')
        os.makedirs(TrainingConfig.MODEL_DIR, exist_ok=True)
        
        patience_counter = 0
        
        # 6. 训练循环
        logger.log("\n" + "=" * 80)
        logger.log("步骤 6: 开始训练")
        logger.log("=" * 80 + "\n")
        
        for epoch in range(1, TrainingConfig.NUM_EPOCHS + 1):
            epoch_start_time = time.time()
            
            logger.log(f"\nEpoch {epoch}/{TrainingConfig.NUM_EPOCHS}")
            logger.log("-" * 80)
            
            # 训练
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device, logger, epoch
            )
            
            # 验证
            val_loss, val_acc = validate(
                model, val_loader, criterion, device, logger, epoch
            )
            
            # 学习率调整
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # 计算耗时
            elapsed_time = time.time() - epoch_start_time
            
            # 汇总日志
            logger.log(f"\nEpoch {epoch} 总结:")
            logger.log(f"  训练 - Loss: {train_loss:.6f}, Acc: {train_acc:.4f}")
            logger.log(f"  验证 - Loss: {val_loss:.6f}, Acc: {val_acc:.4f}")
            logger.log(f"  学习率: {current_lr:.8f}")
            logger.log(f"  耗时: {elapsed_time:.2f}s")
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                }, best_model_path)
                logger.log(f"  🏆 新的最佳验证准确率: {val_acc:.4f}, 模型已保存")
            else:
                patience_counter += 1
                logger.log(f"  耐心值: {patience_counter}/{TrainingConfig.EARLY_STOP_PATIENCE}")
            
            # 早停
            if patience_counter >= TrainingConfig.EARLY_STOP_PATIENCE:
                logger.log(f"\n⏹️ 早停触发，停止训练")
                break
            
            # 定期保存
            if epoch % TrainingConfig.SAVE_INTERVAL == 0:
                checkpoint_path = os.path.join(
                    TrainingConfig.MODEL_DIR, 
                    f'checkpoint_epoch_{epoch}.pth'
                )
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                }, checkpoint_path)
                logger.log(f"  💾 Checkpoint已保存: {checkpoint_path}")
        
        # 7. 训练完成
        logger.log("\n" + "=" * 80)
        logger.log("✅ 训练完成！")
        logger.log("=" * 80)
        logger.log(f"\n最佳验证准确率: {best_val_acc:.4f}")
        logger.log(f"最佳模型: {best_model_path}")
        logger.log(f"\n日志文件:")
        logger.log(f"  - 训练日志: {logger.log_file}")
        logger.log(f"  - 指标CSV: {logger.metrics_file}")
        logger.log(f"  - 错误日志: {logger.error_file}")
    
    except Exception as e:
        logger.log_error("训练过程出现异常", e)
        raise


if __name__ == "__main__":
    main()
