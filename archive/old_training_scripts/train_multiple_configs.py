#!/usr/bin/env python3
"""
使用不同配置训练多个Transformer模型
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
from train_with_detailed_logging import ImprovedTransformer, TradingDataset, DetailedLogger, train_epoch, validate


def get_model_configs():
    """定义不同的模型配置"""
    configs = {
        '小型模型 (128维)': {
            'hidden_dim': 128,
            'num_heads': 4,
            'num_layers': 2,
            'dropout': 0.1
        },
        '中型模型 (256维)': {
            'hidden_dim': 256,
            'num_heads': 8,
            'num_layers': 3,
            'dropout': 0.1
        },
        '大型模型 (512维)': {
            'hidden_dim': 512,
            'num_heads': 8,
            'num_layers': 4,
            'dropout': 0.2
        },
        '深层模型 (128维-6层)': {
            'hidden_dim': 128,
            'num_heads': 4,
            'num_layers': 6,
            'dropout': 0.15
        },
        '宽层模型 (384维-2层)': {
            'hidden_dim': 384,
            'num_heads': 6,
            'num_layers': 2,
            'dropout': 0.1
        },
        '超大模型 (768维)': {
            'hidden_dim': 768,
            'num_heads': 8,
            'num_layers': 4,
            'dropout': 0.2
        },
    }
    return configs


def train_model_with_config(config_name, config, train_loader, val_loader, device, logger, output_dir, input_dim):
    """使用指定配置训练模型"""
    logger.log(f"\n{'='*80}")
    logger.log(f"🚀 开始训练: {config_name}")
    logger.log(f"{'='*80}")
    logger.log(f"配置: {config}")
    
    try:
        # 创建模型
        model = ImprovedTransformer(
            input_dim=input_dim,
            hidden_dim=config['hidden_dim'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            num_classes=3
        ).to(device)
        
        # 统计参数数量
        total_params = sum(p.numel() for p in model.parameters())
        logger.log(f"模型参数数量: {total_params:,}")
        
        # 设置优化器和损失函数
        optimizer = optim.Adam(model.parameters(), lr=TrainingConfig.LEARNING_RATE)
        
        # 计算类别权重
        label_counts = np.bincount([label for _, label in train_loader.dataset])
        class_weights = torch.FloatTensor([1.0 / count if count > 0 else 1.0 for count in label_counts])
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        class_weights = class_weights.to(device)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=TrainingConfig.PATIENCE // 2
        )
        
        # 训练
        best_val_acc = 0
        patience_counter = 0
        results = {
            'config_name': config_name,
            'config': config,
            'total_params': total_params,
            'epochs': [],
            'best_val_acc': 0,
            'best_epoch': 0,
            'total_time': 0
        }
        
        start_time = time.time()
        
        for epoch in range(1, min(TrainingConfig.MAX_EPOCHS, 30) + 1):  # 最多30轮
            logger.log(f"\nEpoch {epoch}")
            logger.log("-" * 80)
            
            # 训练
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device, logger, epoch
            )
            
            # 验证
            val_loss, val_acc = validate(model, val_loader, criterion, device, logger, epoch)
            
            # 学习率调度
            scheduler.step(val_acc)
            current_lr = optimizer.param_groups[0]['lr']
            
            # 记录结果
            epoch_result = {
                'epoch': epoch,
                'train_loss': float(train_loss),
                'train_acc': float(train_acc),
                'val_loss': float(val_loss),
                'val_acc': float(val_acc),
                'lr': float(current_lr)
            }
            results['epochs'].append(epoch_result)
            
            logger.log(f"Epoch {epoch:2d} - Train: Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
                      f"Val: Loss={val_loss:.4f}, Acc={val_acc:.4f} | LR={current_lr:.6f}")
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                results['best_val_acc'] = float(best_val_acc)
                results['best_epoch'] = epoch
                
                # 保存模型
                safe_name = config_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '-')
                model_path = os.path.join(output_dir, f'{safe_name}_best.pth')
                torch.save(model.state_dict(), model_path)
                logger.log(f"  🏆 新的最佳准确率: {best_val_acc:.4f}, 模型已保存")
                patience_counter = 0
            else:
                patience_counter += 1
                
            # 早停
            if patience_counter >= TrainingConfig.PATIENCE:
                logger.log(f"  ⏹️ 早停触发")
                break
        
        total_time = time.time() - start_time
        results['total_time'] = float(total_time)
        
        logger.log(f"\n✅ {config_name} 训练完成!")
        logger.log(f"  最佳验证准确率: {best_val_acc:.4f} (Epoch {results['best_epoch']})")
        logger.log(f"  训练耗时: {total_time:.2f}秒")
        logger.log(f"  参数数量: {total_params:,}")
        
        return results
        
    except Exception as e:
        logger.log_error(f"❌ {config_name} 训练失败: {e}", e)
        import traceback
        traceback.print_exc()
        return None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练多个配置的模型')
    parser.add_argument('--train-file', type=str, required=True, help='训练数据文件')
    parser.add_argument('--val-file', type=str, required=True, help='验证数据文件')
    parser.add_argument('--output-dir', type=str, default='/home/cx/trading_data/model_comparison', help='输出目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志
    logger = DetailedLogger(log_dir)
    
    logger.log("="*80)
    logger.log("🚀 开始训练多个配置的模型")
    logger.log("="*80)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and TrainingConfig.DEVICE == 'cuda' else 'cpu')
    logger.log(f"使用设备: {device}")
    
    # 加载数据
    logger.log(f"\n加载数据...")
    logger.log(f"训练集: {args.train_file}")
    logger.log(f"验证集: {args.val_file}")
    
    train_df = pd.read_csv(args.train_file, index_col=0)
    val_df = pd.read_csv(args.val_file, index_col=0)
    
    logger.log(f"训练集大小: {len(train_df)}")
    logger.log(f"验证集大小: {len(val_df)}")
    
    # 准备数据集
    feature_cols = FeatureConfig.get_all_features()
    input_dim = len(feature_cols)
    logger.log(f"特征数量: {input_dim}")
    
    train_dataset = TradingDataset(train_df, feature_cols)
    val_dataset = TradingDataset(val_df, feature_cols)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=TrainingConfig.BATCH_SIZE,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=TrainingConfig.BATCH_SIZE,
        shuffle=False
    )
    
    # 获取所有配置
    all_configs = get_model_configs()
    logger.log(f"\n找到 {len(all_configs)} 个模型配置:")
    for name in all_configs.keys():
        logger.log(f"  - {name}")
    
    # 训练所有配置
    all_results = []
    
    for config_name, config in all_configs.items():
        result = train_model_with_config(
            config_name, 
            config, 
            train_loader, 
            val_loader, 
            device, 
            logger,
            args.output_dir,
            input_dim
        )
        
        if result:
            all_results.append(result)
    
    # 保存所有结果
    results_file = os.path.join(args.output_dir, 'model_comparison_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    logger.log(f"\n{'='*80}")
    logger.log("📊 所有模型训练完成！")
    logger.log(f"{'='*80}")
    
    # 排序并显示结果
    all_results.sort(key=lambda x: x['best_val_acc'], reverse=True)
    
    logger.log(f"\n排名结果 (按验证准确率):")
    logger.log("-" * 80)
    for i, result in enumerate(all_results, 1):
        logger.log(f"{i}. {result['config_name']}: "
                  f"{result['best_val_acc']:.4f} (Epoch {result['best_epoch']}, "
                  f"{result['total_params']:,} 参数, {result['total_time']:.1f}s)")
    
    logger.log(f"\n结果已保存到: {results_file}")
    
    # 生成Markdown报告
    md_file = os.path.join(args.output_dir, 'comparison_report.md')
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write("# 模型配置对比报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## 排名结果\n\n")
        f.write("| 排名 | 模型配置 | 验证准确率 | 最佳Epoch | 参数数量 | 训练时间(秒) |\n")
        f.write("|------|---------|-----------|----------|----------|-------------|\n")
        for i, result in enumerate(all_results, 1):
            f.write(f"| {i} | {result['config_name']} | {result['best_val_acc']:.4f} | "
                   f"{result['best_epoch']} | {result['total_params']:,} | {result['total_time']:.1f} |\n")
        
        f.write("\n## 详细配置\n\n")
        for result in all_results:
            f.write(f"### {result['config_name']}\n\n")
            f.write(f"- **验证准确率**: {result['best_val_acc']:.4f}\n")
            f.write(f"- **最佳Epoch**: {result['best_epoch']}\n")
            f.write(f"- **参数数量**: {result['total_params']:,}\n")
            f.write(f"- **训练时间**: {result['total_time']:.1f}秒\n")
            f.write(f"- **配置**:\n")
            for k, v in result['config'].items():
                f.write(f"  - {k}: {v}\n")
            f.write("\n")
    
    logger.log(f"Markdown报告已保存到: {md_file}")


if __name__ == "__main__":
    main()
