#!/usr/bin/env python3
"""
训练所有模型并比较结果
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

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import TrainingConfig, FeatureConfig
from train_with_detailed_logging import TradingDataset, DetailedLogger, train_epoch, validate

# 导入所有策略模型
from strategies.llm_strategy import LLMTradingStrategy
from strategies.large_model_strategy import LargeModelStrategy
from strategies.huge_transformer_strategy import HugeTransformerStrategy
from strategies.enhanced_transformer_strategy import EnhancedTransformerStrategy
from strategies.rl_trading_strategy import RLTradingStrategy
from strategies.large_transformer_strategy import LargeTransformerStrategy
from strategies import model_comparison_strategy


def get_all_models():
    """获取所有可训练的模型"""
    models = {
        'LLM策略': LLMTradingStrategy,
        '大模型策略': LargeModelStrategy,
        '超大Transformer策略': HugeTransformerStrategy,
        '增强型Transformer策略': EnhancedTransformerStrategy,
        '强化学习策略': RLTradingStrategy,
        '大型Transformer策略': LargeTransformerStrategy,
        '模型对比策略': model_comparison_strategy.ModelComparisonStrategy,
    }
    return models


def train_single_model(model_name, model_class, train_loader, val_loader, device, logger, output_dir):
    """训练单个模型"""
    logger.log(f"\n{'='*80}")
    logger.log(f"🚀 开始训练: {model_name}")
    logger.log(f"{'='*80}")
    
    try:
        # 获取正确的特征数量
        num_features = len(FeatureConfig.get_all_features())
        
        # 创建策略实例（传递正确的input_size）
        try:
            strategy = model_class(input_size=num_features)
        except TypeError:
            # 如果策略类不接受input_size参数，使用默认构造
            strategy = model_class()
        
        # 检查是否是强化学习策略（需要特殊训练逻辑）
        if model_name == '强化学习策略':
            logger.log(f"⚠️ {model_name} 需要特殊的强化学习训练流程，当前跳过")
            logger.log(f"💡 提示：强化学习策略需要环境交互训练，不适合监督学习流程")
            return None
        
        # 获取内部的PyTorch模型
        if hasattr(strategy, 'model'):
            model = strategy.model
            model = model.to(device)
        elif hasattr(strategy, 'network'):
            # 某些策略使用network属性
            model = strategy.network
            model = model.to(device)
        elif hasattr(strategy, 'lstm_model'):
            # ModelComparisonStrategy使用lstm_model作为主模型
            model = strategy.lstm_model
            model = model.to(device)
        elif hasattr(strategy, 'to'):
            # 如果策略本身就是模型
            model = strategy
            model = model.to(device)
        else:
            # 策略不是标准模型
            logger.log(f"⚠️ {model_name} 没有可训练的模型，跳过标准训练流程")
            return None
        
        # 设置优化器和损失函数
        optimizer = optim.Adam(model.parameters(), lr=TrainingConfig.LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=TrainingConfig.PATIENCE // 2
        )
        
        # 训练
        best_val_acc = 0
        patience_counter = 0
        results = {
            'model_name': model_name,
            'epochs': [],
            'best_val_acc': 0,
            'best_epoch': 0,
            'total_time': 0
        }
        
        start_time = time.time()
        
        for epoch in range(1, TrainingConfig.MAX_EPOCHS + 1):
            logger.log(f"\nEpoch {epoch}/{TrainingConfig.MAX_EPOCHS}")
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
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'lr': current_lr
            }
            results['epochs'].append(epoch_result)
            
            logger.log(f"Epoch {epoch} - Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
            logger.log(f"Epoch {epoch} - Val: Loss={val_loss:.4f}, Acc={val_acc:.4f}")
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                results['best_val_acc'] = best_val_acc
                results['best_epoch'] = epoch
                
                model_path = os.path.join(output_dir, f'{model_name}_best.pth')
                torch.save(model.state_dict(), model_path)
                logger.log(f"🏆 新的最佳准确率: {best_val_acc:.4f}, 模型已保存")
                patience_counter = 0
            else:
                patience_counter += 1
                
            # 早停
            if patience_counter >= TrainingConfig.PATIENCE:
                logger.log(f"⏹️ 早停触发，停止训练")
                break
        
        total_time = time.time() - start_time
        results['total_time'] = total_time
        
        logger.log(f"\n✅ {model_name} 训练完成!")
        logger.log(f"最佳验证准确率: {best_val_acc:.4f} (Epoch {results['best_epoch']})")
        logger.log(f"训练耗时: {total_time:.2f}秒")
        
        return results
        
    except Exception as e:
        logger.log_error(f"❌ {model_name} 训练失败: {e}", e)
        import traceback
        traceback.print_exc()
        return None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练所有模型')
    parser.add_argument('--train-file', type=str, required=True, help='训练数据文件')
    parser.add_argument('--val-file', type=str, required=True, help='验证数据文件')
    parser.add_argument('--output-dir', type=str, default='/home/cx/trading_data/all_models', help='输出目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志
    logger = DetailedLogger(log_dir)
    
    logger.log("="*80)
    logger.log("🚀 开始训练所有模型")
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
    
    # 获取所有模型
    all_models = get_all_models()
    logger.log(f"\n找到 {len(all_models)} 个模型:")
    for name in all_models.keys():
        logger.log(f"  - {name}")
    
    # 训练所有模型
    all_results = []
    
    for model_name, model_class in all_models.items():
        result = train_single_model(
            model_name, 
            model_class, 
            train_loader, 
            val_loader, 
            device, 
            logger,
            args.output_dir
        )
        
        if result:
            all_results.append(result)
    
    # 保存所有结果
    results_file = os.path.join(args.output_dir, 'all_models_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    logger.log(f"\n{'='*80}")
    logger.log("📊 所有模型训练完成！")
    logger.log(f"{'='*80}")
    
    # 排序并显示结果
    all_results.sort(key=lambda x: x['best_val_acc'], reverse=True)
    
    logger.log(f"\n排名结果:")
    logger.log("-" * 80)
    for i, result in enumerate(all_results, 1):
        logger.log(f"{i}. {result['model_name']}: {result['best_val_acc']:.4f} (Epoch {result['best_epoch']}, {result['total_time']:.1f}s)")
    
    logger.log(f"\n结果已保存到: {results_file}")


if __name__ == "__main__":
    main()
