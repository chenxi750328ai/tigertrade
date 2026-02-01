#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用期货数据下载和模型训练系统
支持任意标的：数据采集 + 模型训练 + 测试评估
"""

import sys
import os
import time
import argparse
from datetime import datetime

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/cx/tigertrade')

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='通用期货数据下载和模型训练系统')
    parser.add_argument('--symbol', type=str, default=None, 
                        help='期货标的代码（如SIL2603, GC2603等），默认使用配置文件中的FUTURE_SYMBOL')
    parser.add_argument('--real-api', action='store_true', help='使用真实API获取数据')
    parser.add_argument('--days', type=int, default=60, help='获取天数（默认60天）')
    parser.add_argument('--min-records', type=int, default=20000, help='最少记录数（默认20000）')
    parser.add_argument('--max-records', type=int, default=50000, help='最大记录数（默认50000）')
    parser.add_argument('--output-dir', type=str, default=None, 
                        help='输出目录（默认根据标的自动生成）')
    
    args = parser.parse_args()
    
    # 获取标的代码
    if args.symbol:
        symbol = args.symbol
        # 临时设置环境变量
        os.environ['TRADING_SYMBOL'] = symbol
    else:
        # 使用配置文件中的默认标的
        from src import tiger1
        symbol = tiger1.FUTURE_SYMBOL
    
    # 自动生成输出目录（如果未指定）
    if not args.output_dir:
        # 从完整标的提取简短代码（如SIL.COMEX.202603 -> SIL2603）
        symbol_short = symbol.replace('.', '_').replace('COMEX', '').replace('__', '_')
        args.output_dir = f'/home/cx/trading_data/{symbol_short}_dataset'
    
    print("\n" + "=" * 80)
    print("🚀 通用期货数据下载和模型训练完整流程")
    print("=" * 80)
    print(f"标的代码: {symbol}")
    print(f"使用真实API: {args.real_api}")
    print(f"目标天数: {args.days}")
    print(f"最少记录数: {args.min_records}")
    print(f"最大记录数: {args.max_records}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 80)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 阶段1: 数据采集
    print("\n" + "=" * 80)
    print("📥 阶段1: 数据采集")
    print("=" * 80)
    
    from collect_large_dataset import LargeDatasetCollector
    
    collector = LargeDatasetCollector(
        use_real_api=args.real_api,
        days=args.days,
        max_records=args.max_records
    )
    
    # 更新输出目录
    collector.output_dir = args.output_dir
    
    # 运行数据采集
    files = collector.run()
    
    if not files or 'train' not in files or 'val' not in files or 'test' not in files:
        print("\n❌ 数据采集失败！")
        return 1
    
    # 检查数据量是否满足要求
    import pandas as pd
    train_df = pd.read_csv(files['train'])
    val_df = pd.read_csv(files['val'])
    test_df = pd.read_csv(files['test'])
    
    total_records = len(train_df) + len(val_df) + len(test_df)
    
    print(f"\n📊 数据统计:")
    print(f"  训练集: {len(train_df)} 条")
    print(f"  验证集: {len(val_df)} 条")
    print(f"  测试集: {len(test_df)} 条")
    print(f"  总计: {total_records} 条")
    
    if total_records < args.min_records:
        print(f"\n⚠️ 数据量不足！需要至少 {args.min_records} 条，实际只有 {total_records} 条")
        print("   建议：增加 --days 参数或使用 --real-api 获取真实数据")
        return 1
    
    print(f"\n✅ 数据量满足要求！")
    
    # 阶段2: 模型训练
    print("\n" + "=" * 80)
    print("🤖 阶段2: 训练所有模型")
    print("=" * 80)
    
    # 导入训练模块
    from train_all_models import get_all_models, train_single_model
    from train_with_detailed_logging import DetailedLogger, TradingDataset
    from config import TrainingConfig, FeatureConfig
    import torch
    from torch.utils.data import DataLoader
    
    # 创建日志
    log_dir = os.path.join(args.output_dir, 'training_logs')
    os.makedirs(log_dir, exist_ok=True)
    logger = DetailedLogger(log_dir)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and TrainingConfig.DEVICE == 'cuda' else 'cpu')
    logger.log(f"使用设备: {device}")
    
    # 准备数据集
    feature_cols = FeatureConfig.get_all_features()
    train_dataset = TradingDataset(train_df, feature_cols)
    val_dataset = TradingDataset(val_df, feature_cols)
    test_dataset = TradingDataset(test_df, feature_cols)
    
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
    test_loader = DataLoader(
        test_dataset,
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
    model_dir = os.path.join(args.output_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    for model_name, model_class in all_models.items():
        result = train_single_model(
            model_name, 
            model_class, 
            train_loader, 
            val_loader, 
            device, 
            logger,
            model_dir
        )
        
        if result:
            all_results.append(result)
    
    # 阶段3: 模型测试评估
    print("\n" + "=" * 80)
    print("📊 阶段3: 测试评估")
    print("=" * 80)
    
    import torch.nn as nn
    
    test_results = []
    criterion = nn.CrossEntropyLoss()
    
    for result in all_results:
        model_name = result['model_name']
        logger.log(f"\n测试 {model_name}...")
        
        try:
            # 加载最佳模型
            model_path = os.path.join(model_dir, f'{model_name}_best.pth')
            if not os.path.exists(model_path):
                logger.log(f"  ⚠️ 模型文件不存在: {model_path}")
                continue
            
            # 重新创建模型实例
            all_models_dict = get_all_models()
            model_class = all_models_dict[model_name]
            strategy = model_class()
            
            # 获取实际的模型对象
            if hasattr(strategy, 'model'):
                model = strategy.model
            elif hasattr(strategy, 'network'):
                model = strategy.network
            elif hasattr(strategy, 'lstm_model'):
                model = strategy.lstm_model
            elif hasattr(strategy, 'to'):
                model = strategy
            else:
                logger.log(f"  ⚠️ {model_name} 无法获取模型对象，跳过测试")
                continue
            
            model = model.to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            
            # 在测试集上评估
            test_loss = 0
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)
                    
                    # 为Transformer模型添加seq_len维度
                    if len(batch_X.shape) == 2:
                        batch_X = batch_X.unsqueeze(1)  # (batch, features) -> (batch, 1, features)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    test_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += batch_y.size(0)
                    test_correct += (predicted == batch_y).sum().item()
            
            test_acc = test_correct / test_total if test_total > 0 else 0
            test_loss = test_loss / len(test_loader) if len(test_loader) > 0 else 0
            
            logger.log(f"  ✅ 测试准确率: {test_acc:.4f}, 测试损失: {test_loss:.4f}")
            
            test_results.append({
                'model_name': model_name,
                'test_acc': test_acc,
                'test_loss': test_loss,
                'val_acc': result['best_val_acc']
            })
            
        except Exception as e:
            logger.log(f"  ❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 保存所有结果
    import json
    
    results_file = os.path.join(args.output_dir, 'all_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'training_results': all_results,
            'test_results': test_results,
            'data_files': files,
            'config': {
                'real_api': args.real_api,
                'days': args.days,
                'total_records': total_records,
                'train_records': len(train_df),
                'val_records': len(val_df),
                'test_records': len(test_df)
            }
        }, f, indent=2, ensure_ascii=False)
    
    # 生成最终报告
    print("\n" + "=" * 80)
    print("📋 最终报告")
    print("=" * 80)
    
    # 按测试准确率排序
    test_results.sort(key=lambda x: x['test_acc'], reverse=True)
    
    print(f"\n模型排名（按测试集准确率）:")
    print("-" * 80)
    print(f"{'排名':<6} {'模型名称':<30} {'验证准确率':<15} {'测试准确率':<15}")
    print("-" * 80)
    
    for i, result in enumerate(test_results, 1):
        print(f"{i:<6} {result['model_name']:<30} {result['val_acc']:<15.4f} {result['test_acc']:<15.4f}")
    
    print("\n" + "=" * 80)
    print("✅ 完整流程执行完毕！")
    print("=" * 80)
    print(f"\n生成的文件:")
    print(f"  - 数据目录: {args.output_dir}")
    print(f"  - 模型目录: {model_dir}")
    print(f"  - 日志目录: {log_dir}")
    print(f"  - 结果文件: {results_file}")
    
    # 保存报告文件
    report_file = os.path.join(args.output_dir, 'final_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("SIL2603数据下载和模型训练最终报告\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"生成时间: {datetime.now()}\n\n")
        f.write(f"数据统计:\n")
        f.write(f"  - 训练集: {len(train_df)} 条\n")
        f.write(f"  - 验证集: {len(val_df)} 条\n")
        f.write(f"  - 测试集: {len(test_df)} 条\n")
        f.write(f"  - 总计: {total_records} 条\n\n")
        f.write(f"模型排名（按测试集准确率）:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'排名':<6} {'模型名称':<30} {'验证准确率':<15} {'测试准确率':<15}\n")
        f.write("-" * 80 + "\n")
        for i, result in enumerate(test_results, 1):
            f.write(f"{i:<6} {result['model_name']:<30} {result['val_acc']:<15.4f} {result['test_acc']:<15.4f}\n")
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"  - 报告文件: {report_file}")
    print()
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
