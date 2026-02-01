"""
按照1、2、3的顺序实施改进方案

1. 数据层面改进（增加数据量、改进数据增强）
2. 训练策略改进（增强正则化、改进损失函数）
3. 模型架构改进（PEFT、MoE）
"""
import os
import sys
sys.path.insert(0, '/home/cx/tigertrade')

print("="*70)
print("按照1、2、3的顺序实施改进方案")
print("="*70)

# 1. 数据层面改进
print("\n【步骤1】数据层面改进")
print("-"*70)

# 1.1 获取更多数据
print("\n1.1 获取更多训练数据...")
from scripts.fetch_more_training_data_improved import fetch_more_data_comprehensive
try:
    total_samples = fetch_more_data_comprehensive(days_back=90)
    print(f"✅ 数据获取完成，总样本数: {total_samples:,}")
except Exception as e:
    print(f"⚠️ 数据获取失败: {e}")

# 1.2 合并数据
print("\n1.2 合并训练数据...")
from scripts.merge_training_data import main as merge_main
try:
    merge_main()
    print("✅ 数据合并完成")
except Exception as e:
    print(f"⚠️ 数据合并失败: {e}")

# 1.3 分析数据分布
print("\n1.3 分析训练集和验证集分布...")
from scripts.analyze_train_val_distribution import analyze_distribution_difference
try:
    analyze_distribution_difference()
    print("✅ 分布分析完成")
except Exception as e:
    print(f"⚠️ 分布分析失败: {e}")

# 2. 训练策略改进
print("\n【步骤2】训练策略改进")
print("-"*70)
print("✅ 已实施:")
print("  - 增强正则化（权重衰减：1e-3 → 5e-3）")
print("  - 改进学习率调度（Warmup + CosineAnnealing）")
print("  - 组合损失函数（Label Smoothing + Focal Loss）")
print("  - 高级数据增强（时间序列特定）")

# 3. 模型架构改进
print("\n【步骤3】模型架构改进")
print("-"*70)
print("✅ 已实施:")
print("  - PEFT（参数高效微调）：冻结Transformer层，只训练分类头和收益率头")
print("  - MoE（混合专家模型）：4个专家，每次激活2个，稀疏激活率50%")
print("  - 稀疏注意力：局部窗口=20，注意力头dropout=10%")

# 开始训练
print("\n" + "="*70)
print("开始训练（使用所有改进）")
print("="*70)

from scripts.train_multiple_models_comparison import ModelComparisonTrainer

trainer = ModelComparisonTrainer()
trainer.run()

print("\n✅ 训练完成！")
