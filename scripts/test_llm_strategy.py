#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试LLM策略模型
"""

import sys
import glob
import pandas as pd
import numpy as np
sys.path.insert(0, '/home/cx/tigertrade')

from src.strategies.llm_strategy import LLMTradingStrategy

def test_model():
    """测试训练好的模型"""
    print('📊 测试LLM策略模型...')
    print('=' * 70)
    
    # 加载训练数据
    files = sorted(glob.glob('/home/cx/trading_data/training_data_multitimeframe_*.csv'))
    if not files:
        print('❌ 找不到训练数据文件')
        return
    
    # 找到数据量最大的文件
    max_size = 0
    best_file = None
    for f in files:
        df_test = pd.read_csv(f)
        if len(df_test) > max_size:
            max_size = len(df_test)
            best_file = f
    
    df = pd.read_csv(best_file)
    print(f'📄 使用数据文件: {best_file}')
    print(f'📊 数据量: {len(df)}条, 特征维度: {len(df.columns) - 1}维（不含timestamp）')
    
    # 初始化策略（会自动加载最新模型）
    print('\n🔧 初始化策略...')
    strategy = LLMTradingStrategy(mode='hybrid', predict_profit=True)
    
    # 测试预测
    print('\n🧪 测试模型预测（使用最后50个数据点）...')
    print('=' * 70)
    
    seq_length = 30
    test_start = max(seq_length, len(df) - 50)
    test_indices = range(test_start, len(df))
    
    predictions = []
    for idx in test_indices:
        try:
            row = df.iloc[idx]
            result = strategy.predict_action(row)
            
            # 处理不同的返回值格式
            if isinstance(result, tuple):
                if len(result) == 2:
                    action, confidence = result
                elif len(result) == 3:
                    action, confidence, profit = result
                elif len(result) == 4:
                    action, confidence, profit, grid_adjustment = result
                else:
                    action = result[0]
                    confidence = 0.5
            else:
                action = result
                confidence = 0.5
            
            action_names = {0: '不操作', 1: '买入', 2: '卖出'}
            predictions.append({
                'idx': idx,
                'price': row['price_current'],
                'action': action,
                'action_name': action_names.get(action, '未知'),
                'confidence': confidence
            })
        except Exception as e:
            print(f'  ⚠️ 索引 {idx} 预测失败: {e}')
            continue
    
    # 统计预测结果
    if predictions:
        print(f'\n📊 预测统计（共{len(predictions)}个样本）:')
        print('=' * 70)
        
        action_counts = {}
        for p in predictions:
            action = p['action_name']
            action_counts[action] = action_counts.get(action, 0) + 1
        
        for action, count in sorted(action_counts.items()):
            pct = count / len(predictions) * 100
            print(f'  {action}: {count}次 ({pct:.1f}%)')
        
        avg_confidence = np.mean([p['confidence'] for p in predictions])
        print(f'\n  平均置信度: {avg_confidence:.3f}')
        
        # 显示前10个预测
        print(f'\n📋 前10个预测结果:')
        print('-' * 70)
        print(f'{"索引":<8} {"价格":<10} {"动作":<8} {"置信度":<10}')
        print('-' * 70)
        for p in predictions[:10]:
            print(f'{p["idx"]:<8} {p["price"]:<10.2f} {p["action_name"]:<8} {p["confidence"]:<10.3f}')
    
    print('\n✅ 模型测试完成！')

if __name__ == '__main__':
    test_model()
