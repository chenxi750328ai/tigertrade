#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
后台运行序列长度测试
"""

import sys
import os
import signal
import json
from datetime import datetime

sys.path.insert(0, '/home/cx/tigertrade')

from scripts.analysis.sequence_length_tester import SequenceLengthTester

# 全局变量用于优雅退出
should_stop = False

def signal_handler(sig, frame):
    """处理中断信号"""
    global should_stop
    print("\n⚠️ 收到中断信号，将在当前测试完成后退出...")
    should_stop = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def main():
    """主函数"""
    global should_stop
    
    print("🚀 开始序列长度测试（后台模式）")
    print("="*60)
    print("💡 提示: 按 Ctrl+C 可以优雅退出（当前测试完成后）")
    print("="*60)
    
    # 关键序列长度（根据理论分析）
    key_lengths = [10, 50, 100, 150, 200, 250, 300]
    
    tester = SequenceLengthTester(
        data_dir='/home/cx/trading_data',
        min_length=10,
        max_length=300,
        step=50,
        convergence_window=3,
        convergence_threshold=0.02
    )
    
    # 加载数据
    df = tester.load_training_data()
    if df is None:
        print("❌ 无法加载数据")
        return
    
    print(f"✅ 数据加载成功: {len(df)}条")
    
    # 分割数据
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx].reset_index(drop=True)
    df_val = df.iloc[split_idx:].reset_index(drop=True)
    
    print(f"📊 数据分割: 训练集{len(df_train)}条, 验证集{len(df_val)}条")
    print("="*60)
    
    # 测试关键长度
    for seq_len in key_lengths:
        if should_stop:
            print("\n⚠️ 测试被中断")
            break
            
        print(f"\n{'='*60}")
        print(f"测试序列长度: {seq_len}")
        print(f"{'='*60}")
        
        try:
            # 准备数据
            print("📊 准备序列数据...")
            X_train, y_train = tester.prepare_data_with_sequence(df_train, seq_len)
            X_val, y_val = tester.prepare_data_with_sequence(df_val, seq_len)
            
            if len(X_train) == 0 or len(X_val) == 0:
                print(f"⚠️ 数据不足，跳过序列长度{seq_len}")
                continue
            
            print(f"训练集: {len(X_train)} 样本, 验证集: {len(X_val)} 样本")
            
            # 训练并评估
            print("🔬 训练和评估模型...")
            results = tester.train_and_evaluate(
                seq_length=seq_len,
                X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val
            )
            
            # 记录结果
            result_record = {
                'seq_length': seq_len,
                'accuracy': results['accuracy'],
                'loss': results['loss'],
                'prediction_variance': results['prediction_variance'],
                'composite_score': results['composite_score']
            }
            tester.results.append(result_record)
            
            print(f"✅ 结果: 准确率={results['accuracy']:.4f}, "
                  f"损失={results['loss']:.4f}, "
                  f"综合评分={results['composite_score']:.4f}")
            
            # 保存中间结果
            if tester.results:
                output_file = f'/home/cx/trading_data/sequence_test_progress_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'results': tester.results,
                        'timestamp': datetime.now().isoformat()
                    }, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            print(f"❌ 测试序列长度 {seq_len} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 找到最优长度
    if tester.results:
        best_result = max(tester.results, key=lambda x: x['composite_score'])
        optimal_length = best_result['seq_length']
        
        print("\n" + "="*60)
        print("📊 测试结果总结")
        print("="*60)
        print(f"{'序列长度':<10} {'准确率':<10} {'损失':<10} {'综合评分':<12}")
        print("-"*60)
        
        for r in sorted(tester.results, key=lambda x: x['seq_length']):
            print(f"{r['seq_length']:<10} {r['accuracy']:<10.4f} {r['loss']:<10.4f} {r['composite_score']:<12.4f}")
        
        print("\n" + "="*60)
        print(f"🏆 最优序列长度: {optimal_length}")
        print(f"   最佳综合评分: {best_result['composite_score']:.4f}")
        print(f"   准确率: {best_result['accuracy']:.4f}")
        print(f"   损失: {best_result['loss']:.4f}")
        print("="*60)
        
        # 保存最终结果
        output_file = f'/home/cx/trading_data/sequence_test_final_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'results': tester.results,
                'optimal_length': optimal_length,
                'best_result': best_result,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 结果已保存到: {output_file}")
        
        # 绘制图表
        try:
            tester.plot_results()
            print("📊 图表已生成")
        except Exception as e:
            print(f"⚠️ 生成图表失败: {e}")
        
    else:
        print("❌ 没有成功完成任何测试")

if __name__ == "__main__":
    main()
