#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
监控序列长度测试进度
"""

import os
import json
import glob
from datetime import datetime

def monitor_test():
    """监控测试进度"""
    print("📊 序列长度测试监控")
    print("="*60)
    
    # 1. 检查测试进程
    import subprocess
    result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
    test_processes = [line for line in result.stdout.split('\n') 
                     if 'run_sequence_test_background' in line or 'quick_sequence_test' in line]
    
    if test_processes:
        print("✅ 测试进程正在运行:")
        for proc in test_processes[:3]:
            print(f"   {proc[:80]}")
    else:
        print("⚠️ 未发现测试进程（可能已完成或未启动）")
    
    print("\n" + "-"*60)
    
    # 2. 检查日志文件
    log_file = '/tmp/sequence_test_background.log'
    if os.path.exists(log_file):
        print(f"📄 日志文件: {log_file}")
        print("   最新日志:")
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            for line in lines[-10:]:
                print(f"   {line.rstrip()}")
    else:
        print("⚠️ 日志文件不存在")
    
    print("\n" + "-"*60)
    
    # 3. 检查进度文件
    progress_files = glob.glob('/home/cx/trading_data/sequence_test_progress_*.json')
    if progress_files:
        latest_progress = max(progress_files, key=os.path.getmtime)
        print(f"📂 最新进度文件: {latest_progress}")
        
        with open(latest_progress, 'r', encoding='utf-8') as f:
            data = json.load(f)
            results = data.get('results', [])
            
            if results:
                print(f"\n   已完成测试: {len(results)} 个序列长度")
                print(f"   {'序列长度':<10} {'准确率':<10} {'损失':<10} {'综合评分':<12}")
                print("   " + "-"*50)
                
                for r in sorted(results, key=lambda x: x['seq_length']):
                    print(f"   {r['seq_length']:<10} {r['accuracy']:<10.4f} "
                          f"{r['loss']:<10.4f} {r['composite_score']:<12.4f}")
                
                # 当前最佳
                if results:
                    best = max(results, key=lambda x: x['composite_score'])
                    print(f"\n   🏆 当前最佳: 序列长度{best['seq_length']} "
                          f"(准确率: {best['accuracy']:.4f})")
    else:
        print("⚠️ 未找到进度文件（测试可能刚开始）")
    
    print("\n" + "-"*60)
    
    # 4. 检查最终结果
    final_files = glob.glob('/home/cx/trading_data/sequence_test_final_*.json')
    if final_files:
        latest_final = max(final_files, key=os.path.getmtime)
        print(f"✅ 找到最终结果: {latest_final}")
        
        with open(latest_final, 'r', encoding='utf-8') as f:
            data = json.load(f)
            optimal = data.get('optimal_length', 'N/A')
            best_result = data.get('best_result', {})
            
            print(f"\n   🏆 最优序列长度: {optimal}")
            if best_result:
                print(f"   准确率: {best_result.get('accuracy', 0):.4f}")
                print(f"   损失: {best_result.get('loss', 0):.4f}")
                print(f"   综合评分: {best_result.get('composite_score', 0):.4f}")
    else:
        print("⏳ 测试尚未完成（等待最终结果）")
    
    print("\n" + "="*60)
    print("💡 提示:")
    print("   - 查看实时日志: tail -f /tmp/sequence_test_background.log")
    print("   - 重新运行监控: python scripts/analysis/monitor_sequence_test.py")
    print("="*60)

if __name__ == "__main__":
    monitor_test()
