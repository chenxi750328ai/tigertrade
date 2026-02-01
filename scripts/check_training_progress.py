"""
检查训练进度
"""
import os
import time
from datetime import datetime

def check_training_progress():
    """检查训练进度"""
    data_dir = '/home/cx/trading_data'
    
    print("="*70)
    print("训练进度检查")
    print("="*70)
    print(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 检查无监督预训练
    print("📊 无监督预训练:")
    print("-" * 70)
    pretrained_model = os.path.join(data_dir, 'pretrained_return_model.pth')
    if os.path.exists(pretrained_model):
        mtime = datetime.fromtimestamp(os.path.getmtime(pretrained_model))
        size = os.path.getsize(pretrained_model) / (1024 * 1024)  # MB
        print(f"  ✅ 预训练模型已存在")
        print(f"     文件: pretrained_return_model.pth")
        print(f"     大小: {size:.2f} MB")
        print(f"     修改时间: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 检查是否是最新的（5分钟内）
        if (datetime.now() - mtime).total_seconds() < 300:
            print(f"     ⚡ 正在训练中（最近5分钟内有更新）")
        else:
            print(f"     ⏸️  可能已完成或已停止")
    else:
        print(f"  ⏳ 预训练模型尚未生成")
    
    # 检查多模型对比训练
    print("\n📊 多模型对比训练:")
    print("-" * 70)
    
    lstm_model = os.path.join(data_dir, 'best_lstm_improved.pth')
    comparison_results = os.path.join(data_dir, 'model_comparison_results.txt')
    
    if os.path.exists(lstm_model):
        mtime = datetime.fromtimestamp(os.path.getmtime(lstm_model))
        size = os.path.getsize(lstm_model) / (1024 * 1024)  # MB
        print(f"  ✅ LSTM模型已存在")
        print(f"     文件: best_lstm_improved.pth")
        print(f"     大小: {size:.2f} MB")
        print(f"     修改时间: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print(f"  ⏳ LSTM模型尚未生成")
    
    if os.path.exists(comparison_results):
        mtime = datetime.fromtimestamp(os.path.getmtime(comparison_results))
        print(f"  ✅ 对比结果已存在")
        print(f"     文件: model_comparison_results.txt")
        print(f"     修改时间: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\n  对比结果内容:")
        print("-" * 70)
        with open(comparison_results, 'r', encoding='utf-8') as f:
            print(f.read())
    else:
        print(f"  ⏳ 对比结果尚未生成")
    
    # 检查训练日志
    print("\n📊 训练日志:")
    print("-" * 70)
    log_file = '/tmp/unsupervised_training.log'
    if os.path.exists(log_file):
        mtime = datetime.fromtimestamp(os.path.getmtime(log_file))
        size = os.path.getsize(log_file)
        print(f"  ✅ 日志文件存在")
        print(f"     文件: {log_file}")
        print(f"     大小: {size:,} bytes")
        print(f"     修改时间: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 显示最后几行
        if size > 0:
            print(f"\n  最后10行日志:")
            print("-" * 70)
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines[-10:]:
                    print(f"  {line.rstrip()}")
    else:
        print(f"  ⏳ 日志文件不存在")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    check_training_progress()
