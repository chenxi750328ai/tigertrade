import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

# 仅导入最基础的模型
from llm_strategy import LLMTradingStrategy

def debug_anomaly_detection():
    print("🔍 开始使用异常检测调试训练过程...")
    
    # 创建一个小的模拟数据集
    n_samples = 100
    X = np.random.randn(n_samples, 10).astype(np.float32)
    y = np.random.randint(0, 3, size=(n_samples,)).astype(np.int64)  # 3分类问题
    
    # 转换为DataFrame格式，模拟实际数据
    df = pd.DataFrame({
        'price_current': X[:, 0],
        'grid_lower': X[:, 1],
        'grid_upper': X[:, 2],
        'atr': X[:, 3],
        'rsi_1m': X[:, 4],
        'rsi_5m': X[:, 5],
        'buffer': X[:, 6],
        'threshold': X[:, 7],
        'near_lower': (X[:, 8] > 0).astype(int),
        'rsi_ok': (X[:, 9] > 0).astype(int)
    })
    
    print(f"📊 数据形状: X={X.shape}, y={y.shape}")
    
    # 初始化策略
    strategy = LLMTradingStrategy()
    print(f"✅ 模型初始化成功，参数数量: {sum(p.numel() for p in strategy.model.parameters())}")
    
    # 启用PyTorch的异常检测
    torch.autograd.set_detect_anomaly(True)
    
    # 尝试训练
    try:
        print("🚀 开始训练...")
        strategy.train_model(df)
        print("✅ 训练完成")
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_anomaly_detection()