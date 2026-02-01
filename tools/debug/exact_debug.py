import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

from llm_strategy import LLMTradingStrategy

def exact_debug():
    print("🔍 精确定位inplace操作问题...")
    
    # 创建一个非常小的数据集用于调试
    n_samples = 4
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
    
    # 启用异常检测
    torch.autograd.set_detect_anomaly(True)
    
    # 只做一次前向和反向传播
    try:
        print("🚀 开始单步训练...")
        
        # 准备数据
        X_data, y_data = [], []
        look_ahead = 1  # 使用最小的前瞻窗口
        
        for i in range(len(df) - look_ahead):
            row = df.iloc[i]
            features = strategy.prepare_features(row)
            X_data.append(features)
            
            # 简化的标签生成
            label = np.random.randint(0, 3)  # 随机标签用于测试
            y_data.append(label)
        
        if len(X_data) < 1:
            print("数据不足")
            return
        
        X_data = np.array(X_data)
        y_data = np.array(y_data)
        
        # 转换为张量
        X_tensor = torch.tensor(X_data, dtype=torch.float32).unsqueeze(1).to(strategy.device)
        y_tensor = torch.tensor(y_data, dtype=torch.long).to(strategy.device)
        
        print(f"数据张量形状: X={X_tensor.shape}, y={y_tensor.shape}")
        
        # 创建模型、优化器和损失函数
        model = strategy.model
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        
        # 前向传播
        print("➡️ 执行前向传播...")
        outputs = model(X_tensor)
        print(f"输出形状: {outputs.shape}")
        
        loss = criterion(outputs, y_tensor)
        print(f"损失值: {loss.item()}")
        
        # 反向传播
        print("⬅️ 执行反向传播...")
        optimizer.zero_grad()
        loss.backward()  # 这里可能会出错
        optimizer.step()
        
        print("✅ 单步训练成功完成，未发现问题")
        
    except Exception as e:
        print(f"❌ 发现问题: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    exact_debug()