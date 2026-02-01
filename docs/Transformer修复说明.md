# Transformer训练修复说明

**修复时间**: 2026-01-23  
**问题**: Transformer训练时出现维度错误和数据量不足

---

## ❌ 一、发现的问题

### 1.1 维度错误

**问题**: `query should be unbatched 2D or batched 3D tensor but received 4-D query tensor`

**原因**: 
- Transformer的输入应该是3D (batch, seq, features)
- 但代码中使用了`unsqueeze(1)`，导致变成4D

**修复**:
- 移除`unsqueeze(1)`操作
- 确保X_train已经是3D形状
- 添加形状检查和reshape逻辑

### 1.2 数据量不足

**问题**: Transformer训练时只有1个类别

**原因**: 
- Transformer的数据准备方式与LSTM不一致
- 使用单点特征而不是序列特征

**修复**:
- 修改数据准备方式，使用序列特征（与LSTM一致）
- 构建序列：使用最近seq_length个时间步的特征

### 1.3 类别权重错误

**问题**: 只有1个类别时，类别权重计算错误

**修复**:
- 检查类别数量
- 如果不足3个，使用默认损失函数（不使用类别权重）

---

## 🔧 二、修复详情

### 2.1 数据准备方式统一

**之前**:
```python
for i in range(len(df) - look_ahead):
    row = df.iloc[i]
    features = self.prepare_features(row)  # 单点特征
    X.append(features)
```

**现在**:
```python
for i in range(min_required, len(df)):
    # 准备序列特征（历史seq_length个时间步）
    sequence_features = []
    for j in range(max(0, i - seq_length + 1), i + 1):
        row = df.iloc[j]
        features = self.prepare_features(row)
        sequence_features.append(features)
    # 构建序列
    sequence = np.array(sequence_features[-seq_length:], dtype=np.float32)
    X.append(sequence)
```

### 2.2 维度处理

**之前**:
```python
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(self.device)
```

**现在**:
```python
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
# 确保是3D (batch, seq, features)
if len(X_train_tensor.shape) == 2:
    X_train_tensor = X_train_tensor.view(-1, seq_length, feature_size)
elif len(X_train_tensor.shape) == 3:
    pass  # 已经是正确形状
```

### 2.3 类别权重处理

**之前**:
```python
class_weights = self.calculate_class_weights(y)
self.criterion = nn.CrossEntropyLoss(weight=class_weights)
```

**现在**:
```python
unique_labels = np.unique(y)
if len(unique_labels) < 3:
    class_weights = None
else:
    class_weights = self.calculate_class_weights(y)

if class_weights is not None and len(class_weights) == 3:
    self.criterion = nn.CrossEntropyLoss(weight=class_weights)
else:
    self.criterion = nn.CrossEntropyLoss()  # 不使用类别权重
```

---

## ✅ 三、修复状态

### 3.1 已修复 ✅

- ✅ 维度错误（移除unsqueeze，确保3D形状）
- ✅ 数据准备方式（使用序列特征）
- ✅ 类别权重处理（检查类别数量）

### 3.2 测试状态

- ✅ Transformer训练逻辑测试通过
- ⏳ 完整对比测试正在运行中

---

**状态**: Transformer训练问题已修复，完整对比测试正在运行
