import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
from datetime import datetime
import threading
import time
import glob
import argparse
from typing import Tuple, Optional

# 导入基础策略接口
try:
    from src.strategies.base_strategy import BaseTradingStrategy
except ImportError:
    # 如果无法导入，创建一个占位符类（向后兼容）
    from abc import ABC
    BaseTradingStrategy = ABC

class TradingLSTM(nn.Module):
    """用于交易决策的LSTM模型（支持动作预测、收益率预测和网格参数调整）"""
    def __init__(self, input_size=46, hidden_size=128, num_layers=3, output_size=3, 
                 predict_grid_adjustment=True, predict_profit=False):
        super(TradingLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.predict_grid_adjustment = predict_grid_adjustment
        self.predict_profit = predict_profit
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.3)
        
        # 动作预测头（3类分类：不操作、买入、卖出）
        self.action_head = nn.Linear(hidden_size, output_size)
        
        # 初始化权重（使用Xavier初始化，可能有助于训练）
        self._initialize_weights()
        
        # 收益率预测头（回归，直接预测收益率）
        # 改进：使用更深的网络，但不使用BatchNorm（避免单样本推理问题）
        if predict_profit:
            self.profit_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),  # 使用LayerNorm替代BatchNorm
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.LayerNorm(hidden_size // 2),  # 使用LayerNorm替代BatchNorm
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size // 2, 1)
            )
        else:
            self.profit_head = None
        
        # 网格调整系数预测头（回归，范围 [0.8, 1.2]）
        if predict_grid_adjustment:
            self.grid_adjustment_head = nn.Linear(hidden_size, 1)
        else:
            self.grid_adjustment_head = None
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    # 使用Xavier初始化
                    torch.nn.init.xavier_uniform_(param)
                else:
                    torch.nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                torch.nn.init.constant_(param, 0.0)
    
    def forward(self, x):
        # 确保输入维度正确
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # 添加序列维度 (batch, seq, features)
        
        # 初始化隐藏状态和细胞状态 - 使用 detach() 确保梯度图分离
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, 
                         dtype=x.dtype, device=x.device, requires_grad=False)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, 
                         dtype=x.dtype, device=x.device, requires_grad=False)
        
        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))
        
        # 取最后一个时间步的输出
        out = out[:, -1, :]
        
        # 应用Dropout
        out = self.dropout(out)
        
        # 动作预测（分类）
        action_logits = self.action_head(out)
        
        # 返回值组装
        outputs = []
        outputs.append(action_logits)
        
        # 收益率预测（回归，如果启用）
        if self.predict_profit and self.profit_head is not None:
            # BatchNorm在单样本推理时需要特殊处理
            if out.size(0) == 1:
                # 单样本推理，使用eval模式（使用running stats）
                self.profit_head.eval()
                profit = self.profit_head(out)
                self.profit_head.train()
            else:
                profit = self.profit_head(out)
            # 改进：在forward中不应用ReLU和clamp，让模型自由学习
            # ReLU和clamp只在predict_action中应用（用于推理时的输出限制）
            # 这样训练时可以使用原始输出计算损失，不会影响梯度传播
            outputs.append(profit)  # 直接输出原始值，不限制
        
        # 网格调整系数预测（回归，范围 [0.8, 1.2]）
        if self.predict_grid_adjustment and self.grid_adjustment_head is not None:
            grid_adjustment_raw = self.grid_adjustment_head(out)
            # 使用sigmoid映射到 [0.8, 1.2]
            grid_adjustment = torch.sigmoid(grid_adjustment_raw) * 0.4 + 0.8
            outputs.append(grid_adjustment)
        
        # 返回结果
        if len(outputs) == 1:
            return outputs[0]
        elif len(outputs) == 2:
            return tuple(outputs)
        else:
            return tuple(outputs)


class LLMTradingStrategy(BaseTradingStrategy):
    """LLM交易策略（支持两种模式：计算模式和大模型识别模式）"""
    
    def __init__(self, data_dir='/home/cx/trading_data', model_path=None, mode='hybrid', predict_profit=False):
        """
        初始化LLM交易策略
        
        Args:
            data_dir: 数据目录
            model_path: 模型路径
            mode: 策略模式
                - 'hybrid': 计算模式（规则计算参数，模型预测动作和调整）
                - 'pure_ml': 大模型识别模式（模型自己识别所有特征和参数）
            predict_profit: 是否预测收益率（直接优化收益目标）
        """
        self.mode = mode  # 'hybrid' 或 'pure_ml'
        self.predict_profit = predict_profit  # 是否预测收益率
        # 检查GPU可用性
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"Using device: {self.device}")
        else:
            self.device = torch.device('cpu')
            print(f"Using device: {self.device}")
        
        self.data_dir = data_dir
        
        # 初始化模型（根据模式选择不同的输入输出）
        if mode == 'hybrid':
            # 计算模式：输入计算好的特征（46维，包含多时间尺度特征），输出动作和网格调整系数
            input_size = 46
            predict_grid_adjustment = True
        elif mode == 'pure_ml':
            # 大模型识别模式：输入原始OHLCV数据（序列），输出动作和网格参数
            input_size = 10  # open, high, low, close, volume (1m + 5m)
            predict_grid_adjustment = True
            # 注意：pure_ml模式需要更大的模型来识别特征
        else:
            raise ValueError(f"未知的模式: {mode}，支持的模式: 'hybrid', 'pure_ml'")
        
        self.model = TradingLSTM(
            input_size=input_size,
            hidden_size=128,  # 从64增加到128
            num_layers=3,      # 从2增加到3
            output_size=3,
            predict_grid_adjustment=predict_grid_adjustment,
            predict_profit=predict_profit
        ).to(self.device)
        
        # 直接初始化优化器，而不是设为None
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        self.criterion = None  # 将在训练时动态设置
        # 收益率损失函数（如果启用收益率预测）
        # 改进：使用MSELoss替代HuberLoss，更敏感，有助于收益率预测头学习
        self.profit_criterion = nn.MSELoss() if predict_profit else None
        
        # 控制训练和推理的标志
        self.should_train = True
        self.model_lock = threading.Lock()
        
        # 序列长度配置（根据测试结果，序列长度10表现最好）
        self._seq_length = 10  # 根据测试结果，序列长度10准确率最高（48.05%）
        self._historical_data = None  # 历史数据缓存
    
    @property
    def seq_length(self) -> int:
        """返回策略需要的序列长度"""
        return self._seq_length
    
    @property
    def strategy_name(self) -> str:
        """返回策略名称"""
        return f"LSTM ({self.mode})"
        
        # 如果提供了模型路径，则加载模型
        if model_path and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                if 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"✅ 从 {model_path} 加载模型成功")
            except Exception as e:
                print(f"❌ 加载模型失败: {e}，使用初始模型")
        
        # 注意：不自动启动训练线程，由用户决定是否启动
        # self.training_thread = threading.Thread(target=self.train_continuously, daemon=True)
        # self.training_thread.start()

    def calculate_class_weights(self, y):
        """计算类别权重以处理不平衡数据"""
        import numpy as np
        
        # 计算每个类别的样本数
        classes, counts = np.unique(y, return_counts=True)
        total_samples = len(y)
        
        # 计算每个类别的权重 (总样本数 / (类别数 * 每个类别的样本数))
        weights = []
        for count in counts:
            weight = total_samples / (len(classes) * count)
            weights.append(weight)
        
        # 转换为tensor
        class_weights = torch.FloatTensor(weights).to(self.device)
        return class_weights

    def prepare_sequence_features(self, df, current_idx, seq_length):
        """
        准备历史序列特征
        
        Args:
            df: 数据框
            current_idx: 当前索引
            seq_length: 序列长度
        
        Returns:
            sequence: (seq_length, feature_size) 的数组
        """
        start_idx = max(0, current_idx - seq_length + 1)
        sequence_df = df.iloc[start_idx:current_idx+1]
        
        sequences = []
        for _, row in sequence_df.iterrows():
            features = self.prepare_features(row)
            sequences.append(features)
        
        # 确定特征大小
        feature_size = 46 if self.mode == 'hybrid' else 10  # hybrid模式现在包含多时间尺度特征，所以是46维
        
        # 如果序列不足seq_length，用第一个值填充
        while len(sequences) < seq_length:
            if sequences:
                sequences.insert(0, sequences[0])
            else:
                sequences.insert(0, [0.0] * feature_size)
        
        return np.array(sequences, dtype=np.float32)
    
    def calculate_optimal_grid_adjustment(self, current_price, future_prices, grid_base):
        """
        计算最优网格调整系数（基于历史数据）
        
        Args:
            current_price: 当前价格
            future_prices: 未来价格序列
            grid_base: 基础网格间距
        
        Returns:
            optimal_adjustment: 最优调整系数 [0.8, 1.2]
        """
        if grid_base <= 0 or len(future_prices) == 0:
            return 1.0
        
        best_adjustment = 1.0
        best_profit = -float('inf')
        
        # 尝试不同的调整系数
        for adjustment in np.arange(0.8, 1.21, 0.05):
            grid_step = grid_base * adjustment
            grid_upper = current_price + grid_step / 2
            grid_lower = current_price - grid_step / 2
            
            # 计算在此网格参数下的收益
            # 买入：价格达到上轨时卖出
            buy_profit = 0.0
            if max(future_prices) >= grid_upper:
                buy_profit = (grid_upper - current_price) / current_price
            
            # 卖出：价格达到下轨时买入（做空收益）
            sell_profit = 0.0
            if min(future_prices) <= grid_lower:
                sell_profit = (current_price - grid_lower) / current_price
            
            # 总收益（取较大者）
            total_profit = max(buy_profit, sell_profit)
            
            if total_profit > best_profit:
                best_profit = total_profit
                best_adjustment = adjustment
        
        return best_adjustment
    
    def train_model(self, df, seq_length=10, max_epochs=50, patience=10, train_grid_adjustment=True):
        """
        训练模型（支持序列输入和网格调整系数训练）
        
        Args:
            df: 训练数据框
            seq_length: 序列长度
            max_epochs: 最大训练轮次
            patience: 早停耐心值
            train_grid_adjustment: 是否训练网格调整系数
        """
        try:
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            
            # 重新初始化优化器以避免状态问题
            # 使用更小的学习率，更稳定的训练
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0005, weight_decay=1e-4)
            scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6)
            
            # 准备训练数据，使用基于盈利的标签
            X, y, y_grid = [], [], []  # y_grid: 网格调整系数标签
            y_profit = []  # 收益率标签（始终初始化，避免局部变量作用域问题）
            
            # 改进：基于时间长度计算look_ahead，而不是固定步数
            # 网格交易通常需要2-4小时的持仓周期，使用2小时作为目标预测时长
            target_time_hours = 2.0  # 预测未来2小时
            # 计算数据的时间间隔（假设是1分钟K线）
            if 'timestamp' in df.columns and len(df) > 1:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                avg_interval = df['timestamp'].diff().dropna().median()
                if pd.notna(avg_interval):
                    # 计算步数：目标时间 / 平均间隔
                    look_ahead = int(pd.Timedelta(hours=target_time_hours) / avg_interval)
                    print(f"📊 数据时间间隔: {avg_interval}, 目标预测时长: {target_time_hours}小时, 计算得到look_ahead: {look_ahead}步")
                else:
                    look_ahead = 120  # 默认：2小时 = 120分钟（1分钟K线）
                    print(f"⚠️ 无法计算时间间隔，使用默认look_ahead: {look_ahead}步（{target_time_hours}小时）")
            else:
                look_ahead = 120  # 默认：2小时 = 120分钟（1分钟K线）
                print(f"⚠️ 数据中没有timestamp，使用默认look_ahead: {look_ahead}步（{target_time_hours}小时）")
            
            min_required = seq_length + look_ahead  # 需要至少seq_length + look_ahead个数据点
            
            print(f"📊 使用序列长度: {seq_length}, 需要至少 {min_required} 个数据点")
            print(f"📊 训练配置: 最大轮次={max_epochs}, 早停耐心={patience}, 网格调整训练={train_grid_adjustment}")
            
            for i in range(min_required, len(df)):
                # 计算未来look_ahead步的盈利（先检查数据是否足够）
                if i + look_ahead >= len(df):
                    break  # 数据不足，跳出循环
                
                # 准备序列特征（历史seq_length个时间步）
                sequence = self.prepare_sequence_features(df, i, seq_length)
                X.append(sequence)
                
                # 计算未来look_ahead步的盈利
                current_price = df.iloc[i]['price_current']
                future_prices = df.iloc[i+1:i+look_ahead+1]['price_current'].values
                
                if len(future_prices) == 0:
                    # 如果未来价格不足，移除刚添加的序列
                    X.pop()
                    continue
                
                # 获取基础网格参数
                grid_lower = df.iloc[i].get('grid_lower', current_price * 0.99) if hasattr(df.iloc[i], 'get') else df.iloc[i].get('grid_lower', current_price * 0.99)
                grid_upper = df.iloc[i].get('grid_upper', current_price * 1.01) if hasattr(df.iloc[i], 'get') else df.iloc[i].get('grid_upper', current_price * 1.01)
                grid_base = max(grid_upper - grid_lower, 0.01)  # 确保grid_base > 0
                
                # 计算最大盈利和最大亏损
                max_future_price = max(future_prices)
                min_future_price = min(future_prices)
                
                buy_profit = (max_future_price - current_price) / current_price
                sell_profit = (current_price - min_future_price) / current_price
                
                # 创建动作标签: 0=不操作, 1=买入, 2=卖出
                # 优化标签生成逻辑：降低阈值，增加交易信号
                # 改进：考虑持仓状态（如果有持仓，优先考虑卖出）
                profit_threshold = 0.003  # 从0.005降低到0.003，增加交易机会
                min_diff = 0.002  # 从0.003降低到0.002，更容易区分方向
                
                # 尝试从数据中获取持仓状态（如果存在）
                current_position = 0
                if 'current_position' in df.columns:
                    try:
                        current_position = int(df.iloc[i].get('current_position', 0))
                    except:
                        current_position = 0
                
                # 根据持仓状态调整标签生成逻辑
                # 改进：为了训练数据平衡，仍然生成所有3个标签，但实际交易时会根据持仓状态过滤
                if current_position > 0:
                    # 有持仓，优先考虑卖出
                    if sell_profit > profit_threshold:
                        label = 2  # 卖出
                    elif buy_profit > profit_threshold:
                        label = 1  # 买入（加仓）
                    else:
                        label = 0  # 不操作
                else:
                    # 无持仓，优先买入，但如果sell_profit明显更大，仍然标记为卖出（让模型学习）
                    # 实际交易时会根据持仓状态过滤，但训练时让模型学习所有情况
                    if abs(buy_profit - sell_profit) >= min_diff:
                        if buy_profit > sell_profit and buy_profit > profit_threshold:
                            label = 1  # 买入
                        elif sell_profit > buy_profit and sell_profit > profit_threshold:
                            label = 2  # 卖出（训练时允许，实际交易时会过滤）
                        else:
                            label = 0  # 不操作
                    else:
                        label = 0  # 不操作
                
                y.append(label)
                
                # 计算实际收益率（如果启用收益率预测）
                # 改进：根据动作标签选择对应的收益率，而不是取最大值
                if self.predict_profit:
                    buy_profit = (max(future_prices) - current_price) / current_price
                    sell_profit = (current_price - min(future_prices)) / current_price
                    # 根据动作标签选择对应的收益率
                    if label == 1:  # 买入
                        actual_profit = buy_profit
                    elif label == 2:  # 卖出
                        actual_profit = sell_profit
                    else:  # 不操作
                        actual_profit = 0.0
                    y_profit.append(actual_profit)
                
                # 计算最优网格调整系数（如果启用）- 确保与y同时添加
                if train_grid_adjustment and self.model.predict_grid_adjustment:
                    optimal_adjustment = self.calculate_optimal_grid_adjustment(
                        current_price, future_prices, grid_base
                    )
                    y_grid.append(optimal_adjustment)
                else:
                    y_grid.append(1.0)  # 默认不调整
            
            # 最终检查：确保X, y, y_grid, y_profit长度一致
            if self.predict_profit and y_profit is not None:
                min_len = min(len(X), len(y), len(y_grid), len(y_profit))
                if len(X) != min_len or len(y) != min_len or len(y_grid) != min_len or len(y_profit) != min_len:
                    print(f"⚠️ 警告: 数据长度不一致，调整到最小长度 {min_len}")
                    print(f"   X={len(X)}, y={len(y)}, y_grid={len(y_grid)}, y_profit={len(y_profit)}")
                    X = X[:min_len]
                    y = y[:min_len]
                    y_grid = y_grid[:min_len]
                    y_profit = y_profit[:min_len]
            else:
                min_len = min(len(X), len(y), len(y_grid))
                if len(X) != min_len or len(y) != min_len or len(y_grid) != min_len:
                    print(f"⚠️ 警告: 数据长度不一致，调整到最小长度 {min_len}")
                    print(f"   X={len(X)}, y={len(y)}, y_grid={len(y_grid)}")
                    X = X[:min_len]
                    y = y[:min_len]
                    y_grid = y_grid[:min_len]
            
            if len(X) < 10:  # 需要至少10个样本进行训练
                print("数据不足，跳过训练")
                return
            
            # 分析标签分布
            unique, counts = np.unique(y, return_counts=True)
            label_distribution = dict(zip(unique, counts))
            print(f"标签分布: {label_distribution}")
            
            # 如果某个类别占比过高，可能存在数据不平衡问题
            total_samples = len(y)
            for label, count in label_distribution.items():
                percentage = count / total_samples * 100
                print(f"标签 {label} 占比: {percentage:.2f}%")
            
            # 计算类别权重
            class_weights = self.calculate_class_weights(y)
            print(f"类别权重: {class_weights}")
            
            # 更新损失函数
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            self.grid_criterion = nn.MSELoss() if train_grid_adjustment else None
            # 收益率损失函数（使用Huber损失，对异常值更鲁棒）
            self.profit_criterion = nn.HuberLoss(delta=0.01) if self.predict_profit else None
            
            # 分割训练集和验证集 (80% 训练, 20% 验证)
            split_idx = int(len(X) * 0.8)
            
            # 确保所有列表长度一致
            if self.predict_profit:
                assert len(X) == len(y) == len(y_grid) == len(y_profit), \
                    f"数据长度不一致: X={len(X)}, y={len(y)}, y_grid={len(y_grid)}, y_profit={len(y_profit)}"
            else:
                assert len(X) == len(y) == len(y_grid), \
                    f"数据长度不一致: X={len(X)}, y={len(y)}, y_grid={len(y_grid)}"
            
            X_train = X[:split_idx]
            y_train = y[:split_idx]
            y_grid_train = y_grid[:split_idx]
            y_profit_train = y_profit[:split_idx] if self.predict_profit else None
            X_val = X[split_idx:]
            y_val = y[split_idx:]
            y_grid_val = y_grid[split_idx:]
            y_profit_val = y_profit[split_idx:] if self.predict_profit else None
            
            # 再次检查分割后的长度
            if self.predict_profit:
                assert len(X_train) == len(y_train) == len(y_grid_train) == len(y_profit_train), \
                    f"训练集长度不一致"
                assert len(X_val) == len(y_val) == len(y_grid_val) == len(y_profit_val), \
                    f"验证集长度不一致"
            else:
                assert len(X_train) == len(y_train) == len(y_grid_train), \
                    f"训练集长度不一致"
                assert len(X_val) == len(y_val) == len(y_grid_val), \
                    f"验证集长度不一致"
            
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            y_grid_train = np.array(y_grid_train, dtype=np.float32)
            y_profit_train = np.array(y_profit_train, dtype=np.float32) if self.predict_profit else None
            X_val = np.array(X_val)
            y_val = np.array(y_val)
            y_grid_val = np.array(y_grid_val, dtype=np.float32)
            y_profit_val = np.array(y_profit_val, dtype=np.float32) if self.predict_profit else None
            
            # 转换为张量
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
            y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(self.device)
            y_grid_train_tensor = torch.tensor(y_grid_train, dtype=torch.float32).to(self.device)
            y_profit_train_tensor = torch.tensor(y_profit_train, dtype=torch.float32).to(self.device) if self.predict_profit else None
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(self.device)
            y_grid_val_tensor = torch.tensor(y_grid_val, dtype=torch.float32).to(self.device)
            y_profit_val_tensor = torch.tensor(y_profit_val, dtype=torch.float32).to(self.device) if self.predict_profit else None
            
            print(f"✅ 数据准备完成: 训练集形状 {X_train_tensor.shape}, 验证集形状 {X_val_tensor.shape}")
            if self.predict_profit:
                print(f"📊 收益率统计: 训练集 min={y_profit_train.min():.4f}, max={y_profit_train.max():.4f}, mean={y_profit_train.mean():.4f}")
            
            # 创建数据集（包含网格调整系数和收益率标签）
            dataset_items = [X_train_tensor, y_train_tensor]
            val_dataset_items = [X_val_tensor, y_val_tensor]
            
            if self.predict_profit:
                dataset_items.append(y_profit_train_tensor)
                val_dataset_items.append(y_profit_val_tensor)
            
            if train_grid_adjustment and len(y_grid_train_tensor) == len(y_train_tensor):
                dataset_items.append(y_grid_train_tensor)
                val_dataset_items.append(y_grid_val_tensor)
            
            train_dataset = TensorDataset(*dataset_items)
            val_dataset = TensorDataset(*val_dataset_items)
            
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
            
            # 训练模型
            self.model.train()  # 设置为训练模式
            best_val_acc = 0.0
            patience_counter = 0
            
            for epoch in range(max_epochs):
                # 训练阶段
                total_loss = 0
                num_batches = 0
                correct_predictions = 0
                total_predictions = 0
                
                for batch_data in train_loader:
                    # 处理批次数据（可能包含收益率和网格调整系数标签）
                    batch_x = batch_data[0]
                    batch_y = batch_data[1]
                    batch_y_profit = None
                    batch_y_grid = None
                    
                    # 解析批次数据
                    if self.predict_profit:
                        if len(batch_data) >= 3:
                            batch_y_profit = batch_data[2].to(self.device)
                        if len(batch_data) >= 4:
                            batch_y_grid = batch_data[3].to(self.device)
                    else:
                        if len(batch_data) >= 3:
                            batch_y_grid = batch_data[2].to(self.device)
                    
                    # 创建新的张量副本以避免版本冲突
                    batch_x = batch_x.clone().detach().to(self.device)
                    batch_y = batch_y.clone().detach().to(self.device)
                    
                    # batch_x已经是(batch, seq, features)形状，不需要再unsqueeze
                    # 如果已经是3D，直接使用；如果是2D，需要unsqueeze
                    if len(batch_x.shape) == 2:
                        # (batch, features) -> (batch, 1, features)
                        batch_x = batch_x.unsqueeze(1).contiguous()
                    elif len(batch_x.shape) == 3:
                        # 已经是(batch, seq, features)，直接使用
                        pass
                    else:
                        # 其他情况，尝试reshape
                        batch_x = batch_x.view(batch_x.size(0), -1, batch_x.size(-1)).contiguous()
                    
                    self.optimizer.zero_grad()
                    model_output = self.model(batch_x)
                    
                    # 处理模型输出和计算损失
                    if isinstance(model_output, tuple):
                        # 解析模型输出
                        if len(model_output) == 2:
                            if self.predict_profit:
                                # 收益率 + 网格调整
                                action_logits, profit = model_output
                                grid_adjustment = None
                            else:
                                # 动作 + 网格调整
                                action_logits, grid_adjustment = model_output
                                profit = None
                        elif len(model_output) == 3:
                            # 动作 + 收益率 + 网格调整
                            action_logits, profit, grid_adjustment = model_output
                        else:
                            action_logits = model_output[0]
                            profit = model_output[1] if len(model_output) > 1 else None
                            grid_adjustment = model_output[2] if len(model_output) > 2 else None
                        
                        # 动作分类损失
                        action_loss = self.criterion(action_logits, batch_y)
                        loss = action_loss
                        
                        # 收益率回归损失（如果启用）
                        if self.predict_profit and profit is not None and batch_y_profit is not None and self.profit_criterion is not None:
                            profit_loss = self.profit_criterion(profit.squeeze(), batch_y_profit)
                            # 收益率损失权重增加到1.0（与动作分类同等重要，因为这是主要目标）
                            loss = loss + 1.0 * profit_loss
                        
                        # 网格调整系数回归损失（如果启用）
                        if train_grid_adjustment and grid_adjustment is not None and batch_y_grid is not None and self.grid_criterion is not None:
                            grid_loss = self.grid_criterion(grid_adjustment.squeeze(), batch_y_grid)
                            # 组合损失（网格损失权重0.1）
                            loss = loss + 0.1 * grid_loss
                    else:
                        action_logits = model_output
                        loss = self.criterion(action_logits, batch_y)
                    
                    loss.backward()
                    # 梯度裁剪，防止梯度爆炸
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    
                    # 计算准确率
                    predictions = torch.argmax(action_logits, dim=1)
                    correct_predictions += (predictions == batch_y).sum().item()
                    total_predictions += batch_y.size(0)
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    # 累计收益率预测误差（如果启用）
                    if self.predict_profit and profit is not None and batch_y_profit is not None:
                        profit_errors = torch.abs(profit.squeeze() - batch_y_profit)
                        if not hasattr(self, '_train_profit_errors'):
                            self._train_profit_errors = []
                        self._train_profit_errors.extend(profit_errors.detach().cpu().numpy().tolist())
                
                train_avg_loss = total_loss / num_batches
                train_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
                train_profit_mae = np.mean(self._train_profit_errors) if hasattr(self, '_train_profit_errors') and len(self._train_profit_errors) > 0 else None
                if hasattr(self, '_train_profit_errors'):
                    delattr(self, '_train_profit_errors')
                
                # 验证阶段
                self.model.eval()  # 设置为评估模式
                val_correct = 0
                val_total = 0
                val_loss = 0
                with torch.no_grad():
                    for batch_data in val_loader:
                        # 处理批次数据（与训练集相同）
                        batch_x = batch_data[0]
                        batch_y = batch_data[1]
                        batch_y_profit = None
                        batch_y_grid = None
                        
                        # 解析批次数据
                        if self.predict_profit:
                            if len(batch_data) >= 3:
                                batch_y_profit = batch_data[2]
                            if len(batch_data) >= 4:
                                batch_y_grid = batch_data[3]
                        else:
                            if len(batch_data) >= 3:
                                batch_y_grid = batch_data[2]
                        
                        batch_x = batch_x.clone().detach().to(self.device)
                        batch_y = batch_y.clone().detach().to(self.device)
                        
                        # batch_x已经是(batch, seq, features)形状，不需要再unsqueeze
                        if len(batch_x.shape) == 2:
                            batch_x = batch_x.unsqueeze(1).contiguous()
                        elif len(batch_x.shape) == 3:
                            pass  # 已经是正确形状
                        else:
                            batch_x = batch_x.view(batch_x.size(0), -1, batch_x.size(-1)).contiguous()
                        
                        model_output = self.model(batch_x)
                        
                        # 处理模型输出（与训练阶段相同的逻辑）
                        if isinstance(model_output, tuple):
                            if len(model_output) == 2:
                                if self.predict_profit:
                                    # 收益率 + 网格调整（不应该发生）
                                    action_logits, profit_or_grid = model_output
                                    grid_adjustment = None
                                    profit = profit_or_grid
                                else:
                                    # 动作 + 网格调整
                                    action_logits, grid_adjustment = model_output
                                    profit = None
                            elif len(model_output) == 3:
                                # 动作 + 收益率 + 网格调整
                                action_logits, profit, grid_adjustment = model_output
                            else:
                                action_logits = model_output[0]
                                profit = model_output[1] if len(model_output) > 1 and self.predict_profit else None
                                grid_adjustment = model_output[2] if len(model_output) > 2 else None
                        else:
                            action_logits = model_output
                            grid_adjustment = None
                            profit = None
                        
                        # 计算损失
                        action_loss = self.criterion(action_logits, batch_y)
                        loss = action_loss
                        
                        # 收益率损失（如果启用）
                        if self.predict_profit and profit is not None and batch_y_profit is not None and self.profit_criterion is not None:
                            profit_loss = self.profit_criterion(profit.squeeze(), batch_y_profit)
                            # 收益率损失权重增加到1.0（与动作分类同等重要）
                            loss = loss + 1.0 * profit_loss
                        
                        # 网格调整损失（如果启用）
                        if train_grid_adjustment and grid_adjustment is not None and batch_y_grid is not None and self.grid_criterion is not None:
                            batch_y_grid = batch_y_grid.to(self.device)
                            grid_loss = self.grid_criterion(grid_adjustment.squeeze(), batch_y_grid)
                            loss = loss + 0.1 * grid_loss
                        
                        val_loss += loss.item()
                        predictions = torch.argmax(action_logits, dim=1)
                        val_correct += (predictions == batch_y).sum().item()
                        val_total += batch_y.size(0)
                        
                        # 累计收益率预测误差（如果启用）
                        if self.predict_profit and profit is not None and batch_y_profit is not None:
                            profit_errors = torch.abs(profit.squeeze() - batch_y_profit)
                            if not hasattr(self, '_val_profit_errors'):
                                self._val_profit_errors = []
                            self._val_profit_errors.extend(profit_errors.detach().cpu().numpy().tolist())
                
                val_avg_loss = val_loss / len(val_loader)
                val_accuracy = val_correct / val_total if val_total > 0 else 0
                val_profit_mae = np.mean(self._val_profit_errors) if hasattr(self, '_val_profit_errors') and len(self._val_profit_errors) > 0 else None
                if hasattr(self, '_val_profit_errors'):
                    delattr(self, '_val_profit_errors')
                
                # 学习率调度
                scheduler.step(val_accuracy)
                
                print(f"训练轮次 {epoch+1}/{max_epochs}")
                print(f"  训练 - 损失: {train_avg_loss:.4f}, 准确率: {train_accuracy:.3f}")
                if train_profit_mae is not None:
                    print(f"  训练 - 收益率预测误差(MAE): {train_profit_mae:.4f}")
                print(f"  验证 - 损失: {val_avg_loss:.4f}, 准确率: {val_accuracy:.3f}")
                if val_profit_mae is not None:
                    print(f"  验证 - 收益率预测误差(MAE): {val_profit_mae:.4f}")
                print(f"  学习率: {self.optimizer.param_groups[0]['lr']:.6f}")
                
                # 保存最佳模型和早停
                if val_accuracy > best_val_acc:
                    best_val_acc = val_accuracy
                    patience_counter = 0
                    print(f"  🏆 新的最佳验证准确率: {best_val_acc:.3f}")
                    # 保存最佳模型
                    best_model_path = os.path.join(self.data_dir, 'best_model.pth')
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'best_val_acc': best_val_acc,
                        'epoch': epoch
                    }, best_model_path)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"  ⏹️ 早停于第 {epoch+1} 轮（验证准确率未提升 {patience} 轮）")
                        break
                
                self.model.train()  # 重新设置为训练模式以进行下一轮训练
        
        except Exception as e:
            print(f"训练过程错误: {e}")
            import traceback
            traceback.print_exc()

    def prepare_features(self, row):
        """从数据行中准备特征向量（支持Series和字典，根据模式选择不同的特征）"""
        try:
            # 统一访问方式：同时支持Series和字典
            def get_value(key, default=0):
                if isinstance(row, dict):
                    return row.get(key, default)
                elif isinstance(row, pd.Series):
                    return row.get(key, default) if key in row.index else default
                else:
                    return getattr(row, key, default)
            
            def get_value_safe(key, default=0, check_na=True):
                val = get_value(key, default)
                if check_na and pd.isna(val):
                    return default
                return val
            
            if self.mode == 'hybrid':
                # 计算模式：使用计算好的特征（47维，包含多时间尺度特征）
                # 获取Tick价格（如果存在）
                tick_price = get_value_safe('tick_price', 0)
                if tick_price == 0:
                    # 如果tick_price不存在，使用price_current
                    tick_price = get_value_safe('price_current', 0)
                
                # Tick相关特征
                tick_price_change = get_value_safe('tick_price_change', 0)
                tick_volatility = get_value_safe('tick_volatility', 0)
                tick_volume = get_value_safe('tick_volume', 0)
                tick_count = get_value_safe('tick_count', 0)
                tick_buy_volume = get_value_safe('tick_buy_volume', 0)
                tick_sell_volume = get_value_safe('tick_sell_volume', 0)
                
                # 计算Tick买卖比例（如果可用）
                tick_buy_sell_ratio = get_value_safe('tick_buy_sell_ratio', 0.5)
                if tick_buy_volume + tick_sell_volume > 0:
                    tick_buy_sell_ratio = tick_buy_volume / (tick_buy_volume + tick_sell_volume)
                
                # 1分钟特征
                atr_1m = get_value_safe('atr_1m', get_value_safe('atr', 0))
                rsi_1m = get_value_safe('rsi_1m', 50, check_na=True)
                boll_upper_1m = get_value_safe('boll_upper_1m', get_value('boll_upper', 0))
                boll_mid_1m = get_value_safe('boll_mid_1m', get_value('boll_mid', 0))
                boll_lower_1m = get_value_safe('boll_lower_1m', get_value('boll_lower', 0))
                boll_position_1m = get_value_safe('boll_position_1m', get_value('boll_position', 0.5))
                volatility_1m = get_value_safe('volatility_1m', get_value('volatility', 0))
                volume_1m = get_value_safe('volume_1m', 0)
                
                # 5分钟特征
                price_5m = get_value_safe('price_5m', get_value_safe('price_current', 0))
                rsi_5m = get_value_safe('rsi_5m', 50, check_na=True)
                atr_5m = get_value_safe('atr_5m', atr_1m)
                boll_upper_5m = get_value_safe('boll_upper_5m', boll_upper_1m)
                boll_mid_5m = get_value_safe('boll_mid_5m', boll_mid_1m)
                boll_lower_5m = get_value_safe('boll_lower_5m', boll_lower_1m)
                boll_position_5m = get_value_safe('boll_position_5m', boll_position_1m)
                volume_5m = get_value_safe('volume_5m', volume_1m)
                
                # 1小时特征
                price_1h = get_value_safe('price_1h', get_value_safe('price_current', 0))
                rsi_1h = get_value_safe('rsi_1h', 50, check_na=True)
                atr_1h = get_value_safe('atr_1h', atr_1m)
                boll_upper_1h = get_value_safe('boll_upper_1h', boll_upper_1m)
                boll_mid_1h = get_value_safe('boll_mid_1h', boll_mid_1m)
                boll_lower_1h = get_value_safe('boll_lower_1h', boll_lower_1m)
                boll_position_1h = get_value_safe('boll_position_1h', boll_position_1m)
                volume_1h = get_value_safe('volume_1h', volume_1m)
                trend_1h = get_value_safe('trend_1h', 0.5)  # 0=下跌, 0.5=横盘, 1=上涨
                
                # 日线特征
                price_1d = get_value_safe('price_1d', get_value_safe('price_current', 0))
                rsi_1d = get_value_safe('rsi_1d', 50, check_na=True)
                atr_1d = get_value_safe('atr_1d', atr_1m)
                boll_upper_1d = get_value_safe('boll_upper_1d', boll_upper_1m)
                boll_mid_1d = get_value_safe('boll_mid_1d', boll_mid_1m)
                boll_lower_1d = get_value_safe('boll_lower_1d', boll_lower_1m)
                boll_position_1d = get_value_safe('boll_position_1d', boll_position_1m)
                volume_1d = get_value_safe('volume_1d', volume_1m)
                trend_1d = get_value_safe('trend_1d', 0.5)
                ma_5d = get_value_safe('ma_5d', price_1d)
                ma_10d = get_value_safe('ma_10d', price_1d)
                ma_20d = get_value_safe('ma_20d', price_1d)
                
                # 网格参数
                grid_lower = get_value('grid_lower', boll_lower_1m)
                grid_upper = get_value('grid_upper', boll_upper_1m)
                
                # 构建47维特征向量
                features = [
                    # 基础特征（1分钟）- 16维
                    get_value_safe('price_current', 0),  # 0: K线价格
                    tick_price,  # 1: 真实Tick价格
                    tick_price_change,  # 2: Tick价格变化
                    tick_volatility,  # 3: Tick波动率
                    tick_volume,  # 4: Tick成交量
                    tick_count,  # 5: Tick数量
                    tick_buy_sell_ratio,  # 6: Tick买卖比例
                    atr_1m,  # 7: 1分钟ATR
                    rsi_1m,  # 8: 1分钟RSI
                    boll_upper_1m,  # 9: 1分钟布林带上轨
                    boll_mid_1m,  # 10: 1分钟布林带中轨
                    boll_lower_1m,  # 11: 1分钟布林带下轨
                    boll_position_1m,  # 12: 1分钟布林带位置
                    volatility_1m,  # 13: 1分钟波动率
                    volume_1m,  # 14: 1分钟成交量
                    # 5分钟特征 - 8维
                    price_5m,  # 15: 5分钟价格
                    rsi_5m,  # 16: 5分钟RSI
                    atr_5m,  # 17: 5分钟ATR
                    boll_upper_5m,  # 18: 5分钟布林带上轨
                    boll_mid_5m,  # 19: 5分钟布林带中轨
                    boll_lower_5m,  # 20: 5分钟布林带下轨
                    boll_position_5m,  # 21: 5分钟布林带位置
                    volume_5m,  # 22: 5分钟成交量
                    # 1小时特征 - 9维
                    price_1h,  # 23: 1小时价格
                    rsi_1h,  # 24: 1小时RSI
                    atr_1h,  # 25: 1小时ATR
                    boll_upper_1h,  # 26: 1小时布林带上轨
                    boll_mid_1h,  # 27: 1小时布林带中轨
                    boll_lower_1h,  # 28: 1小时布林带下轨
                    boll_position_1h,  # 29: 1小时布林带位置
                    volume_1h,  # 30: 1小时成交量
                    trend_1h,  # 31: 1小时趋势
                    # 日线特征 - 11维
                    price_1d,  # 32: 日线价格
                    rsi_1d,  # 33: 日线RSI
                    atr_1d,  # 34: 日线ATR
                    boll_upper_1d,  # 35: 日线布林带上轨
                    boll_mid_1d,  # 36: 日线布林带中轨
                    boll_lower_1d,  # 37: 日线布林带下轨
                    boll_position_1d,  # 38: 日线布林带位置
                    volume_1d,  # 39: 日线成交量
                    trend_1d,  # 40: 日线趋势
                    ma_5d,  # 41: 5日均线
                    ma_10d,  # 42: 10日均线
                    ma_20d,  # 43: 20日均线
                    # 网格参数 - 2维
                    grid_lower,  # 44: 网格下轨
                    grid_upper,  # 45: 网格上轨
                ]
                feature_size = 46  # 46维多时间尺度特征（不包含timestamp）
            elif self.mode == 'pure_ml':
                # 大模型识别模式：只使用原始OHLCV数据（10维：1m和5m各5个）
                # 1分钟数据
                features = [
                    get_value_safe('open_1m', 0),
                    get_value_safe('high_1m', 0),
                    get_value_safe('low_1m', 0),
                    get_value_safe('close_1m', get_value_safe('price_current', 0)),
                    get_value_safe('volume_1m', 0),
                    # 5分钟数据
                    get_value_safe('open_5m', 0),
                    get_value_safe('high_5m', 0),
                    get_value_safe('low_5m', 0),
                    get_value_safe('close_5m', get_value_safe('price_current', 0)),
                    get_value_safe('volume_5m', 0)
                ]
                feature_size = 10
            else:
                raise ValueError(f"未知的模式: {self.mode}")
            
            # 归一化特征
            features_np = np.array(features, dtype=np.float32)
            mean_val = np.mean(features_np)
            std_val = np.std(features_np) + 1e-8
            normalized_features = (features_np - mean_val) / std_val
            return normalized_features.tolist()
        except Exception as e:
            print(f"prepare_features错误: {e}")
            import traceback
            traceback.print_exc()
            # 返回默认特征值
            feature_size = 46 if self.mode == 'hybrid' else 10  # hybrid模式现在包含多时间尺度特征
            return [0.0] * feature_size
    
    
    
    def predict_action(self, current_data, historical_data=None):
        """
        使用模型预测交易动作
        
        Args:
            current_data: 当前数据字典
            historical_data: 历史数据DataFrame（可选）
        
        Returns:
            (action, confidence, profit_prediction) 或 (action, confidence)
        """
        with self.model_lock:
            try:
                # 准备输入数据（支持序列输入）
                # 如果提供了历史数据和序列长度，使用序列数据
                if hasattr(self, '_seq_length') and self._seq_length > 1 and hasattr(self, '_historical_data'):
                    historical_data = self._historical_data
                    seq_length = self._seq_length
                    if historical_data is not None and len(historical_data) >= seq_length:
                        # 使用历史序列数据
                        # 假设current_data是historical_data的最后一行或当前行
                        try:
                            if isinstance(current_data, pd.Series):
                                current_idx = len(historical_data) - 1
                            else:
                                current_idx = len(historical_data) - 1
                            sequence = self.prepare_sequence_features(historical_data, current_idx, seq_length)
                            input_tensor = torch.tensor([sequence], dtype=torch.float32).to(self.device)
                        except Exception as e:
                            # 降级到单点特征
                            features = self.prepare_features(current_data)
                            input_tensor = torch.tensor([features], dtype=torch.float32).unsqueeze(1).to(self.device)
                    else:
                        # 数据不足，使用单点特征
                        features = self.prepare_features(current_data)
                        input_tensor = torch.tensor([features], dtype=torch.float32).unsqueeze(1).to(self.device)
                else:
                    # 默认使用单点特征（向后兼容）
                    features = self.prepare_features(current_data)
                    input_tensor = torch.tensor([features], dtype=torch.float32).unsqueeze(1).to(self.device)
                
                # 预测
                with torch.no_grad():
                    self.model.eval()
                    model_output = self.model(input_tensor)
                    
                    # 处理模型输出（可能是动作logits、(action_logits, grid_adjustment)或(action_logits, profit, grid_adjustment)）
                    action_logits = None
                    profit = None
                    grid_adjustment_value = 1.0
                    
                    if isinstance(model_output, tuple):
                        if len(model_output) == 2:
                            if self.predict_profit:
                                # 收益率 + 网格调整（没有动作头？这不应该发生）
                                action_logits, profit_or_grid = model_output
                                if self.model.predict_grid_adjustment:
                                    profit, grid_adjustment = profit_or_grid, model_output[1] if len(model_output) > 1 else None
                                    grid_adjustment_value = float(grid_adjustment.cpu().item()) if grid_adjustment is not None else 1.0
                                else:
                                    profit = profit_or_grid
                            else:
                                # 动作 + 网格调整
                                action_logits, grid_adjustment = model_output
                                grid_adjustment_value = float(grid_adjustment.cpu().item())
                        elif len(model_output) == 3:
                            # 动作 + 收益率 + 网格调整
                            action_logits, profit, grid_adjustment = model_output
                            grid_adjustment_value = float(grid_adjustment.cpu().item())
                        else:
                            action_logits = model_output[0]
                            profit = model_output[1] if len(model_output) > 1 and self.predict_profit else None
                            grid_adjustment = model_output[2] if len(model_output) > 2 else None
                            grid_adjustment_value = float(grid_adjustment.cpu().item()) if grid_adjustment is not None else 1.0
                    else:
                        action_logits = model_output
                    
                    # 计算动作概率
                    probabilities = torch.softmax(action_logits, dim=1).cpu().numpy().flatten() if action_logits.dim() > 1 else torch.softmax(action_logits, dim=1).cpu().numpy()
                    if hasattr(probabilities, 'shape') and probabilities.shape and len(probabilities.shape) > 1:
                        probabilities = probabilities[0]
                    elif not isinstance(probabilities, np.ndarray):
                        probabilities = np.array([probabilities])
                    
                    # 返回最可能的动作: 0=不操作, 1=买入, 2=卖出
                    action = np.argmax(probabilities)
                    base_confidence = probabilities[action]
                    
                    # 如果启用了收益率预测，使用收益率预测来调整置信度
                    if self.predict_profit and profit is not None:
                        # 在推理时应用ReLU和clamp限制输出范围
                        profit = torch.relu(profit)  # 确保非负
                        profit = torch.clamp(profit, max=0.3)  # 限制上限为0.3（30%）
                        profit_value = float(profit.cpu().item())
                        
                        # 基于收益率预测调整置信度
                        if action == 1:  # 买入
                            # 预测收益率越高，置信度越高
                            # 假设5%为高收益，将收益率映射到[0, 1]
                            profit_confidence = min(1.0, max(0.0, profit_value / 0.05))
                            # 结合动作分类置信度和收益率置信度（收益率权重更高）
                            confidence = (base_confidence * 0.3 + profit_confidence * 0.7)
                        elif action == 2:  # 卖出
                            # 预测收益率越低（负值），置信度越高
                            profit_confidence = min(1.0, max(0.0, abs(profit_value) / 0.05))
                            confidence = (base_confidence * 0.3 + profit_confidence * 0.7)
                        else:  # 不操作
                            # 收益率接近0时，不操作的置信度高
                            profit_confidence = 1.0 - min(1.0, abs(profit_value) / 0.02)
                            confidence = (base_confidence * 0.5 + profit_confidence * 0.5)
                    else:
                        # 没有收益率预测，使用动作分类的置信度
                        confidence = base_confidence
                        profit_value = None
                    
                    # 返回结果（统一接口：(action, confidence, profit_prediction)）
                    if self.predict_profit and profit is not None:
                        profit_value = float(profit.cpu().item())
                        return int(action), float(confidence), float(profit_value)
                    else:
                        return int(action), float(confidence), None
            except Exception as e:
                print(f"预测错误: {e}")
                import traceback
                traceback.print_exc()
                return 0, 0.0, None  # 默认不操作
    
    def load_training_data(self):
        """加载训练数据"""
        # 获取最新的数据文件
        all_data_files = []
        
        # 获取所有数据目录
        all_data_dirs = glob.glob(os.path.join(self.data_dir, '202*-*-*'))
        if all_data_dirs:
            # 按日期排序，获取最新的几个文件
            sorted_dirs = sorted(all_data_dirs, reverse=True)
            for data_dir in sorted_dirs[:7]:  # 使用最近7天的数据
                # 包含原始数据和扩展数据
                data_files = glob.glob(os.path.join(data_dir, 'trading_data_*.csv'))
                data_files.extend(glob.glob(os.path.join(data_dir, 'extended_trading_data_*.csv')))
                data_files.extend(glob.glob(os.path.join(data_dir, 'prepared_features_*.csv')))  # 也包含准备好的特征数据
                all_data_files.extend(data_files)
        
        if not all_data_files:
            return None
        
        # 按修改时间排序，获取最新的文件
        data_file = sorted(all_data_files, key=os.path.getmtime, reverse=True)[0]
        
        try:
            df = pd.read_csv(data_file)
            # 清理数据
            df = df.dropna(subset=['price_current', 'grid_lower', 'grid_upper', 'atr', 'rsi_1m', 'rsi_5m'])
            
            if len(df) < 10:  # 需要至少10个样本进行训练
                return None
                
            return df
        except Exception as e:
            print(f"加载训练数据错误: {e}")
            return None
    
    def train_continuously(self):
        """连续训练模型的后台线程"""
        while self.should_train:
            try:
                # 加载数据
                df = self.load_training_data()
                if df is not None and len(df) > 0:
                    print(f"开始训练模型，数据量: {len(df)}")
                    with self.model_lock:
                        self.train_model(df)
                    print("模型训练完成")
                
                # 训练较慢，每30分钟训练一次
                time.sleep(1800)
                
            except Exception as e:
                print(f"训练线程错误: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(60)  # 出错后等待1分钟后继续
    
    def save_model(self, path):
        """保存模型"""
        with self.model_lock:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, path)
            print(f"模型已保存到 {path}")


def main():
    parser = argparse.ArgumentParser(description='LLM Trading Strategy')
    parser.add_argument('--mode', choices=['train', 'predict', 'both'], default='both',
                        help='运行模式: train(仅训练), predict(仅预测), both(全部)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='模型保存或加载路径')
    args = parser.parse_args()
    
    strategy = LLMTradingStrategy(model_path=args.model_path)
    
    if args.mode == 'train':
        print("仅运行训练模式...")
        # 训练模式，不退出
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            print("训练已停止")
    elif args.mode == 'predict':
        print("仅运行预测模式...")
        # 预测模式，演示如何使用
        sample_data = {
            'price_current': 93.5,
            'grid_lower': 93.0,
            'grid_upper': 94.0,
            'atr': 0.2,
            'rsi_1m': 30.0,
            'rsi_5m': 40.0,
            'buffer': 0.05,
            'threshold': 93.05,
            'near_lower': True,
            'rsi_ok': True
        }
        
        action, confidence = strategy.predict_action(sample_data)
        action_map = {0: "不操作", 1: "买入", 2: "卖出"}
        print(f"预测动作: {action_map[action]}, 置信度: {confidence:.3f}")
        
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            print("程序已停止")
    else:
        print("运行训练和预测模式...")
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            print("程序已停止")
    
    # 如果提供了保存路径，保存模型
    if args.model_path:
        strategy.save_model(args.model_path)


if __name__ == "__main__":
    main()