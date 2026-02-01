"""
支持收益率预测的GRU模型
与LSTM类似但使用GRU单元，提供不同的架构对比
"""
import torch
import torch.nn as nn
import numpy as np


class TradingGRUWithProfit(nn.Module):
    """支持收益率预测的GRU模型"""
    def __init__(self, input_size=46, hidden_size=128, num_layers=3, output_size=3, predict_profit=True):
        super(TradingGRUWithProfit, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.predict_profit = predict_profit
        
        # GRU层（与LSTM类似但使用GRU单元）
        self.gru = nn.GRU(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=0.2 if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(0.3)
        
        # 动作分类头
        self.action_head = nn.Linear(hidden_size, output_size)
        
        # 收益率预测头（如果启用）
        if predict_profit:
            self.profit_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.LayerNorm(hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size // 2, 1)
            )
        else:
            self.profit_head = None
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        # GRU前向传播
        out, _ = self.gru(x)  # (batch_size, seq_len, hidden_size)
        
        # 使用最后一个时间步的输出
        out = out[:, -1, :]  # (batch_size, hidden_size)
        out = self.dropout(out)
        
        # 动作分类
        action_logits = self.action_head(out)
        
        # 返回值组装
        outputs = []
        outputs.append(action_logits)
        
        # 收益率预测（如果启用）
        if self.predict_profit and self.profit_head is not None:
            profit = self.profit_head(out)
            # 不在这里应用ReLU和clamp，让模型自由学习
            outputs.append(profit)
        
        if len(outputs) == 1:
            return outputs[0]
        else:
            return tuple(outputs)
