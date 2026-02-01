"""
支持收益率预测的Transformer模型
基于LargeTradingTransformer，添加收益率预测头
"""
import torch
import torch.nn as nn
import numpy as np


class TradingTransformerWithProfit(nn.Module):
    """支持收益率预测的Transformer模型"""
    def __init__(self, input_size=46, nhead=8, num_layers=6, output_size=3, d_model=256, predict_profit=True):
        super(TradingTransformerWithProfit, self).__init__()
        self.d_model = d_model
        self.predict_profit = predict_profit
        
        # 输入投影（支持46维特征）
        self.input_projection = nn.Linear(input_size, d_model)
        
        # 位置编码 - 使用正弦余弦编码
        self.register_buffer('pos_encoding', self._create_positional_encoding(1000, d_model))
        
        # 多层Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*4,  # FFN隐藏层是d_model的4倍
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.dropout = nn.Dropout(0.3)
        
        # 动作分类头
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model//2, d_model//4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model//4, output_size)
        )
        
        # 收益率预测头（如果启用）
        if predict_profit:
            self.profit_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(d_model, d_model // 2),
                nn.LayerNorm(d_model // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(d_model // 2, 1)
            )
        else:
            self.profit_head = None
        
        # 初始化权重
        self._init_weights()
    
    def _create_positional_encoding(self, max_len, d_model):
        """创建正弦余弦位置编码"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # (1, max_len, d_model)
    
    def _init_weights(self):
        """初始化权重"""
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        batch_size, seq_len = x.size(0), x.size(1)
        
        # 输入投影
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # 添加位置编码
        if seq_len <= self.pos_encoding.size(1):
            pos_enc = self.pos_encoding[:, :seq_len, :]
        else:
            # 如果序列太长，扩展位置编码
            extended_pe = self._create_positional_encoding(seq_len, self.d_model).to(x.device)
            pos_enc = extended_pe[:, :seq_len, :]
        
        x = x + pos_enc
        
        # Transformer编码
        out = self.transformer(x)  # (batch_size, seq_len, d_model)
        
        # 使用最后一个时间步的输出
        out = out[:, -1, :]  # (batch_size, d_model)
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
