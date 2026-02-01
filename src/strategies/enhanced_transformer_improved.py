"""
改进版Enhanced Transformer模型 - 解决过拟合问题
1. 减少模型容量（d_model: 512→256, num_layers: 8→6）
2. 增加正则化（dropout: 0.1→0.2）
3. 保持架构优势（注意力池化、GELU激活）
"""
import torch
import torch.nn as nn
import numpy as np


class ImprovedEnhancedTradingTransformerWithProfit(nn.Module):
    """改进版Enhanced Transformer - 减少容量，增加正则化"""
    def __init__(self, input_size=46, nhead=8, num_layers=6, output_size=3, d_model=256, predict_profit=True):
        super(ImprovedEnhancedTradingTransformerWithProfit, self).__init__()
        self.d_model = d_model
        self.predict_profit = predict_profit
        
        # 输入投影（支持46维特征）
        self.input_projection = nn.Linear(input_size, d_model)
        
        # 位置编码 - 使用正弦余弦编码
        self.register_buffer('pos_encoding', self._create_positional_encoding(1000, d_model))
        
        # 多层Transformer编码器（减少层数：8→6，增加dropout：0.1→0.2）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*4,  # FFN隐藏层是d_model的4倍
            dropout=0.2,  # 增加dropout从0.1到0.2
            batch_first=True,
            activation='gelu'  # 使用GELU激活函数
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 注意力池化层 - 用于整合序列信息
        self.attention_pool = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        
        self.dropout = nn.Dropout(0.4)  # 增加dropout从0.3到0.4
        
        # 动作分类头（增加正则化）
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.GELU(),
            nn.LayerNorm(d_model//2),
            nn.Dropout(0.4),  # 增加dropout
            nn.Linear(d_model//2, d_model//4),
            nn.GELU(),
            nn.LayerNorm(d_model//4),
            nn.Dropout(0.3),  # 增加dropout
            nn.Linear(d_model//4, output_size)
        )
        
        # 收益率预测头（如果启用，增加正则化）
        if predict_profit:
            self.profit_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(0.4),  # 增加dropout
                nn.Linear(d_model, d_model // 2),
                nn.LayerNorm(d_model // 2),
                nn.GELU(),
                nn.Dropout(0.3),  # 增加dropout
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
        
        # 注意力池化 - 使用最后一个token作为查询
        attn_out, _ = self.attention_pool(
            query=out[:, -1:, :],  # (batch, 1, d_model)
            key=out,               # (batch, seq_len, d_model)
            value=out              # (batch, seq_len, d_model)
        )
        
        # 展平注意力输出
        pooled_out = attn_out.squeeze(1)  # (batch, d_model)
        pooled_out = self.dropout(pooled_out)
        
        # 动作分类
        action_logits = self.action_head(pooled_out)
        
        # 返回值组装
        outputs = []
        outputs.append(action_logits)
        
        # 收益率预测（如果启用）
        if self.predict_profit and self.profit_head is not None:
            profit = self.profit_head(pooled_out)
            # 不在这里应用ReLU和clamp，让模型自由学习
            outputs.append(profit)
        
        if len(outputs) == 1:
            return outputs[0]
        else:
            return tuple(outputs)
