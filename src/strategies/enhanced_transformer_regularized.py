"""
保持大容量的Enhanced Transformer模型 - 使用正则化技术解决过拟合
不降低模型容量，而是使用：
1. 标签平滑（Label Smoothing）
2. Dropout路径正则化
3. 更好的权重初始化
4. Layer Normalization增强
"""
import torch
import torch.nn as nn
import numpy as np


class LabelSmoothingCrossEntropy(nn.Module):
    """标签平滑的交叉熵损失（支持类别权重）"""
    def __init__(self, smoothing=0.2, weight=None):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.weight = weight
    
    def forward(self, pred, target):
        log_prob = torch.nn.functional.log_softmax(pred, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_prob)
            true_dist.fill_(self.smoothing / (pred.size(1) - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)
        
        # 计算损失
        loss = torch.sum(-true_dist * log_prob, dim=1)
        
        # 应用类别权重（如果有）
        if self.weight is not None:
            weight_tensor = self.weight[target]
            loss = loss * weight_tensor
        
        return torch.mean(loss)


class RegularizedEnhancedTradingTransformerWithProfit(nn.Module):
    """保持大容量的Enhanced Transformer - 使用正则化技术"""
    def __init__(self, input_size=46, nhead=8, num_layers=8, output_size=3, d_model=512, predict_profit=True):
        super(RegularizedEnhancedTradingTransformerWithProfit, self).__init__()
        self.d_model = d_model
        self.predict_profit = predict_profit
        
        # 输入投影（支持46维特征）
        self.input_projection = nn.Linear(input_size, d_model)
        
        # 位置编码 - 使用正弦余弦编码
        self.register_buffer('pos_encoding', self._create_positional_encoding(1000, d_model))
        
        # 多层Transformer编码器（保持大容量，但增加dropout）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*4,
            dropout=0.3,  # 增加dropout到0.3（但保持模型容量）
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 注意力池化层
        self.attention_pool = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True, dropout=0.3)
        
        # 更强的dropout
        self.dropout = nn.Dropout(0.5)
        
        # 动作分类头（保持大容量，但增加正则化）
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.GELU(),
            nn.LayerNorm(d_model//2),
            nn.Dropout(0.5),  # 更强的dropout
            nn.Linear(d_model//2, d_model//4),
            nn.GELU(),
            nn.LayerNorm(d_model//4),
            nn.Dropout(0.4),
            nn.Linear(d_model//4, output_size)
        )
        
        # 收益率预测头
        if predict_profit:
            self.profit_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(0.5),  # 更强的dropout
                nn.Linear(d_model, d_model // 2),
                nn.LayerNorm(d_model // 2),
                nn.GELU(),
                nn.Dropout(0.4),
                nn.Linear(d_model // 2, 1)
            )
        else:
            self.profit_head = None
        
        # 更好的权重初始化
        self._init_weights()
    
    def _create_positional_encoding(self, max_len, d_model):
        """创建正弦余弦位置编码"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def _init_weights(self):
        """更好的权重初始化"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    # 使用Kaiming初始化（适合GELU）
                    if 'transformer' in name or 'attention' in name:
                        nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='relu')
                    else:
                        nn.init.xavier_uniform_(param, gain=1.0)
                elif len(param.shape) == 1:
                    # 一维权重（如LayerNorm）使用较小的初始化
                    nn.init.constant_(param, 1.0)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        batch_size, seq_len = x.size(0), x.size(1)
        
        # 输入投影
        x = self.input_projection(x)
        
        # 添加位置编码
        if seq_len <= self.pos_encoding.size(1):
            pos_enc = self.pos_encoding[:, :seq_len, :]
        else:
            extended_pe = self._create_positional_encoding(seq_len, self.d_model).to(x.device)
            pos_enc = extended_pe[:, :seq_len, :]
        
        x = x + pos_enc
        
        # Transformer编码（应用dropout）
        out = self.transformer(x)
        
        # 注意力池化
        attn_out, _ = self.attention_pool(
            query=out[:, -1:, :],
            key=out,
            value=out
        )
        
        # 展平注意力输出
        pooled_out = attn_out.squeeze(1)
        pooled_out = self.dropout(pooled_out)
        
        # 动作分类
        action_logits = self.action_head(pooled_out)
        
        # 返回值组装
        outputs = []
        outputs.append(action_logits)
        
        # 收益率预测
        if self.predict_profit and self.profit_head is not None:
            profit = self.profit_head(pooled_out)
            outputs.append(profit)
        
        if len(outputs) == 1:
            return outputs[0]
        else:
            return tuple(outputs)
