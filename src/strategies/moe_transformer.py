"""
MoE (Mixture of Experts) Transformer - 混合专家模型
核心原理：将单一模型拆分为多个"专家层"，训练/推理时只激活部分专家
减少实际参与计算的参数量，避免模型过度记忆训练集特征
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Expert(nn.Module):
    """单个专家网络（FFN）"""
    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super(Expert, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = nn.GELU()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class MoELayer(nn.Module):
    """MoE层：包含多个专家和一个门控网络"""
    def __init__(self, d_model, dim_feedforward, num_experts=8, top_k=2, dropout=0.1):
        """
        Args:
            d_model: 模型维度
            dim_feedforward: FFN隐藏层维度
            num_experts: 专家数量
            top_k: 每次激活的专家数量（稀疏激活）
            dropout: Dropout率
        """
        super(MoELayer, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # 创建多个专家
        self.experts = nn.ModuleList([
            Expert(d_model, dim_feedforward, dropout) 
            for _ in range(num_experts)
        ])
        
        # 门控网络（Gating Network）：决定激活哪些专家
        self.gate = nn.Linear(d_model, num_experts)
        
        # 负载均衡损失（Load Balancing Loss）的权重
        self.load_balance_weight = 0.01
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            output: (batch_size, seq_len, d_model)
            aux_loss: 负载均衡损失（用于训练）
        """
        batch_size, seq_len, d_model = x.shape
        
        # 门控网络计算每个专家的权重
        gate_logits = self.gate(x)  # (batch_size, seq_len, num_experts)
        
        # 获取top_k个专家
        top_k_gate_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        top_k_gate_probs = F.softmax(top_k_gate_logits, dim=-1)  # (batch_size, seq_len, top_k)
        
        # 初始化输出
        output = torch.zeros_like(x)
        
        # 对每个专家计算输出
        for i, expert in enumerate(self.experts):
            # 找到使用当前专家的位置
            expert_mask = (top_k_indices == i)  # (batch_size, seq_len, top_k)
            
            if expert_mask.any():
                # 计算当前专家在这些位置的权重
                expert_weights = torch.zeros(batch_size, seq_len, device=x.device)
                for k in range(self.top_k):
                    expert_weights += (expert_mask[:, :, k].float() * top_k_gate_probs[:, :, k])
                
                # 应用专家
                expert_output = expert(x)  # (batch_size, seq_len, d_model)
                
                # 加权求和
                output += expert_output * expert_weights.unsqueeze(-1)
        
        # 计算负载均衡损失（鼓励均匀使用所有专家）
        gate_probs = F.softmax(gate_logits, dim=-1)  # (batch_size, seq_len, num_experts)
        # 计算每个专家的平均使用率
        expert_usage = gate_probs.mean(dim=[0, 1])  # (num_experts,)
        # 负载均衡损失：鼓励所有专家使用率接近
        aux_loss = self.num_experts * torch.sum(expert_usage ** 2)
        
        return output, aux_loss


class SparseMultiheadAttention(nn.Module):
    """稀疏多头注意力：只关注局部窗口或随机屏蔽部分注意力头"""
    def __init__(self, embed_dim, num_heads, dropout=0.1, window_size=None, 
                 attention_dropout_rate=0.1):
        """
        Args:
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            dropout: Dropout率
            window_size: 局部窗口大小（None表示全局注意力）
            attention_dropout_rate: 注意力头的dropout率（随机屏蔽部分头）
        """
        super(SparseMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.attention_dropout_rate = attention_dropout_rate
        
        # 标准多头注意力
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
    
    def forward(self, query, key, value, key_padding_mask=None):
        """
        稀疏注意力：如果window_size不为None，只计算局部窗口内的注意力
        """
        batch_size, seq_len, _ = query.shape
        
        if self.window_size is not None and self.window_size < seq_len:
            # 局部窗口注意力：每个位置只关注window_size内的邻居
            # 创建掩码
            mask = torch.zeros(seq_len, seq_len, device=query.device, dtype=torch.bool)
            for i in range(seq_len):
                start = max(0, i - self.window_size // 2)
                end = min(seq_len, i + self.window_size // 2 + 1)
                mask[i, start:end] = True
            
            # 应用掩码（通过key_padding_mask）
            # 注意：这里简化实现，实际可以使用更复杂的稀疏注意力机制
            attn_output, attn_weights = self.attention(
                query, key, value, 
                key_padding_mask=key_padding_mask
            )
            
            # 应用窗口掩码到注意力权重
            if attn_weights is not None:
                attn_weights = attn_weights.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
                attn_weights = F.softmax(attn_weights, dim=-1)
        else:
            # 全局注意力
            attn_output, attn_weights = self.attention(
                query, key, value,
                key_padding_mask=key_padding_mask
            )
        
        # 随机屏蔽部分注意力头（训练时）
        if self.training and self.attention_dropout_rate > 0:
            # 随机选择要屏蔽的头
            num_heads_to_drop = int(self.num_heads * self.attention_dropout_rate)
            if num_heads_to_drop > 0:
                # 简化实现：直接应用dropout
                attn_output = F.dropout(attn_output, p=self.attention_dropout_rate, training=self.training)
        
        return attn_output, attn_weights


class MoETransformerEncoderLayer(nn.Module):
    """使用MoE和稀疏注意力的Transformer编码器层"""
    def __init__(self, d_model, nhead, dim_feedforward, num_experts=8, top_k=2,
                 dropout=0.1, window_size=None, attention_dropout_rate=0.1):
        super(MoETransformerEncoderLayer, self).__init__()
        self.d_model = d_model
        
        # 稀疏多头注意力
        self.self_attn = SparseMultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            window_size=window_size,
            attention_dropout_rate=attention_dropout_rate
        )
        
        # MoE层（替代标准FFN）
        self.moe = MoELayer(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            num_experts=num_experts,
            top_k=top_k,
            dropout=dropout
        )
        
        # 归一化层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            src: (batch_size, seq_len, d_model)
        Returns:
            output: (batch_size, seq_len, d_model)
            aux_loss: MoE的负载均衡损失
        """
        # 自注意力 + 残差连接
        attn_output, _ = self.self_attn(src, src, src, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        
        # MoE + 残差连接
        moe_output, aux_loss = self.moe(src)
        src = src + self.dropout2(moe_output)
        src = self.norm2(src)
        
        return src, aux_loss


class MoETradingTransformerWithProfit(nn.Module):
    """使用MoE和稀疏注意力的交易Transformer模型"""
    def __init__(self, input_size=46, nhead=8, num_layers=6, output_size=3, 
                 d_model=256, predict_profit=True, num_experts=4, top_k=2,
                 window_size=20, attention_dropout_rate=0.1):
        """
        Args:
            input_size: 输入特征维度
            nhead: 注意力头数
            num_layers: Transformer层数
            output_size: 输出类别数
            d_model: 模型维度
            predict_profit: 是否预测收益率
            num_experts: MoE专家数量
            top_k: 每次激活的专家数量
            window_size: 稀疏注意力的窗口大小（None表示全局）
            attention_dropout_rate: 注意力头dropout率
        """
        super(MoETradingTransformerWithProfit, self).__init__()
        self.d_model = d_model
        self.predict_profit = predict_profit
        
        # 输入投影
        self.input_projection = nn.Linear(input_size, d_model)
        
        # 位置编码
        self.register_buffer('pos_encoding', self._create_positional_encoding(1000, d_model))
        
        # MoE Transformer编码器
        encoder_layer = MoETransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            num_experts=num_experts,
            top_k=top_k,
            dropout=0.1,
            window_size=window_size,
            attention_dropout_rate=attention_dropout_rate
        )
        self.transformer = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        
        # 注意力池化
        self.attention_pool = SparseMultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=0.1,
            window_size=None,  # 池化层使用全局注意力
            attention_dropout_rate=0.0
        )
        
        self.dropout = nn.Dropout(0.3)
        
        # 动作分类头
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.GELU(),
            nn.LayerNorm(d_model//2),
            nn.Dropout(0.3),
            nn.Linear(d_model//2, d_model//4),
            nn.GELU(),
            nn.LayerNorm(d_model//4),
            nn.Dropout(0.2),
            nn.Linear(d_model//4, output_size)
        )
        
        # 收益率预测头
        if predict_profit:
            self.profit_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(d_model, d_model // 2),
                nn.LayerNorm(d_model // 2),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(d_model // 2, 1)
            )
        else:
            self.profit_head = None
        
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
            if 'weight' in name:
                if len(param.shape) >= 2:
                    if 'transformer' in name or 'attention' in name:
                        nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='relu')
                    else:
                        nn.init.xavier_uniform_(param, gain=1.0)
                elif len(param.shape) == 1:
                    nn.init.constant_(param, 1.0)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_size)
        Returns:
            action_logits: (batch_size, output_size)
            profit: (batch_size, 1) 或 None
            aux_loss: MoE的负载均衡损失总和
        """
        batch_size, seq_len = x.size(0), x.size(1)
        
        # 输入投影
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # 添加位置编码
        if seq_len <= self.pos_encoding.size(1):
            pos_enc = self.pos_encoding[:, :seq_len, :]
        else:
            extended_pe = self._create_positional_encoding(seq_len, self.d_model).to(x.device)
            pos_enc = extended_pe[:, :seq_len, :]
        
        x = x + pos_enc
        
        # Transformer编码（MoE层）
        total_aux_loss = 0
        for layer in self.transformer:
            x, aux_loss = layer(x)
            total_aux_loss = total_aux_loss + aux_loss
        
        # 注意力池化
        attn_out, _ = self.attention_pool(
            query=x[:, -1:, :],  # (batch, 1, d_model)
            key=x,                # (batch, seq_len, d_model)
            value=x               # (batch, seq_len, d_model)
        )
        
        pooled_out = attn_out.squeeze(1)  # (batch, d_model)
        pooled_out = self.dropout(pooled_out)
        
        # 动作分类
        action_logits = self.action_head(pooled_out)
        
        outputs = [action_logits]
        
        # 收益率预测
        if self.predict_profit and self.profit_head is not None:
            profit = self.profit_head(pooled_out)
            outputs.append(profit)
        
        # 返回输出和辅助损失
        if len(outputs) > 1:
            return (tuple(outputs), total_aux_loss)
        else:
            return (outputs[0], total_aux_loss)
