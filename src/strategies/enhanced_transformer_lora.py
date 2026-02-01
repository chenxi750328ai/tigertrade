"""
使用LoRA的Enhanced Transformer模型
保持大容量，但只训练少量LoRA参数
"""
import torch
import torch.nn as nn
import numpy as np
from .enhanced_transformer_with_profit import EnhancedTradingTransformerWithProfit


class LoRALinear(nn.Module):
    """LoRA线性层"""
    def __init__(self, linear_layer, rank=8, alpha=16, dropout=0.1):
        super(LoRALinear, self).__init__()
        self.linear = linear_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # 冻结原始层
        for param in self.linear.parameters():
            param.requires_grad = False
        
        # LoRA矩阵
        in_features = linear_layer.in_features
        out_features = linear_layer.out_features
        
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        original_out = self.linear(x)
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        return original_out + lora_out


class EnhancedTransformerWithLoRA(nn.Module):
    """使用LoRA的Enhanced Transformer"""
    def __init__(self, base_model: EnhancedTradingTransformerWithProfit, 
                 lora_rank=8, lora_alpha=16, apply_to_attention=True):
        super(EnhancedTransformerWithLoRA, self).__init__()
        self.base_model = base_model
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        
        # 冻结基础模型的所有参数
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # 应用LoRA到注意力层
        if apply_to_attention:
            self._apply_lora_to_attention()
        
        # 分类头和收益率头保持可训练（或也应用LoRA）
        # 这里我们保持它们可训练，因为它们相对较小
    
    def _apply_lora_to_attention(self):
        """将LoRA应用到Transformer的注意力层"""
        # TransformerEncoderLayer中的线性层名称：
        # - self_attn.in_proj_weight (qkv投影)
        # - self_attn.out_proj (输出投影)
        # - linear1, linear2 (FFN层)
        
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Linear):
                # 应用到注意力相关的线性层
                if 'self_attn' in name or 'attention_pool' in name:
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    
                    parent = self.base_model
                    for part in parent_name.split('.'):
                        if part:
                            parent = getattr(parent, part)
                    
                    lora_layer = LoRALinear(module, rank=self.lora_rank, alpha=self.lora_alpha)
                    setattr(parent, child_name, lora_layer)
    
    def forward(self, x):
        return self.base_model(x)
    
    def get_trainable_params(self):
        """获取可训练参数数量"""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.base_model.parameters())
        return trainable, total
