"""
LoRA (Low-Rank Adaptation) - 参数高效微调
只训练少量新增参数（0.1%-5%），冻结主权重，大幅降低过拟合风险
"""
import torch
import torch.nn as nn
import numpy as np


class LoRALayer(nn.Module):
    """LoRA层：在原有线性层基础上添加低秩矩阵"""
    def __init__(self, original_layer, rank=8, alpha=16, dropout=0.1):
        """
        Args:
            original_layer: 原始的nn.Linear层
            rank: LoRA的秩（低秩矩阵的维度）
            alpha: LoRA的缩放因子（通常alpha = 2 * rank）
            dropout: LoRA的dropout率
        """
        super(LoRALayer, self).__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # 冻结原始层
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        # LoRA的A和B矩阵
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # 原始输出
        original_out = self.original_layer(x)
        
        # LoRA输出
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        
        return original_out + lora_out


def apply_lora_to_transformer(model, rank=8, alpha=16, target_modules=None):
    """
    将LoRA应用到Transformer模型
    
    Args:
        model: Transformer模型
        rank: LoRA的秩
        alpha: LoRA的缩放因子
        target_modules: 要应用LoRA的模块名称列表（如['q_proj', 'v_proj']）
    """
    if target_modules is None:
        # 默认应用到注意力层的query和value投影
        target_modules = ['q_proj', 'v_proj']
    
    lora_params = 0
    total_params = sum(p.numel() for p in model.parameters())
    
    # 遍历模型的所有模块
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 检查是否是目标模块
            if any(target in name for target in target_modules):
                # 创建LoRA层替换原始层
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                parent_module = model
                for part in parent_name.split('.'):
                    if part:
                        parent_module = getattr(parent_module, part)
                
                lora_layer = LoRALayer(module, rank=rank, alpha=alpha)
                setattr(parent_module, child_name, lora_layer)
                
                lora_params += rank * (module.in_features + module.out_features)
    
    trainable_params = lora_params
    print(f"LoRA参数统计:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数（LoRA）: {trainable_params:,}")
    print(f"  可训练参数占比: {trainable_params/total_params*100:.3f}%")
    
    return model
