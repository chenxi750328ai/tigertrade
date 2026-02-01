"""
MoE Transformer交易策略
使用最佳模型（MoE Transformer）进行交易预测
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import threading
from typing import Tuple, Optional, Dict, Any
import sys
sys.path.insert(0, '/home/cx/tigertrade')

from src.strategies.moe_transformer import MoETradingTransformerWithProfit
from src.strategies.llm_strategy import LLMTradingStrategy
from src.strategies.base_strategy import BaseTradingStrategy


class MoETradingStrategy(BaseTradingStrategy):
    """MoE Transformer交易策略（基于LLMTradingStrategy，但使用MoE Transformer模型）"""
    
    def __init__(self, data_dir='/home/cx/trading_data', model_path=None, seq_length=500):
        """
        初始化MoE Transformer交易策略
        
        Args:
            data_dir: 数据目录
            model_path: 模型路径（默认使用最佳MoE模型）
            seq_length: 序列长度（默认500，与训练时一致）
        """
        # 检查GPU可用性
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"Using device: {self.device}")
        else:
            self.device = torch.device('cpu')
            print(f"Using device: {self.device}")
        
        self.data_dir = data_dir
        self._seq_length = seq_length
        self._historical_data = None
        
        # 初始化MoE Transformer模型（需要先检查保存的模型配置）
        input_size = 46  # 多时间尺度特征维度
        
        # 尝试加载模型以获取配置
        model_config = {
            'input_size': input_size,
            'nhead': 8,
            'num_layers': 8,  # 从错误信息看是8层
            'output_size': 3,
            'd_model': 512,  # 从错误信息看是512
            'predict_profit': True,
            'num_experts': 8,  # 从错误信息看是8个专家
            'top_k': 2,
            'window_size': 20,
            'attention_dropout_rate': 0.1
        }
        
        # 如果提供了模型路径，尝试读取配置
        if model_path is None:
            model_path = os.path.join(data_dir, 'best_moe_transformer.pth')
        
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                if isinstance(checkpoint, dict):
                    state_dict = checkpoint.get('model_state_dict', checkpoint)
                else:
                    state_dict = checkpoint
                
                # 从state_dict推断配置
                if 'input_projection.weight' in state_dict:
                    model_config['d_model'] = state_dict['input_projection.weight'].shape[0]
                if 'transformer.0.moe.gate.weight' in state_dict:
                    model_config['num_experts'] = state_dict['transformer.0.moe.gate.weight'].shape[0]
                # 计算层数
                layer_keys = [k for k in state_dict.keys() if k.startswith('transformer.') and '.' in k]
                if layer_keys:
                    max_layer = max([int(k.split('.')[1]) for k in layer_keys if k.split('.')[1].isdigit()])
                    model_config['num_layers'] = max_layer + 1
                
                print(f"📊 从模型文件推断配置: d_model={model_config['d_model']}, num_layers={model_config['num_layers']}, num_experts={model_config['num_experts']}")
            except Exception as e:
                print(f"⚠️ 无法读取模型配置，使用默认配置: {e}")
        
        self.model = MoETradingTransformerWithProfit(
            input_size=model_config['input_size'],
            nhead=model_config['nhead'],
            num_layers=model_config['num_layers'],
            output_size=model_config['output_size'],
            d_model=model_config['d_model'],
            predict_profit=model_config['predict_profit'],
            num_experts=model_config['num_experts'],
            top_k=model_config['top_k'],
            window_size=model_config['window_size'],
            attention_dropout_rate=model_config['attention_dropout_rate']
        ).to(self.device)
        
        # 模型锁
        self.model_lock = threading.Lock()
        
        # 加载模型
        if model_path is None:
            model_path = os.path.join(data_dir, 'best_moe_transformer.pth')
        
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                # 处理不同的checkpoint格式
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        self.model.load_state_dict(checkpoint)
                else:
                    self.model.load_state_dict(checkpoint)
                print(f"✅ 从 {model_path} 加载MoE Transformer模型成功")
            except Exception as e:
                print(f"❌ 加载MoE Transformer模型失败: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"⚠️ 模型文件不存在: {model_path}，使用初始模型")
    
    @property
    def seq_length(self) -> int:
        """返回策略需要的序列长度"""
        return self._seq_length
    
    @property
    def strategy_name(self) -> str:
        """返回策略名称"""
        return "MoE Transformer"
    
    def prepare_features(self, row):
        """准备特征（与LLMTradingStrategy一致）"""
        # 使用LLMTradingStrategy的特征准备逻辑
        base_strategy = LLMTradingStrategy(mode='hybrid', predict_profit=True)
        return base_strategy.prepare_features(row)
    
    def prepare_sequence_features(self, df, current_idx, seq_length):
        """准备历史序列特征"""
        start_idx = max(0, current_idx - seq_length + 1)
        sequence_df = df.iloc[start_idx:current_idx+1]
        
        sequences = []
        for _, row in sequence_df.iterrows():
            features = self.prepare_features(row)
            sequences.append(features)
        
        # 如果序列不足seq_length，用第一个值填充
        feature_size = 46
        while len(sequences) < seq_length:
            if sequences:
                sequences.insert(0, sequences[0])
            else:
                sequences.insert(0, [0.0] * feature_size)
        
        return np.array(sequences, dtype=np.float32)
    
    def predict_action(self, current_data: Dict[str, Any], historical_data: Optional[pd.DataFrame] = None) -> Tuple[int, float, Optional[float]]:
        """
        使用MoE Transformer模型预测交易动作
        
        Args:
            current_data: 当前数据字典
            historical_data: 历史数据DataFrame（可选）
        
        Returns:
            (action, confidence, profit_prediction)
            - action: 0=不操作, 1=买入, 2=卖出
            - confidence: 置信度 [0, 1]
            - profit_prediction: 预测收益率（可选）
        """
        with self.model_lock:
            try:
                # 准备序列数据
                if historical_data is not None and len(historical_data) >= self._seq_length:
                    # 使用历史数据构建序列
                    sequence = self.prepare_sequence_features(
                        historical_data, 
                        len(historical_data) - 1, 
                        self._seq_length
                    )
                else:
                    # 如果没有历史数据，使用当前数据重复填充
                    features = self.prepare_features(current_data)
                    sequence = np.tile(features, (self._seq_length, 1))
                
                # 转换为tensor: (batch_size=1, seq_length, feature_size)
                input_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # 模型预测
                with torch.no_grad():
                    self.model.eval()
                    model_output = self.model(input_tensor)
                    
                    # MoE模型返回((outputs), aux_loss)
                    if isinstance(model_output, tuple):
                        outputs, aux_loss = model_output
                    else:
                        outputs = model_output
                    
                    # 处理输出
                    if isinstance(outputs, tuple):
                        action_logits, profit_pred = outputs
                    else:
                        action_logits = outputs
                        profit_pred = None
                    
                    # 动作预测
                    action_probs = torch.softmax(action_logits, dim=-1)
                    action = torch.argmax(action_probs, dim=-1).cpu().numpy()[0]
                    confidence = action_probs[0][action].cpu().numpy()
                    
                    # 收益率预测（如果启用）
                    if profit_pred is not None:
                        profit_value = float(profit_pred.cpu().numpy()[0])
                        # 应用ReLU和clamp（与训练时推理一致）
                        profit_value = max(0.0, min(profit_value, 0.3))
                        return int(action), float(confidence), profit_value
                    else:
                        return int(action), float(confidence)
            
            except Exception as e:
                print(f"❌ MoE Transformer预测错误: {e}")
                import traceback
                traceback.print_exc()
                return 0, 0.0  # 默认：不操作，置信度0
