# MoE Transformer 策略设计文档

> 本文档描述 MoE（混合专家）Transformer 交易策略的算法原理、参数含义与训练流程。实现见 `src/strategies/moe_strategy.py`、`src/strategies/moe_transformer.py`。

---

## 1. 策略概述

MoE Transformer 策略使用**混合专家 + Transformer** 时序模型，对多时间尺度特征进行编码，输出**动作分类**（不操作/买入/卖出）与**收益率预测**，在满足风控条件下生成交易信号。为 DEMO/实盘主推策略之一。

---

## 2. 算法原理

### 2.1 模型结构

- **输入**：多时间尺度特征序列（默认 46 维），序列长度 500（与训练一致）。
- **编码**：Transformer 编码器 + **MoE 层**（每层含多个专家 FFN + 门控，Top-K 稀疏激活）。
- **输出**：
  - 动作头：3 类分类（不操作、买入、卖出）。
  - 收益头：回归预测收益率（predict_profit=True 时）。

### 2.2 MoE（混合专家）与稀疏注意力

- **MoE**：将部分层替换为多个“专家”FFN，门控网络按输入选择 Top-K 个专家参与计算，实现稀疏激活、降低过拟合。
- **稀疏注意力**：可选限制注意力范围或稀疏头，减少计算与过拟合。
- 详细原理与实现见：[MoE和稀疏注意力方案说明](../MoE和稀疏注意力方案说明.md)。

### 2.3 信号生成

- 结合**动作置信度**与**预测收益**，在满足仓位与风控约束下发出买入/卖出；否则观望。

---

## 3. 主要参数含义

| 参数 | 含义 | 典型值/说明 |
|------|------|-------------|
| input_size | 输入特征维度 | 46（多时间尺度特征） |
| d_model | Transformer 隐藏维度 | 512 |
| num_layers | Transformer 层数 | 8 |
| nhead | 注意力头数 | 8 |
| num_experts | MoE 专家数量 | 8 |
| top_k | 每层激活的专家数 | 2 |
| output_size | 动作类别数 | 3 |
| predict_profit | 是否预测收益率 | True |
| seq_length | 推理序列长度 | 500（需与训练一致） |
| window_size | 局部窗口等 | 20 |
| attention_dropout_rate | 注意力 dropout | 0.1 |

---

## 4. 训练过程

1. **数据**：历史 K 线 + 多时间尺度特征（1m/5m 等），标签为下一阶段涨跌/收益与动作。
2. **脚本**：多模型训练见 `scripts/train_multiple_models_comparison.py`；MoE 单独训练见项目内 MoE 相关脚本。
3. **保存**：最佳模型保存为 `best_moe_transformer.pth`（路径可由 `model_path` 指定）。
4. **与 LSTM 对比**：见 [Transformer_vs_LSTM理论分析](../Transformer_vs_LSTM理论分析.md)。

---

## 5. 相关文档

- [MoE和稀疏注意力方案说明](../MoE和稀疏注意力方案说明.md) — MoE 与稀疏注意力原理与实现
- [多时间尺度设计方案](../多时间尺度设计方案.md) — 特征与数据
- [训练输入数据说明](../训练输入数据说明.md) — 训练数据格式
