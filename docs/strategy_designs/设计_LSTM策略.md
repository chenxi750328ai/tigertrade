# LSTM 策略设计文档

> 本文档描述基于 LSTM 的时序预测交易策略的算法原理、参数含义与训练流程。实现见 `src/strategies/llm_strategy.py`（LSTM 以 LLMTradingStrategy + mode=hybrid 使用 TradingLSTM）。

---

## 1. 策略概述

LSTM 策略使用** LSTM 编码 + 全连接头**的时序模型，输入多时间尺度特征，输出**动作分类**（不操作/买入/卖出），可选**收益率预测**与**网格调整系数**。与 MoE 同属“大模型/时序预测”族，常作为对比基线或备选模型。

---

## 2. 算法原理

### 2.1 模型结构

- **输入**：多时间尺度特征（默认 46 维），序列形式输入 LSTM。
- **编码**：多层 LSTM（batch_first=True，dropout=0.2），取最后时间步隐藏状态。
- **输出头**：
  - **动作头**：3 类分类（不操作、买入、卖出）。
  - **收益头**（predict_profit=True）：回归预测收益率，含 LayerNorm + 多层全连接。
  - **网格调整头**（predict_grid_adjustment=True）：回归，范围 [0.8, 1.2]。

### 2.2 与 Transformer/MoE 的差异

- LSTM 顺序编码，无自注意力；参数量相对较小，小数据集上易收敛。
- 理论对比与“为何小模型有时更好”见：[Transformer_vs_LSTM理论分析](../Transformer_vs_LSTM理论分析.md)。

### 2.3 信号生成

- 由预测动作与可选收益/网格系数，在风控约束下生成交易信号；逻辑与 MoE 策略类似（通过 LLMTradingStrategy 基类）。

---

## 3. 主要参数含义

| 参数 | 含义 | 典型值/说明 |
|------|------|-------------|
| input_size | 输入特征维度 | 46 |
| hidden_size | LSTM 隐藏层维度 | 128 |
| num_layers | LSTM 层数 | 3 |
| output_size | 动作类别数 | 3 |
| predict_profit | 是否预测收益率 | True |
| predict_grid_adjustment | 是否预测网格调整系数 | True |
| mode | 运行模式 | hybrid（计算+模型） |

---

## 4. 训练过程

1. **数据**：与 MoE 相同的多时间尺度历史数据与标签。
2. **脚本**：`scripts/train_multiple_models_comparison.py` 等，训练 LSTM 并保存为 `best_lstm_improved.pth` 或项目配置的路径。
3. **注册**：策略工厂中以 `lstm` 注册，对应 LLMTradingStrategy 的 LSTM 实现。

---

## 5. 相关文档

- [Transformer_vs_LSTM理论分析](../Transformer_vs_LSTM理论分析.md) — LSTM 与 Transformer 对比与适用场景
- [两种策略模式设计](../两种策略模式设计.md) — 计算模式 vs 大模型识别模式
- [训练输入数据说明](../训练输入数据说明.md) — 训练数据格式
