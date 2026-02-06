# LSTM 策略设计文档

> 本文档描述基于 LSTM 的时序预测交易策略的**算法设计**、**训练过程设计**与**数据分析设计**。实现见 `src/strategies/llm_strategy.py`（LSTM 以 LLMTradingStrategy + mode=hybrid 使用 TradingLSTM）。

---

## 1. 策略概述

LSTM 策略使用** LSTM 编码 + 全连接头**的时序模型，输入多时间尺度特征，输出**动作分类**（不操作/买入/卖出），可选**收益率预测**与**网格调整系数**。与 MoE 同属“大模型/时序预测”族，常作为对比基线或备选模型。

---

## 2. 算法设计

### 2.1 模型结构

- **输入**：多时间尺度特征序列，默认 46 维，形状 (batch, seq_len, input_size)；若为 2D 则自动 unsqueeze。
- **编码**：多层 LSTM（batch_first=True，dropout=0.2），取**最后时间步**隐藏状态作为序列表示。
- **输出头**：动作头（3 类 logits）；收益头（LayerNorm + 全连接 → 标量）；网格调整头（范围 [0.8, 1.2]）。

### 2.2 与 Transformer/MoE 的差异

- LSTM 顺序编码、无自注意力，参数量较小，小数据集更易收敛。见 [Transformer_vs_LSTM理论分析](../Transformer_vs_LSTM理论分析.md)。

### 2.3 损失与信号生成

- **损失**：动作交叉熵（可选类别权重）+ 收益 MSE + 网格调整 MSE。
- **推理**：由预测动作与可选收益/网格系数，在风控约束下生成交易信号（共用 LLMTradingStrategy 基类）。

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

## 4. 训练过程设计

- **数据划分**：按时间 80%/20% 训练/验证，不打乱顺序。
- **流程**：从 `training_data_multitimeframe_*.csv` 滑窗得到 (X, y_action, y_profit)；多 epoch + Adam；早停按验证准确率/损失；保存最佳为 `best_lstm_improved.pth`。
- **脚本**：`scripts/train_multiple_models_comparison.py` 中 `prepare_data_for_model(df, model_type='lstm', seq_length=...)`；策略工厂以 `lstm` 注册。

## 5. 数据分析设计

- **特征来源**：与 MoE 一致，多时间尺度 + 可选 Tick（见 [训练输入数据说明](../训练输入数据说明.md)、[多时间尺度设计方案](../多时间尺度设计方案.md)）。
- **标签**：未来 N 根 K 线涨跌与阈值 → 动作 0/1/2；对应时段实际收益率 → 收益标签；网格调整可选 [0.8, 1.2]。
- **验证**：严格按时间切分验证集，用于早停与模型选择；与 MoE/Transformer 同脚本对比。

---

## 6. 相关文档

- [Transformer_vs_LSTM理论分析](../Transformer_vs_LSTM理论分析.md) — LSTM 与 Transformer 对比与适用场景
- [两种策略模式设计](../两种策略模式设计.md) — 计算模式 vs 大模型识别模式
- [训练输入数据说明](../训练输入数据说明.md) — 训练数据格式
- [多时间尺度设计方案](../多时间尺度设计方案.md) — 特征与数据
