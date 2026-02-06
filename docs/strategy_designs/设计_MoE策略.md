# MoE Transformer 策略设计文档

> 本文档描述 MoE（混合专家）Transformer 交易策略的**算法设计**、**训练过程设计**与**数据分析设计**。实现见 `src/strategies/moe_strategy.py`、`src/strategies/moe_transformer.py`。

---

## 1. 策略概述

MoE Transformer 策略使用**混合专家 + Transformer** 时序模型，对多时间尺度特征进行编码，输出**动作分类**（不操作/买入/卖出）与**收益率预测**，在满足风控条件下生成交易信号。为 DEMO/实盘主推策略之一。

---

## 2. 算法设计

### 2.1 模型结构

- **输入**：多时间尺度特征序列（默认 46 维），序列长度 500（与训练一致）。形状 `(batch, seq_len=500, input_size=46)`。
- **编码**：Transformer 编码器 + **MoE 层**（部分 FFN 替换为多个专家 + 门控，Top-K 稀疏激活）。
- **输出**：
  - **动作头**：线性层 → 3 类 logits（不操作、买入、卖出），推理时 argmax 或结合阈值得动作。
  - **收益头**（predict_profit=True）：回归预测未来一段收益率的标量，用于过滤低收益信号。

### 2.2 MoE（混合专家）与稀疏注意力

- **MoE**：将部分 Transformer 层的 FFN 替换为 N 个“专家”FFN + 门控网络；门控按当前隐藏状态输出权重，仅 Top-K 个专家参与前向，实现稀疏激活、参数量大但计算可控、利于减轻过拟合。
- **稀疏注意力**：可选限制注意力窗口或使用稀疏头，减少长序列计算与过拟合。详见 [MoE和稀疏注意力方案说明](../MoE和稀疏注意力方案说明.md)。

### 2.3 损失与优化目标

- **动作**：交叉熵损失（可选类别权重以应对不平衡）。
- **收益**：MSE 或 Smooth L1 对预测收益与标签收益。
- 总损失可为 `loss_action + λ * loss_profit`，λ 为超参。

### 2.4 信号生成（推理）

- 输入当前序列 → 得到动作 logits 与预测收益。
- 取动作置信度（如 softmax 最大概率）与预测收益，在满足仓位、风控（止损止盈、日亏损上限等）下决定是否发出买入/卖出；否则观望。

---

## 3. 训练过程设计

### 3.1 数据划分

- **训练集 / 验证集**：按时间划分（如 80% 训练、20% 验证），避免未来信息泄露；不做随机打乱。
- **序列构造**：每个样本为一段连续时间步的特征序列 + 对应标签（下一段动作与收益）。

### 3.2 训练流程

1. **数据加载**：从多时间尺度特征文件（如 `training_data_multitimeframe_*.csv` 或 merged）按 seq_length 滑窗构造 (X, y_action, y_profit)。
2. **Epoch 与 batch**：多 epoch 遍历训练集；batch_size 根据显存设定（如 32、64）。
3. **优化器**：常用 Adam，学习率可设 1e-4～1e-3，可选学习率衰减或 warmup。
4. **早停**：根据验证集动作准确率或验证损失，若干 epoch 无提升则停止，并恢复最佳权重。
5. **保存**：最佳模型保存为 `best_moe_transformer.pth`（路径可由配置 `model_path` 指定），可同时保存 optimizer 与 epoch 便于恢复。

### 3.3 脚本与入口

- **多模型对比训练**：`scripts/train_multiple_models_comparison.py`，同时训练 LSTM、Transformer、Enhanced Transformer、MoE 等，便于对比。
- MoE 单独训练逻辑在同一脚本或项目内 MoE 相关模块中调用 `MoETradingTransformerWithProfit` 与上述数据接口。

### 3.4 与 LSTM 对比

- 见 [Transformer_vs_LSTM理论分析](../Transformer_vs_LSTM理论分析.md)：Transformer 擅长长程依赖，LSTM 参数量小、小数据更易收敛。

---

## 4. 数据分析设计

### 4.1 特征来源

- **多时间尺度**：1 分钟、5 分钟（及可选 1h/日线）K 线衍生指标（价格、ATR、RSI、BOLL、成交量等），对齐到同一时间轴。详见 [多时间尺度设计方案](../多时间尺度设计方案.md)。
- **维度**：默认 46 维（与 `input_size` 一致），包含价、量、技术指标与可选 Tick 相关字段。见 [训练输入数据说明](../训练输入数据说明.md)。

### 4.2 标签定义

- **动作标签**：根据未来一段时间（如 120 根 K 线）内的涨跌与阈值，定义为 0=不操作、1=买入、2=卖出；考虑持仓状态（当前多/空）与收益阈值、最小价差，避免噪音标签。
- **收益标签**：与动作一致时段内的实际收益率（如做多取区间最大涨幅，做空取区间最大跌幅），用于收益头监督。

### 4.3 验证集与评估

- **验证集**：严格按时间切分，仅用验证集做早停与模型选择，不参与训练。
- **评估指标**：动作准确率、混淆矩阵；收益头 MSE/MAE；可选回测夏普、胜率等（需单独回测脚本）。

### 4.4 数据管道

- 原始 K 线 / Tick → 特征计算与多时间尺度对齐 → 写入 `training_data_multitimeframe_*.csv` 或 merged 文件。
- 训练脚本读取该文件，按 seq_length 滑窗生成 (X, y_action, y_profit)，再 DataLoader 喂给模型。

---

## 5. 主要参数含义

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
| seq_length | 推理/训练序列长度 | 500（需一致） |
| window_size | 局部窗口等 | 20 |
| attention_dropout_rate | 注意力 dropout | 0.1 |

---

## 6. 相关文档

- [MoE和稀疏注意力方案说明](../MoE和稀疏注意力方案说明.md) — MoE 与稀疏注意力原理与实现
- [多时间尺度设计方案](../多时间尺度设计方案.md) — 特征与数据
- [训练输入数据说明](../训练输入数据说明.md) — 训练数据格式
- [Transformer_vs_LSTM理论分析](../Transformer_vs_LSTM理论分析.md) — 与 LSTM 对比
