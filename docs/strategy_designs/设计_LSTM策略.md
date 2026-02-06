# LSTM 策略设计文档

> 本文档描述基于 LSTM 的时序预测交易策略的**算法设计**、**训练过程设计**与**数据分析设计**。实现见 `src/strategies/llm_strategy.py`（LSTM 以 LLMTradingStrategy + mode=hybrid 使用 TradingLSTM）。

---

## 1. 策略概述

LSTM 策略使用 **LSTM 编码 + 全连接头** 的时序模型，输入多时间尺度特征，输出**动作分类**（不操作/买入/卖出），可选**收益率预测**与**网格调整系数**。与 MoE 同属“大模型/时序预测”族，常作为对比基线或备选模型。

---

## 2. LSTM 在金融量化中的实现原理

LSTM 在金融量化中的本质是：用**门控循环单元**（遗忘门、输入门、输出门）与**细胞状态**对金融时序做**顺序编码**，解决简单 RNN 的长期依赖梯度消失问题，同时参数量与计算量通常小于 Transformer，在小样本、中短序列场景下更易收敛。核心可拆解为三层。

### 2.1 金融数据的时序适配与编码

- **输入层**：将原始金融数据（K 线开高低收量、多时间尺度指标、可选 Tick）组织为**时序序列**，每个时间步对应一个特征向量（默认 46 维）。与 Transformer 不同，LSTM **无需位置编码**，因其按时间步顺序依次处理，时间顺序由输入顺序天然表达。
- **特征归一化**：与 Transformer 一致，需在进入模型前对多量纲特征做 Z-score、Min-Max 等归一化，避免数值范围差异影响门控与隐藏状态。

### 2.2 门控与长短期记忆的核心作用

- **遗忘门 / 输入门 / 输出门**：控制上一时刻细胞状态有多少被保留、当前输入有多少写入新状态、当前隐藏状态有多少输出。使模型能学习「何时记住长期趋势、何时关注近期波动」，适合金融中的趋势延续与反转模式。
- **顺序建模**：对单标的，LSTM 沿时间步依次更新，可捕捉「量价异动→后续涨跌」「波动率放大→方向选择」等中短期依赖；与 Transformer 的全局注意力相比，更偏**局部到中程**的依赖，长程依赖依赖层数与隐藏维度，通常不如 Transformer 直接。
- **多层 LSTM**：堆叠多层（如 3 层）可增强表示能力，层间 dropout 缓解过拟合。

### 2.3 下游任务适配

- **预测类任务**：取**最后时间步**的隐藏状态作为整段序列的表示，接全连接得到动作 logits（3 类）与可选收益标量、网格调整系数（范围 [0.8, 1.2]）。
- **策略信号生成**：与 MoE 策略共用基类逻辑，由动作置信度与预测收益/网格系数，在风控约束下生成买/卖/观望信号。

---

## 3. 模型架构（本策略实现）

本策略对应实现为 `TradingLSTM`（`src/strategies/llm_strategy.py`），数据流与层级如下。

| 层级 | 模块 | 输入 → 输出 | 说明 |
|------|------|-------------|------|
| 输入 | — | (B, L, 46) 或 (B, 46) | 若 2D 则 unsqueeze 为 (B, 1, 46) |
| 编码 | `nn.LSTM` | (B, L, input_size) → (B, L, hidden_size) | 多层、batch_first=True、dropout=0.2 |
| 表示 | 取最后时间步 | (B, L, hidden_size) → (B, hidden_size) | `out[:, -1, :]` |
| 正则 | `Dropout(0.3)` | (B, hidden_size) → (B, hidden_size) | 降低过拟合 |
| 输出 | `action_head` | (B, hidden_size) → (B, 3) | 单层 Linear → 动作 logits |
| 输出 | `profit_head` | (B, hidden_size) → (B, 1) | Linear+LayerNorm+ReLU+Dropout 多层 → 收益率标量 |
| 输出 | `grid_adjustment_head` | (B, hidden_size) → (B, 1) | Linear + sigoid×0.4+0.8 → [0.8, 1.2] |

- **维度**：`input_size=46`，`hidden_size=128`，`num_layers=3`，`output_size=3`。权重采用 Xavier 初始化。

---

## 4. 预测原理（从输入到交易信号）

1. **输入**：当前时刻及历史共 `seq_length` 个时间步的多维特征（46 维），形状 (1, seq_length, 46) 或 (B, seq_length, 46)。
2. **编码**：LSTM 按时间步顺序前向，得到 (B, L, hidden_size)；取最后时间步 (B, hidden_size)，再经 Dropout。
3. **输出**：`action_head` → 3 维 logits，softmax/argmax 得动作；`profit_head` → 标量预测收益；`grid_adjustment_head` → 网格调整系数（若启用）。
4. **信号生成**：结合动作置信度与预测收益，在仓位、止损止盈、日亏损上限等风控下决定是否发出买入/卖出；否则观望。与 MoE 共用基类策略逻辑。

---

## 5. 模型可识别的金融数据核心特征

LSTM 通过门控与顺序编码，可侧重捕捉以下类型的特征（部分由模型自动学习）：

- **量价类**：量价匹配/背离、趋势延续/反转（均线、RSI、BOLL 等）、波动率变化。因顺序处理，对**近期与中短期**窗口更敏感，适合分钟级到日内的模式。
- **局部到中程依赖**：如「过去 20～50 根 K 线的形态 → 当前方向」；长于数百步的依赖依赖层数与容量，通常弱于 Transformer。
- **与 Transformer 的差异**：LSTM 无显式跨时间注意力权重，可解释性更多依赖事后分析隐藏状态或梯度；优势在于参数量小、训练与推理资源占用低、小数据更易收敛。见 [Transformer_vs_LSTM理论分析](../Transformer_vs_LSTM理论分析.md)。
- **落地注意**：金融非平稳、信噪比低，需正则（Dropout、权重衰减）、早停、严格按时间划分验证集；避免过拟合与数据挖掘陷阱。

---

## 6. 优势与落地注意事项

- **优势**：参数量与计算量相对较小；顺序计算对中短序列友好；小样本下易收敛；无需位置编码，实现简单。
- **注意事项**：长序列（如 500 步）下长程依赖弱于 Transformer；需保证训练与推理的 seq_length、input_size 一致；单样本推理时 LayerNorm 使用 running stats（本实现已处理）。

---

## 7. 算法设计（小结与损失）

### 7.1 模型结构小结

- **输入**：多时间尺度特征序列，默认 46 维，形状 (batch, seq_len, input_size)；若为 2D 则自动 unsqueeze。
- **编码**：多层 LSTM（batch_first=True，dropout=0.2），取**最后时间步**隐藏状态作为序列表示。
- **输出头**：动作头（3 类 logits）；收益头（LayerNorm + 全连接 → 标量）；网格调整头（范围 [0.8, 1.2]）。

### 7.2 与 Transformer/MoE 的差异

- LSTM 顺序编码、无自注意力，参数量较小，小数据集更易收敛。见 [Transformer_vs_LSTM理论分析](../Transformer_vs_LSTM理论分析.md)。

### 7.3 损失与信号生成

- **损失**：动作交叉熵（可选类别权重）+ 收益 MSE + 网格调整 MSE。
- **推理**：由预测动作与可选收益/网格系数，在风控约束下生成交易信号（共用 LLMTradingStrategy 基类）。

---

## 8. 主要参数含义

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

## 9. 训练过程设计

- **数据划分**：按时间 80%/20% 训练/验证，不打乱顺序。
- **流程**：从 `training_data_multitimeframe_*.csv` 滑窗得到 (X, y_action, y_profit)；多 epoch + Adam；早停按验证准确率/损失；保存最佳为 `best_lstm_improved.pth`。
- **脚本**：`scripts/train_multiple_models_comparison.py` 中 `prepare_data_for_model(df, model_type='lstm', seq_length=...)`；策略工厂以 `lstm` 注册。

## 10. 数据分析设计

- **特征来源**：与 MoE 一致，多时间尺度 + 可选 Tick（见 [训练输入数据说明](../训练输入数据说明.md)、[多时间尺度设计方案](../多时间尺度设计方案.md)）。
- **标签**：未来 N 根 K 线涨跌与阈值 → 动作 0/1/2；对应时段实际收益率 → 收益标签；网格调整可选 [0.8, 1.2]。
- **验证**：严格按时间切分验证集，用于早停与模型选择；与 MoE/Transformer 同脚本对比。

---

## 11. 相关文档

- [Transformer_vs_LSTM理论分析](../Transformer_vs_LSTM理论分析.md) — LSTM 与 Transformer 对比与适用场景
- [两种策略模式设计](../两种策略模式设计.md) — 计算模式 vs 大模型识别模式
- [训练输入数据说明](../训练输入数据说明.md) — 训练数据格式
- [多时间尺度设计方案](../多时间尺度设计方案.md) — 特征与数据
