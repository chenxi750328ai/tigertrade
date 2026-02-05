# 网格策略与 BOLL 网格策略设计文档

> 本文档描述基于价格区间与布林带的网格交易策略的算法原理、参数含义及与时段自适应的关系。实现见 `src/tiger1.py`（`grid_trading_strategy`、`grid_trading_strategy_pro1`、`boll1m_grid_strategy`）及 `src/strategies/time_period_strategy.py`。

---

## 1. 策略概述

- **网格策略**：以 5 分钟 Bollinger Bands 中轨/上轨/下轨（或时段自适应区间）为 **grid_lower / grid_upper**，价格接近下轨且 1 分钟 RSI 低位时考虑买入，接近上轨或触发止盈/止损时卖出。
- **BOLL 网格策略**：网格族的 1 分钟侧重变体（boll1m_grid_strategy），同样以 5 分钟布林带为边界，结合 1 分钟 K 线与 RSI 判断入场与出场。

两者共用一套参数体系与时段自适应逻辑，区别主要在调用入口与部分阈值细节。

---

## 2. 算法原理

### 2.1 数据流

1. **行情**：1 分钟与 5 分钟 K 线（`get_kline_data`），时区与格式标准化，满足最少 K 线数（MIN_KLINES）。
2. **指标**：5 分钟 BOLL、ATR；1 分钟 RSI、成交量（`calculate_indicators`）。
3. **网格边界**：以 5 分钟 Boll 中轨/上轨/下轨为基准，结合 ATR 微调；BOLL 发散或 ATR 放大时动态调整以减少频繁交易。

### 2.2 开仓逻辑（买入）

- 价格接近或低于 `grid_lower`，且 1 分钟 RSI 处于低位（阈值随趋势/时段可调）。
- pro1 变体：允许短期 RSI 反转、价/RSI 背离或成交量突增之一放宽入场。
- 下单前执行 `check_risk_control`：仓位上限、单笔亏损（ATR×合约乘数）、当日亏损上限等。

### 2.3 止盈/止损

- **止损**：ATR 倍数，低波动时设 ATR 下限；在 BOLL 下轨下方留结构缓冲（STOP_LOSS_STRUCT_MULTIPLIER）。
- **止盈**：以 `grid_upper` 减 ATR 偏移或最小 tick 余量，提高成交概率（TAKE_PROFIT_ATR_OFFSET、TAKE_PROFIT_MIN_OFFSET）。

### 2.4 时段自适应

- 不同时段可配置不同波动率、流动性与 **max_position**；DEMO 下 max_position 受 DEMO_MAX_POSITION 硬顶限制。
- 详见：[时段自适应策略使用说明](../时段自适应策略使用说明.md)。

---

## 3. 主要参数含义

| 参数 | 含义 | 典型值/说明 |
|------|------|-------------|
| GRID_MAX_POSITION | 最大持仓手数 | 3（DEMO 受 DEMO_MAX_POSITION 限制） |
| GRID_ATR_PERIOD | ATR 周期 | 14 |
| GRID_BOLL_PERIOD | BOLL 周期 | 20 |
| GRID_BOLL_STD | BOLL 标准差倍数 | 2 |
| GRID_RSI_PERIOD_1M / 5M | 1m/5m RSI 周期 | 14 |
| STOP_LOSS_MULTIPLIER | 止损 ATR 倍数 | 1.2 |
| STOP_LOSS_ATR_FLOOR | 低波动 ATR 下限 | 0.25（可环境变量） |
| STOP_LOSS_STRUCT_MULTIPLIER | 相对下轨结构缓冲 | 0.35 |
| TAKE_PROFIT_ATR_OFFSET | 止盈相对上轨 ATR 余量 | 0.2 |
| TAKE_PROFIT_MIN_OFFSET | 止盈最小价格余量 | 0.02 |
| BOLL_DIVERGENCE_THRESHOLD | BOLL 发散判定 | 0.2 |
| ATR_AMPLIFICATION_THRESHOLD | ATR 放大判定 | 0.3 |
| DAILY_LOSS_LIMIT | 日亏损上限（美元） | 2000 |
| SINGLE_TRADE_LOSS / MAX_SINGLE_LOSS | 单笔最大亏损 | 3000 / 5000 |

---

## 4. 训练与回测

- 网格/BOLL 为**规则策略**，无神经网络训练；参数通过历史回测或优化脚本调优（如 `parameter_grid_search`、算法优化报告中的 optimal_parameters）。
- 与“计算模式/大模型识别模式”的关系见：[两种策略模式设计](../两种策略模式设计.md)。

---

## 5. 相关文档

- [时段自适应策略使用说明](../时段自适应策略使用说明.md) — 时段配置与网格参数动态调整
- [两种策略模式设计](../两种策略模式设计.md) — 计算模式 vs 大模型识别模式
- [盈利目标和风险控制](../盈利目标和风险控制.md) — 风控与止损止盈
- `src/tiger1.py` 模块头部注释 — 网格/BOLL 算法逐步说明
