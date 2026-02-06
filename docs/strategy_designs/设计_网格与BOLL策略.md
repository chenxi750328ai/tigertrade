# 网格策略与 BOLL 网格策略设计文档

> 本文档描述基于价格区间与布林带的网格交易策略的**算法设计**、**训练/回测过程设计**与**数据分析设计**。实现见 `src/tiger1.py`（`grid_trading_strategy`、`grid_trading_strategy_pro1`、`boll1m_grid_strategy`）及 `src/strategies/time_period_strategy.py`。

---

## 1. 策略概述

- **网格策略**：以 5 分钟 Bollinger Bands 中轨/上轨/下轨（或时段自适应区间）为 **grid_lower / grid_upper**，价格接近下轨且 1 分钟 RSI 低位时考虑买入，接近上轨或触发止盈/止损时卖出。
- **BOLL 网格策略**：网格族的 1 分钟侧重变体（boll1m_grid_strategy），同样以 5 分钟布林带为边界，结合 1 分钟 K 线与 RSI 判断入场与出场。

两者共用一套参数体系与时段自适应逻辑，区别主要在调用入口与部分阈值细节。

---

## 2. 算法设计

### 2.1 输入与数据流

1. **行情输入**：1 分钟与 5 分钟 K 线（`get_kline_data`），时区与格式标准化，满足最少 K 线数（MIN_KLINES）。
2. **指标计算**：5 分钟 BOLL（中轨/上轨/下轨）、ATR；1 分钟 RSI、成交量（`calculate_indicators`）。
3. **网格边界**：以 5 分钟 Boll 中轨/上轨/下轨为基准，结合 ATR 微调；当 BOLL 发散或 ATR 放大超过阈值时，可动态放宽或收紧区间，减少震荡区频繁交易。

### 2.2 输出与决策

- **输出**：二元决策——是否发出买入/卖出信号（及手数）；无模型概率，纯规则。
- **开仓（买入）**：价格 ≤ grid_lower（或接近）+ 1 分钟 RSI 低位；pro1 变体可叠加 RSI 反转、价量背离、成交量突增等条件放宽。
- **平仓**：止盈（价格接近 grid_upper 减偏移）或止损（ATR 倍数/结构缓冲）触发时卖出。

### 2.3 风控与约束

- **仓位**：不超过 GRID_MAX_POSITION（DEMO 受 DEMO_MAX_POSITION 限制）。
- **单笔亏损**：ATR × 合约乘数 × 倍数上限；低波动时 ATR 设下限（STOP_LOSS_ATR_FLOOR）。
- **日亏损**：超过 DAILY_LOSS_LIMIT 则停止开新仓。
- **止损/止盈**：见下方参数表与 [盈利目标和风险控制](../盈利目标和风险控制.md)。

### 2.4 时段自适应

- 不同时段可配置不同波动率、流动性与 **max_position**；实现见 [时段自适应策略使用说明](../时段自适应策略使用说明.md)。

---

## 3. 训练/回测过程设计（参数优化）

网格与 BOLL 为**规则策略**，无神经网络训练；“训练”即**参数搜索与回测评估**。

### 3.1 回测数据

- **数据来源**：历史 K 线（如 `data/processed/test.csv` 或项目约定的回测 CSV），需包含 1m/5m 价格与足够长度以计算 BOLL、ATR、RSI。
- **回测逻辑**：按时间顺序模拟：到价则开/平仓，记录每笔盈亏与回撤，产出 return_pct、win_rate、num_trades 等。

### 3.2 参数搜索

- **脚本**：`scripts/parameter_grid_search.py` 中的 `grid_search_optimal_params(name)`，对 `grid` / `boll` 在参数网格上枚举或搜索（如 ATR 周期、BOLL 周期、RSI 阈值、止损倍数等）。
- **目标**：最大化夏普或收益、或满足约束下的收益；产出 optimal_parameters 与 backtest_metrics（return_pct、win_rate、num_trades），写入算法优化报告。

### 3.3 与每日例行衔接

- 每日例行脚本 `optimize_algorithm_and_profitability.py` 调用 `parameter_grid_search`，将 grid/boll 的最优参数与回测效果写入 `algorithm_optimization_report.json` 的 `strategy_performance`，供策略对比报告展示。

---

## 4. 数据分析设计

### 4.1 特征/数据来源

- **行情**：1 分钟、5 分钟 K 线（开高低收、量），来自交易所或数据层 API。
- **衍生指标**：BOLL（周期、标准差倍数）、ATR（周期）、RSI（1m/5m 周期）、成交量；均在回测与实盘一致地计算，保证可复现。

### 4.2 标签/目标（回测用）

- **回测无显式“标签”**：规则直接根据当前价格与指标判断买卖；评估目标为**回测结果**（收益率、胜率、交易次数、最大回撤等）。
- **参数搜索的目标变量**：即上述指标（如 return_pct、sharpe），用于比较不同参数组合。

### 4.3 验证与稳健性

- **样本外**：回测可按时间划分“训练期”与“验证期”，在验证期上评估选定参数，避免过拟合到单一时段。
- **稳健性**：可在不同品种、不同时段重复回测，检查参数是否泛化；文档与报告中标明回测区间与数据来源。

### 4.4 数据管道

- 原始 K 线 → 清洗与对齐（时区、缺失值）→ 计算 BOLL/ATR/RSI → 写入 `data/processed/test.csv` 或回测输入；实盘侧同样由实时 K 线计算指标后送入策略逻辑。

---

## 5. 主要参数含义

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

## 6. 相关文档

- [时段自适应策略使用说明](../时段自适应策略使用说明.md) — 时段配置与网格参数动态调整
- [两种策略模式设计](../两种策略模式设计.md) — 计算模式 vs 大模型识别模式
- [盈利目标和风险控制](../盈利目标和风险控制.md) — 风控与止损止盈
- `src/tiger1.py` 模块头部注释 — 网格/BOLL 算法逐步说明
