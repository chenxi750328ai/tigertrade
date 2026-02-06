# LSTM 策略

*报告生成时间：2026-02-06T09:16:31.830738*

## 📄 设计文档（算法与参数详解）

- **→ [设计_LSTM策略](../../strategy_designs/设计_LSTM策略.md)** — 算法原理、参数含义、训练流程与实现细节。

## 算法说明

基于 LSTM 的时序预测策略（与 LLM 策略同架构，mode=hybrid）。
- **模型**：LSTM 编码 + 全连接输出，支持 predict_profit 收益预测。
- **信号**：与 MoE 类似，由预测方向与收益生成交易信号。
- **训练**：同多时间尺度历史数据。
- **适用**：作为对比基线或备选模型。

更完整的说明（模型结构、信号逻辑、训练与回测）请参见上方 **设计文档**：[设计_LSTM策略](../../strategy_designs/设计_LSTM策略.md)。

## 运行效果

### 实盘/DEMO 效果

（来自 API 历史订单、`today_yield.json`、DEMO 多日志汇总。）

| 指标 | 值 |
| --- | --- |
| profitability | 0 |
| win_rate | 0 |

### 今日收益率

- 6.65%

## 每日收益与算法优化在干啥

**每日「收益与算法优化」在干啥**
- **结果分析**：用 API 历史订单算收益率/胜率（若有）；用 DEMO 多日志汇总订单与止损止盈统计；用 today_yield 展示今日收益率。
- **算法优化**：对网格/BOLL 做参数网格回测（需 `data/processed/test.csv`），得到最优参数与回测收益/胜率，写入报告。
- **报告产出**：更新 `algorithm_optimization_report.json`/`.md`、本策略算法与运行效果报告；报告内「效果数据来源」会写明本次用了哪些数据。

**咋干的（步骤）**
1. 加载历史订单（API）→ 若无则收益率为空。
2. 计算收益率（解析订单盈亏）→ 当前未解析时为空。
3. 分析策略表现：汇总所有 DEMO 日志（demo_*.log、demo_run_20h_*.log）→ 主单成功、止损止盈条数等；读 today_yield.json。
4. 优化参数：对 grid、boll 跑网格回测（parameter_grid_search）→ 最优参数与 return_pct、win_rate。
5. 生成算法优化报告（含效果数据来源说明）并调用本脚本刷新策略报告。

**脚本**：`python scripts/optimize_algorithm_and_profitability.py`。详见 `docs/每日例行_效果数据说明.md`。

详见：[每日例行_效果数据说明](../../每日例行_效果数据说明.md)。
