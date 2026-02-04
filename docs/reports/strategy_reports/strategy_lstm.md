# LSTM 策略

*报告生成时间：2026-02-04T11:13:31.705795*

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

回测/优化指标当前为占位或未写入；DEMO 多日汇总见 **MoE Transformer** 策略报告中的「DEMO 运行统计」。
- 数据源更新时间：2026-02-04 11:09（`algorithm_optimization_report.json`）
- 运行 **收益与算法优化**（`python scripts/optimize_algorithm_and_profitability.py`）或回测后，再运行 `python scripts/generate_strategy_reports.py` 可刷新。
