# BOLL 网格策略 策略

*报告生成时间：2026-02-04T11:13:31.705795*

## 📄 设计文档（算法与参数详解）

- **→ [设计_网格与BOLL策略](../../strategy_designs/设计_网格与BOLL策略.md)** — 算法原理、参数含义、训练流程与实现细节。

## 算法说明

基于布林带的 1 分钟网格变体（boll1m_grid_strategy）。
- **逻辑**：使用 5 分钟布林带中轨与上下轨作为区间边界，结合 1 分钟 K 线与 RSI 判断入场与出场。
- **与 grid 关系**：同属网格族，参数与时段配置可单独调优。
- **适用**：与网格策略类似，侧重 1m 与 5m 结合。

更完整的说明（模型结构、信号逻辑、训练与回测）请参见上方 **设计文档**：[设计_网格与BOLL策略](../../strategy_designs/设计_网格与BOLL策略.md)。

## 运行效果

回测/优化指标当前为占位或未写入；DEMO 多日汇总见 **MoE Transformer** 策略报告中的「DEMO 运行统计」。
- 数据源更新时间：2026-02-04 11:09（`algorithm_optimization_report.json`）
- 运行 **收益与算法优化**（`python scripts/optimize_algorithm_and_profitability.py`）或回测后，再运行 `python scripts/generate_strategy_reports.py` 可刷新。
