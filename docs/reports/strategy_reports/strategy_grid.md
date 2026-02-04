# 网格策略 策略

*报告生成时间：2026-02-04T11:13:31.705795*

## 📄 设计文档（算法与参数详解）

- **→ [设计_网格与BOLL策略](../../strategy_designs/设计_网格与BOLL策略.md)** — 算法原理、参数含义、训练流程与实现细节。

## 算法说明

基于价格区间的网格交易策略。
- **逻辑**：以 5 分钟 Boll 中轨/上轨/下轨或时段自适应区间作为 grid_lower / grid_upper，价格接近下轨且 1 分钟 RSI 低位时考虑买入，接近上轨或止盈/止损条件时卖出。
- **参数**：网格间距、RSI 阈值、时段相关 max_position 等由时段自适应策略调整。
- **适用**：震荡市、区间行情。

更完整的说明（模型结构、信号逻辑、训练与回测）请参见上方 **设计文档**：[设计_网格与BOLL策略](../../strategy_designs/设计_网格与BOLL策略.md)。

## 运行效果

回测/优化指标当前为占位或未写入；DEMO 多日汇总见 **MoE Transformer** 策略报告中的「DEMO 运行统计」。
- 数据源更新时间：2026-02-04 11:09（`algorithm_optimization_report.json`）
- 运行 **收益与算法优化**（`python scripts/optimize_algorithm_and_profitability.py`）或回测后，再运行 `python scripts/generate_strategy_reports.py` 可刷新。
