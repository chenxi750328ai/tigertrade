# LSTM 策略

*报告生成时间：2026-02-04T10:31:05.759452*

## 算法说明

基于 LSTM 的时序预测策略（与 LLM 策略同架构，mode=hybrid）。
- **模型**：LSTM 编码 + 全连接输出，支持 predict_profit 收益预测。
- **信号**：与 MoE 类似，由预测方向与收益生成交易信号。
- **训练**：同多时间尺度历史数据。
- **适用**：作为对比基线或备选模型。

## 运行效果

| 指标 | 值 |
| --- | --- |
| profitability | 0 |
| win_rate | 0 |
| sharpe_ratio | 0 |
| max_drawdown | 0 |
