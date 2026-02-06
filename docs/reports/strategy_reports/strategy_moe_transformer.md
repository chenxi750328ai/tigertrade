# MoE Transformer 策略

*报告生成时间：2026-02-06T10:06:38.567009*

## 📄 设计文档（算法与参数详解）

- **→ [设计_MoE策略](../../strategy_designs/设计_MoE策略.md)** — 算法原理、参数含义、训练流程与实现细节。

## 算法说明

基于混合专家（Mixture of Experts）的 Transformer 时序预测策略。
- **模型**：多专家 Transformer，输入多时间尺度特征（如 46 维），输出方向/收益预测。
- **信号**：结合方向置信度与预测收益，在满足风控条件下发出买入/卖出/观望。
- **训练**：历史 K 线 + 技术指标，预测下一阶段涨跌与收益；支持 LoRA/微调。
- **适用**：DEMO/实盘主推策略之一，适合中短周期趋势与波动。

更完整的说明（模型结构、信号逻辑、训练与回测）请参见上方 **设计文档**：[设计_MoE策略](../../strategy_designs/设计_MoE策略.md)。

## 运行效果

### 回测效果

（来自历史数据回测，如 `parameter_grid_search`、训练阶段回测。）

| 指标 | 值 |
| --- | --- |
| return_pct | — |
| win_rate | 0 |
| num_trades | — |

### 实盘/DEMO 效果

（来自 API 历史订单、`today_yield.json`、DEMO 多日志汇总。）

| 指标 | 值 |
| --- | --- |
| profitability | 0 |
| win_rate | 0 |
| today_yield_pct | 6.65% |
| demo_order_success | 8662 |
| demo_sl_tp_log | 89334 |
| demo_execute_buy_calls | 33354 |
| demo_success_orders_sum | 0 |
| demo_fail_orders_sum | 2131197 |
| demo_logs_scanned | 18 |

### DEMO 运行统计（多日/多日志汇总）

共扫描 **18** 个 DEMO 日志文件（`demo_*.log`、`demo_run_20h_*.log`），汇总如下。

| 项 | 值 |
| --- | --- |
| 主单成功次数（汇总） | 8662 |
| 成功订单数（日志内统计汇总） | 0 |
| 失败订单数（日志内统计汇总） | 2131197 |
| 止损/止盈相关日志条数 | 89334 |
| 买入动作/execute_buy 次数 | 33354 |
| 日志总行数 | 8548094 |
| 最大仓位（各日志中出现过的最大值） | 8 手 |

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
