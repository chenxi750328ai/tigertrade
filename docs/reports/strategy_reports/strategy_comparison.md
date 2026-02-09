# 策略对比报告

*报告生成时间：2026-02-09T15:19:41.933673*

**算法版本**：2.0（重大变更与对比见 [algorithm_versions.md](../../algorithm_versions.md)）

## 数据来源与「结果不全」说明

- **回测效果**：当前仅 **grid / boll** 由 `parameter_grid_search`（data/processed/test.csv）产出；moe_transformer、lstm 需单独回测或训练阶段产出，故表中可能为 —。
- **实盘/DEMO 效果**：**demo_*** 等列来自 DEMO 多日志汇总；同次运行四策略共用统计，故 grid/boll/lstm 与 MoE 数字一致。
- **今日收益率**：来自 `docs/today_yield.json`。若为 —，请运行 **收益与算法优化**（`python scripts/optimize_algorithm_and_profitability.py`）或单独运行 `python scripts/update_today_yield_for_status.py`，会从报告或 DEMO 日志更新后再刷新本报告。

## 回测效果对比

（来自历史数据回测，如 `parameter_grid_search`。）

| 策略 | return_pct | win_rate | num_trades |
| --- | --- | --- | --- |
| moe_transformer | — | 0 | — |
| lstm | — | 0 | — |
| grid | 8.447142857142854 | 100.0 | 1 |
| boll | 8.447142857142854 | 100.0 | 1 |

## 实盘/DEMO 效果对比

（来自 API 订单、today_yield、DEMO 多日志汇总。）

| 策略 | profitability | win_rate | today_yield_pct | demo_order_success | demo_sl_tp_log | demo_execute_buy_calls | demo_success_orders_sum | demo_fail_orders_sum | demo_logs_scanned |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| moe_transformer | 0 | 0 | 6.65% | 8662 | 89334 | 33354 | 0 | 2131197 | 18 |
| lstm | 0 | 0 | 6.65% | 8662 | 89334 | 33354 | 0 | 2131197 | 18 |
| grid | 0 | 100.0 | 6.65% | 8662 | 89334 | 33354 | 0 | 2131197 | 18 |
| boll | 0 | 100.0 | 6.65% | 8662 | 89334 | 33354 | 0 | 2131197 | 18 |

**数据完整度**：回测 4/4 策略有数据；实盘/DEMO 4/4 策略有日志汇总；今日收益率见下。

## 今日收益率（DEMO/实盘）

- 日期：2026-02-09
- 收益率：6.65%

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
