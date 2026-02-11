# 策略对比报告

*报告生成时间：2026-02-11T12:04:31.988861*

**算法版本**：2.0（重大变更与对比见 [algorithm_versions.md](../../algorithm_versions.md)）

## 数据来源与「结果不全」说明

- **回测效果**：**grid / boll** 由 `parameter_grid_search` 参数网格回测（**双向**：long/short）；**moe_transformer、lstm** 由 `scripts/backtest_model_strategies.py` 用 test.csv 信号回测（**双向**：1=多/平空，2=空/平多），四策略均有 num_trades/return_pct/win_rate。
- **回测 vs 实盘**：回测与实盘仅数据来源不同，策略与运行过程应一致才有参考意义；若回测笔数远少于实盘说明不一致需对齐。详见 [algorithm_optimization_report.md](algorithm_optimization_report.md)「回测与实盘差异说明」。
- **实盘/DEMO 效果**：**demo_*** 等列来自 DEMO 多日志汇总；同次运行四策略共用统计，故 grid/boll/lstm 与 MoE 数字一致。
- **今日收益率**：来自 `docs/today_yield.json`。若为 —，请运行 **收益与算法优化**（`python scripts/optimize_algorithm_and_profitability.py`）或单独运行 `python scripts/update_today_yield_for_status.py`，会从报告或 DEMO 日志更新后再刷新本报告。

## 回测效果对比

（回测数据：历史 K 线回测，含**回测收益率、回测胜率、回测笔数**。）

| 策略 | num_trades | return_pct | avg_per_trade_pct | top_per_trade_pct | win_rate |
| --- | --- | --- | --- | --- | --- |
| moe_transformer | 33 | 2.46 | 0.01 | 0.03 | 100.0 |
| lstm | 33 | 2.46 | 0.01 | 0.03 | 100.0 |
| grid | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| boll | 11 | 7.74 | 0.7 | 7.09 | 36.36 |

*说明*：**num_trades**=实际成交笔数；**return_pct**=总收益率；**avg_per_trade_pct**=单笔平均%；**top_per_trade_pct**=单笔TOP%；**win_rate**=胜率。

## 实盘/DEMO 效果对比

（实盘表与回测表**同结构**：笔数、收益率（核对/推算）、单笔均、单笔TOP、胜率；仅收益率区分「老虎核对」与「未核对推算」。DEMO 日志汇总见下表。）

| 策略 | num_trades | return_pct_verified | return_pct_estimated | avg_per_trade_pct | top_per_trade_pct | win_rate |
| --- | --- | --- | --- | --- | --- | --- |
| moe_transformer | 9743（DEMO主单，见下表） | —（见根因说明） | —（见根因说明） | —（见根因说明） | —（见根因说明） | —（见根因说明） |
| lstm | 9743（DEMO主单，见下表） | —（见根因说明） | —（见根因说明） | —（见根因说明） | —（见根因说明） | —（见根因说明） |
| grid | 9743（DEMO主单，见下表） | —（见根因说明） | —（见根因说明） | —（见根因说明） | —（见根因说明） | —（见根因说明） |
| boll | 9743（DEMO主单，见下表） | —（见根因说明） | —（见根因说明） | —（见根因说明） | —（见根因说明） | —（见根因说明） |

*说明*：与回测表同指标；**return_pct_verified**=老虎核对收益率，**return_pct_estimated**=未核对推算；无数据时为 —（见根因说明）。

### DEMO 日志汇总

| 策略 | demo_order_success | demo_sl_tp_log | demo_execute_buy_calls | demo_success_orders_sum | demo_fail_orders_sum | demo_logs_scanned |
| --- | --- | --- | --- | --- | --- | --- |
| moe_transformer | 9743 | 91496 | 34638 | 330611 | 4555826 | 20 |
| lstm | 9743 | 91496 | 34638 | 330611 | 4555826 | 20 |
| grid | 9743 | 91496 | 34638 | 330611 | 4555826 | 20 |
| boll | 9743 | 91496 | 34638 | 330611 | 4555826 | 20 |

**数据完整度**：回测 4/4 策略有数据；实盘主表来自老虎 API/今日收益率；DEMO 汇总 4/4 策略。

## 今日收益率（DEMO/实盘）

- 日期：2026-02-11
- **实际收益率（老虎后台核对）**：—（根因见 [算法优化报告](../algorithm_optimization_report.md) 中「本报告空项根因说明」）
- **推算收益率（未核对）**：—（根因见 [算法优化报告](../algorithm_optimization_report.md) 中「本报告空项根因说明」）
- 当前展示：无老虎核对；实盘笔数见上表「num_trades」列（DEMO 主单）
- **空项根因**：实际/推算收益率为空时，原因均写在 [算法优化报告](../algorithm_optimization_report.md) 的「本报告空项根因说明」中，须追根问底、不忽悠。
- （若为 —：运行 `python scripts/optimize_algorithm_and_profitability.py` 或 `update_today_yield_for_status.py` 更新。）

## 指标说明（含义与计算方式）

| 指标项 | 含义 | 计算方式 / 说明 |
| --- | --- | --- |
| return_pct | 回测收益率 | (期末资金 − 10万) / 10万 × 100（%）。来自 data/processed/test.csv 历史 K 线回测。 |
| win_rate | 回测胜率 | 盈利笔数 / 完成笔数 × 100（%）。回测表为回测结果；实盘表为实盘胜率，仅来自 API 历史订单解析。 |
| num_trades | 回测成交笔数 | 回测区间内实际完成的开平仓次数。 |
| avg_per_trade_pct | 单笔平均% | 总收益/笔数，每笔占初始资金%。 |
| top_per_trade_pct | 单笔TOP% | 单笔最大收益占初始资金%。 |
| profitability | 实盘盈亏汇总 | API 历史订单解析得到的总交易数、总盈亏等；无 API 时为 0 或 —。 |
| return_pct_verified | 收益率（核对） | 与回测 return_pct 对应；老虎后台订单/成交数据计算；未拉取或未核对时为 —。 |
| return_pct_estimated | 收益率（推算） | 与回测 return_pct 对应；未与老虎核对时的推算值；无推算时为 —。 |
| yield_verified | 实际收益率（老虎核对） | 用老虎后台订单/成交数据计算出的收益率；未拉取或未核对时为 —。 |
| yield_estimated | 推算收益率（未核对） | 未与老虎核对时的推算值（如 API 报告解析）；无推算时为 —。 |
| today_yield_pct | 今日收益率展示 | 本日在状态/报告中展示的收益率，来自 today_yield.json；须以实际（老虎核对）为准。 |
| demo_order_success | DEMO 主单成功次数 | DEMO 日志中「订单提交成功」等匹配次数（多日志汇总），非老虎后台笔数。 |
| demo_sl_tp_log | DEMO 止损/止盈日志条数 | 日志全文匹配「止损|止盈|已提交止损|已提交止盈」等的出现次数。 |
| demo_execute_buy_calls | DEMO 买入动作次数 | 日志匹配「execute_buy|动作: 买入」的次数。 |
| demo_success_orders_sum | DEMO 成功订单数(日志) | 日志内统计的成功订单数汇总，非老虎后台。 |
| demo_fail_orders_sum | DEMO 失败订单数(日志) | 日志内统计的失败订单数汇总。 |
| demo_logs_scanned | DEMO 扫描日志数 | 参与汇总的 demo_*.log、demo_run_20h_*.log 文件个数。 |

详见 [DEMO实盘收益率_定义与数据来源](../../DEMO实盘收益率_定义与数据来源.md)、[每日例行_效果数据说明](../../每日例行_效果数据说明.md)、[回溯_执行失败为何出现收益率与推算收益率](../../回溯_执行失败为何出现收益率与推算收益率.md)。

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
