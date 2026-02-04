---
name: tiger-quant-algo-params
description: 查询 Tiger 项目的量化交易策略说明、参数区间与默认值、回测/优化报告。Agent 在需要了解策略算法或调参时使用本 SKILL。
---

# Tiger 量化算法与参数 SKILL

## 用途

开放 Tiger 项目的量化交易算法与参数能力：策略列表与算法说明、参数区间与默认值、回测与优化报告，供 Agent 或外部查阅/调参使用。

## 能力点

| 能力 | 说明 | 入口/位置 |
|------|------|-----------|
| 策略列表与算法说明 | 各策略名称、算法描述 | [策略算法与运行效果报告](../../reports/strategy_reports_index.html)、`scripts/generate_strategy_reports.py` 内 STRATEGY_ALGORITHMS |
| 参数与默认值 | 网格/BOLL/时段等参数 | `src/tiger1.py` 内 GRID_*、时段配置；策略类 __init__ 参数 |
| 回测/优化报告 | 收益率、胜率、回撤、建议 | `docs/reports/algorithm_optimization_report.json`、`docs/reports/strategy_reports/strategy_comparison.md` |
| 今日收益率 | DEMO/回测当日收益 | `docs/today_yield.json`、`scripts/update_today_yield_for_status.py` |

## 调用方式

- **报告**：运行 `python scripts/generate_strategy_reports.py` 生成各策略说明与对比报告；运行 `python scripts/optimize_algorithm_and_profitability.py` 生成算法优化报告。
- **策略元数据**：策略算法说明在 `scripts/generate_strategy_reports.py` 的 `STRATEGY_ALGORITHMS` 中维护；策略工厂见 `src/strategies/strategy_factory.py`。
- **参数**：网格与风控在 `src/tiger1.py`（如 `GRID_MAX_POSITION`、`DEMO_MAX_POSITION`、时段 `max_position`）；新策略参数在对应策略类中。

## 输入输出示例

- **策略报告**：读 `docs/reports/strategy_reports/strategy_<id>.md` 得算法说明与运行效果表。
- **对比报告**：读 `docs/reports/strategy_reports/strategy_comparison.md` 得多策略指标与今日收益率。
- **优化报告**：读 `docs/reports/algorithm_optimization_report.json` 得 strategy_performance、optimal_parameters、recommendations。

## 注意事项

- 报告为每日刷新，依赖例行运行 `generate_strategy_reports.py` 或收益与算法优化流程。
- 参数修改后需重启 DEMO 或回测才能生效。

## 相关文档

- [策略算法与运行效果报告](../../reports/strategy_reports_index.html)
- [SKILLs 设计与交易后端解耦](../../SKILLs设计与交易后端解耦.md)
- [盈利目标和风险控制](../../盈利目标和风险控制.md)
