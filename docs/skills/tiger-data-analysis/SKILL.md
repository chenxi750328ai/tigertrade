---
name: tiger-data-analysis
description: 使用 Tiger 项目进行行情与历史数据分析（K 线、Tick、指标计算与统计）。Agent 在需要查询行情、做简单统计或导出数据时使用本 SKILL。
---

# Tiger 数据分析 SKILL

## 用途

开放 Tiger 项目的数据分析能力：行情与历史数据查询、指标计算、简单统计与导出，供 Agent 或外部脚本调用。

## 能力点

| 能力 | 说明 | 入口/脚本 |
|------|------|-----------|
| K 线获取 | 多周期 K 线（1m/5m/1h/1d） | `tiger1.get_kline_data`、`api_adapter.quote_api.get_future_bars` |
| Tick 获取 | 实时 Tick 或最近 Tick | `tiger1.get_tick_data`、数据采集器 |
| 指标计算 | 5m BOLL、1m RSI、ATR 等 | `tiger1` 内 indicators、或策略层 |
| 数据导出/统计 | 导出 CSV、简单统计 | `scripts/` 下数据脚本、`trading_data/` 目录 |

## 调用方式

- **Python**：在项目根目录 `sys.path` 包含项目后，`from src import tiger1 as t1`，调用 `t1.get_kline_data(...)`、`t1.get_tick_data(...)`；或通过 `api_manager.quote_api` 获取行情 API。
- **脚本**：数据采集与合并见 `scripts/启动Tick采集器.sh`、`scripts/merge_recent_data_and_train.py` 等。
- **数据目录**：`/home/cx/trading_data`（可配置），K 线/Tick 按日期与合约分文件存放。

## 输入输出示例

- **get_kline_data(symbol, period, count)**：返回 DataFrame，列含 time, open, high, low, close, volume；不足或失败时可能返回合成数据或 None。
- **get_tick_data(symbol)**：返回当前 Tick 价格或最新价。
- 行情 API 的 `get_future_bars(symbols, period, begin_time, end_time, count)`：返回 DataFrame 或 None。

## 注意事项

- 实盘/真实 API 需配置 Tiger 账户与行情权限；DEMO/Mock 使用模拟数据。
- 数据不足时策略可能使用合成 K 线，分析结果需区分真实与合成。

## 相关文档

- [数据说明](../../数据说明.md)
- [项目目标和自动化流程总结](../../项目目标和自动化流程总结.md)
