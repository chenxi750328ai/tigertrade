# Tiger 项目 SKILLs（能力开放）

本目录存放 Tiger 项目对外开放的 **SKILL** 文档，供 Agent（如 Cursor/Codex、江湖多 Agent）发现与调用项目能力。每个子目录对应一个 SKILL，内含 `SKILL.md` 描述用途、能力点、调用方式与相关文档。

## SKILL 清单

| SKILL | 说明 |
|-------|------|
| [tiger-data-analysis](tiger-data-analysis/SKILL.md) | 数据分析：K 线/Tick 获取、指标计算、统计与导出 |
| [tiger-quant-algo-params](tiger-quant-algo-params/SKILL.md) | 量化算法与参数：策略说明、参数与默认值、回测/优化报告 |
| [tiger-trading-backend](tiger-trading-backend/SKILL.md) | 交易后端接入：统一下单/撤单/查订单与持仓，与券商解耦 |

## 设计说明

- 详见 [SKILLs 设计与交易后端解耦](../SKILLs设计与交易后端解耦.md)。
- 交易后端通过 `src.trading.protocol.TradingBackendProtocol` 与适配器解耦，可接老虎或其他平台。
