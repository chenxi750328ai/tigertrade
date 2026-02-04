# SKILLs 设计与交易后端解耦

> **目的**：通过 SKILLs 开放 Tiger 项目能力，供 Agent/江湖等调用；交易后端与具体券商/平台解耦，可接老虎或其他平台。

---

## 1. SKILLs 设计目标

设计一批 **SKILL**，把 Tiger 的核心能力封装成可被外部（如 Cursor/Codex Agent、江湖多 Agent）调用的能力单元，便于复用与协作。

### 1.1 建议开放的 SKILL 方向

| 能力方向 | 说明 | 示例能力点 |
|----------|------|------------|
| **数据分析** | 行情与历史数据查询、统计、可视化 | K 线/Tick 获取、指标计算、简单统计与导出 |
| **量化交易算法与参数** | 策略说明、参数查询与建议、回测结果 | 策略列表与算法说明、参数区间与默认值、回测/优化报告 |
| **交易后端接入** | 下单、撤单、查询订单与持仓（与具体平台解耦） | 统一订单接口、账户/持仓查询、支持多后端适配器 |

### 1.2 与 Agent/江湖的衔接

- SKILL 的输入/输出、命名与协议尽量与现有 **SKILL 规范**（如 Cursor/Codex 的 SKILL.md、江湖任务约定）对齐，便于被「创建技能」「安装技能」等流程使用。
- 每个 SKILL 应有清晰的能力描述、参数说明与示例，便于 Agent 发现与调用。

### 1.3 实施步骤（建议）

1. **梳理现有能力**：列出 Tiger 当前已实现的数据接口、策略入口、订单/持仓调用点。
2. **定义 SKILL 清单**：按「数据分析」「量化算法与参数」「交易后端」三类，写出每个 SKILL 的名称、用途、输入输出、依赖脚本/API。
3. **编写 SKILL 文档**：按规范格式（如 SKILL.md）编写，并放入仓库统一路径（如 `docs/skills/` 或与现有 skill 目录一致）。
4. **可选实现层**：若需 CLI/HTTP 封装，再增加薄封装层，供 Agent 或 cron 调用。

### 1.4 已落地的 SKILL 文档

| 能力方向 | 路径 |
|----------|------|
| 数据分析 | [docs/skills/tiger-data-analysis/SKILL.md](skills/tiger-data-analysis/SKILL.md) |
| 量化算法与参数 | [docs/skills/tiger-quant-algo-params/SKILL.md](skills/tiger-quant-algo-params/SKILL.md) |
| 交易后端接入 | [docs/skills/tiger-trading-backend/SKILL.md](skills/tiger-trading-backend/SKILL.md) |
| 索引 | [docs/skills/README.md](skills/README.md) |

---

## 2. 交易后端解耦（可接老虎或其他平台）

### 2.1 原则

- **交易后端**指：执行下单、撤单、查询订单/持仓/账户的底层接口，当前实现为 Tiger Open API。
- 设计上应 **解耦**：业务逻辑（策略、风控、信号）不直接依赖「老虎」或某一家券商，而是依赖 **抽象接口**；具体券商/平台通过 **适配器** 接入。

### 2.2 目标架构（概念）

```
策略 / 风控 / 执行层
        ↓
   [统一交易接口]
   - place_order(side, symbol, quantity, price, order_type, ...)
   - cancel_order(order_id)
   - get_orders / get_positions / get_account
        ↓
   [适配器层]
   - TigerTradeApiAdapter（老虎）
   - OtherBrokerAdapter（其他券商/平台，按需实现）
```

- **统一交易接口**：在项目内定义一套稳定的方法签名与数据结构（订单、持仓、账户），所有适配器实现该接口。
- **适配器**：每个后端（老虎、其他券商或模拟）一个适配器，实现同一套接口；通过配置或环境变量选择当前使用的适配器。

### 2.3 与当前代码的关系

- 当前项目已有 `api_adapter`、`RealTradeApiAdapter`、`MockQuoteApiAdapter` 等，可视为适配器雏形。
- **解耦推进**：  
  - 明确「统一交易接口」的抽象（如 Python 的 Protocol/ABC 或单独模块），所有调用方只依赖该抽象。  
  - 将「老虎」专属逻辑收敛到 Tiger 适配器内，避免在策略/执行层写死老虎 API。  
  - 新增其他平台时，仅新增适配器并实现同一接口，不改策略与风控主流程。

### 2.4 与 SKILLs 的关系

- 「交易后端接入」SKILL 面向的是 **统一交易接口**（下单、撤单、查询），而不是某一家 API。
- Agent 通过 SKILL 调用「下单」「查持仓」等能力时，不关心底层是老虎还是其他平台；具体后端由配置决定。

---

## 3. 风险与依赖

| 风险/依赖 | 说明 |
|-----------|------|
| SKILL 规范不统一 | 需与 Cursor/Codex/江湖 的 SKILL 格式或约定对齐，避免重复造轮子 |
| 多后端兼容成本 | 不同券商 API 差异大，统一接口需做合理抽象与折中 |
| 安全与权限 | 交易类 SKILL 暴露给 Agent 时，需考虑鉴权、权限与审计 |

---

## 4. 相关链接

- [项目计划（月度/周计划）](项目计划_月度周计划.md) — 任务项「SKILLs 设计与交易后端解耦」
- [订单执行流程说明](订单执行流程说明.md) — 当前下单与 api_adapter 流程
- [策略算法与运行效果报告](reports/strategy_reports_index.html) — 策略与算法说明
