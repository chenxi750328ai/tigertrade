---
name: tiger-trading-backend
description: 通过统一交易接口进行下单、撤单、查询订单与持仓。与具体券商解耦，当前可实现为老虎或 Mock；其他平台可新增适配器。Agent 在需要执行交易或查询订单/持仓时使用本 SKILL。
---

# Tiger 交易后端接入 SKILL

## 用途

开放 Tiger 项目的交易后端能力：**统一交易接口**（下单、撤单、查订单/持仓/账户），与具体券商/平台解耦；当前支持老虎（Tiger）与 Mock，后续可接入其他平台。

## 能力点

| 能力 | 说明 | 接口 |
|------|------|------|
| 下单 | place_order(symbol, side, order_type, quantity, limit_price, stop_price, ...) | 统一协议 |
| 撤单 | cancel_order(order_id) | 统一协议（部分后端可能未实现） |
| 查订单 | get_orders(account, symbol, limit)、get_order(order_id) | 统一协议 |
| 查持仓/账户 | get_positions(account)、get_account(account) | 统一协议（部分后端可能未实现） |

## 架构（解耦）

- **协议**：`src.trading.protocol.TradingBackendProtocol` 定义统一方法签名。
- **适配器**：`src.api_adapter.RealTradeApiAdapter`（老虎）、`MockTradeApiAdapter`（模拟）；其他券商实现同一协议即可接入。
- **选择后端**：环境变量 `TRADING_BACKEND=tiger` 或 `mock`；未设置时由当前已初始化的 API（真实/模拟）决定。获取当前适配器：`from src.trading.backend_factory import get_trading_backend`。

## 调用方式

- **Python**：项目根目录下，先完成 API 初始化（`api_manager.initialize_real_apis(...)` 或 `initialize_mock_apis(...)`），再通过 `api_manager.trade_api` 或 `get_trading_backend()` 得到适配器，调用 `place_order`、`get_orders` 等。
- **业务层**：策略与风控不直接依赖老虎 API，只依赖 `trade_api` 的接口（即协议）；具体是老虎还是其他平台由配置决定。

## 输入输出示例

- **place_order(symbol, side, order_type, quantity, time_in_force=None, limit_price=None, stop_price=None)**：返回订单对象或订单 ID；失败抛异常。
- **get_orders(account=None, symbol=None, limit=100)**：返回订单列表（结构由适配器决定）。
- **get_order(order_id=..., account=None)**：返回单个订单或 None。

## 注意事项

- 交易类能力涉及资金与风险，暴露给 Agent 时需考虑鉴权、权限与审计。
- 老虎适配器需配置 account 与 Tiger 后台授权；Mock 用于测试/DEMO。
- 新增其他券商时，实现 `TradingBackendProtocol` 并注册到工厂或 api_manager 即可，无需改策略主流程。

## 相关文档

- [SKILLs 设计与交易后端解耦](../../SKILLs设计与交易后端解耦.md)
- [订单执行流程说明](../../订单执行流程说明.md)
- 统一协议定义：仓库内 `src/trading/protocol.py`
