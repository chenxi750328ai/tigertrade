# 老虎 API 限制与平仓脚本说明

> 依据官方文档 [placeOrder](https://quant.itigerup.com/openapi/zh/python/operation/trade/placeOrder.html)、[请求频率](https://quant.itigerup.com/openapi/zh/python/permission/requestLimit.html)、[FAQ-交易](https://quant.itigerup.com/openapi/zh/python/FAQ/trade.html) 整理。**适用于全项目所有下单路径**。

## 项目内下单路径与约束遵从

| 模块 | 路径 | submit≠fill 校验 | 15 pending 处理 | time_in_force | 备注 |
|------|------|-----------------|----------------|---------------|------|
| `tiger1.place_tiger_order` | 策略、网格、boll、LLM、手工单等 | ❌ 仅看 place_order 返回 | ❌ 无 | DAY ✓ | 单笔 1 手为主，风险较低 |
| `OrderExecutor.execute_buy/sell` | run_moe_demo、Feature 测试 | ❌ 仅看 place_order 返回 | ❌ 无 | DAY ✓ | 单笔 1 手 |
| `scripts/close_demo_positions` | 平仓脚本 | ✓ wait_order_fill | ✓ 先撤挂单、分批≤15、等成交 | DAY ✓ | 大批量平仓，必须遵从 |
| `api_adapter.RealTradeApiAdapter` | 底层适配器 | — | — | 透传 DAY ✓ | 由调用方决定 |
| `place_take_profit_order` | 止盈单 | ❌ | ❌ | DAY ✓ | 附加单，主单 LMT |

**建议**：大批量（>15 手）或关键平仓场景，应复用 `close_demo_positions` 的流程（先撤挂单、分批、wait_order_fill）；单笔 1 手策略单可依赖 place_order 返回，但需知「提交成功≠成交」。

## 为何单测未发现平仓问题

1. **平仓脚本此前无单测**：`close_demo_positions.py` 是独立脚本，之前没有对应单元测试。
2. **place_order 成功 ≠ 成交**：单测若只 mock `place_order` 返回 order_id，会误判为成功；实际需 `get_order(id)` 查 `status==FILLED` 才表示成交。
3. **15 单 pending 限制**：老虎对同一品种待成交订单上限 15 个，超限会拒绝新单（reason: `Pending orders for same product exceed the limit(15)`）。此限制在文档中未显式写明，需通过实盘/排查发现。

## 老虎 API 约束汇总

### 下单与成交

| 约束 | 说明 | 应对 |
|------|------|------|
| place_order 返回 ≠ 成交 | 返回 order_id 仅表示提交成功；订单异步执行，可能 REJECTED/EXPIRED | 用 `get_order(id)` 查 status、reason |
| 不可直接开反向仓 | 持仓 100 股时不能直接卖 200 股，需先平仓 | 检查持仓再下单；卖单数量 ≤ 可卖数量 |
| 不支持锁仓 | 不能同时持有多头和空头（同一标的） | 净仓交易，先平后开 |

### 订单类型与时段

| 约束 | 说明 | 应对 |
|------|------|------|
| 市价单(MKT)/止损单(STP) 不支持盘前盘后 | 盘前盘后需 `outside_rth=false`，且 MKT/STP 仅盘中有效 | 盘前盘后用限价单 |
| 模拟账号 MKT 不支持 GTC | time_in_force 仅 DAY | 使用 DAY |
| 附加订单主单仅限价单 | 附加止损/止盈时主单必须 LMT | 主单用 limit_order |

### 频率限制（滚动 60 秒窗口）

| 类型 | 限制 | 接口 |
|------|------|------|
| 高频 | 120 次/分钟 | place_order、cancel_order、get_order、get_orders、get_open_orders、create_order、modify_order |
| 中频 | 60 次/分钟 | get_positions、get_future_bars、get_contract |
| 低频 | 10 次/分钟 | get_future_exchanges、get_symbols |

超限错误：`code=4 msg=rate limit error`。持续超频可能被黑名单。

### 其他（实测/文档）

| 约束 | 说明 | 应对 |
|------|------|------|
| 同品种最多 15 个 pending | 同一期货品种待成交订单总数 ≤ 15 | 先撤挂单；分批 ≤15 手，等成交再下一批 |
| 港股/窝轮数量限制 | 需符合每手股数 | get_trade_metas 取 lot_size |

## 平仓脚本流程（已实现）

1. 校验 `is_mock_mode=False`，确保使用真实 API。
2. **先撤挂单**：`get_open_orders(account, sec_type=FUT)`，过滤 SIL，全部 `cancel_order`。
3. `sync_positions_from_backend`，得到 long_qty、short_qty。
4. 先平空（BUY）、再平多（SELL）。
5. 每批最多 15 手，市价单；提交后 `wait_order_fill` 轮询直至 FILLED 或终态，再提交下一批。
6. 全部完成后 `sync_positions_from_backend` 校验持仓归零。

## 单测覆盖

### `tests/test_close_demo_positions.py`

- `_is_real_order_id`：区分老虎真实 order_id（纯数字）与 Mock（ORDER_/TEST_）。
- `wait_order_fill`：EXPIRED + reason 含 "Pending orders" 时返回未成交。
- `cancel_first_when_15_open_orders`：有 15 笔挂单时先撤单。
- `submit_success_does_not_imply_fill`：强调仅看 place_order 返回值会漏掉“提交成功但被拒”场景。
- `batch_size_respects_limit`：每批不超过 15 手。

### `tests/test_tiger_api_constraints.py`（全项目范围）

- **通用约束**：submit≠fill、MKT 用 DAY、不可开反向仓、不支持锁仓、频率/15 限、MKT/STP 盘中有效、附加单主单 LMT。
- **OrderExecutor**：`test_order_executor_uses_time_in_force_day`、`test_order_executor_does_not_verify_fill`。
- **tiger1.place_tiger_order**：`test_tiger1_uses_time_in_force_day`。
- **api_adapter**：`test_real_trade_adapter_accepts_time_in_force`。
