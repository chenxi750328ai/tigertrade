# 老虎后台订单拉取分析

**拉取时间**: 2026-02-25 16:32:44
**时间范围**: 2026-02-25 00:00:00 ~ 2026-02-25 16:32:37
**原始数据**: [backend_orders_latest.json](backend_orders_latest.json)

## 汇总

| 项目 | 值 |
|------|-----|
| 总订单数 | 26 |
| 已成交(FILLED 等) | 7 |
| 限价 100 的订单数 | 15 |

## 按合约

| 合约 | 笔数 |
|------|-----|
| SIL2605 | 14 |
| SIL2603 | 12 |

## 按状态

| 状态 | 笔数 |
|------|-----|
| EXPIRED | 12 |
| HELD | 7 |
| FILLED | 7 |

## 限价 100 的订单（疑似测试单）

共 15 笔，limit_price=100：

- order_id=9538 symbol=SIL2603 side=BUY status=EXPIRED quantity=1 limit_price=100.0
- order_id=9537 symbol=SIL2603 side=BUY status=EXPIRED quantity=1 limit_price=100.0
- order_id=9536 symbol=SIL2603 side=BUY status=EXPIRED quantity=1 limit_price=100.0
- order_id=9535 symbol=SIL2603 side=BUY status=EXPIRED quantity=1 limit_price=100.0
- order_id=9533 symbol=SIL2603 side=BUY status=EXPIRED quantity=1 limit_price=100.0
- order_id=9531 symbol=SIL2603 side=BUY status=EXPIRED quantity=1 limit_price=100.0
- order_id=9530 symbol=SIL2603 side=BUY status=EXPIRED quantity=1 limit_price=100.0
- order_id=9528 symbol=SIL2603 side=BUY status=EXPIRED quantity=1 limit_price=100.0
- order_id=9552 symbol=SIL2605 side=BUY status=FILLED quantity=1 limit_price=100.0
- order_id=9551 symbol=SIL2605 side=BUY status=FILLED quantity=1 limit_price=100.0
- order_id=9549 symbol=SIL2605 side=BUY status=FILLED quantity=1 limit_price=100.0
- order_id=9547 symbol=SIL2605 side=BUY status=FILLED quantity=1 limit_price=100.0
- order_id=9543 symbol=SIL2605 side=BUY status=FILLED quantity=1 limit_price=100.0
- order_id=9542 symbol=SIL2605 side=BUY status=FILLED quantity=1 limit_price=100.0
- order_id=9540 symbol=SIL2605 side=BUY status=FILLED quantity=1 limit_price=100.0

## 结论与建议

1. **7 笔 FILLED 全是限价 100 买入（SIL2605）**：与测试用例里常用的 `price=100.0` 一致，可判定为**测试/自测单流入实盘**。跑带 real API 或未 mock 的用例时，用了 DEMO 账户且下了真实单。
2. **SIL2603 上 8 笔 BUY limit 100 已 EXPIRED**：同样为测试单，未成交但占用了订单记录。
3. **建议**：
   - 凡跑 pytest/自测，**不得**使用 openapicfg_dem 对真实老虎下单；或统一 mock `trade_api.place_order` / `place_tiger_order`，保证测试路径不发真实单。
   - 黑盒/集成若需「校验后台可见」，应单独用标记（如 `real_api`）且默认不跑，或使用独立测试账户。
   - 硬顶 2 手已存在，但测试多进程/多轮会叠单；需保证同一 DEMO 账户同时只有一处真实下单入口（或测试与实盘完全隔离账户）。
