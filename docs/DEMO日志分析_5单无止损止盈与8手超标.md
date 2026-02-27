# DEMO 日志分析：5 单无止损止盈、8 手超标

## 分析对象

- **日志文件**：`demo_run_20h_20260129_103052.log`（约 370 万行，2026-01-29 10:31 起）
- **运行方式**：`run_moe_demo.py` → `python src/tiger1.py d moe` → **TradingExecutor 架构**（非 tiger1 内部的 place_tiger_order 循环）

---

## 1. 执行路径：走的是 OrderExecutor，不是 place_tiger_order

- 日志里反复出现：
  - `动作: 买入, 置信度: 0.430`
  - `self.order_executor.execute_buy(...)`（`trading_executor.py` → `order_executor.py`）
- **结论**：DEMO 用的是 **TradingExecutor + OrderExecutor** 这条链路，**没有走** `tiger1.place_tiger_order`。
- **影响**：
  - `place_tiger_order` 里「主单成功后自动下止损单/止盈单」的逻辑**不会执行**。
  - OrderExecutor 内部只下**主单**（限价买/卖），没有向交易所提交 STP/LMT 的止损、止盈单。
  - 因此日志里 **grep「止损」「止盈」整份 log 无匹配** 是符合这条路径的。

---

## 2. 风控报错：`__main__` 没有 `check_risk_control`

- 典型堆栈：
  ```
  AttributeError: module '__main__' has no attribute 'check_risk_control'
    ...
    order_executor.execute_buy(...)
    order_executor.py line 64: if not self.risk_manager.check_risk_control(price, 'BUY'):
  ```
- **含义**：OrderExecutor 的 `risk_manager` 指向的是 `__main__`（即运行中的 tiger1 脚本模块），但在调用时 `__main__` 上拿不到 `check_risk_control`，导致每次买前风控检查就崩。
- **本份 log 中**：未出现「订单提交成功」——说明**所有买入尝试都在风控这一步异常退出**，没有真正下单成功。
- 若其它时间/其它进程在修好该问题或用了不同入口后继续跑，则可能出现：风控通过、主单反复成交，但**依然没有止损/止盈单**（因为仍是 OrderExecutor 路径），从而出现 5 单、8 手等现象。

---

## 3. 仓位上限在日志里的体现

- 日志里多次出现：`网格区间: [...], 最大仓位: 2手`（时段：其他低波动时段）。
- 即：当时**显示**的最大仓位是 2 手；若风控正常生效，理论上不应超过 2 手。
- 实际账户到 8 手，可能原因包括：
  - 其它时段曾把 `GRID_MAX_POSITION` 提到 8/10（时段自适应），风控按 8 放行；
  - 或存在多进程/多次运行，每个进程只看自己的 `current_position`，没有按账户总持仓限制；
  - 或曾有一版代码/入口没有正确做仓位上限检查。

---

## 4. 根因小结

| 现象 | 根因（基于日志与代码） |
|------|------------------------|
| 5 单买入都没有对应止损/止盈单 | DEMO 走 TradingExecutor → OrderExecutor，只下主单，**从未**下 STP/止盈单；tiger1 的 place_tiger_order 主单后自动挂止损/止盈逻辑未执行。 |
| 风控报错 `check_risk_control` | OrderExecutor 用 `__main__` 做 risk_manager，运行时 `__main__` 无 `check_risk_control`，导致买前检查抛错。本 log 中因此未出现「订单提交成功」。 |
| 账户 8 手超标 | 时段自适应曾把最大仓位调到 8/10；和/或多进程/多次运行导致仓位累加；且 OrderExecutor 路径本身不挂止损/止盈，风险更大。 |

---

## 5. 建议修复（与后续动作）

1. **OrderExecutor 风控解析**  
   在 `order_executor.py` 中，不要依赖 `__main__` 是否具备 `check_risk_control`；若 `risk_manager` 没有该方法，则回退到已导入的 `tiger1`（或统一用 `tiger1` 做风控），避免 AttributeError，并保证 DEMO 也能做仓位/风控检查。

2. **OrderExecutor 路径也要挂止损/止盈**  
   在 OrderExecutor.execute_buy 成功提交主单后，仿照 tiger1.place_tiger_order：  
   用同一笔的 `stop_loss_price` / `take_profit_price` 向交易所提交 STP 卖单和 LMT 卖单（或调用 tiger1 的挂单封装），保证「每笔买入都有对应止损和止盈单」。

3. **DEMO 仓位硬顶**  
   已做：DEMO 一律用 `DEMO_MAX_POSITION=3`，不随时段放大；风控与时段赋值处都按此限制。

4. **可靠性：超时止盈/止损必跑**
   - 问题：TradingExecutor 循环里**从未**调用 `check_active_take_profits` / `check_timeout_take_profits`，导致挂了很多手也没有卖出，超时止盈止损不生效；若真实交易方向反了，容易爆仓。
   - 已做：  
     - 在 `tiger1` 中新增 `check_orphan_position_timeout_and_stoploss`（对无止盈单登记的「孤儿持仓」做超时平仓和止损）、以及 `run_position_watchdog`（每轮统一跑：主动止盈 + 超时止盈 + 孤儿超时/止损）。  
     - 在 **TradingExecutor.run_loop** 每轮获取 market_data 后**立即**调用 `run_position_watchdog(tick_price, atr, grid_lower)`，再执行预测与下单。  
   - 效果：有仓必查，超时或触发止损会平仓，避免有仓无卖、裸奔爆仓。

5. **后续排查建议**  
   - 若有其它 DEMO 日志（尤其是出现「订单提交成功」的），可再搜：`订单提交成功`、`止损`、`止盈`、`风控`、`已达上限`，确认是哪个入口、哪段时期下的单，以及当时是否有时段放大仓位、多进程等情况。
   - 启动时可选：从 API 拉取账户实际持仓，若 > 3 则打日志并拒绝继续买入，直到人工处理。
