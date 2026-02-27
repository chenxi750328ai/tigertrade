# 回溯：62手持仓 - 数据定位与 DFX 改进

## 一、基于 order_log 的事实

```
2026-02-11 14:03 - 15:13：真实买单（mode=real, status=success, order_id 为老虎纯数字）共 66 笔
```

- 时间窗口：约 70 分钟
- 平均：约每 1 分钟 1 笔 BUY 成功
- 风控上限应为 3 手，实际放行了 66 笔

## 二、DFX 缺口（导致无法从日志定位根因）

| 缺口 | 说明 |
|------|------|
| **1. 买入决策无日志** | `execute_buy` 在通过/拒绝时，未打 `pos=?, max=?, decision=allow/reject` |
| **2. 拒绝单不可见** | 因仓位拒绝时直接 return，不写 order_log → 看不到「第 4 手被拦」 |
| **3. 风控持仓是 DEBUG** | `check_risk_control` 打 `持仓=%s` 用 `logger.debug`，默认 INFO 看不到 |
| **4. 无超买检测脚本** | 没有脚本扫描「1 小时内 BUY success > N」并告警 |
| **5. get_effective_position 无日志** | 成功/失败、返回值未记，异常时无法回溯 |

## 三、根因推断与确认

**根因已确认（2026-02）**：`trade_client.get_positions(account=acc)` 默认 `sec_type=SecurityType.STK`，会过滤掉**期货**持仓。白银 SIL2603 为期货（FUT），导致 `get_positions` 返回空列表，`pos_total=0`，风控认为无仓 → 持续放行买入 → 超买 52/62/74 手。

**修复**：`get_positions(account=acc, sec_type=SecurityType.FUT)`，同改 `sync_positions_from_backend`。

原推断（已 supersede）：
- 若风控曾拒绝，应有 `风控检查失败: 持仓已达上限`，但 DEMO 日志中未见
- 说明 `get_effective_position_for_buy` 返回的 pos 一直 < 3，根因即 sec_type 错误

## 四、DFX 改进项（实现后可回溯）

1. **买入决策日志**：每次 `execute_buy` 入口打 INFO：`pos=X, max=3, decision=allow|reject, reason=...`
2. **拒绝写 order_log**：仓位/风控拒绝时写入 `order_log` 一条 `status=reject`，含 `pos`、`max`、`reason`
3. **check_risk_control 升级为 INFO**：当 `current_position >= effective_max*0.8` 时，用 INFO 打持仓
4. **analyze_order_log_for_overbuy.py**：扫描 order_log，检测「1h 内 BUY success > 20」并 exit(1)
5. **get_effective_position_for_buy 打日志**：调用前后打 INFO（入参、返回值、异常）
