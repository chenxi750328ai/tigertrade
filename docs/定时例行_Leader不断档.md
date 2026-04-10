# 定时例行 · Leader 不断档

## 先说明白：脚本「触发」的是什么

- Cursor / 云端助手 **没有常驻进程**，关会话就停；**没有任何脚本能直接唤醒云端模型**。
- `leader_routine_tick.sh` 现在会做两件事：**刷新例行数据** + **写入跟进信号 + 可选推送到你的 Webhook/命令**（手机、钉钉、自建服务），见下节。
- 打开 Cursor 后，可看 **`run/leader_routine_agent_signal.json`** 或 **[LEADER_TICK_FOR_CURSOR.md](reports/LEADER_TICK_FOR_CURSOR.md)** 继续干活。

项目第一目标仍是 **[盈利目标和风险控制.md](盈利目标和风险控制.md)**。

### 环境变量（写在仓库根 `.env` 即可）

| 变量 | 作用 |
|------|------|
| `LEADER_ROUTINE_WEBHOOK_URL` | 若设置：每次 tick 后对该 URL **POST JSON**（`event`/`utc_time`/`hint_zh` 等） |
| `LEADER_ROUTINE_NOTIFY_CMD` | 若设置：在仓库根执行该 shell；环境变量 `LEADER_ROUTINE_JSON` 为整段 JSON |

示例（仅示意）：`LEADER_ROUTINE_NOTIFY_CMD='curl -fsS -X POST -H "Content-Type: application/json" -d "$LEADER_ROUTINE_JSON" https://your-n8n/webhook/xxx'`

---

## 方式一：crontab（推荐）

仓库根目录 **不要用写死路径**，用你本机克隆路径替换 `/path/to/tigertrade`。

```cron
# 每 15 分钟：轻量心跳（今日收益 + QA + order 导出 + routine_pulse）
*/15 * * * * /path/to/tigertrade/scripts/leader_routine_tick.sh >> /path/to/tigertrade/logs/leader_routine_tick.log 2>&1

# 每 2 分钟：DEMO/训练监控 + 半点异常单检查（已有脚本）
*/2 * * * * /path/to/tigertrade/scripts/cron_routine_monitor.sh >> /path/to/tigertrade/logs/routine_monitor.log 2>&1

# 每日 QA 报告（与 QA 说明一致时可保留）
0 9 * * * /path/to/tigertrade/scripts/cron_qa_monitor.sh >> /path/to/tigertrade/logs/qa_monitor.log 2>&1
```

`leader_routine_tick.sh` 内部会 `cd` 到仓库根（与 `run_20h_demo.sh` 同类逻辑），可移植。

---

## 方式二：一键启动本机进程（推荐）

```bash
cd /path/to/tigertrade
bash scripts/start_leader_routine_daemon.sh
```

- 用 **`nohup`** 在后台跑 `leader_loop_forever.sh`，并把 **PID 写入** `run/leader_routine_daemon.pid`
- 日志默认 **`logs/leader_routine_daemon.log`**
- 间隔默认 **900 秒**，可：`LEADER_LOOP_INTERVAL_SEC=600 bash scripts/start_leader_routine_daemon.sh`
- 停止：`bash scripts/stop_leader_routine_daemon.sh`

（与 Cursor 是否打开无关；进程在你本机一直跑。）

## 方式三：手动 nohup（与方式二等价）

```bash
cd /path/to/tigertrade
chmod +x scripts/leader_loop_forever.sh
LEADER_LOOP_INTERVAL_SEC=900 nohup bash scripts/leader_loop_forever.sh >> logs/leader_loop_forever.log 2>&1 &
```

停止：`pkill -f leader_loop_forever.sh`（注意勿误杀其他进程）。

---

## 方式四：CI / 其他环境

若只在 GitHub Actions 跑，用 **scheduled workflow** 调用 `ROUTINE_PULSE_REFRESH=1 python scripts/write_routine_pulse.py`（注意密钥与 API 配额，勿过于频繁）。

---

## 与 STATUS 的对应关系

- 心跳输出：`docs/reports/routine_pulse_latest.md`
- 订单状态：`docs/order_execution_status.json`（由 pulse 刷新链路触发 export）
- 第一目标说明：`docs/STATUS.md` 文首引用

---

**结论**：「不要停」= **本机守护进程 / crontab 跑脚本**；进程里能串联 **门禁、测试、优化、你自己的 hook**，详见 **[Leader例行进程_能力与配置.md](Leader例行进程_能力与配置.md)**（`LEADER_ROUTINE_PROFILE`、`LEADER_ROUTINE_POST_HOOK`）。

打开 Cursor 对话才能做 **需求讨论与多轮改代码**；例行数据与命令 **不需要** 等对话。
