# Leader 例行「跟进信号」（给 Cursor / 人）

定时任务跑 `scripts/leader_routine_tick.sh` 后，会生成：

| 路径 | 说明 |
|------|------|
| `run/leader_routine_agent_signal.json` | 完整 JSON（`run/` 通常不提交） |
| `docs/reports/leader_routine_agent_signal.json` | 同上副本（已 `.gitignore`，避免刷屏） |

打开仓库时若需「接着干」：**读上述 JSON 的 `hint_zh`**，并对照 `routine_pulse_latest.md`、订单状态继续 Leader 工作。

可选：在 `.env` 配置 `LEADER_ROUTINE_WEBHOOK_URL` 或 `LEADER_ROUTINE_NOTIFY_CMD`，把 tick 推到你手机/钉钉/n8n（**仍不能代替 Cursor 云端常驻**，但能叫醒人）。

详见：`docs/定时例行_Leader不断档.md`。
