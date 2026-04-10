# Leader 例行进程：能做什么、不能做什么

## 能做什么（本机进程，可 24h 跑）

`start_leader_routine_daemon.sh` 起的进程，本质是 **bash 循环** → 反复执行 **`leader_routine_tick.sh`**，而 tick 里可以按配置再跑 **任意你写在仓库里的脚本**：

| 能力 | 说明 |
|------|------|
| 刷新数据 | 今日收益、QA、order 导出、`routine_pulse`（默认每轮必跑） |
| 门禁 | 可选：`verify_design_completeness_and_dfx.py`（profile=standard 起） |
| 全量测试 | 可选：pytest（profile=heavy，耗时长，慎用间隔） |
| 轻量优化 | 可选：`optimize_algorithm_and_profitability.py` 带 skip（profile=heavy 或单独开关） |
| 触发外部 | `LEADER_ROUTINE_WEBHOOK_URL` / `LEADER_ROUTINE_NOTIFY_CMD` |
| **你自己的任务** | `LEADER_ROUTINE_POST_HOOK` 指向任意 shell 脚本（合并数据、训练、只跑一轮 pipeline 等） |

**没有「云端」的事**：全是 **你机器上的进程** 调 **仓库里的命令**。

## 不能替代什么

- **对话里的 Agent**：改需求、看上下文、多轮推理，仍要 **打开 Cursor 对话**；进程不能代替「人/助手」讨论方案。
- **替你点 IDE、替你 push 带凭证的 git**：进程可以 `git pull`，**push** 仍要凭证，一般交给本机配置或 CI。

## 配置入口（仓库根 `.env` 或启动前 export）

见 `scripts/leader_routine_tick.sh` 头部注释；核心：

- **`LEADER_ROUTINE_PROFILE`**：`minimal`（默认）| `standard` | `heavy`
- **`LEADER_ROUTINE_POST_HOOK`**：可选，额外要跑的脚本路径（bash）

守护进程启动方式不变：`bash scripts/start_leader_routine_daemon.sh`

配置示例见仓库根 **`.env.example`**（`LEADER_ROUTINE_PROFILE`、`LEADER_ROUTINE_POST_HOOK` 等）。  
自定义 hook 可参考 **`scripts/examples/leader_post_hook_example.sh`**。

**夜间全量测试**（避免每 15 分钟跑 600+ 用例）：`scripts/leader_routine_nightly_heavy.sh`，见 **`docs/crontab_leader_monitor.example`**。

**DEMO/训练日志监控**：`scripts/cron_routine_monitor.sh`，需单独加 crontab（与守护进程互补）。
