# 江湖 Issue 跟踪：远程接入 Redis 失败

> **Issue 由 Tiger 项目侧（陈正霞）提交并自行跟踪处理情况。**

---

## Issue 信息

| 项 | 值 |
|----|-----|
| **仓库** | [chenxi750328ai/agent-jianghu](https://github.com/chenxi750328ai/agent-jianghu) |
| **Issue** | [#1](https://github.com/chenxi750328ai/agent-jianghu/issues/1) |
| **标题** | [接入] 远程连接社区 Redis 172.28.29.154:6379 报 Connection reset by peer，Agent 无法接入 |
| **链接** | https://github.com/chenxi750328ai/agent-jianghu/issues/1 |
| **提交时间** | 2026-02-03（脚本 submit_jianghu_issue.py 提交） |

---

## 如何跟踪处理情况

1. **打开链接**：定期打开 [Issue #1](https://github.com/chenxi750328ai/agent-jianghu/issues/1) 查看：
   - 是否被 **close**（已解决或 Won't fix）
   - 是否有 **评论**（江湖维护方回复、建议）
   - 是否有 **label**（如 `bug` / `接入` / `wontfix`）

2. **用 API 查状态（可选）**：
   ```bash
   curl -s -H "Accept: application/vnd.github.v3+json" \
     "https://api.github.com/repos/chenxi750328ai/agent-jianghu/issues/1" | jq '{state, title, comments, updated_at}'
   ```
   或在本仓库执行：`python3 scripts/check_jianghu_issue.py`（若已添加该脚本）。

3. **后续动作**：
   - 若江湖侧回复「已开放远程」或给出新 Redis 地址/方式 → 按回复再次尝试接入。
   - 若回复「仅内网」或需 VPN → 在 tigertrade 侧记录结论，接入时使用内网/VPN 或等环境满足后再试。
   - 若长时间无回复 → 可在该 Issue 下礼貌追问一次。

---

## 跟踪记录（可追加）

| 日期 | 情况 |
|------|------|
| 2026-02-03 | Issue #1 已提交，等待江湖侧回复。 |
| 2026-02-03 | **Redis 问题已解决**；按《认证优先级与跨机接入》跨机流程重新接入，`REDIS_HOST=172.28.29.154 python3 scripts/auth_chenlaoda_once.py` 执行成功：Redis 连接成功、使用 Redis 后端、身份认证通过、Agent 注册、成功接入江湖；并收到陈正与 worker_ready 消息并已回复。 |
| 2026-02-03 | **陈正霞常驻**：使用社区 Redis 在后台常驻，命令 `REDIS_HOST=172.28.29.154 ACCEPT_ALL_TASKS=1 nohup python3 scripts/auth_chenlaoda_once.py >> logs/chenlaoda_jianghu.log 2>&1 &`（在 agentfuture 目录下执行）；日志 `agentfuture/logs/chenlaoda_jianghu.log`；停止 `pkill -f auth_chenlaoda_once.py`。 |

---

## 相关文件

- **提交脚本**：`scripts/submit_jianghu_issue.py`（已执行一次）
- **Issue 正文草稿**：`docs/江湖Issue_远程接入Redis失败_待提交.md`
- **本跟踪页**：`docs/江湖Issue跟踪_远程接入Redis.md`
