# 状态页刷新规范（Agent 必读）

**标签**: status, pages, push, routine

---

## 原则

- 状态页（`docs/status.html`）更新后，**必须提交并推送成功**，Pages 才会生效。
- **Agent 自己提交时，要保证 push 成功**；否则用户不在就刷新不了。
- **Agent 自己执行推送，不等待用户**；重试直至成功。仅在多次失败后才告知用户本机执行。

## Agent 执行要求

1. 修改 `docs/status.html` 后：`git add docs/status.html`，`git commit -m "..."`。
2. **推送**：**由 Agent 自己执行** `git push origin main`，**重试直至成功**（不等待用户；多轮重试）。
3. 仅当本环境多次推送仍失败时：在回复中**明确写出**请用户本机执行：
   ```bash
   cd /home/cx/tigertrade && git push origin main
   ```
   并说明「提交已落盘，仅差推送；您执行上述命令后 Pages 即更新」。

## 每日例行区块

刷新时顺带更新「每日例行（最近一次）」表中各行的**最近一次**日期与**状态**（已跑/运行中/进行中/未跑），再提交并推送。

---

参见：`例行工作清单_agent必读.md` 第 7 项。
