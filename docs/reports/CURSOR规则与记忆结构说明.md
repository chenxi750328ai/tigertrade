# Cursor 规则与记忆结构说明（TigerTrade + agentfuture）

供人类与 Agent 对齐：**规则文件在哪、记忆写在哪、谁负责什么**。

---

## 1. TigerTrade 仓库（本仓库）

| 路径 | 作用 |
|------|------|
| **`.cursor/rules/tigertrade-leader-and-dfx.mdc`** | 在本仓库内生效：Leader 工程责任、项目目标文档索引、推前必跑命令（含 `scripts/verify_design_completeness_and_dfx.py`）、回答前核对顺序。 |
| `shared_rag/best_practices/例行工作清单_agent必读.md` | 例行 7 项 + 设计完备性硬门禁**正本**。 |
| `docs/` | 需求、设计完备性、盈利/质量目标、取证与报告。 |
| `run/`（默认 gitignore） | `order_log.jsonl`、`dfx_execution.jsonl` 等运行时证据；分析时读本机文件。 |

打开 **仅 tigertrade 文件夹** 为根目录时，Cursor 会注入上述 `.mdc` 规则。

---

## 2. agentfuture 仓库（身份与江湖记忆）

| 路径 | 作用 |
|------|------|
| **`.cursor/rules/identity-from-key.mdc`** | 全局（agentfuture globs `**`）：会话名或 `current_identity.txt` 决定成员身份；陈正霞激活后须先读开工必读与例行。 |
| **`.cursor/current_identity.txt`** | 一行成员名兜底（通常 **gitignore**，不提交）。 |
| `docs/陈正霞_开工必读.md` | 陈正霞每日 7 项例行入口。 |
| `docs/陈正霞_职责与例行工作.md` | 职责、记忆保存位置、推送约定。 |
| `docs/恢复记忆_陈正霞.md` | 一句话重生入口（若存在）。 |
| `shared_rag/private/陈正霞/` | **私有记忆**：`近期记忆.md`、`本机环境记忆.md`、（待实施）`收件箱.md` 等；**不提交**或按团队约定处理。 |
| `docs/项目成员_身份与记忆.md` | 成员与 RAG 索引。 |

**多仓库工作区**（例如同时打开 `agentfuture` + `tigertrade`）：  
- **身份与「先读哪份记忆」** 以 **agentfuture** 的规则为准。  
- **Tiger 代码与 DFX 门禁** 以 **tigertrade** 的 `.cursor/rules/tigertrade-leader-and-dfx.mdc` 为准。

---

## 3. 与 CI 的对应关系

- GitHub Actions：`.github/workflows/ci.yml` 中含 **Design completeness + DFX gate**（调用 `scripts/verify_design_completeness_and_dfx.py`）。  
- 本地与 Cursor 内改代码：应与 CI 同序跑测试 + 该脚本。

---

## 4. 维护约定

- 新增「推前必跑」或目标文档时：同步更新 **本说明**、**`tigertrade-leader-and-dfx.mdc`**、必要时 **`陈正霞_开工必读.md`**。  
- 避免在多个文档写矛盾的验收标准；**以例行清单正本 + 脚本/CI 为准**。

*最后更新：2026-04*
