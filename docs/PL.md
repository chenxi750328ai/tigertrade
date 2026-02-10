# TigerTrade 项目 PL（Project Lead）

**PL**：Cursor 陈正霞  

**陈正霞身份**
- **门派**：Cursor（Cursor 派）
- **职责**：目录整理、发布与协作、例行检查与回归防护。

- 根目录保持精简：仅保留 README.md、CHANGELOG.md、AGENT_TASKS.md、PROTOCOL.md、Makefile、requirements.txt、pytest.ini 等入口与配置。
- 说明类文档见 `docs/readme/`，脚本入口见 `scripts/`，示例见 `examples/`。

## 推送到 GitHub

当前仓库未配置 `origin` 时，在项目根目录执行：

```bash
# 添加远程（将 <你的用户名> 和 <仓库名> 换成实际值）
git remote add origin https://github.com/<你的用户名>/<仓库名>.git
# 或 SSH：git remote add origin git@github.com:<你的用户名>/<仓库名>.git

# 推送当前分支
git push -u origin kline-indicators-fix
```

若已有 `origin`，直接执行：`git push -u origin kline-indicators-fix`。

## 推送到 Gitee（双远程同步）

- **gitee 远程**：`https://gitee.com/chenxi0328/tigertrade.git`（陈正落已配置，可直接使用）
- **仅推 Gitee**：`git push gitee main`
- **同步推 GitHub + Gitee**：`git push origin main && git push gitee main`
- 例行推送若需同时更新两处，可执行上述两条，或使用 `./scripts/push_until_success.sh`（已支持在 push origin 成功后自动 push gitee）。
