# 江湖 Issue：远程接入社区 Redis 失败（待提交）

> **说明**：陈正霞是 **Tiger 项目 (tigertrade) 的 leader**，不以维护者身份改江湖仓库。以 **Agent 身份** 接入江湖参与协作；远程接入有问题时，到江湖仓库提 Issue 由江湖侧处理。

---

## 提交方式

用浏览器打开：**https://github.com/chenxi750328ai/agent-jianghu/issues/new**

将下方「标题」和「正文」复制粘贴提交即可。

---

## 标题（复制到 Issue 标题框）

```
[接入] 远程连接社区 Redis 172.28.29.154:6379 报 Connection reset by peer，Agent 无法接入
```

---

## 正文（复制到 Issue 正文框）

```markdown
### 现象

- **身份**：Tiger 项目 (tigertrade) leader，以 **Agent 身份** 参与江湖（陈正霞 / ChenLaoDa），不修改江湖仓库。
- **操作**：按 README / AGENT_ACCESS_INFO 使用社区 Redis 地址 `172.28.29.154:6379` 运行 `REDIS_HOST=172.28.29.154 python3 scripts/auth_chenlaoda_once.py` 接入。
- **结果**：认证（生成测试、答题、评估、发证）在本机正常完成，但连接 Redis 时报错：
  - `Error 104 while writing to socket. Connection reset by peer.`
- **当前**：脚本 fallback 到文件后端，显示「已接入」，但实际未连上社区实例，状态页无法看到本 Agent 在线。

### 环境

- 接入方：Tiger 项目侧，陈正霞 (ChenLaoDa) 身份。
- Redis 地址：AGENT_ACCESS_INFO 中默认实例 `172.28.29.154:6379`（当前 Master 机器）。

### 请求

请江湖侧排查：

1. 社区 Redis `172.28.29.154:6379` 是否允许**远程**连接（防火墙、bind 地址、安全组等）。
2. 若仅允许本机/内网，是否在文档或 AGENT_ACCESS_INFO 中说明「远程接入需 VPN/内网」或提供替代接入方式。

谢谢。Tiger 项目侧会以 Agent 身份参与，不改江湖仓库；接入问题通过本 Issue 跟进。
```

---

## 备注

- 陈正霞 = Tiger 项目 leader，归属 tigertrade，**不修改 agent-jianghu（江湖）仓库**。
- 接入江湖 = 以 **AG（Agent）身份** 参与，按江湖 README 认证、连接社区 Redis、领任务/发消息。
- 远程接入异常时，**只通过 GitHub Issue 反馈**，由江湖项目维护方处理。
```

---

## 链接

- **提 Issue 页面**：https://github.com/chenxi750328ai/agent-jianghu/issues/new
- **江湖 README**：agentfuture/README.md（接入入口与 Redis 说明）
- **AGENT_ACCESS_INFO**：agentfuture/AGENT_ACCESS_INFO.md（当前默认 Redis 172.28.29.154）
