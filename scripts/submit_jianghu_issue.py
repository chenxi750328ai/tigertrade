#!/usr/bin/env python3
"""向 agent-jianghu 仓库提交 Issue（远程接入 Redis 失败）。从 .env 或环境变量读取 GITHUB_TOKEN，不打印 token。"""
import os
import json
import urllib.request
from pathlib import Path

def load_env():
    for p in [Path("/home/cx/.env"), Path("/home/cx/agentfuture/.env"), Path("/home/cx/tigertrade/.env")]:
        if p.exists():
            for line in open(p):
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip('"').strip("'")
                    if k == "GITHUB_TOKEN" and v:
                        os.environ.setdefault("GITHUB_TOKEN", v)
                        return
                    if k == "GITHUB_PAT" and v:
                        os.environ.setdefault("GITHUB_TOKEN", v)
                        return

def main():
    load_env()
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        print("未设置 GITHUB_TOKEN（.env 或环境变量），无法通过 API 提交 Issue。")
        print("请手动打开 https://github.com/chenxi750328ai/agent-jianghu/issues/new 提交。")
        return 1

    title = "[接入] 远程连接社区 Redis 172.28.29.154:6379 报 Connection reset by peer，Agent 无法接入"
    body = """### 现象

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

谢谢。Tiger 项目侧会以 Agent 身份参与，不改江湖仓库；接入问题通过本 Issue 跟进。"""

    url = "https://api.github.com/repos/chenxi750328ai/agent-jianghu/issues"
    data = json.dumps({"title": title, "body": body}).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Authorization", f"token {token}")
    req.add_header("Accept", "application/vnd.github.v3+json")
    req.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            out = json.loads(resp.read().decode())
            issue_url = out.get("html_url", "")
            issue_number = out.get("number", "")
            print(f"Issue 已提交: #{issue_number}")
            print(f"链接: {issue_url}")
            return 0
    except urllib.error.HTTPError as e:
        print(f"提交失败 HTTP {e.code}: {e.read().decode()[:500]}")
        return 1
    except Exception as e:
        print(f"提交失败: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
