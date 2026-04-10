#!/usr/bin/env python3
"""检查当前 GITHUB_TOKEN 对指定仓库是否有 push（与 submit_jianghu_issue 读同一套 .env）。"""
import json
import os
import sys
import urllib.request
from pathlib import Path

DEFAULT_REPO = "chenxi750328ai/tigertrade"


def load_token():
    for p in [
        Path(__file__).resolve().parents[1] / ".env",
        Path("/home/cx/tigertrade/.env"),
        Path("/home/cx/agentfuture/.env"),
        Path("/home/cx/.env"),
    ]:
        if not p.exists():
            continue
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k in ("GITHUB_TOKEN", "GITHUB_PAT") and v:
                    return v
    return os.environ.get("GITHUB_TOKEN", "").strip()


def main():
    repo = os.environ.get("GITHUB_CHECK_REPO", DEFAULT_REPO)
    token = load_token()
    if not token:
        print("未找到 GITHUB_TOKEN（tigertrade/.env、agentfuture/.env 等）")
        return 2
    url = f"https://api.github.com/repos/{repo}"
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"token {token}")
    req.add_header("Accept", "application/vnd.github+json")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            d = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode()[:300]}")
        return 1
    perms = d.get("permissions") or {}
    login_url = urllib.request.Request("https://api.github.com/user")
    login_url.add_header("Authorization", f"token {token}")
    login_url.add_header("Accept", "application/vnd.github+json")
    with urllib.request.urlopen(login_url, timeout=15) as r:
        u = json.loads(r.read().decode())
    login = u.get("login", "?")
    push = perms.get("push")
    print(f"Token 对应用户: {login}")
    print(f"仓库: {repo}")
    print(f"API permissions.push: {push}")
    if push:
        print("结论: 可用 HTTPS + PAT 执行 git push（勿泄露 token）。")
        return 0
    print(
        "结论: 当前令牌对该仓库仅有读权限（与江湖 Issue 不同：Issue 可能只在 agent-jianghu 上开了写 issue）。"
        "\n处理: 在 GitHub 给该用户加 tigertrade 的 Write，或使用仓库所有者账号的 PAT（精细令牌须勾选本仓库 contents:write）。"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
