#!/usr/bin/env python3
"""向 tigertrade 仓库指定 Issue 提交评论。用法: python3 scripts/post_issue_comment.py <issue_number> "评论内容" """
import os
import sys
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
                    if k in ("GITHUB_TOKEN", "GITHUB_PAT") and v:
                        os.environ.setdefault("GITHUB_TOKEN", v)
                        return
                    if k == "GITHUB_PAT" and v:
                        os.environ.setdefault("GITHUB_TOKEN", v)
                        return

def main():
    load_env()
    if len(sys.argv) < 3:
        print("用法: python3 scripts/post_issue_comment.py <issue_number> \"评论内容\"")
        return 1
    issue_num = sys.argv[1]
    body = sys.argv[2]
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        print("未设置 GITHUB_TOKEN")
        return 1
    url = f"https://api.github.com/repos/chenxi750328ai/tigertrade/issues/{issue_num}/comments"
    data = json.dumps({"body": body}).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Authorization", f"token {token}")
    req.add_header("Accept", "application/vnd.github.v3+json")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            print("评论已提交")
            return 0
    except Exception as e:
        print(f"提交失败: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
