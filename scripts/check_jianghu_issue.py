#!/usr/bin/env python3
"""查询 agent-jianghu Issue #1 状态与评论数，用于跟踪远程接入 Redis 的处理情况。"""
import json
import urllib.request

ISSUE_URL = "https://api.github.com/repos/chenxi750328ai/agent-jianghu/issues/1"

def main():
    req = urllib.request.Request(ISSUE_URL)
    req.add_header("Accept", "application/vnd.github.v3+json")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            state = data.get("state", "?")
            title = data.get("title", "")[:60]
            comments = data.get("comments", 0)
            updated = data.get("updated_at", "")[:10]
            html_url = data.get("html_url", "")
            print(f"Issue #1: {title}...")
            print(f"  状态: {state}  |  评论数: {comments}  |  更新: {updated}")
            print(f"  链接: {html_url}")
    except Exception as e:
        print(f"查询失败: {e}")

if __name__ == "__main__":
    main()
