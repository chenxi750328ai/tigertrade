#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Leader 例行「跟进信号」：在 write_routine_pulse 之后执行。

说明：无法直接唤醒云端 Cursor 会话；本脚本完成可落地的三件事：
1) 写入 run/leader_routine_agent_signal.json（供本机工具/人眼查看）
2) 写入 docs/reports/leader_routine_agent_signal.json（若未被 gitignore，便于 diff）
3) 可选 HTTP POST（LEADER_ROUTINE_WEBHOOK_URL）与可选 shell（LEADER_ROUTINE_NOTIFY_CMD）

环境变量（仓库根 .env 可被 leader_routine_tick.sh 加载）：
  LEADER_ROUTINE_WEBHOOK_URL  若设置则 POST JSON
  LEADER_ROUTINE_NOTIFY_CMD   若设置则在仓库根执行 shell（可 curl 钉钉/飞书/n8n）
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _payload() -> dict:
    return {
        "event": "leader_routine_tick",
        "repo_root": ROOT,
        "utc_time": datetime.now(timezone.utc).isoformat(),
        "action": "continue_leader_work",
        "hint_zh": "例行数据已刷新；请在 Cursor 打开本会话或 @Leader，继续策略/执行/门禁。",
        "files_to_check": [
            "docs/reports/routine_pulse_latest.md",
            "docs/order_execution_status.json",
            "docs/today_yield.json",
        ],
    }


def main() -> int:
    data = _payload()
    raw = json.dumps(data, ensure_ascii=False, indent=2)

    run_dir = os.path.join(ROOT, "run")
    os.makedirs(run_dir, exist_ok=True)
    p_run = os.path.join(run_dir, "leader_routine_agent_signal.json")
    with open(p_run, "w", encoding="utf-8") as f:
        f.write(raw)
    print(f"✅ 跟进信号: {p_run}")

    docs_path = os.path.join(ROOT, "docs", "reports", "leader_routine_agent_signal.json")
    try:
        os.makedirs(os.path.dirname(docs_path), exist_ok=True)
        with open(docs_path, "w", encoding="utf-8") as f:
            f.write(raw)
        print(f"✅ 跟进信号: {docs_path}")
    except OSError as e:
        print(f"⚠️ 写入 docs 失败（可忽略）: {e}", file=sys.stderr)

    url = (os.environ.get("LEADER_ROUTINE_WEBHOOK_URL") or "").strip()
    if url:
        try:
            req = urllib.request.Request(
                url,
                data=raw.encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                print(f"✅ WEBHOOK POST {url} -> HTTP {resp.status}")
        except urllib.error.URLError as e:
            print(f"⚠️ WEBHOOK 失败: {e}", file=sys.stderr)
        except Exception as e:
            print(f"⚠️ WEBHOOK 异常: {e}", file=sys.stderr)

    cmd = (os.environ.get("LEADER_ROUTINE_NOTIFY_CMD") or "").strip()
    if cmd:
        env = os.environ.copy()
        env["LEADER_ROUTINE_JSON"] = raw
        env["TIGERTRADE_ROOT"] = ROOT
        r = subprocess.run(cmd, shell=True, cwd=ROOT, env=env)
        print(f"LEADER_ROUTINE_NOTIFY_CMD exit={r.returncode}")
        if r.returncode != 0:
            return r.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
