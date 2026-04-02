#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
例行心跳：汇总关键指标写入 docs/reports/routine_pulse_latest.md，便于定期汇报与留痕。
不触发长时间回测；可配合 cron / 人工「继续」时执行。
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read_json(path: str) -> dict | None:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _count_qa_issues(md_path: str) -> str:
    if not os.path.isfile(md_path):
        return "（无 qa_monitor_latest.md）"
    try:
        with open(md_path, "r", encoding="utf-8") as f:
            text = f.read()
        m = re.search(r"发现问题:\s*(\d+)\s*条", text)
        if m:
            return m.group(1)
        # 报告正文无 stdout 行时，按「### N.」计数
        in_section = False
        n = 0
        for line in text.splitlines():
            if line.strip().startswith("## 三、发现的问题"):
                in_section = True
                continue
            if in_section and line.startswith("## ") and "三、" not in line:
                break
            if in_section and re.match(r"^###\s+\d+\.", line.strip()):
                n += 1
        return str(n) if n else "（未解析到条数）"
    except Exception:
        return "（读取失败）"


def main() -> int:
    os.chdir(ROOT)
    today_path = os.path.join(ROOT, "docs", "today_yield.json")
    qa_path = os.path.join(ROOT, "docs", "reports", "qa_monitor_latest.md")
    backend_path = os.path.join(ROOT, "docs", "reports", "backend_positions_analysis.md")
    out_path = os.path.join(ROOT, "docs", "reports", "routine_pulse_latest.md")

    ty = _read_json(today_path) or {}
    qa_issues = _count_qa_issues(qa_path)
    oes_path = os.path.join(ROOT, "docs", "order_execution_status.json")

    pos_line = "（未读 backend_positions_analysis.md）"
    if os.path.isfile(backend_path):
        try:
            with open(backend_path, "r", encoding="utf-8") as f:
                for line in f:
                    if "持仓数量" in line and "**" in line:
                        pos_line = line.strip().lstrip("- ").strip()
                        break
        except Exception:
            pos_line = "（读取 backend 分析失败）"

    # 可选：刷新今日收益与 QA（轻量）
    if os.environ.get("ROUTINE_PULSE_REFRESH", "").strip().lower() in ("1", "true", "yes"):
        subprocess.run(
            [sys.executable, os.path.join(ROOT, "scripts", "update_today_yield_for_status.py")],
            cwd=ROOT,
            check=False,
        )
        subprocess.run(
            [sys.executable, os.path.join(ROOT, "scripts", "qa_progress_quality_monitor.py")],
            cwd=ROOT,
            check=False,
        )
        subprocess.run(
            [sys.executable, os.path.join(ROOT, "scripts", "export_order_log_and_analyze.py")],
            cwd=ROOT,
            check=False,
        )
        ty = _read_json(today_path) or ty
        qa_issues = _count_qa_issues(qa_path)

    oes = _read_json(oes_path) or {}
    oes_line = (
        f"后台今日核对 success={oes.get('real_success_count', '—')}，"
        f"fail={oes.get('real_fail_count', '—')}，"
        f"本地拒={oes.get('local_reject_count', '—')}，API拒={oes.get('api_reject_count', '—')}"
        f"（updated {oes.get('updated', '—')}）"
        if oes
        else "（无 order_execution_status.json，请先跑 optimize 或 export_order_log）"
    )

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# 例行心跳（最新一轮）\n\n")
        f.write(f"- **时间**：{now}\n")
        f.write(f"- **今日收益摘要**：`yield_pct`={ty.get('yield_pct', '—')}；`yield_note`={ty.get('yield_note', '—')}；`source`={ty.get('source', '—')}\n")
        f.write(f"- **QA 监控问题数**：{qa_issues}\n")
        f.write(f"- **后台持仓分析**：{pos_line}\n")
        f.write(f"- **订单执行核对**：{oes_line}\n")
        f.write("\n---\n\n")
        f.write("生成：`python scripts/write_routine_pulse.py`；刷新数据：`ROUTINE_PULSE_REFRESH=1 python scripts/write_routine_pulse.py`\n")

    print(f"✅ 已写入 {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
