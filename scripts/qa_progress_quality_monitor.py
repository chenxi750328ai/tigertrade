#!/usr/bin/env python3
"""
QA 定时分析：监控项目进度与质量，持续发现问题。
- 读取报告/STATUS/计划，与项目目标对比
- 扫描 TODO/FIXME、稳定性结果
- 输出 docs/reports/qa_monitor_latest.md，并可选追加日志
用法: python scripts/qa_progress_quality_monitor.py [--append-log]
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

# 项目目标（与 README/需求分析一致）
GOALS = {
    "win_rate_min": 60.0,       # 胜率 > 60%
    "backtest_return_phase1": 15.0,  # 第1周回测盈利率 > 15%
    "monthly_return_target": 20.0,  # 月盈利率 20%
    "sharpe_min": 2.0,
    "max_drawdown_pct_max": 10.0,
    "profit_loss_ratio_min": 2.0,
    "stability_error_rate_max": 1.0,  # 20h 错误率 < 1%
}

ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "docs" / "reports"
DOCS_DIR = ROOT / "docs"


def load_report_json() -> dict | None:
    p = REPORTS_DIR / "algorithm_optimization_report.json"
    if not p.is_file():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_stability_stats() -> dict | None:
    for loc in [ROOT / "stability_stats.json", ROOT / "run" / "stability_stats.json"]:
        if loc.is_file():
            try:
                with open(loc, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
    return None


def scan_src_todos() -> list[tuple[str, int, str]]:
    out = []
    pattern = re.compile(r"(TODO|FIXME|XXX|HACK)\s*[:\-]?\s*(.+)")
    for py in (ROOT / "src").rglob("*.py"):
        try:
            text = py.read_text(encoding="utf-8", errors="ignore")
            for i, line in enumerate(text.splitlines(), 1):
                m = pattern.search(line)
                if m:
                    out.append((str(py.relative_to(ROOT)), i, line.strip()[:80]))
        except Exception:
            pass
    return out[:50]  # 最多 50 条


def collect_issues(report: dict | None, stability: dict | None, todos: list) -> list[dict]:
    issues = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    if report:
        prof = report.get("profitability") or {}
        win_rate = prof.get("win_rate")
        if win_rate is not None and win_rate < GOALS["win_rate_min"]:
            issues.append({
                "level": "high",
                "area": "报告指标",
                "title": "实盘胜率未达目标",
                "detail": f"当前 {win_rate:.1f}% < 目标 {GOALS['win_rate_min']}%",
                "suggestion": "见 QA_从项目目标把控质量：报告须做达标判定并改进策略/执行。",
            })
        sp = report.get("strategy_performance") or {}
        for name, data in sp.items():
            rp = data.get("return_pct")
            if rp is not None and isinstance(rp, (int, float)) and rp < GOALS["backtest_return_phase1"] and data.get("num_trades", 0) > 0:
                issues.append({
                    "level": "medium",
                    "area": "回测指标",
                    "title": f"回测收益率未达第1周目标（{name}）",
                    "detail": f"return_pct={rp}% < {GOALS['backtest_return_phase1']}%",
                    "suggestion": "阶段性目标回测 > 15%，需优化参数或数据。",
                })
        # 夏普/回撤/盈亏比缺失或为 0
        has_sharpe = any((sp.get(s) or {}).get("sharpe_ratio") for s in sp)
        if not has_sharpe or all((sp.get(s) or {}).get("sharpe_ratio") in (0, None, "") for s in sp):
            issues.append({
                "level": "medium",
                "area": "报告指标",
                "title": "报告未产出有效夏普/回撤/盈亏比",
                "detail": "KPI 需与目标（夏普>2、回撤<10%、盈亏比>2:1）对照。",
                "suggestion": "在报告或脚本中计算并展示，并在 QA 监控中做达标判定。",
            })
    else:
        issues.append({
            "level": "medium",
            "area": "报告",
            "title": "未找到算法优化报告 JSON",
            "detail": "docs/reports/algorithm_optimization_report.json 不存在或无法读取。",
            "suggestion": "运行 optimize_algorithm_and_profitability.py 生成报告。",
        })

    if stability:
        total = (stability.get("successful_iterations") or 0) + (stability.get("failed_iterations") or 0)
        if total > 0:
            err_rate = 100.0 * (stability.get("failed_iterations") or 0) / total
            if err_rate > GOALS["stability_error_rate_max"]:
                issues.append({
                    "level": "high",
                    "area": "20h 稳定性",
                    "title": "稳定性错误率超标",
                    "detail": f"错误率 {err_rate:.2f}% > {GOALS['stability_error_rate_max']}%",
                    "suggestion": "检查 stability_test.log 与 DEMO 日志，修复后重跑。",
                })
    else:
        # 不强制报错，仅提示
        issues.append({
            "level": "low",
            "area": "20h 稳定性",
            "title": "未发现本次 20h 稳定性结果",
            "detail": "stability_stats.json 未在项目根或 run/ 下找到。",
            "suggestion": "若已跑 20h，请将结果落盘到固定路径供监控与门禁使用。",
        })

    if todos:
        issues.append({
            "level": "low",
            "area": "代码",
            "title": f"源码中存在 {len(todos)} 处 TODO/FIXME/XXX",
            "detail": ";\n  ".join([f"{f}:{n}" for f, n, _ in todos[:10]]),
            "suggestion": "建议登记到项目计划或 Issue，避免遗忘。",
        })

    # STATUS 与计划是否近期更新（可选）
    status_md = DOCS_DIR / "STATUS.md"
    plan_md = DOCS_DIR / "项目计划_月度周计划.md"
    if status_md.is_file():
        mtime = status_md.stat().st_mtime
        age_days = (datetime.now().timestamp() - mtime) / 86400
        if age_days > 14:
            issues.append({
                "level": "low",
                "area": "进度",
                "title": "STATUS.md 超过 14 天未更新",
                "detail": f"约 {int(age_days)} 天未修改。",
                "suggestion": "定期更新当前状态与计划，便于监控对齐。",
            })

    return issues


def render_md(issues: list[dict], report_ts: str | None, stability_ts: str | None) -> str:
    lines = [
        "# QA 进度与质量监控报告",
        "",
        f"**生成时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "**说明**：本报告由 `scripts/qa_progress_quality_monitor.py` 定时生成，从项目目标出发对比报告与运行结果，持续发现问题。",
        "",
        "---",
        "",
        "## 一、数据来源",
        "",
        f"- 算法报告：`docs/reports/algorithm_optimization_report.json`" + (f"（时间戳：{report_ts}）" if report_ts else "（未找到）"),
        f"- 20h 稳定性：`stability_stats.json`" + (f"（存在）" if stability_ts else "（未找到）"),
        "- 源码：`src/` 下 TODO/FIXME/XXX 扫描",
        "- 进度：`docs/STATUS.md`、`docs/项目计划_月度周计划.md` 最后修改时间",
        "",
        "---",
        "",
        "## 二、项目目标（门禁基准）",
        "",
        "| 指标 | 目标值 |",
        "|------|--------|",
        f"| 胜率 | > {GOALS['win_rate_min']}% |",
        f"| 第1周回测盈利率 | > {GOALS['backtest_return_phase1']}% |",
        f"| 月盈利率 | {GOALS['monthly_return_target']}% |",
        f"| 夏普比率 | > {GOALS['sharpe_min']} |",
        f"| 最大回撤 | < {GOALS['max_drawdown_pct_max']}% |",
        f"| 盈亏比 | > {GOALS['profit_loss_ratio_min']}:1 |",
        f"| 20h 错误率 | < {GOALS['stability_error_rate_max']}% |",
        "",
        "---",
        "",
        "## 三、发现的问题",
        "",
    ]
    if not issues:
        lines.append("本次未发现新问题。")
    else:
        for i, q in enumerate(issues, 1):
            level = q.get("level", "medium")
            badge = {"high": "🔴 高", "medium": "🟡 中", "low": "🟢 低"}.get(level, level)
            lines.append(f"### {i}. [{badge}] {q.get('title', '')}")
            lines.append("")
            lines.append(f"- **维度**：{q.get('area', '')}")
            lines.append(f"- **说明**：{q.get('detail', '')}")
            lines.append(f"- **建议**：{q.get('suggestion', '')}")
            lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*由 QA 定时分析脚本生成；建议每日或与例行一并运行，结果供陈正霞与项目组跟进。*")
    return "\n".join(lines)


def main():
    append_log = "--append-log" in sys.argv
    report = load_report_json()
    report_ts = report.get("timestamp") if report else None
    stability = load_stability_stats()
    stability_ts = "yes" if stability else None
    todos = scan_src_todos()
    issues = collect_issues(report, stability, todos)
    md = render_md(issues, report_ts, stability_ts)
    out_path = REPORTS_DIR / "qa_monitor_latest.md"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"✅ 已写入: {out_path}")
    print(f"   发现问题: {len(issues)} 条")
    if append_log:
        log_path = ROOT / "logs" / "qa_monitor.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().isoformat()}] issues={len(issues)} path={out_path}\n")
    return 0 if len([i for i in issues if i.get("level") == "high"]) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
