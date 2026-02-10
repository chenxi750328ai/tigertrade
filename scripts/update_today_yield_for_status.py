#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
今日收益率写入 docs/today_yield.json，供状态页 status.html 显示。

规则（必守）：DEMO 实盘收益率 = 仅以老虎后台可核对的数据为准。
- 优先从老虎后台拉取订单（fetch_tiger_yield_for_demo.py），有成交则写入；无则写「—」或「需老虎后台数据核对」。
- 不得用 DEMO 日志解析出的百分比作为实盘收益率（日志含模拟单/失败单，会误导）。
- 报告中的 profitability 若来自 API 订单解析，可作补充；DEMO 汇总仅可标注「主单 N 笔（未核对老虎，非实盘收益率）」且不得以 % 形式冒充收益率。

用法: 在 tigertrade 根目录运行 python scripts/update_today_yield_for_status.py
"""
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
REPORT_JSON = ROOT / "docs" / "reports" / "algorithm_optimization_report.json"
TODAY_YIELD_JSON = DOCS / "today_yield.json"


def from_tiger_backend():
    """
    从老虎后台（DEMO 账户）拉取订单，仅基于成交单得到收益率/成交笔数。
    成功返回 dict(yield_pct, yield_note, source="tiger_backend")；失败返回 None。
    """
    script = ROOT / "scripts" / "fetch_tiger_yield_for_demo.py"
    if not script.exists():
        return None
    try:
        r = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if r.returncode != 0:
            return None
        data = json.loads(r.stdout.strip())
        if data.get("source") != "tiger_backend":
            return None
        return {
            "yield_pct": data.get("yield_pct") or "—",
            "yield_note": data.get("yield_note") or "老虎后台",
            "source": "tiger_backend",
        }
    except Exception:
        return None


def from_report():
    """
    从 algorithm_optimization_report.json 读取收益率。
    仅当 report 中 profitability 来自 API 订单解析（total_trades>0）时才视为可用的实盘相关数据。
    """
    if not REPORT_JSON.exists():
        return None
    try:
        with open(REPORT_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    prof = data.get("profitability")
    if not prof:
        return None
    total = prof.get("total_trades", 0)
    win_rate = prof.get("win_rate", 0)
    avg_profit = prof.get("average_profit", 0)
    total_profit = prof.get("total_profit", 0)
    if total == 0:
        return {"yield_pct": "—", "yield_note": "暂无交易", "source": "report"}
    note = f"胜率 {win_rate:.1f}%"
    if avg_profit is not None:
        note += f" · 均盈 {avg_profit:.2f} USD"
    if total_profit is not None:
        note += f" · 总盈 {total_profit:.2f} USD"
    return {"yield_pct": f"{win_rate:.1f}%", "yield_note": note, "source": "report"}


def fallback_demo_aggregate_only():
    """
    仅作兜底：DEMO 日志汇总笔数。不得以百分比冒充实盘收益率。
    返回 yield_pct="—" 或 "主单 N 笔"，yield_note 必须含「未核对老虎，非实盘收益率」。
    """
    try:
        from scripts.analyze_demo_log import aggregate_demo_logs
        demo = aggregate_demo_logs()
    except Exception:
        return None
    if not demo:
        return None
    n = demo.get("order_success", 0)
    L = demo.get("logs_scanned", 0)
    if L == 0 and n == 0:
        return {"yield_pct": "—", "yield_note": "DEMO 汇总: 未扫描到日志（未核对老虎，非实盘收益率）", "source": "demo_aggregate"}
    # 只展示笔数，不展示 %；注明未核对老虎
    return {
        "yield_pct": f"主单 {n} 笔",
        "yield_note": f"DEMO 汇总: 主单成功 {n} 笔，扫描 {L} 个日志（未核对老虎，非实盘收益率）",
        "source": "demo_aggregate",
    }


def main():
    today = datetime.now().strftime("%Y-%m-%d")
    out = {
        "date": today,
        "yield_pct": "—",
        "yield_note": "需老虎后台数据核对",
        "source": "none",
    }
    try:
        # 1) 优先：老虎后台
        data = from_tiger_backend()
        if data:
            out["yield_pct"] = (data.get("yield_pct") or "—").strip() or "—"
            out["yield_note"] = (data.get("yield_note") or "").strip() or "老虎后台"
            out["source"] = "tiger_backend"
        else:
            # 2) 报告（API 订单解析出的 profitability）
            data = from_report()
            if data:
                out["yield_pct"] = (data.get("yield_pct") or "—").strip() or "—"
                out["yield_note"] = (data.get("yield_note") or "待统计").strip() or "待统计"
                out["source"] = "report"
            else:
                # 3) 兜底：仅 DEMO 汇总笔数，明确标注非实盘收益率
                data = fallback_demo_aggregate_only()
                if data:
                    out["yield_pct"] = (data.get("yield_pct") or "—").strip() or "—"
                    out["yield_note"] = (data.get("yield_note") or "未核对老虎，非实盘收益率").strip()
                    out["source"] = data.get("source", "demo_aggregate")
    except Exception as e:
        out["yield_pct"] = "错误"
        out["yield_note"] = f"更新失败: {str(e)[:120]}"
        out["source"] = "error"

    if not out.get("yield_pct") or not str(out["yield_pct"]).strip():
        out["yield_pct"] = "—"
    if not out.get("yield_note") or not str(out["yield_note"]).strip():
        out["yield_note"] = "需老虎后台数据核对"

    DOCS.mkdir(parents=True, exist_ok=True)
    with open(TODAY_YIELD_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"✅ 已写入 {TODAY_YIELD_JSON}: 今日收益率 -> {out['yield_pct']} (来源: {out['source']})")


if __name__ == "__main__":
    main()
