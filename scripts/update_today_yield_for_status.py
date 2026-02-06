#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 收益与算法优化 报告或 DEMO 日志中提取今日收益率，写入 docs/today_yield.json，
供状态页 status.html 通过 fetch 显示「今日收益率」。
用法: 在 tigertrade 根目录运行 python scripts/update_today_yield_for_status.py
"""
import json
import os
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
REPORT_JSON = ROOT / "docs" / "reports" / "algorithm_optimization_report.json"
TODAY_YIELD_JSON = DOCS / "today_yield.json"
# 与 analyze_demo_log 一致：项目根 + logs 目录都扫
LOGS_DIRS = [ROOT, ROOT / "logs"]


def from_report():
    """从 algorithm_optimization_report.json 读取收益率"""
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
        return {"yield_pct": "—", "yield_note": "暂无交易"}
    # 显示：胜率 + 均盈/总盈
    note = f"胜率 {win_rate:.1f}%"
    if avg_profit is not None:
        note += f" · 均盈 {avg_profit:.2f} USD"
    if total_profit is not None:
        note += f" · 总盈 {total_profit:.2f} USD"
    return {"yield_pct": f"{win_rate:.1f}%", "yield_note": note}


def _collect_demo_log_paths():
    """与 analyze_demo_log.find_all_demo_logs 一致：根目录 + logs 下的 demo_*.log / demo_run_20h_*.log"""
    out = []
    seen = set()
    for d in LOGS_DIRS:
        if not d.exists():
            continue
        for p in list(d.glob("demo_*.log")) + list(d.glob("demo_run_20h_*.log")):
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                out.append(p)
    out.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return out


def from_demo_logs():
    """从最新 DEMO 日志中尝试解析盈亏/收益率行（多目录、多关键字）"""
    import re
    paths = _collect_demo_log_paths()
    # 关键字：持仓盈亏、盈亏、总盈、收益率、今日.*% 等
    patterns = [
        (r"持仓盈亏[=:\s]*([-+]?\d+\.?\d*)", "持仓盈亏"),
        (r"盈亏[:\s]*([-+]?\d+\.?\d*)", "盈亏"),
        (r"总盈[利]?[=:\s]*([-+]?\d+\.?\d*)", "总盈"),
        (r"收益率[=:\s]*([-+]?\d+\.?\d*)%?", "收益率"),
        (r"今日.*?([-+]?\d+\.?\d*)%", "今日%"),
        (r"yield[=:\s]*([-+]?\d+\.?\d*)", "yield"),
    ]
    for log_path in paths[:5]:
        try:
            text = log_path.read_text(encoding="utf-8", errors="ignore")
            for regex, name in patterns:
                m = re.search(regex, text)
                if m:
                    num = m.group(1).strip()
                    if name == "收益率" or "%" in (m.group(0) or ""):
                        return {"yield_pct": num + "%", "yield_note": "DEMO 日志"}
                    return {"yield_pct": num + " USD", "yield_note": "DEMO 日志"}
            # 兼容旧逻辑：整行含关键字再抽数字
            for line in text.splitlines():
                if "持仓盈亏" in line or "盈亏:" in line or "总盈" in line:
                    m = re.search(r"[-+]?\d+\.?\d*", line.replace(",", ""))
                    if m:
                        return {"yield_pct": m.group(0) + " USD", "yield_note": "DEMO 日志"}
        except Exception:
            continue
    return None


def main():
    today = datetime.now().strftime("%Y-%m-%d")
    out = {"date": today, "yield_pct": "—", "yield_note": "待统计（运行收益与算法优化或从 DEMO 日志解析后更新）", "source": "report"}
    try:
        _main_impl(out)
    except Exception as e:
        out["yield_pct"] = "错误"
        out["yield_note"] = f"更新失败: {str(e)[:120]}"
        out["source"] = "error"
    # 禁止写入空字符串
    if not out.get("yield_pct") or not str(out["yield_pct"]).strip():
        out["yield_pct"] = "—"
    if not out.get("yield_note") or not str(out["yield_note"]).strip():
        out["yield_note"] = "待统计"
    DOCS.mkdir(parents=True, exist_ok=True)
    with open(TODAY_YIELD_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"✅ 已写入 {TODAY_YIELD_JSON}: 今日收益率 -> {out['yield_pct']}")


def _main_impl(out):

    data = from_report()
    if data:
        out["source"] = "report"
    else:
        data = from_demo_logs()
        out["source"] = "demo_log" if data else "report"
    if data:
        out["yield_pct"] = (data.get("yield_pct") or "—").strip() or "—"
        out["yield_note"] = (data.get("yield_note") or "待统计").strip() or "待统计"
    else:
        # 无报告且日志未解析出数字时，用 DEMO 汇总兜底，避免「今日收益率」一直为空；出错也写入（错误即数据）
        err_msg = None
        try:
            from scripts.analyze_demo_log import aggregate_demo_logs
            demo = aggregate_demo_logs()
            if demo and (demo.get("logs_scanned", 0) > 0 or demo.get("order_success", 0) > 0):
                n = demo.get("order_success", 0)
                L = demo.get("logs_scanned", 0)
                out["yield_pct"] = f"主单 {n} 笔"
                out["yield_note"] = f"DEMO 汇总: 主单成功 {n} 笔，扫描 {L} 个日志"
                out["source"] = "demo_aggregate"
            elif demo and demo.get("logs_scanned", 0) == 0:
                out["yield_note"] = "DEMO 汇总: 未扫描到日志"
                out["source"] = "demo_aggregate"
        except Exception as e:
            err_msg = str(e)[:120]
            out["yield_pct"] = "解析失败"
            out["yield_note"] = f"错误: {err_msg}"
            out["source"] = "error"
    # 禁止写入空字符串；出错也是数据，便于改进算法
    if not out.get("yield_pct") or not str(out["yield_pct"]).strip():
        out["yield_pct"] = "—"
    if not out.get("yield_note") or not str(out["yield_note"]).strip():
        out["yield_note"] = "待统计"


if __name__ == "__main__":
    main()
