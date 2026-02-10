#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导出 run/order_log.jsonl 为 CSV 并生成分析报告。
说明：mode=mock 表示模拟单（未提交至老虎后台）；mode=real 且 status=success 表示已提交至老虎（DEMO/实盘账户）；mode=real 且 status=fail 表示 API 拒绝，不会出现在后台。

用法:
  python scripts/export_order_log_and_analyze.py
  python scripts/export_order_log_and_analyze.py --out-dir docs/reports
"""
import json
import csv
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
ORDER_LOG = ROOT / "run" / "order_log.jsonl"
DEFAULT_CSV = ROOT / "run" / "order_log_export.csv"
DEFAULT_REPORT = ROOT / "docs" / "reports"


def load_records(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def export_csv(records, out_path):
    if not records:
        return
    fieldnames = [
        "ts", "side", "symbol", "source", "order_type", "qty", "price",
        "order_id", "status", "mode", "stop_loss", "take_profit", "reason", "error"
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in records:
            row = {k: r.get(k, "") for k in fieldnames}
            w.writerow(row)


def _date_from_ts(ts):
    """从时间戳取日期 YYYY-MM-DD，便于按日统计。"""
    if not ts:
        return ""
    s = str(ts).strip()
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10]
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return ""


def analyze(records):
    by_mode_status = defaultdict(int)
    by_source = defaultdict(int)
    real_errors = defaultdict(int)
    real_success_ts = []
    # 按日期：每日的 (real成功, real失败, mock成功, 总条数)
    by_date = defaultdict(lambda: {"real_success": 0, "real_fail": 0, "mock_success": 0, "total": 0})
    for r in records:
        m = r.get("mode", "")
        s = r.get("status", "")
        by_mode_status[(m, s)] += 1
        by_source[r.get("source", "auto")] += 1
        day = _date_from_ts(r.get("ts", ""))
        if day:
            by_date[day]["total"] += 1
            if m == "real":
                if s == "success":
                    by_date[day]["real_success"] += 1
                else:
                    by_date[day]["real_fail"] += 1
            elif m == "mock" and s == "success":
                by_date[day]["mock_success"] += 1
        if m == "real":
            if s == "fail":
                err = (r.get("error") or "").strip() or "unknown"
                if len(err) > 80:
                    err = err[:77] + "..."
                real_errors[err] += 1
            else:
                real_success_ts.append(r.get("ts", ""))
    # 按日期排序，最近在前
    by_date_sorted = dict(sorted(by_date.items(), reverse=True))
    return {
        "total": len(records),
        "by_mode_status": dict(by_mode_status),
        "by_source": dict(by_source),
        "by_date": by_date_sorted,
        "real_errors": dict(sorted(real_errors.items(), key=lambda x: -x[1])),
        "real_success_count": len(real_success_ts),
        "real_success_ts_sample": sorted(real_success_ts)[-10:] if real_success_ts else [],
    }


def write_report(records, stats, out_dir, report_name="order_log_analysis.md"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / report_name

    lines = [
        "# 订单日志导出与分析",
        "",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**数据源**: `run/order_log.jsonl`",
        f"**总条数**: {stats['total']}",
        "",
        "## 一、结论（是否「真的」在老虎后台）",
        "",
        "| 类型 | 含义 | 是否会在老虎后台出现 |",
        "|------|------|----------------------|",
        "| **mode=mock** | 模拟单，未调用老虎 API | **不会** |",
        "| **mode=real, status=success** | 已成功提交至老虎 API | **会**（请在老虎 DEMO/实盘账户对应时间、合约下查询） |",
        "| **mode=real, status=fail** | 调用老虎 API 但被拒绝（如非交易时段、账户限制） | **不会** |",
        "",
        "因此：若后台查不到订单，请先看下方统计中 **mode=real 且 status=success** 的数量与时间；若多为 mock 或 real 多为 fail，则后台无对应记录是预期行为。",
        "",
        "## 二、统计汇总",
        "",
    ]

    # 按 mode/status 汇总
    lines.append("### 按 mode 与 status")
    lines.append("")
    for (m, s), cnt in sorted(stats["by_mode_status"].items(), key=lambda x: -x[1]):
        lines.append(f"- mode=**{m}**, status=**{s}**: {cnt} 条")
    lines.append("")

    lines.append("### 按 source")
    lines.append("")
    for src, cnt in sorted(stats["by_source"].items(), key=lambda x: -x[1]):
        lines.append(f"- source=**{src}**: {cnt} 条")
    lines.append("")

    # 按日期统计：看清「60 多单」等是哪天的
    by_date = stats.get("by_date") or {}
    if by_date:
        lines.append("### 按日期（订单数）")
        lines.append("")
        lines.append("| 日期 | 总条数 | real成功（会出现在老虎） | real失败 | mock成功 |")
        lines.append("| --- | --- | --- | --- | --- |")
        for day, v in by_date.items():
            lines.append(f"| {day} | {v['total']} | {v['real_success']} | {v['real_fail']} | {v['mock_success']} |")
        lines.append("")
        lines.append("*收益率按日统计需老虎后台成交明细或 API 拉取；本表仅订单条数按日。*")
        lines.append("")

    if stats["real_errors"]:
        lines.append("### mode=real 且 status=fail 的典型错误（前 15 条）")
        lines.append("")
        for err, cnt in list(stats["real_errors"].items())[:15]:
            lines.append(f"- `{err}`: {cnt} 次")
        lines.append("")

    if stats["real_success_ts_sample"]:
        lines.append("### mode=real 且 status=success 最近 10 条时间戳（供与老虎后台核对）")
        lines.append("")
        for ts in stats["real_success_ts_sample"]:
            lines.append(f"- {ts}")
        lines.append("")

    lines.append("## 三、说明")
    lines.append("")
    lines.append("- 完整明细已导出为 CSV：`run/order_log_export.csv`（或通过 `--csv` 指定路径）。")
    lines.append("- DEMO 运行（`tiger1 d moe`）时：若 SDK 初始化成功则使用真实 API（openapicfg_dem），订单为 mode=real；若初始化失败则走模拟，订单为 mode=mock。")
    lines.append("- 老虎后台请使用 **DEMO 账户** 对应账户与时间范围查询；实盘账户与 DEMO 账户订单分离。")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_order_execution_status(stats, docs_dir):
    """写入 docs/order_execution_status.json，供状态页展示「订单执行：成功/失败（含API被拒）」；失败多则无实盘收益率。"""
    docs_dir = Path(docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    real_ok = stats.get("real_success_count", 0)
    real_fail = stats.get("by_mode_status", {}).get(("real", "fail"), 0)
    obj = {
        "real_success_count": real_ok,
        "real_fail_count": real_fail,
        "note": "收益率仅来自老虎后台成交；执行失败（含API被拒）时无实盘收益率。",
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    path = docs_dir / "order_execution_status.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path


def main():
    ap = argparse.ArgumentParser(description="导出 order_log.jsonl 为 CSV 并生成分析报告")
    ap.add_argument("--file", type=Path, default=ORDER_LOG, help="order_log.jsonl 路径")
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="导出 CSV 路径")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_REPORT, help="报告输出目录")
    ap.add_argument("--report-name", type=str, default="order_log_analysis.md", help="报告文件名")
    ap.add_argument("--no-status-json", action="store_true", help="不写入 docs/order_execution_status.json")
    args = ap.parse_args()

    if not args.file.exists():
        print(f"📭 文件不存在: {args.file}")
        return 1

    records = load_records(args.file)
    if not records:
        print("📭 无有效记录")
        return 0

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    export_csv(records, args.csv)
    print(f"📄 CSV 已导出: {args.csv} ({len(records)} 条)")

    stats = analyze(records)
    report_path = write_report(records, stats, args.out_dir, args.report_name)
    print(f"📄 分析报告: {report_path}")
    if not args.no_status_json:
        status_path = write_order_execution_status(stats, ROOT / "docs")
        print(f"📄 订单执行状态: {status_path}")

    print("\n--- 统计摘要 ---")
    print(f"  total: {stats['total']}")
    for (m, s), cnt in sorted(stats["by_mode_status"].items(), key=lambda x: -x[1]):
        print(f"  mode={m}, status={s}: {cnt}")
    print(f"  real success 条数（应出现在老虎后台）: {stats['real_success_count']}")
    return 0


if __name__ == "__main__":
    exit(main())
