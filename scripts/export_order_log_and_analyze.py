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
CONFIG_PATH = ROOT / "openapicfg_dem"


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
    local_reject_count = 0
    api_reject_count = 0
    local_reject_keywords = ("ALLOW_REAL_TRADING", "持仓硬顶", "pos=", "order_id无效", "来自mock", "来自测试")
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
                # 区分：本地风控拒绝 vs 真实 API 被拒
                err_raw = r.get("error") or ""
                if any(kw in err_raw for kw in local_reject_keywords):
                    local_reject_count += 1
                else:
                    api_reject_count += 1
            else:
                real_success_ts.append(r.get("ts", ""))
    # real+success 中区分：order_id 疑似测试/mock（与老虎对不上）vs 可能为真实单（纯数字）
    real_success_likely_mock = 0
    real_success_likely_real = 0
    for r in records:
        if r.get("mode") != "real" or r.get("status") != "success":
            continue
        oid = str(r.get("order_id") or "").strip()
        if not oid:
            real_success_likely_mock += 1
            continue
        if "Mock" in oid or oid.startswith("TEST_") or oid == "TEST123" or oid.startswith("ORDER_"):
            real_success_likely_mock += 1
        elif oid.isdigit() and len(oid) >= 10:
            real_success_likely_real += 1
        else:
            real_success_likely_mock += 1
    by_date_sorted = dict(sorted(by_date.items(), reverse=True))
    return {
        "total": len(records),
        "by_mode_status": dict(by_mode_status),
        "by_source": dict(by_source),
        "by_date": by_date_sorted,
        "real_errors": dict(sorted(real_errors.items(), key=lambda x: -x[1])),
        "real_success_count": len(real_success_ts),
        "local_reject_count": local_reject_count,
        "api_reject_count": api_reject_count,
        "real_success_likely_mock": real_success_likely_mock,
        "real_success_likely_real": real_success_likely_real,
        "real_success_ts_sample": sorted(real_success_ts)[-10:] if real_success_ts else [],
    }


def write_report(records, stats, out_dir, report_name="order_log_analysis.md", with_backend=False):
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
        "| **mode=real, status=success** | order_log 记录（本系统认为已提交） | **未必会**：含测试/mock 污染，与老虎后台对不上是常态，**成功数仅以老虎后台为准** |",
        "| **mode=real, status=fail** | 调用老虎 API 但被拒绝（如非交易时段、账户限制） | **不会** |",
        "",
        "因此：若后台查不到订单，请先看下方统计中 **mode=real 且 status=success** 的数量与时间；若多为 mock 或 real 多为 fail，则后台无对应记录是预期行为。",
        "",
    ]
    # 若 real+success 里多数为 Mock/TEST 等，说明与老虎对不上的原因
    likely_mock = stats.get("real_success_likely_mock", 0)
    likely_real = stats.get("real_success_likely_real", 0)
    total_rs = stats.get("real_success_count", 0)
    if total_rs > 0 and likely_mock > 0:
        lines.append("**⚠️ 关于「与老虎后台对不上」**：当前 mode=real 且 status=success 的条数中，**多数 order_id 为 Mock、TEST_、ORDER_ 等**，说明来自**测试或 mock 环境**（当时未真正走老虎 API，或 API 被 mock 后把 mock 返回值写进了 order_log）。这类记录**不是老虎真实成交**，与老虎后台对不上是预期。只有 order_id 为**纯数字且较长**（老虎返回格式）的，才可能是真实单。详见下方「real success 中：疑似测试/mock 与可能真实单」。")
        lines.append("")
    lines.append("## 二、统计汇总")
    lines.append("")

    # 按 mode/status 汇总
    lines.append("### 按 mode 与 status")
    lines.append("")
    for (m, s), cnt in sorted(stats["by_mode_status"].items(), key=lambda x: -x[1]):
        lines.append(f"- mode=**{m}**, status=**{s}**: {cnt} 条")
    lines.append("")
    if total_rs > 0:
        lines.append("### mode=real 且 status=success 中：疑似测试/mock 与可能真实单")
        lines.append("")
        lines.append("| 类型 | 条数 | 说明 |")
        lines.append("| --- | --- | --- |")
        lines.append(f"| 疑似测试/mock（order_id 含 Mock、TEST_、ORDER_ 等） | {likely_mock} | 来自测试或 mock 环境，**非老虎真实成交**，与老虎后台对不上是预期。 |")
        lines.append(f"| 可能为老虎真实单（order_id 为纯数字） | {likely_real} | **未与老虎核对**，可能含测试污染；真实成功数仅以老虎后台为准。 |")
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
        lines.append("| 日期 | 总条数 | order_log 记成功（未核对） | real失败 | mock成功 |")
        lines.append("| --- | --- | --- | --- | --- |")
        for day, v in by_date.items():
            lines.append(f"| {day} | {v['total']} | {v['real_success']} | {v['real_fail']} | {v['mock_success']} |")
        lines.append("")
        lines.append("*收益率按日统计需老虎后台成交明细或 API 拉取；本表仅订单条数按日。*")
        lines.append("")

    # 失败分类：本地风控 vs API 被拒
    local_reject = stats.get("local_reject_count", 0)
    api_reject = stats.get("api_reject_count", 0)
    if local_reject or api_reject:
        lines.append("### mode=real 且 status=fail 分类")
        lines.append("")
        lines.append(f"- **本地风控拒绝**（未发 API）：{local_reject} 条（如 ALLOW_REAL_TRADING、持仓硬顶）")
        lines.append(f"- **真实 API 被拒**（老虎拒绝）：{api_reject} 条（如 1010/1200/account 为空）")
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

    if with_backend:
        backend_orders, match_map = _fetch_backend_orders_and_match(records)
        lines.append("## 四、后台订单与 order_log（DFX）对照")
        lines.append("")
        lines.append("（由本脚本 `--with-backend` 拉取老虎订单，与 order_log 按 order_id 对；可判断每笔后台单是否来自本机 order_log。）")
        lines.append("")
        if not backend_orders:
            lines.append("未拉取到后台订单或 openapicfg_dem 不可用。")
        else:
            lines.append("| 后台 order_id | symbol | side | status | limit_price | order_log 条数 | order_log ts/source/error |")
            lines.append("|--------------|--------|------|--------|-------------|----------------|---------------------------|")
            for o in backend_orders[:50]:
                oid = getattr(o, "order_id", None) or getattr(o, "id", None)
                sym = getattr(o, "symbol", None) or getattr(getattr(o, "contract", None), "symbol", None)
                side = getattr(o, "side", None)
                st = getattr(o, "status", None)
                lp = getattr(o, "limit_price", None) or getattr(o, "price", None)
                log_rows = match_map.get(str(oid), [])
                log_info = ""
                if log_rows:
                    r0 = log_rows[0]
                    log_info = "ts=%s source=%s" % (r0.get("ts", "")[:19], r0.get("source", ""))
                    if r0.get("error"):
                        log_info += " error=%s" % (str(r0.get("error"))[:40])
                else:
                    log_info = "无（可能手工单或别进程）"
                lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (oid, sym, side, st, lp, len(log_rows), log_info[:50]))
            if len(backend_orders) > 50:
                lines.append("| ... | 共 %s 条，仅列前 50 |" % len(backend_orders))
        lines.append("")
        reason = _today_demo_zero_fill_reason(records, stats)
        lines.append("## 五、DEMO 今日 0 成交原因（定位）")
        lines.append("")
        lines.append(reason)
        lines.append("")

    lines.append("## %s、说明" % ("六" if with_backend else "三"))
    lines.append("")
    lines.append("- 完整明细已导出为 CSV：`run/order_log_export.csv`（或通过 `--csv` 指定路径）。")
    lines.append("- DEMO 运行（`tiger1 d moe`）时：若 SDK 初始化成功则使用真实 API（openapicfg_dem），订单为 mode=real；若初始化失败则走模拟，订单为 mode=mock。")
    lines.append("- 老虎后台请使用 **DEMO 账户** 对应账户与时间范围查询；实盘账户与 DEMO 账户订单分离。")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _fetch_backend_orders_and_match(records):
    """拉取老虎后台订单，与 order_log 按 order_id 对照。返回 (backend_list, match_map)。"""
    try:
        from scripts.fetch_tiger_yield_for_demo import fetch_orders_from_tiger, _today_range_local
    except Exception:
        return [], {}
    if not CONFIG_PATH.exists():
        return [], {}
    account = None
    try:
        from tigeropen.tiger_open_config import TigerOpenClientConfig
        cfg = TigerOpenClientConfig(props_path=str(CONFIG_PATH))
        account = getattr(cfg, "account", None)
    except Exception:
        pass
    if not account:
        return [], {}
    start_time, end_time = _today_range_local()
    orders = []
    for sym in ("SIL2603", "SIL2605"):
        orders.extend(fetch_orders_from_tiger(account, sym, limit=300, start_time=start_time, end_time=end_time) or [])
    if not orders:
        for sym in ("SIL2603", "SIL2605"):
            orders.extend(fetch_orders_from_tiger(account, sym, limit=200) or [])
    seen = set()
    unique = []
    for o in orders:
        oid = getattr(o, "order_id", None) or getattr(o, "id", None)
        if oid is not None and oid not in seen:
            seen.add(oid)
            unique.append(o)
    log_by_oid = defaultdict(list)
    for r in records:
        oid = str(r.get("order_id") or "").strip()
        if oid:
            log_by_oid[oid].append(r)
    match_map = {}
    for o in unique:
        oid = str(getattr(o, "order_id", None) or getattr(o, "id", None) or "")
        match_map[oid] = log_by_oid.get(oid, [])
    return unique, match_map


def _today_demo_zero_fill_reason(records, stats):
    """根据 order_log 今日统计，给出 DEMO 今日 0 成交原因（可定位到 DFX/风控）。"""
    today = datetime.now().strftime("%Y-%m-%d")
    by_date = stats.get("by_date") or {}
    day = by_date.get(today, {})
    real_ok = day.get("real_success", 0)
    real_fail = day.get("real_fail", 0)
    total_today = day.get("total", 0)
    if total_today == 0:
        return "今日 order_log 无任何记录 → DEMO 未运行或未触发下单（无 DFX 日志可对）。"
    if real_ok == 0 and real_fail == 0:
        return "今日仅有 mock 成功记录，无 real 单 → DEMO 未用真实 API 下单（或 SDK 初始化失败走 mock）。"
    if real_ok == 0 and real_fail > 0:
        today_errors = []
        for r in records:
            if _date_from_ts(r.get("ts")) != today or r.get("mode") != "real" or r.get("status") != "fail":
                continue
            today_errors.append((r.get("error") or "").strip() or "unknown")
        from collections import Counter
        top = Counter(today_errors).most_common(5)
        err_text = "; ".join("%s(%s次)" % (e[:60], c) for e, c in top)
        return "今日有 real 下单尝试但全部失败 → 原因见 DFX/order_log 的 error：%s。可据此定位风控或 API 拒单。" % (err_text or "—")
    if real_ok > 0:
        return "今日 order_log 有 %s 笔 real success；若老虎后台仍显示 0 成交，可能为时间/合约过滤或 order_id 非老虎返回（见「后台订单与 order_log 对照」）。" % real_ok
    return "今日统计无法推断；见上方按日期与失败原因。"


def _fetch_tiger_verified_count():
    """从老虎后台拉取今日成交数，作为唯一可信的「成功」数。order_log 含测试污染，不可用。"""
    try:
        r = __import__("subprocess").run(
            [__import__("sys").executable, str(ROOT / "scripts" / "fetch_tiger_yield_for_demo.py")],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=15,
        )
        if r.returncode != 0:
            return None
        data = __import__("json").loads(r.stdout.strip())
        return data.get("filled_count", 0) if data.get("is_today_only") else None
    except Exception:
        return None


def write_order_execution_status(stats, docs_dir):
    """写入 docs/order_execution_status.json。成功数仅以老虎后台为准，order_log 含测试污染不展示。"""
    docs_dir = Path(docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    # 成功数：只信老虎后台，order_log 的 638 含测试/mock 污染，不展示
    tiger_ok = _fetch_tiger_verified_count()
    real_ok = tiger_ok if tiger_ok is not None else None  # None 时前端显示 —
    real_fail = stats.get("by_mode_status", {}).get(("real", "fail"), 0)
    local_reject = stats.get("local_reject_count", 0)
    api_reject = stats.get("api_reject_count", 0)
    obj = {
        "real_success_count": real_ok,  # 老虎后台今日成交数；None 时前端显示 —
        "real_fail_count": real_fail,
        "local_reject_count": local_reject,
        "api_reject_count": api_reject,
        "note": "成功数=老虎后台今日成交；order_log含测试污染不展示；失败=本地风控或API被拒。",
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
    ap.add_argument("--with-backend", action="store_true", help="拉取老虎后台订单并与 order_log 对照，并输出 DEMO 今日 0 成交原因")
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
    report_path = write_report(records, stats, args.out_dir, args.report_name, with_backend=args.with_backend)
    print(f"📄 分析报告: {report_path}")
    if not args.no_status_json:
        status_path = write_order_execution_status(stats, ROOT / "docs")
        print(f"📄 订单执行状态: {status_path}")

    print("\n--- 统计摘要 ---")
    print(f"  total: {stats['total']}")
    for (m, s), cnt in sorted(stats["by_mode_status"].items(), key=lambda x: -x[1]):
        print(f"  mode={m}, status={s}: {cnt}")
    print(f"  老虎后台今日成交（order_execution_status）: 见 docs/order_execution_status.json")
    return 0


if __name__ == "__main__":
    exit(main())
