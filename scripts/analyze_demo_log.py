#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEMO 日志分析：看 LOG 就能发现「无止损止盈、超仓、风控报错」等问题。
运行中或跑完后执行：python scripts/analyze_demo_log.py [日志路径]
不传路径则用当前目录下最新的 demo_*.log。
退出码：0=无问题，1=发现问题（便于 cron/CI 用）。
"""
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def find_latest_demo_log(log_path=None):
    if log_path and os.path.isfile(log_path):
        return Path(log_path)
    for d in [ROOT, ROOT / "logs"]:
        logs = sorted(d.glob("demo_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        if logs:
            return logs[0]
    return None


def find_all_demo_logs():
    """返回所有 DEMO 相关日志（demo_*.log、demo_run_20h_*.log），按修改时间倒序，用于多日汇总。"""
    seen = set()
    out = []
    for d in [ROOT, ROOT / "logs"]:
        if not d.exists():
            continue
        for p in d.glob("demo_*.log"):
            if p.resolve() not in seen:
                seen.add(p.resolve())
                out.append(p)
        for p in d.glob("demo_run_20h_*.log"):
            if p.resolve() not in seen:
                seen.add(p.resolve())
                out.append(p)
    out.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return out


def aggregate_demo_logs(log_paths=None):
    """对多份 DEMO 日志做汇总统计，供策略报告与优化报告使用。"""
    paths = log_paths if log_paths is not None else find_all_demo_logs()
    if not paths:
        return None
    agg = {
        "order_success": 0,
        "success_orders_sum": 0,
        "fail_orders_sum": 0,
        "sl_tp_log": 0,
        "execute_buy_calls": 0,
        "lines": 0,
        "max_positions": [],
        "attr_error_risk": False,
        "no_sl_tp_risk": False,
        "max_pos_over_3": False,
        "errors": 0,
        "logs_scanned": len(paths),
        "log_paths_sample": [str(p) for p in paths[:5]],
    }
    for p in paths:
        try:
            r = analyze(p)
            agg["order_success"] += r.get("order_success", 0)
            agg["success_orders_sum"] += r.get("success_orders_sum", 0)
            agg["fail_orders_sum"] += r.get("fail_orders_sum", 0)
            agg["sl_tp_log"] += r.get("sl_tp_log", 0)
            agg["execute_buy_calls"] += r.get("execute_buy_calls", 0)
            agg["lines"] += r.get("lines", 0)
            agg["errors"] += r.get("errors", 0)
            agg["max_positions"].extend(r.get("max_positions") or [])
            if r.get("attr_error_risk"):
                agg["attr_error_risk"] = True
            if r.get("no_sl_tp_risk"):
                agg["no_sl_tp_risk"] = True
            if r.get("max_pos_over_3"):
                agg["max_pos_over_3"] = True
        except Exception:
            continue
    if agg["max_positions"]:
        agg["max_position"] = max(agg["max_positions"])
    else:
        agg["max_position"] = None
    return agg


def analyze(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    # 统计
    buy_success = len(re.findall(r"订单提交成功|Order创建成功|执行买入.*成功", text))
    # 主单成功相关（不同日志措辞）
    order_success = len(re.findall(r"订单提交成功|\[执行买入\].*✅|Order创建成功", text))
    # 日志中打印的「成功订单: N」「失败订单: N」汇总（每轮可能打印一次，取总和）
    success_orders_matches = re.findall(r"成功订单[:\s]*(\d+)", text)
    fail_orders_matches = re.findall(r"失败订单[:\s]*(\d+)", text)
    success_orders_sum = sum(int(x) for x in success_orders_matches if x.isdigit())
    fail_orders_sum = sum(int(x) for x in fail_orders_matches if x.isdigit())
    sl_tp_log = len(re.findall(r"止损|止盈|STP|止盈单|止损单|已提交止损|已提交止盈", text))
    attr_error_risk = "check_risk_control" in text and "AttributeError" in text
    max_pos_matches = re.findall(r"最大仓位[:\s]*(\d+)手?", text)
    max_positions = [int(x) for x in max_pos_matches if x.isdigit()]
    max_pos_over_3 = any(p > 3 for p in max_positions) if max_positions else False
    execute_buy_calls = len(re.findall(r"execute_buy|动作:\s*买入", text))
    errors = len(re.findall(r"❌|ERROR|Error:|Exception|Traceback", text))

    # 判断：有主单成功但几乎没有止损/止盈日志 → 可能无止损止盈
    no_sl_tp_risk = order_success > 2 and sl_tp_log < order_success

    return {
        "path": str(path),
        "order_success": order_success,
        "success_orders_sum": success_orders_sum,
        "fail_orders_sum": fail_orders_sum,
        "sl_tp_log": sl_tp_log,
        "attr_error_risk": attr_error_risk,
        "max_positions": max_positions,
        "max_pos_over_3": max_pos_over_3,
        "execute_buy_calls": execute_buy_calls,
        "errors": errors,
        "no_sl_tp_risk": no_sl_tp_risk,
        "lines": len(lines),
    }


def main():
    log_path = sys.argv[1] if len(sys.argv) > 1 else None
    path = find_latest_demo_log(log_path)
    if not path:
        print("❌ 未找到 DEMO 日志（可传路径: python scripts/analyze_demo_log.py <path>）")
        return 1

    r = analyze(path)
    print("=" * 60)
    print("DEMO 日志分析（看 LOG 发现问题）")
    print("=" * 60)
    print(f"日志: {r['path']} ({r['lines']} 行)")
    print(f"  主单成功/相关: {r['order_success']} 处")
    print(f"  止损/止盈相关: {r['sl_tp_log']} 处")
    print(f"  买入动作/execute_buy: {r['execute_buy_calls']} 处")
    print(f"  错误/异常 出现: {r['errors']} 处")
    if r["max_positions"]:
        print(f"  最大仓位 出现: {max(r['max_positions'])} 手（样本: {r['max_positions'][:5]}）")

    issues = []
    if r["no_sl_tp_risk"]:
        issues.append("存在多笔主单成功但止损/止盈日志很少 → 可能「有买无止损止盈」")
    if r["attr_error_risk"]:
        issues.append("存在 AttributeError check_risk_control → 风控未生效、下单可能异常")
    if r["max_pos_over_3"]:
        issues.append("日志中出现最大仓位 > 3 手 → DEMO 可能超仓")

    if issues:
        print()
        print("⚠️ 发现问题（看 LOG 即可发现）:")
        for i, msg in enumerate(issues, 1):
            print(f"  {i}. {msg}")
        print()
        print("建议: 修复后重新跑 DEMO，并定期执行本脚本或接入 cron。")
        return 1

    print()
    print("✅ 未发现上述可靠性问题（仅做简单扫描，仍建议人工抽查）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
