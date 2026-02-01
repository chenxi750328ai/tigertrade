#!/usr/bin/env python3
"""
订单 LOG 查询工具：多行清晰显示 run/order_log.jsonl

用法:
  python scripts/query_order_log.py           # 最近 20 条
  python scripts/query_order_log.py -n 50     # 最近 50 条
  python scripts/query_order_log.py -n 0      # 全部
  python scripts/query_order_log.py --success # 只看成功
  python scripts/query_order_log.py --fail    # 只看失败
"""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.order_log import ORDER_LOG_FILE


def _ts_fmt(ts_str):
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ts_str


def _order_type_label(ot):
    # market=市价单, limit=限价单, stop_loss=止损单, take_profit=止盈单（不写“现价单”避免和市价混淆）
    labels = {"market": "市价单", "limit": "限价单", "stop_loss": "止损单", "take_profit": "止盈单"}
    return labels.get(ot, ot or "限价单")


def _format_record(record, index):
    source = record.get("source", "auto")
    source_label = "手工订单" if source == "manual" else "自动订单"
    symbol = record.get("symbol", "") or "-"
    order_type = record.get("order_type", "limit")
    type_label = _order_type_label(order_type)
    lines = [
        "",
        "─" * 60,
        f"  #{index}  {_ts_fmt(record.get('ts', ''))}  [{source_label}]",
        "─" * 60,
        f"  合约     : {symbol}",
        f"  订单类型 : {type_label}",
        f"  方向     : {record.get('side', '')}",
        f"  来源     : {source_label}",
        f"  数量     : {record.get('qty', '')} 手",
        f"  价格     : {record.get('price')}",
        f"  订单ID   : {record.get('order_id', '')}",
        f"  状态     : {record.get('status', '')}",
        f"  模式     : {record.get('mode', '')}",
    ]
    # 止损/止盈：建仓时带的计划止损价、止盈价（非订单类型；订单类型见上方「订单类型」）
    if record.get("stop_loss") is not None:
        lines.append(f"  计划止损 : {record['stop_loss']}")
    if record.get("take_profit") is not None:
        lines.append(f"  计划止盈 : {record['take_profit']}")
    if record.get("reason"):
        lines.append(f"  原因     : {record['reason']}")
    if record.get("error"):
        lines.append(f"  错误     : {record['error']}")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="订单 LOG 查询（多行显示）")
    ap.add_argument("-n", "--lines", type=int, default=20, help="显示最近 N 条，0=全部")
    ap.add_argument("--success", action="store_true", help="只看成功单")
    ap.add_argument("--fail", action="store_true", help="只看失败单")
    ap.add_argument("--auto", action="store_true", help="只看自动订单")
    ap.add_argument("--manual", action="store_true", help="只看手工订单")
    ap.add_argument("--file", type=str, default=ORDER_LOG_FILE, help="LOG 文件路径")
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"📭 文件不存在: {path}")
        return 0

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

    if args.success:
        records = [r for r in records if r.get("status") == "success"]
    if args.fail:
        records = [r for r in records if r.get("status") == "fail"]
    if args.auto:
        records = [r for r in records if r.get("source") == "auto"]
    if args.manual:
        records = [r for r in records if r.get("source") == "manual"]

    total = len(records)
    if args.lines > 0:
        records = records[-args.lines:]
    if not records:
        print("📭 无记录")
        return 0

    print("=" * 60)
    print("📋 订单 LOG")
    print("=" * 60)
    print(f"  文件: {path}")
    print(f"  显示: 最近 {len(records)} 条" + (f"（共 {total} 条）" if args.lines > 0 and total > len(records) else f"（共 {total} 条）"))
    print("=" * 60)

    for i, r in enumerate(records, start=1):
        print(_format_record(r, i))

    print("─" * 60)
    print("")
    return 0


if __name__ == "__main__":
    sys.exit(main())
