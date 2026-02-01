#!/usr/bin/env python3
"""
输出 API 失败时的订单参数详情，便于提供给老虎客服排查（为何 APP 可下单、API 报错）。

用法:
  python scripts/print_api_failure_for_support.py   # 从 api_failure_for_support.jsonl 或 order_log.jsonl 取最近一条失败
  python scripts/print_api_failure_for_support.py -n 3   # 最近 3 条
"""
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.order_log import ORDER_LOG_FILE, API_FAILURE_FOR_SUPPORT_FILE


def _read_lines(path: Path) -> list:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _format_api_failure(record: dict, index: int) -> str:
    """客服用：订单参数详情（可直接复制给客服）"""
    lines = [
        "",
        "=" * 60,
        f"  API 失败记录 #{index}（可提供给老虎客服）",
        "=" * 60,
        f"  时间         : {record.get('ts', '')}",
        f"  来源         : {record.get('source', '')} （auto=自动订单, manual=手工订单）",
        f"  方向         : {record.get('side', '')}",
        f"  数量         : {record.get('quantity', '')} 手",
        f"  价格         : {record.get('price')}",
        f"  提交合约     : {record.get('symbol_submitted', '')}",
        f"  订单类型(API): {record.get('order_type_api', '')} （LMT=限价, MKT=市价）",
        f"  有效期限     : {record.get('time_in_force', '')}",
        f"  限价         : {record.get('limit_price')}",
        f"  止损价       : {record.get('stop_price')}",
        f"  订单ID       : {record.get('order_id', '')}",
        f"  错误信息     : {record.get('error', '')}",
        "=" * 60,
        "",
    ]
    return "\n".join(lines)


def _record_from_order_log(line: dict) -> dict:
    """把 order_log 一条转成 api_failure 格式（缺的字段用空）"""
    return {
        "ts": line.get("ts", ""),
        "source": line.get("source", "auto"),
        "side": line.get("side", ""),
        "quantity": line.get("qty"),
        "price": line.get("price"),
        "symbol_submitted": line.get("symbol", ""),
        "order_type_api": "LMT" if line.get("price") else "MKT",
        "time_in_force": "DAY",
        "limit_price": line.get("price"),
        "stop_price": None,
        "order_id": line.get("order_id", ""),
        "error": line.get("error", ""),
    }


def main():
    ap = argparse.ArgumentParser(description="输出 API 失败订单参数（客服用）")
    ap.add_argument("-n", "--num", type=int, default=1, help="显示最近 N 条失败，默认 1")
    ap.add_argument("--file", type=str, default=None, help="指定 api_failure 或 order_log 文件路径")
    args = ap.parse_args()

    records = []
    path_support = Path(API_FAILURE_FOR_SUPPORT_FILE)
    path_order_log = Path(ORDER_LOG_FILE)

    if args.file:
        p = Path(args.file)
        if p.exists():
            for line in _read_lines(p):
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        # 若指定的是 order_log，只保留 fail + real
        if "order_log" in str(p):
            records = [r for r in records if r.get("status") == "fail" and r.get("mode") == "real"]
            records = [_record_from_order_log(r) for r in records]
    else:
        # 优先 api_failure_for_support.jsonl
        for line in _read_lines(path_support):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if not records and path_order_log.exists():
            for line in _read_lines(path_order_log):
                try:
                    r = json.loads(line)
                    if r.get("status") == "fail" and r.get("mode") == "real":
                        records.append(_record_from_order_log(r))
                except json.JSONDecodeError:
                    continue

    if not records:
        print("📭 暂无 API 失败记录。")
        print("   实盘下单失败时会写入:", API_FAILURE_FOR_SUPPORT_FILE)
        print("   或从 order_log.jsonl 中 status=fail, mode=real 的记录查看。")
        return 0

    n = min(args.num, len(records))
    show = records[-n:]
    print("📋 API 失败订单参数详情（可复制给老虎客服）")
    for i, r in enumerate(show, start=1):
        print(_format_api_failure(r, i))
    return 0


if __name__ == "__main__":
    sys.exit(main())
