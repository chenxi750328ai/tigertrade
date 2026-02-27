#!/usr/bin/env python3
"""
扫描 order_log.jsonl，检测超买/超卖：1 小时内 BUY/SELL success (real) > 20 视为异常。
超卖（SELL 过多）= 可能卖出开仓导致空单堆积。
用于 DFX 可服务性，见 docs/回溯_62手持仓_数据定位与DFX改进.md。
用法: python scripts/analyze_order_log_for_overbuy.py
"""
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "run" / "order_log.jsonl"
THRESHOLD = 20  # 1h 内超过此数视为超买
MAX_AGE_HOURS = 24  # 只检测最近 N 小时内的订单，避免历史超买一直告警


def _is_real_success_buy(r):
    if r.get("side") != "BUY" or r.get("status") != "success" or r.get("mode") != "real":
        return False
    oid = str(r.get("order_id", ""))
    return len(oid) >= 10 and oid.isdigit()


def _is_real_success_sell(r):
    if r.get("side") != "SELL" or r.get("status") != "success" or r.get("mode") != "real":
        return False
    oid = str(r.get("order_id", ""))
    return len(oid) >= 10 and oid.isdigit()


def main():
    if not LOG.exists():
        print("order_log 不存在")
        return 0
    buys = []
    sells = []
    with open(LOG) as f:
        for line in f:
            try:
                r = json.loads(line)
                ts = r.get("ts", "")
                if ts:
                    if _is_real_success_buy(r):
                        buys.append(ts)
                    elif _is_real_success_sell(r):
                        sells.append(ts)
            except Exception:
                pass
    cutoff = datetime.utcnow() - timedelta(hours=MAX_AGE_HOURS)
    exit_code = 0
    # 超买检测
    by_hour_buy = defaultdict(int)
    for ts in buys:
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if dt.tzinfo:
                dt_naive = dt.astimezone().replace(tzinfo=None)
            else:
                dt_naive = dt
            if dt_naive < cutoff:
                continue
            bucket = dt_naive.strftime("%Y-%m-%d %H:00")
            by_hour_buy[bucket] += 1
        except Exception:
            pass
    over_buy = [(h, c) for h, c in by_hour_buy.items() if c > THRESHOLD]
    if over_buy:
        print("超买检测: 以下时段 1h 内 BUY success (real) > %s:" % THRESHOLD)
        for h, c in sorted(over_buy, key=lambda x: -x[1]):
            print("  %s: %s 笔" % (h, c))
        exit_code = 1
    # 超卖检测（卖出开仓导致空单堆积）
    by_hour_sell = defaultdict(int)
    for ts in sells:
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if dt.tzinfo:
                dt_naive = dt.astimezone().replace(tzinfo=None)
            else:
                dt_naive = dt
            if dt_naive < cutoff:
                continue
            bucket = dt_naive.strftime("%Y-%m-%d %H:00")
            by_hour_sell[bucket] += 1
        except Exception:
            pass
    over_sell = [(h, c) for h, c in by_hour_sell.items() if c > THRESHOLD]
    if over_sell:
        print("超卖检测（可能卖出开仓导致空单堆积）: 以下时段 1h 内 SELL success (real) > %s:" % THRESHOLD)
        for h, c in sorted(over_sell, key=lambda x: -x[1]):
            print("  %s: %s 笔" % (h, c))
        exit_code = 1
    if exit_code == 0:
        print("未发现超买/超卖（1h 内 BUY/SELL > %s 笔）" % THRESHOLD)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
