#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
核对规则：DEMO 运行的单在老虎后台数据中都能查到就算通过。
老虎后台可以比 DEMO 多（含人工单），不要求一一对应。

用法：在 tigertrade 根目录运行
  python scripts/verify_demo_orders_against_tiger.py

从 run/order_log.jsonl 取 mode=real、status=success 的 order_id（可选 source=auto 视为 DEMO 自动单），
用 openapicfg_dem 拉取老虎订单，检查上述每个 order_id 是否在老虎返回列表中；
全部查到则 exit 0 并打印通过，否则 exit 1 并列出未查到的 order_id。
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORDER_LOG = ROOT / "run" / "order_log.jsonl"
CONFIG_PATH = ROOT / "openapicfg_dem"


def _attr(o, *keys, default=None):
    for k in keys:
        if hasattr(o, k):
            v = getattr(o, k)
            if v is not None:
                return v
        if isinstance(o, dict):
            v = o.get(k)
            if v is not None:
                return v
    return default


def get_demo_real_success_order_ids(only_auto=True):
    """从 order_log.jsonl 取 mode=real, status=success 的 order_id；only_auto 时仅 source=auto。"""
    if not ORDER_LOG.exists():
        return []
    ids = []
    with open(ORDER_LOG, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("mode") != "real" or r.get("status") != "success":
                continue
            if only_auto and r.get("source") not in (None, "auto"):
                continue
            oid = r.get("order_id")
            if oid:
                ids.append(str(oid).strip())
    return ids


def fetch_tiger_order_ids(account, symbol_short, limit=500):
    """拉取老虎 DEMO 账户订单，返回 order_id 集合（字符串）。"""
    try:
        from tigeropen.tiger_open_config import TigerOpenClientConfig
        from tigeropen.trade.trade_client import TradeClient
    except ImportError:
        return None
    if not CONFIG_PATH.exists():
        return None
    try:
        config = TigerOpenClientConfig(props_path=str(CONFIG_PATH))
        client = TradeClient(config)
        acc = account or getattr(config, "account", None)
        if not acc:
            return None
        orders = client.get_orders(account=acc, symbol=symbol_short, limit=limit)
        if orders is None:
            orders = []
        ids = set()
        for o in orders:
            oid = _attr(o, "order_id", "id")
            if oid is not None:
                ids.add(str(oid).strip())
        return ids
    except Exception:
        return None


def main():
    demo_ids = get_demo_real_success_order_ids(only_auto=True)
    if not demo_ids:
        print("DEMO 侧无 mode=real、status=success 的订单（或 order_log 为空），跳过核对。")
        sys.exit(0)

    try:
        from src import tiger1 as t1
        symbol_api = t1._to_api_identifier(getattr(t1, "FUTURE_SYMBOL", "SIL.COMEX.202603"))
    except Exception:
        symbol_api = "SIL2603"
    account = None
    if CONFIG_PATH.exists():
        try:
            from tigeropen.tiger_open_config import TigerOpenClientConfig
            cfg = TigerOpenClientConfig(props_path=str(CONFIG_PATH))
            account = getattr(cfg, "account", None)
        except Exception:
            pass

    tiger_ids = fetch_tiger_order_ids(account, symbol_api)
    if tiger_ids is None:
        print("无法从老虎拉取订单（配置/网络/API），无法核对。")
        sys.exit(1)

    missing = [i for i in demo_ids if i not in tiger_ids]
    if not missing:
        print("核对通过：DEMO 侧 " + str(len(demo_ids)) + " 笔 real+success 订单在老虎后台均能查到。")
        sys.exit(0)
    print("核对未通过：DEMO 侧 " + str(len(demo_ids)) + " 笔中，以下 " + str(len(missing)) + " 笔在老虎后台未查到：")
    for oid in missing[:50]:
        print("  - " + str(oid))
    if len(missing) > 50:
        print("  ... 共 " + str(len(missing)) + " 笔")
    sys.exit(1)


if __name__ == "__main__":
    main()
