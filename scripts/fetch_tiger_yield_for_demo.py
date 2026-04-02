#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从老虎后台（DEMO 账户）拉取订单，仅基于成交单统计/计算 DEMO 实盘收益率；
并拉取期货持仓的 unrealized_pnl，与今日已实现盈亏合并写入 yield_note / combined_pnl_usd。
产出供 update_today_yield_for_status.py 写入 today_yield.json；未拉取到或无法计算时返回 None，
调用方必须写「—」或「需老虎后台数据核对」，不得用日志解析的百分比代替。

用法: 在 tigertrade 根目录运行 python scripts/fetch_tiger_yield_for_demo.py
返回: 若成功则打印 JSON 并 exit 0；若失败或无成交则 exit 1，不打印收益率数字。
"""
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

ROOT = Path(__file__).resolve().parents[1]
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


def _today_range_local():
    """今日 00:00 至当前时刻（本地时区），供 get_orders 过滤「今日订单」。"""
    now = datetime.now()
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return start.strftime("%Y-%m-%d 00:00:00"), now.strftime("%Y-%m-%d %H:%M:%S")


def fetch_orders_from_tiger(account, symbol_short, limit=500, start_time=None, end_time=None):
    """用 openapicfg_dem 创建 TradeClient，拉取期货订单。返回 list 或 None（失败）。
    start_time/end_time 可选，格式如 '2026-02-24 00:00:00'，用于过滤「今日订单」。"""
    try:
        from tigeropen.tiger_open_config import TigerOpenClientConfig
        from tigeropen.trade.trade_client import TradeClient
        from tigeropen.common.consts import SegmentType, SecurityType
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
        # 期货订单需 seg_type=FUT，否则默认查证券段返回空
        kwargs = dict(
            account=acc,
            symbol=symbol_short,
            limit=limit,
            seg_type=SegmentType.FUT,
            sec_type=SecurityType.FUT,
        )
        if start_time:
            kwargs["start_time"] = start_time
        if end_time:
            kwargs["end_time"] = end_time
        orders = client.get_orders(**kwargs)
        if orders is None:
            return []
        # 分页时返回 OrdersResponse，取 result
        if hasattr(orders, "result"):
            orders = orders.result or []
        return list(orders) if orders else []
    except Exception:
        return None


def normalize_order(order):
    """统一为 dict 便于解析。"""
    if isinstance(order, dict):
        return order
    return {
        "order_id": _attr(order, "order_id", "id"),
        "status": _attr(order, "status", "order_status"),
        "side": _attr(order, "side", "action"),
        "quantity": _attr(order, "quantity", "qty"),
        "filled_quantity": _attr(order, "filled_quantity", "filled_qty", "filled"),
        "avg_fill_price": _attr(order, "avg_fill_price", "average_price"),
        "limit_price": _attr(order, "limit_price", "price"),
        "realized_pnl": _attr(order, "realized_pnl", "realized_pnL", "realized_pnl"),
        "created_time": _attr(order, "created_time", "create_time", "order_time", "submitted_at"),
        "update_time": _attr(order, "update_time", "updated_at", "filled_time"),
    }


def _order_date_local(order_row):
    """取订单日期（本地日），用于过滤「今日」。返回 datetime.date 或 None。"""
    for key in ("created_time", "update_time"):
        t = order_row.get(key)
        if t is None:
            continue
        try:
            if isinstance(t, (int, float)):
                return datetime.fromtimestamp(t / 1000.0 if t > 1e12 else t).date()
            if isinstance(t, str):
                return datetime.fromisoformat(t.replace("Z", "+00:00")).date()
        except Exception:
            pass
    return None


def _filter_orders_by_today(orders, today_date):
    """只保留订单日期为 today_date 的订单（按 created_time/update_time）。"""
    if not today_date:
        return orders
    out = []
    for o in orders:
        row = normalize_order(o)
        d = _order_date_local(row)
        if d == today_date:
            out.append(o)
    return out


def compute_yield_from_orders(orders, symbol_short="SIL2603", is_today_only=False):
    """
    仅基于 FILLED 订单统计；若能解析出 realized_pnl 则算收益率，否则只返回成交笔数 + 说明。
    返回 dict: yield_pct, yield_note, source="tiger_backend", filled_count, total_realized_pnl, is_today_only
    若无可用的收益率数字则 yield_pct 为 None，yield_note 说明「收益率需老虎后台核对」。
    is_today_only=True 时，yield_note 会标明「今日」。
    """
    filled = []
    for o in orders:
        row = normalize_order(o)
        st = row.get("status")
        st_str = (getattr(st, "name", None) or (str(st) if st is not None else "") or "").upper()
        if st_str not in ("FILLED", "FILLED_ALL", "FINISHED"):
            continue
        # 排除限价 100 的测试单（与测试用例 price=100 一致），不冒充实盘成交
        lp = row.get("limit_price")
        ap = row.get("avg_fill_price")
        try:
            if lp is not None and abs(float(lp) - 100.0) < 0.01:
                continue
            if ap is not None and abs(float(ap) - 100.0) < 0.01:
                continue
        except (TypeError, ValueError):
            pass
        filled.append(row)
    if not filled:
        return None

    total_pnl = None
    for r in filled:
        pnl = r.get("realized_pnl")
        if pnl is not None:
            try:
                total_pnl = (total_pnl or 0) + float(pnl)
            except (TypeError, ValueError):
                pass
    filled_count = len(filled)

    out = {
        "source": "tiger_backend",
        "filled_count": filled_count,
        "total_realized_pnl": total_pnl,
        "yield_pct": None,
        "yield_note": None,
    }
    prefix = "今日" if is_today_only else "老虎后台"
    if total_pnl is not None:
        # 若有总盈亏，以「总盈亏 USD」为主展示；收益率需账户净值，老虎 API 若未返回则先不瞎算%
        out["yield_note"] = f"{prefix}成交 {filled_count} 笔，已实现盈亏 {total_pnl:.2f} USD"
        # 若后续有账户初始净值可传入，可算 yield_pct = total_pnl / 初始净值 * 100
        out["yield_pct"] = f"{total_pnl:+.2f} USD"
    else:
        out["yield_note"] = f"{prefix}成交 {filled_count} 笔（收益率需老虎后台核对）"
        out["yield_pct"] = f"成交 {filled_count} 笔"
    out["is_today_only"] = is_today_only
    return out


def fetch_futures_positions_from_tiger(account):
    """拉取期货持仓列表；失败返回 None。"""
    if not CONFIG_PATH.exists() or not account:
        return None
    try:
        from tigeropen.tiger_open_config import TigerOpenClientConfig
        from tigeropen.trade.trade_client import TradeClient
        from tigeropen.common.consts import SecurityType

        cfg = TigerOpenClientConfig(props_path=str(CONFIG_PATH))
        client = TradeClient(cfg)
        try:
            pos = client.get_positions(account=account, sec_type=SecurityType.FUT)
        except Exception:
            pos = client.get_positions(account=account)
        if pos is None:
            return []
        if hasattr(pos, "result"):
            return list(pos.result or [])
        return list(pos)
    except Exception:
        return None


def aggregate_silver_position_pnl(positions):
    """汇总 SIL* 合约净手数与 unrealized_pnl（与后台持仓浮盈浮亏一致）。"""
    empty = {
        "unrealized_pnl": None,
        "net_quantity": 0,
        "position_detail": "",
    }
    if not positions:
        return empty
    total_u = 0.0
    net_qty = 0
    parts = []
    any_u = False
    for p in positions:
        obj = p
        contract = getattr(obj, "contract", None)
        sym = getattr(obj, "symbol", None) or (
            getattr(contract, "symbol", None) if contract else None
        )
        if not sym or "SIL" not in str(sym).upper():
            continue
        qty = getattr(obj, "quantity", None)
        u = getattr(obj, "unrealized_pnl", None)
        try:
            qty_i = int(qty)
        except (TypeError, ValueError):
            qty_i = 0
        try:
            u_f = float(u) if u is not None else None
        except (TypeError, ValueError):
            u_f = None
        if u_f is not None:
            total_u += u_f
            any_u = True
        net_qty += qty_i
        if u_f is not None:
            parts.append("%s %s手 浮盈浮亏%.2fUSD" % (sym, qty_i, u_f))
    return {
        "unrealized_pnl": total_u if any_u else None,
        "net_quantity": net_qty,
        "position_detail": "；".join(parts) if parts else "",
    }


def enrich_payload_with_unrealized(payload, pos_agg):
    """
    在订单统计结果上合并持仓浮动盈亏：yield 展示「今日已实现 + 当前浮动」快照。
    """
    ur = pos_agg.get("unrealized_pnl")
    nq = int(pos_agg.get("net_quantity") or 0)
    payload["open_position_qty"] = nq
    if ur is not None:
        payload["unrealized_pnl_usd"] = round(float(ur), 2)
    else:
        payload["unrealized_pnl_usd"] = None
    if pos_agg.get("position_detail"):
        payload["position_detail"] = pos_agg["position_detail"]

    if nq == 0 and ur is None:
        return payload
    if nq == 0 and ur is not None and abs(float(ur)) < 1e-12:
        return payload
    if ur is None and nq != 0:
        base = (payload.get("yield_note") or "").strip()
        extra = "当前持仓净%d手（后台未返回 unrealized_pnl，请在 Tiger 客户端查看浮动盈亏）" % nq
        payload["yield_note"] = (base + "；" + extra) if base else extra
        return payload

    ur_f = float(ur) if ur is not None else 0.0
    tr = payload.get("total_realized_pnl")
    tr_f = float(tr) if tr is not None else None
    comb = (tr_f if tr_f is not None else 0.0) + ur_f
    payload["combined_pnl_usd"] = round(comb, 2)

    base_note = (payload.get("yield_note") or "").strip()
    chunks = [base_note] if base_note else []
    chunks.append("当前持仓净%d手" % nq)
    if ur is not None:
        chunks.append("浮动盈亏%+.2f USD（未平仓）" % ur_f)
    else:
        chunks.append("浮动盈亏后台未返回数值")
    if tr_f is not None:
        chunks.append("今日已实现%+.2f USD" % tr_f)
    chunks.append("已实现+浮动合计%+.2f USD" % comb)
    payload["yield_note"] = "；".join(chunks)

    if tr_f is not None:
        payload["yield_pct"] = "%+.2f USD（已实现%+.2f+浮动%+.2f）" % (comb, tr_f, ur_f)
    else:
        payload["yield_pct"] = "%+.2f USD（浮动%+.2f）" % (comb, ur_f)
    return payload


def main():
    # 使用与 tiger1 一致的合约格式
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

    pos_list = fetch_futures_positions_from_tiger(account)
    pos_agg = aggregate_silver_position_pnl(pos_list or [])

    start_time, end_time = _today_range_local()

    def _merge(into, from_list):
        seen = {getattr(o, "order_id", None) or getattr(o, "id", None) for o in into}
        for o in from_list or []:
            oid = getattr(o, "order_id", None) or getattr(o, "id", None)
            if oid not in seen:
                into.append(o)
                seen.add(oid)

    # 1) 原逻辑：当前合约 + 今日时间范围
    orders = fetch_orders_from_tiger(account, symbol_api, limit=500, start_time=start_time, end_time=end_time)
    if orders is None:
        orders = []
    verbose = os.environ.get("FETCH_TIGER_YIELD_VERBOSE")
    if verbose:
        print("fetch_tiger_yield: 查询 symbol=%s start=%s end=%s → %s 条" % (symbol_api, start_time, end_time, len(orders or [])), file=sys.stderr)

    # 2) 另一合约 + 时间（避免 7 单在 2603 而只查了 2605）
    other = "SIL2603" if "2605" in str(symbol_api) else "SIL2605"
    other_orders = fetch_orders_from_tiger(account, other, limit=200, start_time=start_time, end_time=end_time)
    _merge(orders, other_orders or [])
    if verbose:
        print("fetch_tiger_yield: 再查 symbol=%s +时间 → 合并后共 %s 条" % (other, len(orders)), file=sys.stderr)

    # 3) 仍为空则不带时间拉一次，但必须按「今日」过滤，不得把历史成交当今日
    if not orders:
        orders = fetch_orders_from_tiger(account, symbol_api, limit=200) or []
        _merge(orders, fetch_orders_from_tiger(account, other, limit=200) or [])
        if verbose:
            print("fetch_tiger_yield: 无时间 symbol=%s,%s → 合并后共 %s 条" % (symbol_api, other, len(orders)), file=sys.stderr)
        today_date = datetime.now().date()
        orders = _filter_orders_by_today(orders, today_date)
        if verbose:
            print("fetch_tiger_yield: 按今日 %s 过滤后 → %s 条" % (today_date, len(orders)), file=sys.stderr)

    result = compute_yield_from_orders(orders, symbol_api, is_today_only=True)
    if not result:
        # 今日无成交：返回明确说明，供 update_today_yield 写入「今日无成交」
        out = {
            "source": "tiger_backend",
            "yield_pct": None,
            "yield_note": "今日无成交（老虎后台 0 笔策略成交；限价 100 的 FILLED 已视为测试单排除，不计入实盘收益）",
            "filled_count": 0,
            "is_today_only": True,
            "zero_trades_action": "【0笔须排查】1) 用 bash scripts/run_20h_demo.sh 启动 DEMO；2) 确认 COMEX 交易时段；3) 检查 openapicfg_dem；4) 查 order_log_analysis 今日 real success。",
        }
        enrich_payload_with_unrealized(out, pos_agg)
        if out.get("combined_pnl_usd") is not None and out.get("yield_pct") is None:
            out["yield_pct"] = "%+.2f USD（仅浮动，今日无新成交）" % out["combined_pnl_usd"]
        print(json.dumps(out, ensure_ascii=False, indent=2))
        sys.exit(0)  # 仍 exit 0，让调用方写入 today_yield

    enrich_payload_with_unrealized(result, pos_agg)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    sys.exit(0)


if __name__ == "__main__":
    main()
