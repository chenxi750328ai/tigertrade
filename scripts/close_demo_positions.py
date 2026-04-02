#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
平仓脚本：将 DEMO 账户持仓恢复为 0（多头和空头）
用法:
  cd /home/cx/tigertrade && python scripts/close_demo_positions.py d
  # 若后台有两手持仓但 sync 显示 0，强制平 2 手多头：
  python scripts/close_demo_positions.py d --long 2
  # 强制平空头：--short N
使用 openapicfg_dem 配置，拉取老虎持仓后：
- 多头：SELL 平仓
- 空头：BUY 平仓（买入平仓）
必须真实下单，禁止 mock。
"""
import sys
import os

sys.path.insert(0, '/home/cx/tigertrade')
os.chdir('/home/cx/tigertrade')
# 必须在 import tiger1 前设置，否则会走 mock 不下真实单
if len(sys.argv) < 2 or sys.argv[1] not in ('d', 'c'):
    sys.argv = [sys.argv[0], 'd']

def _parse_force_close():
    """解析 --long N 与 --short N，返回 (force_long, force_short)。未指定则为 None。"""
    force_long, force_short = None, None
    argv = sys.argv
    i = 2  # 跳过 script, d/c
    while i < len(argv) - 1:
        if argv[i] == '--long':
            try:
                force_long = max(0, int(argv[i + 1]))
            except (ValueError, TypeError):
                pass
            i += 2
        elif argv[i] == '--short':
            try:
                force_short = max(0, int(argv[i + 1]))
            except (ValueError, TypeError):
                pass
            i += 2
        else:
            i += 1
    return force_long, force_short

from scripts.close_utils import is_real_order_id as _is_real_order_id, wait_order_fill as _wait_order_fill

def main():
    from src import tiger1 as t1
    from src.api_adapter import api_manager
    print(f"[校验] is_mock_mode={api_manager.is_mock_mode}, trade_api={type(api_manager.trade_api).__name__}")
    if api_manager.is_mock_mode:
        print("❌ 当前为 Mock 模式，不会下真实单。请确保 openapicfg_dem 存在且可加载。")
        return 1
    # 初始化 API（与 tiger1 一致）
    if t1.api_manager.trade_api is None and hasattr(t1, 'trade_client') and t1.trade_client:
        from src.api_adapter import api_manager
        acc = getattr(t1.trade_client.config, 'account', None) if hasattr(t1.trade_client, 'config') else None
        api_manager.initialize_real_apis(t1.quote_client, t1.trade_client, account=acc)
    t1.sync_positions_from_backend()
    # 老虎限制同一品种最多15个pending，先取消现有挂单
    tc = t1.trade_client
    acc = getattr(t1.client_config, 'account', None) or getattr(tc.config, 'account', None)
    try:
        from tigeropen.common.consts import SecurityType
        open_orders = tc.get_open_orders(account=acc, sec_type=SecurityType.FUT) or []
        to_cancel = [o for o in open_orders if 'SIL' in str(getattr(getattr(o, 'contract', None), 'symbol', '') or '')]
        if to_cancel:
            print(f"⚠️ 发现 {len(to_cancel)} 笔 SIL 挂单，先取消...")
            for o in to_cancel:
                oid = getattr(o, 'id', None)
                try:
                    tc.cancel_order(id=oid)
                    print(f"  已撤单 id={oid}")
                except Exception as e:
                    print(f"  撤单 id={oid} 失败: {e}")
            import time
            time.sleep(3)
    except Exception as e:
        print(f"取消挂单时异常: {e}")
    t1.sync_positions_from_backend()
    long_qty = t1.current_position
    short_qty = getattr(t1, 'current_short_position', 0)
    force_long, force_short = _parse_force_close()
    if force_long is not None:
        long_qty = max(long_qty, force_long)
    if force_short is not None:
        short_qty = max(short_qty, force_short)
    if long_qty <= 0 and short_qty <= 0:
        print("✅ 当前无持仓，无需平仓")
        print("若后台实际有仓（例如两手持仓）但此处显示 0，请执行: python scripts/close_demo_positions.py d --long 2")
        return 0
    print(f"⚠️ 当前持仓: 多头 {long_qty} 手, 空头 {short_qty} 手，开始平仓...")
    # 先平空、再平多（避免平多后 net 变空头，再 BUY 时被误判）
    order_of_close = []  # [(side, qty, label), ...]
    if short_qty > 0:
        order_of_close.append(('BUY', short_qty, 'short', 'close_short_positions_restore'))
    if long_qty > 0:
        order_of_close.append(('SELL', long_qty, 'long', 'close_positions_restore'))
    try:
        sym = t1._to_api_identifier(t1.FUTURE_SYMBOL) if hasattr(t1, '_to_api_identifier') else 'SIL2603'
        tick = t1.api_manager.quote_api.get_quote([sym])
        if tick and len(tick) > 0:
            ask = getattr(tick[0], 'ask_price', None)
            last = getattr(tick[0], 'last_price', None)
            bid = getattr(tick[0], 'bid_price', None)
            price = float(ask or last or bid or 82.0)
        else:
            price = 82.0
    except Exception:
        price = 82.0
    # 平多用 SELL 略低于 bid；平空用 BUY 略高于 ask，确保限价单能成交
    price_sell = price - 0.05  # 平多：略低便于成交
    price_buy = price + 0.05  # 平空：略高便于成交
    import time
    long_ok, long_fail = 0, 0
    short_ok, short_fail = 0, 0
    # 老虎限制同一品种最多15个pending，需分批：每批最多15手，等成交后再下一批；或单笔大单
    tc = t1.trade_client
    BATCH = 15  # 每批最多15单，避免超限
    for side, total, label, reason in order_of_close:
        for batch_start in range(0, total, BATCH):
            batch_size = min(BATCH, total - batch_start)
            t1.place_tiger_order(side, batch_size, None, reason=reason)  # 市价单，一次下batch_size手
            ok, oid = getattr(t1, '_last_place_order_result', (False, None))
            if not ok or not _is_real_order_id(oid):
                if label == 'short':
                    short_fail += batch_size
                else:
                    long_fail += batch_size
                why = f"order_id={oid} (Mock)" if oid and not _is_real_order_id(oid) else "API返回失败"
                print(f"  ❌ 第 {batch_start+1}-{batch_start+batch_size}/{total} 手{'空头' if label=='short' else '多头'} 提交失败: {why}")
                continue
            filled, info = _wait_order_fill(tc, oid, max_wait=60, poll_interval=2)
            if filled:
                if label == 'short':
                    short_ok += batch_size
                else:
                    long_ok += batch_size
                print(f"  ✅ 第 {batch_start+1}-{batch_start+batch_size}/{total} 手{'空头' if label=='short' else '多头'} 已成交 order_id={oid}")
            else:
                if label == 'short':
                    short_fail += batch_size
                else:
                    long_fail += batch_size
                print(f"  ❌ 第 {batch_start+1}-{batch_start+batch_size}/{total} 手{'空头' if label=='short' else '多头'} 未成交: {info}")
    print("")
    print(f"📊 平仓汇总: 多头 成功{long_ok}/失败{long_fail}, 空头 成功{short_ok}/失败{short_fail}")
    # 重新同步，校验实际剩余持仓
    import time
    time.sleep(3)
    t1.sync_positions_from_backend()
    remain_long = t1.current_position
    remain_short = getattr(t1, 'current_short_position', 0)
    if remain_long > 0 or remain_short > 0:
        print(f"⚠️ 后台仍有持仓: 多头 {remain_long} 手, 空头 {remain_short} 手。平仓单可能未成交或已拒，请人工核对。")
        return 1
    print("✅ 后台持仓已归零，平仓完成。")
    return 0

if __name__ == '__main__':
    sys.exit(main())
