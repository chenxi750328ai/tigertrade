# -*- coding: utf-8 -*-
"""
平仓脚本可复用逻辑，供 close_demo_positions 与单测共用。
"""
import time


def is_real_order_id(oid):
    """order_id 为纯数字且较长才是老虎真实单，ORDER_/TEST_ 等为 mock"""
    if not oid:
        return False
    s = str(oid).strip()
    if len(s) < 10:
        return False
    if s.startswith('ORDER_') or s.startswith('TEST_') or 'Mock' in s:
        return False
    return s.isdigit()


def wait_order_fill(tc, oid, max_wait=30, poll_interval=1.5):
    """轮询订单直到 FILLED 或终态。老虎限制同一品种最多15个pending，必须等成交后再下下一单。"""
    start = time.time()
    while time.time() - start < max_wait:
        try:
            o = tc.get_order(id=oid)
            if o is None:
                time.sleep(poll_interval)
                continue
            st = getattr(o, 'status', None)
            st_str = (st.name if hasattr(st, 'name') else str(st)).upper()
            if st_str == 'FILLED':
                return True, getattr(o, 'filled', 0)
            if st_str in ('CANCELLED', 'REJECTED', 'EXPIRED'):
                return False, getattr(o, 'reason', None) or st_str
        except Exception:
            pass
        time.sleep(poll_interval)
    return False, 'timeout'
