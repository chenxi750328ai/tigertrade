#!/usr/bin/env python3
"""
订单业务 LOG：用户只看业务订单记录，不看运行日志。

路径：tigertrade/run/order_log.jsonl
格式：每行一个 JSON，便于追加与解析
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Any

_RUN_DIR = Path(__file__).resolve().parents[1] / "run"
ORDER_LOG_FILE = str(_RUN_DIR / "order_log.jsonl")
API_FAILURE_FOR_SUPPORT_FILE = str(_RUN_DIR / "api_failure_for_support.jsonl")


def _ensure_dir():
    _RUN_DIR.mkdir(parents=True, exist_ok=True)


def log_order(
    side: str,
    quantity: int,
    price: float,
    order_id: str,
    status: str,  # "success" | "fail"
    mode: str = "mock",  # "mock" | "real"
    stop_loss_price: Optional[float] = None,
    take_profit_price: Optional[float] = None,
    reason: str = "",
    error: Optional[str] = None,
    extra: Optional[dict] = None,
    source: str = "auto",  # "auto" | "manual" 自动订单 | 手工订单
    symbol: str = "",  # 合约代码，如 SIL2603
    order_type: str = "limit",  # "market" 市价单 | "limit" 限价单(现价单) | "stop_loss" 止损单 | "take_profit" 止盈单
) -> None:
    """
    写入一条订单记录到 order_log.jsonl
    order_type 取值：market=市价单, limit=限价单(现价单), stop_loss=止损单, take_profit=止盈单
    """
    _ensure_dir()
    allowed = ("market", "limit", "stop_loss", "take_profit")
    order_type_val = order_type if order_type in allowed else "limit"
    record = {
        "ts": datetime.now().isoformat(),
        "side": side,
        "symbol": symbol or "",
        "order_type": order_type_val,
        "source": source if source in ("auto", "manual") else "auto",
        "qty": quantity,
        "price": float(price) if price is not None else None,
        "order_id": str(order_id),
        "status": status,
        "mode": mode,
        "stop_loss": float(stop_loss_price) if stop_loss_price is not None else None,
        "take_profit": float(take_profit_price) if take_profit_price is not None else None,
        "reason": reason or "",
        "error": error or "",
    }
    if extra:
        record["extra"] = extra
    try:
        with open(ORDER_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


def log_take_profit(
    side: str,
    qty: int,
    price: float,
    order_id: str,
    status: str,
    mode: str = "mock",
    source: str = "auto",
    symbol: str = "",
) -> None:
    """写入止盈单记录，order_type 固定为 take_profit（止盈单）"""
    log_order(
        side=side,
        quantity=qty,
        price=price,
        order_id=order_id,
        status=status,
        mode=mode,
        reason="take_profit",
        source=source,
        symbol=symbol,
        order_type="take_profit",
    )


def log_stop_loss(
    side: str,
    quantity: int,
    price: float,
    order_id: str,
    status: str,
    mode: str = "mock",
    source: str = "auto",
    symbol: str = "",
    stop_loss_price: Optional[float] = None,
    take_profit_price: Optional[float] = None,
    reason: str = "stop_loss",
    error: Optional[str] = None,
) -> None:
    """写入止损单记录，order_type 固定为 stop_loss（止损单）"""
    log_order(
        side=side,
        quantity=quantity,
        price=price,
        order_id=order_id,
        status=status,
        mode=mode,
        stop_loss_price=stop_loss_price,
        take_profit_price=take_profit_price,
        reason=reason or "stop_loss",
        error=error or "",
        source=source,
        symbol=symbol,
        order_type="stop_loss",
    )


def log_api_failure_for_support(
    side: str,
    quantity: int,
    price: Optional[float],
    symbol_submitted: str,
    order_type_api: str,
    time_in_force: str,
    limit_price: Optional[float],
    stop_price: Optional[float],
    error: str,
    source: str = "auto",
    order_id: str = "",
) -> None:
    """
    API 失败时写入完整订单参数，便于提供给老虎客服排查（为何 APP 可下单、API 报错）。
    写入 run/api_failure_for_support.jsonl，每行一条 JSON。
    """
    _ensure_dir()
    record = {
        "ts": datetime.now().isoformat(),
        "source": source,
        "side": side,
        "quantity": quantity,
        "price": float(price) if price is not None else None,
        "symbol_submitted": symbol_submitted,
        "order_type_api": order_type_api,
        "time_in_force": time_in_force,
        "limit_price": float(limit_price) if limit_price is not None else None,
        "stop_price": float(stop_price) if stop_price is not None else None,
        "order_id": order_id,
        "error": error,
    }
    try:
        with open(API_FAILURE_FOR_SUPPORT_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass
