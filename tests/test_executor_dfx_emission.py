# -*- coding: utf-8 -*-
"""
门禁：OrderExecutor 在组合单成功/失败+回退路径必须写 DFX 事件（可服务性 §3.4）。

与 tests/test_executor_modules.py 行为一致，额外断言 run/dfx_execution.jsonl（测试中为临时文件）内容。
"""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _dfx_events(path: str):
    p = Path(path)
    if not p.is_file():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").strip().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def test_execute_buy_bracket_success_emits_bracket_submitted():
    from src import order_log
    from src.api_adapter import api_manager
    from src.executor import OrderExecutor
    from src import tiger1 as t1

    t1.current_position = 0
    t1.daily_loss = 0
    api_manager.trade_api = None
    api_manager.is_mock_mode = True

    mock_order_result = MagicMock()
    mock_order_result.order_id = "TEST_ORDER_123"
    mock_trade_api = MagicMock(
        spec=[
            "place_order",
            "place_limit_with_bracket",
            "get_order",
            "get_orders",
            "wait_until_buy_filled",
            "account",
        ]
    )
    mock_trade_api.account = "MOCK_ACC"
    mock_trade_api.place_limit_with_bracket.return_value = mock_order_result
    mock_trade_api.get_order.return_value = mock_order_result
    mock_trade_api.get_orders.return_value = []

    executor = OrderExecutor(t1)
    try:
        api_manager.trade_api = mock_trade_api
        api_manager.is_mock_mode = False
        with patch.object(t1, "sync_positions_from_backend", return_value=None), patch.object(
            t1, "get_effective_position_for_buy", return_value=0
        ):
            ok, msg = executor.execute_buy(
                price=100.0,
                atr=0.5,
                grid_lower=98.0,
                grid_upper=105.0,
                confidence=0.6,
            )
        assert ok, msg
        events = [e.get("event") for e in _dfx_events(order_log.DFX_EXECUTION_FILE)]
        assert "bracket_submitted" in events, f"DFX 应含 bracket_submitted，实际 events={events}"
    finally:
        api_manager.trade_api = None
        api_manager.is_mock_mode = True
        t1.current_position = 0
        t1.daily_loss = 0


def test_execute_buy_bracket_fail_fallback_emits_bracket_failed_and_sl_tp_dfx():
    from src import order_log
    from src.api_adapter import api_manager
    from src.executor import OrderExecutor
    from src import tiger1 as t1

    t1.current_position = 0
    t1.daily_loss = 0
    mock_order_result = MagicMock()
    mock_order_result.order_id = "MAIN_BRACKET_FAIL"
    mock_trade_api = MagicMock(
        spec=[
            "place_order",
            "place_limit_with_bracket",
            "get_order",
            "get_orders",
            "wait_until_buy_filled",
            "account",
        ]
    )
    mock_trade_api.account = "MOCK_ACC"
    mock_trade_api.place_limit_with_bracket.side_effect = ValueError("BRACKETS not supported")
    mock_trade_api.place_order.return_value = mock_order_result
    mock_trade_api.get_order.return_value = mock_order_result
    mock_trade_api.get_orders.return_value = []
    mock_trade_api.wait_until_buy_filled.return_value = True

    executor = OrderExecutor(t1)
    try:
        api_manager.trade_api = mock_trade_api
        api_manager.is_mock_mode = False
        with patch.object(t1, "sync_positions_from_backend", return_value=None), patch.object(
            t1, "get_effective_position_for_buy", return_value=0
        ), patch.object(t1, "compute_stop_loss", return_value=(98.0, 2.0)), patch.object(
            executor, "_calculate_take_profit", return_value=104.9
        ):
            ok, msg = executor.execute_buy(
                price=100.0,
                atr=0.5,
                grid_lower=98.0,
                grid_upper=105.0,
                confidence=0.6,
            )
        assert ok, msg
        events = [e.get("event") for e in _dfx_events(order_log.DFX_EXECUTION_FILE)]
        assert "bracket_failed" in events, events
        assert "stop_loss_submitted" in events, events
        assert "take_profit_submitted" in events, events
    finally:
        api_manager.trade_api = None
        api_manager.is_mock_mode = True
        t1.current_position = 0
        t1.daily_loss = 0
