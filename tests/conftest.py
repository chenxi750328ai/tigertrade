# tests/conftest.py
"""Pytest 配置：测试时 order_log 写入临时路径，不污染 run/order_log.jsonl。"""
import os
import tempfile

import pytest


@pytest.fixture(scope="session", autouse=True)
def _order_log_use_temp_dir():
    """所有测试共用临时目录写 order_log，避免 Mock/TEST_ 等写入生产 order_log。"""
    d = tempfile.mkdtemp(prefix="tigertrade_test_order_log_")
    os.makedirs(d, exist_ok=True)
    from src import order_log

    order_log.ORDER_LOG_FILE = os.path.join(d, "order_log.jsonl")
    order_log.API_FAILURE_FOR_SUPPORT_FILE = os.path.join(d, "api_failure_for_support.jsonl")
    yield
    try:
        for f in (order_log.ORDER_LOG_FILE, order_log.API_FAILURE_FOR_SUPPORT_FILE):
            if os.path.isfile(f):
                os.remove(f)
        os.rmdir(d)
    except OSError:
        pass
