# tests/conftest.py
"""Pytest 配置：仓库根加入 sys.path（可移植，不依赖 /home/cx）；order_log / DFX 写入临时目录。"""
import os
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest


@pytest.fixture(scope="session", autouse=True)
def _order_log_use_temp_dir():
    """所有测试共用临时目录写 order_log 与 dfx_execution，避免污染生产 run/。"""
    d = tempfile.mkdtemp(prefix="tigertrade_test_order_log_")
    os.makedirs(d, exist_ok=True)
    from src import order_log

    order_log.ORDER_LOG_FILE = os.path.join(d, "order_log.jsonl")
    order_log.API_FAILURE_FOR_SUPPORT_FILE = os.path.join(d, "api_failure_for_support.jsonl")
    order_log.DFX_EXECUTION_FILE = os.path.join(d, "dfx_execution.jsonl")
    yield
    try:
        for f in (
            order_log.ORDER_LOG_FILE,
            order_log.API_FAILURE_FOR_SUPPORT_FILE,
            order_log.DFX_EXECUTION_FILE,
        ):
            if os.path.isfile(f):
                os.remove(f)
        os.rmdir(d)
    except OSError:
        pass
