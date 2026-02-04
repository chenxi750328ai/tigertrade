"""
交易后端工厂：根据配置返回当前使用的交易后端适配器。
通过环境变量 TRADING_BACKEND 选择：tiger | mock | 未来可扩展 other。
"""
import os
import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)

TRADING_BACKEND_ENV = "TRADING_BACKEND"
# 可选值: tiger (真实老虎), mock (模拟), 后续可加 other_xxx
BACKEND_TIGER = "tiger"
BACKEND_MOCK = "mock"


def get_trading_backend() -> Optional[Any]:
    """
    获取当前交易后端实例。
    依赖 api_manager 已初始化（initialize_real_apis 或 initialize_mock_apis）。
    返回实现 TradingBackendProtocol 的适配器。
    """
    from src.api_adapter import api_manager
    backend = os.getenv(TRADING_BACKEND_ENV, "").strip().lower()
    # 未设置时：根据 api_manager 当前模式推断
    if not backend:
        if getattr(api_manager, "is_mock_mode", True):
            backend = BACKEND_MOCK
        else:
            backend = BACKEND_TIGER
    if backend == BACKEND_MOCK and api_manager.is_mock_mode:
        return api_manager.trade_api
    if backend == BACKEND_TIGER and not api_manager.is_mock_mode:
        return api_manager.trade_api
    # 当前运行时与 TRADING_BACKEND 不一致时，仍返回当前已初始化的 adapter
    logger.debug("TRADING_BACKEND=%s, 实际使用已初始化的 trade_api", backend)
    return api_manager.trade_api


def get_backend_name() -> str:
    """返回当前后端名称，用于日志与 SKILL 说明。"""
    from src.api_adapter import api_manager
    if getattr(api_manager, "is_mock_mode", True):
        return BACKEND_MOCK
    return BACKEND_TIGER
