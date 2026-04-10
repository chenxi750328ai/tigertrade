# -*- coding: utf-8 -*-
"""
Tiger Open SDK 常量兼容层（Leader/执行路径必读）

部分环境 tigeropen.common.consts 缺少 OrderSide，或 ImportError 与 OrderType 不同步。
下单热路径统一从此模块取 OrderType / OrderSide / TimeInForce，避免 order_log 出现
「cannot import name 'OrderSide'」类错误。
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from tigeropen.common.consts import OrderType
except ImportError:
    class OrderType:  # type: ignore
        LMT = "LMT"
        MKT = "MKT"
        STP = "STP"
        STP_LMT = "STP_LMT"

try:
    from tigeropen.common.consts import OrderSide, TimeInForce
except ImportError:
    logger.info("tigeropen.common.consts 无 OrderSide/TimeInForce，使用字符串兼容类（与 SDK 下单工具 action= 字符串一致）")

    class OrderSide:  # type: ignore
        BUY = "BUY"
        SELL = "SELL"

    class TimeInForce:  # type: ignore
        DAY = "DAY"
        GTC = "GTC"

__all__ = ["OrderType", "OrderSide", "TimeInForce"]
