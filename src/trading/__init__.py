"""
交易后端抽象层：统一交易接口，与具体券商/平台解耦。
策略与风控只依赖本包协议；具体实现由适配器提供（老虎、Mock、其他平台）。
"""
from src.trading.protocol import (
    TradingBackendProtocol,
    OrderSide,
    OrderType,
)

__all__ = [
    "TradingBackendProtocol",
    "OrderSide",
    "OrderType",
]
