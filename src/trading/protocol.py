"""
统一交易接口协议：与具体券商/平台解耦。
所有交易后端适配器（老虎、Mock、其他平台）实现本协议；
策略与执行层只依赖本协议，不依赖具体 API。
"""
from __future__ import annotations
from typing import Protocol, Any, Optional, List
from enum import Enum


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MKT = "MKT"
    LMT = "LMT"
    STP = "STP"
    STP_LMT = "STP_LMT"


class TradingBackendProtocol(Protocol):
    """
    统一交易后端协议。
    实现者：RealTradeApiAdapter（老虎）、MockTradeApiAdapter、以及未来的其他券商适配器。
    """

    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: int,
        time_in_force: Optional[str] = None,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        **kwargs: Any,
    ) -> Any:
        """下单。返回订单对象或订单 ID（由适配器决定）。"""
        ...

    def cancel_order(self, order_id: str, **kwargs: Any) -> Any:
        """撤单。部分后端可能暂未实现，可抛出 NotImplementedError。"""
        ...

    def get_orders(
        self,
        account: Optional[str] = None,
        symbol: Optional[str] = None,
        limit: int = 100,
        **kwargs: Any,
    ) -> List[Any]:
        """查询订单列表。"""
        ...

    def get_order(
        self,
        order_id: Optional[str] = None,
        account: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[Any]:
        """查询单个订单。"""
        ...

    def get_positions(self, account: Optional[str] = None, **kwargs: Any) -> List[Any]:
        """查询持仓。部分后端可能暂未实现。"""
        ...

    def get_account(self, account: Optional[str] = None, **kwargs: Any) -> Optional[Any]:
        """查询账户信息。部分后端可能暂未实现。"""
        ...
