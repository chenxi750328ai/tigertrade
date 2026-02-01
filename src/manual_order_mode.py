#!/usr/bin/env python3
"""
手工订单模式：用户自定义建仓逻辑（trigger/confirm/entry + stop_loss + take_profit）

做多：价格下降到 trigger → 回升到 confirm → 在 entry 建仓 → 止损 stop_loss / 止盈 take_profit
做空：价格上升到 trigger → 回落到 confirm → 在 entry 建仓 → 止损 stop_loss / 止盈 take_profit

字段名（原 A/B/C/D/E 已改为有意义命名，from_dict 仍兼容旧名）：
  trigger: 触发价（做多=低点，做空=高点）
  confirm: 确认价（做多=回升点，做空=回落点）
  entry:   建仓价
  stop_loss: 止损价
  take_profit: 止盈价
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any
import json
import numpy as np


class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"


class OrderState(str, Enum):
    """订单状态机"""
    PENDING_A = "pending_a"    # 等待触及 trigger
    PENDING_B = "pending_b"    # trigger 已触及，等待触及 confirm
    PENDING_C = "pending_c"    # confirm 已触及，等待触及 entry 建仓
    OPEN = "open"             # 已建仓，监控 stop_loss / take_profit
    CLOSED_STOP = "closed_stop"   # 止损触发
    CLOSED_PROFIT = "closed_profit"  # 止盈触发
    CANCELLED = "cancelled"


def _get(d: Dict[str, Any], *keys: str, default: float = 0) -> float:
    """从 dict 取第一个存在的 key，兼容旧名 A/B/C/D/E"""
    for k in keys:
        if k in d and d[k] is not None:
            return float(d[k])
    return default


@dataclass
class ManualOrderInstruction:
    """手工订单指令（trigger/confirm/entry/stop_loss/take_profit）"""
    direction: Direction
    trigger: float      # 做多=低点，做空=高点；direct_entry 时可忽略
    confirm: float      # 做多=回升点，做空=回落点；direct_entry 时可忽略
    entry: float        # 建仓点
    stop_loss: float    # 止损点
    take_profit: float  # 止盈点
    once: bool = True   # True=一次有效，False=多次有效
    direct_entry: bool = False  # True=不用 trigger/confirm，直接用 entry/stop_loss/take_profit 下单

    def validate(self) -> Optional[str]:
        """校验参数逻辑，返回错误信息"""
        if self.direct_entry:
            # 仅校验 entry / stop_loss / take_profit
            if self.direction == Direction.LONG:
                if not (self.stop_loss < self.entry):
                    return "做多止损应低于建仓价"
                if not (self.take_profit > self.entry):
                    return "做多止盈应高于建仓价"
            else:  # SHORT
                if not (self.stop_loss > self.entry):
                    return "做空止损应高于建仓价"
                if not (self.take_profit < self.entry):
                    return "做空止盈应低于建仓价"
            return None
        if self.direction == Direction.LONG:
            if not (self.trigger < self.confirm <= self.entry):
                return "做多要求: trigger < confirm <= entry"
            if not (self.stop_loss < self.entry):
                return "做多止损应低于建仓价"
            if not (self.take_profit > self.entry):
                return "做多止盈应高于建仓价"
        else:  # SHORT
            if not (self.trigger > self.confirm >= self.entry):
                return "做空要求: trigger > confirm >= entry"
            if not (self.stop_loss > self.entry):
                return "做空止损应高于建仓价"
            if not (self.take_profit < self.entry):
                return "做空止盈应低于建仓价"
        return None

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "direction": self.direction.value,
            "trigger": self.trigger,
            "confirm": self.confirm,
            "entry": self.entry,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "once": self.once,
        }
        if self.direct_entry:
            out["direct_entry"] = True
        return out

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ManualOrderInstruction":
        direction = Direction(d.get("direction", "long"))
        once = d.get("once", True)
        direct_entry = bool(d.get("direct_entry", False))
        entry_val = _get(d, "entry", "C")
        if direct_entry:
            trigger = confirm = entry_val  # 占位，不参与逻辑
        else:
            trigger = _get(d, "trigger", "A")
            confirm = _get(d, "confirm", "B")
        return cls(
            direction=direction,
            trigger=trigger,
            confirm=confirm,
            entry=entry_val,
            stop_loss=_get(d, "stop_loss", "D"),
            take_profit=_get(d, "take_profit", "E"),
            once=bool(once),
            direct_entry=direct_entry,
        )

    @classmethod
    def from_json(cls, s: str) -> "ManualOrderInstruction":
        return cls.from_dict(json.loads(s))


@dataclass
class ManualOrderState:
    """手工订单运行时状态"""
    instruction: ManualOrderInstruction
    state: OrderState = OrderState.PENDING_A  # direct_entry 时在创建后设为 PENDING_C

    def __post_init__(self):
        if getattr(self.instruction, "direct_entry", False):
            self.state = OrderState.PENDING_C
    entry_price: Optional[float] = None
    entry_idx: Optional[int] = None
    exit_price: Optional[float] = None
    exit_idx: Optional[int] = None
    exit_reason: Optional[str] = None  # "stop" | "profit"
    history: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instruction": self.instruction.to_dict(),
            "state": self.state.value,
            "entry_price": self.entry_price,
            "entry_idx": self.entry_idx,
            "exit_price": self.exit_price,
            "exit_idx": self.exit_idx,
            "exit_reason": self.exit_reason,
        }


def _check_hit(value: float, target: float, direction: Direction, is_trigger_down: bool) -> bool:
    """检查是否触及目标价。is_trigger_down=True 表示向下触及触发"""
    if is_trigger_down:
        return value <= target if direction == Direction.LONG else value >= target
    else:
        return value >= target if direction == Direction.LONG else value <= target


def update_order_state(
    order: ManualOrderState,
    bar_low: float,
    bar_high: float,
    bar_close: float,
    bar_idx: int,
) -> Optional[str]:
    """
    根据当前 K 线更新订单状态。返回 None 或 "stop" / "profit" 表示已平仓原因。
    """
    inst = order.instruction
    d = inst.direction

    if order.state == OrderState.CLOSED_STOP or order.state == OrderState.CLOSED_PROFIT or order.state == OrderState.CANCELLED:
        return None

    if order.state == OrderState.PENDING_A:
        hit_a = bar_low <= inst.trigger if d == Direction.LONG else bar_high >= inst.trigger
        if hit_a:
            order.state = OrderState.PENDING_B
        return None

    if order.state == OrderState.PENDING_B:
        hit_b = bar_high >= inst.confirm if d == Direction.LONG else bar_low <= inst.confirm
        if hit_b:
            order.state = OrderState.PENDING_C
        return None

    if order.state == OrderState.PENDING_C:
        hit_c = (bar_high >= inst.entry and bar_low <= inst.entry) if d == Direction.LONG else (bar_low <= inst.entry and bar_high >= inst.entry)
        if hit_c:
            order.state = OrderState.OPEN
            order.entry_price = inst.entry
            order.entry_idx = bar_idx
        return None

    if order.state == OrderState.OPEN:
        hit_d = bar_low <= inst.stop_loss if d == Direction.LONG else bar_high >= inst.stop_loss
        if hit_d:
            order.state = OrderState.CLOSED_STOP
            order.exit_price = inst.stop_loss
            order.exit_idx = bar_idx
            order.exit_reason = "stop"
            return "stop"
        hit_e = bar_high >= inst.take_profit if d == Direction.LONG else bar_low <= inst.take_profit
        if hit_e:
            order.state = OrderState.CLOSED_PROFIT
            order.exit_price = inst.take_profit
            order.exit_idx = bar_idx
            order.exit_reason = "profit"
            return "profit"
    return None


def run_backtest(
    df,
    instruction: ManualOrderInstruction,
    price_col: str = "price_current",
    timestamp_col: str = "timestamp",
    high_col: Optional[str] = None,
    low_col: Optional[str] = None,
    open_col: Optional[str] = None,
) -> Dict[str, Any]:
    """
    回测单个手工订单指令。

    df: 需包含价格列。若缺 open/high/low，则用 close 推算。
    """
    err = instruction.validate()
    if err:
        return {"error": err, "trades": [], "summary": {}}

    # 准备 OHLC
    close = df[price_col].values
    n = len(close)
    if open_col and open_col in df.columns:
        open_ = df[open_col].values
    else:
        open_ = np.roll(close, 1)
        open_[0] = close[0]
    if high_col and high_col in df.columns:
        high = df[high_col].values
    else:
        high = np.maximum(open_, close) * 1.0001  # 微小扩展以允许触及
    if low_col and low_col in df.columns:
        low = df[low_col].values
    else:
        low = np.minimum(open_, close) * 0.9999

    order = ManualOrderState(instruction=instruction)
    trades = []

    for i in range(n):
        result = update_order_state(order, float(low[i]), float(high[i]), float(close[i]), i)
        if result == "stop":
            pnl_pct = (order.exit_price - order.entry_price) / order.entry_price
            if instruction.direction == Direction.SHORT:
                pnl_pct = -pnl_pct
            trades.append({
                "entry_idx": order.entry_idx,
                "exit_idx": order.exit_idx,
                "entry_price": order.entry_price,
                "exit_price": order.exit_price,
                "exit_reason": "stop",
                "pnl_pct": pnl_pct * 100,
            })
            break
        if result == "profit":
            pnl_pct = (order.exit_price - order.entry_price) / order.entry_price
            if instruction.direction == Direction.SHORT:
                pnl_pct = -pnl_pct
            trades.append({
                "entry_idx": order.entry_idx,
                "exit_idx": order.exit_idx,
                "entry_price": order.entry_price,
                "exit_price": order.exit_price,
                "exit_reason": "profit",
                "pnl_pct": pnl_pct * 100,
            })
            break

    if not trades and order.state == OrderState.OPEN:
        # 持仓未平，按最后收盘价结算
        pnl_pct = (close[-1] - order.entry_price) / order.entry_price
        if instruction.direction == Direction.SHORT:
            pnl_pct = -pnl_pct
        trades.append({
            "entry_idx": order.entry_idx,
            "exit_idx": n - 1,
            "entry_price": order.entry_price,
            "exit_price": float(close[-1]),
            "exit_reason": "eod",
            "pnl_pct": pnl_pct * 100,
        })

    summary = {}
    if trades:
        t = trades[0]
        summary = {
            "executed": True,
            "exit_reason": t["exit_reason"],
            "pnl_pct": t["pnl_pct"],
            "entry_price": t["entry_price"],
            "exit_price": t["exit_price"],
        }
    else:
        summary = {"executed": False, "final_state": order.state.value}

    return {"trades": trades, "summary": summary, "final_state": order.state.value}


# ==================== 实时监控集成 ====================

_RUN_DIR = Path(__file__).resolve().parents[1] / "run"
MANUAL_ORDERS_FILE = str(_RUN_DIR / "manual_orders.json")
MANUAL_ORDERS_STATUS_FILE = str(_RUN_DIR / "manual_orders_status.json")


def _ensure_run_dir():
    """确保 run 目录存在"""
    _RUN_DIR.mkdir(parents=True, exist_ok=True)


def load_manual_orders_from_file(path: str = MANUAL_ORDERS_FILE) -> List[ManualOrderInstruction]:
    """从文件加载手工订单指令（JSON 数组或单对象）。读取后清空文件。"""
    _ensure_run_dir()
    try:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            orders = [ManualOrderInstruction.from_dict(o) for o in data]
        else:
            orders = [ManualOrderInstruction.from_dict(data)]
        # 读取后清空，避免重复执行
        with open(path, "w") as f:
            json.dump([], f)
        return orders
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_manual_order(inst: ManualOrderInstruction, path: str = MANUAL_ORDERS_FILE) -> None:
    """追加一条手工订单到文件（供监控进程读取并合并到待执行列表）"""
    _ensure_run_dir()
    try:
        existing = []
        with open(path) as f:
            existing = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing = []
    if not isinstance(existing, list):
        existing = []
    existing.append(inst.to_dict())
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)


def write_status_file(
    active: List[ManualOrderState],
    closed: List[ManualOrderState],
    path: str = MANUAL_ORDERS_STATUS_FILE,
) -> None:
    """将当前订单状态写入状态文件，供用户查看"""
    _ensure_run_dir()
    try:
        data = {
            "active_count": len(active),
            "active": [
                {"instruction": o.instruction.to_dict(), "state": o.state.value}
                for o in active
            ],
            "closed_count": len(closed),
            "closed": [
                {
                    "instruction": o.instruction.to_dict(),
                    "entry_price": o.entry_price,
                    "exit_price": o.exit_price,
                    "exit_reason": o.exit_reason,
                    "pnl_pct": round(
                        (o.exit_price - o.entry_price) / o.entry_price * 100, 2
                    )
                    if o.entry_price and o.exit_price
                    and o.instruction.direction == Direction.LONG
                    else round(
                        (o.entry_price - o.exit_price) / o.entry_price * 100, 2
                    )
                    if o.entry_price and o.exit_price
                    else None,
                }
                for o in closed
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except OSError:
        pass


class ManualOrderMonitor:
    """
    实时监控时的手工订单管理器。
    在监控循环中调用 on_price_update(open, high, low, close) 更新状态。
    """
    def __init__(
        self,
        orders_file: str = MANUAL_ORDERS_FILE,
        status_file: Optional[str] = MANUAL_ORDERS_STATUS_FILE,
        on_fill=None,
        on_close=None,
    ):
        self.orders_file = orders_file
        self.status_file = status_file  # 状态文件路径，None 则不写入
        self.active_orders: List[ManualOrderState] = []
        self.closed_orders: List[ManualOrderState] = []
        self.on_fill = on_fill  # 建仓回调 (order, price)
        self.on_close = on_close  # 平仓回调 (order, price, reason)

    def poll_new_orders(self) -> int:
        """检查文件中的新指令并加入 active_orders"""
        new = load_manual_orders_from_file(self.orders_file)
        for inst in new:
            if inst.validate():
                continue
            self.active_orders.append(ManualOrderState(instruction=inst))
        return len(new)

    def on_price_update(self, bar_open: float, bar_high: float, bar_low: float, bar_close: float, bar_idx: int = 0) -> List[Dict]:
        """
        用最新 K 线更新所有活跃订单状态。返回本 tick 发生的平仓事件列表。
        """
        self.poll_new_orders()
        closed_events = []
        still_active = []
        for order in self.active_orders:
            result = update_order_state(order, bar_low, bar_high, bar_close, bar_idx)
            if result:
                closed_events.append({
                    "order": order,
                    "reason": result,
                    "exit_price": order.exit_price,
                })
                self.closed_orders.append(order)
                if self.on_close:
                    self.on_close(order, order.exit_price, result)
                # 多次有效：平仓后重置，等待下次 A→B→C
                if not order.instruction.once:
                    still_active.append(ManualOrderState(instruction=order.instruction))
            else:
                if order.state == OrderState.OPEN and order.entry_idx == bar_idx and self.on_fill:
                    self.on_fill(order, order.entry_price)
                still_active.append(order)
        self.active_orders = still_active
        # 刷新状态文件供用户查看
        if self.status_file:
            write_status_file(self.active_orders, self.closed_orders, self.status_file)
        return closed_events
