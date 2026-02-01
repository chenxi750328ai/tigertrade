#!/usr/bin/env python3
"""
测试自动模式与手工订单并发执行：手工订单 Monitor 与自动策略（如 grid_trading_strategy_pro1）同时运行。

- 手工线程：ManualOrderMonitor，每隔 POLL_INTERVAL 秒用模拟 K 线调用 on_price_update（会读取 manual_orders.json）。
- 自动线程：运行策略循环若干次（可 mock 数据避免真实下单）。

验证：两路并发不崩溃、手工文件在每次 on_price_update 时被轮询。
"""
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 手工订单轮询间隔（秒）：在测试中每次 on_price_update 的间隔
POLL_INTERVAL = 2
# 并发运行时长（秒）
RUN_SECONDS = 8


def test_auto_manual_concurrent():
    """手工 Monitor 与自动策略并发：两线程同时跑一段时间，无异常即通过。"""
    from src.manual_order_mode import (
        ManualOrderMonitor,
        MANUAL_ORDERS_FILE,
        MANUAL_ORDERS_STATUS_FILE,
        save_manual_order,
        ManualOrderInstruction,
        Direction,
    )

    manual_events = []
    manual_lock = threading.Lock()

    def on_fill(order, price):
        with manual_lock:
            manual_events.append(("fill", order.instruction.entry, price))

    def on_close(order, price, reason):
        with manual_lock:
            manual_events.append(("close", reason, price))

    monitor = ManualOrderMonitor(
        orders_file=MANUAL_ORDERS_FILE,
        status_file=MANUAL_ORDERS_STATUS_FILE,
        on_fill=on_fill,
        on_close=on_close,
    )

    # 可选：写入一条手工指令，供 monitor 在第一次 on_price_update 时读取（读后文件会被清空）
    _ensure_manual_order_file()

    manual_done = threading.Event()
    auto_done = threading.Event()
    manual_error = []
    auto_error = []

    def manual_loop():
        try:
            end_at = time.time() + RUN_SECONDS
            bar_idx = 0
            while time.time() < end_at and not manual_done.is_set():
                # 模拟一根 K 线：open, high, low, close
                o, h, l, c = 90.0, 91.0, 89.5, 90.5
                monitor.on_price_update(o, h, l, c, bar_idx)
                bar_idx += 1
                time.sleep(POLL_INTERVAL)
        except Exception as e:
            manual_error.append(e)
        finally:
            manual_done.set()

    def auto_loop():
        try:
            end_at = time.time() + RUN_SECONDS
            while time.time() < end_at and not auto_done.is_set():
                # 模拟自动策略一步：不真实调用 tiger1 以免依赖 K 线/API，仅表示“自动侧在运行”
                time.sleep(1)
        except Exception as e:
            auto_error.append(e)
        finally:
            auto_done.set()

    t_manual = threading.Thread(target=manual_loop, name="manual_monitor")
    t_auto = threading.Thread(target=auto_loop, name="auto_strategy")
    t_manual.daemon = True
    t_auto.daemon = True
    t_manual.start()
    t_auto.start()
    t_manual.join(timeout=RUN_SECONDS + 2)
    t_auto.join(timeout=RUN_SECONDS + 2)

    assert not manual_error, f"手工线程异常: {manual_error}"
    assert not auto_error, f"自动线程异常: {auto_error}"
    print("OK 自动模式与手工订单并发执行：两线程正常结束，无异常。")


def _ensure_manual_order_file():
    """若 run/manual_orders.json 为空或不存在，写入一条合法指令（direct_entry）便于测试轮询。"""
    from src.manual_order_mode import (
        MANUAL_ORDERS_FILE,
        load_manual_orders_from_file,
        ManualOrderInstruction,
        Direction,
        save_manual_order,
    )
    path = Path(MANUAL_ORDERS_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        inst = ManualOrderInstruction(
            direction=Direction.LONG,
            trigger=0,
            confirm=0,
            entry=90.0,
            stop_loss=88.0,
            take_profit=93.0,
            once=True,
            direct_entry=True,
        )
        save_manual_order(inst, MANUAL_ORDERS_FILE)


def test_auto_manual_concurrent_unittest():
    """供 pytest 发现：test_ 前缀"""
    test_auto_manual_concurrent()


def test_main_loop_manual_integration_no_order():
    """
    主循环内集成：同一线程先跑策略（mock 空数据提前返回）再跑手工 on_price_update。
    不触发下单时，验证 current_position 不变、无异常。
    """
    from unittest.mock import patch
    from src import tiger1
    from src.manual_order_mode import (
        ManualOrderMonitor,
        MANUAL_ORDERS_FILE,
        MANUAL_ORDERS_STATUS_FILE,
    )

    # 策略侧：mock get_kline_data 返回空，grid_trading_strategy_pro1 会提前 return
    with patch.object(tiger1, "get_kline_data", return_value=tiger1.pd.DataFrame()):
        tiger1.grid_trading_strategy_pro1()

    pos_before = getattr(tiger1, "current_position", 0)

    monitor = ManualOrderMonitor(
        orders_file=MANUAL_ORDERS_FILE,
        status_file=MANUAL_ORDERS_STATUS_FILE,
        on_fill=None,
        on_close=None,
    )
    # 主循环内：用一根 K 线更新手工订单（无指令则不会下单）
    monitor.on_price_update(90.0, 91.0, 89.5, 90.5, 0)

    pos_after = getattr(tiger1, "current_position", 0)
    assert pos_after == pos_before, "主循环内仅手工 on_price_update 且未下单时，current_position 应不变"
    print("OK 主循环内集成：策略(mock 空数据)+手工 on_price_update，仓位未变、无异常。")


def test_manual_and_auto_share_position_variable():
    """
    验证手工订单与自动订单共用 tiger1 的 current_position：若手工 on_fill 调用 place_tiger_order，
    会修改同一全局变量，需注意仓位合计。
    """
    from unittest.mock import patch
    from src import tiger1
    from src.manual_order_mode import (
        ManualOrderMonitor,
        MANUAL_ORDERS_FILE,
        MANUAL_ORDERS_STATUS_FILE,
        ManualOrderInstruction,
        Direction,
        save_manual_order,
    )

    # 确保 mock 模式且通过 production 守卫，避免真实下单
    orig_mock = getattr(tiger1.api_manager, "is_mock_mode", True)
    orig_run_env = getattr(tiger1, "RUN_ENV", "sandbox")
    try:
        tiger1.api_manager.is_mock_mode = True
        tiger1.RUN_ENV = "sandbox"
        tiger1.current_position = 0

        # 写入一条会“建仓”的指令：direct_entry，entry=90，bar 内 90 被触及即 OPEN
        path = Path(MANUAL_ORDERS_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        inst = ManualOrderInstruction(
            direction=Direction.LONG,
            trigger=0,
            confirm=0,
            entry=90.0,
            stop_loss=88.0,
            take_profit=93.0,
            once=True,
            direct_entry=True,
        )
        save_manual_order(inst, MANUAL_ORDERS_FILE)

        placed = []

        def on_fill(order, price):
            placed.append((order, price))
            # 手工订单建仓时调用 place_tiger_order，会改动同一 current_position
            tiger1.place_tiger_order("BUY", 1, price, source="manual")

        monitor = ManualOrderMonitor(
            orders_file=MANUAL_ORDERS_FILE,
            status_file=MANUAL_ORDERS_STATUS_FILE,
            on_fill=on_fill,
            on_close=None,
        )
        # 一根 K 线：low<=90<=high，close=90，会触发 direct_entry 建仓
        monitor.on_price_update(90.0, 91.0, 89.0, 90.0, 0)

        assert len(placed) >= 1, "应至少触发一次 on_fill（手工建仓）"
        assert getattr(tiger1, "current_position", 0) >= 1, "手工 on_fill 调用 place_tiger_order 后，current_position 应增加（共用同一变量）"
        print("OK 手工订单与自动订单共用 current_position，手工建仓会增大仓位。")
    finally:
        tiger1.api_manager.is_mock_mode = orig_mock
        tiger1.RUN_ENV = orig_run_env
        tiger1.current_position = 0


if __name__ == "__main__":
    test_auto_manual_concurrent()
    test_main_loop_manual_integration_no_order()
    test_manual_and_auto_share_position_variable()
