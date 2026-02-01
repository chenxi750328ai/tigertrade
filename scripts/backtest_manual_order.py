#!/usr/bin/env python3
"""
手工订单模式回测脚本

用法:
  python scripts/backtest_manual_order.py --long trigger confirm entry stop_loss take_profit
  python scripts/backtest_manual_order.py --short trigger confirm entry stop_loss take_profit
  python scripts/backtest_manual_order.py --json '{"direction":"long","trigger":28.5,"confirm":28.8,"entry":28.9,"stop_loss":28.3,"take_profit":29.2}'
  # 直接下单（不用 trigger/confirm）：--long-direct entry stop_loss take_profit 或 JSON 中 "direct_entry":true
"""
import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.manual_order_mode import (
    ManualOrderInstruction,
    Direction,
    run_backtest,
)


def main():
    ap = argparse.ArgumentParser(description="手工订单模式回测")
    ap.add_argument("--long", nargs=5, type=float, metavar=("trigger", "confirm", "entry", "stop_loss", "take_profit"),
                    help="做多: trigger=低点 confirm=回升点 entry=建仓 stop_loss=止损 take_profit=止盈")
    ap.add_argument("--short", nargs=5, type=float, metavar=("trigger", "confirm", "entry", "stop_loss", "take_profit"),
                    help="做空: trigger=高点 confirm=回落点 entry=建仓 stop_loss=止损 take_profit=止盈")
    ap.add_argument("--long-direct", nargs=3, type=float, metavar=("entry", "stop_loss", "take_profit"),
                    help="做多直接下单: 仅 entry/stop_loss/take_profit，不用 trigger/confirm")
    ap.add_argument("--short-direct", nargs=3, type=float, metavar=("entry", "stop_loss", "take_profit"),
                    help="做空直接下单: 仅 entry/stop_loss/take_profit")
    ap.add_argument("--json", type=str, help="JSON 格式指令")
    ap.add_argument("--data", type=str,
                    default="/home/cx/tigertrade/data/processed/test.csv",
                    help="回测数据路径")
    ap.add_argument("--price-col", type=str, default="price_current",
                    help="价格列名")
    ap.add_argument("--batch", type=str, help="批量回测: JSON 数组文件路径")
    args = ap.parse_args()

    instructions = []
    if args.long:
        trigger, confirm, entry, stop_loss, take_profit = args.long
        instructions.append(ManualOrderInstruction(Direction.LONG, trigger, confirm, entry, stop_loss, take_profit))
    elif args.short:
        trigger, confirm, entry, stop_loss, take_profit = args.short
        instructions.append(ManualOrderInstruction(Direction.SHORT, trigger, confirm, entry, stop_loss, take_profit))
    elif args.long_direct:
        entry, stop_loss, take_profit = args.long_direct
        instructions.append(ManualOrderInstruction(Direction.LONG, 0, 0, entry, stop_loss, take_profit, once=True, direct_entry=True))
    elif args.short_direct:
        entry, stop_loss, take_profit = args.short_direct
        instructions.append(ManualOrderInstruction(Direction.SHORT, 0, 0, entry, stop_loss, take_profit, once=True, direct_entry=True))
    elif args.json:
        instructions.append(ManualOrderInstruction.from_json(args.json))
    elif args.batch:
        with open(args.batch) as f:
            arr = json.load(f)
        for item in arr:
            instructions.append(ManualOrderInstruction.from_dict(item))
    else:
        ap.print_help()
        return 1

    df = pd.read_csv(args.data)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    price_col = args.price_col
    if price_col not in df.columns and "close" in df.columns:
        price_col = "close"

    print("=" * 60)
    print("📊 手工订单模式回测")
    print("=" * 60)
    print(f"数据: {args.data} ({len(df)} 条)")
    print(f"价格列: {price_col}")
    print("=" * 60)

    results = []
    for i, inst in enumerate(instructions):
        err = inst.validate()
        if err:
            print(f"\n❌ 指令 {i+1}: {err}")
            results.append({"instruction": inst.to_dict(), "error": err})
            continue

        out = run_backtest(df, inst, price_col=price_col)
        if "error" in out:
            print(f"\n❌ 指令 {i+1}: {out['error']}")
            results.append(out)
            continue

        summary = out["summary"]
        trades = out["trades"]
        tag = " [direct_entry]" if getattr(inst, "direct_entry", False) else ""
        print(f"\n📋 指令 {i+1}: {inst.direction.value}{tag} entry={inst.entry} stop_loss={inst.stop_loss} take_profit={inst.take_profit}")
        if summary.get("executed"):
            t = trades[0]
            print(f"   ✅ 已执行: 建仓 {t['entry_price']:.4f} @ idx {t['entry_idx']} → "
                  f"平仓 {t['exit_price']:.4f} @ idx {t['exit_idx']} ({t['exit_reason']})")
            print(f"   📈 盈亏: {t['pnl_pct']:.2f}%")
        else:
            print(f"   ⏳ 未触发建仓 (最终状态: {out['final_state']})")
        results.append(out)

    # 汇总
    executed = [r for r in results if isinstance(r, dict) and r.get("summary", {}).get("executed")]
    if executed:
        total_pnl = sum(t["pnl_pct"] for r in executed for t in r.get("trades", []))
        print("\n" + "=" * 60)
        print(f"📊 汇总: {len(executed)}/{len(instructions)} 笔执行, 总盈亏 {total_pnl:.2f}%")
        print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
