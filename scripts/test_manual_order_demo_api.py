#!/usr/bin/env python3
"""
手工订单 DEMO 账户 API 模式测试：下一笔手工单（source=manual），
失败时写入 run/api_failure_for_support.jsonl，并输出失败订单详细参数供发给客服。
需在 openapicfg_dem 所在目录运行，或设置 TIGER_PROPS_PATH；必须带参数 d 使用 DEMO 配置。
"""
import sys
import os
from pathlib import Path

# 确保 tigertrade 在路径中
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

# 若 openapicfg_dem 在上级目录，切到上级以便配置生效
props_path = os.getenv("TIGER_PROPS_PATH", "")
if not props_path:
    cand = root / "openapicfg_dem"
    if not cand.exists():
        cand = root.parent / "openapicfg_dem"
    if cand.exists():
        os.chdir(cand.parent)
        print(f"📁 工作目录: {os.getcwd()} (openapicfg_dem 在此)")

# 必须带 'd' 以使用 DEMO 配置
if len(sys.argv) < 2 or sys.argv[1] != "d":
    sys.argv.insert(1, "d")

# 允许真实交易（sandbox 下会走真实 API）
os.environ["ALLOW_REAL_TRADING"] = "1"

import src.tiger1 as t1


def main():
    print("=" * 60)
    print("🔌 手工订单 DEMO 账户 API 模式测试")
    print("=" * 60)
    print("  将下一笔手工单: BUY 1手 @ 91.63, 止损 90, source=manual")
    print("  若 API 报错，失败详情会写入 run/api_failure_for_support.jsonl")
    print("=" * 60)

    ok = t1.place_tiger_order(
        "BUY",
        1,
        91.63,
        stop_loss_price=90.0,
        take_profit_price=None,
        reason="manual_demo_test",
        source="manual",
    )

    print("=" * 60)
    if ok:
        print("✅ 手工单提交成功，请查看 run/order_log.jsonl")
    else:
        print("❌ 手工单提交失败（预期可能为 1200 等），下面输出失败详情供发给客服：")
        print("=" * 60)
        # 输出客服用失败详情
        from src.order_log import API_FAILURE_FOR_SUPPORT_FILE, ORDER_LOG_FILE
        import json
        records = []
        p = Path(API_FAILURE_FOR_SUPPORT_FILE)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        if not records and Path(ORDER_LOG_FILE).exists():
            with open(ORDER_LOG_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                        if r.get("status") == "fail" and r.get("mode") == "real":
                            records.append({
                                "ts": r.get("ts", ""),
                                "source": r.get("source", "manual"),
                                "side": r.get("side", ""),
                                "quantity": r.get("qty"),
                                "price": r.get("price"),
                                "symbol_submitted": r.get("symbol", ""),
                                "order_type_api": "LMT" if r.get("price") else "MKT",
                                "time_in_force": "DAY",
                                "limit_price": r.get("price"),
                                "stop_price": None,
                                "order_id": r.get("order_id", ""),
                                "error": r.get("error", ""),
                            })
                    except json.JSONDecodeError:
                        continue
        if records:
            r = records[-1]
            print("")
            print("--- 可复制给老虎客服的失败订单参数 ---")
            print(f"  时间         : {r.get('ts', '')}")
            print(f"  来源         : {r.get('source', '')} （manual=手工订单）")
            print(f"  方向         : {r.get('side', '')}")
            print(f"  数量         : {r.get('quantity', '')} 手")
            print(f"  价格         : {r.get('price')}")
            print(f"  提交合约     : {r.get('symbol_submitted', '')}")
            print(f"  订单类型(API): {r.get('order_type_api', '')} （LMT=限价, MKT=市价）")
            print(f"  有效期限     : {r.get('time_in_force', '')}")
            print(f"  限价         : {r.get('limit_price')}")
            print(f"  止损价       : {r.get('stop_price')}")
            print(f"  订单ID       : {r.get('order_id', '')}")
            print(f"  错误信息     : {r.get('error', '')}")
            print("--- 结束 ---")
        else:
            print("  （暂无失败记录，可稍后运行: python scripts/print_api_failure_for_support.py）")
    print("=" * 60)
    return 0 if ok else 0  # 始终返回 0，方便管道后续命令


if __name__ == "__main__":
    sys.exit(main())
