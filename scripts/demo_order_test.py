#!/usr/bin/env python3
"""
用真实 DEMO 账户下一笔测试单，并写入订单 LOG。
需在 openapicfg_dem 所在目录运行，或设置 TIGER_PROPS_PATH。
"""
import sys
import os
from pathlib import Path

# 确保 tigertrade 在路径中
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 若 openapicfg_dem 在上级目录，切到上级以便 ./openapicfg_dem 生效
props_path = os.getenv("TIGER_PROPS_PATH", "")
if not props_path:
    cand = Path(__file__).resolve().parents[1] / "openapicfg_dem"
    if not cand.exists():
        cand = Path(__file__).resolve().parents[2] / "openapicfg_dem"
    if cand.exists():
        os.chdir(cand.parent)
        print(f"📁 工作目录: {os.getcwd()} (openapicfg_dem 在此)")

# 必须带 'd' 以使用 DEMO 配置
if len(sys.argv) < 2 or sys.argv[1] != "d":
    sys.argv.insert(1, "d")

# 导入并执行：会初始化真实 API 并调用 verify_api_connection（内含 place_tiger_order）
import src.tiger1 as t1

def main():
    print("=" * 60)
    print("🔌 真实 DEMO 账户 - 连接并下一笔测试单")
    print("=" * 60)
    ok = t1.verify_api_connection()
    print("=" * 60)
    if ok:
        print("✅ 连接成功，已下一笔测试单，请查看 run/order_log.jsonl")
    else:
        print("❌ 连接或下单失败")
    print("=" * 60)
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
