#!/usr/bin/env python3
"""
诊断 1010 account 为空 的根因。
复现 source=auto 路径（OrderExecutor.execute_buy），在 place_order 前打印 api_manager/trade_api 状态。
运行: python scripts/diagnose_account_1010.py
"""
import sys
import os
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
os.chdir(root)

# 模拟 MOE 启动：argv 含 d moe
sys.argv = ['tiger1.py', 'd', 'moe_transformer']

# 根因验证：模拟 cwd 非项目根（如 cron/subprocess 场景）
import os
_saved_cwd = os.getcwd()
os.chdir('/tmp')  # 故意切到非项目根

print("=" * 60)
print("诊断 account 1010 根因（cwd=/tmp 模拟 subprocess 场景）")
print("=" * 60)

# 1. 加载 tiger1（会执行其顶层初始化）
print("\n[1] 加载 tiger1 (argv=%s)..." % sys.argv)
import src.tiger1 as t1

print("  client_config: %s" % (t1.client_config is not None))
print("  trade_client: %s" % (t1.trade_client is not None))
if t1.client_config:
    print("  client_config.account: %s" % getattr(t1.client_config, 'account', None))

# 2. 检查 api_manager
from src.api_adapter import api_manager
print("\n[2] api_manager 状态:")
print("  trade_api: %s" % (api_manager.trade_api is not None))
print("  _account: %s" % getattr(api_manager, '_account', None))
if api_manager.trade_api:
    print("  trade_api.account: %s" % getattr(api_manager.trade_api, 'account', None))

# 3. 模拟 OrderExecutor.execute_buy 的调用路径
print("\n[3] 模拟 OrderExecutor.execute_buy 路径...")
from src.executor.trading_executor import TradingExecutor
from src.executor import MarketDataProvider, OrderExecutor
from src.strategies.strategy_factory import StrategyFactory

# 创建策略与执行器（与 tiger1 一致）
strategy = StrategyFactory.create('moe_transformer')
market_data = MarketDataProvider()
_main = sys.modules.get('__main__')
_risk = t1 if hasattr(t1, 'check_risk_control') else None
order_executor = OrderExecutor(_risk, state_fallback=_main)

print("  OrderExecutor 创建后 trade_api: %s" % (api_manager.trade_api is not None))
print("  trade_api.account: %s" % (getattr(api_manager.trade_api, 'account', None) if api_manager.trade_api else 'N/A'))

# 4. 在 place_order 前打点（不实际下单）
print("\n[4] 若此时 account 为空，place_order 会报 1010")
acc = getattr(api_manager.trade_api, 'account', None) if api_manager.trade_api else None
if not acc:
    print("  ❌ 根因: api_manager.trade_api.account 为空")
    print("  可能原因: initialize_real_apis 时 account 未正确传入或 config 未加载")
else:
    print("  ✅ account 已设置: %s" % (str(acc)[:12] + "..."))

# 5. 检查 tiger1 的初始化条件
print("\n[5] tiger1 初始化条件回溯:")
print("  len(sys.argv)>1: %s" % (len(sys.argv) > 1))
print("  argv[1] in (d,c): %s" % (sys.argv[1] in ('d', 'c') if len(sys.argv) > 1 else False))
print("  client_config 加载: %s" % ("成功" if t1.client_config else "失败/未执行"))
if t1.client_config:
    print("  config.account: %s" % getattr(t1.client_config, 'account', None))
print("=" * 60)

# 恢复 cwd
os.chdir(_saved_cwd)
