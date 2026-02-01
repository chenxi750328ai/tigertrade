#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
授权检查脚本：验证当前 account 是否已授权给当前 API 用户。
若在 Tiger 后台看不到订单，请先在 Tiger 后台完成账户授权，再运行本脚本验证。
"""

import sys
import signal
sys.path.insert(0, '/home/cx/tigertrade')

def main():
    print("=" * 60)
    print("🔐 Tiger API 账户授权检查")
    print("=" * 60)
    
    try:
        from tigeropen.tiger_open_config import TigerOpenClientConfig
        from tigeropen.trade.trade_client import TradeClient
        from src import tiger1 as t1
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return 1
    
    try:
        client_config = TigerOpenClientConfig(props_path='./openapicfg_dem')
        trade_client = TradeClient(client_config)
        account = client_config.account
        tiger_id = getattr(client_config, 'tiger_id', 'N/A')
        
        print(f"   账户 (account): {account}")
        print(f"   API 用户 (tiger_id): {tiger_id}")
        print()
        
        # 用 get_orders 测试：若未授权会直接报 not authorized
        symbol = t1._to_api_identifier(t1.FUTURE_SYMBOL) if hasattr(t1, '_to_api_identifier') else 'SIL2603'
        used_alarm = False
        if getattr(signal, 'SIGALRM', None) is not None and getattr(signal, 'alarm', None):
            def _timeout(signum, frame):
                raise TimeoutError("请求超时（网络或服务较慢）")
            signal.signal(signal.SIGALRM, _timeout)
            signal.alarm(15)
            used_alarm = True
        try:
            trade_client.get_orders(account=account, symbol=symbol, limit=5)
        finally:
            if used_alarm:
                signal.alarm(0)
        
        print("✅ 授权检查通过：当前账户已授权给当前 API 用户。")
        print("   若仍看不到订单，请确认是否在用同一账户/同一环境下单。")
        return 0
        
    except Exception as e:
        err = str(e)
        if 'not authorized' in err.lower() or 'authorized' in err.lower():
            print("❌ 授权失败：当前账户未授权给当前 API 用户。")
            print()
            print("   错误信息:", err[:200])
            print()
            print("👉 请按以下步骤在 Tiger 后台操作：")
            print("   1. 登录 Tiger 证券后台（网页）")
            print("   2. 找到「API 管理」/「开发者」→「账户授权」")
            print("   3. 将 account 授权给 API 用户（tiger_id）")
            print("   4. 保存后重新运行本脚本验证")
            print()
            print("   详细说明见: docs/后台看不到订单_必读_授权配置步骤.md")
            return 1
        print(f"❌ 其他错误: {err}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
