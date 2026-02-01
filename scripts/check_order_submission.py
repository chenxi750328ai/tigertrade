#!/usr/bin/env python3
"""
检查订单提交问题的诊断脚本
"""
import sys
import os
sys.path.insert(0, '/home/cx/tigertrade')

from src.api_adapter import api_manager
from src import tiger1 as t1

print("="*70)
print("🔍 订单提交问题诊断")
print("="*70)
print()

# 1. 检查API管理器状态
print("1. API管理器状态:")
print(f"   is_mock_mode: {api_manager.is_mock_mode}")
print(f"   trade_api: {api_manager.trade_api}")
print(f"   trade_api类型: {type(api_manager.trade_api).__name__ if api_manager.trade_api else 'None'}")
print(f"   quote_api: {api_manager.quote_api}")
print(f"   _account: {getattr(api_manager, '_account', 'N/A')}")
print()

# 2. 检查tiger1的客户端状态
print("2. tiger1客户端状态:")
print(f"   trade_client: {t1.trade_client}")
print(f"   quote_client: {t1.quote_client}")
if t1.trade_client and hasattr(t1.trade_client, 'config'):
    print(f"   trade_client.config.account: {getattr(t1.trade_client.config, 'account', 'N/A')}")
if t1.quote_client and hasattr(t1.quote_client, 'config'):
    print(f"   quote_client.config.account: {getattr(t1.quote_client.config, 'account', 'N/A')}")
print()

# 3. 检查trade_api的account
if api_manager.trade_api:
    print("3. trade_api详细信息:")
    print(f"   account属性: {getattr(api_manager.trade_api, 'account', 'N/A')}")
    print(f"   client属性: {getattr(api_manager.trade_api, 'client', 'N/A')}")
    if hasattr(api_manager.trade_api, 'client') and api_manager.trade_api.client:
        client = api_manager.trade_api.client
        print(f"   client类型: {type(client).__name__}")
        if hasattr(client, 'config'):
            print(f"   client.config.account: {getattr(client.config, 'account', 'N/A')}")
print()

# 4. 尝试重新初始化（如果需要）
if api_manager.trade_api is None and t1.trade_client is not None:
    print("4. 尝试重新初始化API...")
    try:
        account = None
        if hasattr(t1.trade_client, 'config'):
            account = getattr(t1.trade_client.config, 'account', None)
        api_manager.initialize_real_apis(t1.quote_client, t1.trade_client, account=account)
        print(f"   ✅ 重新初始化成功")
        print(f"   trade_api: {api_manager.trade_api}")
        print(f"   account: {getattr(api_manager.trade_api, 'account', 'N/A') if api_manager.trade_api else 'N/A'}")
    except Exception as e:
        print(f"   ❌ 重新初始化失败: {e}")
        import traceback
        traceback.print_exc()
print()

# 5. 测试订单提交（如果API已初始化）
if api_manager.trade_api:
    print("5. 测试订单提交（模拟）...")
    try:
        # 检查account
        account = getattr(api_manager.trade_api, 'account', None)
        if not account:
            print("   ⚠️ account为空，订单提交可能失败")
        else:
            print(f"   ✅ account已设置: {account}")
            print("   （实际下单测试需要真实交易时段）")
    except Exception as e:
        print(f"   ❌ 测试失败: {e}")
else:
    print("5. ⚠️ trade_api未初始化，无法测试订单提交")
print()

print("="*70)
print("诊断完成")
print("="*70)
