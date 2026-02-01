#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
订单查询工具
用于查询Tiger API中的订单信息
"""

import sys
sys.path.insert(0, '/home/cx/tigertrade')

from tigeropen.tiger_open_config import TigerOpenClientConfig
from tigeropen.trade.trade_client import TradeClient
from src import tiger1 as t1

def query_orders():
    """查询订单"""
    print("="*70)
    print("📋 查询订单")
    print("="*70)
    
    try:
        client_config = TigerOpenClientConfig(props_path='./openapicfg_dem')
        trade_client = TradeClient(client_config)
        account = client_config.account
        
        print(f"账户: {account}")
        print(f"交易标的: {t1.FUTURE_SYMBOL}")
        
        # 转换symbol格式：SIL.COMEX.202603 -> SIL2603（Tiger API查询订单需要使用简短格式）
        symbol_to_query = t1._to_api_identifier(t1.FUTURE_SYMBOL)
        print(f"查询使用的symbol: {symbol_to_query}")
        
        # 尝试查询订单
        try:
            all_orders = trade_client.get_orders(
                account=account,
                symbol=symbol_to_query,  # 使用转换后的格式 SIL2603
                limit=50
            )
            
            if all_orders:
                print(f"\n✅ 查询到 {len(all_orders)} 条订单：\n")
                for i, order in enumerate(all_orders, 1):
                    order_id = getattr(order, 'order_id', getattr(order, 'id', None))
                    status = getattr(order, 'status', getattr(order, 'order_status', None))
                    side = getattr(order, 'side', getattr(order, 'action', None))
                    quantity = getattr(order, 'quantity', getattr(order, 'qty', None))
                    price = getattr(order, 'limit_price', getattr(order, 'price', None))
                    
                    print(f"订单 {i}:")
                    print(f"  ID: {order_id}")
                    print(f"  状态: {status}")
                    print(f"  方向: {side}")
                    print(f"  数量: {quantity}")
                    print(f"  价格: {price}")
                    print()
            else:
                print("\n⚠️ 没有查询到订单")
        except Exception as e:
            print(f"\n❌ 查询失败: {e}")
            if 'not authorized' in str(e).lower():
                print("\n提示: 账户授权问题，需要在Tiger后台配置account授权")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")

if __name__ == '__main__':
    query_orders()
