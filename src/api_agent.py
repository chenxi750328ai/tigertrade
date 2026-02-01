"""
API Agent - 模拟Tiger Open API的各种功能
用于提高测试覆盖率，提供完整的模拟API响应
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import time
from tigeropen.common.consts import Language, Market, BarPeriod, QuoteRight, OrderStatus, OrderType, Currency
from tigeropen.tiger_open_config import TigerOpenClientConfig
from tigeropen.quote.quote_client import QuoteClient
from tigeropen.trade.trade_client import TradeClient


class MockQuoteClient:
    """模拟行情客户端"""
    
    def __init__(self):
        self.simulated_data = {}
    
    def get_stock_briefs(self, symbols):
        """模拟获取股票行情"""
        data = []
        for symbol in symbols:
            data.append({
                'symbol': symbol,
                'price': round(random.uniform(50, 200), 2),
                'change': round(random.uniform(-5, 5), 2),
                'chg_ratio': round(random.uniform(-0.05, 0.05), 4),
                'volume': random.randint(1000000, 5000000),
                'timestamp': datetime.now()
            })
        return pd.DataFrame(data)
    
    def get_future_exchanges(self):
        """模拟获取期货交易所"""
        data = [{
            'code': 'CME',
            'name': '芝加哥商品交易所',
            'zone': 'America/Chicago',
            'timezone': 'UTC-6',
            'status': 'OPEN'
        }, {
            'code': 'COMEX',
            'name': '纽约商品交易所',
            'zone': 'America/New_York',
            'timezone': 'UTC-5',
            'status': 'OPEN'
        }]
        return pd.DataFrame(data)
    
    def get_future_contracts(self, exchange_code):
        """模拟获取期货合约"""
        data = [{
            'contract_code': f'SIL{datetime.now().year % 100 + 1:02d}{random.randint(1, 12):02d}',
            'name': '白银期货',
            'multiplier': 5000,
            'last_trading_date': datetime.now() + timedelta(days=random.randint(30, 180)),
            'exchange_code': exchange_code,
            'product_code': 'SIL',
            'status': 'ACTIVE'
        } for _ in range(5)]
        return pd.DataFrame(data)
    
    def get_all_future_contracts(self, product_code):
        """模拟获取全部期货合约"""
        data = [{
            'contract_code': f'{product_code}{datetime.now().year % 100 + 1:02d}{random.randint(1, 12):02d}',
            'name': f'{product_code}期货',
            'multiplier': 5000,
            'last_trading_date': datetime.now() + timedelta(days=random.randint(30, 180)),
            'exchange_code': 'COMEX',
            'product_code': product_code,
            'status': 'ACTIVE'
        } for _ in range(10)]
        return pd.DataFrame(data)
    
    def get_current_future_contract(self, product_code):
        """模拟获取当前期货合约"""
        return {
            'contract_code': f'{product_code}{datetime.now().year % 100 + 1:02d}{random.randint(1, 12):02d}',
            'name': f'{product_code}主力合约',
            'multiplier': 5000,
            'last_trading_date': datetime.now() + timedelta(days=90),
            'exchange_code': 'COMEX'
        }
    
    def get_quote_permission(self):
        """模拟获取行情权限"""
        return {
            'real_time': True,
            'delay': False,
            'permission': 'REAL_TIME'
        }
    
    def get_future_brief(self, symbols):
        """模拟获取期货简要信息"""
        data = []
        for symbol in symbols:
            data.append({
                'symbol': symbol,
                'price': round(random.uniform(20, 30), 2),
                'high': round(random.uniform(25, 35), 2),
                'low': round(random.uniform(15, 25), 2),
                'volume': random.randint(10000, 50000),
                'timestamp': datetime.now()
            })
        return pd.DataFrame(data)
    
    def get_future_bars(self, symbols, period, begin_time, end_time, count, right):
        """模拟获取期货K线数据"""
        data = []
        now = datetime.now()
        
        # 如果符号列表为空或计数为0，则返回空DataFrame
        if not symbols or count <= 0:
            return pd.DataFrame()
        
        for symbol in symbols:
            for i in range(count):
                time_offset = None
                # 使用字符串形式来比较BarPeriod
                if period == '1min' or period == BarPeriod.ONE_MINUTE:
                    time_offset = timedelta(minutes=i)
                elif period == '5min' or period == BarPeriod.FIVE_MINUTES:
                    time_offset = timedelta(minutes=i*5)
                elif period == '1hour' or period == BarPeriod.ONE_HOUR:
                    time_offset = timedelta(hours=i)
                elif period == '1day' or period == BarPeriod.ONE_DAY:
                    time_offset = timedelta(days=i)
                else:
                    # 如果不匹配已知的BarPeriod，则使用分钟作为默认值
                    time_offset = timedelta(minutes=i)
                
                ts = now - time_offset
                base_price = 90 + random.uniform(-5, 5)
                high = base_price + abs(random.uniform(0, 1))
                low = base_price - abs(random.uniform(0, 1))
                open_price = base_price - random.uniform(-0.5, 0.5)
                close_price = base_price + random.uniform(-0.5, 0.5)
                
                data.append({
                    'time': ts,
                    'symbol': symbol,
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close_price,
                    'volume': random.randint(100, 1000),
                    'oi': random.randint(5000, 10000)
                })
        
        if data:
            df = pd.DataFrame(data)
            df.set_index('time', inplace=True)
            return df
        else:
            return pd.DataFrame()
    
    def get_bars(self, symbols, period, begin_time, end_time, count, right):
        """模拟获取K线数据（股票）"""
        data = []
        now = datetime.now()
        
        # 如果符号列表为空或计数为0，则返回空DataFrame
        if not symbols or count <= 0:
            return pd.DataFrame()
        
        for symbol in symbols:
            for i in range(count):
                time_offset = timedelta(minutes=i) if (period == '1min' or period == BarPeriod.ONE_MINUTE) else timedelta(hours=i)
                ts = now - time_offset
                base_price = 90 + random.uniform(-5, 5)
                high = base_price + abs(random.uniform(0, 1))
                low = base_price - abs(random.uniform(0, 1))
                open_price = base_price - random.uniform(-0.5, 0.5)
                close_price = base_price + random.uniform(-0.5, 0.5)
                
                data.append({
                    'time': ts,
                    'symbol': symbol,
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close_price,
                    'volume': random.randint(100, 1000)
                })
        
        if data:
            df = pd.DataFrame(data)
            df.set_index('time', inplace=True)
            return df
        else:
            return pd.DataFrame()


class MockTradeClient:
    """模拟交易客户端"""
    
    def __init__(self):
        self.orders = {}
        self.positions = {}
        self.account_info = {
            'account_id': 'TEST_ACCOUNT_123456',
            'net_liquidation': 100000,
            'total_cash_value': 100000,
            'accrued_cash': 0,
            'available_funds': 100000,
            'excess_liquidity': 100000,
            'maint_margin_req': 0,
            'initial_margin_req': 0,
            'position_value': 0,
            'realized_pnl': 0,
            'unrealized_pnl': 0
        }
    
    def place_order(self, symbol, side, order_type, quantity, time_in_force='DAY', limit_price=None, stop_price=None):
        """模拟下单"""
        order_id = f"ORDER_{int(time.time())}_{random.randint(100000, 999999)}"
        
        # 创建一个模拟的Order对象
        class MockOrder:
            def __init__(self, order_id, symbol, side, order_type, quantity, time_in_force):
                self.order_id = order_id
                self.symbol = symbol
                self.side = side
                self.order_type = order_type
                self.quantity = quantity
                self.time_in_force = time_in_force
                self.limit_price = limit_price
                self.stop_price = stop_price
                self.order_status = 'HELD'  # 使用字符串而不是枚举
        
        order = MockOrder(order_id, symbol, side, order_type, quantity, time_in_force)
        
        self.orders[order_id] = order
        return order
    
    def cancel_order(self, order_id):
        """模拟取消订单"""
        if order_id in self.orders:
            self.orders[order_id].order_status = 'CANCELLED'
            return True
        return False
    
    def modify_order(self, order_id, quantity=None, limit_price=None, stop_price=None):
        """模拟修改订单"""
        if order_id in self.orders:
            order = self.orders[order_id]
            if quantity is not None:
                order.quantity = quantity
            if limit_price is not None:
                order.limit_price = limit_price
            if stop_price is not None:
                order.stop_price = stop_price
            return order
        return None
    
    def get_orders(self, order_status=None, sec_type=None):
        """模拟获取订单列表"""
        orders = list(self.orders.values())
        if order_status:
            orders = [o for o in orders if o.order_status == order_status]
        return orders
    
    def get_positions(self):
        """模拟获取持仓"""
        return list(self.positions.values())
    
    def get_account_info(self):
        """模拟获取账户信息"""
        return self.account_info


class APIAgent:
    """API代理 - 统一的API模拟接口"""
    
    def __init__(self, use_mock=True):
        self.use_mock = use_mock
        
        if use_mock:
            self.quote_client = MockQuoteClient()
            self.trade_client = MockTradeClient()
        else:
            # 真实API客户端
            # 这里需要真实的配置，但为了测试目的，我们总是使用模拟
            self.quote_client = MockQuoteClient()
            self.trade_client = MockTradeClient()
    
    def get_kline_data(self, symbols, period, count=100, start_time=None, end_time=None):
        """获取K线数据的统一接口"""
        try:
            if self.use_mock:
                return self.quote_client.get_future_bars(
                    symbols, 
                    period, 
                    start_time, 
                    end_time, 
                    count, 
                    None
                )
            else:
                return self.quote_client.get_future_bars(
                    symbols, 
                    period, 
                    start_time, 
                    end_time, 
                    count, 
                    None
                )
        except Exception as e:
            print(f"获取K线数据失败: {e}")
            # 返回一个默认的DataFrame
            return pd.DataFrame({
                'open': [90.0] * count,
                'high': [90.5] * count,
                'low': [89.5] * count,
                'close': [90.1] * count,
                'volume': [100] * count
            }, index=pd.date_range(start=datetime.now(), periods=count, freq='1min'))
    
    def get_account_info(self):
        """获取账户信息"""
        return self.trade_client.get_account_info()
    
    def place_order(self, symbol, side, order_type, quantity, time_in_force='DAY', limit_price=None, stop_price=None):
        """下单"""
        return self.trade_client.place_order(
            symbol, side, order_type, quantity, time_in_force, limit_price, stop_price
        )


# 创建全局API代理实例
api_agent = APIAgent(use_mock=True)