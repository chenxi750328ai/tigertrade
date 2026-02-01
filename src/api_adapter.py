"""
API适配器 - 为tiger1.py提供可注入的API接口
允许在测试时使用模拟实现，在生产时使用真实API
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
import time
import sys

from tigeropen.common.consts import BarPeriod, OrderType


class QuoteApiInterface(ABC):
    """行情API接口抽象"""
    
    @abstractmethod
    def get_stock_briefs(self, symbols):
        """获取股票简要信息"""
        pass
    
    @abstractmethod
    def get_future_exchanges(self):
        """获取期货交易所"""
        pass
    
    @abstractmethod
    def get_future_contracts(self, exchange_code):
        """获取期货合约"""
        pass
    
    @abstractmethod
    def get_all_future_contracts(self, product_code):
        """获取所有期货合约"""
        pass
    
    @abstractmethod
    def get_current_future_contract(self, product_code):
        """获取当前期货合约"""
        pass
    
    @abstractmethod
    def get_quote_permission(self):
        """获取行情权限"""
        pass
    
    @abstractmethod
    def get_future_brief(self, symbols):
        """获取期货简要信息"""
        pass
    
    @abstractmethod
    def get_future_bars(self, symbols, period, begin_time, end_time, count, right):
        """获取期货K线数据"""
        pass


class TradeApiInterface(ABC):
    """交易API接口抽象"""
    
    @abstractmethod
    def place_order(self, symbol, side, order_type, quantity, time_in_force, limit_price=None, stop_price=None):
        """下单"""
        pass


class RealQuoteApiAdapter(QuoteApiInterface):
    """真实行情API适配器"""
    
    def __init__(self, client):
        self.client = client
    
    def get_stock_briefs(self, symbols):
        return self.client.get_stock_briefs(symbols)
    
    def get_future_exchanges(self):
        return self.client.get_future_exchanges()
    
    def get_future_contracts(self, exchange_code):
        return self.client.get_future_contracts(exchange_code)
    
    def get_all_future_contracts(self, product_code):
        return self.client.get_all_future_contracts(product_code)
    
    def get_current_future_contract(self, product_code):
        return self.client.get_current_future_contract(product_code)
    
    def get_quote_permission(self):
        return self.client.get_quote_permission()
    
    def get_future_brief(self, symbols):
        return self.client.get_future_brief(symbols)
    
    def get_future_bars(self, symbols, period, begin_time, end_time, count, right):
        return self.client.get_future_bars(symbols, period, begin_time, end_time, count, right)


class RealTradeApiAdapter(TradeApiInterface):
    """真实交易API适配器"""
    
    def __init__(self, client, account=None):
        self.client = client
        # 优先使用传入的account，否则从client获取
        if account:
            self.account = account
        else:
            self.account = getattr(client, 'account', None)
            if self.account is None and hasattr(client, 'config'):
                self.account = getattr(client.config, 'account', None)
    
    def place_order(self, symbol, side, order_type, quantity, time_in_force, limit_price=None, stop_price=None):
        """下单 - 创建Order对象并调用TradeClient.place_order"""
        try:
            from tigeropen.trade.domain.order import Order
            from tigeropen.common.util.contract_utils import future_contract, stock_contract
            from tigeropen.common.consts import Currency
            
            # 获取账户信息（从保存的account或client中获取）
            account = self.account
            if account is None:
                account = getattr(self.client, 'account', None)
            if account is None:
                # 尝试从config获取
                if hasattr(self.client, 'config'):
                    account = getattr(self.client.config, 'account', None)
            
            # 如果account仍然为空，尝试从api_manager获取（如果可用）
            if not account:
                try:
                    from src.api_adapter import api_manager
                    if hasattr(api_manager, '_account') and api_manager._account:
                        account = api_manager._account
                        # 同时更新self.account以便下次使用
                        self.account = account
                        print(f"✅ 从api_manager获取account成功: {account}")
                    elif hasattr(api_manager, 'trade_api') and hasattr(api_manager.trade_api, 'account'):
                        account = api_manager.trade_api.account
                        self.account = account
                        print(f"✅ 从api_manager.trade_api获取account成功: {account}")
                except Exception as e:
                    print(f"⚠️ 从api_manager获取account失败: {e}")
            
            # 确保account不为空
            if not account:
                error_msg = f"account不能为空，无法创建订单。self.account={self.account}, client.account={getattr(self.client, 'account', None)}, client.config.account={getattr(self.client.config, 'account', None) if hasattr(self.client, 'config') else 'N/A'}"
                print(f"❌ {error_msg}")
                raise ValueError(error_msg)
            
            # 转换symbol格式：SIL.COMEX.202603 -> SIL2603（根据API文档，期货使用简短格式）
            symbol_to_use = symbol
            try:
                # 尝试使用tiger1的转换函数
                import sys
                if 'tiger1' in sys.modules:
                    t1_module = sys.modules['tiger1']
                    if hasattr(t1_module, '_to_api_identifier'):
                        symbol_to_use = t1_module._to_api_identifier(symbol)
                        print(f"🔍 [下单调试] 使用tiger1._to_api_identifier转换: {symbol} -> {symbol_to_use}")
                
                # 如果没有转换函数，手动转换
                if symbol_to_use == symbol and '.' in symbol_to_use:
                    # 简单转换：SIL.COMEX.202603 -> SIL2603
                    parts = symbol_to_use.split('.')
                    if len(parts) >= 3:
                        base = parts[0]  # SIL
                        datepart = parts[-1]  # 202603
                        if len(datepart) == 6 and datepart.isdigit():
                            symbol_to_use = f"{base}{datepart[-4:]}"  # SIL2603
                            print(f"🔍 [下单调试] 手动转换symbol: {symbol} -> {symbol_to_use}")
            except Exception as e:
                print(f"⚠️ symbol转换失败，使用原格式: {e}")
                import traceback
                traceback.print_exc()
            
            tiger_id_used = getattr(getattr(self.client, 'config', None), 'tiger_id', None) or 'N/A'
            print(f"🔍 [下单调试] 当前请求 account={account}, tiger_id={tiger_id_used}（来自配置文件），请与 Tiger 后台「API 账户授权」中一致")
            print(f"🔍 [下单调试] symbol={symbol} -> {symbol_to_use}, side={side}, order_type={order_type}, quantity={quantity}, limit_price={limit_price}")
            
            # 创建Contract对象：与原始根目录 tiger1.py 一致，仅 symbol+currency（原始可下单成功）
            # 原始：contract = future_contract(symbol=contract_symbol, currency=FUTURE_CURRENCY)，无 expiry/exchange
            contract = None
            try:
                contract = future_contract(symbol=symbol_to_use, currency=Currency.USD)
                print(f"✅ 使用future_contract创建合约成功: {symbol_to_use}（与原始 tiger1 一致：仅 symbol+currency）")
            except (TypeError, ValueError, Exception) as e:
                print(f"⚠️ future_contract创建失败: {e}，尝试stock_contract")
                # 如果失败，尝试股票合约（兼容旧代码）
                try:
                    contract = stock_contract(symbol_to_use, Currency.USD)
                    print(f"⚠️ 使用stock_contract创建合约（可能不正确）: {symbol_to_use}")
                except (TypeError, ValueError):
                    contract = stock_contract(symbol_to_use)
            
            if contract is None:
                raise ValueError(f"无法创建合约对象，symbol={symbol_to_use}")
            
            # 创建Order对象 - 根据API文档，应该使用limit_order或market_order函数
            print(f"🔍 [下单调试] 准备创建Order: account={account}, symbol={symbol} -> {symbol_to_use}, side={side}, order_type={order_type}, quantity={quantity}, limit_price={limit_price}")
            
            # 根据API文档，应该使用order_utils中的函数创建订单
            try:
                from tigeropen.common.util.order_utils import limit_order, market_order
                
                if order_type == 'LMT' or order_type == OrderType.LMT:
                    # 限价单：使用limit_order函数
                    order = limit_order(
                        account=account,
                        contract=contract,
                        action=side,  # BUY or SELL
                        limit_price=limit_price,
                        quantity=quantity
                    )
                    print(f"✅ 使用limit_order创建限价单成功")
                else:
                    # 市价单：使用market_order函数
                    order = market_order(
                        account=account,
                        contract=contract,
                        action=side,  # BUY or SELL
                        quantity=quantity
                    )
                    print(f"✅ 使用market_order创建市价单成功")
            except ImportError:
                # 如果order_utils不可用，fallback到直接创建Order对象
                print(f"⚠️ order_utils不可用，使用Order直接创建")
                order = Order(
                    account=account,
                    contract=contract,
                    action=side,  # BUY or SELL
                    order_type=order_type,  # LMT or MKT
                    quantity=quantity,
                    time_in_force=time_in_force,  # DAY or GTC
                    limit_price=limit_price,
                    aux_price=stop_price
                )
            print(f"🔍 [下单调试] Order创建成功: order.account={order.account}, order.contract={order.contract}")
            
            # 调用TradeClient.place_order，它接受Order对象
            print(f"🔍 [下单调试] 调用client.place_order，Order详情:")
            print(f"   Order.account: {order.account}")
            print(f"   Order.contract.symbol: {order.contract.symbol if hasattr(order, 'contract') and order.contract else 'N/A'}")
            print(f"   Order.action: {order.action}")
            print(f"   Order.order_type: {order.order_type}")
            print(f"   Order.quantity: {order.quantity}")
            print(f"   Order.limit_price: {order.limit_price}")
            
            result = self.client.place_order(order)
            print(f"🔍 [下单调试] place_order调用成功: result={result}")
            print(f"🔍 [下单调试] result类型: {type(result)}")
            if hasattr(result, '__dict__'):
                print(f"🔍 [下单调试] result属性: {result.__dict__}")
            return result
        except (ImportError, AttributeError, TypeError, ValueError) as e:
            # TradeClient.place_order 只接受 (order, lang=None)，无多参数 fallback
            print(f"⚠️ [下单调试] Order创建失败: {e}")
            raise Exception(f"下单失败: 无法创建Order对象或调用API - Order创建错误: {e}")
        except Exception as e:
            # 捕获所有其他异常（包括API返回的错误）
            error_msg = str(e)
            # 明确识别授权错误
            if 'not authorized' in error_msg.lower() or 'authorized' in error_msg.lower() or 'authorization' in error_msg.lower():
                auth_error = f"授权失败: {error_msg}。需要在Tiger后台配置account授权给API用户。"
                print(f"❌ [下单调试] {auth_error}")
                raise ValueError(auth_error)
            print(f"❌ [下单调试] 下单异常: {e}")
            raise


class MockQuoteApiAdapter(QuoteApiInterface):
    """模拟行情API适配器"""
    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        # 添加一个计数器来控制返回不同值
        self.call_count = 0
        self.get_future_bars_call_count = 0
        self.get_future_bars_by_page_call_count = 0

    def get_bars(self, symbol, period, count, start_time=None, end_time=None):
        """获取K线数据"""
        self.call_count += 1
        
        # 根据调用次数返回不同的数据以覆盖不同分支
        if self.call_count % 5 == 0:  # 每第5次调用返回空数据
            return pd.DataFrame()
        elif self.call_count % 5 == 1:  # 每第1次调用返回正常数据
            return pd.DataFrame({
                'time': pd.date_range('2026-01-16 12:00', periods=count, freq='1min'),
                'open': [90.0 + i*0.01 + np.random.normal(0, 0.05) for i in range(count)],
                'high': [90.1 + i*0.01 + np.random.normal(0, 0.05) for i in range(count)],
                'low': [89.9 + i*0.01 + np.random.normal(0, 0.05) for i in range(count)],
                'close': [90.0 + i*0.01 + np.random.normal(0, 0.05) for i in range(count)],
                'volume': [100 + np.random.randint(0, 50) for _ in range(count)]
            })
        elif self.call_count % 5 == 2:  # 每第2次调用返回包含NaN的数据
            df = pd.DataFrame({
                'time': pd.date_range('2026-01-16 12:00', periods=count, freq='1min'),
                'open': [np.nan if i == count//2 else 90.0 + i*0.01 for i in range(count)],
                'high': [np.nan if i == count//2 else 90.1 + i*0.01 for i in range(count)],
                'low': [np.nan if i == count//2 else 89.9 + i*0.01 for i in range(count)],
                'close': [np.nan if i == count//2 else 90.0 + i*0.01 for i in range(count)],
                'volume': [np.nan if i == count//2 else 100 + i for i in range(count)]
            })
            return df
        elif self.call_count % 5 == 3:  # 每第3次调用返回极值数据
            df = pd.DataFrame({
                'time': pd.date_range('2026-01-16 12:00', periods=count, freq='1min'),
                'open': [float('inf') if i == count//2 else 90.0 for i in range(count)],
                'high': [float('inf') if i == count//2 else 91.0 for i in range(count)],
                'low': [float('-inf') if i == count//2 else 89.0 for i in range(count)],
                'close': [float('inf') if i == count//2 else 90.5 for i in range(count)],
                'volume': [float('inf') if i == count//2 else 100 for i in range(count)]
            })
            return df
        else:  # 每第4次调用返回时间戳数据
            df = pd.DataFrame({
                'time': [(datetime.now() - timedelta(minutes=count-i)).timestamp()*1000 for i in range(count)],
                'open': [90.0 + i*0.01 for i in range(count)],
                'high': [90.1 + i*0.01 for i in range(count)],
                'low': [89.9 + i*0.01 for i in range(count)],
                'close': [90.0 + i*0.01 for i in range(count)],
                'volume': [100 + i for i in range(count)]
            })
            return df

    def get_future_bars(self, symbols, period, begin_time, end_time, count, next_page_token=None):
        """获取期货K线数据"""
        self.get_future_bars_call_count += 1
        
        # 根据调用次数返回不同的数据
        if self.get_future_bars_call_count % 7 == 0:  # 每第7次调用返回None
            return None
        elif self.get_future_bars_call_count % 7 == 1:  # 每第1次调用返回正常数据
            return pd.DataFrame({
                'time': pd.date_range('2026-01-16 12:00', periods=count, freq='1min'),
                'open': [90.0 + i*0.01 for i in range(count)],
                'high': [90.1 + i*0.01 for i in range(count)],
                'low': [89.9 + i*0.01 for i in range(count)],
                'close': [90.0 + i*0.01 for i in range(count)],
                'volume': [100 + i for i in range(count)]
            })
        elif self.get_future_bars_call_count % 7 == 2:  # 每第2次调用返回包含字符串时间的数据
            return pd.DataFrame({
                'time': [(datetime.now() - timedelta(minutes=count-i)).strftime('%Y-%m-%d %H:%M:%S') for i in range(count)],
                'open': [90.0 + i*0.01 for i in range(count)],
                'high': [90.1 + i*0.01 for i in range(count)],
                'low': [89.9 + i*0.01 for i in range(count)],
                'close': [90.0 + i*0.01 for i in range(count)],
                'volume': [100 + i for i in range(count)]
            })
        elif self.get_future_bars_call_count % 7 == 3:  # 每第3次调用返回数值时间戳
            return pd.DataFrame({
                'time': [(datetime.now() - timedelta(minutes=count-i)).timestamp() for i in range(count)],
                'open': [90.0 + i*0.01 for i in range(count)],
                'high': [90.1 + i*0.01 for i in range(count)],
                'low': [89.9 + i*0.01 for i in range(count)],
                'close': [90.0 + i*0.01 for i in range(count)],
                'volume': [100 + i for i in range(count)]
            })
        elif self.get_future_bars_call_count % 7 == 4:  # 每第4次调用返回只有少量数据
            limited_count = min(2, count)
            return pd.DataFrame({
                'time': pd.date_range('2026-01-16 12:00', periods=limited_count, freq='1min'),
                'open': [90.0 + i*0.01 for i in range(limited_count)],
                'high': [90.1 + i*0.01 for i in range(limited_count)],
                'low': [89.9 + i*0.01 for i in range(limited_count)],
                'close': [90.0 + i*0.01 for i in range(limited_count)],
                'volume': [100 + i for i in range(limited_count)]
            })
        elif self.get_future_bars_call_count % 7 == 5:  # 每第5次调用返回缺少列的数据
            return pd.DataFrame({
                'time': pd.date_range('2026-01-16 12:00', periods=count, freq='1min'),
                'open': [90.0 + i*0.01 for i in range(count)],
                'high': [90.1 + i*0.01 for i in range(count)],
                # 缺少 'low', 'close', 'volume' 列
            })
        else:  # 每第6次调用返回完全正常的数据但包含异常值
            df = pd.DataFrame({
                'time': pd.date_range('2026-01-16 12:00', periods=count, freq='1min'),
                'open': [90.0 + i*0.01 for i in range(count)],
                'high': [90.1 + i*0.01 for i in range(count)],
                'low': [89.9 + i*0.01 for i in range(count)],
                'close': [90.0 + i*0.01 for i in range(count)],
                'volume': [100 + i for i in range(count)]
            })
            # 添加一个next_page_token列来触发分页相关代码
            if count > 5:  # 只有在数据量大时才添加
                df['next_page_token'] = [None for _ in range(len(df)-1)] + ['some_token']
            return df

    def get_future_bars_by_page(self, identifier, period, begin_time, end_time, size, page_size, time_field, page_token=None):
        """分页获取期货K线数据"""
        self.get_future_bars_by_page_call_count += 1
        
        # 根据调用次数返回不同的响应类型来触发分页逻辑
        if self.get_future_bars_by_page_call_count % 5 == 0:
            # 返回DataFrame和next_token
            df = pd.DataFrame({
                'time': pd.date_range('2026-01-16 12:00', periods=min(size, 5), freq='1min'),  # 限制大小
                'open': [90.0 + i*0.01 for i in range(min(size, 5))],
                'high': [90.1 + i*0.01 for i in range(min(size, 5))],
                'low': [89.9 + i*0.01 for i in range(min(size, 5))],
                'close': [90.0 + i*0.01 for i in range(min(size, 5))],
                'volume': [100 + i for i in range(min(size, 5))],
                'next_page_token': ['token123' if i < min(size, 5)-1 else None for i in range(min(size, 5))]
            })
            return df, 'next_token' if size > 5 else None
        elif self.get_future_bars_by_page_call_count % 5 == 1:
            # 返回只有DataFrame
            df = pd.DataFrame({
                'time': pd.date_range('2026-01-16 12:00', periods=min(size, 3), freq='1min'),  # 更小的数据集
                'open': [90.0 + i*0.01 for i in range(min(size, 3))],
                'high': [90.1 + i*0.01 for i in range(min(size, 3))],
                'low': [89.9 + i*0.01 for i in range(min(size, 3))],
                'close': [90.0 + i*0.01 for i in range(min(size, 3))],
                'volume': [100 + i for i in range(min(size, 3))]
            })
            return df
        elif self.get_future_bars_by_page_call_count % 5 == 2:
            # 返回字典格式
            return {
                'df': pd.DataFrame({
                    'time': pd.date_range('2026-01-16 12:00', periods=min(size, 4), freq='1min'),
                    'open': [90.0 + i*0.01 for i in range(min(size, 4))],
                    'high': [90.1 + i*0.01 for i in range(min(size, 4))],
                    'low': [89.9 + i*0.01 for i in range(min(size, 4))],
                    'close': [90.0 + i*0.01 for i in range(min(size, 4))],
                    'volume': [100 + i for i in range(min(size, 4))]
                }),
                'next_page_token': 'dict_token' if size > 4 else None
            }
        elif self.get_future_bars_by_page_call_count % 5 == 3:
            # 返回一个包含bar对象的列表
            class MockBar:
                def __init__(self, time, open, high, low, close, volume):
                    self.time = time
                    self.open = open
                    self.high = high
                    self.low = low
                    self.close = close
                    self.volume = volume
                    
            bars = [
                MockBar(
                    pd.Timestamp('2026-01-16 12:00') + timedelta(minutes=i),
                    90.0 + i*0.01,
                    90.1 + i*0.01,
                    89.9 + i*0.01,
                    90.0 + i*0.01,
                    100 + i
                ) for i in range(min(size, 6))
            ]
            return bars
        else:
            # 返回包含next_page_token列的DataFrame
            df = pd.DataFrame({
                'time': pd.date_range('2026-01-16 12:00', periods=min(size, 7), freq='1min'),
                'open': [90.0 + i*0.01 for i in range(min(size, 7))],
                'high': [90.1 + i*0.01 for i in range(min(size, 7))],
                'low': [89.9 + i*0.01 for i in range(min(size, 7))],
                'close': [90.0 + i*0.01 for i in range(min(size, 7))],
                'volume': [100 + i for i in range(min(size, 7))],
                'next_page_token': [None if i == min(size, 7)-1 else f'token_{i}' for i in range(min(size, 7))]
            })
            return df

    def get_stock_briefs(self, symbols):
        """获取股票简介"""
        # return a pandas DataFrame for compatibility with callers/tests
        return pd.DataFrame([{"symbol": sym, "name": f"Name for {sym}", "currency": "USD"} for sym in symbols])

    def get_future_contracts(self, exchange_code, product_code, year, month):
        """获取期货合约"""
        return []

    def get_current_future_contract(self, product_code):
        """获取当前期货合约"""
        return {}

    def get_all_future_contracts(self, exchange_code, product_code):
        """获取所有期货合约"""
        return []

    def get_future_exchanges(self):
        """获取期货交易所"""
        return pd.DataFrame([
            {"code": "CME", "name": "Chicago Mercantile Exchange"},
            {"code": "NYMEX", "name": "New York Mercantile Exchange"}
        ])

    def get_future_brief(self, symbols):
        """获取期货简要信息"""
        return pd.DataFrame([
            {"identifier": sym, "name": f"Future for {sym}", "currency": "USD", "multiplier": 100}
            for sym in symbols
        ])

    def get_quote_permission(self):
        """获取行情权限"""
        return {"permission": True}


class MockTradeApiAdapter(TradeApiInterface):
    """模拟交易API适配器"""
    
    def __init__(self, account=None):
        import random
        import time
        self.random = random
        self.time = time
        self.account = account  # Mock也保存account，用于验证
        
        # 模拟订单存储
        self.orders = {}
    
    def place_order(self, symbol=None, side=None, order_type=None, quantity=None, time_in_force=None, limit_price=None, stop_price=None, order=None):
        """模拟下单 - 支持Order对象或参数"""
        # 如果传入Order对象（真实API的调用方式）
        if order is not None:
            # 验证Order对象必须有account
            if not hasattr(order, 'account') or not order.account:
                raise ValueError("Order对象必须包含account字段")
            symbol = order.contract.symbol if hasattr(order, 'contract') and hasattr(order.contract, 'symbol') else symbol
            side = order.action if hasattr(order, 'action') else side
            order_type = order.order_type if hasattr(order, 'order_type') else order_type
            quantity = order.quantity if hasattr(order, 'quantity') else quantity
            time_in_force = order.time_in_force if hasattr(order, 'time_in_force') else time_in_force
            limit_price = order.limit_price if hasattr(order, 'limit_price') else limit_price
        
        # 验证account（如果使用参数方式调用，account应该在self.account中）
        # 注意：Mock模式下，如果account未设置，也应该模拟真实API的行为
        if not self.account:
            # 在Mock模式下，如果没有account，可以选择：
            # 1. 抛出异常（模拟真实API）
            # 2. 允许下单（仅用于测试）
            # 这里选择抛出异常，确保测试能发现account问题
            raise ValueError("account不能为空，无法创建订单（Mock模式验证）")
        
        order_id = f"ORDER_{int(self.time.time())}_{self.random.randint(100000, 999999)}"
        
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
        
        # 存储订单（用于后续查询验证）
        self.orders[order_id] = order
        return order
    
    def get_order(self, account=None, id=None, order_id=None, **kwargs):
        """查询单个订单 - Mock实现"""
        # 优先使用order_id，其次使用id
        target_id = order_id or id
        if target_id:
            # 转换为字符串匹配（因为order_id可能是int或str）
            target_id_str = str(target_id)
            for stored_id, order in self.orders.items():
                if str(stored_id) == target_id_str:
                    return order
        return None
    
    def get_orders(self, account=None, symbol=None, limit=100, **kwargs):
        """查询订单列表 - Mock实现"""
        orders = list(self.orders.values())
        
        # 按symbol过滤（支持两种格式：SIL.COMEX.202603 和 SIL2603）
        if symbol:
            # 尝试匹配原始symbol或转换后的symbol
            symbol_variants = [symbol]
            # 如果symbol是完整格式，添加转换后的格式
            if '.' in symbol:
                parts = symbol.split('.')
                if len(parts) >= 2:
                    base = parts[0]
                    datepart = parts[-1]
                    if len(datepart) == 6 and datepart.isdigit():
                        symbol_variants.append(f"{base}{datepart[-4:]}")
            # 如果symbol是简短格式，添加完整格式（如果可能）
            elif len(symbol) >= 6 and symbol[-4:].isdigit():
                base = symbol[:-4]
                datepart = symbol[-4:]
                symbol_variants.append(f"{base}.COMEX.{datepart}")
            
            orders = [o for o in orders if hasattr(o, 'symbol') and o.symbol in symbol_variants]
        
        # 限制数量
        if limit:
            orders = orders[:limit]
        
        return orders


class ApiAdapterManager:
    """API适配器管理器"""
    
    def __init__(self):
        self.quote_api = None
        self.trade_api = None
        self.is_mock_mode = False
    
    def initialize_real_apis(self, quote_client, trade_client, account=None):
        """初始化真实API"""
        self.quote_api = RealQuoteApiAdapter(quote_client)
        
        # 确定account值：优先使用传入的account，否则从trade_client.config获取
        if account:
            final_account = account
        elif hasattr(trade_client, 'config'):
            final_account = getattr(trade_client.config, 'account', None)
        else:
            final_account = None
        
        # 如果仍然没有account，尝试从quote_client.config获取（通常两者配置相同）
        if not final_account and hasattr(quote_client, 'config'):
            final_account = getattr(quote_client.config, 'account', None)
        
        # 如果还是没有，尝试从环境变量获取
        if not final_account:
            import os
            final_account = os.getenv('ACCOUNT') or os.getenv('TIGER_ACCOUNT')
        
        # 创建trade_adapter时直接传入account
        trade_adapter = RealTradeApiAdapter(trade_client, account=final_account)
        
        # 保存account以便后续使用（强制设置，即使为None也要保存）
        self._account = final_account
        trade_adapter.account = final_account  # 确保设置
        
        if final_account:
            print(f"✅ [API初始化] account已设置: {final_account}")
            print(f"✅ [API初始化] 验证: api_manager._account={self._account}, trade_api.account={trade_adapter.account}")
        else:
            print(f"⚠️ [API初始化] account为空，可能导致下单失败")
            print(f"⚠️ [API初始化] 调试信息: account参数={account}, trade_client.config.account={getattr(trade_client.config, 'account', None) if hasattr(trade_client, 'config') else 'N/A'}")
        
        self.trade_api = trade_adapter
        self.is_mock_mode = False
    
    def initialize_mock_apis(self, account=None):
        """初始化模拟API"""
        self.quote_api = MockQuoteApiAdapter()
        self.trade_api = MockTradeApiAdapter(account=account)
        self._account = account  # Mock模式也保存account
        self.is_mock_mode = True


# 创建全局API管理器实例
api_manager = ApiAdapterManager()