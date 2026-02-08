"""
订单执行器
统一的订单执行逻辑 - 直接调用API，不依赖tiger1.place_tiger_order
"""
from typing import Tuple, Optional
import math
import sys
import time
import random
sys.path.insert(0, '/home/cx/tigertrade')

from src.api_adapter import api_manager

# 导入tiger1模块 - 处理脚本运行时的导入问题
try:
    import importlib
    import sys
    # 如果tiger1.py作为脚本运行，__main__模块就是tiger1
    if '__main__' in sys.modules and hasattr(sys.modules['__main__'], 'check_risk_control'):
        # 使用__main__模块
        t1 = sys.modules['__main__']
    else:
        # 正常导入
        from src import tiger1 as t1
except (ImportError, AttributeError):
    # 如果导入失败，尝试直接导入
    try:
        import tiger1 as t1
    except ImportError:
        # 最后的fallback：创建一个mock对象
        class MockTiger1:
            def check_risk_control(self, price, side):
                return True
            def compute_stop_loss(self, price, atr, grid_lower):
                return price - atr * 2, atr * 2
            @property
            def current_position(self):
                return 0
        t1 = MockTiger1()

# 导入订单类型（如果可用）
try:
    from tigeropen.common.consts import OrderType, OrderSide, TimeInForce
except ImportError:
    # 如果导入失败，使用字符串常量
    class OrderType:
        LMT = 'LMT'
        MKT = 'MKT'
    class OrderSide:
        BUY = 'BUY'
        SELL = 'SELL'
    class TimeInForce:
        DAY = 'DAY'


class OrderExecutor:
    """订单执行器 - 统一的下单逻辑（直接调用API，模块化设计）"""
    
    def __init__(self, risk_manager=None):
        """
        初始化订单执行器
        
        Args:
            risk_manager: 风险管理模块（默认使用tiger1，仅用于风控检查和止损计算）
        """
        self.risk_manager = risk_manager or t1
        # 当 tiger1 以 python src/tiger1.py 运行时 __main__ 可能尚未定义 check_risk_control（主块在定义之前），回退到导入的 t1
        self._risk_fallback = None
        if not callable(getattr(self.risk_manager, 'check_risk_control', None)):
            self._risk_fallback = self.risk_manager  # 保留原引用用于同步 state
            self.risk_manager = t1
        # 下单逻辑直接使用api_manager，不依赖tiger1.place_tiger_order
        
        # 确保API已初始化
        self._ensure_api_initialized()
    
    def _ensure_api_initialized(self):
        """确保API已正确初始化"""
        # 如果trade_api未初始化，尝试初始化
        if api_manager.trade_api is None:
            print("⚠️ [OrderExecutor] trade_api未初始化，尝试初始化...")
            
            # 检查tiger1是否有可用的客户端
            if hasattr(t1, 'trade_client') and t1.trade_client is not None:
                account = None
                if hasattr(t1.trade_client, 'config'):
                    account = getattr(t1.trade_client.config, 'account', None)
                api_manager.initialize_real_apis(t1.quote_client, t1.trade_client, account=account)
                print(f"✅ [OrderExecutor] API初始化成功，account={account}")
            else:
                # 如果没有客户端，尝试从配置文件创建
                try:
                    from tigeropen.tiger_open_config import TigerOpenClientConfig
                    from tigeropen.quote.quote_client import QuoteClient
                    from tigeropen.trade.trade_client import TradeClient
                    
                    # 尝试demo配置
                    try:
                        client_config = TigerOpenClientConfig(props_path='./openapicfg_dem')
                        quote_client = QuoteClient(client_config)
                        trade_client = TradeClient(client_config)
                        account = getattr(client_config, 'account', None)
                        
                        api_manager.initialize_real_apis(quote_client, trade_client, account=account)
                        print(f"✅ [OrderExecutor] 从配置文件初始化API成功，account={account}")
                        
                        # 更新tiger1的客户端引用
                        t1.quote_client = quote_client
                        t1.trade_client = trade_client
                    except Exception as e:
                        print(f"⚠️ [OrderExecutor] 从配置文件初始化失败: {e}")
                except ImportError as e:
                    print(f"⚠️ [OrderExecutor] 无法导入Tiger SDK: {e}")
    
    def execute_buy(self, 
                   price: float, 
                   atr: float, 
                   grid_lower: float, 
                   grid_upper: float, 
                   confidence: float, 
                   profit_pred: Optional[float] = None) -> Tuple[bool, str]:
        """
        执行买入订单
        
        Args:
            price: 买入价格
            atr: ATR值
            grid_lower: 网格下轨
            grid_upper: 网格上轨
            confidence: 置信度
            profit_pred: 预测收益率（可选）
        
        Returns:
            (是否成功, 消息)
        """
        # 解析实际用于风控的模块（日志显示曾出现 __main__ 无 check_risk_control，调用时动态回退到 t1）
        risk = self.risk_manager if callable(getattr(self.risk_manager, 'check_risk_control', None)) else t1
        # 若使用了 t1 回退，先同步 __main__ 的 state 到 t1，保证风控看到的是当前运行状态
        if self._risk_fallback is not None:
            for attr in ('current_position', 'daily_loss', 'grid_lower', 'grid_upper', 'atr_5m'):
                if hasattr(self._risk_fallback, attr):
                    setattr(risk, attr, getattr(self._risk_fallback, attr))
        # 风控检查 - 需要传入grid_lower参数（check_risk_control内部需要）
        original_grid_lower = getattr(risk, 'grid_lower', None)
        risk.grid_lower = grid_lower
        try:
            risk_check_result = risk.check_risk_control(price, 'BUY')
            if not risk_check_result:
                return False, "风控检查未通过"
        finally:
            if original_grid_lower is not None:
                risk.grid_lower = original_grid_lower
            elif hasattr(risk, 'grid_lower'):
                delattr(risk, 'grid_lower')
        
        # 计算止损价格（同样用 risk，避免 compute_stop_loss 缺失）
        compute_sl = getattr(risk, 'compute_stop_loss', None) or getattr(t1, 'compute_stop_loss')
        stop_loss_price, projected_loss = compute_sl(price, atr, grid_lower)
        
        if stop_loss_price is None:
            return False, "止损计算失败"
        
        # 计算止盈价格
        take_profit_price = self._calculate_take_profit(grid_upper, atr)
        
        # 构建下单原因
        reason = f'策略预测(置信度:{confidence:.3f}'
        if profit_pred is not None:
            reason += f',收益率:{profit_pred*100:.2f}%'
        reason += ')'
        
        # 执行下单（直接调用API，不依赖tiger1.place_tiger_order）
        try:
            # 获取交易API
            trade_api = api_manager.trade_api
            if trade_api is None:
                # 尝试重新初始化API（如果可能）
                print("⚠️ [订单执行] trade_api为None，尝试重新初始化...")
                # 检查是否有可用的客户端
                if hasattr(t1, 'trade_client') and t1.trade_client is not None:
                    account = getattr(t1.trade_client.config, 'account', None) if hasattr(t1.trade_client, 'config') else None
                    api_manager.initialize_real_apis(t1.quote_client, t1.trade_client, account=account)
                    trade_api = api_manager.trade_api
                    print(f"✅ [订单执行] API重新初始化成功，account={account}")
                
                if trade_api is None:
                    return False, "交易API未初始化，无法下单"
            
            # 准备订单参数
            order_side = OrderSide.BUY
            order_type = OrderType.LMT  # 限价单（有价格时）
            order_quantity = 1  # 数量：1手
            
            # 转换symbol格式：SIL.COMEX.202603 -> SIL2603（Tiger API可能期望简短格式）
            symbol_to_use = t1.FUTURE_SYMBOL
            if hasattr(t1, '_to_api_identifier'):
                symbol_to_use = t1._to_api_identifier(t1.FUTURE_SYMBOL)
            elif '.' in symbol_to_use:
                # 简单转换：SIL.COMEX.202603 -> SIL2603
                parts = symbol_to_use.split('.')
                if len(parts) >= 2:
                    base = parts[0]
                    datepart = parts[-1]
                    if len(datepart) == 6 and datepart.isdigit():
                        symbol_to_use = f"{base}{datepart[-4:]}"
            
            # 按合约最小变动价位取整，避免 tick size 报错
            min_tick = getattr(t1, 'MIN_TICK', 0.01)
            try:
                if hasattr(t1, 'get_future_brief_info') and hasattr(t1, '_to_api_identifier'):
                    brief = t1.get_future_brief_info(t1._to_api_identifier(t1.FUTURE_SYMBOL) or t1.FUTURE_SYMBOL)
                    min_tick = float(brief.get('min_tick', min_tick)) if brief else min_tick
            except Exception:
                min_tick = float(getattr(t1, 'FUTURE_TICK_SIZE', 0.01) or 0.01)
            if min_tick <= 0:
                min_tick = 0.01
            limit_price = round(price / min_tick) * min_tick if min_tick > 0 else price
            nd = max(0, int(round(-math.log10(min_tick)))) if min_tick > 0 else 2
            limit_price = round(limit_price, nd)
            
            # 提交订单（使用位置参数，兼容Tiger API）
            order_result = trade_api.place_order(
                symbol_to_use,  # 使用转换后的symbol（SIL2603格式）
                order_side,
                order_type,
                order_quantity,
                TimeInForce.DAY,
                limit_price,  # limit_price（已按 min_tick 取整）
                None    # stop_price
            )
            
            # 处理返回结果
            if hasattr(order_result, 'order_id'):
                order_id = order_result.order_id
            elif isinstance(order_result, dict):
                order_id = order_result.get('order_id') or order_result.get('id')
            else:
                order_id = str(order_result)
            
            # 更新持仓状态（与tiger1保持一致）
            t1.current_position += order_quantity
            
            # 记录订单信息（与tiger1保持一致）
            order_id_str = f"ORDER_{int(time.time())}_{random.randint(1000, 9999)}"
            for i in range(order_quantity):
                individual_order_id = f"{order_id_str}_qty_{i+1}"
                t1.open_orders[individual_order_id] = {
                    'quantity': 1,
                    'price': price,
                    'timestamp': time.time(),
                    'type': 'buy',
                    'reason': reason
                }
            
            # 记录持仓信息
            for pos_id in range(t1.current_position - order_quantity, t1.current_position):
                t1.position_entry_times[pos_id] = time.time()
                t1.position_entry_prices[pos_id] = price
            
            return True, f"订单提交成功 | 价格={limit_price:.3f}, 止损={stop_loss_price:.3f}, 止盈={take_profit_price:.3f}, 订单ID={order_id}"
            
        except Exception as e:
            error_msg = str(e)
            # 明确识别授权错误
            if 'not authorized' in error_msg.lower() or 'authorized' in error_msg.lower() or 'authorization' in error_msg.lower():
                return False, f"❌ 授权失败: {error_msg}。需要在Tiger后台配置account授权给API用户。"
            # 明确识别其他常见错误
            if 'account' in error_msg.lower() and ('empty' in error_msg.lower() or '不能为空' in error_msg.lower()):
                return False, f"❌ account为空: {error_msg}。请检查account配置。"
            return False, f"下单异常: {error_msg}"
    
    def execute_sell(self, price: float, confidence: float) -> Tuple[bool, str]:
        """
        执行卖出订单
        
        Args:
            price: 卖出价格
            confidence: 置信度
        
        Returns:
            (是否成功, 消息)
        """
        # 若使用了 t1 回退，先同步 __main__ 的 state
        if self._risk_fallback is not None:
            for attr in ('current_position', 'daily_loss'):
                if hasattr(self._risk_fallback, attr):
                    setattr(self.risk_manager, attr, getattr(self._risk_fallback, attr))
        # 持仓检查
        if self.risk_manager.current_position <= 0:
            return False, "无持仓，无法卖出"
        
        # 执行下单（直接调用API，不依赖tiger1.place_tiger_order）
        try:
            # 获取交易API
            trade_api = api_manager.trade_api
            if trade_api is None:
                # 尝试重新初始化API（如果可能）
                print("⚠️ [订单执行] trade_api为None，尝试重新初始化...")
                # 检查是否有可用的客户端
                if hasattr(t1, 'trade_client') and t1.trade_client is not None:
                    account = getattr(t1.trade_client.config, 'account', None) if hasattr(t1.trade_client, 'config') else None
                    api_manager.initialize_real_apis(t1.quote_client, t1.trade_client, account=account)
                    trade_api = api_manager.trade_api
                    print(f"✅ [订单执行] API重新初始化成功，account={account}")
                
                if trade_api is None:
                    return False, "交易API未初始化，无法下单"
            
            # 准备订单参数
            order_side = OrderSide.SELL
            order_type = OrderType.LMT  # 限价单
            order_quantity = 1  # 数量：1手
            
            # 转换symbol格式：SIL.COMEX.202603 -> SIL2603（Tiger API可能期望简短格式）
            symbol_to_use = t1.FUTURE_SYMBOL
            if hasattr(t1, '_to_api_identifier'):
                symbol_to_use = t1._to_api_identifier(t1.FUTURE_SYMBOL)
            elif '.' in symbol_to_use:
                # 简单转换：SIL.COMEX.202603 -> SIL2603
                parts = symbol_to_use.split('.')
                if len(parts) >= 2:
                    base = parts[0]
                    datepart = parts[-1]
                    if len(datepart) == 6 and datepart.isdigit():
                        symbol_to_use = f"{base}{datepart[-4:]}"
            
            # 按合约最小变动价位取整，避免 tick size 报错
            min_tick = getattr(t1, 'MIN_TICK', 0.01)
            try:
                if hasattr(t1, 'get_future_brief_info') and hasattr(t1, '_to_api_identifier'):
                    brief = t1.get_future_brief_info(t1._to_api_identifier(t1.FUTURE_SYMBOL) or t1.FUTURE_SYMBOL)
                    min_tick = float(brief.get('min_tick', min_tick)) if brief else min_tick
            except Exception:
                min_tick = float(getattr(t1, 'FUTURE_TICK_SIZE', 0.01) or 0.01)
            if min_tick <= 0:
                min_tick = 0.01
            limit_price = round(price / min_tick) * min_tick if min_tick > 0 else price
            nd = max(0, int(round(-math.log10(min_tick)))) if min_tick > 0 else 2
            limit_price = round(limit_price, nd)
            
            # 提交订单（使用位置参数，兼容Tiger API）
            order_result = trade_api.place_order(
                symbol_to_use,  # 使用转换后的symbol（SIL2603格式）
                order_side,
                order_type,
                order_quantity,
                TimeInForce.DAY,
                limit_price,  # limit_price（已按 min_tick 取整）
                None    # stop_price
            )
            
            # 处理返回结果
            if hasattr(order_result, 'order_id'):
                order_id = order_result.order_id
            elif isinstance(order_result, dict):
                order_id = order_result.get('order_id') or order_result.get('id')
            else:
                order_id = str(order_result)
            
            # 更新持仓状态（与tiger1保持一致）
            t1.current_position -= order_quantity
            if t1.current_position < 0:
                t1.current_position = 0
            
            # 按先进先出原则匹配买单进行平仓
            remaining_qty = order_quantity
            while remaining_qty > 0 and t1.open_orders:
                oldest_buy_order_id = next(iter(t1.open_orders))
                buy_info = t1.open_orders.pop(oldest_buy_order_id)
                remaining_qty -= buy_info['quantity']
                
                # 记录已平仓交易
                closed_order_id = f"CLOSED_{int(time.time())}_{random.randint(1000, 9999)}"
                t1.closed_positions[closed_order_id] = {
                    'buy_order_id': oldest_buy_order_id,
                    'sell_order_id': order_id,
                    'buy_price': buy_info['price'],
                    'sell_price': price,
                    'profit': (price - buy_info['price']) * t1.FUTURE_MULTIPLIER
                }
            
            return True, f"订单提交成功 | 价格={limit_price:.3f}, 持仓={t1.current_position}手, 订单ID={order_id}"
            
        except Exception as e:
            error_msg = str(e)
            # 明确识别授权错误
            if 'not authorized' in error_msg.lower() or 'authorized' in error_msg.lower() or 'authorization' in error_msg.lower():
                return False, f"❌ 授权失败: {error_msg}。需要在Tiger后台配置account授权给API用户。"
            # 明确识别其他常见错误
            if 'account' in error_msg.lower() and ('empty' in error_msg.lower() or '不能为空' in error_msg.lower()):
                return False, f"❌ account为空: {error_msg}。请检查account配置。"
            return False, f"下单异常: {error_msg}"
    
    def _calculate_take_profit(self, grid_upper: float, atr: float) -> float:
        """
        计算止盈价格
        
        Args:
            grid_upper: 网格上轨
            atr: ATR值
        
        Returns:
            止盈价格
        """
        tp_offset = max(
            t1.TAKE_PROFIT_ATR_OFFSET * (atr if atr else 0), 
            t1.TAKE_PROFIT_MIN_OFFSET
        )
        return grid_upper - tp_offset
