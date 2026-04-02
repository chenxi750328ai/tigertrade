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

try:
    from src import order_log
except ImportError:
    order_log = None
try:
    from src.order_log import log_dfx as _log_dfx
except ImportError:
    def _log_dfx(*_a, **_k):
        """order_log 不可用时静默跳过（极少见）；正常环境必有 src.order_log.log_dfx"""
        pass

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
        STP = 'STP'
    class OrderSide:
        BUY = 'BUY'
        SELL = 'SELL'
    class TimeInForce:
        DAY = 'DAY'


class OrderExecutor:
    """订单执行器 - 统一的下单逻辑（直接调用API，模块化设计）"""
    
    def __init__(self, risk_manager=None, state_fallback=None):
        """
        初始化订单执行器
        
        Args:
            risk_manager: 风险管理模块（默认使用tiger1，仅用于风控检查和止损计算）
            state_fallback: 状态来源模块（current_position/daily_loss 等），风控检查前会从此同步到 risk_manager
        """
        self.risk_manager = risk_manager or t1
        # 当 tiger1 以 python src/tiger1.py 运行时 __main__ 可能尚未定义 check_risk_control（主块在定义之前），回退到导入的 t1
        self._risk_fallback = state_fallback
        if not callable(getattr(self.risk_manager, 'check_risk_control', None)):
            if self._risk_fallback is None:
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
                # 根因：Tiger SDK 的 TradeClient.config 为 None，不能用 trade_client.config.account
                account = getattr(t1.client_config, 'account', None) if getattr(t1, 'client_config', None) else None
                if account is None and hasattr(t1.trade_client, 'config') and t1.trade_client.config is not None:
                    account = getattr(t1.trade_client.config, 'account', None)
                api_manager.initialize_real_apis(t1.quote_client, t1.trade_client, account=account)
                print(f"✅ [OrderExecutor] API初始化成功，account={account}")
            else:
                # 如果没有客户端，尝试从配置文件创建（根因修复：用绝对路径，避免 cwd 依赖导致 1010）
                try:
                    from tigeropen.tiger_open_config import TigerOpenClientConfig
                    from tigeropen.quote.quote_client import QuoteClient
                    from tigeropen.trade.trade_client import TradeClient
                    from pathlib import Path
                    _root = Path(__file__).resolve().parents[2]
                    _cfg_path = str(_root / "openapicfg_dem")
                    # 尝试demo配置
                    try:
                        client_config = TigerOpenClientConfig(props_path=_cfg_path)
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
        # 每次买入前从老虎后台同步持仓
        if hasattr(t1, 'sync_positions_from_backend') and callable(t1.sync_positions_from_backend):
            try:
                t1.sync_positions_from_backend()
            except Exception:
                pass
        # 硬顶：以老虎后台为准（持仓+待成交买单），与 tiger1.DEMO_MAX_POSITION 一致，避免多轮/多进程叠单
        HARD_MAX = getattr(t1, 'DEMO_MAX_POSITION', 2)
        try:
            fn = getattr(t1, 'get_effective_position_for_buy', None)
            pos = int((fn() if callable(fn) else getattr(t1, 'current_position', 0)) or 0)
        except Exception as e:
            pos = HARD_MAX  # 任何异常时保守拒绝
            if hasattr(t1, 'logger') and t1.logger:
                t1.logger.warning("[DFX] get_effective_position_for_buy 异常: %s，保守拒绝", e)
        # DFX：每次买入决策打 INFO，便于回溯 52/62 手问题（见 docs/回溯_62手持仓_数据定位与DFX改进.md）
        import logging
        _log = logging.getLogger('order_executor')
        if pos >= HARD_MAX:
            _log.info("[DFX] 买入拒绝 | pos=%s >= max=%s", pos, HARD_MAX)
            return False, f"持仓已达硬顶({pos}>={HARD_MAX}手)，拒绝买入"
        _log.info("[DFX] 买入放行 | pos=%s < max=%s", pos, HARD_MAX)
        # 解析实际用于风控的模块
        risk = self.risk_manager if callable(getattr(self.risk_manager, 'check_risk_control', None)) else t1
        # 同步其他 state；current_position 必须用上面 sync 后的值，不得被 __main__ 覆盖（MOE 路径 __main__ 不更新，覆盖会导致风控始终看到 0→超限 52 手）
        if self._risk_fallback is not None:
            for attr in ('daily_loss', 'grid_lower', 'grid_upper', 'atr_5m'):
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
                    account = getattr(t1.client_config, 'account', None) if getattr(t1, 'client_config', None) else None
                    if account is None and getattr(t1.trade_client, 'config', None) is not None:
                        account = getattr(t1.trade_client.config, 'account', None)
                    api_manager.initialize_real_apis(t1.quote_client, t1.trade_client, account=account)
                    trade_api = api_manager.trade_api
                    print(f"✅ [订单执行] API重新初始化成功，account={account}")
                
                if trade_api is None:
                    return False, "交易API未初始化，无法下单"
            
            # 准备订单参数
            order_side = OrderSide.BUY
            order_type = OrderType.LMT  # 限价单（有价格时）
            order_quantity = 1  # 数量：1手
            _log = __import__('logging').getLogger('order_executor')
            
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

            # 止损+止盈为必须项；优先老虎 BRACKETS（limit_order_with_legs + LOSS + PROFIT）
            _sl = round(stop_loss_price / min_tick) * min_tick if min_tick > 0 else stop_loss_price
            _tp = round(take_profit_price / min_tick) * min_tick if min_tick > 0 else take_profit_price
            _sl = round(_sl, nd)
            _tp = round(_tp, nd)

            used_bracket = False
            bracket_fn = getattr(trade_api, "place_limit_with_bracket", None)
            if callable(bracket_fn):
                try:
                    order_result = bracket_fn(
                        symbol_to_use,
                        order_side,
                        order_quantity,
                        limit_price,
                        _sl,
                        _tp,
                        TimeInForce.DAY,
                    )
                    used_bracket = True
                    _log.info("[OrderExecutor] 已提交老虎 BRACKETS 组合单（限价+止损+止盈）")
                    _boid = getattr(order_result, "order_id", None) or (
                        order_result.get("order_id") if isinstance(order_result, dict) else None
                    ) or str(order_result)
                    _log_dfx(
                        "bracket_submitted",
                        "OrderExecutor limit_order_with_legs",
                        order_id=str(_boid),
                        symbol=str(symbol_to_use),
                        limit_price=float(limit_price),
                        stop_loss=float(_sl),
                        take_profit=float(_tp),
                        quantity=int(order_quantity),
                    )
                except Exception as br_e:
                    _log.warning("[OrderExecutor] 组合单失败，回退为限价单+成交后止损/止盈: %s", br_e)
                    _log_dfx(
                        "bracket_failed",
                        str(br_e),
                        symbol=str(symbol_to_use),
                        limit_price=float(limit_price),
                        stop_loss=float(_sl),
                        take_profit=float(_tp),
                        quantity=int(order_quantity),
                    )

            if not used_bracket:
                order_result = trade_api.place_order(
                    symbol_to_use,
                    order_side,
                    order_type,
                    order_quantity,
                    TimeInForce.DAY,
                    limit_price,
                    None,
                )
            
            # 处理返回结果
            if hasattr(order_result, 'order_id'):
                order_id = order_result.order_id
            elif isinstance(order_result, dict):
                order_id = order_result.get('order_id') or order_result.get('id')
            else:
                order_id = str(order_result)
            
            # 仅当后台能查到该单才记为成功，避免 LOG 显示成功但后台无订单
            account = getattr(trade_api, 'account', None) or getattr(t1, 'client_config', None) and getattr(t1.client_config, 'account', None)
            if account and callable(getattr(trade_api, 'get_orders', None)):
                try:
                    found = False
                    _oid = str(order_id)
                    for _ in range(8):  # 给后端订单索引一点传播时间（约 8 秒）
                        if callable(getattr(trade_api, 'get_order', None)):
                            one = trade_api.get_order(account=account, id=order_id)
                            if one is not None:
                                found = True
                                break
                        recent = trade_api.get_orders(account=account, symbol=symbol_to_use, limit=30)
                        for o in (recent or []):
                            oid = getattr(o, 'order_id', None) or getattr(o, 'id', None)
                            if oid is not None and str(oid) == _oid:
                                found = True
                                break
                        if found:
                            break
                        time.sleep(1)
                    if not found:
                        _log = __import__('logging').getLogger('order_executor')
                        _log.warning("[OrderExecutor] 订单已提交但后台未查到 order_id=%s，不记为成功", order_id)
                        return False, "订单已提交后 8 秒内后台未查到，可能被拒或延迟，请核对后台"
                except Exception as verify_e:
                    _log = __import__('logging').getLogger('order_executor')
                    _log.warning("[OrderExecutor] 后台校验异常: %s，不记为成功", verify_e)
                    return False, f"订单已提交但后台校验失败: {verify_e}，请核对后台"
            
            # 未走组合单时：主单 FILLED 后再挂 STP/TP（与 place_tiger_order 回退路径一致）
            if not used_bracket and (stop_loss_price is not None or take_profit_price is not None):
                _wait_fn = getattr(trade_api, "wait_until_buy_filled", None)
                if callable(_wait_fn):
                    if not _wait_fn(order_id, symbol_to_use, timeout_sec=120, poll_sec=1.0):
                        _log.warning(
                            "[OrderExecutor] 买单 %s 超时未 FILLED，跳过交易所止损/止盈；请依赖策略软止损或成交后补挂",
                            order_id,
                        )
                        _log_dfx(
                            "sl_tp_skipped",
                            "buy_not_filled_within_timeout",
                            parent_order_id=str(order_id),
                            symbol=str(symbol_to_use),
                            timeout_sec=120,
                        )
                        _sl = None
                        _tp = None
                    else:
                        _sl = round(stop_loss_price / min_tick) * min_tick if stop_loss_price is not None and min_tick > 0 else None
                        _tp = round(take_profit_price / min_tick) * min_tick if take_profit_price is not None and min_tick > 0 else None
                else:
                    _sl = round(stop_loss_price / min_tick) * min_tick if stop_loss_price is not None and min_tick > 0 else None
                    _tp = round(take_profit_price / min_tick) * min_tick if take_profit_price is not None and min_tick > 0 else None
                if _sl is not None:
                    try:
                        _stp = getattr(OrderType, 'STP', 'STP')
                        sl_ret = trade_api.place_order(
                            symbol=symbol_to_use,
                            side=OrderSide.SELL,
                            order_type=_stp,
                            quantity=order_quantity,
                            time_in_force=TimeInForce.DAY,
                            limit_price=None,
                            stop_price=_sl,
                        )
                        _sl_cid = getattr(sl_ret, "order_id", None) or (
                            sl_ret.get("order_id") if isinstance(sl_ret, dict) else None
                        ) or str(sl_ret)
                        _log.info("[OrderExecutor] 已提交止损单 | SELL %s手 | 触发价=%.3f", order_quantity, _sl)
                        _log_dfx(
                            "stop_loss_submitted",
                            "",
                            parent_order_id=str(order_id),
                            child_order_id=str(_sl_cid),
                            symbol=str(symbol_to_use),
                            stop_price=float(_sl),
                            quantity=int(order_quantity),
                        )
                    except Exception as sl_e:
                        _log.warning("[OrderExecutor] 止损单提交失败（主单已成交）：%s", sl_e)
                        _log_dfx(
                            "stop_loss_submit_failed",
                            str(sl_e),
                            parent_order_id=str(order_id),
                            symbol=str(symbol_to_use),
                            stop_price=float(_sl),
                            quantity=int(order_quantity),
                        )
                if _tp is not None:
                    try:
                        _lmt = getattr(OrderType, 'LMT', 'LMT')
                        tp_ret = trade_api.place_order(
                            symbol=symbol_to_use,
                            side=OrderSide.SELL,
                            order_type=_lmt,
                            quantity=order_quantity,
                            time_in_force=TimeInForce.DAY,
                            limit_price=_tp,
                            stop_price=None,
                        )
                        _tp_cid = getattr(tp_ret, "order_id", None) or (
                            tp_ret.get("order_id") if isinstance(tp_ret, dict) else None
                        ) or str(tp_ret)
                        _log.info("[OrderExecutor] 已提交止盈单 | SELL %s手 | 价格=%.3f", order_quantity, _tp)
                        _log_dfx(
                            "take_profit_submitted",
                            "",
                            parent_order_id=str(order_id),
                            child_order_id=str(_tp_cid),
                            symbol=str(symbol_to_use),
                            limit_price=float(_tp),
                            quantity=int(order_quantity),
                        )
                    except Exception as tp_e:
                        _log.warning("[OrderExecutor] 止盈单提交失败（主单已成交）：%s", tp_e)
                        _log_dfx(
                            "take_profit_submit_failed",
                            str(tp_e),
                            parent_order_id=str(order_id),
                            symbol=str(symbol_to_use),
                            limit_price=float(_tp),
                            quantity=int(order_quantity),
                        )
            
            # 更新持仓状态（与tiger1保持一致）
            t1.current_position += order_quantity
            try:
                t1._last_buy_success_ts = time.time()
            except Exception:
                pass
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
            
            # 写入 order_log，便于报告/分析
            if order_log:
                order_log.log_order(
                    'BUY', order_quantity, limit_price, str(order_id),
                    'success', 'real', stop_loss_price, take_profit_price,
                    reason=reason or 'auto', source='auto',
                    symbol=getattr(t1, '_to_api_identifier', lambda x: x)(t1.FUTURE_SYMBOL) if hasattr(t1, '_to_api_identifier') else getattr(t1, 'FUTURE_SYMBOL', ''),
                    order_type='limit_bracket' if used_bracket else 'limit',
                    run_env=getattr(t1, 'RUN_ENV', None),
                )
            
            return True, f"订单提交成功 | 价格={limit_price:.3f}, 止损={stop_loss_price:.3f}, 止盈={take_profit_price:.3f}, 订单ID={order_id}"
            
        except Exception as e:
            error_msg = str(e)
            if order_log:
                _sym = getattr(t1, '_to_api_identifier', lambda x: x)(t1.FUTURE_SYMBOL) if hasattr(t1, '_to_api_identifier') else getattr(t1, 'FUTURE_SYMBOL', '')
                order_log.log_order('BUY', 1, price, f"ORDER_{int(time.time())}_{random.randint(1000,9999)}", 'fail', 'real', stop_loss_price, take_profit_price, reason=reason or 'auto', error=error_msg, source='auto', symbol=_sym, order_type='limit', run_env=getattr(t1, 'RUN_ENV', None))
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
        # 每次卖出前也同步持仓，与 execute_buy 一致，确保看到最新状态（含刚成交的买单）
        if hasattr(t1, 'sync_positions_from_backend') and callable(t1.sync_positions_from_backend):
            try:
                t1.sync_positions_from_backend()
            except Exception:
                pass
        # 若使用了 t1 回退，先同步 __main__ 的 state
        if self._risk_fallback is not None:
            for attr in ('current_position', 'current_short_position', 'daily_loss'):
                if hasattr(self._risk_fallback, attr):
                    setattr(self.risk_manager, attr, getattr(self._risk_fallback, attr))
        # 持仓检查：无多头时卖出 = 卖出开仓（开空），必须阻止；空头硬顶 3 手
        if self.risk_manager.current_position <= 0:
            short_pos = 0
            try:
                fn = getattr(t1, 'get_effective_short_position_for_sell', None)
                short_pos = int((fn() if callable(fn) else getattr(t1, 'current_short_position', 0)) or 0)
            except Exception:
                short_pos = 999
            if short_pos >= 3:
                _log = __import__('logging').getLogger('order_executor')
                _log.info("[DFX] 卖出拒绝 | 空头=%s >= max=3（卖出开仓达硬顶）", short_pos)
                return False, f"空头已达硬顶({short_pos}>=3手)，拒绝卖出开仓"
            _log = __import__('logging').getLogger('order_executor')
            _log.info("[DFX] 卖出拒绝 | 无多头持仓，拒绝卖出开仓（会开空）")
            return False, "无多头持仓，无法卖出（不允许卖出开仓）"
        _log = __import__('logging').getLogger('order_executor')
        _log.info("[DFX] 卖出放行 | 多头=%s，平多 1 手", self.risk_manager.current_position)
        # 执行下单（直接调用API，不依赖tiger1.place_tiger_order）
        try:
            # 获取交易API
            trade_api = api_manager.trade_api
            if trade_api is None:
                # 尝试重新初始化API（如果可能）
                print("⚠️ [订单执行] trade_api为None，尝试重新初始化...")
                # 检查是否有可用的客户端
                if hasattr(t1, 'trade_client') and t1.trade_client is not None:
                    account = getattr(t1.client_config, 'account', None) if getattr(t1, 'client_config', None) else None
                    if account is None and getattr(t1.trade_client, 'config', None) is not None:
                        account = getattr(t1.trade_client.config, 'account', None)
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
            
            # 仅当后台能查到该单才记为成功，避免 LOG 显示成功但后台无订单
            account = getattr(trade_api, 'account', None) or getattr(t1, 'client_config', None) and getattr(t1.client_config, 'account', None)
            if account and callable(getattr(trade_api, 'get_orders', None)):
                try:
                    found = False
                    _oid = str(order_id)
                    for _ in range(8):
                        if callable(getattr(trade_api, 'get_order', None)):
                            one = trade_api.get_order(account=account, id=order_id)
                            if one is not None:
                                found = True
                                break
                        recent = trade_api.get_orders(account=account, symbol=symbol_to_use, limit=30)
                        for o in (recent or []):
                            oid = getattr(o, 'order_id', None) or getattr(o, 'id', None)
                            if oid is not None and str(oid) == _oid:
                                found = True
                                break
                        if found:
                            break
                        time.sleep(1)
                    if not found:
                        _log = __import__('logging').getLogger('order_executor')
                        _log.warning("[OrderExecutor] 卖单已提交但后台未查到 order_id=%s，不记为成功", order_id)
                        return False, "订单已提交后 8 秒内后台未查到，可能被拒或延迟，请核对后台"
                except Exception as verify_e:
                    _log = __import__('logging').getLogger('order_executor')
                    _log.warning("[OrderExecutor] 卖单后台校验异常: %s，不记为成功", verify_e)
                    return False, f"订单已提交但后台校验失败: {verify_e}，请核对后台"
            
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
            
            # 写入 order_log
            if order_log:
                _sym = getattr(t1, '_to_api_identifier', lambda x: x)(t1.FUTURE_SYMBOL) if hasattr(t1, '_to_api_identifier') else getattr(t1, 'FUTURE_SYMBOL', '')
                order_log.log_order('SELL', order_quantity, limit_price, str(order_id), 'success', 'real', None, None, reason='auto', source='auto', symbol=_sym, order_type='limit', run_env=getattr(t1, 'RUN_ENV', None))
            
            return True, f"订单提交成功 | 价格={limit_price:.3f}, 持仓={t1.current_position}手, 订单ID={order_id}"
            
        except Exception as e:
            error_msg = str(e)
            if order_log:
                _sym = getattr(t1, '_to_api_identifier', lambda x: x)(t1.FUTURE_SYMBOL) if hasattr(t1, '_to_api_identifier') else getattr(t1, 'FUTURE_SYMBOL', '')
                order_log.log_order('SELL', 1, price, f"ORDER_{int(time.time())}_{random.randint(1000,9999)}", 'fail', 'real', None, None, reason='auto', error=error_msg, source='auto', symbol=_sym, order_type='limit', run_env=getattr(t1, 'RUN_ENV', None))
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
