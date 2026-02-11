#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
优化算法和收益率（每日例行：结果分析 + 算法优化）
- 例行工作目标：提升收益率。每轮 = 例行 + 解决问题 + 优化；有问题先尝试解决再继续，不光是跑完或 exit。
- 结果分析：API 历史订单 → 收益率；DEMO 多日志汇总 → 策略表现；网格/BOLL 回测 → 最优参数与 return_pct/win_rate。
- 效果数据来源与缺口说明见：docs/每日例行_效果数据说明.md、报告内「效果数据来源」节。
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

sys.path.insert(0, '/home/cx/tigertrade')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

logger = logging.getLogger(__name__)


def _normalize_order(order):
    """将订单统一为 dict，便于解析（与 fetch_tiger_yield_for_demo 一致）。"""
    def _attr(o, *keys, default=None):
        for k in keys:
            if hasattr(o, k):
                v = getattr(o, k)
                if v is not None:
                    return v
            if isinstance(o, dict):
                v = o.get(k)
                if v is not None:
                    return v
        return default
    if isinstance(order, dict):
        return order
    return {
        "order_id": _attr(order, "order_id", "id"),
        "status": _attr(order, "status", "order_status"),
        "side": _attr(order, "side", "action"),
        "quantity": _attr(order, "quantity", "qty"),
        "filled_quantity": _attr(order, "filled_quantity", "filled_qty"),
        "avg_fill_price": _attr(order, "avg_fill_price", "average_price"),
        "limit_price": _attr(order, "limit_price", "price"),
        "realized_pnl": _attr(order, "realized_pnl", "realized_pnL"),
    }


def _fetch_orders_from_tiger_direct(limit=1000):
    """当 api_manager 未初始化或无 get_orders 时，用 openapicfg_dem 直接拉老虎订单（与 fetch_tiger_yield_for_demo 一致）。
    返回 (orders_list_or_none, reason_if_fail)。orders 非空时 reason 为 None。"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(root, "openapicfg_dem")
    if not os.path.isdir(config_path):
        return None, "openapicfg_dem 目录不存在"
    try:
        from tigeropen.tiger_open_config import TigerOpenClientConfig
        from tigeropen.trade.trade_client import TradeClient
    except ImportError as e:
        return None, "tigeropen 未安装或不可用: %s" % (e,)
    try:
        config = TigerOpenClientConfig(props_path=config_path)
        client = TradeClient(config)
        acc = getattr(config, "account", None)
        if not acc:
            return None, "openapicfg_dem 中 account 为空"
        try:
            from src import tiger1 as t1
            symbol_api = t1._to_api_identifier(getattr(t1, "FUTURE_SYMBOL", "SIL.COMEX.202603"))
        except Exception:
            symbol_api = "SIL2603"
        orders = client.get_orders(account=acc, symbol=symbol_api, limit=limit)
        if orders is None:
            return [], "老虎 API get_orders 返回 None"
        if len(orders) == 0:
            return [], "老虎 API 返回 0 笔订单（该账户/合约可能无订单或未授权）"
        return orders, None
    except Exception as e:
        logger.debug("直接拉取老虎订单失败: %s", e)
        return None, "openapicfg_dem 拉取异常: %s" % (str(e)[:200],)


def _fetch_positions_from_tiger_direct():
    """用 openapicfg_dem 直接拉老虎持仓。返回 (positions_list_or_none, reason_if_fail)。"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(root, "openapicfg_dem")
    if not os.path.isdir(config_path):
        return None, "openapicfg_dem 目录不存在"
    try:
        from tigeropen.tiger_open_config import TigerOpenClientConfig
        from tigeropen.trade.trade_client import TradeClient
    except ImportError as e:
        return None, "tigeropen 未安装或不可用: %s" % (e,)
    try:
        config = TigerOpenClientConfig(props_path=config_path)
        client = TradeClient(config)
        acc = getattr(config, "account", None)
        if not acc:
            return None, "openapicfg_dem 中 account 为空"
        if not hasattr(client, 'get_positions'):
            return None, "TradeClient 无 get_positions 方法"
        positions = client.get_positions(account=acc)
        if positions is None:
            return [], "老虎 API get_positions 返回 None"
        return positions, None
    except Exception as e:
        logger.debug("直接拉取老虎持仓失败: %s", e)
        return None, "openapicfg_dem 拉取持仓异常: %s" % (str(e)[:200],)


def analyze_backend_orders_and_positions(orders, positions=None):
    """从后台订单（及可选持仓）分析：是否存在「有持仓但止损/止盈单已撤」的情况（可能保证金不足或强平）。
    返回 dict: position_qty, position_source, open_sl_tp_count, cancelled_sl_tp_count, alert_message, detail_lines。"""
    result = {
        "position_qty": 0,
        "position_source": "unknown",
        "open_sl_tp_count": 0,
        "cancelled_sl_tp_count": 0,
        "alert_message": None,
        "detail_lines": [],
    }
    # 从持仓 API 获取真实持仓（若可用）
    if positions is not None and len(positions) > 0:
        try:
            from src import tiger1 as t1
            symbol_api = t1._to_api_identifier(getattr(t1, "FUTURE_SYMBOL", "SIL.COMEX.202603"))
        except Exception:
            symbol_api = "SIL2603"
        total = 0
        for p in positions:
            obj = p if not isinstance(p, dict) else p
            sym = getattr(obj, "symbol", None) or (obj.get("symbol") if isinstance(obj, dict) else None)
            qty = getattr(obj, "quantity", None) or getattr(obj, "qty", None) or (obj.get("quantity") or obj.get("qty") if isinstance(obj, dict) else None)
            if sym and (symbol_api in str(sym) or "SIL" in str(sym)):
                try:
                    total += int(qty or 0)
                except (TypeError, ValueError):
                    pass
        result["position_qty"] = total
        result["position_source"] = "get_positions"
        result["detail_lines"].append("后台持仓（get_positions）: %s 手（合约含 SIL）。" % total)
    # 从订单推断：已成交买量 - 已成交卖量 ≈ 净持仓（近似）
    if orders:
        buy_filled = 0
        sell_filled = 0
        open_sl_tp = 0
        cancelled_sl_tp = 0
        for o in orders:
            row = _normalize_order(o)
            st = (str(row.get("status") or "")).upper()
            side = (str(row.get("side") or "")).upper()
            qty = row.get("filled_quantity") or row.get("quantity") or 0
            try:
                qty = int(float(qty))
            except (TypeError, ValueError):
                qty = 0
            if st in ("FILLED", "FILLED_ALL", "FINISHED"):
                if side in ("BUY", "LONG"):
                    buy_filled += qty
                elif side in ("SELL", "SHORT"):
                    sell_filled += qty
            # 止损/止盈单：通常有 stop_price 或 order_type 为 STOP/STOP_LIMIT 等；已撤单算入 cancelled
            has_stop = row.get("stop_price") is not None or "stop" in str(row.get("order_type") or "").lower()
            if has_stop or "STOP" in st or "止" in str(row.get("order_type") or ""):
                if st in ("CANCELLED", "CANCELED", "REJECTED", "EXPIRED"):
                    cancelled_sl_tp += 1
                elif st in ("PENDING", "OPEN", "LIVE", "NEW", "SUBMITTED", ""):
                    open_sl_tp += 1
        inferred_position = max(0, buy_filled - sell_filled)
        if result["position_source"] == "unknown":
            result["position_qty"] = inferred_position
            result["position_source"] = "inferred_from_orders"
            result["detail_lines"].append("从订单推断净持仓: %s 手（已成交买-卖）。" % inferred_position)
        result["open_sl_tp_count"] = open_sl_tp
        result["cancelled_sl_tp_count"] = cancelled_sl_tp
        result["detail_lines"].append("订单中疑似止损/止盈: 未撤=%s 笔，已撤/拒/过期=%s 笔。" % (open_sl_tp, cancelled_sl_tp))
    # 告警：有持仓但无有效止损止盈单（或大量被撤）
    pos = result["position_qty"]
    if pos > 0 and (result["open_sl_tp_count"] == 0 and result["cancelled_sl_tp_count"] > 0):
        result["alert_message"] = "检测到后台持仓 %s 手，但对应止损/止盈单已撤（可能保证金不足或强平）。当前风控未覆盖「有仓无止损止盈」的补单或告警，建议改进。" % pos
        result["detail_lines"].append("⚠️ " + result["alert_message"])
    elif pos > 0 and result["open_sl_tp_count"] == 0:
        result["alert_message"] = "检测到后台持仓 %s 手，未发现有效止损/止盈单（可能被系统撤单或未提交）。建议检查风控与补单逻辑。" % pos
        result["detail_lines"].append("⚠️ " + result["alert_message"])
    return result


def load_trading_history():
    """加载历史交易记录：先试 api_manager，若无订单则用 openapicfg_dem 直接拉老虎。
    返回 (orders, backend_empty_reason)。orders 非空时 backend_empty_reason 为 None；
    orders 为空时 backend_empty_reason 为字符串，供报告写入「实际收益率为空」的根因。"""
    logger.info("📊 加载历史交易记录...")
    orders = []
    reasons = []

    try:
        from src.api_adapter import api_manager
        if api_manager.trade_api and hasattr(api_manager.trade_api, 'get_orders'):
            from src import tiger1 as t1
            symbol_to_query = t1._to_api_identifier('SIL.COMEX.202603')
            orders = api_manager.trade_api.get_orders(
                account=getattr(api_manager, '_account', None),
                symbol=symbol_to_query,
                limit=1000
            )
            if orders:
                logger.info("✅ 从 api_manager 加载了 %s 条历史订单", len(orders))
                return orders, None
            reasons.append("api_manager.get_orders 返回 0 笔")
        else:
            reasons.append("api_manager 未初始化或 trade_api 无 get_orders")
    except Exception as e:
        reasons.append("api_manager 拉取异常: %s" % (str(e)[:150],))

    # 无订单时：用 openapicfg_dem 直接拉老虎（报告生成环境常未初始化 api_manager）
    direct, direct_reason = _fetch_orders_from_tiger_direct(limit=1000)
    if direct is not None and len(direct) > 0:
        logger.info("✅ 从老虎 API（openapicfg_dem）加载了 %s 条历史订单", len(direct))
        return direct, None
    if direct_reason:
        reasons.append("openapicfg_dem: %s" % direct_reason)
    elif direct is not None:
        reasons.append("openapicfg_dem 拉取返回 0 笔")

    backend_empty_reason = "；".join(reasons) if reasons else "未拉取到订单（原因未记录）"
    logger.warning("⚠️ 无法加载历史交易记录：%s", backend_empty_reason)
    return [], backend_empty_reason


def calculate_profitability(orders):
    """根据订单列表计算收益率；订单可为老虎 API 返回的对象或 dict，仅统计已成交（FILLED）且能解析盈亏的。"""
    logger.info("💰 计算收益率...")

    if not orders:
        logger.warning("⚠️ 没有交易记录，无法计算收益率")
        return None

    try:
        filled = []
        for o in orders:
            row = _normalize_order(o)
            st = (row.get("status") or "").upper()
            if st in ("FILLED", "FILLED_ALL", "FINISHED"):
                filled.append(row)

        total_profit = 0.0
        total_trades = len(filled)
        winning_trades = 0
        losing_trades = 0

        for r in filled:
            pnl = r.get("realized_pnl")
            if pnl is not None:
                try:
                    p = float(pnl)
                    total_profit += p
                    if p > 0:
                        winning_trades += 1
                    elif p < 0:
                        losing_trades += 1
                except (TypeError, ValueError):
                    pass

        profitability = {
            'total_profit': total_profit,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
            'average_profit': (total_profit / total_trades) if total_trades > 0 else 0
        }

        logger.info("✅ 收益率计算完成（已成交笔数=%s）", profitability["total_trades"])
        logger.info("  总交易数: %s", profitability["total_trades"])
        logger.info("  胜率: %.2f%%", profitability["win_rate"])
        logger.info("  平均收益: %.2f", profitability["average_profit"])

        return profitability

    except Exception as e:
        logger.error("❌ 计算收益率失败: %s", e)
        return None


def analyze_strategy_performance():
    """分析策略表现：从 DEMO 日志、today_yield 等汇总可用的运行效果，供策略报告展示。永远返回四策略的 dict，出错也填占位（错误即数据）。"""
    logger.info("📈 分析策略表现...")
    strategies = ['moe_transformer', 'lstm', 'grid', 'boll']
    performance_data = {}
    for s in strategies:
        performance_data[s] = {
            'profitability': 0,
            'win_rate': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0
        }

    try:
        # 从所有 DEMO 日志汇总统计（多日多文件，主推 DEMO 策略为 moe_transformer）
        # 同次汇总一并填入 grid/boll/lstm，避免对比报告里 demo_* 列为空
        try:
            from scripts.analyze_demo_log import aggregate_demo_logs
            demo = aggregate_demo_logs()
            if demo and demo.get('logs_scanned', 0) > 0:
                demo_fields = {
                    'demo_order_success': demo.get('order_success', 0),
                    'demo_sl_tp_log': demo.get('sl_tp_log', 0),
                    'demo_execute_buy_calls': demo.get('execute_buy_calls', 0),
                    'demo_success_orders_sum': demo.get('success_orders_sum', 0),
                    'demo_fail_orders_sum': demo.get('fail_orders_sum', 0),
                    'demo_logs_scanned': demo.get('logs_scanned', 0),
                }
                for sid in ('moe_transformer', 'lstm', 'grid', 'boll'):
                    for k, v in demo_fields.items():
                        performance_data[sid][k] = v
                logger.info("  DEMO 多日志汇总: 扫描 %s 个日志, order_success=%s, sl_tp=%s（已填入四策略）",
                            demo.get('logs_scanned'), demo.get('order_success'), demo.get('sl_tp_log'))
        except Exception as e:
            logger.warning("DEMO 日志统计未合并（已记入占位）: %s", e)
            for sid in strategies:
                performance_data[sid]['demo_note'] = f"汇总异常: {str(e)[:80]}"

        # 从 today_yield 补充今日收益率（四策略都填，便于报告统一展示）
        try:
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            yield_path = os.path.join(root, 'docs', 'today_yield.json')
            if os.path.isfile(yield_path):
                with open(yield_path, 'r', encoding='utf-8') as f:
                    y = json.load(f)
                pct = y.get('yield_pct') or y.get('yield_note')
                if pct and str(pct).strip() not in ('', '—'):
                    for sid in strategies:
                        performance_data[sid]['today_yield_pct'] = str(pct)
        except Exception as e:
            logger.debug("today_yield 未合并: %s", e)

    except Exception as e:
        logger.error(f"❌ 策略表现分析失败（仍返回占位数据）: {e}")
        for sid in strategies:
            performance_data[sid]['error_note'] = str(e)[:80]

    return performance_data


def ensure_test_csv(root, timeout_sec=300):
    """若缺 data/processed/test.csv 则尝试生成（解决问题后继续），成功返回 True。"""
    test_csv = os.path.join(root, 'data', 'processed', 'test.csv')
    if os.path.isfile(test_csv):
        return True
    logger.warning("缺 data/processed/test.csv，回测无法产出；尝试自动生成...")
    import subprocess
    scripts_dir = os.path.join(root, 'scripts')
    for script in ('data_preprocessing.py', 'merge_recent_data_and_train.py'):
        path = os.path.join(scripts_dir, script)
        if not os.path.isfile(path):
            continue
        try:
            r = subprocess.run(
                [sys.executable, path],
                cwd=root,
                timeout=timeout_sec,
                capture_output=True,
                text=True,
            )
            if os.path.isfile(test_csv):
                logger.info("✅ 已生成 test.csv（%s），继续回测", script)
                return True
            if r.returncode != 0 and r.stderr:
                logger.debug("%s: %s", script, r.stderr[:200])
        except subprocess.TimeoutExpired:
            logger.warning("%s 超时，跳过", script)
        except Exception as e:
            logger.debug("%s 未产出 test.csv: %s", script, e)
    return False


def optimize_parameters():
    """
    优化策略参数：对 grid/boll 做网格回测，返回最优参数及回测效果（供报告写入）。
    返回 (optimal_params, backtest_metrics)。backtest_metrics 用于填入 strategy_performance 的收益率/胜率。
    """
    logger.info("⚙️ 优化策略参数（网格/BOLL 回测）...")
    optimal_params = {}
    backtest_metrics = {}
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_csv = os.path.join(root, 'data', 'processed', 'test.csv')
    try:
        # grid：优先与实盘同一套代码的回测（仅数据源为文件）
        if os.path.isfile(test_csv):
            try:
                from src import tiger1 as t1
                r = t1.backtest_grid_trading_strategy_pro1(single_csv_path=test_csv, bars_1m=2000, bars_5m=1000, lookahead=120, step_seconds=5)
                if r and isinstance(r, dict) and (r.get('signals_evaluated') or r.get('num_trades')) is not None:
                    backtest_metrics['grid'] = {
                        'return_pct': r.get('return_pct'),
                        'win_rate': r.get('win_rate'),
                        'num_trades': r.get('num_trades'),
                        'avg_per_trade_pct': r.get('avg_per_trade_pct'),
                        'top_per_trade_pct': r.get('top_per_trade_pct'),
                    }
                    optimal_params['grid'] = {'source': 'backtest_grid_trading_strategy_pro1', 'same_code_as_live': True}
                    logger.info("  grid 回测(与实盘同逻辑): 收益=%.2f%%, 胜率=%.1f%%, 笔数=%s, 单笔均=%.2f%%, 单笔TOP=%.2f%%",
                                r.get('return_pct', 0) or 0, r.get('win_rate', 0) or 0, r.get('num_trades'),
                                r.get('avg_per_trade_pct') or 0, r.get('top_per_trade_pct') or 0)
            except Exception as e:
                logger.debug("  grid 同逻辑回测未产出: %s", e)
        if 'grid' not in backtest_metrics:
            from scripts.parameter_grid_search import grid_search_optimal_params
            try:
                r = grid_search_optimal_params('grid')
                if r and isinstance(r, dict):
                    optimal_params['grid'] = r.get('params', r)
                    if 'return_pct' in r or 'win_rate' in r:
                        backtest_metrics['grid'] = {
                            'return_pct': r.get('return_pct'), 'win_rate': r.get('win_rate'),
                            'num_trades': r.get('num_trades'), 'avg_per_trade_pct': r.get('avg_per_trade_pct'),
                            'top_per_trade_pct': r.get('top_per_trade_pct'),
                        }
                        logger.info("  grid 回测(参数网格): 收益=%.2f%%, 胜率=%.1f%%, 笔数=%s",
                                    r.get('return_pct', 0) or 0, r.get('win_rate', 0) or 0, r.get('num_trades'))
            except Exception as e:
                logger.debug("  grid 回测未产出: %s", e)
        # boll：仍用参数网格回测（后续可接 boll1m 同逻辑回测）
        from scripts.parameter_grid_search import grid_search_optimal_params
        try:
            r = grid_search_optimal_params('boll')
            if r and isinstance(r, dict):
                optimal_params['boll'] = r.get('params', r)
                if 'return_pct' in r or 'win_rate' in r:
                    backtest_metrics['boll'] = {
                        'return_pct': r.get('return_pct'), 'win_rate': r.get('win_rate'),
                        'num_trades': r.get('num_trades'), 'avg_per_trade_pct': r.get('avg_per_trade_pct'),
                        'top_per_trade_pct': r.get('top_per_trade_pct'),
                    }
                    logger.info("  boll 回测: 收益=%.2f%%, 胜率=%.1f%%, 笔数=%s",
                                r.get('return_pct', 0) or 0, r.get('win_rate', 0) or 0, r.get('num_trades'))
        except Exception as e:
            logger.debug("  boll 回测未产出: %s", e)
        # moe_transformer、lstm：用同一套 test.csv 信号回测，产出 num_trades/return_pct/win_rate
        try:
            from scripts.backtest_model_strategies import run_backtest_model_strategies
            model_bt = run_backtest_model_strategies()
            for name, m in (model_bt or {}).items():
                if m and isinstance(m, dict):
                    backtest_metrics[name] = {
                        'return_pct': m.get('return_pct'),
                        'win_rate': m.get('win_rate'),
                        'num_trades': m.get('num_trades'),
                        'avg_per_trade_pct': m.get('avg_per_trade_pct'),
                        'top_per_trade_pct': m.get('top_per_trade_pct'),
                    }
                    logger.info("  %s 回测: 收益=%.2f%%, 胜率=%.1f%%, 笔数=%s, 单笔均=%.2f%%, 单笔TOP=%.2f%%",
                                name, m.get('return_pct', 0) or 0, m.get('win_rate', 0) or 0, m.get('num_trades'),
                                m.get('avg_per_trade_pct') or 0, m.get('top_per_trade_pct') or 0)
        except Exception as e:
            logger.debug("  model strategies 回测未产出: %s", e)
        logger.info("✅ 参数优化完成" if (optimal_params or backtest_metrics) else "⚠️ 无回测数据（需 data/processed/test.csv）")
        return optimal_params, backtest_metrics
    except Exception as e:
        logger.warning("⚠️ 参数优化失败: %s", e)
        return {}, {}


def generate_optimization_report(profitability, performance, optimal_params, backend_empty_reason=None, backend_analysis=None):
    """生成优化报告。backend_empty_reason：当实际收益率（老虎核对）为空时，写入报告的空项根因说明。
    backend_analysis：后台订单与持仓分析结果（有仓无止损止盈等），若有 alert_message 则写入报告。"""
    logger.info("📝 生成优化报告...")
    try:
        from src.algorithm_version import get_current_version
        algo_version = get_current_version()
    except Exception:
        algo_version = "—"
    report = {
        'timestamp': datetime.now().isoformat(),
        'algorithm_version': algo_version,
        'profitability': profitability,
        'strategy_performance': performance,
        'optimal_parameters': optimal_params,
        'recommendations': []
    }
    
    # 生成优化建议
    if profitability:
        if profitability['win_rate'] < 50:
            report['recommendations'].append({
                'priority': 'high',
                'issue': '胜率过低',
                'suggestion': '需要优化策略参数或改进策略逻辑'
            })
        
        if profitability['average_profit'] < 0:
            report['recommendations'].append({
                'priority': 'high',
                'issue': '平均收益为负',
                'suggestion': '需要重新评估策略有效性'
            })
    
    # 效果数据来源说明（避免「每天在干啥」说不清）
    data_sources = []
    data_sources.append("**DEMO 实盘收益率须以老虎后台为准**，未核对上的数据不得作为实盘收益率。见 docs/DEMO实盘收益率_定义与数据来源.md")
    data_sources.append("**日志与老虎后台不一致说明**：日志含模拟单与失败单，只有 mode=real 且 status=success 的才在老虎后台；核对规则为「DEMO 的单在老虎都能查到即通过」，老虎可更多（含人工单）。")
    if profitability and profitability.get('total_trades'):
        data_sources.append("收益率/胜率：来自 API 历史订单（老虎后台可核对）")
    else:
        data_sources.append("收益率/胜率：API 历史订单 暂无或未解析（原因与后台订单来源见 [为何报告里后台核对数据为空_说明](../为何报告里后台核对数据为空_说明.md)）")
    perf = report.get('strategy_performance') or {}
    if perf.get('moe_transformer', {}).get('demo_order_success') or perf.get('moe_transformer', {}).get('demo_logs_scanned'):
        data_sources.append("DEMO：多日志汇总（同次运行，四策略共用统计；订单成功、止损止盈等）")
    else:
        data_sources.append("DEMO：仅订单/日志计数（未发现 demo_*.log 时为空）")
    if optimal_params or (perf and any((perf.get(s) or {}).get('num_trades') not in (None, '—') for s in ('grid', 'boll', 'moe_transformer', 'lstm'))):
        data_sources.append("回测：grid/BOLL 为参数网格回测，moe_transformer/lstm 为 test.csv 信号回测（scripts/backtest_model_strategies.py），产出 num_trades/return_pct/win_rate")
    else:
        data_sources.append("回测未运行或 缺 data/processed/test.csv，无效果数据")
    report['data_sources'] = data_sources

    reports_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'docs', 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    with open(os.path.join(reports_dir, 'algorithm_optimization_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str, ensure_ascii=False)

    # 生成Markdown报告
    with open(os.path.join(reports_dir, 'algorithm_optimization_report.md'), 'w', encoding='utf-8') as f:
        f.write("# 算法优化和收益率分析报告\n\n")
        f.write(f"生成时间: {report['timestamp']}\n\n")
        f.write(f"**算法版本**: {report.get('algorithm_version', '—')}（重大变更见 [algorithm_versions.md](../algorithm_versions.md)）\n\n")
        f.write("## 效果数据来源（本次例行用了啥）\n\n")
        for line in report.get('data_sources', data_sources):
            f.write(f"- {line}\n")
        f.write("\n**数据来源与指标含义**（return_pct、num_trades、win_rate、demo_sl_tp_log、demo_execute_buy_calls 等）见 [每日例行_效果数据说明](../每日例行_效果数据说明.md)、[需求分析和Feature测试设计](../需求分析和Feature测试设计.md) 附录。\n\n")
        # 本报告空项根因说明（必须写清为何空，不能留白让用户猜）
        f.write("## 本报告空项根因说明\n\n")
        f.write("以下为空时，**原因均写明**，便于追根问底、不忽悠。\n\n")
        _ver_reason = "无（本次已从老虎拉取到订单并解析）" if (profitability and profitability.get('total_trades')) else (backend_empty_reason or "未记录拉取步骤，无法给出根因")
        f.write("- **实际收益率（老虎后台核对）为空** → 原因：%s\n" % _ver_reason)
        f.write("- **推算收益率（未核对）为空** → 原因：order_log 仅有主单/止损单/止盈单的**提交记录**，无每笔「最后是止损还是止盈」及**平仓价**，无法在无老虎成交或日志平仓记录的前提下做可信推算；若将来实现须与老虎对账验证。详见 [回溯_执行失败为何出现收益率与推算收益率](../回溯_执行失败为何出现收益率与推算收益率.md)。\n\n")
        f.write("## 日志与老虎后台差异说明（必读）\n\n")
        f.write("系统日志（order_log、DEMO 运行日志）记录的是**本进程的每次下单尝试与结果**，包含：模拟单（未发老虎）、真实但被拒单、真实且成功单。**只有「mode=real 且 status=success」的才会在老虎后台出现**，故日志条数/内容与老虎后台不一致是正常现象。DEMO 实盘收益率须以老虎后台为准；核对规则：**DEMO 运行的单在老虎后台都能查到就算通过**，老虎后台可以更多（含人工单）。详见 [DEMO实盘收益率_定义与数据来源](../DEMO实盘收益率_定义与数据来源.md)、[order_log_analysis](order_log_analysis.md)。\n\n")
        f.write("**执行失败（含 API 被拒）**：发了 API 被拒属于**执行失败**，状态页与订单日志分析中会体现「成功 N 笔、失败（含API被拒）M 笔」。**若多为失败则不应有实盘收益率**；今日收益率仅来自老虎后台成交，执行失败时无实盘收益。\n\n")
        # 后台订单与持仓分析（有仓无止损止盈）
        if backend_analysis and (backend_analysis.get("alert_message") or backend_analysis.get("detail_lines")):
            f.write("## 后台订单与持仓分析\n\n")
            f.write("从后台拉取订单（及可选持仓）后的分析结果，用于发现**有持仓但止损/止盈单已撤**（可能保证金不足或强平）等情况。详见 [后台订单与持仓分析_有仓无止损止盈_风控改进](../后台订单与持仓分析_有仓无止损止盈_风控改进.md)。\n\n")
            f.write("- 持仓数量（来源）: %s 手（%s）\n" % (backend_analysis.get("position_qty", 0), backend_analysis.get("position_source", "—")))
            f.write("- 未撤止损/止盈单: %s 笔；已撤/拒/过期: %s 笔\n" % (backend_analysis.get("open_sl_tp_count", 0), backend_analysis.get("cancelled_sl_tp_count", 0)))
            if backend_analysis.get("alert_message"):
                f.write("\n**⚠️ 风控建议**: %s\n\n" % backend_analysis["alert_message"])
        # 实盘：实际（老虎核对）与推算（未核对）
        try:
            y_path = os.path.join(os.path.dirname(reports_dir), 'today_yield.json')
            if os.path.isfile(y_path):
                with open(y_path, 'r', encoding='utf-8') as yf:
                    y = json.load(yf)
                ysrc = y.get('source') or 'none'
                yp = (y.get('yield_pct') or y.get('yield_note') or '—').strip() or '—'
                f.write("## 实盘收益率：实际（老虎核对）与推算（未核对）\n\n")
                f.write("| 项目 | 值 | 说明 |\n")
                f.write("| --- | --- | --- |\n")
                _ver = yp if ysrc in ('tiger_backend', 'report') and yp != '—' else '—'
                if ysrc == 'none' and _ver == '—':
                    _ver = '—（原因见上方「本报告空项根因说明」）'
                _est = (yp + '（未核对）') if ysrc == 'none' and yp != '—' else '—'
                if _est == '—':
                    _est = '—（原因见上方「本报告空项根因说明」）'
                f.write("| **实际收益率（老虎后台核对）** | %s | 仅老虎后台订单/成交计算；未拉取或未核对时见上方空项根因。 |\n" % _ver)
                f.write("| **推算收益率（未核对）** | %s | 未与老虎核对时的推算值；无推算时见上方空项根因。 |\n" % _est)
                f.write("\n")
        except Exception:
            pass
        if profitability:
            f.write("## 收益率分析（API 订单解析）\n\n")
            f.write(f"- 总交易数: {profitability['total_trades']}\n")
            f.write(f"- **实盘胜率**: {profitability['win_rate']:.2f}%\n")
            f.write(f"- 平均收益: {profitability['average_profit']:.2f}\n\n")

        f.write("## 回测与实盘差异说明（必读）\n\n")
        f.write("### 回测和实盘应一致才有参考意义\n\n")
        f.write("**原则**：回测与实盘**仅允许数据来源不同**（回测用历史切片，实盘用实时行情）；**策略逻辑、运行过程（调用频率、开平仓规则）必须完全一致**，否则回测没有参考意义。\n\n")
        f.write("**回测数据量**：历史区间可以很长，回测可用的数据量、可产生的交易次数通常**不少于**实盘某一段的运行；若出现回测笔数远少于实盘，说明回测与实盘**不一致**（例如回测用的是另一套逻辑、或调用方式不同），需要对齐，而不是用「数据粒度」「时间跨度」等借口。\n\n")
        f.write("### 回测是否做多+做空（双向）？\n\n")
        f.write("**设计**：算法为**双向交易**，可做多与做空。\n")
        f.write("- **grid/boll**：`parameter_grid_search` 回测内已含 long 与 short 信号，**符合双向设计**。\n")
        f.write("- **moe_transformer/lstm**：`backtest_model_strategies.py` 已按**双向**实现：信号 1=做多/平空，信号 2=做空/平多，**符合双向设计**。\n\n")
        f.write("### 为何 grid/boll 回测有时 1 笔有时 48 笔？\n\n")
        f.write("**原因**：当前「最优参数」的选取规则是**优先选 num_trades≥2 的组合，再按收益排序**（见 `scripts/parameter_grid_search.py`）。之前按纯收益排序时，最优的那组参数只产生 1 笔交易；改规则后，展示的是「至少 2 笔里收益最高」的那组，所以变成 48 笔。若需改回「只按收益排序」可改回排序逻辑。\n\n")
        f.write("### 回测与实盘代码是否一致？\n\n")
        f.write("**grid/boll**：当前回测用的是 **parameter_grid_search**（RSI + 均线 参数网格），与实盘 **grid_trading_strategy_pro1**（网格上下轨 + ATR + RSI + 反弹/量能）**不是同一套逻辑**。要回测有参考意义，需改代码：让回测调用与实盘同一套规则，或直接使用实盘同逻辑回测入口 **`src.tiger1.backtest_grid_trading_strategy_pro1`**。\n\n")
        f.write("**moe_transformer/lstm**：回测用 test.csv 的 label/衍生信号模拟多空开平，与实盘模型预测输出是否一致取决于训练与推理是否同源。\n\n")
        perf = report.get('strategy_performance') or {}
        if any((perf.get(s) or {}).get('num_trades') not in (None, '—') for s in ('grid', 'boll', 'moe_transformer', 'lstm')):
            f.write("## 回测明细（实际成交笔数、总收益、单笔平均、单笔TOP）\n\n")
            f.write("| 策略 | 成交笔数 | 总收益率% | 单笔平均% | 单笔TOP% | 胜率% |\n")
            f.write("| --- | --- | --- | --- | --- | --- |\n")
            def _num(v):
                if v is None or v == '—': return '—'
                if isinstance(v, (int, float)): return str(round(float(v), 2))
                return str(v)
            for s in ('moe_transformer', 'lstm', 'grid', 'boll'):
                row = perf.get(s) or {}
                f.write("| %s | %s | %s | %s | %s | %s |\n" % (s, _num(row.get('num_trades')), _num(row.get('return_pct')), _num(row.get('avg_per_trade_pct')), _num(row.get('top_per_trade_pct')), _num(row.get('win_rate'))))
            f.write("\n说明：**成交笔数**=实际完成的开平仓次数；**总收益率**= (期末资金−10万)/10万×100；**单笔平均**= 总收益/笔数（每笔占初始资金%）；**单笔TOP**= 单笔最大收益占初始资金%。这样看到「1 笔」即表示本轮回测只成交 1 笔，不是算法只开一仓。\n\n")
        
        f.write("## 指标说明（含义与计算方式）\n\n")
        f.write("| 指标 | 含义 | 计算方式 / 说明 |\n")
        f.write("| --- | --- | --- |\n")
        f.write("| return_pct | 回测收益率 | (期末资金−10万)/10万×100。回测表。 |\n")
        f.write("| win_rate | 回测/实盘胜率 | 回测表=盈利笔数/完成笔数×100；实盘表=仅 API 历史订单解析。 |\n")
        f.write("| num_trades | 回测笔数 | 完成的开平仓次数。 |\n")
        f.write("| yield_verified | 实际收益率（老虎核对） | 老虎后台订单/成交计算；未核对时为 —。 |\n")
        f.write("| yield_estimated | 推算收益率（未核对） | 未核对时的推算值。 |\n")
        f.write("| demo_* | DEMO 日志统计 | 主单成功、止损止盈条数等为日志匹配次数，非老虎后台。详见 [每日例行_效果数据说明](../每日例行_效果数据说明.md)。 |\n")
        f.write("\n")
        if optimal_params:
            f.write("## 优化后的参数\n\n")
            perf = report.get('strategy_performance') or {}
            for strategy, params in optimal_params.items():
                f.write(f"### {strategy}\n\n")
                f.write(f"```json\n{json.dumps(params, indent=2)}\n```\n\n")
            nt_list = []
            for s in optimal_params:
                if isinstance(perf.get(s), dict):
                    nt = perf[s].get('num_trades')
                    if isinstance(nt, (int, float)) and nt <= 1:
                        nt_list.append(nt)
            if nt_list:
                f.write("**回测胜率说明**：本次回测部分策略成交笔数≤1，此时胜率 100% 或 0% 无参考意义，**非算法假定 100% 胜率**；回测逻辑会既有止损也有止盈，多笔时胜率会正常。详见 [回溯_执行失败为何出现收益率与推算收益率](../回溯_执行失败为何出现收益率与推算收益率.md)。\n\n")
        
        if report['recommendations']:
            f.write("## 优化建议\n\n")
            for i, rec in enumerate(report['recommendations'], 1):
                f.write(f"{i}. **{rec['issue']}** ({rec['priority']}优先级)\n")
                f.write(f"   - {rec['suggestion']}\n\n")
    
    logger.info("✅ 优化报告已生成")

    # 报告自检：回测仅 1 笔时应有胜率说明；无 API 时数据来源应标明
    reports_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'docs', 'reports')
    md_path = os.path.join(reports_dir, 'algorithm_optimization_report.md')
    if os.path.exists(md_path):
        with open(md_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
        algo_warnings = []
        perf = report.get('strategy_performance') or {}
        has_single_trade = any(
            isinstance((perf.get(s) or {}).get('num_trades'), (int, float)) and (perf.get(s) or {}).get('num_trades') <= 1
            for s in ('grid', 'boll')
        )
        if has_single_trade and '回测胜率说明' not in md_content and '无参考意义' not in md_content:
            algo_warnings.append("回测存在 num_trades≤1 但报告中未含「回测胜率说明」，易误导。")
        if not (report.get('profitability') and report['profitability'].get('total_trades')):
            if 'API 历史订单 暂无' not in md_content and '暂无或未解析' not in md_content:
                algo_warnings.append("无 API 订单数据但报告未标明「API 历史订单 暂无」。")
        if algo_warnings:
            for w in algo_warnings:
                logger.warning("报告自检: %s", w)
        else:
            logger.info("报告自检: 通过（数据来源与回测说明符合预期）")
    return report


def run_optimization_workflow():
    """运行优化工作流程：结果分析（收益率+策略表现）+ 算法优化（参数回测）+ 报告。"""
    logger.info("="*70)
    logger.info("🚀 开始算法优化和收益率分析")
    logger.info("="*70)
    
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # 1. 加载历史交易记录（API 订单 → 若有则算收益率）
    orders, backend_empty_reason = load_trading_history()
    
    # 1.5 后台订单与持仓分析：有仓无止损止盈检测（可能保证金不足被撤/强平）
    positions, _pos_reason = _fetch_positions_from_tiger_direct()
    backend_analysis = analyze_backend_orders_and_positions(orders, positions)
    try:
        report_dir = os.path.join(root, "docs", "reports")
        os.makedirs(report_dir, exist_ok=True)
        analysis_path = os.path.join(report_dir, "backend_positions_analysis.md")
        with open(analysis_path, "w", encoding="utf-8") as f:
            f.write("# 后台订单与持仓分析\n\n生成时间: %s\n\n" % datetime.now().isoformat())
            f.write("- **持仓数量（来源）**: %s 手（%s）\n" % (backend_analysis["position_qty"], backend_analysis["position_source"]))
            f.write("- **未撤止损/止盈单**: %s 笔\n" % backend_analysis["open_sl_tp_count"])
            f.write("- **已撤/拒/过期止损止盈单**: %s 笔\n\n" % backend_analysis["cancelled_sl_tp_count"])
            for line in backend_analysis["detail_lines"]:
                f.write("- %s\n" % line)
            if backend_analysis.get("alert_message"):
                f.write("\n## 风控建议\n\n%s\n" % backend_analysis["alert_message"])
        logger.info("✅ 后台订单与持仓分析已写入 %s", analysis_path)
    except Exception as e:
        logger.debug("写入 backend_positions_analysis.md 失败: %s", e)
    
    # 2. 计算收益率（依赖订单解析，暂无则 profitability 为 None）
    profitability = calculate_profitability(orders)
    # 有订单但无成交/未解析时，补全根因供报告写入
    if orders and (not profitability or not profitability.get('total_trades')):
        backend_empty_reason = (backend_empty_reason or "").strip() or "已拉取到订单但无已成交（FILLED）笔或无法解析 realized_pnl"
    
    # 3. 分析策略表现（DEMO 多日志汇总 + today_yield）
    performance = analyze_strategy_performance()
    
    # 3.5 缺 test.csv 则先尝试生成（解决问题再继续），再回测
    ensure_test_csv(root)
    
    # 4. 优化参数（网格/BOLL 回测，产出最优参数与回测效果）
    optimal_params, backtest_metrics = optimize_parameters()
    # 把回测效果写入 strategy_performance；失败也写占位，保证每日数据完整（错误即数据）
    if performance:
        for name in ('grid', 'boll', 'moe_transformer', 'lstm'):
            if name not in performance:
                continue
            m = (backtest_metrics or {}).get(name)
            if m and isinstance(m, dict):
                performance[name]['return_pct'] = m.get('return_pct') if m.get('return_pct') is not None else '—'
                performance[name]['win_rate'] = m.get('win_rate') if m.get('win_rate') is not None else '—'
                performance[name]['num_trades'] = m.get('num_trades') if m.get('num_trades') is not None else '—'
                performance[name]['avg_per_trade_pct'] = m.get('avg_per_trade_pct') if m.get('avg_per_trade_pct') is not None else '—'
                performance[name]['top_per_trade_pct'] = m.get('top_per_trade_pct') if m.get('top_per_trade_pct') is not None else '—'
            else:
                performance[name]['return_pct'] = '—'
                performance[name]['win_rate'] = '—'
                performance[name]['num_trades'] = '—'
                performance[name]['avg_per_trade_pct'] = '—'
                performance[name]['top_per_trade_pct'] = '—'
    
    # 5. 生成报告（含效果数据来源说明、后台订单与持仓分析）
    report = generate_optimization_report(profitability, performance, optimal_params, backend_empty_reason=backend_empty_reason, backend_analysis=backend_analysis)
    
    # 5.5 更新今日收益率（写入 docs/today_yield.json），策略报告中的「今日收益率」才不全为 —
    try:
        import subprocess
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(__file__), 'update_today_yield_for_status.py')],
            cwd=root,
            check=False,
        )
    except Exception as e:
        logger.debug("update_today_yield_for_status 未执行: %s", e)

    # 5.6 订单执行状态（成功/失败含API被拒）写入 docs/order_execution_status.json，供状态页展示
    try:
        import subprocess
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(__file__), 'export_order_log_and_analyze.py')],
            cwd=root,
            check=False,
        )
    except Exception as e:
        logger.debug("export_order_log_and_analyze 未执行: %s", e)

    # 6. 生成各策略算法说明与运行效果报告（含对比），供 STATUS 页链接、每日刷新
    try:
        import subprocess
        subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(__file__), 'generate_strategy_reports.py')],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            check=False,
        )
    except Exception as e:
        logger.warning("⚠️ 策略报告生成未执行: %s", e)
    
    # 7. 报告自检：列问题与修复建议，未通过则 exit(1) 便于下一轮继续「例行→解决→优化」
    run_report_self_check(
        profitability=profitability,
        backtest_metrics=backtest_metrics,
        backend_empty_reason=backend_empty_reason,
        root=root,
    )
    
    logger.info("="*70)
    logger.info("✅ 算法优化和收益率分析完成")
    logger.info("报告自检须到网页上查看，本地不算。push 后等待 GitHub Pages 部署完成，再打开部署后的 status 与报告页核对。")
    logger.info("="*70)
    
    return report


def run_report_self_check(profitability, backtest_metrics, backend_empty_reason, root):
    """报告自检：列问题与修复建议。未通过则 exit(1)，便于下一轮继续例行→解决问题→优化收益率。"""
    issues = []
    test_csv = os.path.join(root, 'data', 'processed', 'test.csv')
    has_test_csv = os.path.isfile(test_csv)
    
    # 实盘数据空
    if not (profitability and profitability.get('total_trades')):
        reason = (backend_empty_reason or "").strip() or "未记录"
        issues.append({
            "item": "实盘数据（老虎后台核对）为空",
            "suggestion": "须在有 openapicfg_dem 且账户有成交的环境执行本脚本，才能拉取订单并产出实盘收益率；报告内已写明根因。",
        })
    
    # 回测空：缺 test.csv 或四策略无数据
    strategies = ('grid', 'boll', 'moe_transformer', 'lstm')
    backtest_empty = not backtest_metrics or not any(
        (backtest_metrics.get(s) or {}).get('num_trades') not in (None, '—')
        for s in strategies
    )
    if backtest_empty:
        if not has_test_csv:
            issues.append({
                "item": "回测全空（缺 data/processed/test.csv）",
                "suggestion": "本 run 已尝试自动生成未成功。请先运行: python scripts/merge_recent_data_and_train.py 或 python scripts/data_preprocessing.py，再重新执行本脚本。",
            })
        else:
            issues.append({
                "item": "回测四策略无有效数据（test.csv 已存在但回测未产出）",
                "suggestion": "检查 parameter_grid_search / backtest_model_strategies 是否异常；或 test.csv 行数/列是否满足回测要求（如 close、足够行数）。",
            })
    
    if not issues:
        logger.info("报告自检: 通过（实盘或回测有数据；无数据项已写明根因）")
        return
    
    logger.warning("报告自检: 未通过（以下问题需解决，下一轮继续例行→解决→优化收益率）")
    for i, x in enumerate(issues, 1):
        logger.warning("  %d. %s", i, x["item"])
        logger.warning("     → %s", x["suggestion"])
    logger.warning("例行工作目标=提升收益率；有问题要解决、解决后继续，一直干到问题都解决。")
    sys.exit(1)


if __name__ == '__main__':
    run_optimization_workflow()
