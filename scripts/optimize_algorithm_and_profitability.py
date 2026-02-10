#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
优化算法和收益率（每日例行：结果分析 + 算法优化）
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


def load_trading_history():
    """加载历史交易记录"""
    logger.info("📊 加载历史交易记录...")
    
    try:
        # 从API获取历史订单
        from src.api_adapter import api_manager
        
        if api_manager.trade_api and hasattr(api_manager.trade_api, 'get_orders'):
            # 转换symbol格式：SIL.COMEX.202603 -> SIL2603
            from src import tiger1 as t1
            symbol_to_query = t1._to_api_identifier('SIL.COMEX.202603')
            orders = api_manager.trade_api.get_orders(
                account=api_manager._account,
                symbol=symbol_to_query,  # 使用转换后的格式 SIL2603
                limit=1000
            )
            
            if orders:
                logger.info(f"✅ 加载了 {len(orders)} 条历史订单")
                return orders
        
        logger.warning("⚠️ 无法加载历史交易记录，使用模拟数据")
        return []
        
    except Exception as e:
        logger.error(f"❌ 加载历史交易记录失败: {e}")
        return []


def calculate_profitability(orders):
    """计算收益率"""
    logger.info("💰 计算收益率...")
    
    if not orders:
        logger.warning("⚠️ 没有交易记录，无法计算收益率")
        return None
    
    try:
        # 分析订单数据
        total_profit = 0
        total_trades = 0
        winning_trades = 0
        losing_trades = 0
        
        for order in orders:
            # 这里需要根据实际的订单对象结构来解析
            # 假设订单有price, quantity, side等属性
            pass
        
        profitability = {
            'total_profit': total_profit,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': winning_trades / total_trades * 100 if total_trades > 0 else 0,
            'average_profit': total_profit / total_trades if total_trades > 0 else 0
        }
        
        logger.info(f"✅ 收益率计算完成")
        logger.info(f"  总交易数: {profitability['total_trades']}")
        logger.info(f"  胜率: {profitability['win_rate']:.2f}%")
        logger.info(f"  平均收益: {profitability['average_profit']:.2f}")
        
        return profitability
        
    except Exception as e:
        logger.error(f"❌ 计算收益率失败: {e}")
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


def optimize_parameters():
    """
    优化策略参数：对 grid/boll 做网格回测，返回最优参数及回测效果（供报告写入）。
    返回 (optimal_params, backtest_metrics)。backtest_metrics 用于填入 strategy_performance 的收益率/胜率。
    """
    logger.info("⚙️ 优化策略参数（网格/BOLL 回测）...")
    optimal_params = {}
    backtest_metrics = {}
    try:
        from scripts.parameter_grid_search import grid_search_optimal_params
        for name in ('grid', 'boll'):
            try:
                r = grid_search_optimal_params(name)
                if r and isinstance(r, dict):
                    optimal_params[name] = r.get('params', r)
                    if 'return_pct' in r or 'win_rate' in r:
                        backtest_metrics[name] = {
                            'return_pct': r.get('return_pct'),
                            'win_rate': r.get('win_rate'),
                            'num_trades': r.get('num_trades'),
                        }
                        logger.info("  %s 回测: 收益=%.2f%%, 胜率=%.1f%%, 笔数=%s",
                                    name, r.get('return_pct', 0) or 0, r.get('win_rate', 0) or 0, r.get('num_trades'))
            except Exception as e:
                logger.debug("  %s 回测未产出: %s", name, e)
        logger.info("✅ 参数优化完成" if optimal_params else "⚠️ 无回测数据（需 data/processed/test.csv）")
        return optimal_params, backtest_metrics
    except Exception as e:
        logger.warning("⚠️ 参数优化失败: %s", e)
        return {}, {}


def generate_optimization_report(profitability, performance, optimal_params):
    """生成优化报告"""
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
        data_sources.append("收益率/胜率：API 历史订单 暂无或未解析")
    perf = report.get('strategy_performance') or {}
    if perf.get('moe_transformer', {}).get('demo_order_success') or perf.get('moe_transformer', {}).get('demo_logs_scanned'):
        data_sources.append("DEMO：多日志汇总（同次运行，四策略共用统计；订单成功、止损止盈等）")
    else:
        data_sources.append("DEMO：仅订单/日志计数（未发现 demo_*.log 时为空）")
    if optimal_params:
        data_sources.append("网格/BOLL：回测（data/processed/test.csv）产出最优参数与 return_pct/win_rate")
    else:
        data_sources.append("网格/BOLL：回测未运行或 缺 data/processed/test.csv，无效果数据")
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
        f.write("## 日志与老虎后台差异说明（必读）\n\n")
        f.write("系统日志（order_log、DEMO 运行日志）记录的是**本进程的每次下单尝试与结果**，包含：模拟单（未发老虎）、真实但被拒单、真实且成功单。**只有「mode=real 且 status=success」的才会在老虎后台出现**，故日志条数/内容与老虎后台不一致是正常现象。DEMO 实盘收益率须以老虎后台为准；核对规则：**DEMO 运行的单在老虎后台都能查到就算通过**，老虎后台可以更多（含人工单）。详见 [DEMO实盘收益率_定义与数据来源](../DEMO实盘收益率_定义与数据来源.md)、[order_log_analysis](order_log_analysis.md)。\n\n")
        f.write("**执行失败（含 API 被拒）**：发了 API 被拒属于**执行失败**，状态页与订单日志分析中会体现「成功 N 笔、失败（含API被拒）M 笔」。**若多为失败则不应有实盘收益率**；今日收益率仅来自老虎后台成交，执行失败时无实盘收益。\n\n")
        if profitability:
            f.write("## 收益率分析\n\n")
            f.write(f"- 总交易数: {profitability['total_trades']}\n")
            f.write(f"- 胜率: {profitability['win_rate']:.2f}%\n")
            f.write(f"- 平均收益: {profitability['average_profit']:.2f}\n\n")
        
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
    
    # 1. 加载历史交易记录（API 订单 → 若有则算收益率）
    orders = load_trading_history()
    
    # 2. 计算收益率（依赖订单解析，暂无则 profitability 为 None）
    profitability = calculate_profitability(orders)
    
    # 3. 分析策略表现（DEMO 多日志汇总 + today_yield）
    performance = analyze_strategy_performance()
    
    # 4. 优化参数（网格/BOLL 回测，产出最优参数与回测效果）
    optimal_params, backtest_metrics = optimize_parameters()
    # 把回测效果写入 strategy_performance；失败也写占位，保证每日数据完整（错误即数据）
    if performance:
        for name in ('grid', 'boll'):
            if name not in performance:
                continue
            m = (backtest_metrics or {}).get(name)
            if m and isinstance(m, dict):
                performance[name]['return_pct'] = m.get('return_pct') if m.get('return_pct') is not None else '—'
                performance[name]['win_rate'] = m.get('win_rate') if m.get('win_rate') is not None else '—'
                performance[name]['num_trades'] = m.get('num_trades') if m.get('num_trades') is not None else '—'
            else:
                performance[name]['return_pct'] = '—'
                performance[name]['win_rate'] = '—'
                performance[name]['num_trades'] = '—'
        for name in ('moe_transformer', 'lstm'):
            if name in performance and (performance[name].get('return_pct') is None and performance[name].get('num_trades') is None):
                performance[name]['return_pct'] = '—'
                performance[name]['num_trades'] = '—'
    
    # 5. 生成报告（含效果数据来源说明）
    report = generate_optimization_report(profitability, performance, optimal_params)
    
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
    
    logger.info("="*70)
    logger.info("✅ 算法优化和收益率分析完成")
    logger.info("="*70)
    
    return report


if __name__ == '__main__':
    run_optimization_workflow()
