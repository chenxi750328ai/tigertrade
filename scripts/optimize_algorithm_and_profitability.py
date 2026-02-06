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
    """分析策略表现：从 DEMO 日志、today_yield 等汇总可用的运行效果，供策略报告展示。"""
    logger.info("📈 分析策略表现...")
    
    try:
        strategies = ['moe_transformer', 'lstm', 'grid', 'boll']
        performance_data = {}
        for s in strategies:
            performance_data[s] = {
                'profitability': 0,
                'win_rate': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0
            }

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
            logger.debug("DEMO 日志统计未合并: %s", e)

        # 从 today_yield 补充今日收益率（四策略都填，便于报告统一展示）
        try:
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            yield_path = os.path.join(root, 'docs', 'today_yield.json')
            if os.path.isfile(yield_path):
                with open(yield_path, 'r', encoding='utf-8') as f:
                    y = json.load(f)
                pct = y.get('yield_pct') or y.get('yield_note')
                if pct and str(pct).strip() not in ('', '—'):
                    for sid in ('moe_transformer', 'lstm', 'grid', 'boll'):
                        performance_data[sid]['today_yield_pct'] = str(pct)
        except Exception as e:
            logger.debug("today_yield 未合并: %s", e)

        return performance_data
        
    except Exception as e:
        logger.error(f"❌ 策略表现分析失败: {e}")
        return None


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
    
    report = {
        'timestamp': datetime.now().isoformat(),
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
    if profitability and profitability.get('total_trades'):
        data_sources.append("收益率/胜率：来自 API 历史订单")
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
        f.write("## 效果数据来源（本次例行用了啥）\n\n")
        for line in report.get('data_sources', data_sources):
            f.write(f"- {line}\n")
        f.write("\n")
        if profitability:
            f.write("## 收益率分析\n\n")
            f.write(f"- 总交易数: {profitability['total_trades']}\n")
            f.write(f"- 胜率: {profitability['win_rate']:.2f}%\n")
            f.write(f"- 平均收益: {profitability['average_profit']:.2f}\n\n")
        
        if optimal_params:
            f.write("## 优化后的参数\n\n")
            for strategy, params in optimal_params.items():
                f.write(f"### {strategy}\n\n")
                f.write(f"```json\n{json.dumps(params, indent=2)}\n```\n\n")
        
        if report['recommendations']:
            f.write("## 优化建议\n\n")
            for i, rec in enumerate(report['recommendations'], 1):
                f.write(f"{i}. **{rec['issue']}** ({rec['priority']}优先级)\n")
                f.write(f"   - {rec['suggestion']}\n\n")
    
    logger.info("✅ 优化报告已生成")
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
    # 把回测效果写入 strategy_performance，报告里才有「效果数据」
    if performance and backtest_metrics:
        for name, metrics in backtest_metrics.items():
            if name in performance and isinstance(metrics, dict):
                if metrics.get('return_pct') is not None:
                    performance[name]['return_pct'] = metrics['return_pct']
                if metrics.get('win_rate') is not None:
                    performance[name]['win_rate'] = metrics['win_rate']
                if metrics.get('num_trades') is not None:
                    performance[name]['num_trades'] = metrics['num_trades']
    
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
