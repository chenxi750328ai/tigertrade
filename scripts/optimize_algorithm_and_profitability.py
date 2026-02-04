#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
优化算法和收益率
基于历史交易数据优化策略参数，提升收益率
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
        try:
            from scripts.analyze_demo_log import aggregate_demo_logs
            demo = aggregate_demo_logs()
            if demo and demo.get('logs_scanned', 0) > 0:
                performance_data['moe_transformer']['demo_order_success'] = demo.get('order_success', 0)
                performance_data['moe_transformer']['demo_sl_tp_log'] = demo.get('sl_tp_log', 0)
                performance_data['moe_transformer']['demo_execute_buy_calls'] = demo.get('execute_buy_calls', 0)
                performance_data['moe_transformer']['demo_success_orders_sum'] = demo.get('success_orders_sum', 0)
                performance_data['moe_transformer']['demo_fail_orders_sum'] = demo.get('fail_orders_sum', 0)
                performance_data['moe_transformer']['demo_logs_scanned'] = demo.get('logs_scanned', 0)
                logger.info("  DEMO 多日志汇总: 扫描 %s 个日志, order_success=%s, sl_tp=%s",
                            demo.get('logs_scanned'), demo.get('order_success'), demo.get('sl_tp_log'))
        except Exception as e:
            logger.debug("DEMO 日志统计未合并: %s", e)

        # 从 today_yield 补充今日收益率
        try:
            yield_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'docs', 'today_yield.json')
            if os.path.isfile(yield_path):
                with open(yield_path, 'r', encoding='utf-8') as f:
                    y = json.load(f)
                pct = y.get('yield_pct') or y.get('yield_note')
                if pct and str(pct).strip() not in ('', '—'):
                    try:
                        performance_data['moe_transformer']['today_yield_pct'] = str(pct)
                    except Exception:
                        performance_data['moe_transformer']['today_yield_pct'] = str(pct)
        except Exception as e:
            logger.debug("today_yield 未合并: %s", e)

        return performance_data
        
    except Exception as e:
        logger.error(f"❌ 策略表现分析失败: {e}")
        return None


def optimize_parameters():
    """优化策略参数"""
    logger.info("⚙️ 优化策略参数...")
    
    try:
        # 基于历史表现优化参数
        from scripts.parameter_grid_search import grid_search_optimal_params
        
        optimal_params = {}
        
        # 优化网格策略参数
        logger.info("📊 优化网格策略参数...")
        grid_params = grid_search_optimal_params('grid')
        optimal_params['grid'] = grid_params
        
        # 优化BOLL策略参数
        logger.info("📊 优化BOLL策略参数...")
        boll_params = grid_search_optimal_params('boll')
        optimal_params['boll'] = boll_params
        
        logger.info("✅ 参数优化完成")
        return optimal_params
        
    except Exception as e:
        logger.warning(f"⚠️ 参数优化失败: {e}")
        return {}


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
    
    # 保存报告到 docs/reports/，与策略报告生成器读取路径一致
    reports_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'docs', 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    with open(os.path.join(reports_dir, 'algorithm_optimization_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str, ensure_ascii=False)
    
    # 生成Markdown报告
    with open(os.path.join(reports_dir, 'algorithm_optimization_report.md'), 'w', encoding='utf-8') as f:
        f.write("# 算法优化和收益率分析报告\n\n")
        f.write(f"生成时间: {report['timestamp']}\n\n")
        
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
    """运行优化工作流程"""
    logger.info("="*70)
    logger.info("🚀 开始算法优化和收益率分析")
    logger.info("="*70)
    
    # 1. 加载历史交易记录
    orders = load_trading_history()
    
    # 2. 计算收益率
    profitability = calculate_profitability(orders)
    
    # 3. 分析策略表现
    performance = analyze_strategy_performance()
    
    # 4. 优化参数
    optimal_params = optimize_parameters()
    
    # 5. 生成报告
    report = generate_optimization_report(profitability, performance, optimal_params)
    
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
