#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
每日数据收集和训练流程
1. 收集新的交易数据
2. 数据预处理
3. 训练模型
4. 评估模型性能
5. 优化算法参数
"""

import sys
import os
import json
import logging
from datetime import datetime, timedelta
import traceback

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src import tiger1 as t1
from src.api_adapter import api_manager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('daily_training.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def collect_new_data():
    """收集新的交易数据"""
    logger.info("="*70)
    logger.info("📊 开始收集新的交易数据")
    logger.info("="*70)
    
    try:
        # 初始化API（使用DEMO账户）
        if api_manager.is_mock_mode:
            logger.info("⚠️ 当前为Mock模式，切换到真实API...")
            from tigeropen.tiger_open_config import TigerOpenClientConfig
            from tigeropen.quote.quote_client import QuoteClient
            from tigeropen.trade.trade_client import TradeClient
            
            client_config = TigerOpenClientConfig(props_path='./openapicfg_dem')
            quote_client = QuoteClient(client_config)
            trade_client = TradeClient(client_config)
            api_manager.initialize_real_apis(quote_client, trade_client, account=client_config.account)
        
        # 收集Tick数据
        logger.info("📈 收集Tick数据...")
        try:
            from src.tick_data_collector import TickDataCollector
            collector = TickDataCollector()
            # TickDataCollector可能使用run()方法或start()方法
            # 这里先尝试运行一段时间收集数据
            import threading
            import time
            collector_thread = threading.Thread(target=collector.run, daemon=True)
            collector_thread.start()
            time.sleep(60)  # 运行60秒收集数据
            tick_count = "N/A"  # 实际数量需要从collector获取
            logger.info(f"✅ Tick数据收集已启动")
        except Exception as e:
            logger.warning(f"⚠️ Tick数据收集失败: {e}，继续使用K线数据")
            tick_count = 0
        
        # 收集K线数据
        logger.info("📊 收集K线数据...")
        kline_data = t1.get_kline_data(
            t1.FUTURE_SYMBOL,
            t1.BarPeriod.ONE_MINUTE,
            count=1440  # 最近24小时（1440分钟）
        )
        logger.info(f"✅ 收集到 {len(kline_data) if hasattr(kline_data, '__len__') else 'N/A'} 条K线数据")
        
        return {
            'tick_count': tick_count,
            'kline_count': len(kline_data) if hasattr(kline_data, '__len__') else 0,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ 数据收集失败: {e}")
        logger.error(traceback.format_exc())
        return None


def prepare_training_data():
    """准备训练数据（依赖 scripts.prepare_data.prepare_training_data 或 main）。"""
    logger.info("="*70)
    logger.info("🔄 准备训练数据")
    logger.info("="*70)
    
    try:
        from scripts.prepare_data import prepare_training_data as prep_data
        result = prep_data()
        logger.info("✅ 训练数据准备完成")
        return result
    except ImportError:
        try:
            from scripts.prepare_data import main as prep_main
            result = prep_main()
            logger.info("✅ 训练数据准备完成（通过 main）")
            return result
        except Exception as e:
            logger.error(f"❌ 数据准备失败: {e}")
            logger.error(traceback.format_exc())
            return None
    except Exception as e:
        logger.error(f"❌ 数据准备失败: {e}")
        logger.error(traceback.format_exc())
        return None


def train_models():
    """训练模型"""
    logger.info("="*70)
    logger.info("🤖 开始训练模型")
    logger.info("="*70)
    
    training_results = {}
    
    try:
        # 训练Transformer模型
        logger.info("📊 训练Transformer模型...")
        from src.train_raw_features_transformer import train_transformer_model
        
        transformer_result = train_transformer_model()
        training_results['transformer'] = transformer_result
        logger.info("✅ Transformer模型训练完成")
        
        # 训练LSTM模型
        logger.info("📊 训练LSTM模型...")
        # 这里可以添加LSTM训练逻辑
        
        # 训练MoE模型
        logger.info("📊 训练MoE模型...")
        # 这里可以添加MoE训练逻辑
        
        return training_results
        
    except Exception as e:
        logger.error(f"❌ 模型训练失败: {e}")
        logger.error(traceback.format_exc())
        return None


def evaluate_models():
    """评估模型性能：用真实策略表现（DEMO 日志汇总 + 回测 return_pct/win_rate）。"""
    logger.info("="*70)
    logger.info("📈 评估模型性能（真实数据）")
    logger.info("="*70)
    
    try:
        from scripts.optimize_algorithm_and_profitability import (
            analyze_strategy_performance,
            optimize_parameters,
        )
        performance = analyze_strategy_performance()
        optimal_params, backtest_metrics = optimize_parameters()
        evaluation_results = {}
        for sid in performance or {}:
            evaluation_results[sid] = {
                'profitability': performance[sid].get('return_pct', performance[sid].get('profitability', 0)),
                'win_rate': performance[sid].get('win_rate', 0),
                'demo_order_success': performance[sid].get('demo_order_success'),
                'num_trades': performance[sid].get('num_trades'),
                'status': 'evaluated',
            }
        if backtest_metrics:
            logger.info("  回测指标已并入评估: %s", list(backtest_metrics.keys()))
        return evaluation_results
    except Exception as e:
        logger.error(f"❌ 模型评估失败: {e}")
        logger.error(traceback.format_exc())
        return None


def optimize_algorithm():
    """优化算法参数：真实回测 grid/boll，产出最优参数与报告。"""
    logger.info("="*70)
    logger.info("⚙️ 优化算法参数（回测）")
    logger.info("="*70)
    
    try:
        from scripts.optimize_algorithm_and_profitability import optimize_parameters
        optimal_params, backtest_metrics = optimize_parameters()
        optimization_suggestions = []
        if optimal_params:
            for name, params in optimal_params.items():
                optimization_suggestions.append({
                    'priority': 'medium',
                    'issue': f'{name} 最优参数',
                    'suggestion': f'回测产出: {params}',
                })
        else:
            optimization_suggestions.append({
                'priority': 'low',
                'issue': '无回测数据',
                'suggestion': '需 data/processed/test.csv 后重新运行'
            })
        logger.info("✅ 算法优化完成")
        return optimization_suggestions
    except Exception as e:
        logger.error(f"❌ 算法优化失败: {e}")
        logger.error(traceback.format_exc())
        return None


def analyze_profitability():
    """分析收益率：用 API 历史订单解析；无则标明暂无。"""
    logger.info("="*70)
    logger.info("💰 分析收益率")
    logger.info("="*70)
    
    try:
        from scripts.optimize_algorithm_and_profitability import load_trading_history, calculate_profitability
        orders = load_trading_history()
        profitability_data = calculate_profitability(orders)
        if profitability_data:
            logger.info("✅ 收益率分析完成（API 订单）")
            return profitability_data
        profitability_data = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'average_profit': 0.0,
            'total_profit': 0.0,
            'note': 'API 历史订单暂无或未解析，见 docs/reports/algorithm_optimization_report.md'
        }
        logger.info("✅ 收益率分析完成（暂无订单数据）")
        return profitability_data
    except Exception as e:
        logger.error(f"❌ 收益率分析失败: {e}")
        logger.error(traceback.format_exc())
        return None


def run_daily_workflow():
    """运行每日工作流程"""
    logger.info("="*70)
    logger.info("🚀 开始每日数据收集和训练流程")
    logger.info(f"开始时间: {datetime.now()}")
    logger.info("="*70)
    
    results = {
        'start_time': datetime.now().isoformat(),
        'data_collection': None,
        'training': None,
        'evaluation': None,
        'optimization': None,
        'profitability': None,
        'end_time': None
    }
    
    try:
        # 1. 收集新数据
        data_result = collect_new_data()
        results['data_collection'] = data_result
        
        if not data_result:
            logger.warning("⚠️ 数据收集失败，跳过后续步骤")
            return results
        
        # 2. 准备训练数据
        prep_result = prepare_training_data()
        
        # 3. 训练模型
        training_results = train_models()
        results['training'] = training_results
        
        # 4. 评估模型
        evaluation_results = evaluate_models()
        results['evaluation'] = evaluation_results
        
        # 5. 优化算法
        optimization_results = optimize_algorithm()
        results['optimization'] = optimization_results
        
        # 6. 分析收益率
        profitability_data = analyze_profitability()
        results['profitability'] = profitability_data
        
    except Exception as e:
        logger.error(f"❌ 每日工作流程失败: {e}")
        logger.error(traceback.format_exc())
    
    # 7. 无论前面是否失败，都跑真实优化与报告（回测 + DEMO 汇总 + 报告 + today_yield）
    try:
        from scripts.optimize_algorithm_and_profitability import run_optimization_workflow
        run_optimization_workflow()
        results['optimization_report'] = 'done'
    except Exception as e:
        logger.warning("⚠️ 优化与报告流程未完成: %s", e)
        results['optimization_report'] = str(e)
    
    finally:
        results['end_time'] = datetime.now().isoformat()
        
        # 保存结果
        with open('daily_workflow_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info("="*70)
        logger.info("✅ 每日工作流程完成")
        logger.info(f"结束时间: {datetime.now()}")
        logger.info("="*70)
    
    return results


if __name__ == '__main__':
    run_daily_workflow()
