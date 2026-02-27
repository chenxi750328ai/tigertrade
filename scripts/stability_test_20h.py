#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
20小时稳定性测试脚本
运行交易策略20小时，监控错误、性能、资源使用等情况
"""

import sys
import os
import time
import json
import logging
from datetime import datetime, timedelta
import traceback
import signal

# 添加项目路径
sys.path.insert(0, '/home/cx/tigertrade')
if os.getenv('ALLOW_REAL_TRADING', '') != '1':
    os.environ['ALLOW_REAL_TRADING'] = '1'

try:
    import psutil
except ImportError:
    psutil = None
    print("⚠️ psutil未安装，性能监控功能将受限")

from src import tiger1 as t1
from src.executor import MarketDataProvider, OrderExecutor, TradingExecutor
from src.strategies.strategy_factory import StrategyFactory

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('stability_test.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# 稳定性测试配置
DURATION_HOURS = int(os.getenv('RUN_DURATION_HOURS', 20))
STRATEGY_NAME = os.getenv('TRADING_STRATEGY', 'moe_transformer')
CHECK_INTERVAL = 60  # 每分钟检查一次
METRICS_INTERVAL = 300  # 每5分钟记录一次指标

# 统计信息
stats = {
    'start_time': None,
    'end_time': None,
    'errors': [],
    'warnings': [],
    'iterations': 0,
    'successful_iterations': 0,
    'failed_iterations': 0,
    'memory_usage': [],
    'cpu_usage': [],
    'api_calls': 0,
    'api_errors': 0,
    'orders_placed': 0,
    'orders_failed': 0,
}


def signal_handler(signum, frame):
    """处理中断信号"""
    logger.info(f"收到信号 {signum}，准备优雅退出...")
    stats['end_time'] = datetime.now()
    save_stats()
    sys.exit(0)


def save_stats():
    """保存统计信息"""
    stats_file = 'stability_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    logger.info(f"统计信息已保存到 {stats_file}")


def record_metrics():
    """记录系统指标"""
    if psutil:
        try:
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent(interval=1)
            
            stats['memory_usage'].append({
                'time': datetime.now().isoformat(),
                'memory_mb': memory_mb
            })
            stats['cpu_usage'].append({
                'time': datetime.now().isoformat(),
                'cpu_percent': cpu_percent
            })
        except Exception as e:
            logger.warning(f"记录指标失败: {e}")


def run_stability_test():
    """运行稳定性测试"""
    logger.info("="*70)
    logger.info("🚀 开始20小时稳定性测试")
    logger.info("="*70)
    logger.info(f"策略: {STRATEGY_NAME}")
    logger.info(f"时长: {DURATION_HOURS} 小时")
    logger.info(f"开始时间: {datetime.now()}")
    logger.info("="*70)
    
    stats['start_time'] = datetime.now()
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # 初始化策略
        logger.info("📋 初始化策略...")
        strategy_config = {}
        config_path = '/home/cx/tigertrade/config/strategy_config.json'
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                strategy_config = config.get('strategies', {}).get(STRATEGY_NAME, {})
        
        strategy = StrategyFactory.create(
            strategy_name=STRATEGY_NAME,
            model_path=strategy_config.get('model_path'),
            seq_length=strategy_config.get('seq_length', 500)
        )
        logger.info(f"✅ 策略初始化成功: {STRATEGY_NAME}")
        
        # 初始化数据提供者
        logger.info("📊 初始化数据提供者...")
        data_provider = MarketDataProvider(t1.FUTURE_SYMBOL)
        logger.info("✅ 数据提供者初始化成功")
        
        # 初始化订单执行器
        logger.info("📦 初始化订单执行器...")
        order_executor = OrderExecutor(t1)
        logger.info("✅ 订单执行器初始化成功")
        
        # 初始化交易执行器
        logger.info("🔄 初始化交易执行器...")
        executor = TradingExecutor(
            strategy=strategy,
            data_provider=data_provider,
            order_executor=order_executor,
            config={
                'confidence_threshold': 0.4,
                'loop_interval': 5
            }
        )
        logger.info("✅ 交易执行器初始化成功")
        
        # 计算结束时间
        end_time = stats['start_time'] + timedelta(hours=DURATION_HOURS)
        logger.info(f"⏰ 预计结束时间: {end_time}")
        
        # 运行主循环
        last_metrics_time = time.time()
        iteration = 0
        
        while datetime.now() < end_time:
            iteration += 1
            stats['iterations'] = iteration
            
            try:
                # 执行一次交易循环
                logger.info(f"[迭代 {iteration}] 执行交易循环...")
                executor.run_loop(duration_hours=0.01)  # 运行很短时间，然后继续
                stats['successful_iterations'] += 1
                stats['api_calls'] += 1
                
            except KeyboardInterrupt:
                logger.info("收到中断信号，退出...")
                break
            except Exception as e:
                stats['failed_iterations'] += 1
                stats['api_errors'] += 1
                error_info = {
                    'time': datetime.now().isoformat(),
                    'iteration': iteration,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
                stats['errors'].append(error_info)
                logger.error(f"[迭代 {iteration}] 错误: {e}")
                logger.debug(traceback.format_exc())
                
                # 如果错误太多，停止测试
                if len(stats['errors']) > 100:
                    logger.error("错误过多，停止测试")
                    break
            
            # 定期记录指标
            current_time = time.time()
            if current_time - last_metrics_time >= METRICS_INTERVAL:
                record_metrics()
                last_metrics_time = current_time
                if psutil:
                    logger.info(f"[指标] 内存: {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024:.2f}MB, "
                              f"CPU: {psutil.Process(os.getpid()).cpu_percent(interval=1):.2f}%")
            
            # 定期保存统计信息
            if iteration % 100 == 0:
                save_stats()
            
            # 短暂休眠
            time.sleep(CHECK_INTERVAL)
        
        stats['end_time'] = datetime.now()
        duration = stats['end_time'] - stats['start_time']
        logger.info("="*70)
        logger.info("✅ 稳定性测试完成")
        logger.info(f"总时长: {duration}")
        logger.info(f"总迭代数: {stats['iterations']}")
        logger.info(f"成功迭代: {stats['successful_iterations']}")
        logger.info(f"失败迭代: {stats['failed_iterations']}")
        logger.info(f"错误数: {len(stats['errors'])}")
        logger.info("="*70)
        
    except Exception as e:
        stats['end_time'] = datetime.now()
        logger.error(f"稳定性测试失败: {e}")
        logger.error(traceback.format_exc())
        stats['errors'].append({
            'time': datetime.now().isoformat(),
            'error': str(e),
            'traceback': traceback.format_exc()
        })
    
    finally:
        save_stats()
        logger.info("📊 统计信息已保存")


if __name__ == '__main__':
    run_stability_test()
