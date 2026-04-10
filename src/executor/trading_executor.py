"""
交易执行器
连接策略和执行的核心模块
"""
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
import os
import pandas as pd
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.strategies.base_strategy import BaseTradingStrategy
from .data_provider import MarketDataProvider
from .order_executor import OrderExecutor

try:
    from src import tiger1 as _t1
except ImportError:
    _t1 = None


class TradingExecutor:
    """交易执行器 - 连接策略和执行的核心模块"""
    
    def __init__(self, 
                 strategy: BaseTradingStrategy,
                 data_provider: MarketDataProvider,
                 order_executor: OrderExecutor,
                 config: Optional[Dict[str, Any]] = None):
        """
        初始化交易执行器
        
        Args:
            strategy: 交易策略实例
            data_provider: 数据提供者
            order_executor: 订单执行器
            config: 配置字典
        """
        self.strategy = strategy
        self.data_provider = data_provider
        self.order_executor = order_executor
        self.config = config or {}
        
        # 统计信息
        self.stats = {
            'total_predictions': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'hold_signals': 0,
            'avg_confidence': 0.0,
            'errors': 0,
            'successful_orders': 0,
            'failed_orders': 0
        }
    
    def run_loop(self, duration_hours: int = 20, start_time: Optional[datetime] = None):
        """
        运行交易循环
        
        Args:
            duration_hours: 运行时长（小时）
            start_time: 开始时间（默认当前时间）
        """
        start_time = start_time or datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)
        
        print("="*70)
        print(f"🔄 开始交易循环...")
        print(f"📅 开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📅 结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  运行时长: {duration_hours} 小时")
        print(f"🤖 使用策略: {self.strategy.strategy_name}")
        print("="*70)
        
        iteration = 0
        while datetime.now() < end_time:
            iteration += 1
            elapsed = datetime.now() - start_time
            remaining = end_time - datetime.now()
            
            try:
                # 显示进度
                if iteration % 10 == 0:
                    self._print_progress(elapsed, remaining)
                
                # 1. 获取市场数据
                market_data = self.data_provider.get_market_data(
                    seq_length=self.strategy.seq_length
                )
                
                # 1.5 持仓看门狗：超时止盈/止损必跑，防止有仓无卖、裸奔爆仓（TradingExecutor 路径原先未调用）
                if _t1 and callable(getattr(_t1, 'run_position_watchdog', None)):
                    tick_price = market_data.get('tick_price') or market_data.get('current_data', {}).get('price_current')
                    atr = market_data.get('atr')
                    grid_lower = market_data.get('grid_lower')
                    if tick_price is not None:
                        _t1.run_position_watchdog(tick_price, atr=atr, grid_lower=grid_lower)
                
                # 2. 调用策略预测
                prediction_result = self.strategy.predict_action(
                    market_data['current_data'], 
                    market_data['historical_data']
                )
                
                # 3. 显示预测结果
                self._print_prediction(prediction_result, market_data)
                
                # 4. 执行交易
                self._execute_prediction(
                    prediction_result,
                    market_data
                )
                
                # 5. 更新统计
                self._update_stats(prediction_result)
                
                # 等待下一次循环
                loop_interval = self.config.get('loop_interval', 5)
                time.sleep(loop_interval)
                
            except KeyboardInterrupt:
                print("\n🛑 用户中断，停止运行")
                break
            except Exception as e:
                self.stats['errors'] += 1
                print(f"❌ 错误: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(5)
        
        # 显示最终统计
        self._print_final_stats()
    
    def _execute_prediction(self, prediction_result: Tuple, market_data: Dict[str, Any]):
        """根据预测结果执行交易"""
        action, confidence, profit_pred = self._parse_prediction(prediction_result)
        
        # 置信度阈值检查
        confidence_threshold = self.config.get('confidence_threshold', 0.4)
        
        if action == 0:
            print(f"   ℹ️ 预测：不操作")
            return
        
        if confidence < confidence_threshold:
            print(f"   ⚠️ 置信度{confidence:.3f}低于阈值{confidence_threshold}，不执行交易")
            return
        
        # 获取必要数据
        tick_price = market_data['tick_price']
        atr = market_data['atr']
        grid_lower = market_data['grid_lower']
        grid_upper = market_data['grid_upper']
        current_data = market_data['current_data']
        
        # 执行交易
        if action == 1:  # 买入
            result, message = self.order_executor.execute_buy(
                tick_price, atr, grid_lower, grid_upper, confidence, profit_pred
            )
            print(f"   {'✅' if result else '⚠️'} [执行买入] {message}")
            if result:
                self.stats['successful_orders'] += 1
            else:
                self.stats['failed_orders'] += 1
            
        elif action == 2:  # 卖出
            result, message = self.order_executor.execute_sell(tick_price, confidence)
            print(f"   {'✅' if result else '⚠️'} [执行卖出] {message}")
            if result:
                self.stats['successful_orders'] += 1
            else:
                self.stats['failed_orders'] += 1
    
    def _parse_prediction(self, prediction_result) -> Tuple[int, float, Optional[float]]:
        """解析预测结果"""
        if isinstance(prediction_result, tuple):
            if len(prediction_result) == 3:
                return prediction_result
            elif len(prediction_result) == 2:
                return (*prediction_result, None)
        return (0, 0.0, None)
    
    def _print_prediction(self, prediction_result: Tuple, market_data: Dict[str, Any]):
        """打印预测结果"""
        action, confidence, profit_pred = self._parse_prediction(prediction_result)
        
        action_map = {0: "不操作", 1: "买入", 2: "卖出"}
        tick_price = market_data['tick_price']
        price_current = market_data['price_current']
        atr = market_data['atr']
        grid_lower = market_data['grid_lower']
        grid_upper = market_data['grid_upper']
        threshold = market_data['current_data']['threshold']
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🧠 {self.strategy.strategy_name}预测:")
        print(f"   动作: {action_map[action]}, 置信度: {confidence:.3f}")
        if profit_pred is not None:
            print(f"   预测收益率: {profit_pred*100:.2f}%")
        print(f"   价格 | Tick={tick_price:.3f}, K线={price_current:.3f}, ATR={atr:.3f}")
        print(f"   网格 | [{grid_lower:.3f}, {grid_upper:.3f}], 阈值={threshold:.3f}")
    
    def _print_progress(self, elapsed: timedelta, remaining: timedelta):
        """打印进度信息"""
        print(f"\n📊 进度更新 [{datetime.now().strftime('%H:%M:%S')}]")
        print(f"   已运行: {elapsed}")
        print(f"   剩余: {remaining}")
        print(f"   预测次数: {self.stats['total_predictions']}")
        print(f"   买入信号: {self.stats['buy_signals']}, 卖出信号: {self.stats['sell_signals']}, 持有: {self.stats['hold_signals']}")
        if self.stats['total_predictions'] > 0:
            print(f"   平均置信度: {self.stats['avg_confidence']:.3f}")
        print(f"   成功订单: {self.stats['successful_orders']}, 失败订单: {self.stats['failed_orders']}")
    
    def _update_stats(self, prediction_result: Tuple):
        """更新统计信息"""
        action, confidence, _ = self._parse_prediction(prediction_result)
        
        self.stats['total_predictions'] += 1
        self.stats['avg_confidence'] = (
            self.stats['avg_confidence'] * (self.stats['total_predictions'] - 1) + confidence
        ) / self.stats['total_predictions']
        
        if action == 1:
            self.stats['buy_signals'] += 1
        elif action == 2:
            self.stats['sell_signals'] += 1
        else:
            self.stats['hold_signals'] += 1
    
    def _print_final_stats(self):
        """打印最终统计"""
        print("\n" + "="*70)
        print("📊 运行统计")
        print("="*70)
        print(f"总预测次数: {self.stats.get('total_predictions', 0)}")
        print(f"买入信号: {self.stats.get('buy_signals', 0)}")
        print(f"卖出信号: {self.stats.get('sell_signals', 0)}")
        print(f"持有信号: {self.stats.get('hold_signals', 0)}")
        avg_conf = self.stats.get('avg_confidence', 0.0)
        print(f"平均置信度: {avg_conf:.3f}")
        print(f"成功订单: {self.stats.get('successful_orders', 0)}")
        print(f"失败订单: {self.stats.get('failed_orders', 0)}")
        print(f"错误次数: {self.stats.get('errors', 0)}")
        print("="*70)
        print("✅ 运行完成！")

    def print_stats(self):
        """打印统计信息（公共方法）"""
        self._print_final_stats()

    def get_stats(self) -> dict:
        """获取统计信息（公共方法）"""
        return self.stats.copy()
