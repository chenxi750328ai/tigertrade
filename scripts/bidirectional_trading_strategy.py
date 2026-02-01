#!/usr/bin/env python3
"""
双向交易策略实现
实现做多和做空双向交易策略，目标月收益20%
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional
import logging
import time
from datetime import datetime


class BidirectionalTradingStrategy:
    """
    双向交易策略类
    支持做多和做空两种方向的交易
    """
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 leverage: float = 1.0,
                 transaction_cost: float = 0.001,
                 stop_loss_pct: float = 0.05,
                 take_profit_pct: float = 0.10,
                 min_price_atr_ratio: float = 0.5):
        """
        初始化策略参数
        
        Args:
            initial_capital: 初始资金
            leverage: 杠杆倍数
            transaction_cost: 交易成本（手续费等）
            stop_loss_pct: 止损百分比
            take_profit_pct: 止盈百分比
            min_price_atr_ratio: 价格相对于ATR的最小比例，用于过滤低波动率入场
        """
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.transaction_cost = transaction_cost
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.min_price_atr_ratio = min_price_atr_ratio  # 用ATR来过滤低质量信号
        
        # 交易记录
        self.trades_log = []
        self.position_history = []
        
        # 当前状态
        self.current_capital = initial_capital
        self.current_position = 0  # >0为多头，<0为空头，=0为平仓
        self.entry_price = None
        self.position_direction = None  # 'long', 'short', or None
        self.total_return = 0.0
        self.max_drawdown = 0.0
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            添加了技术指标的DataFrame
        """
        df = data.copy()
        
        # 计算移动平均线
        df['sma_short'] = df['close'].rolling(window=10).mean()
        df['sma_long'] = df['close'].rolling(window=30).mean()
        
        # 计算RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 计算布林带
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # 计算ATR（平均真实波幅）
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        df['atr'] = true_range.rolling(window=14).mean()
        
        # 计算MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['signal'] = df['macd'].ewm(span=9).mean()
        
        return df
    
    def should_enter_long(self, row: pd.Series, prev_row: Optional[pd.Series]) -> bool:
        """
        判断是否应该开多仓
        
        Args:
            row: 当前K线数据
            prev_row: 前一根K线数据
            
        Returns:
            是否开多仓
        """
        # 记录详细的判断过程
        price_current = row['close']
        sma_short = row['sma_short']
        sma_long = row['sma_long']
        rsi = row['rsi']
        bb_lower = row['bb_lower']
        atr = row['atr']
        macd = row['macd']
        signal = row['signal']
        
        # 详细日志记录
        self.logger.info(f"LONG ENTRY CHECK:")
        self.logger.info(f"  Current Price: {price_current:.3f}")
        self.logger.info(f"  SMA Short: {sma_short:.3f}")
        self.logger.info(f"  SMA Long: {sma_long:.3f}")
        self.logger.info(f"  RSI: {rsi:.3f}")
        self.logger.info(f"  BB Lower: {bb_lower:.3f}")
        self.logger.info(f"  ATR: {atr:.3f}")
        self.logger.info(f"  MACD: {macd:.3f}")
        self.logger.info(f"  Signal: {signal:.3f}")
        
        # 过滤低波动率市场（ATR太小说明可能在横盘）
        if atr < price_current * 0.01:  # ATR小于价格的1%
            self.logger.info(f"  Filtered: ATR too low ({atr:.3f} < {price_current * 0.01:.3f})")
            return False
        
        # 多头入场条件 (需要满足4个中的3个)
        condition1 = sma_short > sma_long  # 短期均线上穿长期均线
        condition2 = rsi < 40              # RSI处于超卖区域
        condition3 = price_current <= bb_lower + atr * 0.5  # 价格接近布林带下轨
        condition4 = macd > signal  # MACD上穿信号线
        
        # 至少满足4个条件中的3个才入场
        conditions_met = sum([condition1, condition2, condition3, condition4])
        result = conditions_met >= 3
        
        self.logger.info(f"  Condition 1 (SMA short > long): {condition1} ({sma_short:.3f} > {sma_long:.3f})")
        self.logger.info(f"  Condition 2 (RSI < 40): {condition2} ({rsi:.3f} < 40)")
        self.logger.info(f"  Condition 3 (Price near lower BB): {condition3} ({price_current:.3f} <= {bb_lower + atr * 0.5:.3f})")
        self.logger.info(f"  Condition 4 (MACD > signal): {condition4} ({macd:.3f} > {signal:.3f})")
        self.logger.info(f"  Conditions met: {conditions_met}/4")
        self.logger.info(f"  LONG ENTRY RESULT: {result}")
        
        return result
    
    def should_enter_short(self, row: pd.Series, prev_row: Optional[pd.Series]) -> bool:
        """
        判断是否应该开空仓
        
        Args:
            row: 当前K线数据
            prev_row: 前一根K线数据
            
        Returns:
            是否开空仓
        """
        # 记录详细的判断过程
        price_current = row['close']
        sma_short = row['sma_short']
        sma_long = row['sma_long']
        rsi = row['rsi']
        bb_upper = row['bb_upper']
        atr = row['atr']
        macd = row['macd']
        signal = row['signal']
        
        # 详细日志记录
        self.logger.info(f"SHORT ENTRY CHECK:")
        self.logger.info(f"  Current Price: {price_current:.3f}")
        self.logger.info(f"  SMA Short: {sma_short:.3f}")
        self.logger.info(f"  SMA Long: {sma_long:.3f}")
        self.logger.info(f"  RSI: {rsi:.3f}")
        self.logger.info(f"  BB Upper: {bb_upper:.3f}")
        self.logger.info(f"  ATR: {atr:.3f}")
        self.logger.info(f"  MACD: {macd:.3f}")
        self.logger.info(f"  Signal: {signal:.3f}")
        
        # 过滤低波动率市场（ATR太小说明可能在横盘）
        if atr < price_current * 0.01:  # ATR小于价格的1%
            self.logger.info(f"  Filtered: ATR too low ({atr:.3f} < {price_current * 0.01:.3f})")
            return False
        
        # 空头入场条件 (需要满足4个中的3个)
        condition1 = sma_short < sma_long  # 短期均线下穿长期均线
        condition2 = rsi > 60              # RSI处于超买区域
        condition3 = price_current >= bb_upper - atr * 0.5  # 价格接近布林带上轨
        condition4 = macd < signal  # MACD下穿信号线
        
        # 至少满足4个条件中的3个才入场
        conditions_met = sum([condition1, condition2, condition3, condition4])
        result = conditions_met >= 3
        
        self.logger.info(f"  Condition 1 (SMA short < long): {condition1} ({sma_short:.3f} < {sma_long:.3f})")
        self.logger.info(f"  Condition 2 (RSI > 60): {condition2} ({rsi:.3f} > 60)")
        self.logger.info(f"  Condition 3 (Price near upper BB): {condition3} ({price_current:.3f} >= {bb_upper - atr * 0.5:.3f})")
        self.logger.info(f"  Condition 4 (MACD < signal): {condition4} ({macd:.3f} < {signal:.3f})")
        self.logger.info(f"  Conditions met: {conditions_met}/4")
        self.logger.info(f"  SHORT ENTRY RESULT: {result}")
        
        return result
    
    def should_exit_position(self, row: pd.Series) -> Tuple[bool, str]:
        """
        判断是否应该平仓
        
        Args:
            row: 当前K线数据
            
        Returns:
            (是否平仓, 平仓原因)
        """
        if self.position_direction is None or self.entry_price is None:
            return False, "no position"
        
        current_price = row['close']
        price_change_pct = abs(current_price - self.entry_price) / self.entry_price
        
        # 计算止损/止盈价格
        if self.position_direction == 'long':
            stop_loss_price = self.entry_price * (1 - self.stop_loss_pct)
            take_profit_price = self.entry_price * (1 + self.take_profit_pct)
            
            if current_price <= stop_loss_price:
                return True, "stop_loss_long"
            elif current_price >= take_profit_price:
                return True, "take_profit_long"
        elif self.position_direction == 'short':
            stop_loss_price = self.entry_price * (1 + self.stop_loss_pct)
            take_profit_price = self.entry_price * (1 - self.take_profit_pct)
            
            if current_price >= stop_loss_price:
                return True, "stop_loss_short"
            elif current_price <= take_profit_price:
                return True, "take_profit_short"
        
        # 检查反向信号
        if self.position_direction == 'long' and self.should_enter_short(row, None):
            return True, "reverse_signal_short"
        elif self.position_direction == 'short' and self.should_enter_long(row, None):
            return True, "reverse_signal_long"
        
        return False, "hold"
    
    def execute_trade(self, 
                     date: str, 
                     direction: str, 
                     price: float, 
                     size: float = None) -> Dict:
        """
        执行交易
        
        Args:
            date: 交易日期
            direction: 交易方向 ('long', 'short', 'close')
            price: 交易价格
            size: 交易大小，如果不指定则使用全部可用资金
            
        Returns:
            交易结果字典
        """
        if direction in ['long', 'short']:
            # 开仓
            if self.current_position != 0:
                self.logger.warning(f"Warning: Already in position, closing old position before opening new one")
                # 先平掉旧仓位
                close_result = self.execute_trade(date, 'close', price)
                if not close_result['success']:
                    return close_result
            
            # 计算仓位大小
            if size is None:
                position_value = self.current_capital * self.leverage
                size = position_value / price
            
            # 计算交易成本
            cost = size * price * self.transaction_cost
            
            # 更新状态
            self.current_position = size if direction == 'long' else -size
            self.entry_price = price
            self.position_direction = direction
            
            # 扣除交易成本
            self.current_capital -= cost
            
            trade_record = {
                'date': date,
                'action': f'enter_{direction}',
                'price': price,
                'size': abs(size),
                'position': self.current_position,
                'capital': self.current_capital,
                'cost': cost,
                'direction': direction
            }
            
            self.trades_log.append(trade_record)
            self.logger.info(f"Opened {direction} position at {price:.3f}, size: {size:.3f}")
            
            return {'success': True, 'type': 'entry', 'record': trade_record}
        
        elif direction == 'close':
            # 平仓
            if self.position_direction is None:
                return {'success': False, 'reason': 'no position to close'}
            
            # 计算盈亏
            pnl = 0
            if self.position_direction == 'long':
                pnl = (price - self.entry_price) * self.current_position
            elif self.position_direction == 'short':
                pnl = (self.entry_price - price) * abs(self.current_position)
            
            # 计算交易成本
            cost = abs(self.current_position) * price * self.transaction_cost
            
            # 更新资本
            self.current_capital += pnl - cost
            
            # 记录交易
            trade_record = {
                'date': date,
                'action': 'exit',
                'price': price,
                'size': abs(self.current_position),
                'pnl': pnl,
                'cost': cost,
                'net_pnl': pnl - cost,
                'capital_after': self.current_capital,
                'direction': self.position_direction
            }
            
            self.trades_log.append(trade_record)
            self.logger.info(f"Closed {self.position_direction} position at {price:.3f}, P&L: {pnl:.2f}, net P&L: {pnl-cost:.2f}")
            
            # 重置仓位状态
            self.current_position = 0
            self.entry_price = None
            self.position_direction = None
            
            return {'success': True, 'type': 'exit', 'record': trade_record}
    
    def backtest(self, data: pd.DataFrame) -> Dict:
        """
        执行回测
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            回测结果字典
        """
        # 计算技术指标
        df = self.calculate_indicators(data)
        
        # 重置交易记录
        self.trades_log = []
        self.position_history = []
        self.current_capital = self.initial_capital
        
        # 遍历数据进行回测
        for i in range(max(30, 14, 20), len(df)):  # 确保有足够的历史数据
            current_row = df.iloc[i]
            prev_row = df.iloc[i-1] if i > 0 else None
            
            # 检查是否需要平仓
            should_exit, exit_reason = self.should_exit_position(current_row)
            if should_exit:
                result = self.execute_trade(
                    str(current_row.name),
                    'close',
                    current_row['close']
                )
                
                if result['success']:
                    self.logger.info(f"Position closed due to: {exit_reason}")
                    
            # 如果没有持仓，检查是否需要开仓
            if self.position_direction is None:
                # 检查多头入场
                if self.should_enter_long(current_row, prev_row):
                    result = self.execute_trade(
                        str(current_row.name),
                        'long',
                        current_row['close']
                    )
                    
                    if result['success']:
                        self.logger.info(f"Long entry executed at {current_row['close']:.3f}")
                    
                # 检查空头入场（如果没开多仓）
                elif self.should_enter_short(current_row, prev_row):
                    result = self.execute_trade(
                        str(current_row.name),
                        'short',
                        current_row['close']
                    )
                    
                    if result['success']:
                        self.logger.info(f"Short entry executed at {current_row['close']:.3f}")
        
        # 回测结束后平掉所有剩余仓位
        if self.position_direction is not None:
            last_price = df.iloc[-1]['close']
            self.execute_trade(str(df.index[-1]), 'close', last_price)
            self.logger.info(f"Closing remaining position at end of backtest: {last_price:.3f}")
        
        # 计算最终结果
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital
        self.total_return = total_return
        
        # 计算最大回撤
        capital_over_time = [self.initial_capital]
        temp_cap = self.initial_capital
        for trade in self.trades_log:
            if trade['action'] == 'exit':
                temp_cap = trade['capital_after']
            capital_over_time.append(temp_cap)
        
        if len(capital_over_time) > 1:
            running_max = np.maximum.accumulate(capital_over_time)
            drawdown = (running_max - capital_over_time) / running_max
            self.max_drawdown = np.max(drawdown)
        
        # 返回结果
        results = {
            'initial_capital': self.initial_capital,
            'final_capital': self.current_capital,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'num_trades': len([t for t in self.trades_log if t['action'] in ['enter_long', 'enter_short', 'exit']]),
            'winning_trades': len([t for t in self.trades_log if t.get('net_pnl', 0) > 0]),
            'losing_trades': len([t for t in self.trades_log if t.get('net_pnl', 0) < 0]),
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self._calculate_sharpe_ratio(capital_over_time),
            'trades_log': self.trades_log
        }
        
        return results
    
    def _calculate_sharpe_ratio(self, capital_series: List[float]) -> float:
        """
        计算夏普比率
        """
        if len(capital_series) < 2:
            return 0
        
        returns = np.diff(capital_series) / capital_series[:-1]
        if len(returns) == 0 or np.std(returns) == 0:
            return 0
        
        # 假设无风险利率为0
        sharpe = np.mean(returns) / np.std(returns)
        
        # 年化夏普比率（假设数据是日级别）
        return sharpe * np.sqrt(252)  # 252个交易日
    
    def plot_results(self, results: Dict, data: pd.DataFrame):
        """
        绘制回测结果
        """
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # 资金曲线
        capital_over_time = [self.initial_capital]
        temp_cap = self.initial_capital
        for trade in self.trades_log:
            if trade['action'] == 'exit':
                temp_cap = trade['capital_after']
            capital_over_time.append(temp_cap)
        
        axes[0].plot(capital_over_time)
        axes[0].set_title(f'Capital Curve - Final Return: {results["total_return_pct"]:.2f}%')
        axes[0].set_ylabel('Capital')
        
        # 交易记录
        long_entries = [(i, self.trades_log[i-1]['price']) for i, t in enumerate(self.trades_log) 
                       if t['action'] == 'enter_long']
        short_entries = [(i, self.trades_log[i-1]['price']) for i, t in enumerate(self.trades_log) 
                        if t['action'] == 'enter_short']
        
        if long_entries:
            idx, prices = zip(*long_entries)
            axes[1].scatter(idx, prices, c='green', label='Long Entry', alpha=0.6)
        if short_entries:
            idx, prices = zip(*short_entries)
            axes[1].scatter(idx, prices, c='red', label='Short Entry', alpha=0.6)
        
        axes[1].plot(data['close'].values[:len(capital_over_time)])
        axes[1].set_title('Price and Entry Points')
        axes[1].set_ylabel('Price')
        axes[1].legend()
        
        # P&L分布
        exits = [t for t in self.trades_log if t['action'] == 'exit']
        if exits:
            pnl_values = [t['net_pnl'] for t in exits]
            axes[2].bar(range(len(pnl_values)), pnl_values)
            axes[2].set_title('P&L per Trade')
            axes[2].set_ylabel('Net P&L')
            axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/tmp/bidirectional_strategy_backtest.png')
        plt.show()


def generate_mock_data(start_date: str = '2023-01-01', days: int = 365, volatility_factor: float = 0.02) -> pd.DataFrame:
    """
    生成模拟市场数据
    """
    dates = pd.date_range(start=start_date, periods=days, freq='D')
    
    # 初始价格
    prices = [100]
    for i in range(1, days):
        # 随机游走，但有一定的趋势成分
        trend = 0.0005  # 小的向上趋势
        noise = np.random.normal(0, volatility_factor)  # 可配置的波动率
        daily_return = trend + noise
        new_price = prices[-1] * (1 + daily_return)
        prices.append(new_price)
    
    # 创建OHLCV数据
    data = pd.DataFrame({
        'close': prices,
    }, index=dates)
    
    # 添加高低价，通常是收盘价±一定幅度
    data['high'] = data['close'] * (1 + np.abs(np.random.normal(0, volatility_factor * 0.5, len(data))))
    data['low'] = data['close'] * (1 - np.abs(np.random.normal(0, volatility_factor * 0.5, len(data))))
    data['open'] = data['close'].shift(1) * (1 + np.random.normal(0, volatility_factor * 0.25, len(data)))
    data['open'].iloc[0] = data['close'].iloc[0] * 0.999  # 设置第一个开盘价
    data['volume'] = np.random.randint(1000000, 5000000, len(data))
    
    # 确保高低价符合要求
    data['high'] = np.maximum(data[['open', 'close']].max(axis=1), data['high'])
    data['low'] = np.minimum(data[['open', 'close']].min(axis=1), data['low'])
    
    return data


def main():
    """主函数"""
    print("💰 双向交易策略实现")
    print("="*70)
    print("实现做多和做空双向交易策略，目标月收益20%")
    print("="*70)
    
    # 创建策略实例
    strategy = BidirectionalTradingStrategy(
        initial_capital=100000,
        leverage=2.0,  # 中等杠杆
        transaction_cost=0.001,
        stop_loss_pct=0.08,  # 合理的止损
        take_profit_pct=0.20,  # 合理的盈利目标
        min_price_atr_ratio=0.5
    )
    
    # 生成模拟数据
    print("📊 生成模拟市场数据...")
    data = generate_mock_data(days=365, volatility_factor=0.025)  # 略高于标准的波动率
    print(f"   数据范围: {data.index[0]} 到 {data.index[-1]}")
    print(f"   数据点数: {len(data)}")
    
    # 执行回测
    print("\n🧪 开始回测...")
    start_time = time.time()
    results = strategy.backtest(data)
    end_time = time.time()
    
    print(f"   回测耗时: {end_time - start_time:.2f}秒")
    
    # 输出结果
    print(f"\n📈 回测结果:")
    print(f"   初始资金: {results['initial_capital']:,.2f}")
    print(f"   最终资金: {results['final_capital']:,.2f}")
    print(f"   总收益率: {results['total_return_pct']:.2f}%")
    print(f"   总交易次数: {results['num_trades']}")
    print(f"   盈利交易: {results['winning_trades']}")
    print(f"   亏损交易: {results['losing_trades']}")
    print(f"   最大回撤: {results['max_drawdown']:.2%}")
    print(f"   夏普比率: {results['sharpe_ratio']:.2f}")
    
    # 计算月收益率
    total_months = len(data) / 30  # 近似计算
    monthly_return = (results['final_capital'] / results['initial_capital']) ** (1/total_months) - 1
    print(f"   月平均收益率: {monthly_return*100:.2f}%")
    
    # 判断是否达到目标
    target_met = monthly_return >= 0.20
    print(f"\n🎯 目标达成情况:")
    print(f"   月收益目标 (20%): {'✅ 达成' if target_met else '❌ 未达成'}")
    
    # 提供改进建议
    print(f"\n💡 优化建议:")
    if not target_met:
        print(f"   - 当前策略未能达到月收益20%的目标")
        print(f"   - 建议考虑更复杂的机器学习模型预测价格走势")
        print(f"   - 尝试引入更多市场情绪和技术指标")
        print(f"   - 考虑不同市场环境下的自适应参数调整")
        print(f"   - 实际交易中需考虑滑点和实际市场流动性")
    else:
        print(f"   - 策略表现良好，达到了月收益目标")
        print(f"   - 建议在真实市场中先用小额资金验证")
        print(f"   - 密切监控策略表现，防止过拟合")
    
    # 保存结果到文件
    result_file = "/tmp/bidirectional_strategy_results.json"
    import json
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 结果已保存至: {result_file}")
    
    # 绘制图表
    try:
        strategy.plot_results(results, data)
        print("📊 图表已生成并保存")
    except ImportError:
        print("⚠️ 无法绘制图表（缺少matplotlib库）")
    
    print("\n" + "="*70)
    print("双向交易策略实现完成！")
    print("所有结果已记录，准备向Master汇报")
    print("="*70)


if __name__ == "__main__":
    main()