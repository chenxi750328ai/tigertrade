#!/usr/bin/env python3
"""
整合版策略 v1.0
集成：优化的交易逻辑 + 完整的风险管理
"""

import pandas as pd
import numpy as np
import json
import sys
sys.path.append('/home/cx/tigertrade/scripts')

from risk_management import RiskManager

class IntegratedStrategy:
    """整合风险管理的交易策略"""
    
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = 0
        self.position_value = 0
        self.entry_price = None
        self.direction = None
        
        # 初始化风险管理器
        self.risk_manager = RiskManager(
            max_position_size=0.3,
            max_drawdown=0.2,
            stop_loss_pct=0.02,
            take_profit_pct=0.05
        )
        
        self.trades = []
        self.stop_loss = None
        self.take_profit = None
    
    def calculate_indicators(self, data):
        """计算技术指标（使用数据中已有的或重新计算）"""
        df = data.copy()
        
        # MA
        df['sma_short'] = df['close'].rolling(window=10).mean()
        df['sma_long'] = df['close'].rolling(window=30).mean()
        
        # RSI（数据中已有rsi_14）
        if 'rsi_14' not in df.columns:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
        else:
            df['rsi'] = df['rsi_14']
        
        # ATR（数据中已有atr_14）
        if 'atr_14' in df.columns:
            df['atr'] = df['atr_14']
        else:
            df['atr'] = df['close'].rolling(window=14).std()
        
        return df.dropna()
    
    def backtest(self, data_path):
        """回测"""
        # 读取数据
        data = pd.read_csv(data_path)
        data = self.calculate_indicators(data)
        
        print(f"回测数据: {len(data)} 条")
        
        for i in range(len(data)):
            current_price = data['close'].iloc[i]
            current_atr = data['atr'].iloc[i]
            
            # 检查回撤
            should_stop, drawdown = self.risk_manager.check_drawdown(self.capital)
            if should_stop:
                print(f"⚠️ 回撤超限({drawdown*100:.1f}%)，停止交易")
                break
            
            # 如果有仓位，检查止损止盈
            if self.position != 0:
                should_close, reason = self.risk_manager.should_close_position(
                    self.entry_price,
                    current_price,
                    self.direction,
                    self.stop_loss,
                    self.take_profit
                )
                
                if should_close:
                    # 平仓
                    if self.direction == 'long':
                        profit = self.position * (current_price - self.entry_price)
                    else:
                        profit = self.position * (self.entry_price - current_price)
                    
                    self.capital += profit
                    
                    self.trades.append({
                        'exit_price': current_price,
                        'profit': profit,
                        'close_reason': reason,
                        'capital': self.capital
                    })
                    
                    self.position = 0
                    self.direction = None
                    continue
            
            # 如果没有仓位，检查入场信号
            if self.position == 0:
                # 优化的入场条件（OR逻辑 + 放宽阈值）
                long_signal = (data['sma_short'].iloc[i] > data['sma_long'].iloc[i]) or (data['rsi'].iloc[i] < 40)
                short_signal = (data['sma_short'].iloc[i] < data['sma_long'].iloc[i]) or (data['rsi'].iloc[i] > 60)
                
                if long_signal and not short_signal:
                    # 做多
                    self.entry_price = current_price
                    self.direction = 'long'
                    
                    # 使用风险管理计算仓位
                    stop_loss_price = self.risk_manager.calculate_stop_loss(
                        current_price, 'long', current_atr
                    )
                    position_size = self.risk_manager.calculate_position_size(
                        self.capital, current_price, stop_loss_price
                    )
                    
                    self.position = position_size
                    self.stop_loss = stop_loss_price
                    self.take_profit = self.risk_manager.calculate_take_profit(
                        current_price, 'long'
                    )
                    
                    self.trades.append({
                        'type': 'LONG',
                        'entry_price': current_price,
                        'position': position_size,
                        'stop_loss': self.stop_loss,
                        'take_profit': self.take_profit
                    })
                
                elif short_signal and not long_signal:
                    # 做空
                    self.entry_price = current_price
                    self.direction = 'short'
                    
                    stop_loss_price = self.risk_manager.calculate_stop_loss(
                        current_price, 'short', current_atr
                    )
                    position_size = self.risk_manager.calculate_position_size(
                        self.capital, current_price, stop_loss_price
                    )
                    
                    self.position = position_size
                    self.stop_loss = stop_loss_price
                    self.take_profit = self.risk_manager.calculate_take_profit(
                        current_price, 'short'
                    )
                    
                    self.trades.append({
                        'type': 'SHORT',
                        'entry_price': current_price,
                        'position': position_size,
                        'stop_loss': self.stop_loss,
                        'take_profit': self.take_profit
                    })
        
        # 统计结果
        completed_trades = [t for t in self.trades if 'exit_price' in t]
        
        result = {
            'initial_capital': self.initial_capital,
            'final_capital': self.capital,
            'total_return_pct': (self.capital - self.initial_capital) / self.initial_capital * 100,
            'num_trades': len(completed_trades),
            'winning_trades': len([t for t in completed_trades if t['profit'] > 0]),
            'losing_trades': len([t for t in completed_trades if t['profit'] <= 0]),
            'max_drawdown': self.risk_manager.current_drawdown * 100,
            'trades': completed_trades[:10]  # 前10笔
        }
        
        return result

if __name__ == '__main__':
    print("=== 整合版策略回测 ===\n")
    
    strategy = IntegratedStrategy(initial_capital=100000)
    result = strategy.backtest('/home/cx/tigertrade/data/processed/test.csv')
    
    print(f"\n{'='*50}")
    print("回测结果:")
    print(f"{'='*50}")
    print(f"初始资金: ${result['initial_capital']:,.0f}")
    print(f"最终资金: ${result['final_capital']:,.0f}")
    print(f"总收益率: {result['total_return_pct']:.2f}%")
    print(f"交易次数: {result['num_trades']}")
    print(f"盈利交易: {result['winning_trades']}")
    print(f"亏损交易: {result['losing_trades']}")
    print(f"胜率: {result['winning_trades']/result['num_trades']*100:.1f}%" if result['num_trades'] > 0 else "胜率: N/A")
    print(f"最大回撤: {result['max_drawdown']:.2f}%")
    
    # 保存结果
    with open('/tmp/integrated_strategy_result.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n结果已保存: /tmp/integrated_strategy_result.json")
