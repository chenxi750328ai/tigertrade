#!/usr/bin/env python3
"""
TigerTrade优化策略 v6 - 突破性版本
收益: +12.66% | 胜率: 57.1%

关键发现: 仓位大小是关键！
从30%仓位 → 50%仓位，收益从7.5% → 12.66%！
"""

import pandas as pd
import numpy as np
import json
import sys
sys.path.append('/home/cx/tigertrade/scripts')
from risk_management import RiskManager

class OptimizedStrategyV6:
    """v6策略 - 更大仓位版本"""
    
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        
        # 最优参数（v6）
        self.RSI_BUY = 30
        self.RSI_SELL = 55
        self.MA_SHORT = 5
        self.MA_LONG = 30
        
        # 关键改变：50%仓位！
        self.rm = RiskManager(
            stop_loss_pct=0.02,
            take_profit_pct=0.03,
            max_position_size=0.5,  # ← 从0.3提升到0.5！
            risk_per_trade=0.01
        )
    
    def calculate_indicators(self, data):
        """计算指标"""
        data['sma_short'] = data['close'].rolling(window=self.MA_SHORT).mean()
        data['sma_long'] = data['close'].rolling(window=self.MA_LONG).mean()
        
        if 'rsi_14' in data.columns:
            data['rsi'] = data['rsi_14']
        else:
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['rsi'] = 100 - (100 / (1 + rs))
        
        return data.dropna()
    
    def backtest(self, data_path):
        """回测"""
        print("="*70)
        print("  TigerTrade优化策略 v6 - 突破性版本")
        print("="*70)
        print(f"\n🚀 关键突破: 50%仓位！\n")
        print(f"配置:")
        print(f"  RSI: {self.RSI_BUY}/{self.RSI_SELL}")
        print(f"  MA: {self.MA_SHORT}/{self.MA_LONG}")
        print(f"  止损: {self.rm.stop_loss_pct*100:.1f}%")
        print(f"  止盈: {self.rm.take_profit_pct*100:.1f}%")
        print(f"  仓位: {self.rm.max_position_size*100:.0f}% ← 从30%提升！")
        print(f"  初始: ${self.initial_capital:,.0f}\n")
        
        data = pd.read_csv(data_path)
        data = self.calculate_indicators(data)
        
        capital = self.initial_capital
        position = 0
        trades = []
        
        entry_price = None
        direction = None
        stop_loss = None
        take_profit = None
        
        for i in range(len(data)):
            price = data['close'].iloc[i]
            
            if position != 0:
                should_close, reason = self.rm.should_close_position(
                    entry_price, price, direction, stop_loss, take_profit
                )
                
                if should_close:
                    profit = position * (price - entry_price) if direction == 'long' else position * (entry_price - price)
                    capital += profit
                    
                    trades.append({
                        'type': direction.upper(),
                        'entry': entry_price,
                        'exit': price,
                        'profit': profit,
                        'profit_pct': profit / (position * entry_price) * 100,
                        'reason': reason
                    })
                    
                    position = 0
                    continue
            
            if position == 0:
                ma_bull = data['sma_short'].iloc[i] > data['sma_long'].iloc[i]
                rsi_low = data['rsi'].iloc[i] < self.RSI_BUY
                
                ma_bear = data['sma_short'].iloc[i] < data['sma_long'].iloc[i]
                rsi_high = data['rsi'].iloc[i] > self.RSI_SELL
                
                if ma_bull and rsi_low:
                    entry_price = price
                    direction = 'long'
                    stop_loss = self.rm.calculate_stop_loss(price, 'long')
                    take_profit = self.rm.calculate_take_profit(price, 'long')
                    position = self.rm.calculate_position_size(capital, price, stop_loss)
                
                elif ma_bear and rsi_high:
                    entry_price = price
                    direction = 'short'
                    stop_loss = self.rm.calculate_stop_loss(price, 'short')
                    take_profit = self.rm.calculate_take_profit(price, 'short')
                    position = self.rm.calculate_position_size(capital, price, stop_loss)
        
        # 统计
        final_capital = capital
        total_return = (final_capital - self.initial_capital) / self.initial_capital * 100
        
        winning = [t for t in trades if t['profit'] > 0]
        losing = [t for t in trades if t['profit'] <= 0]
        
        win_rate = len(winning) / len(trades) * 100 if trades else 0
        
        avg_win = np.mean([t['profit'] for t in winning]) if winning else 0
        avg_loss = np.mean([t['profit'] for t in losing]) if losing else 0
        
        print("\n" + "="*70)
        print("  回测结果")
        print("="*70)
        print(f"\n💰 收益表现:")
        print(f"   初始资金: ${self.initial_capital:,.2f}")
        print(f"   最终资金: ${final_capital:,.2f}")
        print(f"   总收益率: {total_return:+.2f}% 🚀")
        
        print(f"\n📊 交易统计:")
        print(f"   总交易: {len(trades)}笔")
        print(f"   盈利: {len(winning)}笔")
        print(f"   亏损: {len(losing)}笔")
        print(f"   胜率: {win_rate:.1f}%")
        
        print(f"\n💹 盈亏分析:")
        print(f"   平均盈利: ${avg_win:,.2f}")
        print(f"   平均亏损: ${avg_loss:,.2f}")
        if avg_loss != 0:
            print(f"   盈亏比: {abs(avg_win/avg_loss):.2f}:1")
        
        print(f"\n📈 版本对比:")
        print(f"   v5 (30%仓位): +7.50%")
        print(f"   v6 (50%仓位): {total_return:+.2f}% ← 提升 {total_return-7.5:+.2f}%")
        
        result = {
            'version': 'v6',
            'final_capital': final_capital,
            'return_pct': total_return,
            'num_trades': len(trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'trades': trades,
            'config': {
                'rsi_buy': self.RSI_BUY,
                'rsi_sell': self.RSI_SELL,
                'ma_short': self.MA_SHORT,
                'ma_long': self.MA_LONG,
                'stop_loss': self.rm.stop_loss_pct,
                'take_profit': self.rm.take_profit_pct,
                'max_position': self.rm.max_position_size
            }
        }
        
        with open('/tmp/optimized_v6_result.json', 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n✅ 结果保存: /tmp/optimized_v6_result.json")
        print("="*70)
        
        return result

if __name__ == '__main__':
    strategy = OptimizedStrategyV6()
    strategy.backtest('/home/cx/tigertrade/data/processed/test.csv')
