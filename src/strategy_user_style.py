#!/usr/bin/env python3
"""
基于用户真实交易风格的策略实现
根据70条SIL交易记录提取的规则
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional

class UserStyleStrategy:
    """
    模仿用户交易风格的策略
    
    核心特征：
    1. 价格下跌时加仓降低成本
    2. 不严格止损，容忍一定浮亏
    3. 多数情况下耐心等待盈利
    """
    
    def __init__(self, config: Dict = None):
        """初始化策略"""
        self.config = config or {}
        
        # 从真实交易数据提取的参数（优化版v2）
        self.entry_price_range = (74.0, 80.0)  # 开仓价格区间
        self.add_position_gap = 0.40  # 加仓间距（从0.55→0.40，更积极）
        self.max_add_times = 4  # 最多加仓次数（从3→4）
        self.max_price_drop = 2.50  # 最大容忍跌幅（从1.03→2.50美元，更宽容）
        self.target_profit_per_round = 400.0  # 目标每轮盈利（从271→400，更耐心）
        self.max_loss_per_round = 5000.0  # 最大单轮亏损（从1861→5000，容忍大浮亏）
        
        # 仓位管理
        self.initial_position = 1  # 初始仓位（手）
        self.add_position_size = 1  # 加仓大小（手）
        self.max_total_position = 5  # 最大总仓位
        
        # 当前状态
        self.current_positions = []  # [(开仓价格, 数量, 时间)]
        self.total_qty = 0
        self.avg_cost = 0.0
        self.round_pnl = 0.0
        self.round_id = 0
        
        # 统计
        self.completed_rounds = []
        self.total_pnl = 0.0
        
    def should_open_position(self, price: float, indicators: Dict) -> bool:
        """
        判断是否应该开仓
        
        条件：
        1. 无持仓
        2. 价格在合理区间
        3. 趋势判断为上涨（简化：使用EMA）
        """
        # 如果已有持仓，不再开新轮
        if self.total_qty > 0:
            return False
        
        # 价格检查
        if not (self.entry_price_range[0] <= price <= self.entry_price_range[1]):
            return False
        
        # 趋势判断（简化版，实际用户会看日K和盘面）
        ema_20 = indicators.get('ema_20', price)
        rsi = indicators.get('rsi', 50)
        
        # 价格接近或低于EMA，且RSI不过高
        if price <= ema_20 * 1.01 and 30 < rsi < 70:
            return True
        
        return False
    
    def should_add_position(self, price: float, indicators: Dict) -> bool:
        """
        判断是否应该加仓
        
        条件：
        1. 有持仓
        2. 价格下跌达到加仓间距
        3. 未超过最大加仓次数
        4. 未超过最大容忍跌幅
        """
        if self.total_qty == 0:
            return False
        
        # 检查加仓次数
        if len(self.current_positions) >= self.max_add_times:
            return False
        
        # 检查仓位上限
        if self.total_qty >= self.max_total_position:
            return False
        
        # 计算当前价格相对于平均成本的跌幅
        price_drop = self.avg_cost - price
        
        # 如果跌幅超过最大容忍，不再加仓（风控）
        if price_drop > self.max_price_drop:
            return False
        
        # 如果价格下跌达到加仓间距的60%（更积极加仓）
        if price_drop >= self.add_position_gap * 0.6:
            # 加上一些随机性，模拟人的判断
            if np.random.random() > 0.2:  # 80%概率加仓（从70%→80%）
                return True
        
        return False
    
    def should_close_position(self, price: float, indicators: Dict) -> Tuple[bool, int]:
        """
        判断是否应该平仓
        
        返回: (是否平仓, 平仓数量)
        
        策略：
        1. 盈利达到目标：全部平仓
        2. 小幅盈利+技术指标好：部分平仓
        3. 亏损但超过最大容忍：止损
        """
        if self.total_qty == 0:
            return False, 0
        
        # 计算当前盈亏
        current_pnl = (price - self.avg_cost) * self.total_qty * 1000  # 白银合约乘数
        
        # 1. 达到目标盈利：全部平仓
        if current_pnl >= self.target_profit_per_round:
            return True, self.total_qty
        
        # 2. 中等盈利（200-400美元）且RSI高位：部分平仓（更耐心）
        rsi = indicators.get('rsi', 50)
        if 200 < current_pnl < 400 and rsi > 70:  # 提高阈值：50→200, RSI 65→70
            # 平掉一半仓位
            close_qty = max(1, self.total_qty // 2)
            return True, close_qty
        
        # 3. 止损（亏损超过最大容忍）
        if current_pnl < -self.max_loss_per_round:
            # 用户实际很少止损，这里保留作为风控
            return True, self.total_qty
        
        # 4. 价格回升且有较好获利：平仓（更耐心）
        if price > self.avg_cost * 1.002 and current_pnl > 150:  # 提高阈值：0.998→1.002, 0→150
            # 根据盈利大小决定平仓数量
            if current_pnl > 250:  # 从100→250
                return True, self.total_qty
            else:
                # 中等盈利，部分平仓
                return True, max(1, self.total_qty // 2)
        
        return False, 0
    
    def open_position(self, price: float, timestamp: datetime):
        """开仓"""
        qty = self.initial_position
        self.current_positions.append((price, qty, timestamp))
        self.total_qty += qty
        self._update_avg_cost()
        
        print(f"📈 开仓: 价格={price:.2f}, 数量={qty}手, 平均成本={self.avg_cost:.2f}")
        return qty
    
    def add_position(self, price: float, timestamp: datetime):
        """加仓"""
        qty = self.add_position_size
        self.current_positions.append((price, qty, timestamp))
        self.total_qty += qty
        self._update_avg_cost()
        
        print(f"➕ 加仓: 价格={price:.2f}, 数量={qty}手, 新平均成本={self.avg_cost:.2f}, 总持仓={self.total_qty}手")
        return qty
    
    def close_position(self, price: float, qty: int, timestamp: datetime):
        """平仓"""
        if qty > self.total_qty:
            qty = self.total_qty
        
        # 计算盈亏
        pnl = (price - self.avg_cost) * qty * 1000
        self.round_pnl += pnl
        self.total_qty -= qty
        
        # 更新持仓列表（简化：从最早的持仓开始平）
        remaining_qty = qty
        new_positions = []
        for pos_price, pos_qty, pos_time in self.current_positions:
            if remaining_qty <= 0:
                new_positions.append((pos_price, pos_qty, pos_time))
            elif remaining_qty >= pos_qty:
                remaining_qty -= pos_qty
            else:
                new_positions.append((pos_price, pos_qty - remaining_qty, pos_time))
                remaining_qty = 0
        
        self.current_positions = new_positions
        self._update_avg_cost()
        
        print(f"📉 平仓: 价格={price:.2f}, 数量={qty}手, 本次盈亏=${pnl:.2f}, 累计盈亏=${self.round_pnl:.2f}, 剩余持仓={self.total_qty}手")
        
        # 如果全部平仓，记录这一轮
        if self.total_qty == 0:
            self._complete_round()
        
        return pnl
    
    def _update_avg_cost(self):
        """更新平均成本"""
        if self.total_qty > 0 and self.current_positions:
            total_cost = sum(price * qty for price, qty, _ in self.current_positions)
            self.avg_cost = total_cost / self.total_qty
        else:
            self.avg_cost = 0.0
    
    def _complete_round(self):
        """完成一轮交易"""
        self.round_id += 1
        self.completed_rounds.append({
            'round_id': self.round_id,
            'pnl': self.round_pnl,
            'num_entries': len(self.current_positions) if self.current_positions else 0
        })
        self.total_pnl += self.round_pnl
        
        print(f"✅ 第{self.round_id}轮完成: 盈亏=${self.round_pnl:.2f}, 总盈亏=${self.total_pnl:.2f}\n")
        
        # 重置轮次状态
        self.round_pnl = 0.0
        self.current_positions = []
    
    def get_signal(self, price: float, indicators: Dict, timestamp: datetime) -> Dict:
        """
        获取交易信号
        
        返回: {
            'action': 'BUY'/'SELL'/'HOLD',
            'quantity': int,
            'reason': str
        }
        """
        # 开仓信号
        if self.should_open_position(price, indicators):
            return {
                'action': 'BUY',
                'quantity': self.initial_position,
                'reason': '开仓条件满足'
            }
        
        # 加仓信号
        if self.should_add_position(price, indicators):
            return {
                'action': 'BUY',
                'quantity': self.add_position_size,
                'reason': f'价格下跌{self.avg_cost - price:.2f}美元，加仓'
            }
        
        # 平仓信号
        should_close, close_qty = self.should_close_position(price, indicators)
        if should_close:
            current_pnl = (price - self.avg_cost) * self.total_qty * 1000
            return {
                'action': 'SELL',
                'quantity': close_qty,
                'reason': f'平仓条件满足，当前盈亏=${current_pnl:.2f}'
            }
        
        return {
            'action': 'HOLD',
            'quantity': 0,
            'reason': '观望'
        }
    
    def get_stats(self) -> Dict:
        """获取策略统计"""
        if not self.completed_rounds:
            return {
                'total_rounds': 0,
                'total_pnl': 0,
                'avg_pnl': 0,
                'win_rate': 0
            }
        
        winning_rounds = [r for r in self.completed_rounds if r['pnl'] > 0]
        
        return {
            'total_rounds': len(self.completed_rounds),
            'total_pnl': self.total_pnl,
            'avg_pnl': self.total_pnl / len(self.completed_rounds),
            'win_rate': len(winning_rounds) / len(self.completed_rounds) * 100,
            'avg_entries': np.mean([r['num_entries'] for r in self.completed_rounds]),
            'max_pnl': max([r['pnl'] for r in self.completed_rounds]),
            'min_pnl': min([r['pnl'] for r in self.completed_rounds])
        }


if __name__ == '__main__':
    print("=" * 80)
    print("🎯 用户风格交易策略")
    print("=" * 80)
    print("\n基于真实交易数据提取的策略参数：")
    print("  • 加仓间距: 0.55美元")
    print("  • 最多加仓: 3次")
    print("  • 目标盈利: $271/轮")
    print("  • 最大亏损: $1,861/轮")
    print("  • 胜率目标: 45-80%")
    print("\n" + "=" * 80)
