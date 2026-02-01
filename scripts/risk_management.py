#!/usr/bin/env python3
"""
风险管理模块
实现止损、止盈、仓位管理、最大回撤控制
"""

import numpy as np
from typing import Dict, Optional, Tuple

class RiskManager:
    """风险管理器"""
    
    def __init__(self,
                 max_position_size: float = 0.3,  # 单次最大仓位30%
                 max_drawdown: float = 0.2,       # 最大回撤20%
                 stop_loss_pct: float = 0.02,     # 止损2%
                 take_profit_pct: float = 0.05,   # 止盈5%
                 risk_per_trade: float = 0.01):   # 每笔交易风险1%
        """
        初始化风险管理参数
        
        Args:
            max_position_size: 单次最大仓位比例
            max_drawdown: 允许的最大回撤比例
            stop_loss_pct: 止损百分比
            take_profit_pct: 止盈百分比
            risk_per_trade: 每笔交易的风险比例
        """
        self.max_position_size = max_position_size
        self.max_drawdown = max_drawdown
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.risk_per_trade = risk_per_trade
        
        # 运行时状态
        self.peak_capital = None
        self.current_drawdown = 0.0
        
    def calculate_position_size(self,
                                capital: float,
                                entry_price: float,
                                stop_loss_price: float,
                                confidence: float = 1.0) -> float:
        """
        计算仓位大小
        
        Args:
            capital: 当前资金
            entry_price: 入场价格
            stop_loss_price: 止损价格
            confidence: 信号置信度(0-1)
            
        Returns:
            建议的仓位大小（股数/合约数）
        """
        # 基于风险的仓位计算
        risk_amount = capital * self.risk_per_trade
        price_risk = abs(entry_price - stop_loss_price)
        
        if price_risk == 0:
            position_size = 0
        else:
            position_size = risk_amount / price_risk
        
        # 限制最大仓位
        max_position = capital * self.max_position_size / entry_price
        position_size = min(position_size, max_position)
        
        # 根据信号置信度调整
        position_size *= confidence
        
        return position_size
    
    def check_drawdown(self, current_capital: float) -> Tuple[bool, float]:
        """
        检查回撤是否超限
        
        Args:
            current_capital: 当前资金
            
        Returns:
            (是否需要停止交易, 当前回撤比例)
        """
        if self.peak_capital is None:
            self.peak_capital = current_capital
        
        # 更新峰值
        if current_capital > self.peak_capital:
            self.peak_capital = current_capital
        
        # 计算回撤
        self.current_drawdown = (self.peak_capital - current_capital) / self.peak_capital
        
        # 判断是否超限
        stop_trading = self.current_drawdown >= self.max_drawdown
        
        return stop_trading, self.current_drawdown
    
    def calculate_stop_loss(self,
                           entry_price: float,
                           direction: str,
                           atr: Optional[float] = None) -> float:
        """
        计算止损价格
        
        Args:
            entry_price: 入场价格
            direction: 'long' 或 'short'
            atr: 平均真实波幅（可选，用于动态止损）
            
        Returns:
            止损价格
        """
        if atr is not None:
            # 基于ATR的动态止损
            stop_distance = atr * 2
        else:
            # 固定百分比止损
            stop_distance = entry_price * self.stop_loss_pct
        
        if direction == 'long':
            stop_loss = entry_price - stop_distance
        else:  # short
            stop_loss = entry_price + stop_distance
        
        return stop_loss
    
    def calculate_take_profit(self,
                             entry_price: float,
                             direction: str,
                             risk_reward_ratio: float = 2.5) -> float:
        """
        计算止盈价格
        
        Args:
            entry_price: 入场价格
            direction: 'long' 或 'short'
            risk_reward_ratio: 风险收益比
            
        Returns:
            止盈价格
        """
        profit_distance = entry_price * self.take_profit_pct * risk_reward_ratio
        
        if direction == 'long':
            take_profit = entry_price + profit_distance
        else:  # short
            take_profit = entry_price - profit_distance
        
        return take_profit
    
    def should_close_position(self,
                            entry_price: float,
                            current_price: float,
                            direction: str,
                            stop_loss: float,
                            take_profit: float) -> Tuple[bool, str]:
        """
        判断是否应该平仓
        
        Args:
            entry_price: 入场价格
            current_price: 当前价格
            direction: 'long' 或 'short'
            stop_loss: 止损价格
            take_profit: 止盈价格
            
        Returns:
            (是否平仓, 原因)
        """
        if direction == 'long':
            if current_price <= stop_loss:
                return True, 'stop_loss'
            elif current_price >= take_profit:
                return True, 'take_profit'
        else:  # short
            if current_price >= stop_loss:
                return True, 'stop_loss'
            elif current_price <= take_profit:
                return True, 'take_profit'
        
        return False, ''
    
    def adjust_position_for_volatility(self,
                                      base_position: float,
                                      current_volatility: float,
                                      avg_volatility: float) -> float:
        """
        根据波动率调整仓位
        
        Args:
            base_position: 基础仓位
            current_volatility: 当前波动率
            avg_volatility: 平均波动率
            
        Returns:
            调整后的仓位
        """
        if avg_volatility == 0:
            return base_position
        
        # 波动率高时减小仓位
        volatility_ratio = current_volatility / avg_volatility
        adjustment_factor = 1 / volatility_ratio if volatility_ratio > 1 else 1
        
        return base_position * adjustment_factor
    
    def get_risk_metrics(self) -> Dict:
        """
        获取当前风险指标
        
        Returns:
            风险指标字典
        """
        return {
            'current_drawdown': self.current_drawdown,
            'max_drawdown_limit': self.max_drawdown,
            'peak_capital': self.peak_capital,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'max_position_size': self.max_position_size,
            'risk_per_trade': self.risk_per_trade
        }


# 使用示例
if __name__ == '__main__':
    print("=== 风险管理模块测试 ===\n")
    
    rm = RiskManager(
        max_position_size=0.3,
        max_drawdown=0.2,
        stop_loss_pct=0.02,
        take_profit_pct=0.05
    )
    
    # 测试仓位计算
    capital = 100000
    entry_price = 25.50
    stop_loss = 25.00
    
    position = rm.calculate_position_size(capital, entry_price, stop_loss)
    print(f"建议仓位: {position:.0f} 股")
    print(f"仓位价值: ${position * entry_price:,.2f}")
    print(f"占总资金: {(position * entry_price / capital * 100):.1f}%\n")
    
    # 测试止损止盈
    stop = rm.calculate_stop_loss(entry_price, 'long')
    profit = rm.calculate_take_profit(entry_price, 'long')
    print(f"入场价: ${entry_price:.2f}")
    print(f"止损价: ${stop:.2f} (-{(entry_price-stop)/entry_price*100:.1f}%)")
    print(f"止盈价: ${profit:.2f} (+{(profit-entry_price)/entry_price*100:.1f}%)\n")
    
    # 测试回撤检查
    should_stop, drawdown = rm.check_drawdown(80000)
    print(f"当前回撤: {drawdown*100:.1f}%")
    print(f"是否停止交易: {'是' if should_stop else '否'}\n")
    
    print("✅ 风险管理模块就绪！")

