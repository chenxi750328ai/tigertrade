#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
时段自适应策略模块
整合数据驱动分析和参考规则，实现时段自适应的网格交易策略
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple
import json

# 添加tigertrade目录到路径
sys.path.insert(0, '/home/cx/tigertrade')

try:
    from scripts.analysis.time_period_analyzer import TimePeriodAnalyzer
    from scripts.analysis.llm_period_analyzer import LLMPeriodAnalyzer
except ImportError as e:
    print(f"⚠️ 无法导入时段分析模块: {e}")
    TimePeriodAnalyzer = None
    LLMPeriodAnalyzer = None


class TimePeriodStrategy:
    """时段自适应策略"""
    
    def __init__(self, symbol="SIL2603", 
                 period_config_file: str = None,
                 use_reference_rules: bool = True):
        """
        初始化时段自适应策略
        
        Args:
            symbol: 合约代码
            period_config_file: 时段配置文件路径（JSON格式）
            use_reference_rules: 是否使用参考规则作为兜底
        """
        self.symbol = symbol
        self.use_reference_rules = use_reference_rules
        
        # 加载时段配置
        self.period_configs = self._load_period_configs(period_config_file)
        
        # 初始化分析器（如果可用）
        if TimePeriodAnalyzer:
            self.analyzer = TimePeriodAnalyzer(symbol=symbol)
        else:
            self.analyzer = None
            print("⚠️ 时段分析器不可用，将使用参考规则")
        
        # 默认参考规则（兜底）
        self.default_configs = self._get_default_configs()
    
    def _load_period_configs(self, config_file: str) -> Dict:
        """加载时段配置"""
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    return config.get('period_configs', {})
            except Exception as e:
                print(f"⚠️ 加载时段配置失败: {e}")
        
        return {}
    
    def _get_default_configs(self) -> Dict:
        """获取默认参考规则配置"""
        return {
            "COMEX_欧美高峰": {
                "time_range": ("20:00", "22:00"),
                "volatility": 2.0,
                "slippage_rate": 0.008,
                "max_position": 10,
                "order_offset": 0.02
            },
            "COMEX_欧美高峰_冬令时": {
                "time_range": ("21:00", "23:00"),
                "volatility": 1.9,
                "slippage_rate": 0.010,
                "max_position": 8,
                "order_offset": 0.03
            },
            "沪银_日盘尖峰": {
                "time_range": ("09:00", "09:30"),
                "volatility": 1.9,
                "slippage_rate": 0.028,
                "max_position": 3,
                "order_offset": 0.05
            },
            "沪银_夜盘联动": {
                "time_range": ("21:00", "21:30"),
                "volatility": 1.7,
                "slippage_rate": 0.015,
                "max_position": 6,
                "order_offset": 0.04
            },
            "COMEX_亚洲低波动": {
                "time_range": ("06:00", "08:00"),
                "volatility": 0.8,
                "slippage_rate": 0.020,
                "max_position": 2,
                "order_offset": 0.05
            },
            "其他低波动时段": {
                "time_range": ("00:00", "23:59"),
                "volatility": 0.6,
                "slippage_rate": 0.015,
                "max_position": 2,
                "order_offset": 0.04
            }
        }
    
    def get_current_period_config(self, current_time: datetime = None) -> Dict:
        """
        获取当前时段的配置
        
        Args:
            current_time: 当前时间（默认使用当前时间）
            
        Returns:
            当前时段的配置字典
        """
        if current_time is None:
            current_time = datetime.now(timezone(timedelta(hours=8)))
        
        # 优先使用数据驱动的配置
        if self.analyzer and self.period_configs:
            period_name = self.analyzer.extract_time_period(current_time)
            if period_name in self.period_configs:
                config = self.period_configs[period_name].copy()
                config['source'] = 'data_driven'
                return config
        
        # 使用参考规则（兜底）
        if self.use_reference_rules:
            period_name = self._match_period_from_time(current_time)
            if period_name in self.default_configs:
                config = self.default_configs[period_name].copy()
                config['source'] = 'reference_rules'
                return config
        
        # 默认配置
        default_config = self.default_configs["其他低波动时段"].copy()
        default_config['source'] = 'default'
        return default_config
    
    def get_period_config(self, period_name: str) -> Optional[Dict]:
        """
        按时段名称获取配置（供测试或外部按名称查询）。
        优先从 period_configs（文件加载），其次 default_configs（参考规则）。
        """
        if self.period_configs and period_name in self.period_configs:
            return self.period_configs[period_name]
        if self.default_configs and period_name in self.default_configs:
            return self.default_configs[period_name]
        return None

    def _match_period_from_time(self, current_time: datetime) -> str:
        """从时间匹配时段"""
        hour = current_time.hour
        minute = current_time.minute
        time_str = f"{hour:02d}:{minute:02d}"
        
        # 检查夏令时/冬令时（简化处理，实际需要更精确的判断）
        # 这里假设3-10月为夏令时，11-2月为冬令时
        month = current_time.month
        is_daylight_saving = 3 <= month <= 10
        
        if (20, 0) <= (hour, minute) < (22, 0) and is_daylight_saving:
            return "COMEX_欧美高峰"
        elif (21, 0) <= (hour, minute) < (23, 0) and not is_daylight_saving:
            return "COMEX_欧美高峰_冬令时"
        elif (9, 0) <= (hour, minute) < (9, 30):
            return "沪银_日盘尖峰"
        elif (21, 0) <= (hour, minute) < (21, 30):
            return "沪银_夜盘联动"
        elif (6, 0) <= (hour, minute) < (8, 0):
            return "COMEX_亚洲低波动"
        else:
            return "其他低波动时段"
    
    def calculate_balance_threshold(self, contract_price: float, 
                                    slippage_rate: float,
                                    safety_factor: float = 1.2) -> float:
        """
        计算网格盈利-滑点平衡阈值
        
        Args:
            contract_price: 合约价格
            slippage_rate: 滑点率
            safety_factor: 安全系数
            
        Returns:
            平衡阈值
        """
        slippage_cost = contract_price * slippage_rate
        balance_threshold = 2 * slippage_cost * safety_factor
        return round(balance_threshold, 4)
    
    def get_grid_parameters(self, current_price: float,
                            current_time: datetime = None) -> Dict:
        """
        获取当前时段的网格参数
        
        Args:
            current_price: 当前价格
            current_time: 当前时间
            
        Returns:
            网格参数字典，包含：
            - grid_step: 网格间距
            - grid_upper: 网格上轨
            - grid_lower: 网格下轨
            - max_position: 最大仓位
            - order_offset: 限价单偏离幅度
            - balance_threshold: 平衡阈值
        """
        period_config = self.get_current_period_config(current_time)
        
        # 计算平衡阈值（最小网格间距，用于确保盈利覆盖滑点）
        balance_threshold = self.calculate_balance_threshold(
            current_price,
            period_config['slippage_rate']
        )
        
        # 计算合理的网格范围（基于价格百分比，更符合实际交易需求）
        volatility = period_config.get('volatility', 1.0)
        
        # 根据波动率计算网格范围百分比
        # 低波动(0.6): ±1.0%, 中波动(1.0): ±1.5%, 高波动(2.0): ±2.5%
        if volatility <= 0.8:
            grid_range_pct = 0.010  # ±1.0%
        elif volatility <= 1.5:
            grid_range_pct = 0.015  # ±1.5%
        else:
            grid_range_pct = 0.025  # ±2.5%
        
        # 计算网格范围（以当前价格为中心）
        grid_range = current_price * grid_range_pct
        
        # 计算网格上下轨
        grid_upper = current_price + grid_range
        grid_lower = current_price - grid_range
        
        # 计算合理的网格间距（基于网格范围，确保有足够的网格层级）
        # 网格间距应该是网格范围的1/4到1/6，这样可以在网格范围内有4-6个网格层级
        num_grid_levels = 5  # 目标网格层级数
        grid_step_from_range = grid_range / num_grid_levels
        
        # 确保网格间距 >= 平衡阈值（这是硬性要求，必须覆盖滑点）
        grid_step = max(balance_threshold, grid_step_from_range, 0.1)
        
        # 如果平衡阈值太大（超过网格范围的1/3），说明滑点率设置可能不合理
        # 在这种情况下，需要调整策略：
        # 1. 如果平衡阈值 > 网格范围的1/2，说明滑点率设置不合理，使用更保守的网格范围
        # 2. 限制网格范围在合理范围内（最多±2%），即使平衡阈值很大
        if balance_threshold > grid_range / 2:
            # 平衡阈值太大，限制网格范围在合理范围内
            max_grid_range_pct = 0.02  # 最多±2%
            max_grid_range = current_price * max_grid_range_pct
            
            # 使用较小的网格范围
            grid_range = min(grid_range, max_grid_range)
            grid_upper = current_price + grid_range
            grid_lower = current_price - grid_range
            
            # 重新计算网格间距（基于调整后的网格范围）
            grid_step_from_range = grid_range / num_grid_levels
            grid_step = max(balance_threshold, grid_step_from_range, 0.1)
            
            # 如果平衡阈值仍然导致网格间距过大，使用平衡阈值但限制网格范围
            if grid_step > grid_range / 2:
                # 使用平衡阈值作为网格间距
                grid_step = balance_threshold
                # 但确保网格范围至少包含3个网格间距
                min_grid_range = grid_step * 3
                # 限制在最大允许范围内
                grid_range = min(min_grid_range, max_grid_range)
                grid_upper = current_price + grid_range
                grid_lower = current_price - grid_range
        
        return {
            'grid_step': grid_step,
            'grid_upper': grid_upper,
            'grid_lower': grid_lower,
            'max_position': period_config['max_position'],
            'order_offset': period_config['order_offset'],
            'balance_threshold': balance_threshold,
            'period_name': self._match_period_from_time(current_time or datetime.now(timezone(timedelta(hours=8)))),
            'config_source': period_config.get('source', 'unknown'),
            'volatility': period_config.get('volatility', 1.0),
            'slippage_rate': period_config['slippage_rate']
        }
    
    def update_period_configs_from_analysis(self, analysis_result: Dict):
        """
        从分析结果更新时段配置
        
        Args:
            analysis_result: 时段分析结果
        """
        if 'period_configs' in analysis_result:
            self.period_configs = analysis_result['period_configs']
            print(f"✅ 已更新时段配置，共{len(self.period_configs)}个时段")
    
    def refresh_analysis(self, days: int = 30):
        """
        刷新时段分析（重新分析历史数据）
        
        Args:
            days: 分析最近N天的数据
        """
        if not self.analyzer:
            print("⚠️ 时段分析器不可用")
            return
        
        print(f"🔄 开始刷新时段分析（最近{days}天）...")
        result = self.analyzer.analyze_from_klines(days=days)
        
        if result:
            self.update_period_configs_from_analysis(result)
            print("✅ 时段分析刷新完成")
        else:
            print("⚠️ 时段分析刷新失败")


def main():
    """测试函数"""
    strategy = TimePeriodStrategy(symbol="SIL2603")
    
    # 获取当前时段配置
    current_config = strategy.get_current_period_config()
    print(f"\n当前时段配置:")
    print(f"  时段: {current_config.get('source', 'unknown')}")
    print(f"  波动率: {current_config.get('volatility', 0)}")
    print(f"  滑点率: {current_config.get('slippage_rate', 0)*100:.2f}%")
    print(f"  最大仓位: {current_config.get('max_position', 0)}")
    
    # 获取网格参数
    grid_params = strategy.get_grid_parameters(25.0)
    print(f"\n网格参数:")
    print(f"  网格间距: {grid_params['grid_step']:.4f}美元")
    print(f"  网格区间: {grid_params['grid_lower']:.2f} ~ {grid_params['grid_upper']:.2f}美元")
    print(f"  平衡阈值: {grid_params['balance_threshold']:.4f}美元")
    print(f"  最大仓位: {grid_params['max_position']}手")
    print(f"  配置来源: {grid_params['config_source']}")


if __name__ == "__main__":
    main()
