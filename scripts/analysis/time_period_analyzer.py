#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据驱动的时段分析模块
从历史数据中自动提取时段特征（波动率、滑点率、流动性等）
支持大模型辅助分析时段特征
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional
import json
from collections import defaultdict

# 添加tigertrade目录到路径
sys.path.insert(0, '/home/cx/tigertrade')

# 延迟导入tiger1，避免循环导入
def get_tiger1_module():
    """延迟导入tiger1模块"""
    try:
        from src import tiger1 as t1
        return t1
    except ImportError:
        try:
            import tiger1 as t1
            return t1
        except ImportError:
            print("⚠️ 无法导入tiger1模块")
            return None


class TimePeriodAnalyzer:
    """数据驱动的时段分析器"""
    
    def __init__(self, symbol="SIL2603", reference_rules_path=None):
        """
        初始化时段分析器
        
        Args:
            symbol: 合约代码
            reference_rules_path: 参考规则文件路径（可选）
        """
        self.symbol = symbol
        self.reference_rules = self._load_reference_rules(reference_rules_path)
        self.analyzed_periods = {}
        
    def _load_reference_rules(self, rules_path):
        """加载参考规则（如果提供）"""
        if rules_path and os.path.exists(rules_path):
            try:
                # 这里可以解析参考规则文件
                # 暂时返回空字典，实际实现时可以解析JSON或Markdown
                return {}
            except Exception as e:
                print(f"⚠️ 加载参考规则失败: {e}")
        return {}
    
    def extract_time_period(self, timestamp: datetime) -> str:
        """
        从时间戳提取时段标签
        
        Args:
            timestamp: 时间戳（需为北京时间）
            
        Returns:
            时段标签，如 "COMEX_欧美高峰", "沪银_日盘尖峰" 等
        """
        # 确保时间戳为北京时间
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone(timedelta(hours=8)))
        elif timestamp.tzinfo != timezone(timedelta(hours=8)):
            # 转换为北京时间
            timestamp = timestamp.astimezone(timezone(timedelta(hours=8)))
        
        hour = timestamp.hour
        minute = timestamp.minute
        time_str = f"{hour:02d}:{minute:02d}"
        
        # 时段匹配规则（可扩展）
        if (20, 0) <= (hour, minute) < (22, 0):
            return "COMEX_欧美高峰"
        elif (21, 0) <= (hour, minute) < (23, 0):
            return "COMEX_欧美高峰_冬令时"
        elif (9, 0) <= (hour, minute) < (9, 30):
            return "沪银_日盘尖峰"
        elif (21, 0) <= (hour, minute) < (21, 30):
            return "沪银_夜盘联动"
        elif (6, 0) <= (hour, minute) < (8, 0):
            return "COMEX_亚洲低波动"
        else:
            return "其他低波动时段"
    
    def analyze_period_volatility(self, df: pd.DataFrame, 
                                   price_col='close', 
                                   time_col='timestamp') -> Dict[str, Dict]:
        """
        按时段分析波动率特征
        
        Args:
            df: 包含价格和时间的数据框
            price_col: 价格列名
            time_col: 时间列名
            
        Returns:
            各时段的波动率统计信息
        """
        if time_col not in df.columns:
            if df.index.name == 'timestamp' or isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
                time_col = df.columns[0] if time_col not in df.columns else time_col
        
        # 提取时段标签
        df['period'] = df[time_col].apply(self.extract_time_period)
        
        # 计算价格变化率
        df['price_change'] = df[price_col].pct_change()
        
        # 按时段分组统计
        period_stats = {}
        for period in df['period'].unique():
            period_data = df[df['period'] == period]
            
            if len(period_data) < 10:  # 数据量太少跳过
                continue
            
            # 计算波动率指标
            price_changes = period_data['price_change'].dropna()
            volatility = price_changes.std() * np.sqrt(252 * 288)  # 年化波动率（假设5分钟数据）
            
            # 计算ATR（如果数据中有）
            if 'atr' in period_data.columns:
                avg_atr = period_data['atr'].mean()
                atr_pct = (avg_atr / period_data[price_col].mean()) * 100
            else:
                avg_atr = None
                atr_pct = None
            
            # 价格范围
            price_range = period_data[price_col].max() - period_data[price_col].min()
            price_range_pct = (price_range / period_data[price_col].mean()) * 100
            
            period_stats[period] = {
                'volatility': volatility,
                'volatility_pct': volatility * 100,
                'avg_atr': avg_atr,
                'atr_pct': atr_pct,
                'price_range': price_range,
                'price_range_pct': price_range_pct,
                'data_count': len(period_data),
                'avg_price': period_data[price_col].mean()
            }
        
        return period_stats
    
    def analyze_period_slippage(self, orders_df: pd.DataFrame) -> Dict[str, Dict]:
        """
        按时段分析滑点率
        
        Args:
            orders_df: 订单数据框，需包含列：
                - timestamp: 订单时间
                - order_price: 挂单价
                - fill_price: 成交价
                - side: 买卖方向（BUY/SELL）
                
        Returns:
            各时段的滑点率统计信息
        """
        if orders_df.empty:
            return {}
        
        # 提取时段标签
        orders_df['period'] = orders_df['timestamp'].apply(self.extract_time_period)
        
        # 计算滑点
        orders_df['slippage'] = np.where(
            orders_df['side'] == 'BUY',
            (orders_df['fill_price'] - orders_df['order_price']) / orders_df['order_price'],
            (orders_df['order_price'] - orders_df['fill_price']) / orders_df['order_price']
        )
        
        # 按时段分组统计
        period_slippage = {}
        for period in orders_df['period'].unique():
            period_orders = orders_df[orders_df['period'] == period]
            
            if len(period_orders) < 5:  # 数据量太少跳过
                continue
            
            slippage_values = period_orders['slippage'].dropna()
            
            period_slippage[period] = {
                'mean_slippage': slippage_values.mean(),
                'median_slippage': slippage_values.median(),
                'std_slippage': slippage_values.std(),
                'p95_slippage': slippage_values.quantile(0.95),
                'order_count': len(period_orders),
                'slippage_pct': slippage_values.mean() * 100
            }
        
        return period_slippage
    
    def analyze_period_liquidity(self, df: pd.DataFrame,
                                 volume_col='volume',
                                 time_col='timestamp') -> Dict[str, Dict]:
        """
        按时段分析流动性特征
        
        Args:
            df: 包含成交量和时间的数据框
            volume_col: 成交量列名
            time_col: 时间列名
            
        Returns:
            各时段的流动性统计信息
        """
        if time_col not in df.columns:
            if df.index.name == 'timestamp' or isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
                time_col = df.columns[0] if time_col not in df.columns else time_col
        
        # 提取时段标签
        df['period'] = df[time_col].apply(self.extract_time_period)
        
        # 按时段分组统计
        period_liquidity = {}
        for period in df['period'].unique():
            period_data = df[df['period'] == period]
            
            if len(period_data) < 10:
                continue
            
            volumes = period_data[volume_col].dropna()
            
            period_liquidity[period] = {
                'mean_volume': volumes.mean(),
                'median_volume': volumes.median(),
                'std_volume': volumes.std(),
                'total_volume': volumes.sum(),
                'data_count': len(period_data),
                'volume_stability': 1 - (volumes.std() / volumes.mean()) if volumes.mean() > 0 else 0
            }
        
        return period_liquidity
    
    def calculate_balance_threshold(self, contract_price: float, 
                                    slippage_rate: float,
                                    safety_factor: float = 1.2) -> float:
        """
        计算网格盈利-滑点平衡阈值
        
        Args:
            contract_price: 合约价格
            slippage_rate: 滑点率（小数形式，如0.008表示0.8%）
            safety_factor: 安全系数，默认1.2
            
        Returns:
            平衡阈值（最小网格间距）
        """
        slippage_cost = contract_price * slippage_rate
        balance_threshold = 2 * slippage_cost * safety_factor
        return round(balance_threshold, 4)
    
    def generate_period_config(self, period_stats: Dict,
                                slippage_stats: Dict = None,
                                liquidity_stats: Dict = None,
                                contract_price: float = 25.0) -> Dict[str, Dict]:
        """
        基于数据分析结果生成时段配置
        
        Args:
            period_stats: 时段波动率统计
            slippage_stats: 时段滑点率统计（可选）
            liquidity_stats: 时段流动性统计（可选）
            contract_price: 合约基准价格
            
        Returns:
            时段配置字典
        """
        period_configs = {}
        
        for period, stats in period_stats.items():
            # 获取滑点率（优先使用实际数据，否则使用参考规则）
            if slippage_stats and period in slippage_stats:
                slippage_rate = slippage_stats[period]['mean_slippage']
            elif self.reference_rules and period in self.reference_rules:
                slippage_rate = self.reference_rules[period].get('slippage_rate', 0.015)
            else:
                # 默认滑点率（根据波动率估算）
                volatility_pct = stats.get('volatility_pct', 100)
                if volatility_pct > 150:
                    slippage_rate = 0.008  # 高波动，低滑点
                elif volatility_pct > 100:
                    slippage_rate = 0.015
                else:
                    slippage_rate = 0.020  # 低波动，高滑点
            
            # 计算平衡阈值
            balance_threshold = self.calculate_balance_threshold(
                contract_price, slippage_rate
            )
            
            # 根据波动率和流动性确定仓位上限
            volatility_pct = stats.get('volatility_pct', 100)
            if liquidity_stats and period in liquidity_stats:
                volume_stability = liquidity_stats[period].get('volume_stability', 0.5)
            else:
                volume_stability = 0.5
            
            # 仓位上限逻辑
            if volatility_pct > 180 and volume_stability > 0.6:
                max_position = 10
            elif volatility_pct > 150:
                max_position = 8
            elif volatility_pct > 120:
                max_position = 6
            elif volatility_pct > 80:
                max_position = 3
            else:
                max_position = 2
            
            # 限价单偏离幅度（根据滑点率调整）
            if slippage_rate < 0.01:
                order_offset = 0.02
            elif slippage_rate < 0.02:
                order_offset = 0.03
            elif slippage_rate < 0.025:
                order_offset = 0.04
            else:
                order_offset = 0.05
            
            period_configs[period] = {
                'volatility': volatility_pct / 100,  # 转换为倍数形式
                'slippage_rate': slippage_rate,
                'balance_threshold': balance_threshold,
                'max_position': max_position,
                'order_offset': order_offset,
                'data_quality': {
                    'volatility_data_count': stats.get('data_count', 0),
                    'slippage_data_count': slippage_stats[period].get('order_count', 0) if slippage_stats and period in slippage_stats else 0
                }
            }
        
        return period_configs
    
    def analyze_from_klines(self, days: int = 30) -> Dict:
        """
        从K线数据中分析时段特征
        
        Args:
            days: 分析最近N天的数据
            
        Returns:
            完整的时段分析结果
        """
        print(f"📊 开始分析最近{days}天的时段特征...")
        
        try:
            # 延迟导入tiger1模块
            t1 = get_tiger1_module()
            if t1 is None:
                print("⚠️ 无法获取K线数据：tiger1模块不可用")
                return {}
            
            # 获取K线数据
            df_5m = t1.get_kline_data(self.symbol, '5min', count=days * 288)
            
            if df_5m.empty:
                print("⚠️ 无法获取K线数据")
                return {}
            
            # 确保时间列为datetime类型
            if not isinstance(df_5m.index, pd.DatetimeIndex):
                if 'timestamp' in df_5m.columns:
                    df_5m['timestamp'] = pd.to_datetime(df_5m['timestamp'])
                    df_5m = df_5m.set_index('timestamp')
                else:
                    print("⚠️ 无法识别时间列")
                    return {}
            
            # 转换为北京时间
            if df_5m.index.tz is None:
                df_5m.index = df_5m.index.tz_localize('UTC').tz_convert('Asia/Shanghai')
            else:
                df_5m.index = df_5m.index.tz_convert('Asia/Shanghai')
            
            df_5m = df_5m.reset_index()
            df_5m['timestamp'] = df_5m['timestamp']
            
            # 分析波动率
            print("📈 分析时段波动率...")
            period_stats = self.analyze_period_volatility(df_5m, 'close', 'timestamp')
            
            # 分析流动性
            print("💧 分析时段流动性...")
            liquidity_stats = self.analyze_period_liquidity(df_5m, 'volume', 'timestamp')
            
            # 尝试分析滑点（如果有订单数据）
            slippage_stats = None
            # TODO: 从交易日志中提取订单数据
            
            # 生成时段配置
            print("⚙️ 生成时段配置...")
            avg_price = df_5m['close'].mean()
            period_configs = self.generate_period_config(
                period_stats, slippage_stats, liquidity_stats, avg_price
            )
            
            # 汇总结果
            result = {
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'symbol': self.symbol,
                'data_period_days': days,
                'period_stats': period_stats,
                'liquidity_stats': liquidity_stats,
                'slippage_stats': slippage_stats,
                'period_configs': period_configs,
                'reference_rules_used': len(self.reference_rules) > 0
            }
            
            print(f"✅ 时段分析完成，共分析{len(period_configs)}个时段")
            return result
            
        except Exception as e:
            print(f"❌ 时段分析失败: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def print_analysis_report(self, analysis_result: Dict):
        """打印分析报告"""
        if not analysis_result:
            print("⚠️ 无分析结果")
            return
        
        print("\n" + "="*60)
        print("📊 时段特征分析报告")
        print("="*60)
        print(f"分析时间: {analysis_result['analysis_date']}")
        print(f"合约: {analysis_result['symbol']}")
        print(f"数据周期: {analysis_result['data_period_days']}天")
        print(f"参考规则使用: {'是' if analysis_result['reference_rules_used'] else '否'}")
        print("\n" + "-"*60)
        
        period_configs = analysis_result.get('period_configs', {})
        period_stats = analysis_result.get('period_stats', {})
        
        print("\n时段配置建议:")
        print("-"*60)
        print(f"{'时段':<20} {'波动率':<10} {'滑点率':<10} {'平衡阈值':<12} {'最大仓位':<10} {'订单偏离':<10}")
        print("-"*60)
        
        for period, config in period_configs.items():
            stats = period_stats.get(period, {})
            print(f"{period:<20} "
                  f"{config['volatility']*100:.1f}%{'':<5} "
                  f"{config['slippage_rate']*100:.2f}%{'':<5} "
                  f"{config['balance_threshold']:.4f}{'':<6} "
                  f"{config['max_position']:<10} "
                  f"{config['order_offset']:.2f}")
        
        print("\n" + "="*60)


def main():
    """主函数"""
    analyzer = TimePeriodAnalyzer(symbol="SIL2603")
    result = analyzer.analyze_from_klines(days=30)
    analyzer.print_analysis_report(result)
    
    # 保存结果到JSON文件
    output_file = f"/home/cx/trading_data/time_period_analysis_{datetime.now().strftime('%Y%m%d')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n💾 分析结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
