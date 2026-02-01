#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SIL2603数据分析报告及tiger1.py参数优化建议
此脚本定期分析采集到的SIL2603数据，输出参数优化建议
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# 添加tigertrade目录到路径
sys.path.insert(0, '/home/cx/tigertrade')

from src import tiger1 as t1


class Sil2603Analyzer:
    """SIL2603数据分析器"""
    
    def __init__(self):
        self.symbol = "SIL2603"
        self.data_collector = None
        self.df = None
        self.current_params = self.get_current_params()
    
    def get_current_params(self):
        """获取当前参数"""
        return {
            'GRID_MAX_POSITION': t1.GRID_MAX_POSITION,
            'GRID_ATR_PERIOD': t1.GRID_ATR_PERIOD,
            'GRID_BOLL_PERIOD': t1.GRID_BOLL_PERIOD,
            'GRID_BOLL_STD': t1.GRID_BOLL_STD,
            'GRID_RSI_PERIOD_1M': t1.GRID_RSI_PERIOD_1M,
            'GRID_RSI_PERIOD_5M': t1.GRID_RSI_PERIOD_5M,
            'DAILY_LOSS_LIMIT': t1.DAILY_LOSS_LIMIT,
            'SINGLE_TRADE_LOSS': t1.SINGLE_TRADE_LOSS,
            'STOP_LOSS_MULTIPLIER': t1.STOP_LOSS_MULTIPLIER,
            'STOP_LOSS_ATR_FLOOR': t1.STOP_LOSS_ATR_FLOOR,
            'TAKE_PROFIT_ATR_OFFSET': t1.TAKE_PROFIT_ATR_OFFSET,
            'TAKE_PROFIT_MIN_OFFSET': t1.TAKE_PROFIT_MIN_OFFSET,
            'BOLL_DIVERGENCE_THRESHOLD': t1.BOLL_DIVERGENCE_THRESHOLD,
            'ATR_AMPLIFICATION_THRESHOLD': t1.ATR_AMPLIFICATION_THRESHOLD,
            'STOP_LOSS_ATR_FACTOR': t1.STOP_LOSS_ATR_FACTOR,
            'MIN_PROFIT_RATIO': t1.MIN_PROFIT_RATIO
        }
    
    def fetch_real_data(self, days=7):
        """获取真实数据"""
        print(f"📊 获取{days}天的真实SIL2603数据...")
        
        try:
            # 获取5分钟数据
            df_5m = t1.get_kline_data(self.symbol, '5min', count=days * 288)  # 每天约288个5分钟数据点
            print(f"📈 获取到{len(df_5m)}条5分钟数据")
            
            # 获取1分钟数据
            df_1m = t1.get_kline_data(self.symbol, '1min', count=days * 1440)  # 每天约1440个1分钟数据点
            print(f"📈 获取到{len(df_1m)}条1分钟数据")
            
            if df_5m.empty:
                print("⚠️ 无法获取5分钟真实数据，使用模拟数据作为备选方案")
                return self.generate_simulated_data(days)
            
            if df_1m.empty:
                print("⚠️ 无法获取1分钟真实数据，使用模拟数据作为备选方案")
                # 从5分钟数据扩展1分钟数据
                df_1m = self.expand_data(df_5m, '1min')
            
            # 手动计算技术指标以避免API问题
            df_5m_features = self.calculate_technical_indicators(df_5m)
            df_1m_features = self.calculate_technical_indicators(df_1m)
            
            # 构建综合数据框
            combined_data = pd.DataFrame({
                'timestamp': df_5m_features.index,
                'close_5m': df_5m_features['close'],
                'high_5m': df_5m_features['high'],
                'low_5m': df_5m_features['low'],
                'open_5m': df_5m_features['open'],
                'volume_5m': df_5m_features['volume'],
                'boll_upper': df_5m_features['boll_upper'],
                'boll_lower': df_5m_features['boll_lower'],
                'boll_middle': df_5m_features['boll_middle'],
                'atr': df_5m_features['atr'],
                'rsi_5m': df_5m_features['rsi']
            })
            
            # 添加1分钟RSI数据（由于频率不同，需要对齐）
            df_1m_aligned = df_1m_features.reindex(combined_data.index, method='nearest')
            combined_data['close_1m'] = df_1m_aligned['close']
            combined_data['rsi_1m'] = df_1m_aligned['rsi']
            
            # 删除包含NaN的行
            combined_data = combined_data.dropna()
            
            print(f"📊 最终真实数据集大小: {len(combined_data)} 条")
            return combined_data
        except Exception as e:
            print(f"❌ 获取真实数据失败: {e}")
            print("🔄 退回到模拟数据模式")
            return self.generate_simulated_data(days)
    
    def expand_data(self, df_original, target_period):
        """扩展数据，从较低频率数据扩展到较高频率"""
        # 这里简单地复制数据来模拟高频数据
        expanded_data = []
        
        for idx in range(len(df_original)-1):
            current_row = df_original.iloc[idx]
            next_row = df_original.iloc[idx+1]
            
            # 在每两个原始数据点之间插值生成中间数据点
            for i in range(5):  # 5分钟数据扩展为1分钟数据，每个间隔分成5份
                if i == 0:
                    # 使用原始数据点
                    temp_df = current_row.copy()
                    temp_df.name = df_original.index[idx]
                    expanded_data.append(temp_df)
                else:
                    # 插值生成中间数据点
                    ratio = i / 5.0
                    interpolated_close = current_row['close'] + (next_row['close'] - current_row['close']) * ratio
                    interpolated_high = max(current_row['high'], next_row['high'])
                    interpolated_low = min(current_row['low'], next_row['low'])
                    
                    temp_data = pd.Series({
                        'open': interpolated_close,
                        'high': interpolated_high,
                        'low': interpolated_low,
                        'close': interpolated_close,
                        'volume': int((current_row['volume'] + next_row['volume']) / 2)
                    })
                    
                    # 计算中间时间点
                    time_diff = (df_original.index[idx+1] - df_original.index[idx]) / 5
                    temp_data.name = df_original.index[idx] + time_diff * i
                    
                    expanded_data.append(temp_data)
        
        if expanded_data:
            df_expanded = pd.DataFrame.from_records([s.to_dict() for s in expanded_data])
            df_expanded.index = [s.name for s in expanded_data]
            return df_expanded
        else:
            return df_original.copy()
    
    def generate_simulated_data(self, days=7):
        """生成模拟数据"""
        print(f"📊 生成{days}天的模拟SIL2603数据...")
        
        # 生成时间序列
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        # 生成5分钟数据（大约每天288个点）
        n_points = days * 288
        timestamps = pd.date_range(start=start_time, end=end_time, periods=n_points)
        
        # 生成模拟价格数据 - 基于真实白银期货的价格范围（参考历史数据）
        base_price = 25.0  # 基础价格水平
        daily_volatility = 0.015  # 每日波动率1.5%
        
        # 生成价格变化
        returns = np.random.normal(0, daily_volatility/np.sqrt(288), n_points)
        prices = [base_price]
        
        for ret in returns[1:]:
            new_price = max(15.0, prices[-1] * (1 + ret))  # 价格不低于15
            prices.append(new_price)
        
        # 添加一些趋势和周期性
        trend_factor = np.linspace(-0.01, 0.02, n_points)  # 微小的趋势
        cycle_factor = 0.3 * np.sin(np.arange(n_points) * 2 * np.pi / 100)  # 周期性波动
        prices = np.array(prices) * (1 + trend_factor + cycle_factor)
        
        # 生成OHLC数据
        opens = prices * np.random.uniform(0.998, 1.002, n_points)
        highs = np.maximum(prices, opens) * np.random.uniform(1.0, 1.005, n_points)
        lows = np.minimum(prices, opens) * np.random.uniform(0.995, 1.0, n_points)
        closes = prices
        
        # 确保高低符合要求
        for i in range(len(highs)):
            if lows[i] > closes[i]:
                lows[i] = closes[i] * 0.999
            if highs[i] < closes[i]:
                highs[i] = closes[i] * 1.001
            if lows[i] > opens[i]:
                lows[i] = min(opens[i], closes[i]) * 0.999
            if highs[i] < opens[i]:
                highs[i] = max(opens[i], closes[i]) * 1.001

        # 生成成交量（随机但有一定趋势）
        volumes = np.random.lognormal(mean=8, sigma=1, size=n_points).astype(int)
        
        # 创建DataFrame
        df_5m = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        }, index=timestamps)
        
        # 高频数据（1分钟）- 从5分钟数据扩展而来
        df_1m = self.expand_data(df_5m, '1min')
        
        print(f"📈 生成了{len(df_1m)}条1分钟数据和{len(df_5m)}条5分钟数据")
        
        # 手动计算技术指标以避免API问题
        df_5m_features = self.calculate_technical_indicators(df_5m)
        df_1m_features = self.calculate_technical_indicators(df_1m)
        
        # 构建综合数据框
        combined_data = pd.DataFrame({
            'timestamp': df_5m_features.index,
            'close_5m': df_5m_features['close'],
            'high_5m': df_5m_features['high'],
            'low_5m': df_5m_features['low'],
            'open_5m': df_5m_features['open'],
            'volume_5m': df_5m_features['volume'],
            'boll_upper': df_5m_features['boll_upper'],
            'boll_lower': df_5m_features['boll_lower'],
            'boll_middle': df_5m_features['boll_middle'],
            'atr': df_5m_features['atr'],
            'rsi_5m': df_5m_features['rsi']
        })
        
        # 添加1分钟RSI数据（由于频率不同，需要对齐）
        df_1m_aligned = df_1m_features.reindex(combined_data.index, method='nearest')
        combined_data['close_1m'] = df_1m_aligned['close']
        combined_data['rsi_1m'] = df_1m_aligned['rsi']
        
        # 删除包含NaN的行
        combined_data = combined_data.dropna()
        
        print(f"📊 最终数据集大小: {len(combined_data)} 条")
        return combined_data
    
    def calculate_technical_indicators(self, df):
        """手动计算技术指标"""
        # 确保数据按时间排序
        df = df.sort_index()
        
        # 计算RSI
        def calculate_rsi(prices, period=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        # 计算ATR
        def calculate_atr(df, period=14):
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            return atr
        
        # 计算布林带
        def calculate_bollinger_bands(prices, period=20, std_dev=2):
            rolling_mean = prices.rolling(window=period).mean()
            rolling_std = prices.rolling(window=period).std()
            
            upper_band = rolling_mean + (rolling_std * std_dev)
            lower_band = rolling_mean - (rolling_std * std_dev)
            middle_band = rolling_mean
            
            return upper_band, lower_band, middle_band
        
        # 计算指标
        df['rsi'] = calculate_rsi(df['close'])
        df['atr'] = calculate_atr(df)
        
        upper, lower, middle = calculate_bollinger_bands(df['close'])
        df['boll_upper'] = upper
        df['boll_lower'] = lower
        df['boll_middle'] = middle
        
        return df
    
    def analyze_market_characteristics(self):
        """分析市场特征"""
        if self.df is None:
            print("⚠️ 数据未加载，请先调用fetch_real_data或generate_simulated_data方法")
            return
        
        print("\n🔍 SIL2603市场特征分析:")
        print("-" * 50)
        
        # 计算基本统计
        close_prices = self.df['close_5m']
        atr_values = self.df['atr']
        
        print(f"价格范围: {close_prices.min():.3f} - {close_prices.max():.3f}")
        print(f"平均价格: {close_prices.mean():.3f}")
        print(f"价格标准差: {close_prices.std():.3f}")
        print(f"ATR范围: {atr_values.min():.3f} - {atr_values.max():.3f}")
        print(f"平均ATR: {atr_values.mean():.3f}")
        print(f"平均RSI(5m): {self.df['rsi_5m'].mean():.2f}")
        print(f"平均RSI(1m): {self.df['rsi_1m'].mean():.2f}")
        
        # 波动性分析
        volatility = close_prices.pct_change().std() * np.sqrt(252)  # 年化波动率
        print(f"年化波动率: {volatility:.4f}")
        
        # 布林带宽度分析
        bb_width = (self.df['boll_upper'] - self.df['boll_lower']) / self.df['boll_middle']
        print(f"平均布林带宽度占比: {bb_width.mean():.4f}")
        
        # RSI超买超卖情况
        rsi_5m = self.df['rsi_5m']
        rsi_1m = self.df['rsi_1m']
        print(f"RSI(5m)超卖(<30)比例: {(rsi_5m < 30).sum() / len(rsi_5m):.2%}")
        print(f"RSI(5m)超买(>70)比例: {(rsi_5m > 70).sum() / len(rsi_5m):.2%}")
        print(f"RSI(1m)超卖(<30)比例: {(rsi_1m < 30).sum() / len(rsi_1m):.2%}")
        print(f"RSI(1m)超买(>70)比例: {(rsi_1m > 70).sum() / len(rsi_1m):.2%}")
    
    def analyze_parameter_sensitivity(self):
        """分析参数敏感性"""
        if self.df is None:
            print("⚠️ 数据未加载，请先调用fetch_real_data或generate_simulated_data方法")
            return
        
        print("\n🔍 参数敏感性分析:")
        print("-" * 50)
        
        # 分析ATR对不同乘数的反应
        atr_values = self.df['atr']
        current_atr_mult = self.current_params['STOP_LOSS_MULTIPLIER']
        
        print(f"当前止损ATR乘数: {current_atr_mult}")
        print("不同ATR乘数下的止损幅度(%)：")
        
        for mult in [0.8, 1.0, 1.2, 1.5, 2.0]:
            avg_stop_loss_pct = (atr_values * mult / self.df['close_5m'] * 100).mean()
            print(f"  {mult}x: {avg_stop_loss_pct:.3f}%")
        
        # 分析布林带周期
        print(f"\n当前布林带周期: {self.current_params['GRID_BOLL_PERIOD']}, 标准差: {self.current_params['GRID_BOLL_STD']}")
        
        # 分析RSI周期
        rsi_1m = self.df['rsi_1m']
        rsi_5m = self.df['rsi_5m']
        print(f"当前1m RSI周期: {self.current_params['GRID_RSI_PERIOD_1M']}")
        print(f"当前5m RSI周期: {self.current_params['GRID_RSI_PERIOD_5M']}")
        print(f"RSI振荡情况 - 1m波动范围: {rsi_1m.max() - rsi_1m.min():.2f}, 5m波动范围: {rsi_5m.max() - rsi_5m.min():.2f}")
        
        # 分析止盈参数
        current_tp_atr_offset = self.current_params['TAKE_PROFIT_ATR_OFFSET']
        current_tp_min_offset = self.current_params['TAKE_PROFIT_MIN_OFFSET']
        print(f"\n当前止盈ATR偏移: {current_tp_atr_offset}, 最小偏移: {current_tp_min_offset}")
        
        # 计算基于ATR的止盈距离
        avg_price = self.df['close_5m'].mean()
        avg_atr = self.df['atr'].mean()
        tp_atr_based = avg_atr * current_tp_atr_offset
        tp_min_based = current_tp_min_offset
        print(f"  平均ATR基础止盈距离: {tp_atr_based:.4f}")
        print(f"  固定最小止盈距离: {tp_min_based:.4f}")
        print(f"  实际平均止盈距离: {max(tp_atr_based, tp_min_based):.4f}")
        print(f"  止盈距离占价格比: {max(tp_atr_based, tp_min_based)/avg_price*100:.3f}%")
    
    def generate_optimization_recommendations(self):
        """生成参数优化建议"""
        if self.df is None:
            print("⚠️ 数据未加载，请先调用fetch_real_data或generate_simulated_data方法")
            return
        
        print("\n💡 参数优化建议:")
        print("-" * 50)
        
        close_prices = self.df['close_5m']
        atr_values = self.df['atr']
        rsi_5m = self.df['rsi_5m']
        rsi_1m = self.df['rsi_1m']
        
        # 价格水平分析
        avg_price = close_prices.mean()
        print(f"📊 基于当前平均价格 ({avg_price:.2f}) 的建议:")
        
        # ATR相关参数优化
        avg_atr = atr_values.mean()
        print(f"📊 基于当前平均ATR ({avg_atr:.4f}) 的建议:")
        
        # 止损参数
        current_sl_mult = self.current_params['STOP_LOSS_MULTIPLIER']
        suggested_sl_mult = round(current_sl_mult, 2)
        
        # 根据市场波动性调整止损
        atr_price_ratio = avg_atr / avg_price
        if atr_price_ratio > 0.02:  # 高波动
            suggested_sl_mult = min(2.0, current_sl_mult + 0.3)
            print(f"⚠️ 市场波动较高 (ATR/价格={atr_price_ratio:.4f})，建议增加止损ATR乘数至: {suggested_sl_mult}")
        elif atr_price_ratio < 0.005:  # 低波动
            suggested_sl_mult = max(0.8, current_sl_mult - 0.2)
            print(f"⚠️ 市场波动较低 (ATR/价格={atr_price_ratio:.4f})，建议减少止损ATR乘数至: {suggested_sl_mult}")
        else:
            print(f"✅ 当前止损ATR乘数 ({current_sl_mult}) 适合当前市场波动水平 (ATR/价格={atr_price_ratio:.4f})")
        
        # 止盈参数
        current_tp_atr = self.current_params['TAKE_PROFIT_ATR_OFFSET']
        current_tp_min = self.current_params['TAKE_PROFIT_MIN_OFFSET']
        
        suggested_tp_atr = current_tp_atr
        suggested_tp_min = current_tp_min
        
        # 根据波动性调整止盈参数
        if avg_atr / avg_price > 0.015:  # 高波动
            suggested_tp_atr = max(current_tp_atr, 0.25)
            suggested_tp_min = max(current_tp_min, 0.025)
            print(f"⚠️ 市场波动较高，建议增加止盈参数: ATR偏移{current_tp_atr}→{suggested_tp_atr}, 最小偏移{current_tp_min}→{suggested_tp_min}")
        else:
            print(f"✅ 当前止盈参数适合当前市场波动")
        
        # RSI参数
        rsi_1m_variability = rsi_1m.max() - rsi_1m.min()
        rsi_5m_variability = rsi_5m.max() - rsi_5m.min()
        
        print(f"📊 RSI变异性 - 1m: {rsi_1m_variability:.2f}, 5m: {rsi_5m_variability:.2f}")
        
        # RSI阈值建议
        if rsi_1m_variability > 60:  # RSI变化很大，可能过于敏感
            print("⚠️ 1分钟RSI变化剧烈，可能过于敏感，建议适当调整RSI阈值范围")
        else:
            print("✅ RSI变化在合理范围内")
        
        # 网格参数建议
        bb_width_avg = (self.df['boll_upper'] - self.df['boll_lower']).mean() / self.df['boll_middle'].mean()
        print(f"📊 平均布林带宽度占比: {bb_width_avg:.4f}")
        
        if bb_width_avg > 0.05:  # 布林带较宽，市场波动大
            print(f"⚠️ 布林带较宽，建议适当增加网格间隔以适应波动")
        elif bb_width_avg < 0.015:  # 布林带较窄，市场平稳
            print(f"⚠️ 布林带较窄，可考虑减小网格间隔以增加交易机会")
        else:
            print(f"✅ 布林带宽度适合当前网格参数")
        
        # 交易频率建议
        price_changes = close_prices.pct_change().abs()
        avg_daily_change = price_changes.resample('D').mean().mean() * 100
        print(f"📊 平均日价格变化: {avg_daily_change:.3f}%")
        
        if avg_daily_change > 2.0:  # 高波动
            print(f"⚠️ 价格波动较大，可能需要更保守的风险控制")
        elif avg_daily_change < 0.5:  # 低波动
            print(f"⚠️ 价格波动较小，可适当增加交易频率")
        else:
            print(f"✅ 价格波动适中，当前风险控制参数合适")
        
        # 综合建议
        print(f"\n🎯 综合参数优化建议:")
        print(f"  - STOP_LOSS_MULTIPLIER: {current_sl_mult} → {suggested_sl_mult}")
        print(f"  - TAKE_PROFIT_ATR_OFFSET: {current_tp_atr} → {suggested_tp_atr}")
        print(f"  - TAKE_PROFIT_MIN_OFFSET: {current_tp_min} → {suggested_tp_min}")
        print(f"  - GRID_MAX_POSITION: {self.current_params['GRID_MAX_POSITION']} (保持不变，根据账户资金调整)")
        print(f"  - DAILY_LOSS_LIMIT: ${self.current_params['DAILY_LOSS_LIMIT']} (根据账户规模调整)")
        
        # 基于分析结果的额外建议
        print(f"\n🔧 基于当前市场状况的额外建议:")
        print(f"  - 考虑调整RSI阈值以适应当前市场波动")
        print(f"  - 根据ATR/价格比率({atr_price_ratio:.4f})，可考虑动态调整网格参数")
        print(f"  - 建议在高波动时期降低单笔交易量以控制风险")
        print(f"  - 根据日变化率({avg_daily_change:.3f}%)，可考虑调整交易频率")
        
        # 建议监控的关键指标
        print(f"\n📊 建议定期监控的指标:")
        print(f"  - ATR/价格比率: 当前 {atr_price_ratio:.4f}")
        print(f"  - RSI 1分钟和5分钟的变异性")
        print(f"  - 布林带宽度变化")
        print(f"  - 价格日变化率")
        print(f"  - 成交量变化趋势")
    
    def run_analysis(self):
        """运行完整的分析流程"""
        print("🚀 开始SIL2603数据分析...")
        
        # 尝试获取真实数据
        self.df = self.fetch_real_data(days=7)
        if self.df is None:
            print("❌ 数据获取失败，终止分析")
            return
        
        # 分析市场特征
        self.analyze_market_characteristics()
        
        # 分析参数敏感性
        self.analyze_parameter_sensitivity()
        
        # 生成优化建议
        self.generate_optimization_recommendations()
        
        print(f"\n✅ SIL2603分析完成！")
        print(f"📈 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    """主函数"""
    analyzer = Sil2603Analyzer()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()