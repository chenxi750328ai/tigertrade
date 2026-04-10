#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据收集和分析系统，用于记录交易数据并优化算法参数
"""

import sys
import os
import json
from pathlib import Path
import csv
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pickle
import threading
import time

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src import tiger1 as t1


class DataCollector:
    """数据收集器"""
    
    def __init__(self, data_file='trading_data.csv'):
        self.data_file = data_file
        self.data = []
        self.fields = [
            'timestamp', 'price_current', 'grid_lower', 'grid_upper', 'atr', 
            'rsi_1m', 'rsi_5m', 'buffer', 'threshold', 'near_lower', 
            'rsi_ok', 'trend_check', 'rebound', 'vol_ok', 'final_decision',
            'take_profit_price', 'stop_loss_price', 'position_size', 'side',
            'deviation_percent', 'atr_multiplier', 'min_buffer_val'
        ]
        
        # 初始化CSV文件
        if not os.path.exists(self.data_file):
            with open(self.data_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.fields)
                writer.writeheader()
    
    def collect_data_point(self, **kwargs):
        """收集数据点"""
        # 获取当前时间戳
        current_timestamp = datetime.now().isoformat()
        data_point = {
            'timestamp': current_timestamp,
        }
        
        # 添加传入的参数
        for field in self.fields:
            if field in kwargs:
                data_point[field] = kwargs[field]
            elif field != 'timestamp':  # 确保不是timestamp字段
                data_point[field] = None
                
        # 确保timestamp字段有值
        if 'timestamp' not in data_point or not data_point['timestamp']:
            data_point['timestamp'] = current_timestamp
        
        # 写入CSV
        with open(self.data_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writerow(data_point)
        
        self.data.append(data_point)
        print(f"📊 数据点已记录: {data_point['timestamp']}")
    
    def load_data(self):
        """加载数据"""
        if os.path.exists(self.data_file):
            df = pd.read_csv(self.data_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        return pd.DataFrame(columns=self.fields)


class TradingAnalyzer:
    """交易分析器"""
    
    def __init__(self, data_collector):
        self.collector = data_collector
        self.df = self.collector.load_data()
    
    def analyze_performance(self):
        """分析性能"""
        if self.df.empty:
            print("⚠️ 没有数据可供分析")
            return {}
        
        analysis = {}
        
        # 计算基础统计
        analysis['total_records'] = len(self.df)
        analysis['successful_signals'] = self.df[self.df['final_decision'] == True].shape[0]
        analysis['success_rate'] = analysis['successful_signals'] / analysis['total_records'] if analysis['total_records'] > 0 else 0
        
        # 分析不同参数设置下的表现
        if 'buffer' in self.df.columns and 'atr' in self.df.columns:
            analysis['avg_buffer'] = self.df['buffer'].mean()
            analysis['avg_atr'] = self.df['atr'].mean()
            
            # 计算不同ATR倍数下的near_lower比率
            self.df['atr_multiplier'] = self.df['buffer'] / self.df['atr'].replace(0, np.nan)
            analysis['avg_atr_multiplier'] = self.df['atr_multiplier'].mean()
        
        # 计算偏离度统计
        if 'deviation_percent' in self.df.columns:
            valid_dev = self.df['deviation_percent'].dropna()
            if len(valid_dev) > 0:
                analysis['avg_deviation'] = valid_dev.mean()
                analysis['std_deviation'] = valid_dev.std()
        
        print(f"📈 性能分析结果:")
        print(f"   总记录数: {analysis['total_records']}")
        print(f"   成功信号数: {analysis['successful_signals']}")
        print(f"   成功率: {analysis['success_rate']:.2%}")
        if 'avg_buffer' in analysis:
            print(f"   平均buffer: {analysis['avg_buffer']:.4f}")
            print(f"   平均ATR: {analysis['avg_atr']:.4f}")
            print(f"   平均ATR倍数: {analysis['avg_atr_multiplier']:.4f}")
        if 'avg_deviation' in analysis:
            print(f"   平均偏离度: {analysis['avg_deviation']:.4f}")
            print(f"   偏离度标准差: {analysis['std_deviation']:.4f}")
        
        return analysis
    
    def optimize_parameters(self):
        """优化参数"""
        if self.df.empty or 'near_lower' not in self.df.columns:
            print("⚠️ 无法优化参数，缺少必要数据")
            return {}
        
        print("🔍 分析不同参数的影响...")
        
        # 分析偏离度与成功率的关系
        if 'deviation_percent' in self.df.columns and 'final_decision' in self.df.columns:
            # 按偏离度分组分析成功率
            self.df['dev_group'] = pd.cut(self.df['deviation_percent'], bins=10, labels=False)
            group_analysis = self.df.groupby('dev_group').agg({
                'final_decision': 'mean',
                'deviation_percent': 'mean'
            }).rename(columns={'final_decision': 'success_rate'})
            
            print("📊 偏离度与成功率关系:")
            for idx, row in group_analysis.iterrows():
                print(f"   偏离度组 {idx}: 平均偏离度={row['deviation_percent']:.3f}, 成功率={row['success_rate']:.2%}")
        
        # 分析ATR乘数对near_lower的影响
        atr_multipliers = [0.03, 0.05, 0.07, 0.1, 0.15, 0.2]
        min_buffers = [0.001, 0.0025, 0.005, 0.01, 0.015, 0.02]
        
        best_params = {
            'atr_multiplier': 0.05,
            'min_buffer': 0.0025,
            'score': 0
        }
        
        results = []
        for mult in atr_multipliers:
            for min_buf in min_buffers:
                # 模拟使用这些参数的near_lower结果
                simulated_buffer = self.df['atr'].apply(lambda x: max(mult * x, min_buf))
                
                # 计算模拟的near_lower
                simulated_near_lower = self.df['price_current'] <= (self.df['grid_lower'] + simulated_buffer)
                
                # 评估效果
                score = self._evaluate_buffer_score(simulated_near_lower, self.df)
                
                results.append({
                    'atr_multiplier': mult,
                    'min_buffer': min_buf,
                    'score': score
                })
                
                if score > best_params['score']:
                    best_params.update({
                        'atr_multiplier': mult,
                        'min_buffer': min_buf,
                        'score': score
                    })
        
        print(f"🎯 最佳参数:")
        print(f"   ATR乘数: {best_params['atr_multiplier']}")
        print(f"   最小buffer: {best_params['min_buffer']}")
        print(f"   评分: {best_params['score']:.4f}")
        
        return best_params
    
    def _evaluate_buffer_score(self, simulated_near_lower, df):
        """评估buffer参数得分"""
        if len(df) == 0:
            return 0
            
        # 计算价格与下轨的距离
        price_distance = (df['price_current'] - df['grid_lower']).abs()
        
        # 对于near_lower为True的点，计算平均距离
        true_points = simulated_near_lower[simulated_near_lower == True]
        if len(true_points) == 0:
            return 0
            
        avg_distance_when_true = price_distance[true_points.index].mean()
        
        # 计算信号密度（信号数量相对于总数的比例）
        signal_density = len(true_points) / len(df)
        
        # 评分：负相关距离 + 适度的信号密度
        score = (1.0 / (1.0 + avg_distance_when_true)) * (0.5 + 0.5 * signal_density)
        
        return score
    
    def visualize_data(self):
        """可视化数据"""
        if self.df.empty:
            print("⚠️ 没有数据可供可视化")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('交易数据可视化分析', fontsize=16)
        
        # 1. 价格和网格上下轨随时间的变化
        if 'price_current' in self.df.columns and 'grid_lower' in self.df.columns and 'grid_upper' in self.df.columns:
            axes[0, 0].plot(self.df['timestamp'], self.df['price_current'], label='Current Price', alpha=0.7)
            axes[0, 0].plot(self.df['timestamp'], self.df['grid_lower'], label='Grid Lower', linestyle='--', alpha=0.7)
            axes[0, 0].plot(self.df['timestamp'], self.df['grid_upper'], label='Grid Upper', linestyle='--', alpha=0.7)
            axes[0, 0].set_title('价格与网格边界')
            axes[0, 0].legend()
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. ATR和Buffer随时间的变化
        if 'atr' in self.df.columns and 'buffer' in self.df.columns:
            axes[0, 1].plot(self.df['timestamp'], self.df['atr'], label='ATR', color='red', alpha=0.7)
            axes[0, 1].plot(self.df['timestamp'], self.df['buffer'], label='Buffer', color='blue', alpha=0.7)
            axes[0, 1].set_title('ATR与Buffer比较')
            axes[0, 1].legend()
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. RSI分布
        if 'rsi_1m' in self.df.columns and 'rsi_5m' in self.df.columns:
            axes[0, 2].hist(self.df['rsi_1m'].dropna(), bins=30, alpha=0.5, label='RSI 1m', density=True)
            axes[0, 2].hist(self.df['rsi_5m'].dropna(), bins=30, alpha=0.5, label='RSI 5m', density=True)
            axes[0, 2].axvline(x=30, color='red', linestyle='--', label='RSI 30 (超卖)')
            axes[0, 2].axvline(x=70, color='green', linestyle='--', label='RSI 70 (超买)')
            axes[0, 2].set_title('RSI分布')
            axes[0, 2].set_xlabel('RSI值')
            axes[0, 2].legend()
        
        # 4. near_lower决策分布
        if 'near_lower' in self.df.columns:
            near_lower_counts = self.df['near_lower'].value_counts()
            axes[1, 0].bar(['False', 'True'], [near_lower_counts.get(False, 0), near_lower_counts.get(True, 0)])
            axes[1, 0].set_title('near_lower决策分布')
            axes[1, 0].set_ylabel('计数')
        
        # 5. 偏离度分布
        if 'deviation_percent' in self.df.columns:
            axes[1, 1].hist(self.df['deviation_percent'].dropna(), bins=30, alpha=0.7)
            axes[1, 1].set_title('价格偏离度分布')
            axes[1, 1].set_xlabel('偏离度')
            axes[1, 1].set_ylabel('频次')
        
        # 6. 成功率随时间变化
        if 'final_decision' in self.df.columns:
            # 计算滑动窗口的成功率
            window_size = min(20, len(self.df)//5)  # 使用较小的窗口
            if window_size > 0:
                rolling_success = self.df['final_decision'].rolling(window=window_size).mean()
                axes[1, 2].plot(self.df['timestamp'], rolling_success, label=f'{window_size}-period rolling success rate')
                axes[1, 2].set_title(f'成功率滑动平均 (窗口={window_size})')
                axes[1, 2].set_ylabel('成功率')
                axes[1, 2].tick_params(axis='x', rotation=45)
            else:
                axes[1, 2].text(0.5, 0.5, '数据不足', horizontalalignment='center', verticalalignment='center', transform=axes[1, 2].transAxes)
        
        plt.tight_layout()
        plt.savefig('/home/cx/trading_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("📊 可视化图表已保存至 /home/cx/trading_analysis.png")


def enhance_strategy_with_logging():
    """增强策略函数以支持数据收集"""
    # 保存原始函数
    original_grid_strategy = t1.grid_trading_strategy_pro1
    
    def enhanced_grid_trading_strategy_pro1():
        """增强版网格交易策略，带数据收集功能"""
        # 导入必要的库
        import sys
        sys.path.insert(0, str(_REPO_ROOT))
        from src import tiger1 as t1_mod
        
        # 获取市场数据
        df_1m = t1_mod.get_kline_data([t1_mod.FUTURE_SYMBOL], '1min', count=30)
        df_5m = t1_mod.get_kline_data([t1_mod.FUTURE_SYMBOL], '5min', count=50)
        if df_1m.empty or df_5m.empty:
            print("⚠️ 数据不足，跳过 enhanced_grid_trading_strategy_pro1")
            return

        indicators = t1_mod.calculate_indicators(df_1m, df_5m)
        if not indicators or '5m' not in indicators or '1m' not in indicators:
            print("⚠️ 指标计算失败，跳过 enhanced_grid_trading_strategy_pro1")
            return

        trend = t1_mod.judge_market_trend(indicators)
        t1_mod.adjust_grid_interval(trend, indicators)

        price_current = indicators['1m']['close']
        rsi_1m = indicators['1m']['rsi']
        rsi_5m = indicators['5m']['rsi']
        atr = indicators['5m']['atr']

        rsi_low_map = {
            'boll_divergence_down': 15,
            'osc_bear': 22,
            'osc_bull': 55,
            'bull_trend': 50,
            'osc_normal': 25
        }
        rsi_low = rsi_low_map.get(trend, 25)

        # 使用动态参数计算
        # 根据经验教训，使用分层动态缓冲区机制
        if atr > 0.2:
            # 高波动市场：使用最小系数
            atr_multiplier = 0.03
            min_buffer_val = 0.002
        elif atr > 0.1:
            # 中等波动市场
            atr_multiplier = 0.05
            min_buffer_val = 0.0025
        else:
            # 低波动市场
            atr_multiplier = 0.07
            min_buffer_val = 0.003

        buffer = max(atr_multiplier * (atr if atr else 0), min_buffer_val)
        threshold = t1_mod.grid_lower + buffer
        near_lower = price_current <= threshold

        # 计算价格相对于下轨的偏离度
        if t1_mod.grid_upper and t1_mod.grid_upper != t1_mod.grid_lower:
            deviation_percent = (price_current - t1_mod.grid_lower) / (t1_mod.grid_upper - t1_mod.grid_lower)
        else:
            deviation_percent = np.nan

        # 2) RSI acceptance: oversold OR reversal OR bullish divergence
        oversold_ok = False
        rsi_rev_ok = False
        rsi_div_ok = False
        try:
            oversold_ok = (rsi_1m is not None) and (rsi_1m <= (rsi_low + 5))

            # build recent RSI series (prefer precomputed, else compute)
            import talib
            try:
                rsis = df_1m['rsi']
            except Exception:
                rsis = talib.RSI(df_1m['close'], timeperiod=t1_mod.GRID_RSI_PERIOD_1M)

            rsis = rsis.dropna() if hasattr(rsis, 'dropna') else rsis
            rsi_prev = float(rsis.iloc[-2]) if hasattr(rsis, 'iloc') and len(rsis) >= 2 else None
            rsi_cap = (rsi_low + 12)

            # reversal: RSI crosses above 50 from below
            if (rsi_prev is not None) and (rsi_1m is not None):
                rsi_rev_ok = (rsi_prev < 50) and (rsi_1m >= 50)

            # bullish divergence: price makes lower low while RSI makes higher low
            try:
                lows = df_1m['low'].dropna()
                low_prev = float(lows.iloc[-2]) if len(lows) >= 2 else None
                low_cur = float(lows.iloc[-1]) if len(lows) >= 1 else None
                rsi_div_ok = (low_cur is not None and low_prev is not None and rsi_prev is not None and
                              (low_cur < low_prev) and (rsi_1m is not None) and (rsi_1m > rsi_prev) and (rsi_1m <= rsi_cap))
            except Exception:
                rsi_div_ok = False
        except Exception:
            oversold_ok = False
            rsi_rev_ok = False
            rsi_div_ok = False

        rsi_ok = oversold_ok or rsi_rev_ok or rsi_div_ok

        # 3) relaxed trend check
        trend_check = (trend in ['osc_bull', 'bull_trend'] and rsi_5m > 45) or \
                      (trend in ['osc_bear', 'boll_divergence_down'] and rsi_5m < 55)

        # 4) momentum / volume backups
        rebound = False
        vol_ok = False
        try:
            closes = df_1m['close'].dropna()
            last = float(closes.iloc[-1])
            prev = float(closes.iloc[-2]) if len(closes) >= 2 else None
            rebound = (prev is not None and last > prev)
            vols = df_1m['volume'].dropna()
            if len(vols) >= 6:
                window = vols.iloc[-6:-1]
                recent_mean = window.mean()
                recent_median = window.median()
                rmax = window.max()
                mean_up = recent_mean * 1.05
                med_up = recent_median * 1.01
                max_up = rmax * 0.95
                threshold = max(mean_up, med_up, max_up)
                vol_ok = vols.iloc[-1] >= max(threshold, 0)
        except Exception:
            rebound = False
            vol_ok = False

        # Final buy decision: near_lower + rsi_ok + (trend_check or rebound or vol_ok)
        final_decision = near_lower and rsi_ok and (trend_check or rebound or vol_ok)
        
        # 记录数据点
        collector = DataCollector()
        collector.collect_data_point(
            price_current=price_current,
            grid_lower=t1_mod.grid_lower,
            grid_upper=t1_mod.grid_upper,
            atr=atr,
            rsi_1m=rsi_1m,
            rsi_5m=rsi_5m,
            buffer=buffer,
            threshold=threshold,
            near_lower=near_lower,
            rsi_ok=rsi_ok,
            trend_check=trend_check,
            rebound=rebound,
            vol_ok=vol_ok,
            final_decision=final_decision,
            deviation_percent=deviation_percent,
            atr_multiplier=atr_multiplier,
            min_buffer_val=min_buffer_val,
            side='BUY'
        )

        if final_decision and t1_mod.check_risk_control(price_current, 'BUY'):
            stop_loss_price, projected_loss = t1_mod.compute_stop_loss(price_current, atr, t1_mod.grid_lower)
            if stop_loss_price is None or not isinstance(projected_loss, (int, float)) or not np.isfinite(projected_loss):
                print("⚠️ 止损计算异常，跳过买入(enhanced)")
                return
            # compute TP with buffer below grid_upper
            import math
            min_tick = 0.01
            try:
                min_tick = float(t1_mod.FUTURE_TICK_SIZE)
            except Exception:
                pass
            tp_offset = max(t1_mod.TAKE_PROFIT_ATR_OFFSET * (atr if atr else 0), t1_mod.TAKE_PROFIT_MIN_OFFSET)
            take_profit_price = max(price_current + min_tick, 
                                   (t1_mod.grid_upper - tp_offset) if t1_mod.grid_upper is not None else price_current + min_tick)
            
            # 更新数据记录，包含止盈止损价格
            collector.collect_data_point(
                price_current=price_current,
                grid_lower=t1_mod.grid_lower,
                grid_upper=t1_mod.grid_upper,
                atr=atr,
                rsi_1m=rsi_1m,
                rsi_5m=rsi_5m,
                buffer=buffer,
                threshold=threshold,
                near_lower=near_lower,
                rsi_ok=rsi_ok,
                trend_check=trend_check,
                rebound=rebound,
                vol_ok=vol_ok,
                final_decision=final_decision,
                take_profit_price=take_profit_price,
                stop_loss_price=stop_loss_price,
                position_size=1,
                deviation_percent=deviation_percent,
                atr_multiplier=atr_multiplier,
                min_buffer_val=min_buffer_val,
                side='BUY'
            )
            
            print(
                f"🎯 enhanced_grid_trading_strategy_pro1: 触发买入条件 -> price={price_current:.4f}, "
                f"rsi_1m={rsi_1m}, rsi_5m={rsi_5m}, atr={atr}, buffer={buffer:.4f}, near_lower={near_lower}, "
                f"rsi_ok={rsi_ok}, trend_check={trend_check}, rebound={rebound}, vol_ok={vol_ok}, "
                f"grid_lower={t1_mod.grid_lower}, grid_upper={t1_mod.grid_upper}, stop_loss={stop_loss_price:.4f}, tp={take_profit_price:.4f}"
            )
            t1_mod.place_tiger_order('BUY', 1, price_current, stop_loss_price)
            try:
                t1_mod.place_take_profit_order('BUY', 1, take_profit_price)
            except Exception:
                pass
    
    # 替换原始函数
    t1.grid_trading_strategy_pro1 = enhanced_grid_trading_strategy_pro1
    print("🔄 策略函数已增强，支持数据收集功能")


def run_optimization_process():
    """运行优化流程"""
    print("🚀 开始运行数据收集和参数优化流程...")
    
    # 1. 创建数据收集器
    collector = DataCollector()
    print(f"📊 数据收集器已创建，数据将保存至: {collector.data_file}")
    
    # 2. 增强策略函数以支持数据收集
    enhance_strategy_with_logging()
    print("🔄 策略函数已增强")
    
    # 3. 创建分析器
    analyzer = TradingAnalyzer(collector)
    print("⚙️ 分析器已创建")
    
    # 4. 加载现有数据并分析
    print("📈 加载现有数据并进行分析...")
    analysis_results = analyzer.analyze_performance()
    
    # 5. 优化参数
    print("🔍 开始参数优化...")
    best_params = analyzer.optimize_parameters()
    
    # 6. 可视化数据
    print("📊 生成可视化图表...")
    analyzer.visualize_data()
    
    print("\n🎯 优化流程完成!")
    print(f"   最佳ATR乘数: {best_params['atr_multiplier']}")
    print(f"   最佳最小buffer: {best_params['min_buffer']}")
    print(f"   评分: {best_params['score']:.4f}")
    
    return {
        'analysis_results': analysis_results,
        'best_params': best_params,
        'data_collector': collector,
        'analyzer': analyzer
    }


if __name__ == "__main__":
    # 运行优化流程
    results = run_optimization_process()
    
    # 打印总结
    print("\n📋 优化总结:")
    print(f"   - 数据文件: trading_data.csv")
    print(f"   - 分析记录数: {results['analysis_results'].get('total_records', 0)}")
    print(f"   - 成功率: {results['analysis_results'].get('success_rate', 0):.2%}")
    print(f"   - 最佳参数: ATR乘数={results['best_params']['atr_multiplier']}, 最小buffer={results['best_params']['min_buffer']}")
    print(f"   - 图表已保存至: /home/cx/trading_analysis.png")