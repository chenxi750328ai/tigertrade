#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
详细分析交易数据文件并提出网格交易参数优化策略
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def analyze_data_freshness_and_optimize():
    """分析数据时效性并提出优化建议"""
    
    print("🔍 开始分析交易数据时效性与优化策略...")
    
    try:
        # 读取数据
        df = pd.read_csv('trading_data.csv')
        print(f"📊 已加载 {len(df)} 条交易数据")
        
        # 检查文件修改时间
        mod_time = os.path.getmtime('trading_data.csv')
        mod_date = datetime.fromtimestamp(mod_time)
        print(f"📁 文件最后修改时间: {mod_date}")
        
        # 检查数据中的时间戳（如果有）
        if 'timestamp' in df.columns:
            # 将时间戳转换为datetime对象
            df['timestamp_dt'] = pd.to_datetime(df['timestamp'], errors='coerce')
            valid_timestamps = df['timestamp_dt'].dropna()
            
            if len(valid_timestamps) > 0:
                earliest_data = valid_timestamps.min()
                latest_data = valid_timestamps.max()
                
                print(f"🕒 最早数据时间: {earliest_data}")
                print(f"🕒 最晚数据时间: {latest_data}")
                
                # 计算数据的新鲜度
                now = datetime.now()
                data_age = now - latest_data
                file_age = now - mod_date
                
                print(f"⏰ 最新数据距今: {data_age.days}天 {data_age.seconds//3600}小时")
                print(f"⏰ 文件距今: {file_age.days}天 {file_age.seconds//3600}小时")
                
                # 数据时效性判断
                if data_age.days <= 1:
                    print("✅ 数据时效性: 非常新鲜（1天内）")
                elif data_age.days <= 7:
                    print("✅ 数据时效性: 较新鲜（1周内）")
                elif data_age.days <= 30:
                    print("⚠️ 数据时效性: 一般（1个月内）")
                else:
                    print("❌ 数据时效性: 较旧（超过1个月）")
            else:
                print("⚠️ 数据中没有有效的时间戳")
        else:
            print("⚠️ CSV文件中没有timestamp列")
        
        # 数据完整性检查
        print(f"\n🔍 数据完整性检查:")
        numeric_cols = ['price_current', 'grid_lower', 'grid_upper', 'atr', 'rsi_1m', 'rsi_5m', 'buffer']
        for col in numeric_cols:
            if col in df.columns:
                valid_count = df[col].notna().sum()
                print(f"  {col}: {valid_count}/{len(df)} 有效数据 ({valid_count/len(df)*100:.1f}%)")
        
        # 数据有效性筛选
        df_valid = df[
            (df['price_current'].notna()) & 
            (df['grid_lower'].notna()) & 
            (df['grid_upper'].notna()) & 
            (df['atr'].notna())
        ].copy()
        
        print(f"\n📊 有效数据记录: {len(df_valid)}/{len(df)} ({len(df_valid)/len(df)*100:.1f}%)")
        
        if len(df_valid) == 0:
            print("❌ 没有有效的数值数据")
            return
        
        # 计算市场特征
        print(f"\n📈 SIL2603市场特征分析:")
        print("-" * 50)
        
        # 价格统计
        price_stats = {
            'min': df_valid['price_current'].min(),
            'max': df_valid['price_current'].max(),
            'mean': df_valid['price_current'].mean(),
            'std': df_valid['price_current'].std(),
            'median': df_valid['price_current'].median()
        }
        
        print(f"价格范围: {price_stats['min']:.3f} - {price_stats['max']:.3f}")
        print(f"平均价格: {price_stats['mean']:.3f}")
        print(f"中位数价格: {price_stats['median']:.3f}")
        print(f"价格标准差: {price_stats['std']:.3f}")
        
        # ATR统计
        atr_stats = {
            'min': df_valid['atr'].min(),
            'max': df_valid['atr'].max(),
            'mean': df_valid['atr'].mean(),
            'std': df_valid['atr'].std()
        }
        
        print(f"ATR范围: {atr_stats['min']:.3f} - {atr_stats['max']:.3f}")
        print(f"平均ATR: {atr_stats['mean']:.3f}")
        print(f"ATR标准差: {atr_stats['std']:.3f}")
        
        # RSI统计
        if 'rsi_1m' in df_valid.columns and 'rsi_5m' in df_valid.columns:
            rsi_1m_stats = {
                'mean': df_valid['rsi_1m'].mean(),
                'min': df_valid['rsi_1m'].min(),
                'max': df_valid['rsi_1m'].max(),
                'std': df_valid['rsi_1m'].std()
            }
            
            rsi_5m_stats = {
                'mean': df_valid['rsi_5m'].mean(),
                'min': df_valid['rsi_5m'].min(),
                'max': df_valid['rsi_5m'].max(),
                'std': df_valid['rsi_5m'].std()
            }
            
            print(f"RSI(1m)范围: {rsi_1m_stats['min']:.2f} - {rsi_1m_stats['max']:.2f} (均值: {rsi_1m_stats['mean']:.2f}, 标准差: {rsi_1m_stats['std']:.2f})")
            print(f"RSI(5m)范围: {rsi_5m_stats['min']:.2f} - {rsi_5m_stats['max']:.2f} (均值: {rsi_5m_stats['mean']:.2f}, 标准差: {rsi_5m_stats['std']:.2f})")
        
        # 波动率分析 (ATR/价格)
        avg_atr_price_ratio = atr_stats['mean'] / price_stats['mean']
        print(f"平均ATR/价格比率: {avg_atr_price_ratio:.4f}")
        
        # 计算价格变化统计
        if len(df_valid) > 1:
            price_changes = df_valid['price_current'].pct_change().abs()
            avg_price_change = price_changes.mean() * 100
            median_price_change = price_changes.median() * 100
            max_price_change = price_changes.max() * 100
            print(f"平均价格变化幅度: {avg_price_change:.3f}% (中位数: {median_price_change:.3f}%, 最大值: {max_price_change:.3f}%)")
        
        # RSI超买超卖情况
        if 'rsi_1m' in df_valid.columns and 'rsi_5m' in df_valid.columns:
            rsi_1m_overbought = (df_valid['rsi_1m'] > 70).sum() / len(df_valid) * 100
            rsi_1m_oversold = (df_valid['rsi_1m'] < 30).sum() / len(df_valid) * 100
            rsi_5m_overbought = (df_valid['rsi_5m'] > 70).sum() / len(df_valid) * 100
            rsi_5m_oversold = (df_valid['rsi_5m'] < 30).sum() / len(df_valid) * 100
            
            print(f"RSI(1m)超买(>70)比例: {rsi_1m_overbought:.2f}%")
            print(f"RSI(1m)超卖(<30)比例: {rsi_1m_oversold:.2f}%")
            print(f"RSI(5m)超买(>70)比例: {rsi_5m_overbought:.2f}%")
            print(f"RSI(5m)超卖(<30)比例: {rsi_5m_oversold:.2f}%")
        
        # 网格参数分析
        grid_stats = {
            'lower_min': df_valid['grid_lower'].min(),
            'lower_max': df_valid['grid_lower'].max(),
            'upper_min': df_valid['grid_upper'].min(),
            'upper_max': df_valid['grid_upper'].max(),
            'lower_mean': df_valid['grid_lower'].mean(),
            'upper_mean': df_valid['grid_upper'].mean(),
            'width_mean': (df_valid['grid_upper'] - df_valid['grid_lower']).mean()
        }
        
        print(f"网格下轨范围: {grid_stats['lower_min']:.3f} - {grid_stats['lower_max']:.3f}")
        print(f"网格上轨范围: {grid_stats['upper_min']:.3f} - {grid_stats['upper_max']:.3f}")
        print(f"平均网格宽度: {grid_stats['width_mean']:.3f}")
        
        # 分析交易决策
        if 'near_lower' in df_valid.columns:
            near_lower_true_pct = (df_valid['near_lower'] == True).sum() / len(df_valid) * 100
            print(f"near_lower为True的比例: {near_lower_true_pct:.2f}%")
        
        if 'rsi_ok' in df_valid.columns:
            rsi_ok_true_pct = (df_valid['rsi_ok'] == True).sum() / len(df_valid) * 100
            print(f"rsi_ok为True的比例: {rsi_ok_true_pct:.2f}%")
        
        if 'final_decision' in df_valid.columns:
            final_decision_true_pct = (df_valid['final_decision'] == True).sum() / len(df_valid) * 100
            print(f"最终决策为True的比例: {final_decision_true_pct:.2f}%")
        
        # 基于真实数据分析参数优化
        print(f"\n💡 基于真实数据的参数优化建议:")
        print("-" * 50)
        
        print(f"📊 基于平均价格 ({price_stats['mean']:.2f}) 的建议:")
        print(f"📊 基于平均ATR ({atr_stats['mean']:.4f}) 的建议:")
        print(f"📊 ATR/价格比率: {avg_atr_price_ratio:.4f}")
        
        # 根据ATR/价格比率判断市场波动性
        if avg_atr_price_ratio > 0.015:
            print("⚠️ 市场波动较高，建议增加止损ATR乘数至: 1.5")
            suggested_stop_loss_mult = 1.5
        elif avg_atr_price_ratio < 0.005:
            print("⚠️ 市场波动较低，建议减少止损ATR乘数至: 0.8")
            suggested_stop_loss_mult = 0.8
        else:
            print("✅ 当前市场波动适中，维持止损ATR乘数: 1.2")
            suggested_stop_loss_mult = 1.2
        
        # 止盈参数建议
        if avg_atr_price_ratio > 0.015:
            print("⚠️ 市场波动较高，建议增加止盈参数: ATR偏移0.2→0.25, 最小偏移0.02→0.025")
            suggested_tp_atr = 0.25
            suggested_tp_min = 0.025
        else:
            print("✅ 当前止盈参数适合当前市场波动")
            suggested_tp_atr = 0.2
            suggested_tp_min = 0.02
        
        # RSI参数建议
        if 'rsi_1m' in df_valid.columns and 'rsi_5m' in df_valid.columns:
            rsi_1m_var = rsi_1m_stats['max'] - rsi_1m_stats['min']
            rsi_5m_var = rsi_5m_stats['max'] - rsi_5m_stats['min']
            print(f"📊 RSI变异性 - 1m: {rsi_1m_var:.2f}, 5m: {rsi_5m_var:.2f}")
            
            if rsi_1m_var > 60:
                print("⚠️ 1分钟RSI变化剧烈，可能过于敏感，建议适当调整RSI阈值范围")
            else:
                print("✅ RSI变化在合理范围内")
        
        # 网格参数建议
        print(f"📊 平均网格宽度: {grid_stats['width_mean']:.3f}")
        
        if grid_stats['width_mean'] > 3:
            print("⚠️ 网格较宽，建议适当缩小网格间隔以增加交易机会")
        elif grid_stats['width_mean'] < 1:
            print("⚠️ 网格较窄，可考虑扩大网格间隔以适应波动")
        else:
            print("✅ 网格宽度适合当前市场")
        
        if 'final_decision' in df_valid.columns:
            trade_frequency = final_decision_true_pct
            print(f"📊 交易触发频率: {trade_frequency:.2f}%")
            if trade_frequency < 5:  # 如果交易频率低于5%
                print("⚠️ 交易频率较低，可考虑降低触发阈值")
            elif trade_frequency > 50:  # 如果交易频率高于50%
                print("⚠️ 交易频率较高，可考虑提高触发阈值")
            else:
                print("✅ 交易频率适中")
        
        # 算法优化建议
        print(f"\n🔧 网格交易算法优化建议:")
        print("-" * 50)
        
        # 分析near_lower、rsi_ok、trend_check等条件的触发频率
        conditions = ['near_lower', 'rsi_ok', 'trend_check', 'rebound', 'vol_ok']
        active_conditions = []
        
        for condition in conditions:
            if condition in df_valid.columns:
                true_pct = (df_valid[condition] == True).sum() / len(df_valid) * 100
                print(f"📊 {condition}触发频率: {true_pct:.2f}%")
                if true_pct < 5:
                    print(f"  ⚠️ {condition}触发频率过低，可能需要调整参数")
                elif true_pct > 80:
                    print(f"  ⚠️ {condition}触发频率过高，可能需要收紧条件")
                else:
                    print(f"  ✅ {condition}触发频率适中")
        
        # 基于final_decision分析整体策略效果
        if 'final_decision' in df_valid.columns:
            final_decision_rate = final_decision_true_pct
            print(f"\n📊 最终决策触发频率: {final_decision_rate:.2f}%")
            
            # 分析各条件对最终决策的贡献
            if all(cond in df_valid.columns for cond in ['near_lower', 'rsi_ok']):
                # 计算条件关联性
                for cond in ['near_lower', 'rsi_ok', 'trend_check', 'rebound', 'vol_ok']:
                    if cond in df_valid.columns:
                        # 计算当条件为True时，最终决策为True的概率
                        if (df_valid[cond] == True).sum() > 0:
                            prob = ((df_valid[cond] == True) & (df_valid['final_decision'] == True)).sum() / (df_valid[cond] == True).sum()
                            print(f"  📊 当{cond}=True时，最终决策为True的概率: {prob:.2%}")
        
        print(f"\n🎯 综合参数优化建议:")
        print(f"  - STOP_LOSS_MULTIPLIER: 1.2 → {suggested_stop_loss_mult}")
        print(f"  - TAKE_PROFIT_ATR_OFFSET: 0.2 → {suggested_tp_atr}")
        print(f"  - TAKE_PROFIT_MIN_OFFSET: 0.02 → {suggested_tp_min}")
        print(f"  - GRID_MAX_POSITION: 3 (保持不变，根据账户资金调整)")
        print(f"  - DAILY_LOSS_LIMIT: $1200 (根据账户规模调整)")
        
        print(f"\n📈 动态调整策略:")
        print(f"  - 根据ATR/价格比率({avg_atr_price_ratio:.4f})，可考虑动态调整网格参数")
        print(f"  - 当RSI变异性大于60时，考虑调整RSI阈值范围")
        print(f"  - 根据交易频率({trade_frequency:.2f}%)，动态调整触发条件")
        print(f"  - 建议在高波动时期降低单笔交易量以控制风险")
        
        print(f"\n📊 建议定期监控的指标:")
        print(f"  - ATR/价格比率: 当前 {avg_atr_price_ratio:.4f}")
        print(f"  - RSI 1分钟和5分钟的变异性")
        print(f"  - 网格宽度变化")
        print(f"  - 价格日变化率")
        print(f"  - 各条件触发频率")
        
        print(f"\n✅ 详细数据分析完成！")
        print(f"📈 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except FileNotFoundError:
        print("❌ 未找到 trading_data.csv 文件")
    except Exception as e:
        print(f"❌ 分析过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_data_freshness_and_optimize()