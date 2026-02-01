#!/usr/bin/env python3
"""
大规模1分钟K线数据采集 - 目标2万+条数据
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DataConfig, LabelConfig, DataSplitConfig, FeatureConfig
from tiger1 import get_kline_data, calculate_indicators, FUTURE_SYMBOL


class Massive1MinCollector:
    """大规模1分钟K线数据采集器"""
    
    def __init__(self, days=90, output_dir='/home/cx/trading_data/massive_1min'):
        """
        初始化
        
        Args:
            days: 获取天数（白银期货SIL.COMEX.202603）
            output_dir: 输出目录
        """
        self.days = days
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.log_file = os.path.join(output_dir, f'collection_log_{self.timestamp}.txt')
        self._log(f"🚀 初始化大规模1分钟数据采集器")
        self._log(f"   交易标的: 白银期货 SIL.COMEX.202603")
        self._log(f"   目标天数: {days} 天")
        self._log(f"   预计1分钟数据量: ~{days * 6 * 60:,} 条（按每天约6小时交易估算）")
    
    def _log(self, message):
        """记录日志"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
    
    def fetch_1min_data(self):
        """获取1分钟K线数据"""
        self._log(f"{'='*80}")
        self._log(f"📥 开始获取1分钟K线数据...")
        
        # 计算需要的数量
        # 白银期货交易时间有限，保守估计每天约6小时有效交易时间
        # 但我们请求更多数据，让API返回实际可用的
        expected_count = self.days * 1440  # 请求足够多，API会返回实际有的
        
        self._log(f"   请求数量: {expected_count:,} 条1分钟K线")
        
        try:
            # 获取1分钟数据
            df_1min = get_kline_data([FUTURE_SYMBOL], "1min", count=expected_count)
            
            if df_1min.empty:
                self._log(f"❌ 1分钟数据为空")
                return None
            
            self._log(f"✅ 成功获取1分钟数据: {len(df_1min):,} 条")
            self._log(f"   时间范围: {df_1min.index[0]} 至 {df_1min.index[-1]}")
            
            # 同时获取5分钟数据用于计算一些指标
            expected_5min = self.days * 24 * 12  # 5分钟数据
            df_5min = get_kline_data([FUTURE_SYMBOL], "5min", count=expected_5min)
            
            if df_5min.empty:
                self._log(f"⚠️ 5分钟数据为空，将只使用1分钟数据")
                df_5min = None
            else:
                self._log(f"✅ 成功获取5分钟数据: {len(df_5min):,} 条")
            
            return df_1min, df_5min
            
        except Exception as e:
            self._log(f"❌ 获取数据失败: {e}")
            import traceback
            self._log(traceback.format_exc())
            return None, None
    
    def calculate_features_from_1min(self, df_1min, df_5min):
        """从1分钟数据计算特征"""
        self._log(f"{'='*80}")
        self._log(f"🔧 开始计算特征...")
        
        features_list = []
        min_len = 60  # 至少需要60根1分钟K线（1小时）
        
        if len(df_1min) < min_len:
            self._log(f"❌ 数据不足，需要至少 {min_len} 条")
            return pd.DataFrame()
        
        self._log(f"   1分钟数据量: {len(df_1min):,}")
        if df_5min is not None:
            self._log(f"   5分钟数据量: {len(df_5min):,}")
        
        window_size = 60  # 使用60分钟作为窗口
        total = len(df_1min) - min_len
        
        self._log(f"   开始计算 {total:,} 条特征...")
        
        for i in range(min_len, len(df_1min)):
            if (i - min_len) % 1000 == 0:
                progress = (i - min_len) / total * 100
                self._log(f"   进度: {progress:.1f}% ({i-min_len:,}/{total:,})")
            
            try:
                # 获取1分钟数据窗口
                window_1m = df_1min.iloc[max(0, i-window_size):i+1]
                current_timestamp = df_1min.index[i]
                
                # 获取对应的5分钟数据
                if df_5min is not None:
                    df_5min_slice = df_5min[df_5min.index <= current_timestamp]
                    if len(df_5min_slice) >= 20:
                        window_5m = df_5min_slice.iloc[-20:]
                    else:
                        window_5m = None
                else:
                    window_5m = None
                
                # 计算指标
                if window_5m is not None and len(window_5m) > 0:
                    inds = calculate_indicators(window_5m, window_1m)
                else:
                    # 如果没有5分钟数据，用1分钟数据代替
                    inds = calculate_indicators(window_1m, window_1m)
                
                if '1m' not in inds:
                    continue
                
                # 提取特征
                price_current = inds['1m']['close']
                
                # ATR（从5分钟或1分钟）
                if '5m' in inds and 'atr' in inds['5m']:
                    atr = inds['5m']['atr']
                else:
                    atr = inds['1m'].get('atr', price_current * 0.01)
                
                # RSI
                rsi_1m = inds['1m'].get('rsi', 50)
                rsi_5m = inds.get('5m', {}).get('rsi', 50)
                
                # 布林带
                boll_upper = inds.get('5m', {}).get('boll_upper', 0)
                boll_mid = inds.get('5m', {}).get('boll_mid', price_current)
                boll_lower = inds.get('5m', {}).get('boll_lower', 0)
                
                # 成交量
                volume_1m = inds['1m'].get('volume', 0)
                
                # 计算价格变化率
                if len(window_1m) >= 2:
                    price_change_1 = (price_current - window_1m['close'].iloc[-2]) / window_1m['close'].iloc[-2] * 100
                else:
                    price_change_1 = 0
                
                if len(window_1m) >= 6:
                    price_change_5 = (price_current - window_1m['close'].iloc[-6]) / window_1m['close'].iloc[-6] * 100
                else:
                    price_change_5 = 0
                
                # 波动率（基于最近60分钟）
                volatility = window_1m['close'].std() / window_1m['close'].mean() * 100 if len(window_1m) > 1 else 0
                
                # 布林带位置
                if (boll_upper - boll_lower) > 0:
                    boll_position = (price_current - boll_lower) / (boll_upper - boll_lower)
                else:
                    boll_position = 0.5
                
                # 网格参数（简化版）
                grid_upper = price_current * 1.005  # 0.5%
                grid_lower = price_current * 0.995
                
                features = {
                    'timestamp': current_timestamp,
                    'price_current': price_current,
                    'atr': atr,
                    'rsi_1m': rsi_1m,
                    'rsi_5m': rsi_5m,
                    'boll_upper': boll_upper,
                    'boll_mid': boll_mid,
                    'boll_lower': boll_lower,
                    'boll_position': boll_position,
                    'volume': volume_1m,
                    'price_change_1min': price_change_1,
                    'price_change_5min': price_change_5,
                    'volatility': volatility,
                }
                
                features_list.append(features)
                
            except Exception as e:
                if (i - min_len) % 1000 == 0:
                    self._log(f"   ⚠️ 计算特征出错 (索引 {i}): {e}")
                continue
        
        df_features = pd.DataFrame(features_list)
        self._log(f"✅ 特征计算完成: {len(df_features):,} 条记录")
        
        return df_features
    
    def generate_labels(self, df, look_ahead=5):
        """生成训练标签"""
        self._log(f"{'='*80}")
        self._log(f"🏷️  开始生成标签...")
        self._log(f"   前瞻周期: {look_ahead} 分钟")
        
        df = df.copy()
        df['label'] = 0  # 0=持有, 1=买入, 2=卖出
        
        # 使用价格变化作为标签
        buy_threshold = 0.3  # 0.3%上涨
        sell_threshold = -0.3  # 0.3%下跌
        
        for i in range(len(df) - look_ahead):
            current_price = df.iloc[i]['price_current']
            future_price = df.iloc[i + look_ahead]['price_current']
            price_change_pct = (future_price - current_price) / current_price * 100
            
            if price_change_pct > buy_threshold:
                df.iloc[i, df.columns.get_loc('label')] = 1  # 买入
            elif price_change_pct < sell_threshold:
                df.iloc[i, df.columns.get_loc('label')] = 2  # 卖出
        
        # 统计标签分布
        label_counts = df['label'].value_counts().sort_index()
        total = len(df)
        
        self._log(f"\n   标签分布:")
        self._log(f"   持有(0): {label_counts.get(0, 0):,} ({label_counts.get(0, 0)/total*100:.1f}%)")
        self._log(f"   买入(1): {label_counts.get(1, 0):,} ({label_counts.get(1, 0)/total*100:.1f}%)")
        self._log(f"   卖出(2): {label_counts.get(2, 0):,} ({label_counts.get(2, 0)/total*100:.1f}%)")
        
        return df
    
    def split_dataset(self, df):
        """划分数据集（使用分层采样）"""
        self._log(f"{'='*80}")
        self._log(f"📊 划分数据集...")
        
        from sklearn.model_selection import train_test_split
        
        # 第一次分割：分离测试集
        train_val, test = train_test_split(
            df,
            test_size=0.15,
            random_state=42,
            stratify=df['label']
        )
        
        # 第二次分割：分离训练集和验证集
        train, val = train_test_split(
            train_val,
            test_size=0.15 / 0.85,  # 确保验证集占总数据的15%
            random_state=42,
            stratify=train_val['label']
        )
        
        self._log(f"\n   数据集大小:")
        self._log(f"   训练集: {len(train):,} 条 ({len(train)/len(df)*100:.1f}%)")
        self._log(f"   验证集: {len(val):,} 条 ({len(val)/len(df)*100:.1f}%)")
        self._log(f"   测试集: {len(test):,} 条 ({len(test)/len(df)*100:.1f}%)")
        
        # 打印各集的标签分布
        for name, data in [('训练集', train), ('验证集', val), ('测试集', test)]:
            counts = data['label'].value_counts().sort_index()
            self._log(f"\n   {name}标签分布:")
            self._log(f"     持有(0): {counts.get(0, 0):,} ({counts.get(0, 0)/len(data)*100:.1f}%)")
            self._log(f"     买入(1): {counts.get(1, 0):,} ({counts.get(1, 0)/len(data)*100:.1f}%)")
            self._log(f"     卖出(2): {counts.get(2, 0):,} ({counts.get(2, 0)/len(data)*100:.1f}%)")
        
        return train, val, test
    
    def save_datasets(self, train, val, test, full_df):
        """保存数据集"""
        self._log(f"{'='*80}")
        self._log(f"💾 保存数据集...")
        
        files = {}
        
        train_file = os.path.join(self.output_dir, f'train_{self.timestamp}.csv')
        train.to_csv(train_file, index=True)
        self._log(f"   ✅ 训练集: {train_file}")
        files['train'] = train_file
        
        val_file = os.path.join(self.output_dir, f'val_{self.timestamp}.csv')
        val.to_csv(val_file, index=True)
        self._log(f"   ✅ 验证集: {val_file}")
        files['val'] = val_file
        
        test_file = os.path.join(self.output_dir, f'test_{self.timestamp}.csv')
        test.to_csv(test_file, index=True)
        self._log(f"   ✅ 测试集: {test_file}")
        files['test'] = test_file
        
        full_file = os.path.join(self.output_dir, f'full_{self.timestamp}.csv')
        full_df.to_csv(full_file, index=True)
        self._log(f"   ✅ 完整数据: {full_file}")
        files['full'] = full_file
        
        return files
    
    def run(self):
        """运行完整流程"""
        try:
            # 1. 获取数据
            result = self.fetch_1min_data()
            if result is None:
                return None
            
            df_1min, df_5min = result
            
            # 2. 计算特征
            df_features = self.calculate_features_from_1min(df_1min, df_5min)
            
            if df_features.empty:
                self._log("❌ 特征计算失败")
                return None
            
            # 3. 生成标签
            df_labeled = self.generate_labels(df_features, look_ahead=5)
            
            # 4. 划分数据集
            train, val, test = self.split_dataset(df_labeled)
            
            # 5. 保存数据集
            files = self.save_datasets(train, val, test, df_labeled)
            
            self._log(f"\n{'='*80}")
            self._log(f"✅ 数据采集完成！")
            self._log(f"{'='*80}")
            self._log(f"\n   总数据量: {len(df_labeled):,} 条")
            self._log(f"   日志文件: {self.log_file}")
            
            return files
            
        except Exception as e:
            self._log(f"❌ 数据采集出错: {e}")
            import traceback
            self._log(traceback.format_exc())
            return None


def main():
    parser = argparse.ArgumentParser(description='大规模1分钟K线数据采集')
    parser.add_argument('--days', type=int, default=90, help='获取天数（默认90天）')
    parser.add_argument('--output-dir', type=str, default='/home/cx/trading_data/massive_1min', help='输出目录')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"🚀 大规模1分钟K线数据采集")
    print(f"{'='*80}")
    print(f"   目标天数: {args.days} 天")
    print(f"   预计数据量: {args.days * 24 * 60:,} 条")
    print(f"{'='*80}\n")
    
    collector = Massive1MinCollector(days=args.days, output_dir=args.output_dir)
    files = collector.run()
    
    if files:
        print(f"\n{'='*80}")
        print(f"✅ 数据采集成功！可以开始训练模型")
        print(f"{'='*80}")
        print(f"\n训练命令示例:")
        print(f"python train_all_real_models.py --train-file {files['train']} --val-file {files['val']}")


if __name__ == "__main__":
    main()
