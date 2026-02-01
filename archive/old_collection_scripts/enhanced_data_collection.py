#!/usr/bin/env python3
"""
增强版数据采集工具 - 改进标注策略和训练/验证集划分
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入tiger1模块的必要函数
from tiger1 import (
    get_kline_data, calculate_indicators, 
    FUTURE_SYMBOL, data_collector
)


class EnhancedDataCollector:
    """增强版数据收集器"""
    
    def __init__(self, days=30, output_dir='/home/cx/trading_data/enhanced'):
        """
        初始化数据收集器
        
        Args:
            days: 获取的天数
            output_dir: 输出目录
        """
        self.days = days
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建日志文件
        self.log_file = os.path.join(output_dir, f'collection_log_{self.timestamp}.txt')
        self._log(f"初始化数据收集器 - 将获取 {days} 天的数据")
    
    def _log(self, message):
        """记录日志"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
    
    def fetch_historical_data(self):
        """获取历史K线数据"""
        self._log(f"=" * 80)
        self._log(f"开始获取历史数据...")
        
        # 计算需要的K线数量
        counts = {
            '1min': self.days * 400,
            '5min': self.days * 100,
        }
        
        historical_data = {}
        
        for period in ['1min', '5min']:
            try:
                self._log(f"  正在获取 {period} 数据...")
                count = counts.get(period, 1000)
                df = get_kline_data([FUTURE_SYMBOL], period, count=count)
                
                if not df.empty:
                    historical_data[period] = df
                    self._log(f"  ✅ {period} 数据获取成功: {len(df)} 条记录")
                    self._log(f"     时间范围: {df.index[0]} 到 {df.index[-1]}")
                else:
                    self._log(f"  ⚠️ {period} 数据为空")
                    
            except Exception as e:
                self._log(f"  ❌ 获取 {period} 数据失败: {e}")
                import traceback
                self._log(traceback.format_exc())
        
        return historical_data
    
    def calculate_features_batch(self, df_5m, df_1m):
        """批量计算特征"""
        self._log(f"=" * 80)
        self._log(f"开始批量计算特征...")
        
        features_list = []
        min_len = 50
        
        if len(df_5m) < min_len or len(df_1m) < min_len:
            self._log(f"⚠️ 数据不足，需要至少 {min_len} 条")
            return pd.DataFrame()
        
        self._log(f"数据量: 5分钟={len(df_5m)}, 1分钟={len(df_1m)}")
        
        window_size = 20
        total = len(df_5m) - min_len
        
        for i in range(min_len, len(df_5m)):
            if (i - min_len) % 100 == 0:
                progress = (i - min_len) / total * 100
                self._log(f"  进度: {progress:.1f}% ({i-min_len}/{total})")
            
            try:
                window_5m = df_5m.iloc[max(0, i-window_size):i+1]
                timestamp_5m = df_5m.index[i]
                df_1m_slice = df_1m[df_1m.index <= timestamp_5m]
                
                if len(df_1m_slice) < min_len:
                    continue
                
                window_1m = df_1m_slice.iloc[-window_size:]
                inds = calculate_indicators(window_5m, window_1m)
                
                if '5m' not in inds or '1m' not in inds:
                    continue
                
                price_current = inds['1m']['close']
                atr = inds['5m']['atr']
                rsi_1m = inds['1m']['rsi']
                rsi_5m = inds['5m']['rsi']
                
                grid_upper = price_current * 1.01
                grid_lower = price_current * 0.99
                buffer = max(atr * 0.3, 0.0025)
                threshold = grid_lower + buffer
                
                boll_upper = inds['5m'].get('boll_upper', 0)
                boll_mid = inds['5m'].get('boll_mid', 0)
                boll_lower = inds['5m'].get('boll_lower', 0)
                
                # 计算更多特征
                volume_1m = inds['1m'].get('volume', 0)
                
                # 价格动量特征
                if len(window_5m) > 1:
                    price_change_1 = (price_current - window_5m['close'].iloc[-2]) / window_5m['close'].iloc[-2] * 100
                    price_change_5 = (price_current - window_5m['close'].iloc[-6]) / window_5m['close'].iloc[-6] * 100 if len(window_5m) > 5 else 0
                else:
                    price_change_1 = 0
                    price_change_5 = 0
                
                # 波动率特征
                volatility = window_5m['close'].std() / window_5m['close'].mean() * 100 if len(window_5m) > 1 else 0
                
                # 布林带位置
                boll_position = (price_current - boll_lower) / (boll_upper - boll_lower) if (boll_upper - boll_lower) > 0 else 0.5
                
                features = {
                    'timestamp': timestamp_5m,
                    'price_current': price_current,
                    'grid_lower': grid_lower,
                    'grid_upper': grid_upper,
                    'atr': atr,
                    'rsi_1m': rsi_1m,
                    'rsi_5m': rsi_5m,
                    'buffer': buffer,
                    'threshold': threshold,
                    'near_lower': price_current <= threshold,
                    'rsi_ok': rsi_1m < 30 or (rsi_5m > 45 and rsi_5m < 55),
                    'boll_upper': boll_upper,
                    'boll_mid': boll_mid,
                    'boll_lower': boll_lower,
                    'boll_position': boll_position,
                    'volume_1m': volume_1m,
                    'price_change_1': price_change_1,
                    'price_change_5': price_change_5,
                    'volatility': volatility,
                }
                
                features_list.append(features)
                
            except Exception as e:
                if i % 100 == 0:
                    self._log(f"  ⚠️ 计算特征时出错 (索引 {i}): {e}")
                continue
        
        df_features = pd.DataFrame(features_list)
        self._log(f"✅ 特征计算完成: {len(df_features)} 条记录")
        
        return df_features
    
    def generate_labels_multi_strategy(self, df):
        """
        使用多种策略生成标签，并进行对比分析
        
        Args:
            df: 特征DataFrame
        
        Returns:
            DataFrame: 添加了多种标签的数据
        """
        self._log(f"=" * 80)
        self._log(f"开始生成训练标签（多策略）...")
        
        df = df.copy()
        
        # 策略1: 固定阈值（原始方法）
        df = self._label_fixed_threshold(df, look_ahead=5, buy_threshold=0.5, sell_threshold=-0.5)
        
        # 策略2: 动态阈值（基于ATR）
        df = self._label_dynamic_threshold(df)
        
        # 策略3: 趋势跟踪
        df = self._label_trend_following(df)
        
        # 策略4: 布林带突破
        df = self._label_bollinger_breakout(df)
        
        # 策略5: 综合策略（投票机制）
        df = self._label_ensemble(df)
        
        # 打印各策略的标签分布对比
        self._print_label_comparison(df)
        
        return df
    
    def _label_fixed_threshold(self, df, look_ahead=5, buy_threshold=0.5, sell_threshold=-0.5):
        """策略1: 固定阈值标注"""
        self._log(f"  策略1: 固定阈值标注 (阈值={buy_threshold}%/-{abs(sell_threshold)}%)")
        
        df['label_fixed'] = 0
        
        for i in range(len(df) - look_ahead):
            current_price = df.iloc[i]['price_current']
            future_price = df.iloc[i + look_ahead]['price_current']
            price_change_pct = (future_price - current_price) / current_price * 100
            
            if price_change_pct > buy_threshold:
                df.iloc[i, df.columns.get_loc('label_fixed')] = 1
            elif price_change_pct < sell_threshold:
                df.iloc[i, df.columns.get_loc('label_fixed')] = 2
        
        return df
    
    def _label_dynamic_threshold(self, df, look_ahead=5):
        """策略2: 动态阈值（基于ATR的自适应阈值）"""
        self._log(f"  策略2: 动态阈值标注 (基于ATR)")
        
        df['label_dynamic'] = 0
        
        for i in range(len(df) - look_ahead):
            current_price = df.iloc[i]['price_current']
            future_price = df.iloc[i + look_ahead]['price_current']
            atr = df.iloc[i]['atr']
            
            # 动态阈值：使用ATR的倍数
            buy_threshold = (atr / current_price) * 100 * 0.5  # ATR的50%
            sell_threshold = -(atr / current_price) * 100 * 0.5
            
            price_change_pct = (future_price - current_price) / current_price * 100
            
            if price_change_pct > buy_threshold:
                df.iloc[i, df.columns.get_loc('label_dynamic')] = 1
            elif price_change_pct < sell_threshold:
                df.iloc[i, df.columns.get_loc('label_dynamic')] = 2
        
        return df
    
    def _label_trend_following(self, df, look_ahead=5):
        """策略3: 趋势跟踪（结合RSI和价格动量）"""
        self._log(f"  策略3: 趋势跟踪标注 (RSI+动量)")
        
        df['label_trend'] = 0
        
        for i in range(len(df) - look_ahead):
            current_price = df.iloc[i]['price_current']
            future_price = df.iloc[i + look_ahead]['price_current']
            rsi_1m = df.iloc[i]['rsi_1m']
            rsi_5m = df.iloc[i]['rsi_5m']
            price_change_1 = df.iloc[i]['price_change_1']
            
            price_change_pct = (future_price - current_price) / current_price * 100
            
            # 买入条件：RSI超卖 + 短期上涨动量
            if rsi_1m < 30 and price_change_1 > 0 and price_change_pct > 0.3:
                df.iloc[i, df.columns.get_loc('label_trend')] = 1
            # 卖出条件：RSI超买 + 短期下跌动量
            elif rsi_1m > 70 and price_change_1 < 0 and price_change_pct < -0.3:
                df.iloc[i, df.columns.get_loc('label_trend')] = 2
        
        return df
    
    def _label_bollinger_breakout(self, df, look_ahead=5):
        """策略4: 布林带突破"""
        self._log(f"  策略4: 布林带突破标注")
        
        df['label_boll'] = 0
        
        for i in range(len(df) - look_ahead):
            price_current = df.iloc[i]['price_current']
            future_price = df.iloc[i + look_ahead]['price_current']
            boll_position = df.iloc[i]['boll_position']
            
            price_change_pct = (future_price - price_current) / price_current * 100
            
            # 买入条件：接近下轨且未来上涨
            if boll_position < 0.2 and price_change_pct > 0.3:
                df.iloc[i, df.columns.get_loc('label_boll')] = 1
            # 卖出条件：接近上轨且未来下跌
            elif boll_position > 0.8 and price_change_pct < -0.3:
                df.iloc[i, df.columns.get_loc('label_boll')] = 2
        
        return df
    
    def _label_ensemble(self, df):
        """策略5: 综合标注（投票机制）"""
        self._log(f"  策略5: 综合标注 (投票机制)")
        
        # 使用投票机制
        df['label_ensemble'] = 0
        
        label_cols = ['label_fixed', 'label_dynamic', 'label_trend', 'label_boll']
        
        for i in range(len(df)):
            votes = df.loc[df.index[i], label_cols].values
            
            # 统计投票
            buy_votes = np.sum(votes == 1)
            sell_votes = np.sum(votes == 2)
            
            # 需要至少2票才确定
            if buy_votes >= 2:
                df.iloc[i, df.columns.get_loc('label_ensemble')] = 1
            elif sell_votes >= 2:
                df.iloc[i, df.columns.get_loc('label_ensemble')] = 2
        
        return df
    
    def _print_label_comparison(self, df):
        """打印各策略的标签分布对比"""
        self._log(f"\n  各策略标签分布对比:")
        self._log(f"  {'-' * 80}")
        self._log(f"  {'策略名称':<20} {'持有(0)':<15} {'买入(1)':<15} {'卖出(2)':<15}")
        self._log(f"  {'-' * 80}")
        
        strategies = {
            'label_fixed': '固定阈值',
            'label_dynamic': '动态阈值(ATR)',
            'label_trend': '趋势跟踪',
            'label_boll': '布林带突破',
            'label_ensemble': '综合投票'
        }
        
        for col, name in strategies.items():
            if col in df.columns:
                counts = df[col].value_counts()
                hold = counts.get(0, 0)
                buy = counts.get(1, 0)
                sell = counts.get(2, 0)
                total = len(df)
                
                self._log(f"  {name:<20} {hold:>6} ({hold/total*100:>5.1f}%)  "
                         f"{buy:>6} ({buy/total*100:>5.1f}%)  "
                         f"{sell:>6} ({sell/total*100:>5.1f}%)")
        
        self._log(f"  {'-' * 80}")
    
    def split_train_val_test(self, df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        划分训练集、验证集和测试集
        
        Args:
            df: 完整数据集
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
        
        Returns:
            train_df, val_df, test_df
        """
        self._log(f"=" * 80)
        self._log(f"划分数据集...")
        self._log(f"  训练集: {train_ratio*100:.0f}%")
        self._log(f"  验证集: {val_ratio*100:.0f}%")
        self._log(f"  测试集: {test_ratio*100:.0f}%")
        
        # 按时间顺序划分（不随机打乱）
        total = len(df)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        
        train_df = df.iloc[:train_size].copy()
        val_df = df.iloc[train_size:train_size+val_size].copy()
        test_df = df.iloc[train_size+val_size:].copy()
        
        self._log(f"\n  数据集大小:")
        self._log(f"    训练集: {len(train_df)} 条")
        self._log(f"    验证集: {len(val_df)} 条")
        self._log(f"    测试集: {len(test_df)} 条")
        
        # 打印各集的标签分布（使用ensemble标签）
        for name, data in [('训练集', train_df), ('验证集', val_df), ('测试集', test_df)]:
            if 'label_ensemble' in data.columns:
                counts = data['label_ensemble'].value_counts()
                self._log(f"\n  {name}标签分布:")
                self._log(f"    持有: {counts.get(0, 0)} ({counts.get(0, 0)/len(data)*100:.1f}%)")
                self._log(f"    买入: {counts.get(1, 0)} ({counts.get(1, 0)/len(data)*100:.1f}%)")
                self._log(f"    卖出: {counts.get(2, 0)} ({counts.get(2, 0)/len(data)*100:.1f}%)")
        
        return train_df, val_df, test_df
    
    def save_datasets(self, train_df, val_df, test_df, full_df):
        """保存所有数据集"""
        self._log(f"=" * 80)
        self._log(f"保存数据集...")
        
        files = {}
        
        # 保存训练集
        train_file = os.path.join(self.output_dir, f'train_{self.timestamp}.csv')
        train_df.to_csv(train_file, index=True, encoding='utf-8')
        self._log(f"  ✅ 训练集: {train_file}")
        files['train'] = train_file
        
        # 保存验证集
        val_file = os.path.join(self.output_dir, f'val_{self.timestamp}.csv')
        val_df.to_csv(val_file, index=True, encoding='utf-8')
        self._log(f"  ✅ 验证集: {val_file}")
        files['val'] = val_file
        
        # 保存测试集
        test_file = os.path.join(self.output_dir, f'test_{self.timestamp}.csv')
        test_df.to_csv(test_file, index=True, encoding='utf-8')
        self._log(f"  ✅ 测试集: {test_file}")
        files['test'] = test_file
        
        # 保存完整数据集
        full_file = os.path.join(self.output_dir, f'full_{self.timestamp}.csv')
        full_df.to_csv(full_file, index=True, encoding='utf-8')
        self._log(f"  ✅ 完整数据: {full_file}")
        files['full'] = full_file
        
        return files
    
    def generate_summary_report(self, df, files):
        """生成数据收集总结报告"""
        report_file = os.path.join(self.output_dir, f'data_summary_{self.timestamp}.md')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 数据收集总结报告\n\n")
            f.write(f"**生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
            
            f.write("## 📊 数据统计\n\n")
            f.write(f"- **总记录数:** {len(df)}\n")
            f.write(f"- **特征数量:** {len(df.columns)}\n")
            f.write(f"- **时间范围:** {df['timestamp'].min()} 到 {df['timestamp'].max()}\n\n")
            
            f.write("## 🏷️ 标注策略对比\n\n")
            
            strategies = {
                'label_fixed': '固定阈值',
                'label_dynamic': '动态阈值(ATR)',
                'label_trend': '趋势跟踪',
                'label_boll': '布林带突破',
                'label_ensemble': '综合投票'
            }
            
            f.write("| 策略 | 持有(0) | 买入(1) | 卖出(2) |\n")
            f.write("|------|---------|---------|----------|\n")
            
            for col, name in strategies.items():
                if col in df.columns:
                    counts = df[col].value_counts()
                    hold = counts.get(0, 0)
                    buy = counts.get(1, 0)
                    sell = counts.get(2, 0)
                    f.write(f"| {name} | {hold} ({hold/len(df)*100:.1f}%) | "
                           f"{buy} ({buy/len(df)*100:.1f}%) | "
                           f"{sell} ({sell/len(df)*100:.1f}%) |\n")
            
            f.write("\n## 📁 生成的文件\n\n")
            for key, path in files.items():
                f.write(f"- **{key}:** `{path}`\n")
            
            f.write("\n## 💡 建议\n\n")
            f.write("### 推荐使用的标注策略\n\n")
            
            # 分析哪个策略最平衡
            best_strategy = None
            best_balance = float('inf')
            
            for col in strategies.keys():
                if col in df.columns:
                    counts = df[col].value_counts()
                    buy = counts.get(1, 0)
                    sell = counts.get(2, 0)
                    # 计算不平衡度
                    if buy + sell > 0:
                        balance = abs(buy - sell) / (buy + sell)
                        if balance < best_balance and (buy + sell) / len(df) > 0.1:
                            best_balance = balance
                            best_strategy = col
            
            if best_strategy:
                f.write(f"**推荐策略:** {strategies[best_strategy]}\n\n")
                f.write(f"原因: 该策略的买入/卖出信号比例最平衡，不平衡度为 {best_balance:.3f}\n\n")
            
            f.write("### 下一步行动\n\n")
            f.write("1. 使用推荐的标注策略训练模型\n")
            f.write("2. 在验证集上调整超参数\n")
            f.write("3. 在测试集上评估最终性能\n")
            f.write("4. 如果某个策略表现特别好，可以单独使用\n")
            f.write("5. 综合投票策略通常更稳健\n")
        
        self._log(f"  ✅ 总结报告: {report_file}")
        return report_file
    
    def run(self):
        """运行完整的数据收集流程"""
        try:
            # 1. 获取历史数据
            historical_data = self.fetch_historical_data()
            
            if not historical_data or '5min' not in historical_data or '1min' not in historical_data:
                self._log("❌ 数据获取失败")
                return None
            
            # 2. 计算特征
            df_features = self.calculate_features_batch(
                historical_data['5min'],
                historical_data['1min']
            )
            
            if df_features.empty:
                self._log("❌ 特征计算失败")
                return None
            
            # 3. 生成多种标注
            df_labeled = self.generate_labels_multi_strategy(df_features)
            
            # 4. 划分数据集
            train_df, val_df, test_df = self.split_train_val_test(df_labeled)
            
            # 5. 保存数据集
            files = self.save_datasets(train_df, val_df, test_df, df_labeled)
            
            # 6. 生成总结报告
            report_file = self.generate_summary_report(df_labeled, files)
            
            self._log(f"\n" + "=" * 80)
            self._log(f"✅ 数据收集完成！")
            self._log(f"=" * 80)
            self._log(f"\n日志文件: {self.log_file}")
            self._log(f"总结报告: {report_file}")
            
            return files
            
        except Exception as e:
            self._log(f"❌ 数据收集过程出错: {e}")
            import traceback
            self._log(traceback.format_exc())
            return None


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("📥 增强版数据采集工具")
    print("=" * 80)
    
    # 解析命令行参数
    days = 30
    if len(sys.argv) > 1:
        try:
            days = int(sys.argv[1])
        except ValueError:
            print("⚠️ 无效的天数，使用默认值30天")
    
    # 创建收集器并运行
    collector = EnhancedDataCollector(days=days)
    files = collector.run()
    
    if files:
        print("\n" + "=" * 80)
        print("✅ 所有数据已准备就绪，可以开始训练模型！")
        print("=" * 80)


if __name__ == "__main__":
    main()
