#!/usr/bin/env python3
"""
大规模数据采集脚本 - 支持获取10万+数据
使用真实API或扩展模拟数据
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DataConfig, LabelConfig, DataSplitConfig, FeatureConfig
from tiger1 import get_kline_data, calculate_indicators, FUTURE_SYMBOL


class LargeDatasetCollector:
    """大规模数据集采集器"""
    
    def __init__(self, use_real_api=False, days=30, max_records=100000):
        """
        初始化
        
        Args:
            use_real_api: 是否使用真实API
            days: 获取天数
            max_records: 最大记录数
        """
        self.use_real_api = use_real_api
        self.days = days
        self.max_records = max_records
        
        # 更新配置
        DataConfig.USE_REAL_API = use_real_api
        DataConfig.DAYS_TO_FETCH = days
        DataConfig.MAX_RECORDS = max_records
        DataConfig.COUNT_1MIN = min(days * DataConfig.BARS_PER_DAY_1MIN, max_records)
        DataConfig.COUNT_5MIN = min(days * DataConfig.BARS_PER_DAY_5MIN, max_records // 5)
        
        # 创建输出目录
        self.output_dir = DataConfig.OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 创建日志文件
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(self.output_dir, f'collection_log_{timestamp}.txt')
        
        self._log("=" * 80)
        self._log("📥 大规模数据采集器初始化")
        self._log("=" * 80)
        self._log(f"使用真实API: {self.use_real_api}")
        self._log(f"目标天数: {self.days}")
        self._log(f"最大记录数: {self.max_records}")
        self._log(f"1分钟K线目标: {DataConfig.COUNT_1MIN}")
        self._log(f"5分钟K线目标: {DataConfig.COUNT_5MIN}")
        self._log("=" * 80)
    
    def _log(self, message):
        """记录日志"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
    
    def fetch_kline_data_with_retry(self, period, count, max_retries=3):
        """
        带重试的K线数据获取
        
        Args:
            period: 周期
            count: 数量
            max_retries: 最大重试次数
        
        Returns:
            DataFrame
        """
        for attempt in range(max_retries):
            try:
                self._log(f"  尝试获取 {period} 数据 (第 {attempt + 1}/{max_retries} 次)...")
                
                if self.use_real_api:
                    # 使用真实API获取数据
                    # 如果count很大，可能需要分批获取
                    if count > 10000:
                        self._log(f"    数据量大于10000，将分批获取...")
                        return self._fetch_in_batches(period, count)
                    else:
                        df = get_kline_data([FUTURE_SYMBOL], period, count=count)
                else:
                    # 模拟模式：扩展生成数据
                    self._log(f"    模拟模式：生成 {count} 条数据...")
                    df = self._generate_mock_data(period, count)
                
                if not df.empty:
                    self._log(f"  ✅ 成功获取 {len(df)} 条 {period} 数据")
                    return df
                else:
                    self._log(f"  ⚠️ 数据为空，重试...")
                    time.sleep(2)
                    
            except Exception as e:
                self._log(f"  ❌ 获取失败: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    import traceback
                    self._log(traceback.format_exc())
        
        return pd.DataFrame()
    
    def _fetch_in_batches(self, period, total_count, batch_size=10000):
        """
        分批获取大量数据
        
        Args:
            period: 周期
            total_count: 总数量
            batch_size: 批次大小
        
        Returns:
            DataFrame
        """
        all_data = []
        num_batches = (total_count + batch_size - 1) // batch_size
        
        self._log(f"    将分 {num_batches} 批获取，每批 {batch_size} 条")
        
        for i in range(num_batches):
            count = min(batch_size, total_count - i * batch_size)
            self._log(f"    批次 {i+1}/{num_batches}: 获取 {count} 条...")
            
            # 计算时间范围
            end_time = datetime.now() - timedelta(days=i * batch_size // DataConfig.BARS_PER_DAY_1MIN)
            start_time = end_time - timedelta(days=count // DataConfig.BARS_PER_DAY_1MIN)
            
            df = get_kline_data(
                [FUTURE_SYMBOL], 
                period, 
                count=count,
                start_time=start_time,
                end_time=end_time
            )
            
            if not df.empty:
                all_data.append(df)
                self._log(f"      ✅ 获取 {len(df)} 条")
            else:
                self._log(f"      ⚠️ 批次 {i+1} 数据为空")
            
            time.sleep(1)  # 避免API限流
        
        if all_data:
            result = pd.concat(all_data, ignore_index=False)
            result = result.sort_index()
            result = result[~result.index.duplicated(keep='first')]
            return result
        else:
            return pd.DataFrame()
    
    def _generate_mock_data(self, period, count):
        """
        生成模拟数据（用于Demo模式）
        
        Args:
            period: 周期
            count: 数量
        
        Returns:
            DataFrame
        """
        # 生成时间索引
        if period == '1min':
            freq = '1T'
        elif period == '5min':
            freq = '5T'
        else:
            freq = '1H'
        
        end_time = datetime.now()
        time_index = pd.date_range(end=end_time, periods=count, freq=freq)
        
        # 生成模拟价格数据（带趋势和波动）
        base_price = 90.0
        trend = np.linspace(0, 5, count)  # 上升趋势
        volatility = np.random.randn(count) * 0.5  # 随机波动
        prices = base_price + trend + volatility
        
        # 生成OHLC数据
        data = {
            'open': prices,
            'high': prices + np.abs(np.random.randn(count) * 0.2),
            'low': prices - np.abs(np.random.randn(count) * 0.2),
            'close': prices + np.random.randn(count) * 0.1,
            'volume': np.random.randint(100, 1000, count)
        }
        
        df = pd.DataFrame(data, index=time_index)
        return df
    
    def calculate_features_optimized(self, df_5m, df_1m):
        """
        优化的批量特征计算
        
        Args:
            df_5m: 5分钟数据
            df_1m: 1分钟数据
        
        Returns:
            DataFrame
        """
        self._log("\n" + "=" * 80)
        self._log("开始批量计算特征（优化版本）...")
        self._log(f"5分钟数据: {len(df_5m)} 条")
        self._log(f"1分钟数据: {len(df_1m)} 条")
        
        features_list = []
        min_len = DataConfig.MIN_REQUIRED_BARS
        window_size = DataConfig.WINDOW_SIZE
        
        if len(df_5m) < min_len or len(df_1m) < min_len:
            self._log(f"⚠️ 数据不足，需要至少 {min_len} 条")
            return pd.DataFrame()
        
        total = len(df_5m) - min_len
        last_progress = 0
        
        for i in range(min_len, len(df_5m)):
            # 进度更新
            progress = int((i - min_len) / total * 100)
            if progress >= last_progress + 10:
                self._log(f"  进度: {progress}% ({i-min_len}/{total})")
                last_progress = progress
            
            try:
                window_5m = df_5m.iloc[max(0, i-window_size):i+1]
                timestamp_5m = df_5m.index[i]
                df_1m_slice = df_1m[df_1m.index <= timestamp_5m]
                
                if len(df_1m_slice) < min_len:
                    continue
                
                window_1m = df_1m_slice.iloc[-window_size:]
                # 注意：calculate_indicators 参数顺序是 (df_1m, df_5m)
                inds = calculate_indicators(window_1m, window_5m)
                
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
                
                volume_1m = inds['1m'].get('volume', 0)
                
                # 价格动量特征
                if len(window_5m) > 1 and 'close' in window_5m.columns:
                    price_change_1 = (price_current - window_5m['close'].iloc[-2]) / window_5m['close'].iloc[-2] * 100
                    price_change_5 = (price_current - window_5m['close'].iloc[-6]) / window_5m['close'].iloc[-6] * 100 if len(window_5m) > 5 else 0
                else:
                    price_change_1 = 0
                    price_change_5 = 0
                
                # 波动率特征
                volatility = window_5m['close'].std() / window_5m['close'].mean() * 100 if (len(window_5m) > 1 and 'close' in window_5m.columns) else 0
                
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
                if i % 1000 == 0:
                    self._log(f"  ⚠️ 计算特征时出错 (索引 {i}): {e}")
                    import traceback
                    self._log(f"  详细错误:\n{traceback.format_exc()}")
                continue
        
        df_features = pd.DataFrame(features_list)
        self._log(f"✅ 特征计算完成: {len(df_features)} 条记录")
        
        return df_features
    
    def generate_labels(self, df):
        """生成标签（使用配置的策略）"""
        self._log("\n" + "=" * 80)
        self._log("生成训练标签...")
        self._log(f"使用策略: {LabelConfig.STRATEGY}")
        self._log(f"向前看周期: {LabelConfig.LOOK_AHEAD}")
        
        df = df.copy()
        look_ahead = LabelConfig.LOOK_AHEAD
        
        # 计算未来价格变化
        price_changes = []
        for i in range(len(df)):
            if i + look_ahead < len(df):
                current = df.iloc[i]['price_current']
                future = df.iloc[i + look_ahead]['price_current']
                pct_change = (future - current) / current * 100
            else:
                pct_change = 0
            price_changes.append(pct_change)
        
        df['future_price_change'] = price_changes
        
        # 根据策略生成标签
        if LabelConfig.STRATEGY == 'percentile':
            df = self._label_percentile(df, look_ahead)
        elif LabelConfig.STRATEGY == 'std':
            df = self._label_std(df, look_ahead)
        elif LabelConfig.STRATEGY == 'hybrid':
            df = self._label_percentile(df, look_ahead)
            df = self._label_std(df, look_ahead)
            df = self._label_hybrid(df)
        else:
            self._log(f"⚠️ 未知策略 {LabelConfig.STRATEGY}，使用百分位数")
            df = self._label_percentile(df, look_ahead)
        
        # 使用主标签列
        if 'label_' + LabelConfig.STRATEGY in df.columns:
            df['label'] = df['label_' + LabelConfig.STRATEGY]
        elif 'label_percentile' in df.columns:
            df['label'] = df['label_percentile']
        
        # 打印标签分布
        if 'label' in df.columns:
            counts = df['label'].value_counts().sort_index()
            self._log(f"\n标签分布:")
            for label, count in counts.items():
                label_name = {0: "持有", 1: "买入", 2: "卖出"}.get(label, "未知")
                self._log(f"  {label_name} ({label}): {count} ({count/len(df)*100:.1f}%)")
        
        return df
    
    def _label_percentile(self, df, look_ahead):
        """百分位数标注"""
        changes = df['future_price_change'].values[:-look_ahead]
        buy_threshold = np.percentile(changes, LabelConfig.PERCENTILE_BUY)
        sell_threshold = np.percentile(changes, LabelConfig.PERCENTILE_SELL)
        
        self._log(f"  百分位数阈值: 买入>{buy_threshold:.6f}%, 卖出<{sell_threshold:.6f}%")
        
        df['label_percentile'] = 0
        for i in range(len(df) - look_ahead):
            change = df.iloc[i]['future_price_change']
            if change > buy_threshold:
                df.iloc[i, df.columns.get_loc('label_percentile')] = 1
            elif change < sell_threshold:
                df.iloc[i, df.columns.get_loc('label_percentile')] = 2
        
        return df
    
    def _label_std(self, df, look_ahead):
        """标准差标注"""
        changes = df['future_price_change'].values[:-look_ahead]
        mean = changes.mean()
        std = changes.std()
        
        buy_threshold = mean + std * LabelConfig.STD_MULTIPLIER
        sell_threshold = mean - std * LabelConfig.STD_MULTIPLIER
        
        self._log(f"  标准差阈值: 买入>{buy_threshold:.6f}%, 卖出<{sell_threshold:.6f}%")
        
        df['label_std'] = 0
        for i in range(len(df) - look_ahead):
            change = df.iloc[i]['future_price_change']
            if change > buy_threshold:
                df.iloc[i, df.columns.get_loc('label_std')] = 1
            elif change < sell_threshold:
                df.iloc[i, df.columns.get_loc('label_std')] = 2
        
        return df
    
    def _label_hybrid(self, df):
        """混合标注"""
        df['label_hybrid'] = 0
        
        for i in range(len(df)):
            votes = []
            if 'label_percentile' in df.columns:
                votes.append(df.iloc[i]['label_percentile'])
            if 'label_std' in df.columns:
                votes.append(df.iloc[i]['label_std'])
            
            if len(votes) > 0:
                buy_votes = sum(1 for v in votes if v == 1)
                sell_votes = sum(1 for v in votes if v == 2)
                
                if buy_votes >= LabelConfig.VOTE_THRESHOLD:
                    df.iloc[i, df.columns.get_loc('label_hybrid')] = 1
                elif sell_votes >= LabelConfig.VOTE_THRESHOLD:
                    df.iloc[i, df.columns.get_loc('label_hybrid')] = 2
        
        return df
    
    def split_dataset(self, df):
        """划分数据集"""
        self._log("\n" + "=" * 80)
        self._log("划分数据集...")
        
        if DataSplitConfig.RANDOM_SPLIT:
            # 分层随机划分（确保各集标签分布相似）
            self._log("使用分层随机划分（保持标签分布均衡）")
            from sklearn.model_selection import train_test_split
            
            # 使用stratify参数进行分层采样
            train_val, test = train_test_split(
                df, 
                test_size=DataSplitConfig.TEST_RATIO,
                random_state=DataSplitConfig.RANDOM_SEED,
                stratify=df['label'] if 'label' in df.columns else None
            )
            train, val = train_test_split(
                train_val,
                test_size=DataSplitConfig.VAL_RATIO / (1 - DataSplitConfig.TEST_RATIO),
                random_state=DataSplitConfig.RANDOM_SEED,
                stratify=train_val['label'] if 'label' in train_val.columns else None
            )
        else:
            # 时间顺序划分
            self._log("使用时间顺序划分")
            n = len(df)
            train_end = int(n * DataSplitConfig.TRAIN_RATIO)
            val_end = int(n * (DataSplitConfig.TRAIN_RATIO + DataSplitConfig.VAL_RATIO))
            
            train = df.iloc[:train_end].copy()
            val = df.iloc[train_end:val_end].copy()
            test = df.iloc[val_end:].copy()
        
        self._log(f"  训练集: {len(train)} 条 ({len(train)/len(df)*100:.1f}%)")
        self._log(f"  验证集: {len(val)} 条 ({len(val)/len(df)*100:.1f}%)")
        self._log(f"  测试集: {len(test)} 条 ({len(test)/len(df)*100:.1f}%)")
        
        # 打印各集标签分布
        if 'label' in df.columns:
            for name, data in [('训练集', train), ('验证集', val), ('测试集', test)]:
                counts = data['label'].value_counts().sort_index()
                self._log(f"\n{name}标签分布:")
                for label, count in counts.items():
                    label_name = {0: "持有", 1: "买入", 2: "卖出"}.get(label, "未知")
                    self._log(f"  {label_name}: {count} ({count/len(data)*100:.1f}%)")
        
        return train, val, test
    
    def save_datasets(self, train, val, test, full):
        """保存数据集"""
        self._log("\n" + "=" * 80)
        self._log("保存数据集...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        files = {}
        for name, data in [('train', train), ('val', val), ('test', test), ('full', full)]:
            filepath = os.path.join(self.output_dir, f'{name}_{timestamp}.csv')
            data.to_csv(filepath, index=True)
            files[name] = filepath
            self._log(f"  ✅ {name}: {filepath}")
        
        # 保存数据集信息
        info_file = os.path.join(self.output_dir, f'dataset_info_{timestamp}.txt')
        with open(info_file, 'w', encoding='utf-8') as f:
            f.write(f"数据集信息\n")
            f.write(f"生成时间: {datetime.now()}\n\n")
            f.write(f"总记录数: {len(full)}\n")
            f.write(f"训练集: {len(train)} 条\n")
            f.write(f"验证集: {len(val)} 条\n")
            f.write(f"测试集: {len(test)} 条\n\n")
            f.write(f"特征数量: {len(full.columns)}\n")
            f.write(f"特征列表:\n")
            for col in full.columns:
                f.write(f"  - {col}\n")
        
        files['info'] = info_file
        self._log(f"  ✅ 信息文件: {info_file}")
        
        return files
    
    def run(self):
        """运行完整流程"""
        try:
            start_time = time.time()
            
            # 1. 获取K线数据
            self._log("\n" + "=" * 80)
            self._log("步骤1: 获取K线数据")
            self._log("=" * 80)
            
            df_1m = self.fetch_kline_data_with_retry('1min', DataConfig.COUNT_1MIN)
            df_5m = self.fetch_kline_data_with_retry('5min', DataConfig.COUNT_5MIN)
            
            if df_1m.empty or df_5m.empty:
                self._log("❌ K线数据获取失败")
                return None
            
            # 2. 计算特征
            self._log("\n" + "=" * 80)
            self._log("步骤2: 计算特征")
            self._log("=" * 80)
            
            df_features = self.calculate_features_optimized(df_5m, df_1m)
            
            if df_features.empty:
                self._log("❌ 特征计算失败")
                return None
            
            # 3. 生成标签
            df_labeled = self.generate_labels(df_features)
            
            # 4. 划分数据集
            train, val, test = self.split_dataset(df_labeled)
            
            # 5. 保存数据集
            files = self.save_datasets(train, val, test, df_labeled)
            
            # 完成
            elapsed = time.time() - start_time
            self._log("\n" + "=" * 80)
            self._log("✅ 数据采集完成！")
            self._log("=" * 80)
            self._log(f"总耗时: {elapsed:.2f}秒")
            self._log(f"日志文件: {self.log_file}")
            
            return files
            
        except Exception as e:
            self._log(f"\n❌ 采集过程出错: {e}")
            import traceback
            self._log(traceback.format_exc())
            return None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='大规模数据采集')
    parser.add_argument('--real-api', action='store_true', help='使用真实API')
    parser.add_argument('--days', type=int, default=30, help='获取天数')
    parser.add_argument('--max-records', type=int, default=100000, help='最大记录数')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("📥 大规模数据采集工具")
    print("=" * 80)
    print(f"使用真实API: {args.real_api}")
    print(f"目标天数: {args.days}")
    print(f"最大记录数: {args.max_records}")
    print("=" * 80)
    
    # 创建采集器并运行
    collector = LargeDatasetCollector(
        use_real_api=args.real_api,
        days=args.days,
        max_records=args.max_records
    )
    
    files = collector.run()
    
    if files:
        print("\n" + "=" * 80)
        print("✅ 数据已准备就绪！")
        print("=" * 80)
        print(f"\n生成的文件:")
        for key, path in files.items():
            print(f"  - {key}: {path}")


if __name__ == "__main__":
    main()
