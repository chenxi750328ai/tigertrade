#!/usr/bin/env python3
"""
TigerTrade数据预处理和特征工程
目标：为模型训练准备高质量数据
"""

import pandas as pd
import numpy as np
import talib
from pathlib import Path
import json
from datetime import datetime

class TigerTradeDataProcessor:
    """数据预处理器"""
    
    def __init__(self, input_file, output_dir):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*70)
        print("🔧 TigerTrade数据预处理器")
        print("="*70)
        print(f"输入文件: {self.input_file}")
        print(f"输出目录: {self.output_dir}")
        print("="*70)
    
    def load_data(self):
        """加载原始数据"""
        print("\n📂 加载数据...")
        self.df = pd.read_csv(self.input_file)
        print(f"✅ 加载完成：{len(self.df)} 条记录")
        print(f"列：{list(self.df.columns)}")
        print(f"\n数据预览：")
        print(self.df.head())
        return self
    
    def clean_data(self):
        """数据清洗"""
        print("\n🧹 数据清洗...")
        
        # 检查缺失值
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            print(f"⚠️  发现缺失值：")
            print(missing[missing > 0])
            
            # 填充缺失值（前向填充）
            self.df = self.df.ffill()
            print(f"✅ 缺失值已填充")
        else:
            print(f"✅ 无缺失值")
        
        # 检查异常值（价格/成交量）
        if 'close' in self.df.columns:
            price_std = self.df['close'].std()
            price_mean = self.df['close'].mean()
            outliers = ((self.df['close'] - price_mean).abs() > 3 * price_std).sum()
            print(f"📊 价格异常值（3σ）：{outliers} 条")
        
        # 确保时间列
        if 'time' in self.df.columns:
            self.df['time'] = pd.to_datetime(self.df['time'])
            self.df.sort_values('time', inplace=True)
            self.df.reset_index(drop=True, inplace=True)
            print(f"✅ 时间序列已排序")
        
        return self
    
    def add_technical_indicators(self):
        """添加技术指标"""
        print("\n📈 计算技术指标...")
        
        # 处理列名（使用price_current作为close）
        if 'close' not in self.df.columns and 'price_current' in self.df.columns:
            self.df['close'] = self.df['price_current']
            print("✅ 使用price_current作为close价格")
        
        if 'close' not in self.df.columns:
            print("❌ 缺少close列，跳过技术指标")
            return self
        
        close = self.df['close'].values
        high = self.df.get('high', self.df['close']).values
        low = self.df.get('low', self.df['close']).values
        
        # 使用volume_1m作为volume（如果存在）
        if 'volume' not in self.df.columns and 'volume_1m' in self.df.columns:
            self.df['volume'] = self.df['volume_1m']
        
        volume = self.df.get('volume', pd.Series([1]*len(self.df))).values
        
        # RSI (相对强弱指标)
        print("  计算RSI...")
        self.df['rsi_14'] = talib.RSI(close, timeperiod=14)
        self.df['rsi_28'] = talib.RSI(close, timeperiod=28)
        
        # MACD (指数平滑异同移动平均线)
        print("  计算MACD...")
        macd, signal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        self.df['macd'] = macd
        self.df['macd_signal'] = signal
        self.df['macd_hist'] = hist
        
        # Bollinger Bands (布林带)
        print("  计算Bollinger Bands...")
        upper, middle, lower = talib.BBANDS(close, timeperiod=20)
        self.df['bb_upper'] = upper
        self.df['bb_middle'] = middle
        self.df['bb_lower'] = lower
        self.df['bb_width'] = (upper - lower) / middle
        
        # 移动平均线
        print("  计算移动平均线...")
        self.df['sma_5'] = talib.SMA(close, timeperiod=5)
        self.df['sma_10'] = talib.SMA(close, timeperiod=10)
        self.df['sma_20'] = talib.SMA(close, timeperiod=20)
        self.df['ema_5'] = talib.EMA(close, timeperiod=5)
        self.df['ema_10'] = talib.EMA(close, timeperiod=10)
        
        # ATR (真实波动幅度)
        print("  计算ATR...")
        self.df['atr_14'] = talib.ATR(high, low, close, timeperiod=14)
        
        # ROC (变动率指标)
        print("  计算ROC...")
        self.df['roc_10'] = talib.ROC(close, timeperiod=10)
        
        # 成交量指标
        if 'volume' in self.df.columns:
            print("  计算成交量指标...")
            self.df['volume_sma_5'] = talib.SMA(volume, timeperiod=5)
            self.df['volume_ratio'] = self.df['volume'] / self.df['volume_sma_5']
        
        print(f"✅ 技术指标计算完成，新增 {len([c for c in self.df.columns if c.startswith(('rsi_', 'macd', 'bb_', 'sma_', 'ema_', 'atr_', 'roc_', 'volume_'))])} 个特征")
        
        return self
    
    def add_custom_features(self):
        """添加自定义特征"""
        print("\n💡 添加自定义特征...")
        
        if 'close' not in self.df.columns:
            return self
        
        # 价格变化率（多时间窗口）
        for window in [1, 5, 10, 30, 60]:
            self.df[f'price_change_{window}'] = self.df['close'].pct_change(window)
        
        # 价格动量
        for window in [5, 10, 20]:
            self.df[f'momentum_{window}'] = self.df['close'] - self.df['close'].shift(window)
        
        # 波动率（多时间窗口）
        for window in [5, 10, 20, 60]:
            self.df[f'volatility_{window}'] = self.df['close'].rolling(window).std()
        
        # 价格位置（在最近N周期的高低范围内的位置）
        for window in [10, 20, 60]:
            rolling_max = self.df['close'].rolling(window).max()
            rolling_min = self.df['close'].rolling(window).min()
            self.df[f'price_position_{window}'] = (self.df['close'] - rolling_min) / (rolling_max - rolling_min + 1e-8)
        
        # 如果有成交量，添加成交量特征
        if 'volume' in self.df.columns:
            for window in [1, 5, 10]:
                self.df[f'volume_change_{window}'] = self.df['volume'].pct_change(window)
        
        custom_count = len([c for c in self.df.columns if any(c.startswith(p) for p in ['price_change_', 'momentum_', 'volatility_', 'price_position_', 'volume_change_'])])
        print(f"✅ 自定义特征添加完成，新增 {custom_count} 个特征")
        
        return self
    
    def create_target(self):
        """创建目标变量（未来收益）"""
        print("\n🎯 创建目标变量...")
        
        if 'close' not in self.df.columns:
            return self
        
        # 未来1/5/10期的收益率
        for horizon in [1, 5, 10]:
            self.df[f'target_return_{horizon}'] = self.df['close'].pct_change(horizon).shift(-horizon)
        
        # 未来趋势（上涨/下跌）
        self.df['target_direction_1'] = (self.df['target_return_1'] > 0).astype(int)
        self.df['target_direction_5'] = (self.df['target_return_5'] > 0).astype(int)
        
        print(f"✅ 目标变量创建完成")
        
        return self
    
    def split_data(self, train_ratio=0.7, val_ratio=0.15):
        """时间序列分割"""
        print("\n✂️  数据分割（时间序列）...")
        
        # 删除包含NaN的行
        self.df.dropna(inplace=True)
        print(f"清洗后数据：{len(self.df)} 条")
        
        n = len(self.df)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)
        
        self.train_df = self.df[:train_size].copy()
        self.val_df = self.df[train_size:train_size+val_size].copy()
        self.test_df = self.df[train_size+val_size:].copy()
        
        print(f"✅ 数据分割完成：")
        print(f"   训练集: {len(self.train_df)} 条 ({len(self.train_df)/n*100:.1f}%)")
        print(f"   验证集: {len(self.val_df)} 条 ({len(self.val_df)/n*100:.1f}%)")
        print(f"   测试集: {len(self.test_df)} 条 ({len(self.test_df)/n*100:.1f}%)")
        
        return self
    
    def save_data(self):
        """保存处理后的数据"""
        print("\n💾 保存数据...")
        
        # 保存完整数据
        full_output = self.output_dir / "processed_data.csv"
        self.df.to_csv(full_output, index=False)
        print(f"✅ 完整数据: {full_output}")
        
        # 保存分割数据
        train_output = self.output_dir / "train.csv"
        val_output = self.output_dir / "val.csv"
        test_output = self.output_dir / "test.csv"
        
        self.train_df.to_csv(train_output, index=False)
        self.val_df.to_csv(val_output, index=False)
        self.test_df.to_csv(test_output, index=False)
        
        print(f"✅ 训练集: {train_output}")
        print(f"✅ 验证集: {val_output}")
        print(f"✅ 测试集: {test_output}")
        
        # 保存特征列表
        feature_cols = [c for c in self.df.columns if c not in ['time', 'open', 'high', 'low', 'close', 'volume'] 
                       and not c.startswith('target_')]
        
        metadata = {
            "total_samples": len(self.df),
            "train_samples": len(self.train_df),
            "val_samples": len(self.val_df),
            "test_samples": len(self.test_df),
            "num_features": len(feature_cols),
            "feature_columns": feature_cols,
            "target_columns": [c for c in self.df.columns if c.startswith('target_')],
            "processed_at": datetime.now().isoformat()
        }
        
        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ 元数据: {metadata_file}")
        print(f"\n📊 特征总数: {len(feature_cols)}")
        
        return self
    
    def generate_report(self):
        """生成数据报告"""
        print("\n📄 生成数据报告...")
        
        report = []
        report.append("="*70)
        report.append("TigerTrade数据预处理报告")
        report.append("="*70)
        report.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\n输入文件: {self.input_file}")
        report.append(f"输出目录: {self.output_dir}")
        
        report.append(f"\n{'='*70}")
        report.append("数据统计")
        report.append(f"{'='*70}")
        report.append(f"原始数据: {len(pd.read_csv(self.input_file))} 条")
        report.append(f"清洗后: {len(self.df)} 条")
        report.append(f"训练集: {len(self.train_df)} 条 ({len(self.train_df)/len(self.df)*100:.1f}%)")
        report.append(f"验证集: {len(self.val_df)} 条 ({len(self.val_df)/len(self.df)*100:.1f}%)")
        report.append(f"测试集: {len(self.test_df)} 条 ({len(self.test_df)/len(self.df)*100:.1f}%)")
        
        feature_cols = [c for c in self.df.columns if c not in ['time', 'open', 'high', 'low', 'close', 'volume'] 
                       and not c.startswith('target_')]
        
        report.append(f"\n{'='*70}")
        report.append("特征工程")
        report.append(f"{'='*70}")
        report.append(f"特征总数: {len(feature_cols)}")
        report.append(f"\n特征类别:")
        
        categories = {
            "技术指标": ['rsi_', 'macd', 'bb_', 'sma_', 'ema_', 'atr_', 'roc_'],
            "价格特征": ['price_change_', 'momentum_', 'price_position_'],
            "波动率特征": ['volatility_'],
            "成交量特征": ['volume_']
        }
        
        for cat_name, prefixes in categories.items():
            cat_features = [c for c in feature_cols if any(c.startswith(p) for p in prefixes)]
            if cat_features:
                report.append(f"  {cat_name}: {len(cat_features)} 个")
        
        report.append(f"\n目标变量: {len([c for c in self.df.columns if c.startswith('target_')])} 个")
        
        report_text = "\n".join(report)
        print(report_text)
        
        # 保存报告
        report_file = self.output_dir / "preprocessing_report.txt"
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        print(f"\n✅ 报告已保存: {report_file}")
        
        return self


def _ensure_sample_data(input_path):
    """若默认输入不存在，在项目内生成示例 CSV 供预处理使用。"""
    p = Path(input_path)
    if p.exists():
        return str(p)
    # 项目内备选路径
    base = Path(__file__).resolve().parent.parent
    alt = base / "data" / "raw" / "sample.csv"
    if alt.exists():
        return str(alt)
    # 生成最小示例：OHLCV + time，约 200 条
    alt.parent.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)
    n = 200
    t0 = pd.Timestamp("2025-01-01 09:00:00", tz="UTC")
    times = pd.date_range(t0, periods=n, freq="1min")
    close = 90.0 + np.cumsum(np.random.randn(n) * 0.1)
    high = close + np.abs(np.random.randn(n) * 0.05)
    low = close - np.abs(np.random.randn(n) * 0.05)
    open_ = np.roll(close, 1)
    open_[0] = 90.0
    df = pd.DataFrame({
        "time": times,
        "open": open_, "high": high, "low": low, "close": close,
        "volume": (100 + np.random.randint(0, 50, n)).astype(float)
    })
    df.to_csv(alt, index=False)
    print(f"📂 已生成示例数据: {alt}")
    return str(alt)


def main():
    """主函数"""
    # 配置：优先使用原始数据，否则用项目内 data/raw 或自动生成示例
    default_input = "/home/cx/trading_data/large_dataset/full_20260121_100827.csv"
    input_file = _ensure_sample_data(default_input)
    output_dir = "/home/cx/tigertrade/data/processed"
    
    # 执行预处理
    processor = TigerTradeDataProcessor(input_file, output_dir)
    
    (processor
     .load_data()
     .clean_data()
     .add_technical_indicators()
     .add_custom_features()
     .create_target()
     .split_data(train_ratio=0.7, val_ratio=0.15)
     .save_data()
     .generate_report())
    
    print("\n" + "="*70)
    print("🎉 数据预处理完成！")
    print("="*70)
    print(f"\n下一步：模型训练")
    print(f"  python train_model.py")


if __name__ == "__main__":
    main()
