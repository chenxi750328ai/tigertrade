import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, timedelta
import torch
import torch.nn as nn
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings("ignore")

class DataDrivenOptimizer:
    """基于数据分析的模型参数优化器"""
    
    def __init__(self, data_dir='/home/cx/trading_data'):
        self.data_dir = data_dir
        self.feature_importance = {}
        
    def load_recent_data(self, days=7):
        """加载最近几天的数据"""
        all_data_files = []
        
        # 获取所有数据目录
        all_data_dirs = glob.glob(os.path.join(self.data_dir, '202*-*-*'))
        if all_data_dirs:
            # 按日期排序，获取最近几天的数据
            sorted_dirs = sorted(all_data_dirs, reverse=True)
            for data_dir in sorted_dirs[:days]:  # 使用最近7天的数据
                data_files = glob.glob(os.path.join(data_dir, 'trading_data_*.csv'))
                all_data_files.extend(data_files)
        
        if not all_data_files:
            return None
        
        # 按修改时间排序，获取所有相关文件
        sorted_files = sorted(all_data_files, key=os.path.getmtime, reverse=True)
        
        # 合并所有数据
        dfs = []
        for file_path in sorted_files:
            try:
                df = pd.read_csv(file_path)
                dfs.append(df)
            except Exception as e:
                print(f"加载文件失败 {file_path}: {e}")
        
        if not dfs:
            return None
        
        combined_df = pd.concat(dfs, ignore_index=True)
        return combined_df.dropna()
    
    def analyze_market_regimes(self, df):
        """分析市场状态（趋势、震荡等）"""
        if df is None or len(df) < 100:
            return {'trend_strength': 0.5, 'volatility': 0.02, 'mean_reversion': 0.5}
        
        # 计算价格变化率
        df['price_change_pct'] = df['price_current'].pct_change()
        
        # 计算趋势强度 (通过价格与其移动平均线的偏离度)
        df['ma_20'] = df['price_current'].rolling(window=20).mean()
        df['trend_strength'] = abs(df['price_current'] - df['ma_20']) / df['ma_20']
        
        # 计算波动率
        df['volatility'] = df['price_change_pct'].rolling(window=20).std()
        
        # 计算均值回归强度 (RSI在30以下或70以上后的回归倾向)
        df['mean_reversion'] = (
            ((df['rsi_1m'] < 30) | (df['rsi_1m'] > 70)) & 
            (df['price_current'] > df['ma_20']) & 
            (df['price_change_pct'].shift(-1) < 0)
        ).astype(int).rolling(window=10).sum() / 10
        
        # 返回统计值
        trend_strength = df['trend_strength'].dropna().mean() or 0.5
        volatility = df['volatility'].dropna().mean() or 0.02
        mean_reversion = df['mean_reversion'].dropna().mean() or 0.5
        
        return {
            'trend_strength': trend_strength,
            'volatility': volatility,
            'mean_reversion': mean_reversion
        }
    
    def optimize_model_params(self, market_regime):
        """根据市场状态优化模型参数"""
        # 根据市场状态调整模型参数
        params = {
            'lstm_hidden_size': 64,
            'lstm_num_layers': 2,
            'transformer_d_model': 256,
            'transformer_nhead': 8,
            'transformer_num_layers': 4,
            'learning_rate': 0.001,
            'dropout_rate': 0.2
        }
        
        # 高波动市场：增加模型复杂度以捕捉复杂模式
        if market_regime['volatility'] > 0.03:
            params['lstm_hidden_size'] = 128
            params['transformer_d_model'] = 512
            params['transformer_num_layers'] = 6
            params['learning_rate'] = 0.0005  # 降低学习率以稳定训练
        
        # 趋势市场：减少正则化，让模型更容易跟随趋势
        if market_regime['trend_strength'] > 0.05:
            params['dropout_rate'] = 0.1
            params['learning_rate'] = 0.0015  # 增加学习率以更快适应趋势
        
        # 均值回归市场：调整奖励函数偏向反转策略
        if market_regime['mean_reversion'] > 0.5:
            params['dropout_rate'] = 0.3  # 增加正则化避免过拟合假突破
        
        return params
    
    def suggest_action_thresholds(self, market_regime):
        """根据市场状态建议操作阈值"""
        # 根据市场状态调整操作阈值
        thresholds = {
            'min_confidence': 0.6,  # 最小置信度
            'min_price_change': 0.005,  # 最小价格变化百分比
            'max_risk_ratio': 0.02,  # 最大风险比率
        }
        
        # 高波动市场：提高阈值以减少错误信号
        if market_regime['volatility'] > 0.03:
            thresholds['min_confidence'] = 0.7
            thresholds['min_price_change'] = 0.008
        
        # 趋势市场：降低阈值以抓住趋势机会
        if market_regime['trend_strength'] > 0.05:
            thresholds['min_confidence'] = 0.55
            thresholds['max_risk_ratio'] = 0.03
        
        # 均值回归市场：提高阈值避免假突破
        if market_regime['mean_reversion'] > 0.5:
            thresholds['min_confidence'] = 0.65
            thresholds['min_price_change'] = 0.007
        
        return thresholds
    
    def get_feature_importance(self, df):
        """分析特征重要性"""
        if df is None or len(df) < 100:
            return {}
        
        # 计算各特征与价格变动的相关性
        features = ['atr', 'rsi_1m', 'rsi_5m', 'grid_lower', 'grid_upper', 'buffer', 'threshold']
        correlations = {}
        
        for feat in features:
            if feat in df.columns and 'price_current' in df.columns:
                corr = df[[feat, 'price_current']].corr().iloc[0, 1]
                correlations[feat] = abs(corr)  # 使用绝对值作为重要性指标
        
        return correlations
    
    def run_analysis_and_optimization(self):
        """运行完整的分析和优化流程"""
        print("🔍 开始数据分析和模型优化...")
        
        # 加载数据
        df = self.load_recent_data(days=7)
        if df is None:
            print("⚠️ 未能加载数据，使用默认参数")
            return {}, {}
        
        print(f"📊 已加载 {len(df)} 条数据记录")
        
        # 分析市场状态
        market_regime = self.analyze_market_regimes(df)
        print(f"📈 市场分析结果:")
        print(f"   趋势强度: {market_regime['trend_strength']:.3f}")
        print(f"   波动率: {market_regime['volatility']:.3f}")
        print(f"   均值回归强度: {market_regime['mean_reversion']:.3f}")
        
        # 分析特征重要性
        feature_importance = self.get_feature_importance(df)
        print(f"💡 特征重要性 (前5个):")
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        for feat, imp in sorted_features:
            print(f"   {feat}: {imp:.3f}")
        
        # 优化模型参数
        model_params = self.optimize_model_params(market_regime)
        print(f"⚙️ 优化的模型参数:")
        for param, value in model_params.items():
            print(f"   {param}: {value}")
        
        # 建议操作阈值
        thresholds = self.suggest_action_thresholds(market_regime)
        print(f"🎯 建议的操作阈值:")
        for thr, value in thresholds.items():
            print(f"   {thr}: {value}")
        
        return model_params, thresholds


def main():
    optimizer = DataDrivenOptimizer()
    model_params, thresholds = optimizer.run_analysis_and_optimization()
    
    print("\n✅ 数据分析和模型优化完成")


if __name__ == "__main__":
    main()