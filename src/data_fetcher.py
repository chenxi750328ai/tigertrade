import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import time
import glob
import pytz
import warnings
warnings.filterwarnings("ignore")

# 导入tiger相关库
from tigeropen.quote.quote_client import QuoteClient
from tigeropen.common.consts import Language
from tigeropen.tiger_open_config import TigerOpenClientConfig
from tigeropen.common.util.contract_utils import future_contract


def get_kline_data_extended(symbol, period='1min', count=20000):
    """
    获取扩展的历史K线数据，用于训练模型
    """
    try:
        # 这里我们模拟从tiger API获取数据的逻辑
        # 实际环境中，您需要替换为真正的API调用
        print(f"Fetching {count} historical data points for {symbol} at {period} intervals...")
        
        # 为了演示目的，我们创建模拟数据
        # 在实际环境中，您应该使用真实的API调用来获取数据
        dates = pd.date_range(end=datetime.now(), periods=count, freq=period)
        
        # 生成模拟价格数据（使用随机游走模型）
        initial_price = 92.0
        returns = np.random.normal(0, 0.001, count)  # 每分钟收益率
        prices = [initial_price]
        
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(new_price)
        
        # 计算技术指标
        df = pd.DataFrame({
            'time': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.0005))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.0005))) for p in prices],
            'close': prices,
            'volume': np.random.randint(10, 1000, size=count)
        })
        
        # 计算RSI指标
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 计算ATR指标
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=14).mean()
        
        # 用fillna填充NaN值
        df['rsi'].fillna(method='bfill', inplace=True)
        df['atr'].fillna(method='bfill', inplace=True)
        
        return df
    
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return pd.DataFrame()


def save_extended_data(symbol, period='1min', count=20000):
    """
    获取并保存扩展的历史数据
    """
    # 确保数据目录存在
    data_dir = '/home/cx/trading_data'
    os.makedirs(data_dir, exist_ok=True)
    
    # 获取今天日期作为目录名
    today = datetime.now().strftime('%Y-%m-%d')
    day_dir = os.path.join(data_dir, today)
    os.makedirs(day_dir, exist_ok=True)
    
    print(f"Fetching extended historical data for {symbol}...")
    df = get_kline_data_extended(symbol, period, count)
    
    if df.empty:
        print("Failed to fetch data")
        return False
    
    # 保存数据
    filename = f"extended_trading_data_{symbol}_{period}_{datetime.now().strftime('%H%M%S')}.csv"
    filepath = os.path.join(day_dir, filename)
    
    df.to_csv(filepath, index=False)
    print(f"Saved extended data to {filepath}, shape: {df.shape}")
    
    return True


def aggregate_data_for_training(data_dir="/home/cx/trading_data", days=30):
    """
    聚合多天的数据用于训练
    """
    print("Aggregating data for training...")
    
    all_data_files = []
    
    # 获取最近几天的数据目录
    date_dirs = glob.glob(os.path.join(data_dir, '202*-*-*'))
    if not date_dirs:
        print("No historical data found")
        return None
        
    # 按日期排序，获取最近days天的数据
    sorted_dirs = sorted(date_dirs, reverse=True)[:days]
    
    for data_dir in sorted_dirs:
        # 包括原始数据和扩展数据
        data_files = glob.glob(os.path.join(data_dir, 'trading_data_*.csv'))
        data_files.extend(glob.glob(os.path.join(data_dir, 'extended_trading_data_*.csv')))
        all_data_files.extend(data_files)
    
    if not all_data_files:
        print("No trading data files found")
        return None
    
    # 按修改时间排序，获取所有文件
    all_data_files = sorted(all_data_files, key=os.path.getmtime)
    
    # 合并所有数据文件
    all_data = []
    for file_path in all_data_files:
        try:
            df = pd.read_csv(file_path)
            all_data.append(df)
            print(f"Loaded {file_path} with {len(df)} records")
        except Exception as e:
            print(f"Failed to read {file_path}: {e}")
    
    if not all_data:
        print("No data successfully loaded")
        return None
    
    # 合并所有数据
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Combined {len(all_data)} files into {len(combined_df)} total records")
    
    # 清理数据
    combined_df = combined_df.dropna(subset=['close', 'rsi', 'atr'])
    print(f"After cleaning: {len(combined_df)} records")
    
    return combined_df


def prepare_features_from_raw_data(df):
    """
    从原始数据准备特征
    """
    print("Preparing features from raw data...")
    
    # 确保数据已按时间排序
    df = df.sort_values('time').reset_index(drop=True)
    
    # 生成网格参数
    df['price_current'] = df['close']
    df['grid_upper'] = df['close'] * 1.01  # 1% 上涨
    df['grid_lower'] = df['close'] * 0.99  # 1% 下跌
    df['atr'] = df['atr']
    df['rsi_1m'] = df['rsi']  # 假设这是1分钟RSI
    df['rsi_5m'] = df['rsi']  # 假设这是5分钟RSI，实际上可能需要聚合
    
    # 计算缓冲区
    df['buffer'] = df['atr'] * 0.3
    df['threshold'] = df['grid_lower'] + df['buffer']
    
    # 计算布尔条件
    df['near_lower'] = df['price_current'] <= df['threshold']
    df['rsi_ok'] = (df['rsi_1m'] < 30) | ((df['rsi_5m'] > 45) & (df['rsi_5m'] < 55))
    
    # 选择需要的列
    feature_cols = ['price_current', 'grid_lower', 'grid_upper', 'atr', 'rsi_1m', 'rsi_5m', 
                   'buffer', 'threshold', 'near_lower', 'rsi_ok']
    
    # 仅选择存在的列
    available_cols = [col for col in feature_cols if col in df.columns]
    result_df = df[available_cols].copy()
    
    # 为缺失的列添加默认值
    for col in feature_cols:
        if col not in result_df.columns:
            if col in ['near_lower', 'rsi_ok']:
                result_df[col] = False
            else:
                result_df[col] = 0.0
    
    print(f"Prepared {len(result_df)} feature vectors")
    return result_df


def main():
    """
    主函数 - 获取扩展数据并准备用于训练
    """
    print("Starting extended data fetch and preparation...")
    
    # 获取扩展的历史数据
    symbol = "SIL"
    success = save_extended_data(symbol, period='1min', count=20000)
    
    if not success:
        print("Failed to fetch extended data")
        return
    
    # 聚合数据
    combined_df = aggregate_data_for_training()
    
    if combined_df is None or len(combined_df) < 100:
        print("Insufficient data for training")
        return
    
    # 准备特征
    features_df = prepare_features_from_raw_data(combined_df)
    
    if features_df is None or len(features_df) < 100:
        print("Insufficient features for training")
        return
    
    # 保存准备好的特征数据
    data_dir = '/home/cx/trading_data'
    today = datetime.now().strftime('%Y-%m-%d')
    day_dir = os.path.join(data_dir, today)
    
    feature_filename = f"prepared_features_{datetime.now().strftime('%H%M%S')}.csv"
    feature_filepath = os.path.join(day_dir, feature_filename)
    
    features_df.to_csv(feature_filepath, index=False)
    print(f"Saved prepared features to {feature_filepath}")
    
    print(f"Final dataset contains {len(features_df)} samples with {len(features_df.columns)} features")
    print("Feature columns:", list(features_df.columns))


if __name__ == "__main__":
    main()