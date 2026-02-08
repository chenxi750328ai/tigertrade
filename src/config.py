#!/usr/bin/env python3
"""
配置文件 - 移除所有硬编码参数
"""

import json
import os
from datetime import datetime, timedelta

def _load_trading_file():
    """从 config/trading.json 读取交易配置（若存在）。环境变量优先于文件。"""
    for base in (os.getcwd(), os.path.dirname(os.path.dirname(os.path.abspath(__file__)))):
        path = os.path.join(base, "config", "trading.json")
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
    return {}

_trading_file = _load_trading_file()

class TradingConfig:
    """交易后端与标的配置，供下单、DEMO、行情使用。优先环境变量，其次 config/trading.json，不再硬编码。"""
    BACKEND = os.getenv("TRADING_BACKEND") or _trading_file.get("trading_backend", "tiger")
    SYMBOL = os.getenv("TRADING_SYMBOL") or _trading_file.get("symbol", "SIL.COMEX.202603")
    TICK_SIZE = float(os.getenv("TICK_SIZE") or _trading_file.get("tick_size") or "0.005")

class DataConfig:
    """数据采集配置"""
    
    # API配置
    USE_REAL_API = os.getenv('USE_REAL_API', 'False').lower() == 'true'
    TIGER_CONFIG_PATH = os.getenv('TIGER_CONFIG_PATH', './openapicfg_dem')
    
    # 数据采集参数
    SYMBOL = os.getenv('SYMBOL', 'NQ')  # 期货代码
    DAYS_TO_FETCH = int(os.getenv('DAYS_TO_FETCH', '30'))  # 获取天数
    MAX_RECORDS = int(os.getenv('MAX_RECORDS', '100000'))  # 最大记录数
    
    # 时间周期
    PERIOD_1MIN = '1min'
    PERIOD_5MIN = '5min'
    
    # K线数量计算
    # 真实市场：每天约390分钟交易时间（6.5小时）
    # 1分钟: 390条/天
    # 5分钟: 78条/天
    BARS_PER_DAY_1MIN = 390
    BARS_PER_DAY_5MIN = 78
    
    # 计算需要的K线数量
    COUNT_1MIN = min(DAYS_TO_FETCH * BARS_PER_DAY_1MIN, MAX_RECORDS)
    COUNT_5MIN = min(DAYS_TO_FETCH * BARS_PER_DAY_5MIN, MAX_RECORDS // 5)
    
    # 特征计算参数
    MIN_REQUIRED_BARS = int(os.getenv('MIN_REQUIRED_BARS', '50'))
    WINDOW_SIZE = int(os.getenv('WINDOW_SIZE', '20'))
    
    # 输出目录
    OUTPUT_DIR = os.getenv('OUTPUT_DIR', '/home/cx/trading_data/large_dataset')
    
    @classmethod
    def get_time_range(cls):
        """获取时间范围"""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=cls.DAYS_TO_FETCH)
        return start_time, end_time
    
    @classmethod
    def print_config(cls):
        """打印配置信息"""
        print("=" * 80)
        print("📋 数据采集配置")
        print("=" * 80)
        print(f"使用真实API: {cls.USE_REAL_API}")
        print(f"期货代码: {cls.SYMBOL}")
        print(f"获取天数: {cls.DAYS_TO_FETCH}")
        print(f"最大记录数: {cls.MAX_RECORDS}")
        print(f"1分钟K线数量: {cls.COUNT_1MIN}")
        print(f"5分钟K线数量: {cls.COUNT_5MIN}")
        print(f"输出目录: {cls.OUTPUT_DIR}")
        print("=" * 80)


class LabelConfig:
    """标注配置"""
    
    # 标注策略选择
    STRATEGY = os.getenv('LABEL_STRATEGY', 'percentile')  # percentile, std, hybrid
    
    # 向前看周期
    LOOK_AHEAD = int(os.getenv('LOOK_AHEAD', '5'))
    
    # 固定阈值策略参数
    FIXED_BUY_THRESHOLD = float(os.getenv('FIXED_BUY_THRESHOLD', '0.5'))  # %
    FIXED_SELL_THRESHOLD = float(os.getenv('FIXED_SELL_THRESHOLD', '-0.5'))  # %
    
    # 百分位数策略参数
    PERCENTILE_BUY = int(os.getenv('PERCENTILE_BUY', '67'))
    PERCENTILE_SELL = int(os.getenv('PERCENTILE_SELL', '33'))
    
    # 标准差策略参数
    STD_MULTIPLIER = float(os.getenv('STD_MULTIPLIER', '0.25'))
    
    # 投票阈值（混合策略）
    VOTE_THRESHOLD = int(os.getenv('VOTE_THRESHOLD', '2'))
    
    @classmethod
    def print_config(cls):
        """打印配置信息"""
        print("\n" + "=" * 80)
        print("🏷️ 标注配置")
        print("=" * 80)
        print(f"标注策略: {cls.STRATEGY}")
        print(f"向前看周期: {cls.LOOK_AHEAD}")
        print(f"百分位数阈值: 买入>{cls.PERCENTILE_BUY}%, 卖出<{cls.PERCENTILE_SELL}%")
        print(f"标准差倍数: {cls.STD_MULTIPLIER}")
        print("=" * 80)


class DataSplitConfig:
    """数据划分配置"""
    
    TRAIN_RATIO = float(os.getenv('TRAIN_RATIO', '0.7'))
    VAL_RATIO = float(os.getenv('VAL_RATIO', '0.15'))
    TEST_RATIO = float(os.getenv('TEST_RATIO', '0.15'))
    
    # 是否使用随机划分（而非时间顺序）
    # ⚠️ 必须使用False（时间序列分割），Random会导致时间泄漏！
    RANDOM_SPLIT = os.getenv('RANDOM_SPLIT', 'False').lower() == 'true'
    RANDOM_SEED = int(os.getenv('RANDOM_SEED', '42'))
    
    @classmethod
    def print_config(cls):
        """打印配置信息"""
        print("\n" + "=" * 80)
        print("📊 数据划分配置")
        print("=" * 80)
        print(f"训练集比例: {cls.TRAIN_RATIO*100:.0f}%")
        print(f"验证集比例: {cls.VAL_RATIO*100:.0f}%")
        print(f"测试集比例: {cls.TEST_RATIO*100:.0f}%")
        print(f"随机划分: {cls.RANDOM_SPLIT}")
        if cls.RANDOM_SPLIT:
            print(f"随机种子: {cls.RANDOM_SEED}")
        print("=" * 80)


class TrainingConfig:
    """训练配置"""
    
    # 模型参数
    HIDDEN_DIM = int(os.getenv('HIDDEN_DIM', '128'))
    NUM_HEADS = int(os.getenv('NUM_HEADS', '4'))
    NUM_LAYERS = int(os.getenv('NUM_LAYERS', '2'))
    DROPOUT = float(os.getenv('DROPOUT', '0.1'))
    
    # 训练参数
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', '32'))
    LEARNING_RATE = float(os.getenv('LEARNING_RATE', '0.001'))
    NUM_EPOCHS = int(os.getenv('NUM_EPOCHS', '50'))
    MAX_EPOCHS = NUM_EPOCHS  # 添加MAX_EPOCHS别名以兼容旧代码
    
    # 早停参数
    EARLY_STOP_PATIENCE = int(os.getenv('EARLY_STOP_PATIENCE', '10'))
    PATIENCE = int(os.getenv('PATIENCE', '10'))  # 添加PATIENCE别名以兼容旧代码
    
    # 学习率调度
    LR_SCHEDULER = os.getenv('LR_SCHEDULER', 'ReduceLROnPlateau')  # ReduceLROnPlateau, StepLR, CosineAnnealingLR
    LR_PATIENCE = int(os.getenv('LR_PATIENCE', '5'))
    LR_FACTOR = float(os.getenv('LR_FACTOR', '0.5'))
    
    # 梯度裁剪
    GRAD_CLIP = float(os.getenv('GRAD_CLIP', '1.0'))
    
    # 日志配置
    LOG_INTERVAL = int(os.getenv('LOG_INTERVAL', '10'))  # 每N个batch打印一次
    SAVE_INTERVAL = int(os.getenv('SAVE_INTERVAL', '5'))  # 每N个epoch保存一次
    
    # 输出目录
    MODEL_DIR = os.getenv('MODEL_DIR', '/home/cx/trading_data/models')
    LOG_DIR = os.getenv('LOG_DIR', '/home/cx/trading_data/training_logs')
    
    # 设备
    DEVICE = os.getenv('DEVICE', 'cuda')  # cuda or cpu
    
    # 调试模式
    DEBUG_MODE = os.getenv('DEBUG_MODE', 'False').lower() == 'true'
    CHECK_GRADIENTS = os.getenv('CHECK_GRADIENTS', 'True').lower() == 'true'
    
    @classmethod
    def print_config(cls):
        """打印配置信息"""
        print("\n" + "=" * 80)
        print("🎓 训练配置")
        print("=" * 80)
        print(f"批次大小: {cls.BATCH_SIZE}")
        print(f"学习率: {cls.LEARNING_RATE}")
        print(f"训练轮数: {cls.NUM_EPOCHS}")
        print(f"隐藏维度: {cls.HIDDEN_DIM}")
        print(f"注意力头数: {cls.NUM_HEADS}")
        print(f"Transformer层数: {cls.NUM_LAYERS}")
        print(f"Dropout: {cls.DROPOUT}")
        print(f"梯度裁剪: {cls.GRAD_CLIP}")
        print(f"早停耐心值: {cls.EARLY_STOP_PATIENCE}")
        print(f"调试模式: {cls.DEBUG_MODE}")
        print(f"梯度检查: {cls.CHECK_GRADIENTS}")
        print("=" * 80)


class FeatureConfig:
    """特征配置"""
    
    # 基础特征
    BASIC_FEATURES = [
        'price_current',
        'atr',
        'rsi_1m',
        'rsi_5m',
    ]
    
    # 布林带特征
    BOLL_FEATURES = [
        'boll_upper',
        'boll_mid',
        'boll_lower',
        'boll_position',
    ]
    
    # 动量特征
    MOMENTUM_FEATURES = [
        'price_change_1',
        'price_change_5',
    ]
    
    # 波动率特征
    VOLATILITY_FEATURES = [
        'volatility',
    ]
    
    # 成交量特征
    VOLUME_FEATURES = [
        'volume_1m',
    ]
    
    @classmethod
    def get_all_features(cls):
        """获取所有特征"""
        return (cls.BASIC_FEATURES + 
                cls.BOLL_FEATURES + 
                cls.MOMENTUM_FEATURES + 
                cls.VOLATILITY_FEATURES + 
                cls.VOLUME_FEATURES)
    
    @classmethod
    def get_selected_features(cls):
        """获取选中的特征"""
        selected = os.getenv('FEATURES', 'all')
        if selected == 'all':
            return cls.get_all_features()
        else:
            return [f.strip() for f in selected.split(',')]
    
    @classmethod
    def print_config(cls):
        """打印配置信息"""
        features = cls.get_selected_features()
        print("\n" + "=" * 80)
        print("🔧 特征配置")
        print("=" * 80)
        print(f"特征数量: {len(features)}")
        print("特征列表:")
        for i, feat in enumerate(features, 1):
            print(f"  {i}. {feat}")
        print("=" * 80)


# 全局配置打印函数
def print_all_configs():
    """打印所有配置"""
    DataConfig.print_config()
    LabelConfig.print_config()
    DataSplitConfig.print_config()
    FeatureConfig.print_config()
    TrainingConfig.print_config()


if __name__ == "__main__":
    print_all_configs()
