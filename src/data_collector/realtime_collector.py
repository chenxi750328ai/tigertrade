"""
实时数据采集器
从tiger1.py的DataCollector类提取
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tigeropen.common.consts import BarPeriod
from tigeropen.quote.quote_client import QuoteClient
from tigeropen.tiger_open_config import TigerOpenClientConfig


class RealTimeDataCollector:
    """
    实时数据采集器
    
    功能：
    - 获取最新K线数据（1分钟、5分钟、1小时）
    - 数据缓存和更新
    - 异常处理和重试
    
    使用示例：
        collector = RealTimeDataCollector(symbol='SIL2603')
        data = collector.get_latest_klines(period='1m', count=100)
    """
    
    def __init__(self, symbol='SIL2603', config_path='./openapicfg_dem'):
        """
        初始化
        
        Args:
            symbol: 合约代码
            config_path: Tiger API配置路径
        """
        self.symbol = symbol
        self.config_path = config_path
        
        # 初始化Tiger API客户端
        self._init_client()
        
        # 数据缓存
        self.cache = {
            '1m': None,
            '5m': None,
            '1h': None
        }
        
        # 最后更新时间
        self.last_update = {
            '1m': None,
            '5m': None,
            '1h': None
        }
    
    def _init_client(self):
        """初始化Tiger API客户端"""
        try:
            client_config = TigerOpenClientConfig(props_path=self.config_path)
            self.quote_client = QuoteClient(client_config)
            print(f"✅ Tiger API客户端初始化成功")
        except Exception as e:
            print(f"❌ Tiger API客户端初始化失败: {e}")
            self.quote_client = None
    
    def get_latest_klines(self, period='1m', count=100, force_refresh=False):
        """
        获取最新K线数据
        
        Args:
            period: 周期 ('1m', '5m', '1h', 'day')
            count: 数量
            force_refresh: 是否强制刷新（忽略缓存）
        
        Returns:
            pd.DataFrame: K线数据
        """
        # 检查缓存
        if not force_refresh and self._cache_valid(period):
            return self.cache[period]
        
        # 从API获取
        try:
            df = self._fetch_from_api(period, count)
            
            # 更新缓存
            self.cache[period] = df
            self.last_update[period] = datetime.now()
            
            return df
            
        except Exception as e:
            print(f"❌ 获取K线数据失败 ({period}): {e}")
            
            # 返回缓存（如果有）
            if self.cache[period] is not None:
                print(f"⚠️ 返回缓存数据")
                return self.cache[period]
            
            return None
    
    def _cache_valid(self, period):
        """检查缓存是否有效"""
        if self.cache[period] is None:
            return False
        
        if self.last_update[period] is None:
            return False
        
        # 缓存有效期
        validity = {
            '1m': timedelta(minutes=1),
            '5m': timedelta(minutes=5),
            '1h': timedelta(hours=1),
            'day': timedelta(hours=24)
        }
        
        elapsed = datetime.now() - self.last_update[period]
        return elapsed < validity.get(period, timedelta(minutes=1))
    
    def _fetch_from_api(self, period, count):
        """从API获取数据"""
        if self.quote_client is None:
            raise Exception("Tiger API客户端未初始化")
        
        # 周期映射
        period_map = {
            '1m': BarPeriod.ONE_MINUTE,
            '5m': BarPeriod.FIVE_MINUTES,
            '1h': BarPeriod.ONE_HOUR,
            'day': BarPeriod.DAY
        }
        
        bar_period = period_map.get(period, BarPeriod.ONE_MINUTE)
        
        # 获取数据
        df = self.quote_client.get_future_bars(
            identifiers=self.symbol,
            period=bar_period,
            limit=count
        )
        
        # 数据处理
        if df is not None and not df.empty:
            if 'time' in df.columns:
                df['datetime'] = pd.to_datetime(df['time'], unit='ms')
            
            # 标准化列名
            df = df.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            })
            
            # 排序
            df = df.sort_values('datetime').reset_index(drop=True)
        
        return df
    
    def get_multi_period_data(self, periods=['1m', '5m', '1h'], counts=None):
        """
        获取多周期数据
        
        Args:
            periods: 周期列表
            counts: 每个周期的数量（None则使用默认值）
        
        Returns:
            dict: {period: DataFrame}
        """
        if counts is None:
            counts = {
                '1m': 100,
                '5m': 100,
                '1h': 100,
                'day': 60
            }
        
        result = {}
        for period in periods:
            count = counts.get(period, 100)
            df = self.get_latest_klines(period, count)
            result[period] = df
        
        return result
    
    def get_latest_price(self):
        """获取最新价格"""
        df = self.get_latest_klines('1m', count=1)
        if df is not None and not df.empty:
            return df['close'].iloc[-1]
        return None


# 向后兼容：保持与原tiger1.py相同的接口
class DataCollector(RealTimeDataCollector):
    """向后兼容的别名"""
    pass
