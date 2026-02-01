"""
K线数据获取器
历史K线数据获取
"""

import pandas as pd
from datetime import datetime, timedelta
from tigeropen.common.consts import BarPeriod
from tigeropen.quote.quote_client import QuoteClient
from tigeropen.tiger_open_config import TigerOpenClientConfig


class KLineFetcher:
    """
    K线数据获取器
    
    功能：
    - 获取历史K线数据
    - 支持日期范围查询
    - 数据保存
    """
    
    def __init__(self, config_path='./openapicfg_dem'):
        client_config = TigerOpenClientConfig(props_path=config_path)
        self.quote_client = QuoteClient(client_config)
    
    def fetch_historical_klines(self, symbol, period='day', days=90):
        """
        获取历史K线
        
        Args:
            symbol: 合约代码
            period: 周期
            days: 天数
        
        Returns:
            pd.DataFrame
        """
        period_map = {
            '1m': BarPeriod.ONE_MINUTE,
            '5m': BarPeriod.FIVE_MINUTES,
            '1h': BarPeriod.ONE_HOUR,
            'day': BarPeriod.DAY
        }
        
        bar_period = period_map.get(period, BarPeriod.DAY)
        
        try:
            df = self.quote_client.get_future_bars(
                identifiers=symbol,
                period=bar_period,
                limit=2000  # 最大值
            )
            
            if df is not None and not df.empty:
                if 'time' in df.columns:
                    df['datetime'] = pd.to_datetime(df['time'], unit='ms')
                df = df.sort_values('datetime').reset_index(drop=True)
                
                # 只返回最近N天
                if 'datetime' in df.columns:
                    cutoff = datetime.now() - timedelta(days=days)
                    df = df[df['datetime'] >= cutoff]
                
                return df
        
        except Exception as e:
            print(f"❌ 获取历史K线失败: {e}")
        
        return None
