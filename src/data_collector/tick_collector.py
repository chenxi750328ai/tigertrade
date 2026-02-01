"""
Tick数据采集器
从已有的tick_data_collector.py整合
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from tigeropen.tiger_open_config import TigerOpenClientConfig
from tigeropen.quote.quote_client import QuoteClient


class TickDataCollector:
    """
    Tick数据采集器
    
    功能：
    - 获取实时Tick数据
    - 持续采集和保存
    - 数据去重和管理
    """
    
    def __init__(self, symbol='SIL2603', save_dir='/home/cx/trading_data/ticks'):
        self.symbol = symbol
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化客户端
        client_config = TigerOpenClientConfig(props_path='./openapicfg_dem')
        self.quote_client = QuoteClient(client_config)
    
    def get_latest_ticks(self, count=1000):
        """获取最新Tick数据"""
        try:
            ticks = self.quote_client.get_future_trade_ticks(
                identifier=self.symbol,
                begin_index=0,
                end_index=count - 1,
                limit=count
            )
            
            if ticks is not None and not ticks.empty:
                if 'time' in ticks.columns:
                    ticks['datetime'] = pd.to_datetime(ticks['time'], unit='ms')
                return ticks
            
        except Exception as e:
            print(f"❌ 获取Tick失败: {e}")
        
        return None
    
    def save_ticks(self, ticks):
        """保存Tick数据"""
        if ticks is None or ticks.empty:
            return
        
        date_str = datetime.now().strftime('%Y%m%d')
        filename = self.save_dir / f'{self.symbol}_ticks_{date_str}.csv'
        
        # 追加模式
        if filename.exists():
            existing = pd.read_csv(filename)
            ticks = pd.concat([existing, ticks], ignore_index=True)
            ticks = ticks.drop_duplicates(subset=['datetime', 'price'])
        
        ticks.to_csv(filename, index=False)
        return filename
