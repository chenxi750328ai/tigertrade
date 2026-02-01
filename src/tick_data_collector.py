#!/usr/bin/env python3
"""
Tick数据持续采集器
在交易时段持续采集并本地保存，积累海量历史数据
"""

import sys
import os
import time
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tigeropen.tiger_open_config import TigerOpenClientConfig
from tigeropen.quote.quote_client import QuoteClient


class TickDataCollector:
    """Tick数据持续采集器"""
    
    def __init__(self, symbol='SIL2603', save_dir='/home/cx/trading_data/ticks'):
        """
        初始化采集器
        
        Args:
            symbol: 期货合约代码
            save_dir: 数据保存目录
        """
        self.symbol = symbol
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建客户端
        client_config = TigerOpenClientConfig(props_path='./openapicfg_dem')
        self.quote_client = QuoteClient(client_config)
        
        # 记录最后采集的索引
        self.last_index = 0
        self.tick_buffer = []  # 缓冲区
        self.buffer_size = 1000  # 每1000条写入一次
        
        # 日志
        self.log_file = self.save_dir / 'collector.log'
        
        self._log(f"{'='*80}")
        self._log(f"✅ Tick采集器初始化完成")
        self._log(f"合约: {self.symbol}")
        self._log(f"保存目录: {self.save_dir}")
        self._log(f"{'='*80}")
    
    def _log(self, message):
        """记录日志"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_msg + '\n')
    
    def fetch_latest_ticks(self, batch_size=1000):
        """
        获取最新的Tick数据
        
        Args:
            batch_size: 每批获取数量
        
        Returns:
            DataFrame or None
        """
        try:
            ticks = self.quote_client.get_future_trade_ticks(
                identifier=self.symbol,
                begin_index=0,
                end_index=batch_size - 1,
                limit=batch_size
            )
            
            if ticks is not None and not ticks.empty:
                if 'time' in ticks.columns:
                    ticks['datetime'] = pd.to_datetime(ticks['time'], unit='ms')
                
                return ticks
            else:
                return None
                
        except Exception as e:
            self._log(f"❌ 获取Tick失败: {e}")
            return None
    
    def save_batch(self, df, batch_type='realtime'):
        """
        保存一批数据
        
        Args:
            df: DataFrame
            batch_type: 'realtime' 或 'backfill'
        """
        if df.empty:
            return
        
        # 按日期分文件保存
        if 'datetime' in df.columns:
            date_str = df['datetime'].iloc[-1].strftime('%Y%m%d')
        else:
            date_str = datetime.now().strftime('%Y%m%d')
        
        filename = self.save_dir / f'{self.symbol}_ticks_{date_str}.csv'
        
        # 追加模式保存
        if filename.exists():
            # 读取已有数据，去重后合并
            existing = pd.read_csv(filename)
            if 'datetime' in existing.columns:
                existing['datetime'] = pd.to_datetime(existing['datetime'])
            
            combined = pd.concat([existing, df], ignore_index=True)
            combined = combined.drop_duplicates(subset=['datetime', 'price'])
            combined = combined.sort_values('datetime')
            combined.to_csv(filename, index=False)
            
            new_count = len(combined) - len(existing)
            self._log(f"📝 追加 {new_count} 条新Tick到 {filename.name}")
        else:
            df.to_csv(filename, index=False)
            self._log(f"📝 创建新文件 {filename.name}，保存 {len(df)} 条Tick")
    
    def collect_historical_ticks(self, max_batches=100):
        """
        回填历史Tick数据
        
        Args:
            max_batches: 最多获取批次数
        """
        self._log(f"\n{'='*60}")
        self._log(f"📥 开始回填历史Tick数据...")
        self._log(f"{'='*60}")
        
        batch_size = 1000
        all_ticks = []
        
        for batch_num in range(max_batches):
            begin_idx = batch_num * batch_size
            end_idx = (batch_num + 1) * batch_size - 1
            
            try:
                ticks = self.quote_client.get_future_trade_ticks(
                    identifier=self.symbol,
                    begin_index=begin_idx,
                    end_index=end_idx,
                    limit=batch_size
                )
                
                if ticks is not None and not ticks.empty:
                    if 'time' in ticks.columns:
                        ticks['datetime'] = pd.to_datetime(ticks['time'], unit='ms')
                    
                    all_ticks.append(ticks)
                    
                    time_info = f"{ticks['datetime'].min().strftime('%m-%d %H:%M')} - {ticks['datetime'].max().strftime('%H:%M')}"
                    self._log(f"批次{batch_num+1:3d} ({begin_idx:6d}-{end_idx:6d}): ✅ {len(ticks):4d}条 | {time_info}")
                    
                    # 每10批保存一次
                    if (batch_num + 1) % 10 == 0:
                        df_batch = pd.concat(all_ticks[-10:], ignore_index=True)
                        self.save_batch(df_batch, 'backfill')
                    
                    # 避免频率限制
                    time.sleep(0.6)
                else:
                    self._log(f"批次{batch_num+1}: 数据为空，停止回填")
                    break
                    
            except Exception as e:
                error_msg = str(e)
                self._log(f"批次{batch_num+1}: ❌ {error_msg[:80]}")
                
                if 'rate limit' in error_msg.lower():
                    self._log(f"  → 触发频率限制，等待60秒...")
                    time.sleep(60)
                else:
                    break
        
        # 保存剩余数据
        if all_ticks:
            df_all = pd.concat(all_ticks, ignore_index=True)
            df_all = df_all.sort_values('datetime').drop_duplicates(subset=['datetime', 'price'])
            self.save_batch(df_all, 'backfill')
            
            self._log(f"\n✅ 历史回填完成: 总计 {len(df_all)} 条Tick")
            self._log(f"时间范围: {df_all['datetime'].min()} 至 {df_all['datetime'].max()}")
    
    def run_realtime_collector(self, interval_seconds=60):
        """
        实时采集模式（持续运行）
        
        Args:
            interval_seconds: 采集间隔（秒）
        """
        self._log(f"\n{'='*60}")
        self._log(f"🔄 启动实时Tick采集（间隔{interval_seconds}秒）")
        self._log(f"按Ctrl+C停止")
        self._log(f"{'='*60}")
        
        try:
            while True:
                # 获取最新Tick
                ticks = self.fetch_latest_ticks(batch_size=500)
                
                if ticks is not None and not ticks.empty:
                    # 去重（避免重复保存）
                    if self.tick_buffer:
                        existing_times = set(self.tick_buffer[-1]['datetime']) if 'datetime' in self.tick_buffer[-1].columns else set()
                        new_ticks = ticks[~ticks['datetime'].isin(existing_times)] if 'datetime' in ticks.columns else ticks
                    else:
                        new_ticks = ticks
                    
                    if not new_ticks.empty:
                        self.tick_buffer.append(new_ticks)
                        
                        latest_time = new_ticks['datetime'].max().strftime('%H:%M:%S') if 'datetime' in new_ticks.columns else 'N/A'
                        latest_price = new_ticks['price'].iloc[-1] if 'price' in new_ticks.columns else 0
                        
                        self._log(f"✅ 采集 {len(new_ticks)} 条新Tick | 最新: {latest_time} ${latest_price:.2f}")
                        
                        # 缓冲区满了就保存
                        total_buffered = sum(len(df) for df in self.tick_buffer)
                        if total_buffered >= self.buffer_size:
                            df_to_save = pd.concat(self.tick_buffer, ignore_index=True)
                            self.save_batch(df_to_save, 'realtime')
                            self.tick_buffer = []
                            self._log(f"💾 缓冲区已保存，共 {total_buffered} 条")
                
                # 等待下一次采集
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            self._log(f"\n⏹️  收到停止信号，保存缓冲区数据...")
            if self.tick_buffer:
                df_to_save = pd.concat(self.tick_buffer, ignore_index=True)
                self.save_batch(df_to_save, 'realtime')
            self._log(f"✅ 采集器已停止")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Tick数据持续采集器')
    parser.add_argument('--symbol', default='SIL2603', help='期货合约代码')
    parser.add_argument('--mode', choices=['backfill', 'realtime', 'both'], default='both', 
                       help='运行模式：backfill（回填历史）, realtime（实时采集）, both（两者）')
    parser.add_argument('--interval', type=int, default=60, help='实时采集间隔（秒）')
    parser.add_argument('--max-backfill', type=int, default=100, help='最大回填批次数')
    
    args = parser.parse_args()
    
    collector = TickDataCollector(symbol=args.symbol)
    
    if args.mode in ['backfill', 'both']:
        collector.collect_historical_ticks(max_batches=args.max_backfill)
    
    if args.mode in ['realtime', 'both']:
        collector.run_realtime_collector(interval_seconds=args.interval)


if __name__ == '__main__':
    main()
