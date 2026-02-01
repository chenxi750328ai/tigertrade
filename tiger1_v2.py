#!/usr/bin/env python3
"""
TigerTrade V2.0 - 实盘交易主程序
模块化架构，协调各模块执行实时交易

使用方法:
    python tiger1_v2.py --strategy transformer --interval 60
"""

import time
import argparse
from datetime import datetime
from pathlib import Path

# 导入各模块
from src.data_collector import RealTimeDataCollector
from src.strategies import get_strategy
from src.risk import RiskManager


class LiveTradingSystem:
    """
    实盘交易系统
    
    协调各模块，执行实时交易
    """
    
    def __init__(self, strategy_name='grid', symbol='SIL2603', interval=60):
        """
        初始化
        
        Args:
            strategy_name: 策略名称 ('grid', 'transformer')
            symbol: 合约代码
            interval: 执行间隔（秒）
        """
        self.symbol = symbol
        self.interval = interval
        
        print(f"="*80)
        print(f"🚀 TigerTrade V2.0 实盘交易系统")
        print(f"="*80)
        print(f"策略: {strategy_name}")
        print(f"合约: {symbol}")
        print(f"间隔: {interval}秒")
        print(f"="*80)
        
        # 初始化各模块
        self._init_modules(strategy_name)
        
        # 状态
        self.is_running = False
        self.position = 0.0
        self.entry_price = 0.0
        self.account_value = 10000.0  # TODO: 从API获取
        
    def _init_modules(self, strategy_name):
        """初始化各模块"""
        print(f"\n初始化模块...")
        
        # Module 1: 数据采集
        print(f"  [1/3] 数据采集器...")
        self.data_collector = RealTimeDataCollector(symbol=self.symbol)
        
        # Module 5: 策略引擎
        print(f"  [2/3] 策略引擎 ({strategy_name})...")
        self.strategy = get_strategy(strategy_name)
        
        # Module 7: 风险控制
        print(f"  [3/3] 风险管理...")
        self.risk_manager = RiskManager()
        
        print(f"✅ 所有模块初始化完成\n")
    
    def run(self):
        """主循环"""
        self.is_running = True
        
        print(f"{'='*80}")
        print(f"开始实盘交易 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
        print(f"{'='*80}\n")
        
        iteration = 0
        
        try:
            while self.is_running:
                iteration += 1
                print(f"\n{'─'*80}")
                print(f"第 {iteration} 轮 | {datetime.now().strftime('%H:%M:%S')}")
                print(f"{'─'*80}")
                
                # 执行一次交易循环
                self._execute_one_cycle()
                
                # 等待下一个周期
                print(f"\n⏳ 等待 {self.interval} 秒...")
                time.sleep(self.interval)
                
        except KeyboardInterrupt:
            print(f"\n\n⏹️  收到停止信号")
            self.stop()
        except Exception as e:
            print(f"\n\n❌ 发生错误: {e}")
            import traceback
            traceback.print_exc()
            self.stop()
    
    def _execute_one_cycle(self):
        """执行一次交易循环"""
        try:
            # 1. 获取实时数据 (Module 1)
            print(f"📊 获取实时数据...")
            data = self.data_collector.get_multi_period_data(
                periods=['1m', '5m', '1h'],
                counts={'1m': 100, '5m': 100, '1h': 100}
            )
            
            if data['1m'] is None or data['1m'].empty:
                print(f"⚠️ 数据获取失败，跳过本轮")
                return
            
            current_price = data['1m']['close'].iloc[-1]
            print(f"   当前价格: ${current_price:.2f}")
            
            # 2. 策略信号 (Module 5)
            print(f"🎯 生成交易信号...")
            signal = self.strategy.generate_signal(
                data,
                entry_price=self.entry_price,
                position=self.position
            )
            
            print(f"   信号: {signal['action']}")
            print(f"   置信度: {signal['confidence']:.2f}")
            print(f"   原因: {signal['reason']}")
            
            # 3. 风险检查 (Module 7)
            if signal['action'] != 'HOLD':
                print(f"🛡️  风险检查...")
                if not self.risk_manager.check_signal(signal, self.account_value):
                    print(f"   ⛔ 风险检查未通过，取消交易")
                    return
                print(f"   ✅ 风险检查通过")
            
            # 4. 执行交易
            self._execute_signal(signal, current_price)
            
            # 5. 状态显示
            self._print_status(current_price)
            
        except Exception as e:
            print(f"❌ 执行失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _execute_signal(self, signal, current_price):
        """执行交易信号"""
        action = signal['action']
        
        if action == 'BUY' and self.position == 0:
            print(f"\n📈 执行买入")
            print(f"   仓位: {signal['position_size']*100:.0f}%")
            print(f"   价格: ${current_price:.2f}")
            
            # TODO: 实际下单
            # order = place_tiger_order(...)
            
            self.position = signal['position_size']
            self.entry_price = current_price
            
        elif action == 'SELL' and self.position > 0:
            profit = (current_price - self.entry_price) / self.entry_price
            print(f"\n📉 执行卖出")
            print(f"   盈亏: {profit*100:+.2f}%")
            print(f"   价格: ${current_price:.2f}")
            
            # TODO: 实际下单
            # order = place_tiger_order(...)
            
            self.position = 0.0
            self.entry_price = 0.0
        
        elif action == 'HOLD':
            pass  # 无操作
    
    def _print_status(self, current_price):
        """打印当前状态"""
        print(f"\n{'─'*40}")
        print(f"💼 当前状态")
        print(f"{'─'*40}")
        print(f"持仓: {self.position*100:.0f}%")
        
        if self.position > 0:
            unrealized_pnl = (current_price - self.entry_price) / self.entry_price
            print(f"入场价: ${self.entry_price:.2f}")
            print(f"浮动盈亏: {unrealized_pnl*100:+.2f}%")
        
        print(f"账户价值: ${self.account_value:.2f}")
        print(f"{'─'*40}")
    
    def stop(self):
        """停止系统"""
        self.is_running = False
        print(f"\n{'='*80}")
        print(f"✅ 系统已停止")
        print(f"{'='*80}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='TigerTrade V2.0 实盘交易')
    parser.add_argument('--strategy', default='grid', 
                       help='策略名称 (grid, transformer)')
    parser.add_argument('--symbol', default='SIL2603',
                       help='合约代码')
    parser.add_argument('--interval', type=int, default=60,
                       help='执行间隔（秒）')
    
    args = parser.parse_args()
    
    # 创建系统
    system = LiveTradingSystem(
        strategy_name=args.strategy,
        symbol=args.symbol,
        interval=args.interval
    )
    
    # 运行
    system.run()


if __name__ == '__main__':
    main()
