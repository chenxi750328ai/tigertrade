#!/usr/bin/env python3
"""
交易记录分析工具
分析用户的真实交易记录，提取策略规则，验证是否运气
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json

class TradingRecordAnalyzer:
    """交易记录分析器"""
    
    def __init__(self):
        self.rounds = []  # 每一轮完整交易
        
    def load_records(self, filepath):
        """加载交易记录"""
        # 支持多种格式：CSV, Excel, JSON
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.xlsx'):
            df = pd.read_excel(filepath)
        elif filepath.endswith('.json'):
            with open(filepath) as f:
                df = pd.DataFrame(json.load(f))
        else:
            raise ValueError("不支持的文件格式")
        
        return df
    
    def analyze_round(self, round_df):
        """分析单轮交易"""
        
        trades = []
        for _, row in round_df.iterrows():
            trades.append({
                'time': row.get('time', row.get('timestamp', None)),
                'action': row.get('action', row.get('side', None)),  # BUY/SELL
                'price': float(row.get('price', 0)),
                'quantity': int(row.get('quantity', row.get('size', 1)))
            })
        
        # 计算统计
        buy_trades = [t for t in trades if t['action'] in ['BUY', 'LONG', '买入']]
        sell_trades = [t for t in trades if t['action'] in ['SELL', 'SHORT', '卖出', '平仓']]
        
        if not buy_trades or not sell_trades:
            return None
        
        # 平均成本
        total_quantity = sum(t['quantity'] for t in buy_trades)
        avg_cost = sum(t['price'] * t['quantity'] for t in buy_trades) / total_quantity
        
        # 卖出均价
        avg_sell = sum(t['price'] * t['quantity'] for t in sell_trades) / sum(t['quantity'] for t in sell_trades)
        
        # 盈亏
        profit = (avg_sell - avg_cost) * total_quantity
        
        # 最大持仓
        max_position = 0
        current_position = 0
        for t in trades:
            if t['action'] in ['BUY', 'LONG', '买入']:
                current_position += t['quantity']
                max_position = max(max_position, current_position)
            else:
                current_position -= t['quantity']
        
        # 最大浮亏（简化计算）
        min_price = min(t['price'] for t in trades)
        max_drawdown = (avg_cost - min_price) * total_quantity
        
        # 加仓间距
        if len(buy_trades) > 1:
            price_gaps = []
            for i in range(1, len(buy_trades)):
                gap = buy_trades[i-1]['price'] - buy_trades[i]['price']
                price_gaps.append(gap)
            avg_gap = np.mean(price_gaps)
        else:
            avg_gap = 0
        
        return {
            'num_entries': len(buy_trades),
            'max_position': max_position,
            'avg_cost': avg_cost,
            'avg_sell': avg_sell,
            'profit': profit,
            'profit_pct': (avg_sell - avg_cost) / avg_cost * 100,
            'max_drawdown': max_drawdown,
            'avg_gap': avg_gap,
            'first_price': buy_trades[0]['price'],
            'last_buy_price': buy_trades[-1]['price'],
            'trades': trades
        }
    
    def analyze_all(self, records):
        """分析所有交易"""
        
        print("=" * 80)
        print("📊 交易记录分析报告")
        print("=" * 80)
        
        results = []
        for i, round_data in enumerate(records, 1):
            result = self.analyze_round(round_data)
            if result:
                result['round_id'] = i
                results.append(result)
        
        if not results:
            print("❌ 没有有效的交易记录")
            return
        
        df = pd.DataFrame(results)
        
        # 总体统计
        print(f"\n【总体统计】")
        print(f"  总交易轮数: {len(results)}")
        print(f"  胜率: {(df['profit'] > 0).sum() / len(results) * 100:.1f}%")
        print(f"  总盈利: ${df['profit'].sum():,.2f}")
        print(f"  平均盈利: ${df['profit'].mean():,.2f}")
        print(f"  平均盈利率: {df['profit_pct'].mean():.2f}%")
        print(f"  最大单轮盈利: ${df['profit'].max():,.2f}")
        print(f"  最大单轮亏损: ${df['profit'].min():,.2f}")
        
        # 加仓统计
        print(f"\n【加仓统计】")
        print(f"  平均加仓次数: {df['num_entries'].mean():.1f}")
        print(f"  最多加仓次数: {df['num_entries'].max()}")
        print(f"  平均最大持仓: {df['max_position'].mean():.1f}手")
        print(f"  最大持仓峰值: {df['max_position'].max()}手")
        
        # 风险统计
        print(f"\n【风险统计】")
        print(f"  平均最大浮亏: ${df['max_drawdown'].mean():,.2f}")
        print(f"  最大浮亏峰值: ${df['max_drawdown'].max():,.2f}")
        print(f"  平均加仓间距: {df['avg_gap'].mean():.2f}点")
        
        # 策略规则提取
        print(f"\n【策略规则提取】")
        print(f"\n1. 加仓触发条件：")
        print(f"   • 平均加仓间距: {df['avg_gap'].mean():.2f}点")
        print(f"   • 标准差: {df['avg_gap'].std():.2f}点")
        print(f"   → 建议规则: 价格下跌{df['avg_gap'].mean():.0f}±{df['avg_gap'].std():.0f}点时加仓")
        
        print(f"\n2. 仓位管理：")
        print(f"   • 平均首次开仓价: {df['first_price'].mean():.2f}")
        print(f"   • 平均最低加仓价: {df['last_buy_price'].mean():.2f}")
        print(f"   • 平均价格跌幅: {(1 - df['last_buy_price'].mean() / df['first_price'].mean()) * 100:.1f}%")
        
        print(f"\n3. 止盈策略：")
        winning = df[df['profit'] > 0]
        if len(winning) > 0:
            print(f"   • 平均盈利率: {winning['profit_pct'].mean():.2f}%")
            print(f"   • 最小盈利率: {winning['profit_pct'].min():.2f}%")
            print(f"   • 最大盈利率: {winning['profit_pct'].max():.2f}%")
            print(f"   → 建议规则: 盈利达到{winning['profit_pct'].min():.1f}%以上考虑止盈")
        
        # 风险评估
        print(f"\n【风险评估】")
        total_profit = df['profit'].sum()
        max_loss = abs(df['profit'].min())
        risk_reward = total_profit / max_loss if max_loss > 0 else float('inf')
        
        print(f"  总盈利: ${total_profit:,.2f}")
        print(f"  最大单次亏损: ${max_loss:,.2f}")
        print(f"  盈亏比: {risk_reward:.2f}:1")
        print(f"  → 一次最大亏损需要{max_loss/df['profit'].mean():.1f}次平均盈利才能补回")
        
        # 是策略还是运气？
        print(f"\n【策略 vs 运气评估】")
        print(f"\n✅ 支持"策略有效"的证据：")
        if len(results) > 20:
            print(f"  • 样本量充足: {len(results)}轮交易")
        if df['profit'].std() / df['profit'].mean() < 2:
            print(f"  • 盈利稳定: 标准差/均值比率 = {df['profit'].std() / df['profit'].mean():.2f}")
        if (df['profit'] > 0).sum() / len(results) > 0.8:
            print(f"  • 高胜率: {(df['profit'] > 0).sum() / len(results) * 100:.1f}%")
        
        print(f"\n⚠️  潜在风险：")
        if df['max_position'].max() >= 5:
            print(f"  • 最大持仓达{df['max_position'].max()}手，资金压力大")
        if df['max_drawdown'].max() > 5000:
            print(f"  • 最大浮亏达${df['max_drawdown'].max():,.2f}，心理压力大")
        if len(results) < 30:
            print(f"  • 样本量较小({len(results)}轮)，可能还未经历极端行情")
        if risk_reward < 5:
            print(f"  • 盈亏比偏低({risk_reward:.1f}:1)，一次失败影响大")
        
        # 保存分析结果
        df.to_csv('/home/cx/tigertrade/analysis/trading_analysis_results.csv', index=False)
        print(f"\n💾 详细结果已保存至: /home/cx/tigertrade/analysis/trading_analysis_results.csv")
        
        return df

if __name__ == '__main__':
    print("交易记录分析工具已准备就绪！")
    print("\n使用方法：")
    print("1. 准备交易记录文件（CSV/Excel/JSON）")
    print("2. analyzer = TradingRecordAnalyzer()")
    print("3. records = analyzer.load_records('your_file.csv')")
    print("4. analyzer.analyze_all(records)")
