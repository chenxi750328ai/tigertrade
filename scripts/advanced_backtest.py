#!/usr/bin/env python3
"""
高级回测系统 - 目标：20%月收益

改进点：
1. 做空机制
2. 杠杆交易（2-3倍）
3. 动态仓位管理
4. 多策略组合
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

class AdvancedBacktester:
    """高级回测系统"""
    
    def __init__(
        self,
        model_path: str,
        data_path: str,
        initial_capital: float = 100000.0,
        leverage: float = 2.0,
        max_position_size: float = 0.3  # 最大仓位30%
    ):
        """
        初始化回测器
        
        Args:
            model_path: 模型路径
            data_path: 测试数据路径
            initial_capital: 初始资金
            leverage: 杠杆倍数
            max_position_size: 最大仓位比例
        """
        self.model = self._load_model(model_path)
        self.data = pd.read_csv(data_path)
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.max_position_size = max_position_size
        
        # 回测状态
        self.capital = initial_capital
        self.position = 0  # 持仓数量（正数=多头，负数=空头）
        self.position_value = 0
        self.trades = []
        self.equity_curve = []
        
    def _load_model(self, model_path: str):
        """加载模型"""
        print(f"📦 加载模型: {model_path}")
        model = torch.load(model_path, map_location='cpu', weights_only=False)
        model.eval()
        return model
    
    def calculate_position_size(
        self,
        prediction: float,
        confidence: float,
        current_price: float
    ) -> float:
        """
        动态仓位管理
        
        Args:
            prediction: 预测收益率
            confidence: 预测置信度
            current_price: 当前价格
            
        Returns:
            建议仓位大小（正数=做多，负数=做空）
        """
        # 基础仓位：根据预测幅度
        base_size = abs(prediction) * 10  # 预测1%收益 → 10%仓位
        
        # 置信度调整
        adjusted_size = base_size * confidence
        
        # 限制最大仓位
        position_ratio = min(adjusted_size, self.max_position_size)
        
        # 计算实际股数（考虑杠杆）
        available_capital = self.capital * self.leverage
        position_capital = available_capital * position_ratio
        shares = position_capital / current_price
        
        # 做多或做空
        if prediction > 0:
            return shares  # 做多
        else:
            return -shares  # 做空
    
    def execute_trade(
        self,
        target_position: float,
        current_price: float,
        timestamp: pd.Timestamp,
        prediction: float
    ):
        """
        执行交易
        
        Args:
            target_position: 目标仓位
            current_price: 当前价格
            timestamp: 时间戳
            prediction: 预测值
        """
        # 计算交易量
        trade_amount = target_position - self.position
        
        if abs(trade_amount) < 0.01:  # 忽略微小调整
            return
        
        # 交易成本（手续费 + 滑点）
        commission_rate = 0.0003  # 0.03%手续费
        slippage_rate = 0.0002  # 0.02%滑点
        total_cost_rate = commission_rate + slippage_rate
        
        trade_value = abs(trade_amount) * current_price
        trade_cost = trade_value * total_cost_rate
        
        # 更新持仓
        self.position = target_position
        self.position_value = self.position * current_price
        
        # 更新资金（扣除交易成本）
        self.capital -= trade_cost
        
        # 记录交易
        trade_record = {
            'timestamp': timestamp,
            'price': current_price,
            'amount': trade_amount,
            'position_after': self.position,
            'capital_after': self.capital,
            'cost': trade_cost,
            'prediction': prediction,
            'action': 'BUY' if trade_amount > 0 else 'SELL'
        }
        self.trades.append(trade_record)
    
    def update_equity(self, current_price: float, timestamp: pd.Timestamp):
        """更新权益曲线"""
        # 当前持仓市值
        position_value = self.position * current_price
        
        # 总权益 = 现金 + 持仓市值
        total_equity = self.capital + position_value
        
        # 记录权益
        self.equity_curve.append({
            'timestamp': timestamp,
            'capital': self.capital,
            'position_value': position_value,
            'total_equity': total_equity,
            'position': self.position
        })
    
    def run_backtest(self):
        """执行回测"""
        print("="*70)
        print("🚀 开始高级回测")
        print("="*70)
        print(f"初始资金: ${self.initial_capital:,.2f}")
        print(f"杠杆倍数: {self.leverage}x")
        print(f"最大仓位: {self.max_position_size*100}%")
        print("="*70)
        
        # 准备特征
        feature_columns = [col for col in self.data.columns 
                          if col not in ['timestamp', 'target', 'price_current']]
        
        features = self.data[feature_columns].values
        
        # 标准化特征
        features_mean = np.mean(features, axis=0)
        features_std = np.std(features, axis=0)
        features_std[features_std == 0] = 1
        features_normalized = (features - features_mean) / features_std
        
        # 转换为tensor
        features_tensor = torch.FloatTensor(features_normalized).unsqueeze(1)
        
        # 批量预测
        print("\n📊 生成预测...")
        with torch.no_grad():
            predictions = self.model(features_tensor).squeeze().numpy()
        
        print(f"   预测数据点: {len(predictions)}")
        print(f"   预测范围: [{predictions.min():.4f}, {predictions.max():.4f}]")
        
        # 模拟交易
        print("\n💹 开始交易模拟...")
        
        for i in range(len(self.data)):
            row = self.data.iloc[i]
            current_price = row['price_current']
            timestamp = pd.to_datetime(row['timestamp']) if 'timestamp' in row else pd.Timestamp.now()
            prediction = predictions[i]
            
            # 计算置信度（基于预测绝对值）
            confidence = min(abs(prediction) / 0.05, 1.0)  # 预测5%收益=100%置信
            
            # 策略1: 高置信度大仓位
            if abs(prediction) > 0.02 and confidence > 0.4:
                target_position = self.calculate_position_size(
                    prediction, confidence, current_price
                )
                self.execute_trade(target_position, current_price, timestamp, prediction)
            
            # 策略2: 止损（持仓亏损超过3%）
            elif self.position != 0:
                current_value = self.position * current_price
                entry_value = self.position_value  # 上次交易后的价值
                
                if entry_value != 0:
                    pnl_ratio = (current_value - entry_value) / abs(entry_value)
                    
                    if pnl_ratio < -0.03:  # 亏损超过3%，平仓
                        self.execute_trade(0, current_price, timestamp, prediction)
            
            # 更新权益
            self.update_equity(current_price, timestamp)
            
            # 进度显示
            if (i + 1) % 1000 == 0:
                equity = self.equity_curve[-1]['total_equity']
                return_pct = (equity - self.initial_capital) / self.initial_capital * 100
                print(f"   进度: {i+1}/{len(self.data)} | 当前收益: {return_pct:+.2f}%")
        
        # 回测结束，平掉所有仓位
        if self.position != 0:
            final_price = self.data.iloc[-1]['price_current']
            final_timestamp = pd.to_datetime(self.data.iloc[-1]['timestamp']) if 'timestamp' in self.data.iloc[-1] else pd.Timestamp.now()
            self.execute_trade(0, final_price, final_timestamp, 0)
        
        print("\n" + "="*70)
        print("✅ 回测完成")
        print("="*70)
    
    def calculate_metrics(self):
        """计算回测指标"""
        equity_df = pd.DataFrame(self.equity_curve)
        
        # 总收益
        final_equity = equity_df['total_equity'].iloc[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        
        # 计算测试数据的时间跨度
        if 'timestamp' in self.data:
            start_date = pd.to_datetime(self.data['timestamp'].iloc[0])
            end_date = pd.to_datetime(self.data['timestamp'].iloc[-1])
            days = (end_date - start_date).days
        else:
            # 假设是1分钟数据
            days = len(self.data) / (60 * 24)  # 转换为天数
        
        # 月化收益
        months = days / 30.0
        monthly_return = (total_return / months) if months > 0 else 0
        
        # 计算收益率序列
        equity_df['returns'] = equity_df['total_equity'].pct_change()
        
        # Sharpe比率（假设无风险利率=0）
        returns_mean = equity_df['returns'].mean()
        returns_std = equity_df['returns'].std()
        sharpe_ratio = (returns_mean / returns_std * np.sqrt(252)) if returns_std > 0 else 0
        
        # 最大回撤
        equity_df['cummax'] = equity_df['total_equity'].cummax()
        equity_df['drawdown'] = (equity_df['total_equity'] - equity_df['cummax']) / equity_df['cummax']
        max_drawdown = equity_df['drawdown'].min()
        
        # 交易统计
        winning_trades = [t for t in self.trades if t.get('prediction', 0) * t.get('amount', 0) > 0]
        total_trades = len(self.trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        metrics = {
            'initial_capital': self.initial_capital,
            'final_equity': final_equity,
            'total_return': total_return * 100,
            'monthly_return': monthly_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown * 100,
            'total_trades': total_trades,
            'win_rate': win_rate * 100,
            'test_days': days,
            'leverage': self.leverage
        }
        
        return metrics
    
    def print_summary(self, metrics: dict):
        """打印回测总结"""
        print("\n" + "="*70)
        print("📊 回测结果总结")
        print("="*70)
        
        print(f"\n💰 收益指标:")
        print(f"   初始资金: ${metrics['initial_capital']:,.2f}")
        print(f"   最终权益: ${metrics['final_equity']:,.2f}")
        print(f"   总收益率: {metrics['total_return']:+.2f}%")
        print(f"   月收益率: {metrics['monthly_return']:+.2f}%")
        
        # 目标达成检查
        target_achieved = metrics['monthly_return'] >= 20.0
        target_symbol = "✅" if target_achieved else "❌"
        print(f"\n🎯 目标达成: {target_symbol}")
        print(f"   目标月收益: 20.00%")
        print(f"   实际月收益: {metrics['monthly_return']:+.2f}%")
        print(f"   差距: {metrics['monthly_return'] - 20.0:+.2f}%")
        
        print(f"\n📈 风险指标:")
        print(f"   Sharpe比率: {metrics['sharpe_ratio']:.2f}")
        print(f"   最大回撤: {metrics['max_drawdown']:.2f}%")
        print(f"   杠杆倍数: {metrics['leverage']}x")
        
        print(f"\n💹 交易统计:")
        print(f"   总交易次数: {metrics['total_trades']}")
        print(f"   胜率: {metrics['win_rate']:.2f}%")
        print(f"   测试天数: {metrics['test_days']:.1f}")
        
        print("\n" + "="*70)
        
        return target_achieved
    
    def save_results(self, output_dir: str = "/home/cx/tigertrade/results"):
        """保存回测结果"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存指标
        metrics = self.calculate_metrics()
        metrics_file = output_path / f"backtest_metrics_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n💾 结果已保存:")
        print(f"   指标: {metrics_file}")
        
        # 保存权益曲线
        equity_df = pd.DataFrame(self.equity_curve)
        equity_file = output_path / f"equity_curve_{timestamp}.csv"
        equity_df.to_csv(equity_file, index=False)
        print(f"   权益曲线: {equity_file}")
        
        # 保存交易记录
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_file = output_path / f"trades_{timestamp}.csv"
            trades_df.to_csv(trades_file, index=False)
            print(f"   交易记录: {trades_file}")
        
        return metrics


def main():
    """主函数"""
    print("="*70)
    print("🎯 TigerTrade高级回测系统")
    print("目标: 月收益率 20%")
    print("="*70)
    
    # 配置
    model_path = "/home/cx/tigertrade/models/transformer_best.pth"
    data_path = "/home/cx/tigertrade/data/processed/test.csv"
    
    # 检查文件
    if not Path(model_path).exists():
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    if not Path(data_path).exists():
        print(f"❌ 数据文件不存在: {data_path}")
        return
    
    # 创建回测器
    backtester = AdvancedBacktester(
        model_path=model_path,
        data_path=data_path,
        initial_capital=100000.0,
        leverage=2.5,  # 2.5倍杠杆
        max_position_size=0.4  # 最大40%仓位
    )
    
    # 执行回测
    backtester.run_backtest()
    
    # 计算指标
    metrics = backtester.calculate_metrics()
    
    # 打印总结
    target_achieved = backtester.print_summary(metrics)
    
    # 保存结果
    backtester.save_results()
    
    # 返回状态
    return 0 if target_achieved else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
