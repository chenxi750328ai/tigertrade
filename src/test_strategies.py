#!/usr/bin/env python3
"""
策略测试脚本 - 测试所有大模型策略的训练和推理效果
"""

import sys
import os
import time
import json
from datetime import datetime
import pandas as pd

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入tiger1模块的必要函数
from tiger1 import (
    get_kline_data, calculate_indicators, 
    FUTURE_SYMBOL, GRID_PERIOD
)

# 导入所有策略模块
try:
    from strategies import llm_strategy
    from strategies import large_model_strategy
    from strategies import huge_transformer_strategy
    from strategies import enhanced_transformer_strategy
    from strategies import rl_trading_strategy
    from strategies import model_comparison_strategy
    from strategies import large_transformer_strategy
except ImportError as e:
    print(f"❌ 导入策略模块失败: {e}")
    sys.exit(1)


class StrategyTester:
    """策略测试器"""
    
    def __init__(self, iterations=10):
        """
        初始化测试器
        
        Args:
            iterations: 每个策略的测试迭代次数
        """
        self.iterations = iterations
        self.results = {}
        
        # 初始化所有策略
        self.strategies = {
            'LLM策略': llm_strategy.LLMTradingStrategy(),
            '大模型策略': large_model_strategy.LargeModelStrategy(),
            '超大Transformer策略': huge_transformer_strategy.HugeTransformerStrategy(),
            '增强型Transformer策略': enhanced_transformer_strategy.EnhancedTransformerStrategy(),
            '强化学习策略': rl_trading_strategy.RLTradingStrategy(),
            '大型Transformer策略': large_transformer_strategy.LargeTransformerStrategy(),
        }
        
        print("=" * 80)
        print("🚀 策略测试器初始化完成")
        print(f"📊 将测试 {len(self.strategies)} 个策略，每个策略运行 {iterations} 次迭代")
        print("=" * 80)
    
    def get_market_data(self):
        """获取市场数据"""
        try:
            df_5m = get_kline_data([FUTURE_SYMBOL], '5min', count=GRID_PERIOD + 5)
            df_1m = get_kline_data([FUTURE_SYMBOL], '1min', count=GRID_PERIOD + 5)
            
            if df_5m.empty or df_1m.empty:
                print("⚠️ 数据为空")
                return None, None, None
            
            # 计算技术指标
            inds = calculate_indicators(df_5m, df_1m)
            if '5m' not in inds or '1m' not in inds:
                print("⚠️ 指标计算失败")
                return None, None, None
            
            return df_5m, df_1m, inds
            
        except Exception as e:
            print(f"❌ 获取市场数据失败: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def prepare_features(self, inds):
        """准备模型特征数据"""
        price_current = inds['1m']['close']
        atr = inds['5m']['atr']
        rsi_1m = inds['1m']['rsi']
        rsi_5m = inds['5m']['rsi']
        
        # 使用硬编码的网格值
        grid_upper = price_current * 1.01  # 1% 上涨
        grid_lower = price_current * 0.99  # 1% 下跌
        
        # 计算缓冲区
        buffer = max(atr * 0.3, 0.0025)
        threshold = grid_lower + buffer
        
        current_data = {
            'price_current': price_current,
            'grid_lower': grid_lower,
            'grid_upper': grid_upper,
            'atr': atr,
            'rsi_1m': rsi_1m,
            'rsi_5m': rsi_5m,
            'buffer': buffer,
            'threshold': threshold,
            'near_lower': price_current <= threshold,
            'rsi_ok': rsi_1m < 30 or (rsi_5m > 45 and rsi_5m < 55)
        }
        
        return current_data
    
    def test_strategy(self, strategy_name, strategy):
        """
        测试单个策略
        
        Args:
            strategy_name: 策略名称
            strategy: 策略实例
        """
        print(f"\n{'=' * 80}")
        print(f"🧪 开始测试: {strategy_name}")
        print(f"{'=' * 80}")
        
        predictions = []
        errors = 0
        
        for i in range(self.iterations):
            try:
                # 获取市场数据
                df_5m, df_1m, inds = self.get_market_data()
                if inds is None:
                    errors += 1
                    time.sleep(2)
                    continue
                
                # 准备特征
                current_data = self.prepare_features(inds)
                
                # 模型预测
                start_time = time.time()
                action, confidence = strategy.predict_action(current_data)
                inference_time = time.time() - start_time
                
                action_map = {0: "持有/不操作", 1: "买入", 2: "卖出"}
                
                # 记录预测结果
                result = {
                    'iteration': i + 1,
                    'timestamp': datetime.now().isoformat(),
                    'action': action,
                    'action_name': action_map.get(action, "未知"),
                    'confidence': confidence,
                    'inference_time': inference_time,
                    'price': current_data['price_current'],
                    'atr': current_data['atr'],
                    'rsi_1m': current_data['rsi_1m'],
                    'rsi_5m': current_data['rsi_5m'],
                }
                predictions.append(result)
                
                # 打印进度
                print(f"  [{i+1}/{self.iterations}] "
                      f"预测: {result['action_name']} | "
                      f"置信度: {confidence:.3f} | "
                      f"推理时间: {inference_time*1000:.2f}ms | "
                      f"价格: {current_data['price_current']:.3f}")
                
                # 等待一段时间
                time.sleep(1)
                
            except Exception as e:
                print(f"  ❌ 迭代 {i+1} 失败: {e}")
                errors += 1
                import traceback
                traceback.print_exc()
        
        # 保存结果
        self.results[strategy_name] = {
            'predictions': predictions,
            'total_iterations': self.iterations,
            'successful_iterations': len(predictions),
            'errors': errors
        }
        
        # 打印统计
        if predictions:
            self._print_statistics(strategy_name, predictions)
    
    def _print_statistics(self, strategy_name, predictions):
        """打印策略统计信息"""
        print(f"\n{'─' * 80}")
        print(f"📈 {strategy_name} - 统计结果")
        print(f"{'─' * 80}")
        
        # 基本统计
        total = len(predictions)
        buy_count = sum(1 for p in predictions if p['action'] == 1)
        sell_count = sum(1 for p in predictions if p['action'] == 2)
        hold_count = sum(1 for p in predictions if p['action'] == 0)
        
        avg_confidence = sum(p['confidence'] for p in predictions) / total
        avg_inference_time = sum(p['inference_time'] for p in predictions) / total
        
        print(f"  总预测次数: {total}")
        print(f"  买入信号: {buy_count} ({buy_count/total*100:.1f}%)")
        print(f"  卖出信号: {sell_count} ({sell_count/total*100:.1f}%)")
        print(f"  持有信号: {hold_count} ({hold_count/total*100:.1f}%)")
        print(f"  平均置信度: {avg_confidence:.3f}")
        print(f"  平均推理时间: {avg_inference_time*1000:.2f}ms")
        
        # 高置信度预测
        high_conf = [p for p in predictions if p['confidence'] > 0.7]
        if high_conf:
            print(f"\n  高置信度预测 (>0.7): {len(high_conf)} 次")
            buy_high = sum(1 for p in high_conf if p['action'] == 1)
            sell_high = sum(1 for p in high_conf if p['action'] == 2)
            print(f"    买入: {buy_high}, 卖出: {sell_high}")
    
    def run_all_tests(self):
        """运行所有策略测试"""
        print("\n" + "=" * 80)
        print("🚀 开始测试所有策略")
        print("=" * 80)
        
        start_time = time.time()
        
        for strategy_name, strategy in self.strategies.items():
            self.test_strategy(strategy_name, strategy)
        
        total_time = time.time() - start_time
        
        # 打印总结报告
        self._print_summary(total_time)
        
        # 保存结果到文件
        self._save_results()
    
    def _print_summary(self, total_time):
        """打印总结报告"""
        print("\n" + "=" * 80)
        print("📊 测试总结报告")
        print("=" * 80)
        
        print(f"\n总测试时间: {total_time:.2f}秒")
        print(f"\n策略对比:")
        print(f"{'策略名称':<25} {'成功率':<10} {'平均置信度':<12} {'平均推理时间':<15} {'买入%':<10} {'卖出%':<10}")
        print("-" * 80)
        
        for strategy_name, result in self.results.items():
            if result['predictions']:
                predictions = result['predictions']
                success_rate = len(predictions) / result['total_iterations'] * 100
                avg_conf = sum(p['confidence'] for p in predictions) / len(predictions)
                avg_time = sum(p['inference_time'] for p in predictions) / len(predictions) * 1000
                buy_pct = sum(1 for p in predictions if p['action'] == 1) / len(predictions) * 100
                sell_pct = sum(1 for p in predictions if p['action'] == 2) / len(predictions) * 100
                
                print(f"{strategy_name:<25} {success_rate:>6.1f}%   {avg_conf:>8.3f}    "
                      f"{avg_time:>10.2f}ms    {buy_pct:>6.1f}%   {sell_pct:>6.1f}%")
    
    def _save_results(self):
        """保存测试结果到文件"""
        try:
            # 创建结果目录
            results_dir = '/home/cx/trading_data/strategy_tests'
            os.makedirs(results_dir, exist_ok=True)
            
            # 生成文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            json_file = os.path.join(results_dir, f'test_results_{timestamp}.json')
            
            # 保存为JSON
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            
            print(f"\n✅ 测试结果已保存到: {json_file}")
            
            # 也保存为CSV格式（便于分析）
            csv_file = os.path.join(results_dir, f'test_results_{timestamp}.csv')
            all_predictions = []
            for strategy_name, result in self.results.items():
                for pred in result['predictions']:
                    pred_copy = pred.copy()
                    pred_copy['strategy'] = strategy_name
                    all_predictions.append(pred_copy)
            
            if all_predictions:
                df = pd.DataFrame(all_predictions)
                df.to_csv(csv_file, index=False, encoding='utf-8')
                print(f"✅ CSV格式结果已保存到: {csv_file}")
                
        except Exception as e:
            print(f"⚠️ 保存结果失败: {e}")


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("🧪 Tiger Trade 策略测试工具")
    print("=" * 80)
    
    # 解析命令行参数
    iterations = 10  # 默认每个策略测试10次
    if len(sys.argv) > 1:
        try:
            iterations = int(sys.argv[1])
        except ValueError:
            print("⚠️ 无效的迭代次数，使用默认值10")
    
    # 创建测试器并运行
    tester = StrategyTester(iterations=iterations)
    tester.run_all_tests()
    
    print("\n" + "=" * 80)
    print("✅ 测试完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
