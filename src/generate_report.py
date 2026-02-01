#!/usr/bin/env python3
"""
生成测试报告 - 分析所有策略的表现
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_test_results():
    """加载测试结果"""
    results_dir = '/home/cx/trading_data/strategy_tests'
    
    # 找到最新的测试结果
    files = [f for f in os.listdir(results_dir) if f.startswith('test_results_') and f.endswith('.json')]
    
    if not files:
        print("❌ 未找到测试结果文件")
        return None
    
    # 按时间排序，取最新的
    latest_file = sorted(files)[-1]
    filepath = os.path.join(results_dir, latest_file)
    
    print(f"📂 加载测试结果: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    return results


def generate_summary_report(results):
    """生成总结报告"""
    print("\n" + "=" * 100)
    print("📊 策略测试总结报告")
    print("=" * 100)
    print(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 创建对比表格
    summary_data = []
    
    for strategy_name, result in results.items():
        if result['predictions']:
            predictions = result['predictions']
            
            # 基本统计
            total = result['total_iterations']
            success = result['successful_iterations']
            errors = result['errors']
            success_rate = success / total * 100
            
            # 预测统计
            buy_count = sum(1 for p in predictions if p['action'] == 1)
            sell_count = sum(1 for p in predictions if p['action'] == 2)
            hold_count = sum(1 for p in predictions if p['action'] == 0)
            
            avg_confidence = sum(p['confidence'] for p in predictions) / len(predictions)
            avg_inference_time = sum(p['inference_time'] for p in predictions) / len(predictions) * 1000  # ms
            
            # 高置信度预测
            high_conf_count = sum(1 for p in predictions if p['confidence'] > 0.7)
            high_conf_rate = high_conf_count / len(predictions) * 100
            
            summary_data.append({
                '策略名称': strategy_name,
                '总测试次数': total,
                '成功次数': success,
                '失败次数': errors,
                '成功率(%)': f"{success_rate:.1f}",
                '买入信号': buy_count,
                '卖出信号': sell_count,
                '持有信号': hold_count,
                '平均置信度': f"{avg_confidence:.3f}",
                '高置信度率(%)': f"{high_conf_rate:.1f}",
                '平均推理时间(ms)': f"{avg_inference_time:.2f}",
            })
    
    # 创建DataFrame并打印
    df_summary = pd.DataFrame(summary_data)
    print("\n" + "─" * 100)
    print(df_summary.to_string(index=False))
    print("─" * 100)
    
    return df_summary


def analyze_strategy_patterns(results):
    """分析策略模式"""
    print("\n" + "=" * 100)
    print("🔍 策略行为模式分析")
    print("=" * 100)
    
    for strategy_name, result in results.items():
        if not result['predictions']:
            continue
        
        predictions = result['predictions']
        
        print(f"\n【{strategy_name}】")
        print("─" * 50)
        
        # 行为一致性分析
        actions = [p['action'] for p in predictions]
        action_names = [p['action_name'] for p in predictions]
        
        if len(set(actions)) == 1:
            print(f"  ⚠️ 策略行为单一: 所有预测都是 '{action_names[0]}'")
        else:
            print(f"  ✅ 策略有多样化预测")
        
        # 置信度分析
        confidences = [p['confidence'] for p in predictions]
        avg_conf = sum(confidences) / len(confidences)
        min_conf = min(confidences)
        max_conf = max(confidences)
        
        print(f"  置信度范围: {min_conf:.3f} - {max_conf:.3f} (平均: {avg_conf:.3f})")
        
        # 推理速度分析
        inference_times = [p['inference_time'] * 1000 for p in predictions]
        avg_time = sum(inference_times) / len(inference_times)
        min_time = min(inference_times)
        max_time = max(inference_times)
        
        print(f"  推理时间范围: {min_time:.2f}ms - {max_time:.2f}ms (平均: {avg_time:.2f}ms)")
        
        # 市场条件分析
        prices = [p['price'] for p in predictions]
        atrs = [p['atr'] for p in predictions]
        
        print(f"  价格范围: {min(prices):.3f} - {max(prices):.3f}")
        print(f"  ATR范围: {min(atrs):.3f} - {max(atrs):.3f}")


def rank_strategies(results):
    """策略排名"""
    print("\n" + "=" * 100)
    print("🏆 策略综合排名")
    print("=" * 100)
    
    rankings = []
    
    for strategy_name, result in results.items():
        if not result['predictions']:
            continue
        
        predictions = result['predictions']
        
        # 计算综合得分
        # 考虑因素: 成功率、置信度、多样性、速度
        
        success_rate = result['successful_iterations'] / result['total_iterations']
        avg_confidence = sum(p['confidence'] for p in predictions) / len(predictions)
        
        # 行为多样性（熵）
        actions = [p['action'] for p in predictions]
        action_counts = {0: 0, 1: 0, 2: 0}
        for a in actions:
            action_counts[a] += 1
        
        diversity = 0
        for count in action_counts.values():
            if count > 0:
                p = count / len(actions)
                diversity -= p * (p if p == 0 else __import__('math').log2(p))
        
        # 速度得分（越快越好）
        avg_time = sum(p['inference_time'] for p in predictions) / len(predictions)
        speed_score = 1 / (avg_time * 1000 + 1)  # 归一化
        
        # 综合得分 (权重可调整)
        score = (
            success_rate * 0.3 +         # 30% 成功率
            avg_confidence * 0.3 +       # 30% 置信度
            diversity * 0.2 +            # 20% 多样性
            speed_score * 0.2            # 20% 速度
        )
        
        rankings.append({
            '策略': strategy_name,
            '综合得分': f"{score:.3f}",
            '成功率': f"{success_rate:.2f}",
            '平均置信度': f"{avg_confidence:.3f}",
            '行为多样性': f"{diversity:.3f}",
            '速度得分': f"{speed_score:.3f}",
        })
    
    # 按得分排序
    df_rankings = pd.DataFrame(rankings)
    df_rankings = df_rankings.sort_values('综合得分', ascending=False)
    
    print("\n" + "─" * 100)
    print(df_rankings.to_string(index=False))
    print("─" * 100)
    
    return df_rankings


def generate_recommendations(results):
    """生成建议"""
    print("\n" + "=" * 100)
    print("💡 优化建议")
    print("=" * 100)
    
    recommendations = []
    
    for strategy_name, result in results.items():
        if not result['predictions']:
            continue
        
        predictions = result['predictions']
        
        # 分析并给出建议
        actions = [p['action'] for p in predictions]
        confidences = [p['confidence'] for p in predictions]
        avg_confidence = sum(confidences) / len(confidences)
        
        # 检查行为单一性
        if len(set(actions)) == 1:
            recommendations.append(f"⚠️ {strategy_name}: 预测过于保守，建议调整决策阈值")
        
        # 检查低置信度
        if avg_confidence < 0.5:
            recommendations.append(f"⚠️ {strategy_name}: 平均置信度较低({avg_confidence:.3f})，建议增加训练数据")
        
        # 检查高置信度但无多样性
        if avg_confidence > 0.9 and len(set(actions)) == 1:
            recommendations.append(f"⚠️ {strategy_name}: 高置信度但缺乏多样性，可能过拟合")
        
        # 检查成功率
        success_rate = result['successful_iterations'] / result['total_iterations']
        if success_rate < 0.7:
            recommendations.append(f"⚠️ {strategy_name}: 测试成功率较低({success_rate*100:.1f}%)，数据获取存在问题")
    
    # 通用建议
    recommendations.append("\n📌 通用优化建议:")
    recommendations.append("  1. 收集更多历史数据用于训练（建议至少30天以上）")
    recommendations.append("  2. 在真实市场环境中测试，而非demo模式")
    recommendations.append("  3. 调整特征工程，添加更多技术指标")
    recommendations.append("  4. 使用交叉验证评估模型性能")
    recommendations.append("  5. 实现回测系统验证策略收益")
    
    for rec in recommendations:
        print(rec)


def save_report(df_summary, df_rankings):
    """保存报告"""
    try:
        # 创建报告目录
        report_dir = '/home/cx/trading_data/reports'
        os.makedirs(report_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存总结
        summary_file = os.path.join(report_dir, f'summary_{timestamp}.csv')
        df_summary.to_csv(summary_file, index=False, encoding='utf-8')
        
        # 保存排名
        ranking_file = os.path.join(report_dir, f'rankings_{timestamp}.csv')
        df_rankings.to_csv(ranking_file, index=False, encoding='utf-8')
        
        print(f"\n✅ 报告已保存:")
        print(f"  - 总结: {summary_file}")
        print(f"  - 排名: {ranking_file}")
        
    except Exception as e:
        print(f"\n⚠️ 保存报告失败: {e}")


def main():
    """主函数"""
    print("\n" + "=" * 100)
    print("📊 策略测试报告生成器")
    print("=" * 100)
    
    # 加载测试结果
    results = load_test_results()
    
    if not results:
        return
    
    # 生成各种分析
    df_summary = generate_summary_report(results)
    analyze_strategy_patterns(results)
    df_rankings = rank_strategies(results)
    generate_recommendations(results)
    
    # 保存报告
    save_report(df_summary, df_rankings)
    
    print("\n" + "=" * 100)
    print("✅ 报告生成完成！")
    print("=" * 100)


if __name__ == "__main__":
    main()
