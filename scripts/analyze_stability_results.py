#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析稳定性测试结果
"""

import json
import sys
from datetime import datetime
from collections import defaultdict

def analyze_stability_results(log_file='stability_test.log', stats_file='stability_stats.json'):
    """分析稳定性测试结果"""
    
    print("="*70)
    print("📊 稳定性测试结果分析")
    print("="*70)
    
    # 读取统计信息
    try:
        with open(stats_file, 'r') as f:
            stats = json.load(f)
    except FileNotFoundError:
        print(f"❌ 统计文件不存在: {stats_file}")
        return
    
    # 基本统计
    print("\n📈 基本统计:")
    print(f"  开始时间: {stats.get('start_time', 'N/A')}")
    print(f"  结束时间: {stats.get('end_time', 'N/A')}")
    print(f"  总迭代数: {stats.get('iterations', 0)}")
    print(f"  成功迭代: {stats.get('successful_iterations', 0)}")
    print(f"  失败迭代: {stats.get('failed_iterations', 0)}")
    
    if stats.get('iterations', 0) > 0:
        success_rate = stats.get('successful_iterations', 0) / stats.get('iterations', 1) * 100
        print(f"  成功率: {success_rate:.2f}%")
    else:
        success_rate = 0
    
    # 错误分析
    errors = stats.get('errors', [])
    print(f"\n❌ 错误统计:")
    print(f"  总错误数: {len(errors)}")
    
    if errors:
        error_types = defaultdict(int)
        for error in errors:
            error_msg = error.get('error', 'Unknown')
            error_type = error_msg.split(':')[0] if ':' in error_msg else error_msg
            error_types[error_type] += 1
        
        print(f"  错误类型分布:")
        for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            print(f"    {error_type}: {count}次")
        
        # 最近10个错误
        print(f"\n  最近10个错误:")
        for error in errors[-10:]:
            print(f"    [{error.get('time', 'N/A')}] {error.get('error', 'Unknown')}")
    
    # 性能指标
    memory_usage = stats.get('memory_usage', [])
    cpu_usage = stats.get('cpu_usage', [])
    
    if memory_usage:
        memory_values = [m['memory_mb'] for m in memory_usage]
        print(f"\n💾 内存使用:")
        print(f"  平均: {sum(memory_values) / len(memory_values):.2f}MB")
        print(f"  最大: {max(memory_values):.2f}MB")
        print(f"  最小: {min(memory_values):.2f}MB")
    
    if cpu_usage:
        cpu_values = [c['cpu_percent'] for c in cpu_usage]
        print(f"\n⚡ CPU使用:")
        print(f"  平均: {sum(cpu_values) / len(cpu_values):.2f}%")
        print(f"  最大: {max(cpu_values):.2f}%")
        print(f"  最小: {min(cpu_values):.2f}%")
    
    # API调用统计
    print(f"\n📡 API调用统计:")
    print(f"  总调用数: {stats.get('api_calls', 0)}")
    print(f"  错误数: {stats.get('api_errors', 0)}")
    if stats.get('api_calls', 0) > 0:
        error_rate = stats.get('api_errors', 0) / stats.get('api_calls', 1) * 100
        print(f"  错误率: {error_rate:.2f}%")
    
    # 订单统计
    print(f"\n📦 订单统计:")
    print(f"  下单成功: {stats.get('orders_placed', 0)}")
    print(f"  下单失败: {stats.get('orders_failed', 0)}")
    
    # 生成报告
    report = {
        'analysis_time': datetime.now().isoformat(),
        'summary': {
            'total_iterations': stats.get('iterations', 0),
            'success_rate': success_rate,
            'total_errors': len(errors),
            'error_types': dict(error_types) if errors else {},
        },
        'performance': {
            'avg_memory_mb': sum(memory_values) / len(memory_values) if memory_usage else 0,
            'max_memory_mb': max(memory_values) if memory_usage else 0,
            'avg_cpu_percent': sum(cpu_values) / len(cpu_values) if cpu_usage else 0,
        },
        'recommendations': generate_recommendations(stats)
    }
    
    # 保存分析结果
    with open('stability_analysis.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n✅ 分析完成，结果已保存到 stability_analysis.json")
    print("="*70)


def generate_recommendations(stats):
    """生成优化建议"""
    recommendations = []
    
    errors = stats.get('errors', [])
    if len(errors) > 50:
        recommendations.append({
            'priority': 'high',
            'issue': '错误过多',
            'suggestion': '需要检查错误日志，修复主要错误源'
        })
    
    if stats.get('failed_iterations', 0) > stats.get('successful_iterations', 0):
        recommendations.append({
            'priority': 'high',
            'issue': '失败迭代数超过成功迭代数',
            'suggestion': '需要检查策略逻辑和API连接稳定性'
        })
    
    memory_usage = stats.get('memory_usage', [])
    if memory_usage:
        memory_values = [m['memory_mb'] for m in memory_usage]
        if max(memory_values) > 1000:  # 超过1GB
            recommendations.append({
                'priority': 'medium',
                'issue': '内存使用过高',
                'suggestion': '检查内存泄漏，优化数据缓存策略'
            })
    
    cpu_usage = stats.get('cpu_usage', [])
    if cpu_usage:
        cpu_values = [c['cpu_percent'] for c in cpu_usage]
        if max(cpu_values) > 80:
            recommendations.append({
                'priority': 'medium',
                'issue': 'CPU使用率过高',
                'suggestion': '优化算法性能，考虑异步处理'
            })
    
    if not recommendations:
        recommendations.append({
            'priority': 'low',
            'issue': '无重大问题',
            'suggestion': '系统运行稳定，继续保持'
        })
    
    return recommendations


if __name__ == '__main__':
    log_file = sys.argv[1] if len(sys.argv) > 1 else 'stability_test.log'
    analyze_stability_results(log_file)
