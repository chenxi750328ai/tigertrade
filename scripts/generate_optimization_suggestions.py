#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成优化建议
基于测试数据和稳定性测试结果生成优化建议
"""

import json
import os
from datetime import datetime

def generate_optimization_suggestions():
    """生成优化建议"""
    print("="*70)
    print("💡 生成优化建议")
    print("="*70)
    
    suggestions = {
        'generation_time': datetime.now().isoformat(),
        'suggestions': []
    }
    
    # 读取测试分析结果
    test_analysis = {}
    if os.path.exists('test_analysis.json'):
        with open('test_analysis.json', 'r') as f:
            test_analysis = json.load(f)
    
    # 读取稳定性分析结果
    stability_analysis = {}
    if os.path.exists('stability_analysis.json'):
        with open('stability_analysis.json', 'r') as f:
            stability_analysis = json.load(f)
    
    # 基于测试结果生成建议
    if test_analysis.get('test_results', {}).get('success_rate', 100) < 90:
        suggestions['suggestions'].append({
            'category': '测试质量',
            'priority': 'high',
            'title': '提升测试通过率',
            'description': f"当前测试通过率: {test_analysis.get('test_results', {}).get('success_rate', 0):.2f}%",
            'actions': [
                '修复失败的测试用例',
                '检查测试环境配置',
                '更新过时的测试用例'
            ]
        })
    
    # 基于覆盖率生成建议
    low_coverage_modules = test_analysis.get('issues', [{}])[0].get('modules', []) if test_analysis.get('issues') else []
    if low_coverage_modules:
        suggestions['suggestions'].append({
            'category': '代码覆盖率',
            'priority': 'medium',
            'title': '提升代码覆盖率',
            'description': f"发现{len(low_coverage_modules)}个低覆盖率模块",
            'actions': [
                f"优先提升以下模块的覆盖率: {', '.join([m['file'] for m in low_coverage_modules[:5]])}",
                '补充边界条件测试',
                '增加异常处理测试'
            ]
        })
    
    # 基于稳定性测试生成建议
    if stability_analysis.get('recommendations'):
        for rec in stability_analysis['recommendations']:
            suggestions['suggestions'].append({
                'category': '系统稳定性',
                'priority': rec.get('priority', 'medium'),
                'title': rec.get('issue', '稳定性问题'),
                'description': rec.get('suggestion', ''),
                'actions': [
                    '检查相关日志',
                    '分析根本原因',
                    '实施修复方案'
                ]
            })
    
    # 算法优化建议
    suggestions['suggestions'].append({
        'category': '算法优化',
        'priority': 'low',
        'title': '持续优化交易算法',
        'description': '基于实际交易数据优化策略参数',
        'actions': [
            '分析历史交易数据',
            '优化策略参数',
            'A/B测试不同策略配置',
            '监控策略表现指标'
        ]
    })
    
    # 保存建议
    with open('optimization_suggestions.md', 'w') as f:
        f.write("# 优化建议\n\n")
        f.write(f"生成时间: {suggestions['generation_time']}\n\n")
        
        for i, suggestion in enumerate(suggestions['suggestions'], 1):
            f.write(f"## {i}. {suggestion['title']}\n\n")
            f.write(f"**类别**: {suggestion['category']}\n\n")
            f.write(f"**优先级**: {suggestion['priority']}\n\n")
            f.write(f"**描述**: {suggestion['description']}\n\n")
            f.write("**建议行动**:\n")
            for action in suggestion['actions']:
                f.write(f"- {action}\n")
            f.write("\n")
    
    # 保存JSON格式
    with open('optimization_suggestions.json', 'w') as f:
        json.dump(suggestions, f, indent=2)
    
    print(f"\n✅ 生成了{len(suggestions['suggestions'])}条优化建议")
    print("📄 建议已保存到 optimization_suggestions.md")
    print("="*70)
    
    return suggestions


if __name__ == '__main__':
    generate_optimization_suggestions()
