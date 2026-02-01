#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析测试数据，识别问题和改进点
"""

import json
import xml.etree.ElementTree as ET
import os
from datetime import datetime

def analyze_test_results():
    """分析测试结果"""
    print("="*70)
    print("📊 测试数据分析")
    print("="*70)
    
    analysis = {
        'analysis_time': datetime.now().isoformat(),
        'test_results': {},
        'coverage_analysis': {},
        'issues': [],
        'recommendations': []
    }
    
    # 分析JUnit XML结果
    if os.path.exists('test-results.xml'):
        try:
            tree = ET.parse('test-results.xml')
            root = tree.getroot()
            
            total_tests = int(root.get('tests', 0))
            failures = int(root.get('failures', 0))
            errors = int(root.get('errors', 0))
            
            analysis['test_results'] = {
                'total': total_tests,
                'failures': failures,
                'errors': errors,
                'success_rate': (total_tests - failures - errors) / total_tests * 100 if total_tests > 0 else 0
            }
            
            # 分析失败的测试
            failed_tests = []
            for testcase in root.findall('.//testcase'):
                failure = testcase.find('failure')
                error = testcase.find('error')
                if failure is not None or error is not None:
                    failed_tests.append({
                        'name': testcase.get('name'),
                        'classname': testcase.get('classname'),
                        'type': 'failure' if failure is not None else 'error',
                        'message': (failure or error).get('message', '')
                    })
            
            analysis['test_results']['failed_tests'] = failed_tests
            
        except Exception as e:
            print(f"⚠️ 解析测试结果失败: {e}")
    
    # 分析覆盖率数据
    if os.path.exists('.coverage'):
        try:
            import coverage
            cov = coverage.Coverage()
            cov.load()
            
            # 获取覆盖率报告
            report_data = {}
            for file_path in cov.get_data().measured_files():
                if 'src/' in file_path:
                    rel_path = file_path.split('src/')[-1]
                    analysis_data = cov.analysis(file_path)
                    report_data[rel_path] = {
                        'statements': analysis_data[1],
                        'missing': analysis_data[2],
                        'coverage': (analysis_data[1] - len(analysis_data[2])) / analysis_data[1] * 100 if analysis_data[1] > 0 else 0
                    }
            
            analysis['coverage_analysis'] = report_data
            
            # 识别低覆盖率模块
            low_coverage = []
            for file_path, data in report_data.items():
                if data['coverage'] < 50:
                    low_coverage.append({
                        'file': file_path,
                        'coverage': data['coverage'],
                        'missing_lines': len(data['missing'])
                    })
            
            analysis['issues'].append({
                'type': 'low_coverage',
                'modules': low_coverage
            })
            
        except Exception as e:
            print(f"⚠️ 分析覆盖率失败: {e}")
    
    # 生成建议
    if analysis['test_results'].get('success_rate', 100) < 80:
        analysis['recommendations'].append({
            'priority': 'high',
            'issue': '测试通过率过低',
            'suggestion': '需要修复失败的测试用例'
        })
    
    if analysis['test_results'].get('errors', 0) > 0:
        analysis['recommendations'].append({
            'priority': 'high',
            'issue': '存在测试错误',
            'suggestion': '检查测试环境配置和依赖项'
        })
    
    # 保存分析结果
    with open('test_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print("\n✅ 分析完成，结果已保存到 test_analysis.json")
    print("="*70)
    
    return analysis


if __name__ == '__main__':
    analyze_test_results()
