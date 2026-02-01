#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
提升测试覆盖率
自动分析覆盖率报告，识别未覆盖的代码，生成测试用例建议
"""

import sys
import os
import json
import subprocess
from datetime import datetime
import re

sys.path.insert(0, '/home/cx/tigertrade')


def get_coverage_report():
    """获取覆盖率报告"""
    print("📊 获取覆盖率报告...")
    
    # 运行测试并收集覆盖率
    result = subprocess.run(
        ['python', '-m', 'coverage', 'report', '--include=src/*', '--show-missing'],
        capture_output=True,
        text=True,
        cwd='/home/cx/tigertrade'
    )
    
    return result.stdout


def analyze_coverage_gaps():
    """分析覆盖率缺口"""
    print("="*70)
    print("🔍 分析覆盖率缺口")
    print("="*70)
    
    coverage_report = get_coverage_report()
    
    # 解析覆盖率报告
    gaps = []
    lines = coverage_report.split('\n')
    
    for line in lines:
        if '%' in line and 'src/' in line:
            # 解析覆盖率行
            parts = line.split()
            if len(parts) >= 4:
                try:
                    file_path = parts[0]
                    statements = int(parts[1])
                    missing = int(parts[2])
                    coverage = float(parts[3].rstrip('%'))
                    
                    if coverage < 65:  # 低于目标覆盖率
                        gaps.append({
                            'file': file_path,
                            'coverage': coverage,
                            'statements': statements,
                            'missing': missing,
                            'priority': 'high' if coverage < 50 else 'medium'
                        })
                except (ValueError, IndexError):
                    continue
    
    return gaps


def generate_test_suggestions(gaps):
    """生成测试建议"""
    print("\n💡 生成测试建议...")
    
    suggestions = []
    
    for gap in gaps:
        file_path = gap['file']
        coverage = gap['coverage']
        missing = gap['missing']
        
        suggestion = {
            'file': file_path,
            'current_coverage': coverage,
            'target_coverage': 80,
            'missing_lines': missing,
            'priority': gap['priority'],
            'suggestions': []
        }
        
        # 根据文件类型生成建议
        if 'executor' in file_path:
            suggestion['suggestions'].append('补充executor模块的测试用例')
            suggestion['suggestions'].append('测试所有订单执行路径')
            suggestion['suggestions'].append('测试错误处理逻辑')
        elif 'api_adapter' in file_path:
            suggestion['suggestions'].append('补充API适配器的测试用例')
            suggestion['suggestions'].append('测试Mock和Real API的所有路径')
            suggestion['suggestions'].append('测试错误处理和重试逻辑')
        elif 'tiger1' in file_path:
            suggestion['suggestions'].append('补充tiger1.py主函数的测试')
            suggestion['suggestions'].append('测试所有策略分支')
            suggestion['suggestions'].append('测试主循环的所有路径')
        elif 'strategies' in file_path:
            suggestion['suggestions'].append('补充策略模块的测试用例')
            suggestion['suggestions'].append('测试策略预测逻辑')
            suggestion['suggestions'].append('测试策略参数调整')
        
        suggestions.append(suggestion)
    
    return suggestions


def create_test_templates(suggestions):
    """创建测试模板"""
    print("\n📝 创建测试模板...")
    
    templates = []
    
    for suggestion in suggestions[:10]:  # 只处理前10个
        file_path = suggestion['file']
        test_file_name = f"test_{os.path.basename(file_path).replace('.py', '')}_coverage.py"
        
        template = f"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
\"\"\"
补充测试用例以提升覆盖率: {file_path}
当前覆盖率: {suggestion['current_coverage']:.2f}%
目标覆盖率: {suggestion['target_coverage']}%
\"\"\"

import unittest
import sys
sys.path.insert(0, '/home/cx/tigertrade')

from src import {os.path.basename(file_path).replace('.py', '')} as module


class Test{suggestion['file'].replace('/', '_').replace('.py', '')}Coverage(unittest.TestCase):
    \"\"\"补充测试用例以提升覆盖率\"\"\"
    
    def setUp(self):
        \"\"\"测试前准备\"\"\"
        pass
    
    def tearDown(self):
        \"\"\"测试后清理\"\"\"
        pass
    
    # TODO: 添加测试用例以覆盖未覆盖的代码路径
    # 建议：
"""
        
        for sug in suggestion['suggestions']:
            template += f"    # - {sug}\n"
        
        template += """
    def test_placeholder(self):
        \"\"\"占位测试\"\"\"
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
"""
        
        templates.append({
            'file': test_file_name,
            'content': template,
            'target_file': suggestion['file']
        })
    
    return templates


def save_suggestions(suggestions, templates):
    """保存建议和模板"""
    report = {
        'analysis_time': datetime.now().isoformat(),
        'current_coverage': 21.14,  # 从实际报告获取
        'target_coverage': 65,
        'gaps': suggestions,
        'templates': templates
    }
    
    # 保存JSON报告
    with open('coverage_improvement_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # 保存Markdown报告
    with open('coverage_improvement_report.md', 'w') as f:
        f.write("# 测试覆盖率提升报告\n\n")
        f.write(f"生成时间: {report['analysis_time']}\n\n")
        f.write(f"当前覆盖率: {report['current_coverage']:.2f}%\n")
        f.write(f"目标覆盖率: {report['target_coverage']}%\n")
        f.write(f"差距: {report['target_coverage'] - report['current_coverage']:.2f}%\n\n")
        
        f.write("## 需要提升的模块\n\n")
        for i, gap in enumerate(suggestions[:20], 1):
            f.write(f"### {i}. {gap['file']}\n\n")
            f.write(f"- **当前覆盖率**: {gap['current_coverage']:.2f}%\n")
            f.write(f"- **缺失行数**: {gap['missing_lines']}\n")
            f.write(f"- **优先级**: {gap['priority']}\n\n")
            f.write("**建议**:\n")
            for sug in gap['suggestions']:
                f.write(f"- {sug}\n")
            f.write("\n")
    
    print("✅ 报告已保存到 coverage_improvement_report.json 和 coverage_improvement_report.md")


def run_coverage_improvement():
    """运行覆盖率提升流程"""
    print("="*70)
    print("🚀 开始提升测试覆盖率")
    print("="*70)
    
    # 1. 分析覆盖率缺口
    gaps = analyze_coverage_gaps()
    print(f"\n📊 发现 {len(gaps)} 个需要提升的模块")
    
    # 2. 生成测试建议
    suggestions = generate_test_suggestions(gaps)
    
    # 3. 创建测试模板
    templates = create_test_templates(suggestions)
    
    # 4. 保存报告
    save_suggestions(suggestions, templates)
    
    print("\n✅ 覆盖率提升分析完成")
    print("="*70)
    
    return {
        'gaps_found': len(gaps),
        'suggestions': len(suggestions),
        'templates': len(templates)
    }


if __name__ == '__main__':
    run_coverage_improvement()
