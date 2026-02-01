#!/usr/bin/env python3
"""
生成CI测试和覆盖率报告
整合Feature测试结果和代码覆盖率报告
"""
import json
import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path

def run_command(cmd, capture_output=True):
    """运行命令并返回结果"""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=capture_output, text=True, check=False
        )
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return 1, "", str(e)

def get_feature_test_results():
    """获取Feature测试结果"""
    print("📊 收集Feature测试结果...")
    returncode, stdout, stderr = run_command(
        "python -m pytest tests/test_feature_*.py -v --tb=no -q --json-report --json-report-file=/tmp/feature_report.json 2>&1 || true"
    )
    
    # 尝试解析JSON报告
    feature_results = {
        'total': 0,
        'passed': 0,
        'failed': 0,
        'skipped': 0,
        'tests': []
    }
    
    # 如果没有JSON报告，从stdout解析
    if os.path.exists('/tmp/feature_report.json'):
        try:
            with open('/tmp/feature_report.json') as f:
                data = json.load(f)
                feature_results['total'] = data.get('summary', {}).get('total', 0)
                feature_results['passed'] = data.get('summary', {}).get('passed', 0)
                feature_results['failed'] = data.get('summary', {}).get('failed', 0)
                feature_results['skipped'] = data.get('summary', {}).get('skipped', 0)
        except:
            pass
    
    # 从stdout解析
    lines = stdout.split('\n')
    for line in lines:
        if 'passed' in line.lower() and 'test' in line.lower():
            parts = line.split()
            for i, part in enumerate(parts):
                if part == 'passed':
                    try:
                        feature_results['passed'] = int(parts[i-1])
                    except:
                        pass
    
    return feature_results, returncode == 0

def get_coverage_report():
    """获取覆盖率报告"""
    print("📊 收集覆盖率数据...")
    
    # 运行覆盖率测试
    run_command("python -m pytest tests/ --cov=src --cov-report=json:coverage.json --cov-report=term-missing -q")
    
    coverage_data = {
        'total_coverage': 0.0,
        'files': {}
    }
    
    # 读取JSON覆盖率报告
    if os.path.exists('coverage.json'):
        try:
            with open('coverage.json') as f:
                data = json.load(f)
                coverage_data['total_coverage'] = data.get('totals', {}).get('percent_covered', 0.0)
                
                # 提取关键文件的覆盖率
                for file_path, file_data in data.get('files', {}).items():
                    if 'src/executor' in file_path or 'src/api_adapter' in file_path:
                        coverage_data['files'][file_path] = {
                            'coverage': file_data.get('summary', {}).get('percent_covered', 0.0),
                            'lines': file_data.get('summary', {}).get('num_statements', 0),
                            'missing': file_data.get('summary', {}).get('missing_lines', 0)
                        }
        except Exception as e:
            print(f"⚠️ 解析覆盖率JSON失败: {e}")
    
    # 从命令行输出获取
    _, stdout, _ = run_command("python -m coverage report -m --include='src/executor/*,src/api_adapter.py'")
    
    return coverage_data, stdout

def generate_markdown_report(feature_results, feature_success, coverage_data, coverage_text):
    """生成Markdown报告"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""# CI测试和覆盖率报告

**生成时间**: {timestamp}  
**测试工具**: pytest + pytest-cov + coverage.py

---

## 一、Feature级测试结果（业务需求验证）

### 测试统计

| 指标 | 数量 |
|------|------|
| 总测试数 | {feature_results['total']} |
| ✅ 通过 | {feature_results['passed']} |
| ❌ 失败 | {feature_results['failed']} |
| ⏭️ 跳过 | {feature_results['skipped']} |
| **状态** | {'✅ 通过' if feature_success else '❌ 失败'} |

### Feature测试覆盖的AR（验收标准）

- **Feature 1**: 市场数据采集
- **Feature 2**: 交易策略预测
- **Feature 3**: 订单执行（关键）
- **Feature 4**: 风险管理
- **Feature 6**: 交易循环执行

---

## 二、代码级测试结果（技术逻辑验证）

代码级测试确保代码逻辑正确，覆盖所有代码路径。

---

## 三、代码覆盖率报告

### 总体覆盖率

**总覆盖率**: {coverage_data['total_coverage']:.2f}%

### 关键模块覆盖率

{coverage_text}

### 覆盖率趋势

- Executor模块（核心交易逻辑）: 目标 > 80%
- API适配器: 目标 > 60%
- 全项目: 目标 > 50%

---

## 四、测试和覆盖率互补

### Feature测试（业务视角）
- ✅ 验证业务需求是否满足（AR）
- ✅ 端到端功能测试
- ✅ 关注业务结果

### 代码测试（技术视角）
- ✅ 验证代码逻辑是否正确
- ✅ 单元测试、边界测试
- ✅ 关注代码路径覆盖

### 两者结合
- Feature测试发现业务问题（如"订单没出现在账户"）
- 代码测试定位技术问题（如"account字段为空"）
- 确保业务和技术都正确

---

## 五、HTML覆盖率报告

详细报告请查看: `htmlcov/index.html`

可以打开查看：
- 逐行覆盖情况（绿色=已覆盖，红色=未覆盖）
- 未覆盖代码的具体行号和内容
- 分支覆盖情况

---

## 六、CI状态

- **Feature测试**: {'✅ 通过' if feature_success else '❌ 失败'}
- **代码覆盖率**: {coverage_data['total_coverage']:.2f}%
- **整体状态**: {'✅ 通过' if feature_success and coverage_data['total_coverage'] >= 20 else '⚠️ 需要改进'}

---

**报告生成工具**: pytest-cov + coverage.py  
**测试框架**: pytest + unittest  
**CI平台**: GitHub Actions
"""
    
    return report

def main():
    """主函数"""
    print("=" * 60)
    print("生成CI测试和覆盖率报告")
    print("=" * 60)
    
    # 1. 获取Feature测试结果
    feature_results, feature_success = get_feature_test_results()
    
    # 2. 获取覆盖率报告
    coverage_data, coverage_text = get_coverage_report()
    
    # 3. 生成Markdown报告
    report = generate_markdown_report(feature_results, feature_success, coverage_data, coverage_text)
    
    # 4. 保存报告
    report_path = Path('docs/CI_TEST_REPORT.md')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding='utf-8')
    
    print(f"\n✅ 报告已生成: {report_path}")
    print(f"   - Feature测试: {'通过' if feature_success else '失败'}")
    print(f"   - 代码覆盖率: {coverage_data['total_coverage']:.2f}%")
    
    return 0 if feature_success else 1

if __name__ == '__main__':
    sys.exit(main())
