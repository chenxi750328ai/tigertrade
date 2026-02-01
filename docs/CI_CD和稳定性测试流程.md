# CI/CD和稳定性测试流程

## 📋 概述

项目已建立完整的CI/CD和稳定性测试流程，包括：
1. **每日CI/CD测试** - 自动运行测试套件和覆盖率检查
2. **20小时稳定性测试** - 长时间运行验证系统稳定性
3. **问题检测和修复** - 自动分析问题并生成修复建议
4. **数据分析和优化** - 基于测试数据优化算法

## 🔄 每日CI/CD流程

### 触发时机
- **自动触发**: 每天UTC 00:00（北京时间08:00）
- **手动触发**: 通过GitHub Actions界面
- **代码推送**: 推送到main或develop分支时

### 执行步骤

#### 1. 测试套件运行
- 运行所有测试（520+个测试）
- 收集代码覆盖率
- 检查覆盖率阈值
- 生成测试报告

#### 2. 20小时稳定性测试
- 运行交易策略20小时
- 监控错误、性能、资源使用
- 记录所有异常情况
- 生成稳定性报告

#### 3. 数据分析
- 分析测试结果
- 分析稳定性数据
- 识别问题和改进点
- 生成优化建议

## 📊 稳定性测试详情

### 测试配置
- **运行时长**: 20小时
- **策略**: moe_transformer（可配置）
- **检查间隔**: 60秒
- **指标记录**: 每5分钟

### 监控指标
- **错误统计**: 错误类型、频率、堆栈跟踪
- **性能指标**: CPU使用率、内存使用
- **API调用**: 调用次数、错误率
- **订单统计**: 下单成功/失败次数

### 输出文件
- `stability_test.log` - 详细日志
- `stability_stats.json` - 统计信息
- `stability_analysis.json` - 分析结果
- `stability_report.md` - 可读性报告

## 🔍 问题检测机制

### 自动检测的问题类型
1. **测试失败** - 失败的测试用例
2. **覆盖率不足** - 低于阈值的模块
3. **稳定性问题** - 错误过多、内存泄漏等
4. **性能问题** - CPU/内存使用过高

### 修复建议生成
- 自动分析问题根因
- 生成修复建议
- 优先级排序
- 行动清单

## 📈 数据分析流程

### 测试数据分析
- 测试通过率统计
- 失败测试分类
- 覆盖率分析
- 低覆盖率模块识别

### 稳定性数据分析
- 错误趋势分析
- 性能指标趋势
- 资源使用分析
- 稳定性评分

### 优化建议生成
- 基于测试结果
- 基于稳定性数据
- 算法优化建议
- 代码质量改进

## 🚀 使用方法

### 手动运行CI/CD
```bash
# 运行测试
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m coverage run --source=src -m pytest tests/ -v

# 运行稳定性测试
python scripts/stability_test_20h.py

# 分析结果
python scripts/analyze_stability_results.py
python scripts/analyze_test_data.py
python scripts/generate_optimization_suggestions.py
```

### GitHub Actions
CI/CD流程会自动运行，无需手动操作。查看结果：
1. 访问GitHub Actions页面
2. 查看最新的workflow运行结果
3. 下载artifacts查看详细报告

## 📝 报告文件

### 测试报告
- `test-results.xml` - JUnit格式测试结果
- `test_analysis.json` - 测试数据分析
- `htmlcov/` - HTML覆盖率报告

### 稳定性报告
- `stability_test.log` - 详细日志
- `stability_stats.json` - 统计数据
- `stability_analysis.json` - 分析结果

### 优化建议
- `optimization_suggestions.md` - Markdown格式建议
- `optimization_suggestions.json` - JSON格式建议

## 🎯 目标

### 测试目标
- **测试通过率**: > 90%
- **Executor模块覆盖率**: > 80% ✅（当前86.22%）
- **总体覆盖率**: > 65%（当前21.14%，需要提升）

### 稳定性目标
- **错误率**: < 1%
- **内存使用**: < 1GB
- **CPU使用**: < 80%
- **20小时无崩溃**: ✅

## ✅ 当前状态

- ✅ CI/CD流程已配置
- ✅ 稳定性测试脚本已创建
- ✅ 数据分析脚本已创建
- ✅ 优化建议生成已实现
- ⏳ 需要配置GitHub Actions secrets（API配置）

## 📚 相关文件

- `.github/workflows/daily_ci_cd.yml` - CI/CD配置
- `scripts/stability_test_20h.py` - 稳定性测试脚本
- `scripts/analyze_stability_results.py` - 稳定性分析
- `scripts/analyze_test_data.py` - 测试数据分析
- `scripts/generate_optimization_suggestions.py` - 优化建议生成

## 🔧 配置说明

### 环境变量
- `TRADING_STRATEGY` - 策略名称（默认: moe_transformer）
- `RUN_DURATION_HOURS` - 运行时长（默认: 20）
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD` - 禁用ROS插件（必须）

### GitHub Secrets（需要配置）
- `TIGER_API_CONFIG` - Tiger API配置（如果需要真实API测试）

## 📊 监控和告警

### 告警条件
- 测试通过率 < 80%
- 错误数 > 50
- 内存使用 > 1GB
- CPU使用 > 80%

### 通知方式
- GitHub Actions通知
- 邮件通知（可配置）
- Slack通知（可配置）

## 🎯 下一步

1. **配置GitHub Actions** - 设置secrets和通知
2. **修复失败的测试** - 提升测试通过率
3. **提升覆盖率** - 补充测试用例
4. **优化算法** - 基于数据分析结果
