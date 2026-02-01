# pytest和coverage使用指南

## 前置要求

### 安装依赖

确保已安装pytest和pytest-cov：

```bash
pip install pytest pytest-cov coverage
```

## 快速开始

### 运行所有测试（带覆盖率）

```bash
# 方法1：使用脚本（推荐）
bash scripts/run_pytest_tests.sh

# 方法2：直接运行
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ --cov=src --cov-report=html
```

### 运行特定测试文件

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_account_传递_端到端.py -v
```

### 运行特定测试用例

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_account_传递_端到端.py::TestAccount传递端到端::test_account_从配置传递到下单 -v
```

## 重要说明

### 为什么需要`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`？

系统中安装了ROS相关的pytest插件，这些插件会干扰pytest对unittest.TestCase的识别和收集。使用`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`环境变量可以完全禁用ROS插件的自动加载，确保测试能够正常运行。

### 如果不使用环境变量会怎样？

如果不使用`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`，pytest可能无法收集或运行测试，会显示：
- `collected 0 items / 1 skipped`
- `ERROR: found no collectors`
- `PluginValidationError`

## 常用命令

### 收集测试（不运行）

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ --collect-only
```

### 运行测试（详细输出）

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -v
```

### 运行测试（快速模式）

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -q
```

### 运行测试（遇到第一个失败就停止）

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -x
```

### 生成覆盖率报告

**推荐方法**：使用`coverage run`包装pytest（避免ROS插件问题）：

```bash
# 运行测试并收集覆盖率数据
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m coverage run --source=src -m pytest tests/ -v

# 生成终端报告
python -m coverage report --show-missing

# 生成HTML报告
python -m coverage html
# 然后打开 htmlcov/index.html 查看详细报告

# 或使用脚本（推荐）
bash scripts/run_tests_with_coverage_v2.sh
```

**备选方法**：使用pytest-cov插件（可能遇到ROS插件冲突）：

```bash
# 终端报告
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ --cov=src --cov-report=term

# HTML报告
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ --cov=src --cov-report=html
```

### 查看覆盖率报告

```bash
# 查看终端报告
python -m coverage report

# 查看HTML报告
# 打开 htmlcov/index.html
```

## 配置文件

### pytest.ini

项目根目录下的`pytest.ini`文件包含了pytest的配置：

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
```

### coverage配置

覆盖率配置也在`pytest.ini`中：

```ini
[coverage:run]
source = src
omit = 
    */tests/*
    */test_*.py
    ...
```

## 测试脚本

项目提供了两个便捷脚本：

1. **scripts/run_pytest_tests.sh** - 运行pytest测试并生成覆盖率报告
2. **scripts/run_tests_with_coverage.sh** - 运行测试并生成覆盖率报告（已更新）

## 故障排除

### 问题：pytest无法收集测试

**解决方案**：确保使用`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`环境变量：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -v
```

### 问题：测试文件在导入时执行了副作用代码

**解决方案**：确保测试文件中的API初始化等副作用代码在`setUpClass`或`setUp`方法中，而不是在模块级别。

### 问题：覆盖率报告显示"No data to report"

**解决方案**：确保使用`--cov=src`选项，并且测试实际运行了代码：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ --cov=src --cov-report=term
```

## 最佳实践

1. **始终使用环境变量**：运行pytest时始终使用`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`
2. **使用脚本**：优先使用提供的测试脚本，而不是直接运行pytest命令
3. **检查覆盖率**：定期检查代码覆盖率，确保测试覆盖了关键功能
4. **修复失败的测试**：及时修复失败的测试，保持测试套件的健康
5. **避免模块级别副作用**：不要在测试文件的模块级别执行副作用代码

## 相关文档

- [pytest问题解决总结.md](./pytest问题解决总结.md) - 详细的问题分析和解决方案
- [pytest和coverage使用说明.md](./pytest和coverage使用说明.md) - 之前的说明文档
