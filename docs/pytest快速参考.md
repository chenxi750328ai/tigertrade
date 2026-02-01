# pytest快速参考指南

## 🚀 快速开始

### 运行所有测试

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -v
```

### 运行测试并收集覆盖率

```bash
# 方法1：使用脚本（推荐）
bash scripts/run_tests_with_coverage_v2.sh

# 方法2：手动运行
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m coverage run --source=src -m pytest tests/ -v
python -m coverage report --show-missing
python -m coverage html
```

## 📝 常用命令

### 运行特定测试文件

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_account_传递_端到端.py -v
```

### 运行特定测试用例

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_account_传递_端到端.py::TestAccount传递端到端::test_account_从配置传递到下单 -v
```

### 运行多个测试文件

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_account_传递_端到端.py tests/test_feature_buy_silver_comprehensive.py -v
```

### 快速模式（不显示详细信息）

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -q
```

### 遇到第一个失败就停止

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -x
```

### 只收集测试（不运行）

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ --collect-only
```

## 📊 覆盖率相关

### 收集覆盖率数据

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m coverage run --source=src -m pytest tests/ -v
```

### 查看覆盖率报告

```bash
# 终端报告
python -m coverage report --show-missing

# HTML报告
python -m coverage html
# 然后打开 htmlcov/index.html
```

### 查看特定模块的覆盖率

```bash
python -m coverage report --include="src/api_adapter.py" --show-missing
```

## ⚠️ 重要提示

1. **必须使用环境变量**：运行pytest时务必使用`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`，否则ROS插件会干扰测试
2. **覆盖率收集**：使用`coverage run`包装pytest，而不是`--cov`选项
3. **使用脚本**：推荐使用提供的脚本，它们已经配置好了正确的选项

## 🔧 故障排除

### 问题：pytest无法收集测试

**解决**：确保使用了`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`

### 问题：覆盖率报告为空

**解决**：使用`coverage run`包装pytest，而不是`--cov`选项

### 问题：测试运行很慢

**解决**：使用`-q`选项减少输出，或只运行特定测试文件

## 📚 更多信息

- [pytest使用指南.md](./pytest使用指南.md) - 完整的使用指南
- [pytest问题解决总结.md](./pytest问题解决总结.md) - 问题分析和解决方案
- [pytest和coverage最终解决方案.md](./pytest和coverage最终解决方案.md) - 最终解决方案
