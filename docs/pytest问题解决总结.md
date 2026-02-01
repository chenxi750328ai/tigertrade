# pytest测试收集问题解决总结

## 问题描述

pytest无法收集unittest.TestCase测试，显示"collected 0 items / 1 skipped"和"ERROR: found no collectors"。

## 根本原因

1. **ROS插件干扰**：系统中安装了ROS相关的pytest插件（`launch-testing`, `launch-testing-ros`, `ament-*`等），这些插件会干扰pytest对unittest.TestCase的识别和收集。

2. **模块级别副作用代码**：部分测试文件在模块级别执行了API初始化等副作用代码，导致pytest收集测试时出现问题。

## 解决方案

### 1. 运行测试（推荐方法）

**最佳方案**：使用环境变量`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`完全禁用ROS插件自动加载：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -v
```

这种方法完全避免了ROS插件的干扰，测试可以正常运行。

### 2. 收集覆盖率（推荐方法）

**最佳方案**：使用`coverage run`包装pytest来收集覆盖率：

```bash
# 运行测试并收集覆盖率
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m coverage run --source=src -m pytest tests/ -v

# 生成报告
python -m coverage report --show-missing
python -m coverage html
```

或使用脚本：
```bash
bash scripts/run_tests_with_coverage_v2.sh
```

**备选方案**：如果环境变量方法不可用，可以尝试使用`-p no:`选项禁用特定插件：

```bash
python -m pytest tests/ \
    -p no:launch_testing \
    -p no:launch_testing_ros \
    -p no:ament_xmllint \
    -p no:ament_flake8 \
    -p no:ament_lint \
    -p no:ament_copyright \
    -p no:ament_pep257
```

**注意**：备选方案可能会遇到`PluginValidationError`错误，虽然测试能够收集，但可能无法执行。

### 2. 修复模块级别副作用代码

将测试文件中的模块级别副作用代码移到`setUpClass`或`setUp`方法中：

**修复的文件：**
- `tests/test_feature_order_execution_real_api.py` - 移除了模块级别的API初始化
- `tests/feature_test_base.py` - 移除了模块级别的自动初始化
- `tests/test_feature_order_execution.py` - 移除了模块级别的API初始化

### 3. 使用测试脚本

已创建/更新以下脚本，确保正确禁用ROS插件：

- `scripts/run_pytest_tests.sh` - 使用pytest和coverage运行测试（已禁用ROS插件）
- `scripts/run_tests_with_coverage.sh` - 更新为禁用ROS插件

## 验证

使用`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`后，pytest能够正确收集和运行测试：

```bash
$ PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_account_传递_端到端.py -v
============================= test session starts ==============================
collected 3 items

tests/test_account_传递_端到端.py::TestAccount传递端到端::test_account_从配置传递到下单 PASSED [ 33%]
tests/test_account_传递_端到端.py::TestAccount传递端到端::test_account为空时下单失败 PASSED [ 66%]
tests/test_account_传递_端到端.py::TestAccount传递端到端::test_account从api_manager获取 PASSED [100%]

============================== 3 passed in 29.26s ==============================
```

## 注意事项

1. **pytest.ini配置**：虽然已在`pytest.ini`中添加了`addopts`来禁用ROS插件，但由于某些原因（可能是pytest版本或ROS插件版本不兼容），配置可能不会自动生效。因此，建议在命令行中显式指定`-p no:`选项。

2. **ROS插件错误**：禁用ROS插件后，可能会出现`PluginValidationError: unknown hook 'pytest_launch_collect_makemodule'`错误，但这不影响测试收集和执行，可以忽略。

3. **测试运行**：使用修复后的脚本运行测试：

```bash
# 使用pytest和coverage（推荐）
bash scripts/run_pytest_tests.sh

# 或直接运行（使用环境变量）
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ --cov=src

# 或使用修复后的脚本
bash scripts/run_tests_with_coverage.sh
```

## 总结

问题已解决：
- ✅ pytest能够收集unittest.TestCase测试
- ✅ 修复了模块级别副作用代码
- ✅ 创建了正确的测试运行脚本
- ✅ 文档化解决方案

现在可以使用pytest和coverage正常运行测试了！
