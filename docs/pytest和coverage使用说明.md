# pytest和coverage使用说明

## 问题说明

之前错误地使用了`unittest`来运行测试，但项目配置的是`pytest`和`coverage`。

## 正确的测试运行方式

### 1. 使用pytest运行测试

```bash
# 运行所有测试
cd /home/cx/tigertrade
python -m pytest tests/ -v

# 运行特定测试文件
python -m pytest tests/test_feature_buy_silver_comprehensive.py -v

# 运行特定测试类
python -m pytest tests/test_feature_buy_silver_comprehensive.py::TestFeatureBuySilverMock -v

# 运行特定测试方法
python -m pytest tests/test_feature_buy_silver_comprehensive.py::TestFeatureBuySilverMock::test_buy_silver_account_correct -v
```

### 2. 使用pytest + coverage生成覆盖率报告

```bash
# 运行测试并生成覆盖率报告
cd /home/cx/tigertrade
python -m pytest tests/ \
    -v \
    --cov=src \
    --cov-report=term \
    --cov-report=term-missing \
    --cov-report=html \
    --tb=short

# 查看HTML覆盖率报告
open htmlcov/index.html
# 或
xdg-open htmlcov/index.html
```

### 3. 使用脚本运行（推荐）

```bash
# 使用提供的脚本
cd /home/cx/tigertrade
bash scripts/run_tests_with_coverage.sh
```

## 配置说明

### pytest.ini配置

项目已配置`pytest.ini`：
- `testpaths = tests` - 测试目录
- `python_files = test_*.py` - 测试文件模式
- `python_classes = Test*` - 测试类模式（支持unittest）

### .coveragerc配置

项目已配置`.coveragerc`：
- `source = src` - 源代码目录
- `omit` - 排除的目录和文件
- `show_missing = True` - 显示未覆盖的行

## 当前覆盖率状态

根据最新运行结果：
- **总覆盖率**: 6.67%
- **关键模块覆盖率**:
  - `src/tiger1.py`: 11.56%
  - `src/api_adapter.py`: 25.09%
  - `src/executor/order_executor.py`: 17.97%
  - `src/executor/trading_executor.py`: 13.74%

## 注意事项

1. **pytest可以运行unittest测试**：虽然测试文件使用`unittest.TestCase`，但pytest可以自动识别并运行它们。

2. **测试收集问题**：如果pytest没有收集到测试，检查：
   - 测试文件是否以`test_`开头
   - 测试类是否以`Test`开头
   - 测试方法是否以`test_`开头

3. **覆盖率目标**：目标是100%覆盖率，当前只有6.67%，需要大幅提升。

## 下一步

1. ✅ 修复测试收集问题，确保所有测试都能被pytest识别
2. ⏳ 运行完整测试套件，生成覆盖率报告
3. ⏳ 分析未覆盖的代码，添加更多测试
4. ⏳ 提升覆盖率到100%
