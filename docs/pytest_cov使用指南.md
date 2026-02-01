# pytest-cov + coverage.py 使用指南

## 一、为什么选择 pytest-cov + coverage.py？

### 1.1 优势

**pytest-cov** 是 pytest 的覆盖率插件，**coverage.py** 是 Python 覆盖率工具的标准实现。两者结合是最强的组合：

✅ **pytest-cov的优势**：
- 与 pytest 无缝集成
- 支持并行测试（pytest-xdist）
- 支持多种报告格式（term, html, xml, json）
- 支持覆盖率阈值检查（--cov-fail-under）
- 支持分支覆盖率（--cov-branch）

✅ **coverage.py的优势**：
- Python 覆盖率工具的标准实现
- 支持多种配置方式（.coveragerc, setup.cfg, pyproject.toml）
- 支持数据合并（多个测试运行的结果合并）
- 支持 HTML 报告（可视化）
- 支持 XML 报告（CI集成）

### 1.2 组合优势

- **pytest-cov** 负责在 pytest 运行时收集覆盖率数据
- **coverage.py** 负责生成报告和分析
- 两者配合，既能在开发时快速查看覆盖率，也能在 CI 中生成详细报告

---

## 二、安装和配置

### 2.1 安装

```bash
pip install pytest pytest-cov coverage[toml]
```

### 2.2 配置文件

**pytest.ini** (pytest配置):
```ini
[pytest]
addopts = 
    --cov=src
    --cov-report=term-missing
    --cov-report=html:htmlcov
    --cov-report=xml:coverage.xml
    --cov-fail-under=50
```

**.coveragerc** (coverage.py配置):
```ini
[run]
source = src
omit = 
    */tests/*
    */__pycache__/*

[report]
precision = 2
show_missing = True
exclude_lines =
    pragma: no cover
    def __repr__
```

---

## 三、基本使用

### 3.1 运行测试并生成覆盖率

```bash
# 基本用法
pytest tests/ --cov=src

# 显示未覆盖的行
pytest tests/ --cov=src --cov-report=term-missing

# 生成HTML报告
pytest tests/ --cov=src --cov-report=html:htmlcov

# 生成XML报告（用于CI）
pytest tests/ --cov=src --cov-report=xml:coverage.xml

# 设置覆盖率阈值（低于50%失败）
pytest tests/ --cov=src --cov-fail-under=50
```

### 3.2 只测试特定模块

```bash
# 只测试executor模块
pytest tests/ --cov=src/executor --cov-report=term-missing

# 只测试特定文件
pytest tests/test_order_executor.py --cov=src/executor/order_executor.py
```

### 3.3 查看覆盖率报告

```bash
# 终端报告
coverage report -m

# HTML报告（浏览器打开）
coverage html
open htmlcov/index.html

# XML报告（CI工具使用）
coverage xml
```

---

## 四、CI集成

### 4.1 GitHub Actions

```yaml
- name: Run tests with coverage
  run: |
    pytest tests/ --cov=src \
      --cov-report=term-missing \
      --cov-report=html:htmlcov \
      --cov-report=xml:coverage.xml \
      --cov-fail-under=50

- name: Upload coverage
  uses: actions/upload-artifact@v4
  with:
    name: coverage-html
    path: htmlcov/
```

### 4.2 本地CI脚本

```bash
#!/bin/bash
# scripts/run_ci_tests.sh

# 运行测试
pytest tests/ --cov=src --cov-report=html:htmlcov --cov-report=xml:coverage.xml

# 检查覆盖率
coverage report --fail-under=50
```

---

## 五、高级功能

### 5.1 分支覆盖率

```bash
pytest tests/ --cov=src --cov-branch
```

### 5.2 合并多个测试运行

```bash
# 第一次运行
pytest tests/test_a.py --cov=src --cov-append

# 第二次运行
pytest tests/test_b.py --cov=src --cov-append

# 生成合并报告
coverage report
```

### 5.3 排除特定代码

**.coveragerc**:
```ini
[report]
exclude_lines =
    pragma: no cover
    def __repr__
    if __name__ == .__main__.:
    if TYPE_CHECKING:
```

代码中：
```python
if some_condition:  # pragma: no cover
    # 这行不会被计入覆盖率
    pass
```

---

## 六、常见问题解决

### 6.1 pytest找不到测试

**问题**: `pytest` 找不到测试文件

**解决**:
```ini
# pytest.ini
[pytest]
testpaths = tests
python_files = test_*.py
```

### 6.2 覆盖率报告为空

**问题**: 覆盖率报告显示0%

**解决**:
1. 检查 `source` 配置是否正确
2. 确保测试文件在 `tests/` 目录
3. 确保源代码在 `src/` 目录
4. 检查 `omit` 配置是否排除了源代码

### 6.3 覆盖率数据不准确

**问题**: 覆盖率数据不准确

**解决**:
1. 使用 `--cov-branch` 启用分支覆盖率
2. 检查 `exclude_lines` 配置
3. 确保测试真正执行了代码

### 6.4 CI中覆盖率失败

**问题**: CI中覆盖率检查失败

**解决**:
```bash
# 设置合理的阈值
pytest tests/ --cov=src --cov-fail-under=50

# 或者分模块检查
pytest tests/ --cov=src/executor --cov-fail-under=80
```

---

## 七、最佳实践

### 7.1 开发时

```bash
# 快速查看覆盖率
pytest tests/ --cov=src --cov-report=term-missing -v

# 查看HTML报告
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html
```

### 7.2 CI中

```bash
# 生成所有格式的报告
pytest tests/ --cov=src \
  --cov-report=term-missing \
  --cov-report=html:htmlcov \
  --cov-report=xml:coverage.xml \
  --cov-fail-under=50
```

### 7.3 报告解读

- **绿色行**: 已覆盖
- **红色行**: 未覆盖
- **黄色行**: 部分覆盖（分支覆盖率）

---

## 八、与unittest的兼容

### 8.1 同时支持pytest和unittest

pytest可以运行unittest测试：

```bash
# unittest测试也能用pytest运行
pytest tests/test_feature_risk_management.py -v

# 同样支持覆盖率
pytest tests/test_feature_risk_management.py --cov=src
```

### 8.2 迁移建议

1. **保持unittest测试**: pytest可以运行unittest测试，无需修改
2. **新测试用pytest**: 新测试使用pytest风格
3. **统一覆盖率**: 使用pytest-cov统一收集覆盖率

---

## 九、总结

**pytest-cov + coverage.py 是最强的组合**：

✅ **功能完整**: 支持所有覆盖率功能  
✅ **集成方便**: 与pytest无缝集成  
✅ **报告丰富**: 支持多种报告格式  
✅ **CI友好**: 支持XML报告和阈值检查  
✅ **兼容性好**: 可以运行unittest测试  

**使用建议**：
- 开发时：`pytest tests/ --cov=src --cov-report=html`
- CI中：`pytest tests/ --cov=src --cov-report=xml --cov-fail-under=50`
- 查看报告：`open htmlcov/index.html`
