# pytest测试收集问题说明

## 问题描述

pytest无法收集测试用例，显示"collected 0 items / 1 skipped"和"ERROR: found no collectors"。

## 当前状态

1. **Python可以成功导入测试模块**：测试类和测试方法都存在
2. **pytest无法收集测试**：显示"no tests collected"
3. **所有测试文件都有同样的问题**

## 可能的原因

1. **测试文件在导入时执行了某些代码**，导致pytest认为文件有问题
2. **pytest配置问题**：`pytest.ini`配置可能有问题
3. **测试文件结构问题**：虽然使用`unittest.TestCase`，但pytest应该能识别

## 临时解决方案

由于pytest无法收集测试，**暂时继续使用unittest**，但使用coverage来生成覆盖率报告：

```bash
# 使用unittest运行测试
cd /home/cx/tigertrade
python -m unittest discover tests/ -v

# 使用coverage运行测试并生成报告
python -m coverage run --source=src -m unittest discover tests/
python -m coverage report --show-missing
python -m coverage html
```

## 下一步

1. ⏳ 调查pytest无法收集测试的根本原因
2. ⏳ 修复pytest配置或测试文件结构
3. ⏳ 确保pytest可以正常收集和运行测试
4. ⏳ 使用pytest + pytest-cov生成覆盖率报告

## 当前可用的测试方式

虽然pytest有问题，但可以使用以下方式：

1. **unittest + coverage**（当前可用）
   ```bash
   python -m coverage run --source=src -m unittest discover tests/
   python -m coverage report
   python -m coverage html
   ```

2. **直接运行测试文件**（如果测试文件有main函数）
   ```bash
   python tests/test_feature_buy_silver_comprehensive.py
   ```

## 总结

- ✅ 项目配置了pytest和coverage
- ❌ pytest无法收集测试（需要调查原因）
- ✅ 可以使用unittest + coverage作为临时方案
- ⏳ 需要修复pytest测试收集问题
