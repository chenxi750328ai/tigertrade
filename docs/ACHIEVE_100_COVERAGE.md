# 达到100%代码覆盖率和分支覆盖率的方案

## 当前状态

### 测试通过率
- ✅ **总测试数**: 63
- ✅ **通过测试**: 62 (98.41%)
- ⚠️ **失败测试**: 0
- ❌ **错误测试**: 1

### 代码覆盖率
- **当前覆盖率**: 52%
- **目标覆盖率**: 100%

## 未覆盖的主要代码区域

### 1. 主函数 (2233-2721行) - 约500行未覆盖
**问题**: 主函数包含while True循环，需要特殊处理

**解决方案**:
```python
# 使用mock来模拟while循环
def test_main_function_all_strategies():
    with patch('time.sleep', return_value=None):  # 避免实际等待
        with patch('builtins.input', return_value='q'):  # 模拟退出
            # 测试每个策略分支
            for strategy in ['backtest', 'llm', 'grid', 'boll', 'compare', 'large', 'huge', 'optimize', 'all']:
                with patch('sys.argv', ['tiger1.py', 'd', strategy]):
                    # 使用threading.Timer来中断循环
                    timer = threading.Timer(0.1, lambda: sys.exit(0))
                    timer.start()
                    try:
                        # 导入并执行主函数
                        import tiger1
                        exec(open('tiger1.py').read())
                    except SystemExit:
                        pass
```

### 2. get_kline_data详细实现 (918-1009行) - 约90行未覆盖
**问题**: 分页逻辑、时间转换、异常处理的所有分支

**解决方案**:
```python
def test_get_kline_data_all_scenarios():
    # 测试所有分页返回格式
    # 测试所有时间格式转换
    # 测试所有异常情况
    # 测试所有边界条件
```

### 3. 策略函数的详细分支 - 约200行未覆盖
**问题**: 各种条件组合、边界情况

**解决方案**:
```python
# 使用参数化测试覆盖所有条件组合
@pytest.mark.parametrize("rsi_1m,rsi_5m,price,expected", [
    (10, 20, 88.0, True),   # 买入条件
    (80, 70, 92.0, False),  # 卖出条件
    # ... 更多组合
])
def test_strategy_conditions(rsi_1m, rsi_5m, price, expected):
    # 测试各种条件组合
```

## 实施步骤

### 步骤1: 修复剩余的错误测试
```bash
# 修复test_main_function_execution中的导入错误
```

### 步骤2: 添加主函数测试
创建 `test_main_function_direct.py`:
```python
import threading
import sys
from unittest.mock import patch

def test_main_all_strategies():
    """直接测试主函数的所有策略路径"""
    for strategy in ['backtest', 'llm', 'grid', 'boll', 'compare', 'large', 'huge', 'optimize', 'all']:
        # 使用Timer中断循环
        timer = threading.Timer(0.1, lambda: sys.exit(0))
        timer.start()
        # 测试主函数
```

### 步骤3: 添加get_kline_data的全面测试
创建 `test_get_kline_comprehensive.py`:
```python
def test_get_kline_all_formats():
    """测试所有数据格式和转换"""
    # DataFrame格式
    # 列表格式
    # 字典格式
    # 时间戳格式
    # 字符串时间格式
    # 分页token处理
    # 异常处理
```

### 步骤4: 添加策略函数的参数化测试
使用pytest的parametrize来覆盖所有条件组合

### 步骤5: 运行完整测试套件
```bash
python -m coverage run --source=. --include="tiger1.py" -m pytest test_*.py -v
python -m coverage report --include="tiger1.py" --show-missing
python -m coverage html --include="tiger1.py" -d htmlcov
```

## 快速命令

### 运行所有测试
```bash
cd /home/cx/tigertrade
python run_test_clean.py
```

### 生成覆盖率报告
```bash
cd /home/cx/tigertrade
./run_100_coverage_test.sh
```

### 查看详细覆盖率
```bash
python -m coverage report --include="tiger1.py" --show-missing | less
```

## 注意事项

1. **主函数测试**: 需要使用Timer或mock来避免无限循环
2. **时间相关**: 使用mock来避免实际等待
3. **并发测试**: 主函数的'all'策略使用多线程，需要特殊处理
4. **环境变量**: 某些代码路径需要特定的环境变量

## 预期结果

完成所有步骤后：
- ✅ 测试通过率: 100%
- ✅ 代码覆盖率: 100%
- ✅ 分支覆盖率: 100%
