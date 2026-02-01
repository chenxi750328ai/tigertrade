# Tiger Trade 测试工具使用指南

本文档介绍如何使用测试工具来评估和优化交易策略。

---

## 📋 目录

1. [快速开始](#快速开始)
2. [测试工具说明](#测试工具说明)
3. [使用示例](#使用示例)
4. [测试结果解读](#测试结果解读)
5. [常见问题](#常见问题)

---

## 🚀 快速开始

### 一键运行所有测试

```bash
cd /home/cx/tigertrade
./run_all_tests.sh
```

这将自动执行：
1. 策略性能测试（默认10次迭代）
2. 历史数据获取（默认7天）
3. 生成分析报告

### 自定义参数运行

```bash
# 每个策略测试20次，获取14天的历史数据
./run_all_tests.sh 20 14
```

---

## 🛠️ 测试工具说明

### 1. test_strategies.py - 策略测试工具

**功能:**
- 测试所有大模型交易策略
- 评估训练效果和推理性能
- 生成详细的测试数据

**使用方法:**
```bash
cd /home/cx/tigertrade/src
python test_strategies.py [迭代次数]

# 示例：每个策略测试15次
python test_strategies.py 15
```

**测试的策略:**
- LLM策略
- 大模型策略
- 超大Transformer策略
- 增强型Transformer策略
- 强化学习策略
- 大型Transformer策略

**输出文件:**
- `/home/cx/trading_data/strategy_tests/test_results_YYYYMMDD_HHMMSS.json`
- `/home/cx/trading_data/strategy_tests/test_results_YYYYMMDD_HHMMSS.csv`

---

### 2. fetch_more_data.py - 数据采集工具

**功能:**
- 通过API获取历史K线数据
- 计算技术指标特征
- 生成带标签的训练数据

**使用方法:**
```bash
cd /home/cx/tigertrade/src
python fetch_more_data.py [天数]

# 示例：获取30天的历史数据
python fetch_more_data.py 30
```

**数据类型:**
- 1分钟K线数据
- 5分钟K线数据
- 计算好的技术特征（ATR、RSI、布林带等）
- 训练标签（买入/卖出/持有）

**输出文件:**
- `/home/cx/trading_data/historical/kline_1min_YYYYMMDD_HHMMSS.csv`
- `/home/cx/trading_data/historical/kline_5min_YYYYMMDD_HHMMSS.csv`
- `/home/cx/trading_data/historical/training_data_YYYYMMDD_HHMMSS.csv`

---

### 3. generate_report.py - 报告生成工具

**功能:**
- 分析策略测试结果
- 生成综合排名
- 提供优化建议

**使用方法:**
```bash
cd /home/cx/tigertrade/src
python generate_report.py
```

**生成的分析:**
- 策略对比总结
- 行为模式分析
- 综合排名
- 优化建议

**输出文件:**
- `/home/cx/trading_data/reports/summary_YYYYMMDD_HHMMSS.csv`
- `/home/cx/trading_data/reports/rankings_YYYYMMDD_HHMMSS.csv`

---

## 💡 使用示例

### 示例1: 快速评估策略性能

```bash
# 运行5次快速测试
cd /home/cx/tigertrade/src
python test_strategies.py 5
python generate_report.py
```

### 示例2: 收集大量训练数据

```bash
# 获取60天的历史数据
cd /home/cx/tigertrade/src
python fetch_more_data.py 60
```

### 示例3: 完整的测试流程

```bash
# 1. 获取历史数据
python fetch_more_data.py 30

# 2. 运行策略测试
python test_strategies.py 20

# 3. 生成分析报告
python generate_report.py

# 4. 查看总结文档
cat /home/cx/tigertrade/TESTING_SUMMARY.md
```

### 示例4: 定期自动化测试

可以设置定时任务（cron）自动运行测试：

```bash
# 编辑crontab
crontab -e

# 添加以下行：每天早上8点运行测试
0 8 * * * /home/cx/tigertrade/run_all_tests.sh 10 7 >> /home/cx/tigertrade/test.log 2>&1
```

---

## 📊 测试结果解读

### 关键指标说明

#### 1. 成功率
- **定义:** 成功完成预测的次数占总测试次数的百分比
- **正常范围:** 80-100%
- **低于80%:** 可能存在数据获取问题或模型错误

#### 2. 平均置信度
- **定义:** 模型预测的平均置信度
- **解读:**
  - 0.0-0.3: 低置信度，模型不确定
  - 0.3-0.7: 中等置信度，需要谨慎
  - 0.7-1.0: 高置信度，模型较有把握

#### 3. 行为多样性
- **定义:** 预测结果的多样性（熵）
- **范围:** 0-1.58（对于3个类别）
- **解读:**
  - 0.0: 完全单一，所有预测相同
  - 0.5-1.0: 中等多样性
  - 1.0-1.58: 高度多样性

#### 4. 推理时间
- **定义:** 单次预测所需的时间
- **参考:**
  - < 5ms: 优秀
  - 5-10ms: 良好
  - > 10ms: 需要优化

#### 5. 综合得分
- **计算公式:** 
  ```
  得分 = 成功率 × 0.3 + 置信度 × 0.3 + 多样性 × 0.2 + 速度 × 0.2
  ```
- **范围:** 0-1
- **解读:**
  - > 0.5: 优秀
  - 0.3-0.5: 良好
  - < 0.3: 需要改进

---

## 📈 查看结果的方法

### 1. 查看JSON格式的详细结果

```bash
cat /home/cx/trading_data/strategy_tests/test_results_*.json | python -m json.tool
```

### 2. 使用pandas分析CSV数据

```python
import pandas as pd

# 读取测试结果
df = pd.read_csv('/home/cx/trading_data/strategy_tests/test_results_*.csv')

# 查看各策略的预测分布
print(df.groupby('strategy')['action_name'].value_counts())

# 查看置信度统计
print(df.groupby('strategy')['confidence'].describe())
```

### 3. 查看总结文档

```bash
# 使用Markdown查看器
cat /home/cx/tigertrade/TESTING_SUMMARY.md

# 或者在浏览器中打开（如果支持）
xdg-open /home/cx/tigertrade/TESTING_SUMMARY.md
```

---

## ❓ 常见问题

### Q1: 为什么测试成功率很低（40-60%）？

**原因:** Demo模式下数据不稳定，部分K线数据为空。

**解决方案:**
- 在真实交易环境中测试
- 增加错误重试机制
- 使用已保存的历史数据进行回测

### Q2: 为什么所有策略都预测"持有"？

**原因:** 
- 训练数据不足
- 决策阈值设置过高
- 模型过于保守

**解决方案:**
- 收集更多训练数据
- 调整策略的决策阈值
- 平衡训练数据的标签分布

### Q3: 如何提高模型的预测准确性？

**建议:**
1. 收集至少30天的历史数据
2. 增加特征工程（更多技术指标）
3. 使用交叉验证调整超参数
4. 实现回测系统验证收益
5. 在真实市场环境中测试

### Q4: 哪个策略最适合实盘交易？

**建议:**
- **首选:** 大型Transformer策略（综合得分最高）
- **备选:** 强化学习策略（唯一有多样性预测）
- **快速场景:** 增强型Transformer策略（推理最快）

**注意:** 任何策略在实盘前都需要充分的纸面交易验证。

### Q5: 如何获取更多数据？

**方法:**
```bash
# 获取更长时间的数据
python fetch_more_data.py 60  # 60天

# 或者手动调用API获取
# 可以修改fetch_more_data.py中的参数
```

**注意:** API可能有调用频率限制，建议分批获取。

### Q6: 如何调整策略参数？

策略参数通常在各策略模块中定义：
- `strategies/llm_strategy.py`
- `strategies/large_model_strategy.py`
- 等等...

修改后需要重新训练模型。

---

## 🔧 高级用法

### 自定义测试脚本

可以基于`test_strategies.py`创建自定义测试：

```python
from test_strategies import StrategyTester

# 创建测试器
tester = StrategyTester(iterations=50)

# 只测试特定策略
tester.test_strategy('LLM策略', tester.strategies['LLM策略'])

# 获取结果
results = tester.results
```

### 批量数据处理

```python
from fetch_more_data import fetch_historical_data, calculate_features_batch

# 获取数据
data = fetch_historical_data(days=30)

# 计算特征
features = calculate_features_batch(data['5min'], data['1min'])
```

---

## 📞 技术支持

如有问题或建议，请：
1. 查看日志文件
2. 检查数据文件是否生成
3. 确认API连接正常
4. 查看错误堆栈信息

---

## 📝 更新日志

### 2026-01-20
- ✅ 创建策略测试工具
- ✅ 实现数据采集功能
- ✅ 添加报告生成系统
- ✅ 完成一键测试脚本

---

**最后更新:** 2026-01-20  
**版本:** 1.0.0
