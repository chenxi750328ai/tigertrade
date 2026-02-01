# 🚀 快速开始指南

立即开始使用新的数据采集和训练系统！

---

## ⚡ 5分钟快速体验

### 1. 采集数据（模拟模式）

```bash
cd /home/cx/tigertrade/src

# 采集10天数据，约730条记录
python collect_large_dataset.py --days 10 --max-records 50000
```

**输出:**
```
✅ 数据已准备就绪！
生成的文件:
  - train: /home/cx/trading_data/large_dataset/train_*.csv    (510条)
  - val: /home/cx/trading_data/large_dataset/val_*.csv        (110条)
  - test: /home/cx/trading_data/large_dataset/test_*.csv      (110条)
```

### 2. 训练模型

```bash
# 使用刚采集的数据训练
python train_with_detailed_logging.py \
    --train-file /home/cx/trading_data/large_dataset/train_20260120_144703.csv \
    --val-file /home/cx/trading_data/large_dataset/val_20260120_144703.csv
```

**查看日志:**
```bash
# 实时查看训练进度
tail -f /home/cx/trading_data/training_logs/training_*.log

# 查看错误（如果有）
cat /home/cx/trading_data/training_logs/errors_*.log
```

### 3. 分析结果

```bash
# 查看训练指标
python -c "
import pandas as pd
df = pd.read_csv('/home/cx/trading_data/training_logs/metrics_*.csv')
print('每个epoch的平均准确率:')
print(df[df['phase']=='train'].groupby('epoch')['accuracy'].mean())
"
```

---

## 📊 采集大规模数据

### 方式1: 模拟模式（推荐测试用）

```bash
# 10万条数据（约需2分钟）
python collect_large_dataset.py --days 300 --max-records 100000
```

### 方式2: 真实API（需要配置）

```bash
# 先配置API
export USE_REAL_API=true
export TIGER_CONFIG_PATH=./openapicfg_prod

# 采集真实数据
python collect_large_dataset.py --real-api --days 365 --max-records 500000
```

---

## ⚙️ 自定义配置

### 通过环境变量

```bash
# 数据采集配置
export DAYS_TO_FETCH=60
export MAX_RECORDS=200000

# 标注配置
export LABEL_STRATEGY=percentile    # 或 std, hybrid
export LOOK_AHEAD=10                # 向前看10个周期

# 训练配置
export BATCH_SIZE=64
export LEARNING_RATE=0.0001
export NUM_EPOCHS=100
export HIDDEN_DIM=256
export DEBUG_MODE=true

# 运行
python collect_large_dataset.py
python train_with_detailed_logging.py --train-file ... --val-file ...
```

### 查看所有配置

```bash
python config.py
```

---

## 🔍 检查配置

```bash
cd /home/cx/tigertrade/src

# 查看当前配置
python config.py

# 输出示例：
# ================================================================================
# 📋 数据采集配置
# ================================================================================
# 使用真实API: False
# 期货代码: NQ
# 获取天数: 30
# 最大记录数: 100000
# ...
```

---

## 📁 文件位置

### 数据文件
```
/home/cx/trading_data/large_dataset/
├── train_*.csv          # 训练集
├── val_*.csv            # 验证集
├── test_*.csv           # 测试集
├── full_*.csv           # 完整数据
└── dataset_info_*.txt   # 数据集信息
```

### 模型文件
```
/home/cx/trading_data/models/
├── best_model.pth              # 最佳模型
└── checkpoint_epoch_*.pth      # 定期检查点
```

### 日志文件
```
/home/cx/trading_data/training_logs/
├── training_*.log      # 完整训练日志
├── metrics_*.csv       # 每批次指标
└── errors_*.log        # 错误日志
```

---

## 🐛 故障排除

### 问题1: ModuleNotFoundError

```bash
# 确保在正确的目录
cd /home/cx/tigertrade/src

# 检查Python路径
python -c "import sys; print('\n'.join(sys.path))"
```

### 问题2: inplace操作错误

新版本已经避免了所有inplace操作。如果还遇到，请：

```bash
export DEBUG_MODE=true
python train_with_detailed_logging.py ...

# 查看错误日志
cat /home/cx/trading_data/training_logs/errors_*.log
```

### 问题3: 内存不足

```bash
# 减小批次大小
export BATCH_SIZE=16

# 或减少数据量
python collect_large_dataset.py --max-records 10000
```

### 问题4: GPU内存不足

```bash
# 使用CPU
export DEVICE=cpu

# 或减小模型
export HIDDEN_DIM=64
export NUM_LAYERS=2
```

---

## 💡 实用命令

### 查看数据

```bash
# 查看训练数据前10行
head -10 /home/cx/trading_data/large_dataset/train_*.csv

# 统计数据量
wc -l /home/cx/trading_data/large_dataset/*.csv

# 查看标签分布
python -c "
import pandas as pd
df = pd.read_csv('/home/cx/trading_data/large_dataset/train_*.csv', index_col=0)
print(df['label'].value_counts())
"
```

### 监控训练

```bash
# 实时监控
watch -n 1 'tail -20 /home/cx/trading_data/training_logs/training_*.log'

# 查看GPU使用
watch -n 1 nvidia-smi

# 查看进程
ps aux | grep python
```

### 清理文件

```bash
# 清理旧日志
rm /home/cx/trading_data/training_logs/training_*.log

# 清理旧模型（保留最新的）
cd /home/cx/trading_data/models
ls -t checkpoint_*.pth | tail -n +6 | xargs rm  # 保留最新5个
```

---

## 📚 完整文档

- **配置说明:** `/home/cx/tigertrade/数据和训练改进完成.md`
- **测试报告:** `/home/cx/tigertrade/数据优化完成报告.md`
- **原始测试:** `/home/cx/tigertrade/测试完成报告.md`

---

## ✅ 验证安装

运行这个命令验证一切正常：

```bash
cd /home/cx/tigertrade/src

# 1. 检查配置
python config.py | head -20

# 2. 快速数据测试
python collect_large_dataset.py --days 1 --max-records 1000

# 3. 检查生成的文件
ls -lh /home/cx/trading_data/large_dataset/

# 如果以上都成功，说明系统可以正常工作！
```

---

## 🎯 下一步

1. **实验不同配置**
   ```bash
   # 尝试不同的标注策略
   export LABEL_STRATEGY=std
   python collect_large_dataset.py --days 30 --max-records 10000
   ```

2. **比较模型性能**
   ```bash
   # 训练小模型
   export HIDDEN_DIM=64
   python train_with_detailed_logging.py ...
   
   # 训练大模型
   export HIDDEN_DIM=256
   python train_with_detailed_logging.py ...
   ```

3. **收集真实数据**
   ```bash
   # 配置真实API后
   python collect_large_dataset.py --real-api --days 365 --max-records 500000
   ```

---

**有问题？** 查看完整文档或检查日志文件！

**版本:** v2.0  
**更新时间:** 2026-01-20
