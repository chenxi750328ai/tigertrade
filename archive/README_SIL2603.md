# SIL2603数据下载和全模型训练系统

## 🎯 快速开始（3步）

### 1️⃣ 运行测试（验证系统）

```bash
cd /home/cx/tigertrade
python3 quick_test_sil2603.py
```

**预期结果**: 所有测试通过 ✅

### 2️⃣ 启动完整流程

```bash
cd /home/cx/tigertrade
./run_sil2603_download_and_train.sh
```

### 3️⃣ 查看结果

```bash
cat /home/cx/trading_data/sil2603_large_dataset/final_report.txt
```

## 📚 文档导航

| 文档 | 说明 | 位置 |
|------|------|------|
| **项目完成总结** | 项目完成情况、功能特性、预期性能 | [项目完成总结.md](./项目完成总结.md) |
| **详细使用说明** | 完整的参数说明、配置调整、故障排除 | [SIL2603_数据采集和训练说明.md](./SIL2603_数据采集和训练说明.md) |
| **配置文件** | 所有可配置参数 | [src/config.py](./src/config.py) |

## 🛠️ 核心脚本

| 脚本 | 功能 | 用途 |
|------|------|------|
| `quick_test_sil2603.py` | 快速测试 | 验证系统组件是否正常 |
| `run_sil2603_download_and_train.sh` | 一键启动 | 执行完整的数据下载和训练流程 |
| `src/download_sil2603_and_train.py` | 主程序 | 数据采集+模型训练+测试评估 |

## 🎓 训练的模型（6个）

1. ✅ LLM策略 (LLMTradingStrategy)
2. ✅ 大模型策略 (LargeModelStrategy)
3. ✅ 超大Transformer策略 (HugeTransformerStrategy)
4. ✅ 增强型Transformer策略 (EnhancedTransformerStrategy)
5. ✅ 强化学习策略 (RLTradingStrategy)
6. ✅ 大型Transformer策略 (LargeTransformerStrategy)

## 💡 常见用法

### 基本用法（模拟模式）

```bash
python3 src/download_sil2603_and_train.py --days 60 --min-records 20000
```

### 使用真实API

```bash
python3 src/download_sil2603_and_train.py --real-api --days 90 --min-records 30000
```

### 大规模数据采集

```bash
python3 src/download_sil2603_and_train.py --days 120 --max-records 100000
```

### 后台运行

```bash
nohup ./run_sil2603_download_and_train.sh > output.log 2>&1 &
tail -f output.log
```

## 📊 系统要求

- **Python**: 3.x
- **PyTorch**: 2.x
- **GPU**: 推荐（已验证RTX 4090）
- **内存**: 16GB+
- **磁盘**: 10GB+

## ✅ 验证状态

| 组件 | 状态 |
|------|------|
| 配置文件 | ✅ 通过 |
| PyTorch环境 | ✅ 通过 (CUDA 12.1, RTX 4090) |
| Tiger API | ✅ 通过 |
| 模型导入 | ✅ 通过 (6个模型) |
| 数据采集 | ✅ 通过 (特征计算正常) |

## 📈 预期输出

完成后生成以下文件：

```
/home/cx/trading_data/sil2603_large_dataset/
├── 数据集（CSV格式）
│   ├── train_*.csv          # 训练集
│   ├── val_*.csv            # 验证集
│   ├── test_*.csv           # 测试集
│   └── full_*.csv           # 完整数据
├── 模型（PyTorch格式）
│   ├── LLM策略_best.pth
│   ├── 大模型策略_best.pth
│   └── ... (共6个模型)
├── 日志
│   ├── collection_log_*.txt
│   └── training_logs/
├── 结果报告
│   ├── final_report.txt     # 文本格式
│   └── all_results.json     # JSON格式
└── 数据集信息
    └── dataset_info_*.txt
```

## 🎯 参考现有脚本

项目中已有的数据采集和分析脚本（可作为参考）：

### 数据采集脚本
```
scripts/data_collection/
├── simple_data_collector.py      # 简单数据收集
├── data_collector_analyzer.py    # 数据收集+分析
└── data_collection_analyzer.py   # 数据收集分析器
```

### 定时分析脚本
```
scripts/analysis/
├── hourly_analysis.py             # 每小时分析
├── analyze_today_data.py          # 今日数据分析
├── analyze_real_data.py           # 真实数据分析
└── sil2603_analysis_report.py     # SIL2603专项分析
```

## 🔧 参数调整

### 调整数据量

编辑 `run_sil2603_download_and_train.sh`:

```bash
DAYS=120             # 获取天数
MIN_RECORDS=50000    # 最少记录数
MAX_RECORDS=100000   # 最大记录数
```

### 调整训练参数

编辑 `src/config.py`:

```python
class TrainingConfig:
    BATCH_SIZE = 64           # 批次大小
    LEARNING_RATE = 0.0001    # 学习率
    NUM_EPOCHS = 100          # 训练轮数
```

## 🐛 故障排除

### 数据量不足
→ 增加 `--days` 参数或使用 `--real-api`

### GPU内存不足
→ 减少 `BATCH_SIZE` 或使用 `DEVICE = 'cpu'`

### 训练时间过长
→ 减少 `NUM_EPOCHS` 或训练部分模型

详细故障排除请参阅：[SIL2603_数据采集和训练说明.md](./SIL2603_数据采集和训练说明.md)

## 📞 获取帮助

1. **查看日志**: `training_logs/error_*.log`
2. **运行测试**: `python3 quick_test_sil2603.py`
3. **检查配置**: `src/config.py`
4. **参阅文档**: `SIL2603_数据采集和训练说明.md`

## 🚀 立即开始

```bash
# 1. 进入项目目录
cd /home/cx/tigertrade

# 2. 运行快速测试
python3 quick_test_sil2603.py

# 3. 启动完整流程
./run_sil2603_download_and_train.sh
```

---

**项目状态**: ✅ 已完成并通过测试

**创建日期**: 2026-01-20

**目标**: 下载SIL2603数据（>20000条）+ 训练所有模型 + 生成评估报告

**系统环境**: Linux (WSL2), Python 3.x, PyTorch 2.1.0+cu121, CUDA 12.1, RTX 4090
