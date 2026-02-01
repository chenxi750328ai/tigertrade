# Archive - 归档文件说明

本目录包含已归档的旧版本脚本文件。这些文件已被更新的版本替代，但保留以供参考。

## 归档时间

2026-01-20

## 归档原因

项目整理，移除重复和冗余文件，保持代码库清晰。

## 目录结构

```
archive/
├── old_training_scripts/          # 旧训练脚本
│   ├── extended_training.py       # 扩展训练（400行）- 被train_all_models.py替代
│   ├── train_all_real_models.py   # 真实模型训练（313行）- 功能已合并
│   └── train_multiple_configs.py  # 多配置训练（315行）- 功能已合并
│
└── old_collection_scripts/        # 旧数据采集脚本
    ├── collect_massive_1min_data.py    # 大规模1分钟数据（385行）
    ├── enhanced_data_collection.py     # 增强数据收集（587行）
    └── data_collection_analyzer.py     # 数据收集分析（483行）- 重复文件
```

## 替代方案

### 训练相关

| 旧文件 | 替代方案 |
|--------|----------|
| `extended_training.py` | 使用 `src/train_all_models.py` |
| `train_all_real_models.py` | 使用 `src/train_all_models.py` |
| `train_multiple_configs.py` | 使用 `src/train_all_models.py` 或修改 `src/config.py` |

### 数据采集相关

| 旧文件 | 替代方案 |
|--------|----------|
| `collect_massive_1min_data.py` | 使用 `src/collect_large_dataset.py` |
| `enhanced_data_collection.py` | 使用 `src/collect_large_dataset.py` |
| `data_collection_analyzer.py` | 使用 `scripts/data_collection/data_collector_analyzer.py` |

## 当前推荐使用

### 完整流程
```bash
# 数据采集 + 模型训练 + 测试评估
python3 src/download_sil2603_and_train.py --days 60 --min-records 20000
```

### 仅数据采集
```bash
python3 src/collect_large_dataset.py --days 60 --max-records 50000
```

### 仅模型训练
```bash
python3 src/train_all_models.py --train-file train.csv --val-file val.csv
```

## 注意事项

1. 这些文件**不会被删除**，仅归档保存
2. 如需使用归档文件，可以直接运行或参考其代码
3. 不建议在生产环境使用归档文件
4. 归档文件可能包含过时的API调用或配置

## 恢复文件

如需恢复任何归档文件：

```bash
# 从归档恢复到原位置
cp archive/old_training_scripts/extended_training.py /home/cx/tigertrade/
```

---

**维护建议**: 定期检查归档文件，超过6个月未使用可考虑删除
