# API支持的K线周期说明

**更新时间**: 2026-01-23  
**状态**: ✅ 已更新代码支持所有周期

---

## 📊 一、TigerOpen API支持的K线周期

通过检查`tigeropen.common.consts.BarPeriod`，发现API支持以下所有周期：

### 1.1 分钟级周期

| 周期代码 | BarPeriod常量 | 说明 |
|---------|--------------|------|
| `1min` | `ONE_MINUTE` | 1分钟K线 |
| `3min` | `THREE_MINUTES` | 3分钟K线 |
| `5min` | `FIVE_MINUTES` | 5分钟K线 |
| `10min` | `TEN_MINUTES` | 10分钟K线 |
| `15min` | `FIFTEEN_MINUTES` | 15分钟K线 |
| `30min` | `HALF_HOUR` | 30分钟K线 |
| `45min` | `FORTY_FIVE_MINUTES` | 45分钟K线 |

### 1.2 小时级周期

| 周期代码 | BarPeriod常量 | 说明 |
|---------|--------------|------|
| `1h` | `ONE_HOUR` | 1小时K线 |
| `2h` | `TWO_HOURS` | 2小时K线 |
| `3h` | `THREE_HOURS` | 3小时K线 |
| `4h` | `FOUR_HOURS` | 4小时K线 |
| `6h` | `SIX_HOURS` | 6小时K线 |

### 1.3 日级及以上周期

| 周期代码 | BarPeriod常量 | 说明 |
|---------|--------------|------|
| `1d` | `DAY` | 日线K线 |
| `1w` | `WEEK` | 周线K线 |
| `1M` | `MONTH` | 月线K线 |
| `1y` | `YEAR` | 年线K线 |

---

## ✅ 二、代码更新

### 2.1 `tiger1.py` - `get_kline_data`函数

已更新`period_map`，支持所有周期：

```python
period_map = {
    "1min": BarPeriod.ONE_MINUTE,
    "3min": BarPeriod.THREE_MINUTES,
    "5min": BarPeriod.FIVE_MINUTES,
    "10min": BarPeriod.TEN_MINUTES,
    "15min": BarPeriod.FIFTEEN_MINUTES,
    "30min": BarPeriod.HALF_HOUR,
    "45min": BarPeriod.FORTY_FIVE_MINUTES,
    "1h": BarPeriod.ONE_HOUR,
    "2h": BarPeriod.TWO_HOURS,
    "3h": BarPeriod.THREE_HOURS,
    "4h": BarPeriod.FOUR_HOURS,
    "6h": BarPeriod.SIX_HOURS,
    "1d": BarPeriod.DAY,
    "1w": BarPeriod.WEEK,
    "1M": BarPeriod.MONTH,
    "1y": BarPeriod.YEAR,
}
```

### 2.2 `generate_multitimeframe_training_data_direct.py`

已更新以支持周线和月线：

- ✅ 添加`count_1w`和`count_1M`参数
- ✅ 获取周线和月线数据
- ✅ 计算周线和月线技术指标
- ✅ 在训练数据中包含周线和月线特征（待实现）

---

## 🎯 三、使用示例

### 3.1 获取不同周期的K线数据

```python
from src import tiger1 as t1

# 获取1分钟数据
df_1m = t1.get_kline_data('SIL2603', '1min', count=1000)

# 获取5分钟数据
df_5m = t1.get_kline_data('SIL2603', '5min', count=200)

# 获取1小时数据
df_1h = t1.get_kline_data('SIL2603', '1h', count=100)

# 获取日线数据
df_1d = t1.get_kline_data('SIL2603', '1d', count=30)

# 获取周线数据
df_1w = t1.get_kline_data('SIL2603', '1w', count=10)

# 获取月线数据
df_1M = t1.get_kline_data('SIL2603', '1M', count=12)
```

### 3.2 生成包含周线和月线的训练数据

```bash
python scripts/analysis/generate_multitimeframe_training_data_direct.py \
    --count-1m 10000 \
    --count-5m 2000 \
    --count-1h 500 \
    --count-1d 100 \
    --count-1w 50 \
    --count-1M 12
```

---

## 📈 四、多时间尺度策略建议

### 4.1 推荐的周期组合

**短期策略**:
- 1分钟、5分钟、15分钟、1小时

**中期策略**:
- 5分钟、1小时、4小时、日线

**长期策略**:
- 1小时、日线、周线、月线

**全周期策略**:
- 1分钟、5分钟、1小时、日线、周线、月线

### 4.2 特征维度扩展

当前多时间尺度特征（46维）：
- 1分钟：16维
- 5分钟：8维
- 1小时：9维
- 日线：11维
- 网格参数：2维

**添加周线和月线后**（预计60+维）：
- 1分钟：16维
- 5分钟：8维
- 1小时：9维
- 日线：11维
- **周线：11维**（新增）
- **月线：11维**（新增）
- 网格参数：2维

---

## ✅ 五、总结

### 5.1 完成状态

- ✅ 更新`tiger1.py`支持所有周期
- ✅ 更新训练数据生成脚本支持周线和月线
- ✅ 测试验证API支持周线和月线数据

### 5.2 下一步

- ⏳ 在训练数据生成中添加周线和月线特征
- ⏳ 更新模型输入维度（从46维扩展到60+维）
- ⏳ 重新训练模型，评估多时间尺度（包含周线、月线）的效果

---

**状态**: API支持所有周期，代码已更新，可以开始使用周线和月线数据
