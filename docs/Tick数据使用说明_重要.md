# Tick数据使用说明（重要！）

**更新时间**: 2026-01-23

---

## ⚠️ 重要提醒

**Tick数据是从DEMO账户真实获取的，不是伪造的！**

- ✅ Tick数据来源：通过 `tick_data_collector.py` 从DEMO账户真实采集
- ✅ 数据保存位置：`/home/cx/trading_data/ticks/SIL2603_ticks_YYYYMMDD.csv`
- ❌ **不要**自己生成或伪造Tick数据
- ❌ **不要**从K线数据推导Tick数据

---

## 📊 一、Tick数据采集

### 1.1 采集器

**文件**: `src/tick_data_collector.py`

**功能**:
- 从DEMO账户通过Tiger API获取真实Tick数据
- 使用 `quote_client.get_future_trade_ticks()` 方法
- 实时采集并保存到CSV文件

**配置**:
- 使用DEMO账户配置：`TigerOpenClientConfig(props_path='./openapicfg_dem')`
- 保存目录：`/home/cx/trading_data/ticks/`

### 1.2 启动采集器

```bash
cd /home/cx/tigertrade
python src/tick_data_collector.py --mode both
```

**或者使用启动脚本**:
```bash
bash 启动Tick采集器.sh
```

### 1.3 数据文件格式

**文件**: `SIL2603_ticks_YYYYMMDD.csv`

**列**:
- `identifier`: 合约代码
- `index`: Tick索引
- `price`: Tick价格（真实成交价格，从DEMO账户获取）
- `volume`: Tick成交量
- `time`: 时间戳（毫秒）
- `datetime`: 日期时间

---

## 🔧 二、训练数据生成

### 2.1 使用真实Tick数据

**脚本**: `scripts/analysis/generate_training_data_from_klines.py`

**流程**:
1. 从 `/home/cx/trading_data/ticks/` 目录加载真实Tick数据文件
2. 合并所有Tick文件（按时间排序）
3. 匹配Tick数据到K线（时间窗口匹配）
4. 计算Tick特征（价格、成交量、波动率等）
5. 生成训练数据（包含真实Tick特征）

**重要**:
- ✅ 使用从DEMO账户采集的真实Tick数据
- ❌ 不使用K线价格作为Tick价格（除非Tick数据不存在）

### 2.2 代码示例

```python
# 加载真实的Tick数据（从DEMO账户采集器保存的文件）
tick_dir = '/home/cx/trading_data/ticks'
tick_files = glob.glob(os.path.join(tick_dir, 'SIL2603_ticks_*.csv'))

if tick_files:
    # 合并所有Tick文件（真实数据）
    all_ticks = []
    for tick_file in sorted(tick_files):
        df_ticks = pd.read_csv(tick_file)
        # 处理时间列
        if 'time' in df_ticks.columns:
            df_ticks['datetime'] = pd.to_datetime(df_ticks['time'], unit='ms')
        all_ticks.append(df_ticks)
    
    tick_data = pd.concat(all_ticks, ignore_index=True)
    tick_data = tick_data.sort_values('datetime').reset_index(drop=True)
```

---

## ✅ 三、验证Tick数据真实性

### 3.1 检查数据文件

```bash
# 查看Tick数据文件
ls -lh /home/cx/trading_data/ticks/*.csv

# 查看文件内容（前几行）
head -5 /home/cx/trading_data/ticks/SIL2603_ticks_*.csv
```

### 3.2 检查采集器状态

```bash
# 查看采集器进程
ps aux | grep tick_data_collector

# 查看采集器日志
tail -f /home/cx/trading_data/ticks/collector.log
```

### 3.3 验证数据来源

**Tick数据必须**:
- ✅ 来自DEMO账户（通过Tiger API）
- ✅ 由 `tick_data_collector.py` 采集
- ✅ 保存在 `/home/cx/trading_data/ticks/` 目录
- ✅ 包含真实的 `price`、`volume`、`time` 列

**Tick数据不能**:
- ❌ 从K线数据推导
- ❌ 随机生成
- ❌ 使用模拟数据（除非在测试模式下）

---

## 🎯 四、常见错误

### 4.1 错误：使用K线价格作为Tick价格

**错误代码**:
```python
tick_price = row['close']  # 这是K线价格，不是真实Tick价格！
```

**正确做法**:
```python
# 从真实Tick数据文件加载
tick_data = load_real_tick_data()  # 从 /home/cx/trading_data/ticks/ 加载
tick_price = get_tick_price_from_real_data(tick_data, kline_time)
```

### 4.2 错误：生成模拟Tick数据

**错误代码**:
```python
# 不要这样做！
tick_price = base_price + random.uniform(-0.1, 0.1)
```

**正确做法**:
```python
# 使用真实采集的Tick数据
tick_data = pd.read_csv('/home/cx/trading_data/ticks/SIL2603_ticks_20260122.csv')
tick_price = tick_data['price'].iloc[-1]  # 使用真实价格
```

---

## 📝 五、总结

### 5.1 核心原则

1. **Tick数据必须从DEMO账户真实获取**
2. **使用 `tick_data_collector.py` 采集真实数据**
3. **训练数据生成脚本必须使用真实Tick数据文件**
4. **不要伪造或生成Tick数据**

### 5.2 数据流程

```
DEMO账户 (Tiger API)
    ↓
tick_data_collector.py (采集器)
    ↓
/home/cx/trading_data/ticks/SIL2603_ticks_*.csv (真实数据文件)
    ↓
generate_training_data_from_klines.py (训练数据生成)
    ↓
training_data_from_klines_*.csv (包含真实Tick特征的训练数据)
```

---

**重要**: 始终使用从DEMO账户真实采集的Tick数据，不要伪造！
