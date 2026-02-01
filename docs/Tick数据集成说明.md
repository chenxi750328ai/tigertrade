# Tick数据集成说明

**更新时间**: 2026-01-23  
**状态**: 已集成真实Tick数据到训练输入

---

## ✅ 一、Tick数据的重要性

### 1.1 为什么Tick数据很重要

**用户反馈**: "tick数据是真实获取的，不是从K线数据里伪造的，这个很重要，我和你说过我的人工策略，和实时数据关系很大"

**Tick数据的优势**:
- ✅ **更精确的价格**: Tick价格是实时成交价格，比K线收盘价更准确
- ✅ **更及时的信息**: Tick数据反映最新的市场状态
- ✅ **成交量信息**: 可以区分买入和卖出成交量
- ✅ **价格波动**: 可以看到K线周期内的价格波动细节

### 1.2 与手工策略的关系

**手工策略依赖实时数据**:
- 手工策略可能基于Tick价格的实时变化做决策
- Tick价格与K线价格的差异可能包含重要信息
- Tick成交量可以反映市场情绪（买卖力量对比）

---

## 📊 二、Tick数据特征

### 2.1 新增的Tick特征（18维）

**之前（12维）**:
1. price_current
2. atr
3. rsi_1m
4. rsi_5m
5. grid_lower
6. grid_upper
7. boll_upper
8. boll_mid
9. boll_lower
10. boll_position
11. volatility
12. volume_1m

**现在（18维，包含真实Tick数据）**:
1. **price_current** - K线价格
2. **tick_price** - 真实Tick价格（重要！）
3. **tick_price_change** - Tick价格相对于K线价格的变化
4. **tick_volatility** - Tick价格波动率
5. **tick_volume** - Tick成交量
6. **tick_count** - Tick数量
7. **tick_buy_sell_ratio** - Tick买卖比例
8. atr
9. rsi_1m
10. rsi_5m
11. grid_lower
12. grid_upper
13. boll_upper
14. boll_mid
15. boll_lower
16. boll_position
17. volatility
18. volume_1m

---

## 🔧 三、实现细节

### 3.1 训练数据生成

**脚本**: `scripts/analysis/generate_training_data_from_klines.py`

**Tick数据来源**:
- 从 `/home/cx/trading_data/ticks/` 目录加载真实的Tick数据文件
- 文件格式: `SIL2603_ticks_YYYYMMDD.csv`
- 列: `identifier, index, price, volume, time, datetime`

**Tick数据匹配**:
- 对于每个K线时间点，找到该时间窗口内的Tick数据
- 时间窗口: K线时间 ± 30秒
- 使用最新的Tick价格作为该K线的Tick价格

**Tick特征计算**:
```python
# Tick价格（最新Tick）
tick_price = ticks_in_window['price'].iloc[-1]

# Tick价格变化（相对于K线价格）
tick_price_change = (tick_price - kline_price) / kline_price

# Tick波动率（该窗口内Tick价格的标准差）
tick_volatility = ticks_in_window['price'].std() / kline_price

# Tick成交量
tick_volume = ticks_in_window['volume'].sum()

# Tick数量
tick_count = len(ticks_in_window)

# 买卖成交量
tick_buy_volume = buy_ticks['volume'].sum()
tick_sell_volume = sell_ticks['volume'].sum()

# 买卖比例
tick_buy_sell_ratio = tick_buy_volume / (tick_buy_volume + tick_sell_volume)
```

### 3.2 特征提取更新

**代码位置**: `src/strategies/llm_strategy.py` 第711-740行

**更新内容**:
- 从训练数据中提取Tick相关特征
- 如果Tick数据不存在，使用默认值（向后兼容）
- 特征维度从12维增加到18维

---

## 📈 四、Tick数据文件

### 4.1 文件位置

**Tick数据目录**: `/home/cx/trading_data/ticks/`

**文件格式**: `SIL2603_ticks_YYYYMMDD.csv`

**文件列**:
- `identifier`: 合约代码
- `index`: Tick索引
- `price`: Tick价格（重要！）
- `volume`: Tick成交量
- `time`: 时间戳（毫秒）
- `datetime`: 日期时间

### 4.2 数据采集

**采集器**: `src/tick_data_collector.py`

**采集方式**:
- 使用Tiger API的 `get_future_trade_ticks` 方法
- 实时采集并保存到CSV文件
- 按日期分文件保存

---

## 🎯 五、使用说明

### 5.1 生成包含Tick数据的训练数据

```bash
cd /home/cx/tigertrade
python scripts/analysis/generate_training_data_from_klines.py
```

**要求**:
- Tick数据文件必须存在于 `/home/cx/trading_data/ticks/`
- 如果Tick数据不存在，将使用K线价格作为Tick价格（向后兼容）

### 5.2 训练模型

```python
from src.strategies.llm_strategy import LLMTradingStrategy

# 使用包含Tick数据的训练数据
strategy = LLMTradingStrategy(mode='hybrid', predict_profit=True)
strategy.train_model(df, seq_length=10, max_epochs=50, patience=10)
```

**注意**:
- 训练数据必须包含Tick相关列（tick_price, tick_price_change等）
- 如果缺少Tick数据，特征提取会使用默认值

---

## ✅ 六、总结

### 6.1 已完成

1. ✅ 更新训练数据生成脚本，使用真实的Tick数据
2. ✅ 添加Tick相关特征（tick_price, tick_price_change, tick_volatility等）
3. ✅ 更新特征提取，支持18维特征（包含Tick数据）
4. ✅ 更新模型输入维度（从12维增加到18维）

### 6.2 核心改进

- **真实Tick数据**: 使用采集器保存的真实Tick数据，而不是从K线伪造
- **更多Tick特征**: 包含价格、成交量、波动率、买卖比例等
- **向后兼容**: 如果Tick数据不存在，使用K线价格作为默认值

---

**状态**: Tick数据已集成到训练输入中，特征维度从12维增加到18维
