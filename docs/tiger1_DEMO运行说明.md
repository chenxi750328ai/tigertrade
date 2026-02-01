# tiger1策略DEMO账户运行说明

**启动时间**: 2026-01-23  
**运行模式**: DEMO/Sandbox  
**运行时长**: 20小时

---

## 📊 一、运行状态

### 启动信息
- **PID**: 见 `/tmp/tiger1_demo.pid`
- **日志文件**: `logs/tiger1_demo_YYYYMMDD_HHMMSS.log`
- **预计结束时间**: 启动后20小时

### 当前配置
- **账户模式**: DEMO/Sandbox（使用 `'d'` 参数）
- **策略类型**: 默认运行所有策略（grid + boll）
- **时段自适应**: 已启用
- **序列长度优化**: 已集成

---

## 🔍 二、监控命令

### 1. 查看实时日志
```bash
# 查看最新日志
tail -f /home/cx/tigertrade/logs/tiger1_demo_*.log

# 或使用监控脚本
python scripts/analysis/monitor_tiger1.py
```

### 2. 检查进程状态
```bash
# 检查进程
ps aux | grep "tiger1.py d"

# 检查PID文件
cat /tmp/tiger1_demo.pid
```

### 3. 查看数据收集
```bash
# 查看今日数据
ls -lh /home/cx/trading_data/$(date +%Y-%m-%d)/

# 查看最新数据文件
tail -f /home/cx/trading_data/$(date +%Y-%m-%d)/trading_data_*.csv
```

---

## 🛑 三、停止命令

### 方法1：使用PID文件
```bash
kill $(cat /tmp/tiger1_demo.pid)
```

### 方法2：查找进程并终止
```bash
pkill -f "tiger1.py d"
```

### 方法3：使用监控脚本
```bash
# 监控脚本会显示PID，然后使用kill命令
kill <PID>
```

### 方法4：延长运行时间
```bash
# 如果策略已在运行，可以手动延长（需要重启）
# 或者等待当前运行结束后，下次启动将自动使用20小时配置
bash scripts/extend_tiger1_runtime.sh
```

---

## 📈 四、策略特性

### 4.1 时段自适应策略
- **自动识别交易时段**: COMEX、沪银等
- **动态调整参数**: 网格间距、仓位、滑点阈值
- **数据驱动**: 优先使用历史数据分析结果
- **参考规则**: 使用RAG系统中的参考规则作为备选

### 4.2 网格策略
- **动态网格**: 基于布林带和ATR
- **风控**: 止损、止盈、仓位控制
- **时段适配**: 根据时段调整网格参数

### 4.3 数据收集
- **实时数据**: 每5秒收集一次市场数据
- **策略决策**: 记录所有决策和参数
- **数据存储**: 保存到 `/home/cx/trading_data/YYYY-MM-DD/`

---

## 📝 五、日志说明

### 日志内容
- **策略决策**: 买入/卖出信号及原因
- **参数调整**: 网格、ATR、RSI等指标
- **时段识别**: 当前交易时段及配置
- **风控检查**: 风险控制决策
- **API调用**: 数据获取和订单执行

### 日志格式
```
[时间戳] 日志内容
```

---

## ⚙️ 六、配置说明

### 6.1 DEMO模式配置
- **配置文件**: `./openapicfg_dem/tiger_openapi_config.properties`
- **环境**: `env=sandbox`（模拟环境）
- **下单模拟**: 在sandbox模式下，失败的下单会被模拟为成功

### 6.2 策略参数
- **网格周期**: 20个K线
- **风控参数**: 
  - 日亏损上限: $1200
  - 单笔最大亏损: $1000
  - 止损ATR倍数: 1.2
- **运行间隔**: 5秒

---

## 🔄 七、自动停止

策略会在以下情况自动停止：
1. **达到20小时运行时间**: 自动停止
2. **手动中断**: Ctrl+C 或 kill命令
3. **异常退出**: 程序异常时会记录日志

---

## 📊 八、运行后分析

### 8.1 数据文件
运行结束后，可以分析：
- `/home/cx/trading_data/YYYY-MM-DD/trading_data_*.csv`: 策略决策数据
- `/home/cx/tigertrade/logs/tiger1_demo_*.log`: 完整运行日志

### 8.2 分析工具
```bash
# 分析今日数据
python scripts/analysis/analyze_today_data.py

# 时段分析
python scripts/analysis/time_period_analyzer.py

# 策略效果分析
python scripts/analysis/analyze_real_data.py
```

---

## 💡 九、注意事项

1. **DEMO模式**: 所有交易都是模拟的，不会产生真实盈亏
2. **数据收集**: 确保有足够的磁盘空间存储数据
3. **网络连接**: 需要稳定的网络连接获取市场数据
4. **时段识别**: 策略会自动识别交易时段，无需手动调整

---

## 🎯 十、预期效果

### 10.1 策略行为
- **自动交易**: 根据市场条件自动买入/卖出
- **时段适配**: 不同时段使用不同参数
- **风控执行**: 严格执行止损和仓位控制

### 10.2 数据产出
- **决策记录**: 所有交易决策和原因
- **参数记录**: 网格、ATR、RSI等指标
- **时段数据**: 时段特征和适配参数

---

**最后更新**: 2026-01-23
