# MoE Transformer DEMO运行状态

**启动时间**: 2026-01-27 17:24  
**运行时长**: 20小时  
**预计结束时间**: 2026-01-28 13:24

---

## 一、运行配置

### 模型信息
- **模型类型**: MoE Transformer（最佳模型）
- **验证准确率**: 67.43%
- **收益率MAE**: 0.103424
- **序列长度**: 500步
- **特征维度**: 46（多时间尺度）

### 运行参数
- **账户类型**: DEMO账户
- **运行时长**: 20小时
- **预测间隔**: 5秒
- **日志文件**: `/tmp/moe_demo.log`

---

## 二、当前状态

### ✅ 运行中
- **进程ID**: 357766
- **CPU使用率**: ~43%
- **内存使用**: ~1.3GB
- **状态**: 正常运行

### 📊 预测示例
- **动作**: 不操作
- **置信度**: 0.634-0.644
- **预测收益率**: 2.25%-4.85%

---

## 三、监控方式

### 实时监控
```bash
# 查看实时日志
tail -f /tmp/moe_demo.log

# 查看进程状态
ps aux | grep "python.*run_moe_demo"

# 查看监控日志
tail -f /tmp/moe_demo_monitor.log
```

### 统计信息
脚本会自动统计：
- 总预测次数
- 买入信号数量
- 卖出信号数量
- 持有信号数量
- 平均置信度
- 错误次数

---

## 四、已知问题

### ⚠️ API连接错误
- **现象**: 偶尔出现 `'NoneType' object has no attribute 'get_future_bars'`
- **影响**: 不影响模型运行，脚本会自动重试
- **原因**: API连接暂时中断
- **处理**: 脚本会自动重试，继续运行

---

## 五、文件位置

### 脚本文件
- **主脚本**: `/home/cx/tigertrade/scripts/run_moe_demo.py`
- **策略类**: `/home/cx/tigertrade/src/strategies/moe_strategy.py`
- **监控脚本**: `/home/cx/tigertrade/scripts/monitor_moe_demo.sh`

### 日志文件
- **主日志**: `/tmp/moe_demo.log`
- **监控日志**: `/tmp/moe_demo_monitor.log`

### 模型文件
- **模型路径**: `/home/cx/trading_data/best_moe_transformer.pth`

---

## 六、停止运行

如果需要提前停止：

```bash
# 查找进程
ps aux | grep "python.*run_moe_demo"

# 停止进程（替换PID）
kill <PID>

# 或者使用pkill
pkill -f "run_moe_demo"
```

---

## 七、运行完成后

运行完成后，脚本会自动输出统计信息：
- 总预测次数
- 买入/卖出/持有信号统计
- 平均置信度
- 错误次数

---

**最后更新**: 2026-01-27 17:25
