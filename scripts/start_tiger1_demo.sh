#!/bin/bash
# 在DEMO账户运行tiger1策略8小时

cd /home/cx/tigertrade

# 创建日志目录
mkdir -p logs

# 日志文件
LOG_FILE="logs/tiger1_demo_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="/tmp/tiger1_demo.pid"

echo "============================================================"
echo "🚀 启动tiger1策略（DEMO账户，运行20小时）"
echo "============================================================"
echo "📅 开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "⏰ 预计结束时间: $(date -d '+20 hours' '+%Y-%m-%d %H:%M:%S')"
echo "📝 日志文件: $LOG_FILE"
echo "============================================================"

# 启动策略（使用'd'参数表示DEMO/sandbox模式，'llm'表示使用LLM模型策略）
nohup python src/tiger1.py d llm > "$LOG_FILE" 2>&1 &
PID=$!

# 保存PID
echo $PID > "$PID_FILE"

echo "✅ 策略已启动 (PID: $PID)"
echo "📝 日志文件: $LOG_FILE"
echo ""
echo "💡 监控命令:"
echo "   tail -f $LOG_FILE"
echo "   python scripts/analysis/monitor_tiger1.py"
echo ""
echo "💡 停止命令:"
echo "   kill $PID"
echo "   或: kill \$(cat $PID_FILE)"
echo "============================================================"

# 等待20小时
sleep 72000  # 20小时 = 72000秒

# 20小时后自动停止
echo ""
echo "⏰ 已达到20小时运行时间，正在停止..."
kill $PID 2>/dev/null
rm -f "$PID_FILE"
echo "✅ 策略已停止"
