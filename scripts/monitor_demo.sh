#!/bin/bash
# 监控20小时DEMO运行状态

LOG_FILE=$(ls -t /home/cx/tigertrade/logs/demo_20h_*.log 2>/dev/null | head -1)

if [ -z "$LOG_FILE" ]; then
    echo "❌ 未找到运行日志"
    exit 1
fi

echo "=========================================="
echo "监控DEMO运行状态"
echo "=========================================="
echo "📝 日志文件: $LOG_FILE"
echo ""

# 检查进程
PID=$(ps aux | grep "run_moe_demo" | grep -v grep | awk '{print $2}')
if [ -z "$PID" ]; then
    echo "❌ 进程未运行"
    exit 1
else
    echo "✅ 进程运行中 (PID: $PID)"
fi

echo ""
echo "最近的关键日志:"
echo "----------------------------------------"

# 显示最近的account相关日志
echo "📋 Account状态:"
tail -100 "$LOG_FILE" | grep -E "account|Account" | tail -5

echo ""
echo "📊 下单记录:"
tail -100 "$LOG_FILE" | grep -E "下单|place_order|Order|执行买入|执行卖出" | tail -5

echo ""
echo "❌ 错误记录:"
tail -100 "$LOG_FILE" | grep -E "❌|ERROR|Error|失败|失败" | tail -5

echo ""
echo "=========================================="
echo "实时监控 (Ctrl+C退出):"
echo "tail -f $LOG_FILE"
echo "=========================================="
