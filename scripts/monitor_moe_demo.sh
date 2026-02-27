#!/bin/bash
# 监控MoE Transformer DEMO运行进度
# 使用实际 demo 日志：logs/demo_20h_*.log

cd "$(dirname "$0")/.." || exit 1
LOG_FILE=$(ls -t logs/demo_20h_*.log 2>/dev/null | head -1)
LOG_FILE="${LOG_FILE:-/tmp/moe_demo.log}"
CHECK_INTERVAL=30

echo "=========================================="
echo "监控MoE Transformer DEMO运行进度"
echo "=========================================="
echo "日志: $LOG_FILE"
echo ""

while true; do
    clear
    echo "=========================================="
    echo "MoE Transformer DEMO监控 - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    echo ""
    
    # 检查进程
    if pgrep -f "run_moe_demo.py" >/dev/null || pgrep -f "tiger1.py.*moe" >/dev/null; then
        echo "✅ DEMO进程正在运行"
        ps aux | grep -E "run_moe_demo|tiger1.*moe" | grep -v grep | head -1 | awk '{print "  PID:", $2, "CPU:", $3"%", "MEM:", $4"%"}'
    else
        echo "⚠️ DEMO进程未运行"
    fi
    
    # 检查最近是否有错误
    if [ -f "$LOG_FILE" ]; then
        ERR=$(grep -E "Error|Exception|失败|Traceback" "$LOG_FILE" | tail -3)
        if [ -n "$ERR" ]; then
            echo ""
            echo "⚠️ 最近错误:"
            echo "$ERR"
        fi
    fi
    
    echo ""
    echo "📊 最新日志（最后15行）："
    echo "----------------------------------------"
    tail -15 "$LOG_FILE" 2>/dev/null || echo "日志不存在"
    
    echo ""
    echo "----------------------------------------"
    echo "下次更新: $CHECK_INTERVAL 秒后..."
    sleep $CHECK_INTERVAL
done
