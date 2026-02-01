#!/bin/bash
# 监控MoE Transformer DEMO运行进度

LOG_FILE="/tmp/moe_demo.log"
CHECK_INTERVAL=30

echo "=========================================="
echo "监控MoE Transformer DEMO运行进度"
echo "=========================================="
echo ""

while true; do
    clear
    echo "=========================================="
    echo "MoE Transformer DEMO监控 - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    echo ""
    
    # 检查进程
    if ps aux | grep -q "python.*run_moe_demo" | grep -v grep; then
        echo "✅ DEMO进程正在运行"
        ps aux | grep "python.*run_moe_demo" | grep -v grep | head -1 | awk '{print "  PID:", $2, "CPU:", $3"%", "MEM:", $4"%", "运行时间:", $10, $11}'
    else
        echo "⚠️ DEMO进程未运行"
    fi
    
    echo ""
    echo "📊 最新日志输出（最后20行）："
    echo "----------------------------------------"
    tail -20 "$LOG_FILE" 2>/dev/null || echo "日志文件不存在或为空"
    
    echo ""
    echo "----------------------------------------"
    echo "下次更新: $CHECK_INTERVAL 秒后..."
    sleep $CHECK_INTERVAL
done
