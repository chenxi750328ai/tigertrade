#!/bin/bash
# 监控训练进度脚本

LOG_FILE="/tmp/improvements_123.log"
TRAINING_LOG="/tmp/model_comparison_fixed_v2.log"
CHECK_INTERVAL=30  # 每30秒检查一次

echo "=========================================="
echo "开始监控训练进度"
echo "=========================================="
echo ""

while true; do
    clear
    echo "=========================================="
    echo "训练进度监控 - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    echo ""
    
    # 检查进程状态
    if ps aux | grep -q "python.*implement_improvements_123" | grep -v grep; then
        echo "✅ 主脚本正在运行"
        ps aux | grep "python.*implement_improvements_123" | grep -v grep | awk '{print "  PID:", $2, "CPU:", $3"%", "MEM:", $4"%"}'
    else
        echo "⚠️ 主脚本未运行"
    fi
    
    echo ""
    echo "📊 最新日志输出（最后20行）："
    echo "----------------------------------------"
    tail -20 "$LOG_FILE" 2>/dev/null | tail -20
    echo ""
    
    # 检查是否开始训练
    if grep -q "开始训练\|训练\|Training\|Epoch\|epoch" "$LOG_FILE" 2>/dev/null | tail -1; then
        echo ""
        echo "🎯 训练已开始！"
        if [ -f "$TRAINING_LOG" ]; then
            echo ""
            echo "📈 训练日志（最后10行）："
            echo "----------------------------------------"
            tail -10 "$TRAINING_LOG" 2>/dev/null
        fi
    fi
    
    # 检查是否完成
    if grep -q "✅ 训练完成\|训练完成\|Training completed" "$LOG_FILE" 2>/dev/null; then
        echo ""
        echo "🎉 训练已完成！"
        break
    fi
    
    echo ""
    echo "----------------------------------------"
    echo "下次更新: $CHECK_INTERVAL 秒后..."
    sleep $CHECK_INTERVAL
done

echo ""
echo "=========================================="
echo "监控结束"
echo "=========================================="
