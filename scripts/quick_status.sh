#!/bin/bash
# 快速状态监控脚本

LOG_FILE="/tmp/moe_demo.log"

echo "=========================================="
echo "📊 DEMO运行状态快速监控"
echo "=========================================="
echo ""

# 进程状态
echo "【进程状态】"
PROC_LINE=$(ps aux | grep "python.*run_moe_demo" | grep -v grep | head -1)
if [ -n "$PROC_LINE" ]; then
    echo "$PROC_LINE" | awk '{print "  PID:", $2, "| CPU:", $3"%", "| 内存:", $4"%", "| 运行时间:", $10, $11}'
else
    echo "  ⚠️ 进程未运行"
fi

echo ""

# API状态
echo "【API连接】"
if grep -q "Mock模式: False" "$LOG_FILE" 2>/dev/null; then
    echo "  ✅ 真实DEMO账户"
else
    echo "  ⚠️ Mock模式或未知"
fi

echo ""

# 策略信息
echo "【策略信息】"
STRATEGY=$(grep "使用策略:" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oP '使用策略:\s*\K\S+' || echo "未知")
echo "  当前策略: $STRATEGY"

echo ""

# 运行时间
echo "【运行时间】"
START_TIME=$(grep "开始时间:" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oP '开始时间:\s*\K[0-9-]+\s+[0-9:]+' || echo "")
END_TIME=$(grep "结束时间:" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oP '结束时间:\s*\K[0-9-]+\s+[0-9:]+' || echo "")

if [ -n "$START_TIME" ]; then
    echo "  开始时间: $START_TIME"
    START_EPOCH=$(date -d "$START_TIME" +%s 2>/dev/null || echo "")
    if [ -n "$START_EPOCH" ]; then
        NOW_EPOCH=$(date +%s)
        ELAPSED=$((NOW_EPOCH - START_EPOCH))
        HOURS=$((ELAPSED / 3600))
        MINUTES=$(((ELAPSED % 3600) / 60))
        SECONDS=$((ELAPSED % 60))
        echo "  已运行: ${HOURS}小时${MINUTES}分钟${SECONDS}秒"
    fi
fi

if [ -n "$END_TIME" ]; then
    echo "  结束时间: $END_TIME"
    END_EPOCH=$(date -d "$END_TIME" +%s 2>/dev/null || echo "")
    if [ -n "$END_EPOCH" ]; then
        NOW_EPOCH=$(date +%s)
        REMAINING=$((END_EPOCH - NOW_EPOCH))
        if [ $REMAINING -gt 0 ]; then
            HOURS=$((REMAINING / 3600))
            MINUTES=$(((REMAINING % 3600) / 60))
            echo "  剩余时间: ${HOURS}小时${MINUTES}分钟"
        else
            echo "  ✅ 已完成"
        fi
    fi
fi

echo ""

# 统计信息
echo "【统计信息】"
TOTAL_PRED=$(grep -c "预测:" "$LOG_FILE" 2>/dev/null || echo "0")
BUY_SIGNALS=$(grep -c "动作: 买入" "$LOG_FILE" 2>/dev/null || echo "0")
SELL_SIGNALS=$(grep -c "动作: 卖出" "$LOG_FILE" 2>/dev/null || echo "0")
HOLD_SIGNALS=$(grep -c "动作: 不操作" "$LOG_FILE" 2>/dev/null || echo "0")
ERRORS=$(grep -c "❌" "$LOG_FILE" 2>/dev/null || echo "0")

echo "  总预测次数: $TOTAL_PRED"
echo "  买入: $BUY_SIGNALS | 卖出: $SELL_SIGNALS | 持有: $HOLD_SIGNALS"
echo "  错误次数: $ERRORS"

# 平均置信度
if [ -f "$LOG_FILE" ]; then
    AVG_CONF=$(grep -oP '置信度:\s*\K[\d.]+' "$LOG_FILE" 2>/dev/null | awk '{sum+=$1; count++} END {if(count>0) printf "%.3f", sum/count; else print "N/A"}')
    if [ "$AVG_CONF" != "N/A" ]; then
        echo "  平均置信度: $AVG_CONF"
    fi
fi

echo ""

# 最新预测
echo "【最新预测结果】"
tail -20 "$LOG_FILE" 2>/dev/null | grep -E "(预测:|动作:|置信度:|预测收益率:)" | tail -4

echo ""
echo "=========================================="
echo "💡 实时监控: python scripts/monitor_demo_status.py"
echo "💡 查看日志: tail -f $LOG_FILE"
echo "=========================================="
