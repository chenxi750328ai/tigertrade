#!/bin/bash
# 监控公平对比测试进度

LOG_FILE="/tmp/fair_comparison_full.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "⚠️ 日志文件不存在，测试可能尚未开始"
    exit 1
fi

echo "📊 公平对比测试监控"
echo "===================="
echo ""

# 检查进程是否在运行
if pgrep -f "fair_model_comparison.py" > /dev/null; then
    echo "✅ 测试正在运行中"
    echo ""
else
    echo "⏹️ 测试已完成或未运行"
    echo ""
fi

# 显示最新进度
echo "📈 最新进度:"
echo "---"
tail -30 "$LOG_FILE" | grep -E "训练轮次|准确率|完成|评估结果|LSTM|Transformer|收益" | tail -10

echo ""
echo "📊 完整日志: $LOG_FILE"
echo "查看完整日志: tail -f $LOG_FILE"
