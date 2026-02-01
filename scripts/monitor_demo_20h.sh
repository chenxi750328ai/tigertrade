#!/bin/bash
# DEMO 20小时运行监控脚本

cd /home/cx/tigertrade

LOG_FILE="demo_run_20h_$(date +%Y%m%d)_*.log"
LATEST_LOG=$(ls -t $LOG_FILE 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo "❌ 未找到DEMO日志文件"
    exit 1
fi

echo "=========================================="
echo "DEMO 20小时运行监控"
echo "=========================================="
echo "日志文件: $LATEST_LOG"
echo "监控时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo

# 1. 进程状态
DEMO_PID=$(pgrep -f "run_moe_demo\|tiger1.py.*moe" | head -1)
if [ -z "$DEMO_PID" ]; then
    echo "❌ DEMO进程未运行"
    exit 1
else
    echo "✅ 进程ID: $DEMO_PID"
    ETIME=$(ps -p $DEMO_PID -o etime --no-headers 2>/dev/null | tr -d ' ')
    if [ ! -z "$ETIME" ]; then
        echo "   运行时长: $ETIME"
    fi
fi
echo

# 2. 关键指标
echo "📈 关键指标:"
echo "   下单尝试: $(grep -c "下单调试\|execute_buy\|execute_sell" $LATEST_LOG 2>/dev/null || echo 0)"
echo "   下单成功: $(grep -c "订单提交成功\|Order创建成功" $LATEST_LOG 2>/dev/null || echo 0)"
echo "   下单失败: $(grep -c "下单失败\|下单异常\|授权失败\|account不能为空" $LATEST_LOG 2>/dev/null || echo 0)"
echo "   预测次数: $(grep -c "MoE Transformer预测\|预测结果" $LATEST_LOG 2>/dev/null || echo 0)"
echo "   错误总数: $(grep -c "❌\|ERROR\|错误\|Exception" $LATEST_LOG 2>/dev/null || echo 0)"
echo "   警告总数: $(grep -c "⚠️\|WARNING\|警告" $LATEST_LOG 2>/dev/null || echo 0)"
echo

# 3. 最近错误（最后5条）
echo "📋 最近错误（最后5条）:"
grep -E "❌|ERROR|错误|Exception|Traceback" $LATEST_LOG 2>/dev/null | tail -5 | sed 's/^/   /' || echo "   无错误"
echo

# 4. 最近日志（最后10行）
echo "📋 最近日志（最后10行）:"
tail -10 $LATEST_LOG | sed 's/^/   /'
echo

echo "=========================================="
