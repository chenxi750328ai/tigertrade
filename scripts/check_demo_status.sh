#!/bin/bash
# DEMO运行状态查询脚本

cd /home/cx/tigertrade

echo "=========================================="
echo "DEMO运行状态查询"
echo "=========================================="
echo

# 1. 进程状态
echo "📊 进程状态:"
DEMO_PID=$(pgrep -f "run_moe_demo" | head -1)
if [ -z "$DEMO_PID" ]; then
    echo "❌ DEMO进程未运行"
else
    echo "✅ 进程ID: $DEMO_PID"
    ETIME=$(ps -p $DEMO_PID -o etime --no-headers 2>/dev/null | tr -d ' ')
    if [ ! -z "$ETIME" ]; then
        echo "   运行时长: $ETIME"
    fi
fi
echo

# 2. 日志文件
echo "📄 日志文件:"
LATEST_LOG=$(ls -t demo_run_20h_*.log 2>/dev/null | head -1)
if [ -z "$LATEST_LOG" ]; then
    echo "❌ 未找到日志文件"
else
    echo "✅ 最新日志: $(basename $LATEST_LOG)"
    SIZE=$(ls -lh $LATEST_LOG | awk '{print $5}')
    echo "   文件大小: $SIZE"
    MTIME=$(stat -c %y $LATEST_LOG 2>/dev/null | cut -d. -f1)
    if [ ! -z "$MTIME" ]; then
        echo "   最后更新: $MTIME"
    fi
fi
echo

# 3. 关键指标
if [ ! -z "$LATEST_LOG" ]; then
    echo "📈 关键指标:"
    echo "   下单尝试: $(grep -c "下单调试" $LATEST_LOG 2>/dev/null || echo 0)"
    echo "   下单成功: $(grep -c "Order创建成功" $LATEST_LOG 2>/dev/null || echo 0)"
    echo "   下单失败: $(grep -c "下单失败\|下单异常\|授权失败" $LATEST_LOG 2>/dev/null || echo 0)"
    echo "   预测次数: $(grep -c "MoE Transformer预测" $LATEST_LOG 2>/dev/null || echo 0)"
    echo "   错误总数: $(grep -c "❌\|ERROR\|错误" $LATEST_LOG 2>/dev/null || echo 0)"
fi
echo

# 4. 最近日志（最后10行）
if [ ! -z "$LATEST_LOG" ]; then
    echo "📋 最近日志（最后10行）:"
    tail -10 $LATEST_LOG | sed 's/^/   /'
fi

echo "=========================================="
