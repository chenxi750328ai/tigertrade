#!/bin/bash
# 延长tiger1策略运行时间到20小时

PID_FILE="/tmp/tiger1_demo.pid"
LOG_FILE="/tmp/tiger1_runtime_extension.log"

echo "============================================================"
echo "🔄 延长tiger1策略运行时间到20小时"
echo "============================================================"

# 检查是否有正在运行的进程
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "✅ 找到正在运行的策略进程 (PID: $PID)"
        echo "💡 当前策略将继续运行，新的启动脚本已配置为20小时"
        echo "   如需重启以应用新配置，请先停止当前进程:"
        echo "   kill $PID"
    else
        echo "⚠️ PID文件存在但进程未运行，清理PID文件"
        rm -f "$PID_FILE"
    fi
else
    echo "ℹ️ 未找到运行中的策略进程"
fi

echo ""
echo "✅ 配置已更新:"
echo "   - start_tiger1_demo.sh: 已设置为20小时"
echo "   - run_tiger1_demo_8h.py: 已设置为20小时"
echo ""
echo "💡 下次启动时将运行20小时"
echo "============================================================"
