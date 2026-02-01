#!/bin/bash
# Tick数据持续采集守护进程

cd /home/cx/tigertrade

echo "========================================================================"
echo "🚀 启动Tick数据持续采集器"
echo "========================================================================"

# 检查是否已在运行
if ps aux | grep -v grep | grep "tick_data_collector.py" > /dev/null; then
    echo "⚠️  采集器已在运行"
    echo "进程信息:"
    ps aux | grep -v grep | grep "tick_data_collector.py"
    echo ""
    echo "如需重启，请先运行: pkill -f tick_data_collector.py"
    exit 1
fi

# 创建日志目录
mkdir -p logs

# 后台启动采集器（实时模式）
nohup python src/tick_data_collector.py \
    --symbol SIL2603 \
    --mode realtime \
    --interval 60 \
    > logs/tick_collector_$(date +%Y%m%d_%H%M%S).log 2>&1 &

PID=$!

echo "✅ 采集器已启动"
echo "   PID: $PID"
echo "   合约: SIL2603"
echo "   模式: 实时采集（每60秒）"
echo "   数据目录: /home/cx/trading_data/ticks/"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 管理命令:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  查看状态: ps aux | grep tick_data_collector"
echo "  查看日志: tail -f /home/cx/trading_data/ticks/collector.log"
echo "  查看数据: ls -lh /home/cx/trading_data/ticks/*.csv"
echo "  停止采集: pkill -f tick_data_collector.py"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "✅ 采集器将持续运行，每60秒采集一次最新Tick数据"
echo "✅ 数据自动按日期分文件保存，去重合并"
echo "✅ 这样就可以积累无限多的历史数据！"
echo "========================================================================"
