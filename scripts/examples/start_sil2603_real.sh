#!/bin/bash
# SIL2603真实数据训练启动脚本
# 使用Tiger DEMO账户获取真实市场K线数据

set -e

echo "================================================================================"
echo "🚀 SIL2603 真实市场数据训练"
echo "================================================================================"
echo ""
echo "标的: SIL2603 (白银期货)"
echo "数据来源: Tiger DEMO账户 - 真实市场K线"
echo "模型数量: 7个深度学习模型"
echo "获取天数: 90天"
echo "预期数据: 20,000+条真实记录"
echo ""
echo "================================================================================"

cd /home/cx/tigertrade

START_TIME=$(date +%s)

# 使用 --real-api 获取真实数据
python3 src/download_and_train.py \
    --symbol SIL2603 \
    --real-api \
    --days 90 \
    --min-records 20000 \
    --max-records 50000

EXIT_CODE=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "================================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 训练完成！"
    echo "================================================================================"
    echo "总耗时: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒"
    echo ""
    echo "结果位置: /home/cx/trading_data/SIL2603_dataset/"
    echo ""
    echo "查看报告:"
    echo "  cat /home/cx/trading_data/SIL2603_dataset/final_report.txt"
else
    echo "❌ 训练失败，退出码: $EXIT_CODE"
fi
echo "================================================================================"

exit $EXIT_CODE
