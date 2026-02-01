#!/bin/bash
# SIL2603 数据获取和模型训练启动脚本

set -e

echo "================================================================================"
echo "🚀 启动SIL2603数据获取和模型训练"
echo "================================================================================"
echo ""
echo "标的: SIL2603 (白银期货)"
echo "任务: 数据采集 → 7个模型训练 → 测试评估"
echo ""
echo "================================================================================"

cd /home/cx/tigertrade

# 开始执行
START_TIME=$(date +%s)

echo ""
echo "开始执行..."
echo ""

# 运行完整流程
python3 src/download_and_train.py \
    --symbol SIL2603 \
    --days 60 \
    --min-records 20000 \
    --max-records 50000

EXIT_CODE=$?

# 计算耗时
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "================================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ SIL2603训练完成！"
    echo "================================================================================"
    echo ""
    echo "总耗时: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒"
    echo ""
    echo "结果位置:"
    echo "  - 输出目录: /home/cx/trading_data/SIL2603_dataset/"
    echo "  - 最终报告: /home/cx/trading_data/SIL2603_dataset/final_report.txt"
    echo "  - 详细结果: /home/cx/trading_data/SIL2603_dataset/all_results.json"
    echo ""
    echo "查看报告:"
    echo "  cat /home/cx/trading_data/SIL2603_dataset/final_report.txt"
else
    echo "❌ 执行出错！退出码: $EXIT_CODE"
fi
echo "================================================================================"

exit $EXIT_CODE
