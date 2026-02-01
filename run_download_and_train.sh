#!/bin/bash
# 通用期货数据下载和模型训练启动脚本
# 支持任意标的代码

set -e  # 遇到错误立即退出

echo "========================================================================================================"
echo "🚀 通用期货数据下载和模型训练完整流程"
echo "========================================================================================================"
echo ""
echo "此脚本将执行以下步骤:"
echo "  1. 下载指定标的的期货数据（默认>20000条）"
echo "  2. 对所有模型进行训练（7个模型）"
echo "  3. 在测试集上评估所有模型"
echo "  4. 生成完整的分析报告"
echo ""
echo "========================================================================================================"

# 切换到tigertrade目录
cd /home/cx/tigertrade

# ============================================
# 可配置参数
# ============================================

# 标的代码（期货合约）
# 示例：
#   - SIL2603 或 SIL.COMEX.202603  (白银期货)
#   - GC2603 或 GC.COMEX.202603    (黄金期货)
#   - NQ2603 或 NQ.CME.202603      (纳斯达克期货)
#   - ES2603 或 ES.CME.202603      (标普500期货)
# 留空则使用配置文件中的FUTURE_SYMBOL
SYMBOL=""

# 数据采集参数
DAYS=60              # 获取天数（可根据需要调整）
MIN_RECORDS=20000    # 最少记录数
MAX_RECORDS=50000    # 最大记录数

# 输出目录（留空则根据标的自动生成）
OUTPUT_DIR=""

# 可选：使用真实API（取消注释以下行）
# REAL_API="--real-api"
REAL_API=""

# ============================================
# 显示配置
# ============================================

echo "配置参数:"
if [ -z "$SYMBOL" ]; then
    echo "  - 标的代码: 使用配置文件默认值"
else
    echo "  - 标的代码: $SYMBOL"
fi
echo "  - 获取天数: $DAYS"
echo "  - 最少记录数: $MIN_RECORDS"
echo "  - 最大记录数: $MAX_RECORDS"
if [ -z "$OUTPUT_DIR" ]; then
    echo "  - 输出目录: 自动生成"
else
    echo "  - 输出目录: $OUTPUT_DIR"
fi
echo "  - 使用真实API: $([ -z "$REAL_API" ] && echo "否（模拟模式）" || echo "是")"
echo ""
echo "========================================================================================================"
echo ""

# 确认执行
read -p "按Enter键开始执行，或按Ctrl+C取消... " confirm

# 构建命令参数
CMD_ARGS=""
if [ -n "$SYMBOL" ]; then
    CMD_ARGS="$CMD_ARGS --symbol $SYMBOL"
fi
if [ -n "$REAL_API" ]; then
    CMD_ARGS="$CMD_ARGS $REAL_API"
fi
if [ -n "$OUTPUT_DIR" ]; then
    CMD_ARGS="$CMD_ARGS --output-dir $OUTPUT_DIR"
fi
CMD_ARGS="$CMD_ARGS --days $DAYS --min-records $MIN_RECORDS --max-records $MAX_RECORDS"

# 开始执行
START_TIME=$(date +%s)

echo ""
echo "========================================================================================================"
echo "开始执行..."
echo "========================================================================================================"
echo ""

# 运行Python脚本
python3 src/download_and_train.py $CMD_ARGS

EXIT_CODE=$?

# 计算耗时
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "========================================================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 完整流程执行成功！"
else
    echo "❌ 执行过程中出现错误！退出码: $EXIT_CODE"
fi
echo "========================================================================================================"
echo ""
echo "总耗时: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒"
echo ""

# 获取实际使用的输出目录
if [ -z "$OUTPUT_DIR" ]; then
    # 如果未指定，尝试找到最新的输出目录
    ACTUAL_OUTPUT_DIR=$(ls -td /home/cx/trading_data/*_dataset 2>/dev/null | head -1)
else
    ACTUAL_OUTPUT_DIR="$OUTPUT_DIR"
fi

if [ -n "$ACTUAL_OUTPUT_DIR" ]; then
    echo "结果文件位置:"
    echo "  - 输出目录: $ACTUAL_OUTPUT_DIR"
    echo "  - 最终报告: $ACTUAL_OUTPUT_DIR/final_report.txt"
    echo "  - 详细结果: $ACTUAL_OUTPUT_DIR/all_results.json"
    echo ""
    echo "查看报告:"
    echo "  cat $ACTUAL_OUTPUT_DIR/final_report.txt"
fi

echo ""
echo "========================================================================================================"

exit $EXIT_CODE
