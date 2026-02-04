#!/bin/bash
# 异常订单检查：扫描最新 DEMO 日志，发现「无止损止盈、超仓、风控报错」等问题。
# 用法:
#   手动: bash scripts/run_anomaly_order_check.sh
#   20h 跑完后: 执行本脚本或由 run_moe_demo.py 退出时自动执行
#   cron 定期（如每 30 分钟）: 0,30 * * * * cd /home/cx/tigertrade && bash scripts/run_anomaly_order_check.sh >> logs/anomaly_order_check.log 2>&1

set -e
cd /home/cx/tigertrade
mkdir -p logs
LOG="logs/anomaly_order_check.log"

echo "----------------------------------------"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 异常订单检查"
echo "----------------------------------------"
python scripts/analyze_demo_log.py
RET=$?
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 退出码: $RET"
if [ $RET -ne 0 ]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️ 发现问题，请查看上方输出" >> "$LOG"
fi
exit $RET
