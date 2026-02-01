#!/bin/bash
# 定时监控入口：供 crontab 每 N 分钟调用，把检查结果写入日志并在有真实错误时退出码非 0
# 用法: bash scripts/cron_routine_monitor.sh
# 建议 crontab: 每 2 分钟执行一次（快速发现错误）
#   */2 * * * * cd /home/cx/tigertrade && bash scripts/cron_routine_monitor.sh >> logs/routine_monitor.log 2>&1

cd /home/cx/tigertrade
MONITOR_LOG="logs/routine_monitor.log"
mkdir -p logs

RUN_TIME=$(date '+%Y-%m-%d %H:%M:%S')
DEMO_PID=$(pgrep -f "run_moe_demo.py" | head -1)
TRAIN_PID=$(pgrep -f "train_multiple_models_comparison.py" | head -1)

DEMO_STATUS="未运行"
TRAIN_STATUS="未运行"
OTHER_ERRORS="无"
EXIT_CODE=0
OTHER=""
TRAIN_ERR=""

if [ -n "$DEMO_PID" ]; then
  DEMO_ELAPSED=$(ps -p "$DEMO_PID" -o etime= 2>/dev/null | tr -d ' ')
  DEMO_STATUS="运行中 PID=$DEMO_PID 时长=$DEMO_ELAPSED"
  DEMO_LOG=$(ls -t logs/demo_20h_*.log 2>/dev/null | head -1)
  if [ -n "$DEMO_LOG" ]; then
    # 排除已知：1200、ApiException、Traceback 首行、已修复的 check_risk_control
    OTHER=$(grep -E "Exception|Traceback|AttributeError|Error:" "$DEMO_LOG" 2>/dev/null | grep -v "code=1200" | grep -v "ApiException" | grep -v "Traceback (most recent" | grep -v "check_risk_control" | tail -3)
    if [ -n "$OTHER" ]; then
      OTHER_ERRORS="有（DEMO 见下方）"
      EXIT_CODE=1
    fi
  fi
fi

if [ -n "$TRAIN_PID" ]; then
  TRAIN_ELAPSED=$(ps -p "$TRAIN_PID" -o etime= 2>/dev/null | tr -d ' ')
  TRAIN_STATUS="运行中 PID=$TRAIN_PID 时长=$TRAIN_ELAPSED"
  TRAIN_LOG=$(ls -t logs/train_multiple_models_*.log 2>/dev/null | head -1)
  if [ -n "$TRAIN_LOG" ]; then
    BYTES=$(wc -c < "$TRAIN_LOG")
    MTIME=$(stat -c %Y "$TRAIN_LOG" 2>/dev/null)
    NOW=$(date +%s)
    MINS_AGO=$(( (NOW - MTIME) / 60 ))
    if [ "$MINS_AGO" -gt 60 ] && [ "$BYTES" -lt 5000 ]; then
      TRAIN_STATUS="${TRAIN_STATUS} ⚠️日志${MINS_AGO}分钟未更新"
    fi
    TRAIN_ERR=$(grep -E "Error|Exception|Traceback" "$TRAIN_LOG" 2>/dev/null | tail -2)
    if [ -n "$TRAIN_ERR" ]; then
      [ "$OTHER_ERRORS" != "无" ] && OTHER_ERRORS="有（DEMO+训练）" || OTHER_ERRORS="有（训练见下方）"
      EXIT_CODE=1
    fi
  fi
fi

{
  echo "[$RUN_TIME] DEMO: $DEMO_STATUS | 训练: $TRAIN_STATUS | 其他错误: $OTHER_ERRORS"
  [ -n "$OTHER" ] && echo "$OTHER"
  [ -n "$TRAIN_ERR" ] && echo "$TRAIN_ERR"
} >> "$MONITOR_LOG"

exit $EXIT_CODE
