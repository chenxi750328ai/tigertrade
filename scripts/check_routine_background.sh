#!/bin/bash
# 例行工作后台任务监控：检查 20h DEMO 与多模型训练是否在跑、日志是否有报错
# 用法: bash scripts/check_routine_background.sh

cd /home/cx/tigertrade

echo "=========================================="
echo "例行工作后台任务状态"
echo "=========================================="

# 20h DEMO
DEMO_PID=$(pgrep -f "run_moe_demo.py" | head -1)
if [ -n "$DEMO_PID" ]; then
  echo "[20h DEMO] 运行中 PID=$DEMO_PID"
  ps -p "$DEMO_PID" -o etime= 2>/dev/null | xargs echo "  运行时长:"
  DEMO_LOG=$(ls -t logs/demo_20h_*.log 2>/dev/null | head -1)
  if [ -n "$DEMO_LOG" ]; then
    echo "  日志: $DEMO_LOG"
    N1200=$(grep -c "code=1200" "$DEMO_LOG" 2>/dev/null || echo 0)
    echo "  已知 API 1200 次数: $N1200（时段/账户限制，非代码错误）"
    echo "  最近其他错误（排除 1200 及对应 Traceback/ApiException）:"
    grep -E "Exception|Traceback|AttributeError|Error:" "$DEMO_LOG" 2>/dev/null | grep -v "code=1200" | grep -v "ApiException" | grep -v "Traceback (most recent" | tail -5 || echo "  (无)"
  fi
else
  echo "[20h DEMO] 未运行"
fi

echo ""

# 多模型训练
TRAIN_PID=$(pgrep -f "train_multiple_models_comparison.py" | head -1)
if [ -n "$TRAIN_PID" ]; then
  echo "[多模型训练] 运行中 PID=$TRAIN_PID"
  ps -p "$TRAIN_PID" -o etime= 2>/dev/null | xargs echo "  运行时长:"
  TRAIN_LOG=$(ls -t logs/train_multiple_models_*.log 2>/dev/null | head -1)
  if [ -n "$TRAIN_LOG" ]; then
    BYTES=$(wc -c < "$TRAIN_LOG")
    MTIME=$(stat -c %Y "$TRAIN_LOG" 2>/dev/null)
    NOW=$(date +%s)
    MINS_AGO=$(( (NOW - MTIME) / 60 ))
    echo "  日志: $TRAIN_LOG ($BYTES bytes, 最后写入 ${MINS_AGO} 分钟前)"
    if [ "$MINS_AGO" -gt 30 ] && [ "$BYTES" -lt 5000 ]; then
      echo "  ⚠️ 提示: 日志长时间未更新且较小，可能为 stdout 缓冲；进程若 CPU 高则正常，可设 PYTHONUNBUFFERED=1 重跑以实时看日志"
    fi
    echo "  最近错误:"
    grep -E "Error|Exception|Traceback" "$TRAIN_LOG" 2>/dev/null | tail -3 || echo "  (无)"
  fi
else
  echo "[多模型训练] 未运行"
fi

echo ""
echo "=========================================="
