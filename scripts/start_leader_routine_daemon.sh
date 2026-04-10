#!/usr/bin/env bash
# 启动 Leader 例行后台进程（本机进程，与 Cursor 是否打开无关）
#
# 用法:
#   bash scripts/start_leader_routine_daemon.sh          # 默认每 900 秒一轮
#   LEADER_LOOP_INTERVAL_SEC=600 bash scripts/start_leader_routine_daemon.sh
#
# 停止:
#   bash scripts/stop_leader_routine_daemon.sh
#   或: kill "$(cat run/leader_routine_daemon.pid)"
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export TIGERTRADE_ROOT="$REPO_ROOT"
mkdir -p logs run

if [ -f "$REPO_ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$REPO_ROOT/.env"
  set +a
fi

PID_FILE="$REPO_ROOT/run/leader_routine_daemon.pid"
LOG_FILE="$REPO_ROOT/logs/leader_routine_daemon.log"
INTERVAL="${LEADER_LOOP_INTERVAL_SEC:-900}"

if [ -f "$PID_FILE" ]; then
  OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "已在运行: PID=$OLD_PID （$PID_FILE）"
    echo "若要重启: bash scripts/stop_leader_routine_daemon.sh 后再执行本脚本"
    exit 0
  fi
fi

export LEADER_LOOP_INTERVAL_SEC="$INTERVAL"
nohup bash "$REPO_ROOT/scripts/leader_loop_forever.sh" >>"$LOG_FILE" 2>&1 &
NEW_PID=$!
echo "$NEW_PID" >"$PID_FILE"
echo "已启动 Leader 例行后台: PID=$NEW_PID"
echo "间隔: ${INTERVAL}s | 日志: $LOG_FILE | PID 文件: $PID_FILE"
