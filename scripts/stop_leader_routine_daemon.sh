#!/usr/bin/env bash
# 停止 start_leader_routine_daemon.sh 启动的后台循环
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PID_FILE="$REPO_ROOT/run/leader_routine_daemon.pid"

if [ ! -f "$PID_FILE" ]; then
  echo "无 PID 文件: $PID_FILE"
  exit 0
fi
PID="$(cat "$PID_FILE" 2>/dev/null || true)"
if [ -z "$PID" ]; then
  rm -f "$PID_FILE"
  exit 0
fi
if kill -0 "$PID" 2>/dev/null; then
  kill "$PID" 2>/dev/null || true
  echo "已发送停止信号: PID=$PID"
else
  echo "进程已不存在: PID=$PID"
fi
rm -f "$PID_FILE"
# 子进程 leader_loop_forever 可能仍在，一并结束
pkill -f "scripts/leader_loop_forever.sh" 2>/dev/null && echo "已结束 leader_loop_forever.sh" || true
